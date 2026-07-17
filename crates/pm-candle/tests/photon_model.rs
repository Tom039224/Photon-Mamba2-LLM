//! D.9: End-to-end integration test for `PhotonMamba`.
//!
//! Forward-only — backward / autograd verification belongs in F.4
//! (recorded as PLAN.md's relocation of the original C.5 task).

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{
    ChunkLocalDecoder, ContextChunker, ContextConverter, ContextEncoder, DecoderLevel,
    HierarchicalDecoder, HierarchicalEncoder, HierarchicalLevel, TokenEmbedding,
};
use pm_core::{Ops, PhotonMamba, Tensor};

struct ToyDims {
    vocab: usize,
    d_model: usize,
    chunk_size: usize,
    n_layers_per_level: usize,
    n_levels: usize, // = 2 for our toy
    weight_scale: f32,
}

fn mk_block(bk: &CandleBackend, d_model: usize, weight_scale: f32) -> Mamba2Block<CandleBackend> {
    let cfg = Mamba2Config {
        d_model,
        d_state: 8,
        d_head: 8,
        n_heads: d_model / 8,
        n_groups: 1,
        d_conv: 4,
        block_len: 4,
        rmsnorm_eps: 1e-5,
    };
    Mamba2Block::from_constants(bk, cfg, weight_scale).unwrap()
}

fn build_toy_model(bk: &CandleBackend, dims: &ToyDims) -> PhotonMamba<CandleBackend> {
    assert_eq!(dims.n_levels, 2, "toy currently fixed at L=2");

    let embed =
        TokenEmbedding::from_constants(bk, dims.vocab, dims.d_model, dims.weight_scale).unwrap();

    let lvl0 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(
            (0..dims.n_layers_per_level)
                .map(|_| mk_block(bk, dims.d_model, dims.weight_scale))
                .collect(),
        ),
        chunker: Some(
            ContextChunker::from_constants(
                bk,
                dims.d_model,
                dims.d_model,
                dims.chunk_size,
                dims.weight_scale,
            )
            .unwrap(),
        ),
    };
    let lvl1 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(
            (0..dims.n_layers_per_level)
                .map(|_| mk_block(bk, dims.d_model, dims.weight_scale))
                .collect(),
        ),
        chunker: None,
    };
    let encoder = HierarchicalEncoder::from_levels(vec![lvl0, lvl1]);

    let conv = ContextConverter::from_constants(
        bk,
        dims.d_model,
        dims.d_model,
        dims.chunk_size,
        dims.weight_scale,
    )
    .unwrap();
    let dec_stack = ChunkLocalDecoder::from_layers(
        (0..dims.n_layers_per_level)
            .map(|_| mk_block(bk, dims.d_model, dims.weight_scale))
            .collect(),
        dims.chunk_size,
        dims.chunk_size,
    );
    let decoder = HierarchicalDecoder::from_levels(vec![DecoderLevel::new(conv, dec_stack)]);

    PhotonMamba::new(embed, encoder, decoder)
}

#[test]
fn photon_mamba_toy_forward_end_to_end() {
    let bk = CandleBackend::new_cpu();
    let dims = ToyDims {
        vocab: 32,
        d_model: 16,
        chunk_size: 4,
        n_layers_per_level: 2,
        n_levels: 2,
        weight_scale: 0.05,
    };
    let model = build_toy_model(&bk, &dims);

    let (b, t) = (2, 16);
    // Deterministic token ids in [0, vocab).
    let ids_data: Vec<i64> = (0..b * t).map(|i| (i as i64) % dims.vocab as i64).collect();
    let ids = bk.from_slice_i64(&ids_data, &[b, t]).unwrap();

    let out = model.forward(&bk, &ids).unwrap();

    // Logits: (B, T, vocab).
    assert_eq!(out.logits.shape(), &[b, t, dims.vocab]);
    // Encoded: 2 levels, top is (B, T/chunk, d).
    assert_eq!(out.encoded.encoded.len(), 2);
    assert_eq!(out.encoded.encoded[0].shape(), &[b, t, dims.d_model]);
    assert_eq!(
        out.encoded.encoded[1].shape(),
        &[b, t / dims.chunk_size, dims.d_model]
    );
    // Decoder: 1 prediction (matches L-1).
    assert_eq!(out.predicted.len(), 1);
    assert_eq!(out.predicted[0].shape(), &[b, t, dims.d_model]);

    let v = bk.to_vec_f32(&out.logits).unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "non-finite logits");
    let max_abs = v.iter().map(|x| x.abs()).fold(0f32, f32::max);
    assert!(max_abs > 0.0, "logits collapsed to zero");
}

#[test]
#[should_panic(expected = "decoder must have exactly one fewer level than encoder")]
fn photon_mamba_rejects_inconsistent_level_counts() {
    let bk = CandleBackend::new_cpu();
    let d = 16;
    let c = 4;
    let vocab = 32;

    // Encoder with 2 levels (1 chunker + top).
    let embed = TokenEmbedding::from_constants(&bk, vocab, d, 0.1).unwrap();
    let lvl0 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(vec![mk_block(&bk, d, 0.1)]),
        chunker: Some(ContextChunker::from_constants(&bk, d, d, c, 0.1).unwrap()),
    };
    let lvl1 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(vec![mk_block(&bk, d, 0.1)]),
        chunker: None,
    };
    let encoder = HierarchicalEncoder::from_levels(vec![lvl0, lvl1]);

    // Decoder with 2 levels (should be 1 to match encoder L=2). This
    // mismatch is what PhotonMamba::new must reject.
    let mk_dl = |bk: &CandleBackend| -> DecoderLevel<CandleBackend> {
        let conv = ContextConverter::from_constants(bk, d, d, c, 0.1).unwrap();
        let stack = ChunkLocalDecoder::from_layers(vec![mk_block(bk, d, 0.1)], c, c);
        DecoderLevel::new(conv, stack)
    };
    let decoder = HierarchicalDecoder::from_levels(vec![mk_dl(&bk), mk_dl(&bk)]);

    let _ = PhotonMamba::new(embed, encoder, decoder);
}
