//! Tests for HierarchicalDecoder (D.7).

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{
    ChunkLocalDecoder, ContextChunker, ContextConverter, ContextEncoder, DecoderLevel,
    HierarchicalDecoder, HierarchicalEncoder, HierarchicalLevel,
};
use pm_core::{Ops, Tensor};

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

/// Build a two-level encoder + decoder pair sharing dimensions
/// (`d_model = d`, `c = r = chunk_size`).
struct TwoLevelPair {
    encoder: HierarchicalEncoder<CandleBackend>,
    decoder: HierarchicalDecoder<CandleBackend>,
}

fn build_two_level(bk: &CandleBackend, d: usize, c: usize, weight_scale: f32) -> TwoLevelPair {
    let lvl0 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(vec![mk_block(bk, d, weight_scale)]),
        chunker: Some(ContextChunker::from_constants(bk, d, d, c, weight_scale).unwrap()),
    };
    let lvl1 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(vec![mk_block(bk, d, weight_scale)]),
        chunker: None,
    };
    let encoder = HierarchicalEncoder::from_levels(vec![lvl0, lvl1]);

    let converter = ContextConverter::from_constants(bk, d, d, c, weight_scale).unwrap();
    let decoder_stack = ChunkLocalDecoder::from_layers(
        vec![mk_block(bk, d, weight_scale), mk_block(bk, d, weight_scale)],
        c,
        c,
    );
    let dec_level = DecoderLevel::new(converter, decoder_stack);
    let decoder = HierarchicalDecoder::from_levels(vec![dec_level]);

    TwoLevelPair { encoder, decoder }
}

#[test]
fn hierarchical_decoder_shape_chain_matches_encoder() {
    let bk = CandleBackend::new_cpu();
    let d = 16;
    let c = 4;
    let pair = build_two_level(&bk, d, c, 0.1);

    let (b, t) = (1, 16);
    let x_data: Vec<f32> = (0..b * t * d).map(|i| (i as f32 * 0.05).sin()).collect();
    let x = bk.from_slice_f32(&x_data, &[b, t, d]).unwrap();

    let encoded = pair.encoder.encode(&bk, &x).unwrap();
    // level_inputs[0] = original embeddings; level_inputs[1] = chunker output.
    let level_inputs = vec![x, encoded.chunked[0].clone()];
    let predicted = pair.decoder.decode(&bk, &encoded, &level_inputs).unwrap();

    assert_eq!(predicted.len(), 1);
    // predicted[0] must match the level-0 stream shape.
    assert_eq!(predicted[0].shape(), &[b, t, d]);

    let v = bk.to_vec_f32(&predicted[0]).unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
    let max_abs = v.iter().map(|x| x.abs()).fold(0f32, f32::max);
    assert!(max_abs > 0.0);
}

#[test]
fn hierarchical_decoder_starting_seed_drives_chunk0_first_position() {
    // With weight_scale=0 every Mamba2Block is the zero map and the
    // residual stack acts as identity. The decoder's trailing-C output
    // for chunk k therefore equals chunk k's shifted-stream slice
    // (positions C*k .. C*(k+1) of the shifted stream).
    //
    // We pick a stream whose first C positions of the shift come from
    // the converter's (zero) starting_latent — so predicted[0][:, 0..C]
    // must be zero.
    let bk = CandleBackend::new_cpu();
    let d = 16;
    let c = 4;
    let pair = build_two_level(&bk, d, c, 0.0); // zero block weights

    let (b, t) = (1, 8);
    let x_data: Vec<f32> = (0..b * t * d).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let x = bk.from_slice_f32(&x_data, &[b, t, d]).unwrap();

    let encoded = pair.encoder.encode(&bk, &x).unwrap();
    let level_inputs = vec![x.clone(), encoded.chunked[0].clone()];
    let predicted = pair.decoder.decode(&bk, &encoded, &level_inputs).unwrap();

    let v = bk.to_vec_f32(&predicted[0]).unwrap();
    assert!(
        v.len() >= c * d,
        "v.len() ({}) < c * d ({})",
        v.len(),
        c * d
    );
    // First C positions of predicted[0] should be 0 (starting_latent).
    for (j, &val) in v.iter().enumerate().take(c * d) {
        assert!(
            val.abs() < 1e-5,
            "expected starting_latent zero at flat idx {j}, got {val}"
        );
    }
    // Next C positions should match the original x[..C] (the shift puts
    // x's first C positions into chunk 1's trailing slot when blocks are
    // identity).
    for j in 0..(c * d) {
        let got = v[c * d + j];
        let want = (j as f32 + 1.0) * 0.1; // x_data[0..(c*d)]
        assert!(
            (got - want).abs() < 1e-4,
            "expected x's first C at chunk 1, got {got} vs {want} at j={j}"
        );
    }
}

#[test]
#[should_panic(expected = "level_inputs length mismatch")]
fn hierarchical_decoder_rejects_wrong_level_inputs_count() {
    let bk = CandleBackend::new_cpu();
    let d = 16;
    let c = 4;
    let pair = build_two_level(&bk, d, c, 0.1);

    let (b, t) = (1, 8);
    let x = bk
        .from_slice_f32(&vec![0.1f32; b * t * d], &[b, t, d])
        .unwrap();
    let encoded = pair.encoder.encode(&bk, &x).unwrap();
    // Wrong: only 1 entry instead of 2.
    let level_inputs = vec![x];
    let _ = pair.decoder.decode(&bk, &encoded, &level_inputs);
}
