//! Tests for HierarchicalEncoder (D.6).

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{ContextChunker, ContextEncoder, HierarchicalEncoder, HierarchicalLevel};
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

#[test]
fn hierarchical_encoder_two_levels_shape_chain() {
    let bk = CandleBackend::new_cpu();
    let d0 = 16;
    let d1 = 16; // keep equal — chunker just collapses time
    let c_1 = 4; // chunk size at level 0 → produces level 1

    // Level 0: 2 Mamba2 layers + chunker (d0 -> d1, chunk_size=c_1).
    let lvl0 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(vec![mk_block(&bk, d0, 0.1), mk_block(&bk, d0, 0.1)]),
        chunker: Some(ContextChunker::from_constants(&bk, d0, d1, c_1, 0.1).unwrap()),
    };
    // Level 1 (top): 1 Mamba2 layer, no chunker.
    let lvl1 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(vec![mk_block(&bk, d1, 0.1)]),
        chunker: None,
    };

    let enc = HierarchicalEncoder::from_levels(vec![lvl0, lvl1]);
    assert_eq!(enc.n_levels(), 2);

    let (b, t) = (1, 16);
    let x_data: Vec<f32> = (0..b * t * d0).map(|i| (i as f32 * 0.05).sin()).collect();
    let x = bk.from_slice_f32(&x_data, &[b, t, d0]).unwrap();

    let out = enc.encode(&bk, &x).unwrap();
    assert_eq!(out.encoded.len(), 2);
    assert_eq!(out.chunked.len(), 1);

    // encoded[0] keeps shape (B, T, d0).
    assert_eq!(out.encoded[0].shape(), &[b, t, d0]);
    // chunked[0] is the chunker output: (B, T/c_1, d1).
    assert_eq!(out.chunked[0].shape(), &[b, t / c_1, d1]);
    // encoded[1] (top): same shape as chunked[0].
    assert_eq!(out.encoded[1].shape(), &[b, t / c_1, d1]);

    for tensor in out.encoded.iter().chain(out.chunked.iter()) {
        let v = bk.to_vec_f32(tensor).unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn hierarchical_encoder_single_level_degenerates_to_context_encoder() {
    let bk = CandleBackend::new_cpu();
    let d = 16;
    let lvl = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(vec![mk_block(&bk, d, 0.1)]),
        chunker: None,
    };
    let enc = HierarchicalEncoder::from_levels(vec![lvl]);
    assert_eq!(enc.n_levels(), 1);

    let (b, t) = (1, 8);
    let x_data: Vec<f32> = (0..b * t * d).map(|i| (i as f32 * 0.1).sin()).collect();
    let x = bk.from_slice_f32(&x_data, &[b, t, d]).unwrap();

    let out = enc.encode(&bk, &x).unwrap();
    assert_eq!(out.encoded.len(), 1);
    assert!(out.chunked.is_empty());
    assert_eq!(out.encoded[0].shape(), &[b, t, d]);
}

#[test]
#[should_panic(expected = "chunker present iff not top level")]
fn hierarchical_encoder_rejects_top_with_chunker() {
    let bk = CandleBackend::new_cpu();
    let d = 16;
    let lvl = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(vec![mk_block(&bk, d, 0.1)]),
        chunker: Some(ContextChunker::from_constants(&bk, d, d, 4, 0.1).unwrap()),
    };
    // Single level with a chunker → top has one, invariant violated.
    let _ = HierarchicalEncoder::from_levels(vec![lvl]);
}
