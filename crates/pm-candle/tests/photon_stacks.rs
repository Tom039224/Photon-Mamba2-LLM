//! Tests for ContextEncoder (D.3) and ChunkLocalDecoder (D.5).

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{ChunkLocalDecoder, ContextEncoder};
use pm_core::{Module, Ops, Tensor};

fn small_cfg() -> Mamba2Config {
    Mamba2Config {
        d_model: 16,
        d_state: 8,
        d_head: 8,
        n_heads: 2,
        n_groups: 1,
        d_conv: 4,
        block_len: 4,
        rmsnorm_eps: 1e-5,
    }
}

#[test]
fn context_encoder_preserves_shape_and_is_finite() {
    let bk = CandleBackend::new_cpu();
    let cfg = small_cfg();
    let n_layers = 3;

    let layers: Vec<_> = (0..n_layers)
        .map(|_| Mamba2Block::from_constants(&bk, cfg.clone(), 0.1).unwrap())
        .collect();
    let encoder = ContextEncoder::from_layers(layers);
    assert_eq!(encoder.n_layers(), n_layers);

    let (b, t) = (1, 8);
    let x_data: Vec<f32> = (0..b * t * cfg.d_model)
        .map(|i| (i as f32 * 0.13).sin())
        .collect();
    let x = bk.from_slice_f32(&x_data, &[b, t, cfg.d_model]).unwrap();
    let y = encoder.forward(&bk, &x).unwrap();
    assert_eq!(y.shape(), &[b, t, cfg.d_model]);

    let v = bk.to_vec_f32(&y).unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "encoder output non-finite");
    // Residual stack: output must not collapse to zero.
    let max_abs = v.iter().map(|x| x.abs()).fold(0f32, f32::max);
    assert!(max_abs > 0.0);
}

#[test]
fn context_encoder_with_zero_weight_blocks_acts_as_identity() {
    // weight_scale=0 makes every Mamba2Block produce a zero output (D_skip
    // is zero too); the residual then preserves the input exactly.
    let bk = CandleBackend::new_cpu();
    let cfg = small_cfg();
    let layers: Vec<_> = (0..2)
        .map(|_| Mamba2Block::from_constants(&bk, cfg.clone(), 0.0).unwrap())
        .collect();
    let encoder = ContextEncoder::from_layers(layers);

    let (b, t) = (1, 4);
    let x_data: Vec<f32> = (0..b * t * cfg.d_model).map(|i| (i + 1) as f32).collect();
    let x = bk.from_slice_f32(&x_data, &[b, t, cfg.d_model]).unwrap();
    let y = encoder.forward(&bk, &x).unwrap();
    let v = bk.to_vec_f32(&y).unwrap();
    for (got, want) in v.iter().zip(x_data.iter()) {
        assert!(
            (got - want).abs() < 1e-5,
            "encoder with zero block weights must be identity (got {got}, want {want})"
        );
    }
}

#[test]
fn chunk_local_decoder_shape_and_chunk_parallel_equivalence() {
    let bk = CandleBackend::new_cpu();
    let cfg = small_cfg();
    let layers: Vec<_> = (0..2)
        .map(|_| Mamba2Block::from_constants(&bk, cfg.clone(), 0.1).unwrap())
        .collect();
    let r_l = 2;
    let c_l = 4;
    let decoder = ChunkLocalDecoder::from_layers(layers, r_l, c_l);
    assert_eq!(decoder.bounded_context_len(), r_l + c_l);

    let (b, s) = (1, 3);
    let t = r_l + c_l;
    let d = cfg.d_model;
    let x_data: Vec<f32> = (0..b * s * t * d)
        .map(|i| (i as f32 * 0.07).cos())
        .collect();
    let x = bk.from_slice_f32(&x_data, &[b, s, t, d]).unwrap();
    let y = decoder.forward(&bk, &x).unwrap();
    assert_eq!(y.shape(), &[b, s, t, d]);
    let v = bk.to_vec_f32(&y).unwrap();
    assert!(v.iter().all(|x| x.is_finite()));

    // Chunk-parallel equivalence: extract chunk 1 alone (as B=1, T=t),
    // run it through the same blocks in a tiny encoder, and check the
    // result matches y[:, 1, :, :].
    let chunk_idx = 1;
    let chunk_offset = chunk_idx * t * d;
    let chunk1: Vec<f32> = (0..t * d).map(|j| x_data[chunk_offset + j]).collect();
    let x1 = bk.from_slice_f32(&chunk1, &[1, t, d]).unwrap();
    // The decoder applies the same residual stack to each chunk; we can
    // mimic it via a fresh ContextEncoder over the same blocks.
    let single_layers: Vec<_> = (0..2)
        .map(|_| Mamba2Block::from_constants(&bk, cfg.clone(), 0.1).unwrap())
        .collect();
    let mirror = ContextEncoder::from_layers(single_layers);
    let y1 = mirror.forward(&bk, &x1).unwrap();
    let y1v = bk.to_vec_f32(&y1).unwrap();

    let offset_in_y = chunk_offset; // chunk index 1, flatten (s, t, d) → linear
    for j in 0..t * d {
        let got = v[offset_in_y + j];
        let want = y1v[j];
        assert!(
            (got - want).abs() < 1e-4,
            "chunk-parallel mismatch at j={j}: got {got}, want {want}"
        );
    }
}

#[test]
#[should_panic(expected = "must equal r_l + c_l")]
fn chunk_local_decoder_rejects_wrong_t() {
    let bk = CandleBackend::new_cpu();
    let cfg = small_cfg();
    let layers = vec![Mamba2Block::from_constants(&bk, cfg.clone(), 0.1).unwrap()];
    let decoder = ChunkLocalDecoder::from_layers(layers, 2, 4);
    let x = bk
        .from_slice_f32(&vec![0.5f32; 2 * 5 * cfg.d_model], &[1, 2, 5, cfg.d_model])
        .unwrap();
    let _ = decoder.forward(&bk, &x);
}
