//! Mamba2Block forward smoke test, run through the Candle backend.
//!
//! This exercises the full block (in_proj → conv1d → SiLU → SSD → D-skip
//! → gated RMSNorm → out_proj) end-to-end. With weight_scale=0.1 and a
//! sine-wave input, the output should be finite, the right shape, and
//! not identically zero.

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::{Module, Ops, Tensor};

#[test]
fn mamba2_block_forward_smoke() {
    let bk = CandleBackend::new_cpu();
    let cfg = Mamba2Config {
        d_model: 16,
        d_state: 8,
        d_head: 8,
        n_heads: 2, // d_inner = 16
        n_groups: 1,
        d_conv: 4,
        block_len: 4,
        rmsnorm_eps: 1e-5,
    };
    let block = Mamba2Block::from_constants(&bk, cfg.clone(), 0.1).unwrap();

    let (b, t) = (1, 8);
    let x_data: Vec<f32> = (0..b * t * cfg.d_model)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    let x = bk.from_slice_f32(&x_data, &[b, t, cfg.d_model]).unwrap();

    let y = block.forward(&bk, &x).unwrap();

    assert_eq!(y.shape(), &[b, t, cfg.d_model]);
    let v = bk.to_vec_f32(&y).unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "non-finite output");
    let max_abs = v.iter().map(|x| x.abs()).fold(0f32, f32::max);
    assert!(max_abs > 0.0, "all-zero output indicates broken forward");
    eprintln!(
        "mamba2_block_forward_smoke: out shape {:?}, max_abs={max_abs:.4e}",
        y.shape()
    );
}

#[test]
fn mamba2_block_n_groups_equal_n_heads() {
    let bk = CandleBackend::new_cpu();
    let cfg = Mamba2Config {
        d_model: 16,
        d_state: 8,
        d_head: 8,
        n_heads: 2,
        n_groups: 2, // == n_heads, no broadcast needed
        d_conv: 4,
        block_len: 4,
        rmsnorm_eps: 1e-5,
    };
    let block = Mamba2Block::from_constants(&bk, cfg.clone(), 0.1).unwrap();

    let (b, t) = (1, 8);
    let x = bk
        .from_slice_f32(&vec![0.5f32; b * t * cfg.d_model], &[b, t, cfg.d_model])
        .unwrap();
    let y = block.forward(&bk, &x).unwrap();
    assert_eq!(y.shape(), &[b, t, cfg.d_model]);
    assert!(bk.to_vec_f32(&y).unwrap().iter().all(|x| x.is_finite()));
}

#[test]
fn mamba2_config_shapes_consistent() {
    let cfg = Mamba2Config {
        d_model: 768,
        d_state: 128,
        d_head: 64,
        n_heads: 12, // d_inner = 768
        n_groups: 1,
        d_conv: 4,
        block_len: 64,
        rmsnorm_eps: 1e-5,
    };
    assert_eq!(cfg.d_inner(), 768);
    // xbc = d_inner + 2 * G * N = 768 + 2 * 1 * 128 = 1024
    assert_eq!(cfg.xbc_dim(), 1024);
    // in_proj = d_inner + xbc + n_heads = 768 + 1024 + 12 = 1804
    assert_eq!(cfg.in_proj_dim(), 1804);
}
