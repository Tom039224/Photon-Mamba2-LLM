//! Regression tests for the `rms_norm`-has-no-backward bug.
//!
//! `candle_nn::ops::rms_norm` (candle-nn 0.11.0) is built on
//! `Tensor::apply_op2_no_bwd` — it carries `BackpropOp::none()` and has
//! **no backward**. `pm-candle` used it until `ops_impl.rs` was switched
//! to the differentiable `rms_norm_slow`. With the fused `rms_norm`,
//! backprop through a `Mamba2Block` reached only `out_proj_weight` (the
//! post-norm matmul's rhs leaf) and the tied embedding — every pre-norm
//! parameter (`in_proj_weight`, `conv1d_weight`, `conv1d_bias`, `a_log`,
//! `d_skip`, `dt_bias`, `norm_weight`) silently received `None` in the
//! `GradStore` and was skipped by the optimiser. Every Candle-backend
//! training run before the fix therefore only updated `out_proj_weight`
//! + the embedding table, never the SSD / gating / conv dynamics.
//!
//! These tests run on CPU (fp32) so they guard the fix in the default
//! CI, not only under `--features cuda`. The prior backward coverage
//! was on the `pm-cuda` backend (which has its own differentiable
//! rmsnorm VJP) — the Candle path had no all-parameter grad assertion,
//! which is why the bug went undetected.

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::{Module, Ops, Param, Parameterized, Tensor};

fn small_cfg() -> Mamba2Config {
    Mamba2Config {
        d_model: 16,
        d_state: 8,
        d_head: 8,
        n_heads: 2, // d_inner = 16
        n_groups: 1,
        d_conv: 4,
        block_len: 4,
        rmsnorm_eps: 1e-5,
    }
}

/// Deterministic LCG (matches `train_cmd::perturb_params`) — breaks the
/// `from_constants` weight symmetry so gradients are non-degenerate.
fn lcg_vec(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as u32 as f32 / u32::MAX as f32 - 0.5) * scale
        })
        .collect()
}

fn seed_params(bk: &CandleBackend, params: &[&<CandleBackend as Ops>::Param], seed: u64) {
    for (k, p) in params.iter().enumerate() {
        let shape = p.as_tensor().shape().to_vec();
        let n: usize = shape.iter().product();
        let data = lcg_vec(seed.wrapping_add(k as u64 * 101), n, 0.1);
        let t = bk.from_slice_f32(&data, &shape).unwrap();
        bk.assign(p, &t).unwrap();
    }
}

const PARAM_NAMES: [&str; 8] = [
    "in_proj_weight",
    "conv1d_weight",
    "conv1d_bias",
    "a_log",
    "d_skip",
    "dt_bias",
    "norm_weight",
    "out_proj_weight",
];

fn forward_loss(
    bk: &CandleBackend,
    block: &Mamba2Block<CandleBackend>,
    x: &<CandleBackend as Ops>::Tensor,
) -> <CandleBackend as Ops>::Tensor {
    let y = block.forward(bk, x).unwrap();
    bk.sum_all(&y).unwrap()
}

/// The core regression: after the `rms_norm_slow` fix, backprop through a
/// `Mamba2Block` must reach **every** parameter with a finite, non-zero
/// gradient. Before the fix this asserted 8/8 but only got 1/8
/// (`out_proj_weight`).
#[test]
fn mamba2_block_backward_reaches_all_params() {
    let bk = CandleBackend::new_cpu();
    let cfg = small_cfg();
    let block = Mamba2Block::from_constants(&bk, cfg.clone(), 0.1).unwrap();
    let params = block.collect_params();
    seed_params(&bk, &params, 2024);

    let (b, t) = (1, 8);
    let x_data: Vec<f32> = (0..b * t * cfg.d_model)
        .map(|i| (i as f32 * 0.031).sin() * 0.5)
        .collect();
    let x = bk.from_slice_f32(&x_data, &[b, t, cfg.d_model]).unwrap();

    let loss = forward_loss(&bk, &block, &x);
    let grads = bk.backward(&loss).unwrap();

    let mut missing = Vec::new();
    for (name, p) in PARAM_NAMES.iter().zip(params.iter()) {
        match bk.gradient(&grads, p).unwrap() {
            None => missing.push(format!("{name}: NO GRAD (None in GradStore)")),
            Some(g) => {
                let gv = bk.to_vec_f32(&g).unwrap();
                assert!(
                    gv.iter().all(|v| v.is_finite()),
                    "{name}: gradient has non-finite entries"
                );
                let max_abs = gv.iter().fold(0f32, |m, v| m.max(v.abs()));
                if max_abs <= 1e-9 {
                    missing.push(format!(
                        "{name}: gradient is all ~zero (max|g|={max_abs:.2e})"
                    ));
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "backprop did not reach all params (rms_norm-no-backward regression):\n  {}",
        missing.join("\n  ")
    );
}

/// Correctness (not just presence): central-difference FD grad-check on
/// `in_proj_weight`, which was frozen before the fix. Lenient tolerance
/// and zero-threshold (fp32 FD on a compound SSD forward loses digits to
/// cancellation — see B4.4f T5 / commit `b5895c8`); we only assert on
/// elements whose analytical grad is well above the FD noise floor.
#[test]
fn mamba2_block_fd_grad_check_in_proj_weight() {
    let bk = CandleBackend::new_cpu();
    let cfg = small_cfg();
    let block = Mamba2Block::from_constants(&bk, cfg.clone(), 0.1).unwrap();
    let params = block.collect_params();
    seed_params(&bk, &params, 7);

    let (b, t) = (1, 8);
    let x_data: Vec<f32> = (0..b * t * cfg.d_model)
        .map(|i| (i as f32 * 0.017 + 0.3).cos() * 0.5)
        .collect();
    let x = bk.from_slice_f32(&x_data, &[b, t, cfg.d_model]).unwrap();

    // Analytical grad for in_proj_weight (params[0]).
    let loss = forward_loss(&bk, &block, &x);
    let grads = bk.backward(&loss).unwrap();
    let g_ana = bk
        .to_vec_f32(
            &bk.gradient(&grads, params[0])
                .unwrap()
                .expect("in_proj_weight grad present"),
        )
        .unwrap();

    let shape = params[0].as_tensor().shape().to_vec();
    let n: usize = shape.iter().product();
    let mut w = bk.to_vec_f32(params[0].as_tensor()).unwrap();
    let orig = w.clone();

    let eps = 1e-2f32;
    // Pick the indices with the largest analytical gradient (best-
    // conditioned for FD), so the check is meaningful and not noise.
    let mut idxs: Vec<usize> = (0..n).collect();
    idxs.sort_by(|&i, &j| g_ana[j].abs().partial_cmp(&g_ana[i].abs()).unwrap());
    let mut checked = 0usize;
    for &i in idxs.iter().take(4) {
        if g_ana[i].abs() < 1e-3 {
            continue; // below FD noise floor at eps=1e-2
        }
        w.copy_from_slice(&orig);
        w[i] = orig[i] + eps;
        bk.assign(params[0], &bk.from_slice_f32(&w, &shape).unwrap())
            .unwrap();
        let lp = bk.to_vec_f32(&forward_loss(&bk, &block, &x)).unwrap()[0];

        w[i] = orig[i] - eps;
        bk.assign(params[0], &bk.from_slice_f32(&w, &shape).unwrap())
            .unwrap();
        let lm = bk.to_vec_f32(&forward_loss(&bk, &block, &x)).unwrap()[0];

        let fd = (lp - lm) / (2.0 * eps);
        let rel = (fd - g_ana[i]).abs() / fd.abs().max(g_ana[i].abs()).max(1e-6);
        eprintln!(
            "in_proj_weight[{i}]: ana={:.5e} fd={fd:.5e} rel={rel:.3}",
            g_ana[i]
        );
        assert!(
            rel < 0.15,
            "FD grad-check failed at idx {i}: analytical {:.5e} vs FD {fd:.5e} (rel {rel:.3})",
            g_ana[i]
        );
        checked += 1;
    }
    // restore
    bk.assign(params[0], &bk.from_slice_f32(&orig, &shape).unwrap())
        .unwrap();
    assert!(checked > 0, "no well-conditioned index found for FD check");
}
