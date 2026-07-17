//! F.5 gradient-clipping smoke test.
//!
//! Constructs a single Linear layer, drives a forward + backward with a
//! deliberately large input, then asserts that clipping caps the global
//! gradient norm at the requested threshold.

use pm_candle::{CandleBackend, CandleParam};
use pm_core::nn::Linear;
use pm_core::{Module, Ops, Parameterized};
use pm_train::clip_grad_norm;

fn global_norm(
    bk: &CandleBackend,
    params: &[&CandleParam],
    grads: &candle_core::backprop::GradStore,
) -> anyhow::Result<f32> {
    let mut total: f64 = 0.0;
    for p in params {
        if let Some(g) = bk.gradient(grads, p)? {
            let sq = bk.mul(&g, &g)?;
            let s = bk.sum_all(&sq)?;
            let v = bk.to_vec_f32(&s)?;
            total += f64::from(v[0]);
        }
    }
    Ok(total.sqrt() as f32)
}

#[test]
fn clip_grad_norm_caps_global_norm_at_max() -> anyhow::Result<()> {
    let bk = CandleBackend::new_cpu();
    let layer: Linear<CandleBackend> = Linear::from_constants(&bk, 4, 4, false, 1.0)?;
    let params = layer.collect_params();

    // Large input → large gradients.
    let x = bk.from_slice_f32(&[100.0_f32; 4], &[1, 4])?;
    let y = layer.forward(&bk, &x)?;
    // Scalar loss = mean of y.
    let loss = bk.mean_all(&y)?;
    let mut grads = bk.backward(&loss)?;

    let raw_norm = global_norm(&bk, &params, &grads)?;
    assert!(
        raw_norm > 1.0,
        "raw_norm too small ({raw_norm}) to test clipping"
    );

    let max_norm = 0.5;
    let reported = clip_grad_norm(&bk, &params, &mut grads, max_norm)?;
    assert!(
        (reported - raw_norm).abs() < 1e-3,
        "clip_grad_norm should report the *unclipped* norm: got {reported}, want ≈ {raw_norm}"
    );

    let new_norm = global_norm(&bk, &params, &grads)?;
    assert!(
        new_norm <= max_norm + 1e-3,
        "post-clip norm {new_norm} exceeds max_norm {max_norm}"
    );
    assert!(
        (new_norm - max_norm).abs() < 1e-3,
        "post-clip norm {new_norm} not at the cap {max_norm}"
    );

    Ok(())
}

#[test]
fn clip_grad_norm_below_threshold_is_noop() -> anyhow::Result<()> {
    let bk = CandleBackend::new_cpu();
    let layer: Linear<CandleBackend> = Linear::from_constants(&bk, 4, 4, false, 0.01)?;
    let params = layer.collect_params();

    let x = bk.from_slice_f32(&[0.1_f32; 4], &[1, 4])?;
    let y = layer.forward(&bk, &x)?;
    let loss = bk.mean_all(&y)?;
    let mut grads = bk.backward(&loss)?;

    let raw_norm = global_norm(&bk, &params, &grads)?;
    let max_norm = 1.0;
    assert!(raw_norm < max_norm, "raw_norm too large for noop test");

    let reported = clip_grad_norm(&bk, &params, &mut grads, max_norm)?;
    assert!((reported - raw_norm).abs() < 1e-5);

    let post_norm = global_norm(&bk, &params, &grads)?;
    assert!(
        (post_norm - raw_norm).abs() < 1e-5,
        "no-op clip must not change grads"
    );

    Ok(())
}
