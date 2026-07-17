//! Numerical regression test for the rmsnorm backward kernel geometry
//! rewrite (Phase B'.2d): `rmsnorm_backward_x_f32` moved from
//! one-thread-per-row to one-block-per-row + shared-memory tree
//! reduction, and `rmsnorm_backward_w_f32` moved from one-thread-per-
//! column (768 threads total) to a two-stage chunked-reduction dispatch
//! (`rmsnorm_backward_w_partial_f32` + `reduce_sum_dim_keepdim_f32`).
//!
//! This compares `CudaBackend` gradients directly against
//! `CandleBackend` (the project's numerical reference — see
//! `CLAUDE.md` §数値同等性) rather than finite differences: B4.4f-T5
//! (`b44f_t5_block_fd_grad.rs`) already established that FD at
//! production magnitude is noise-dominated for rmsnorm (13% "smoking
//! gun" that turned out to be FD cancellation, not a VJP bug — see
//! `docs/b4-4f-investigation.md`). Comparing two independent analytical
//! backends sidesteps that noise entirely.
//!
//! Two shapes are covered:
//! - `prod`: `n_rows=512, d_model=768` — matches the production T=512,
//!   d_model=768 config exactly (exercises full-width blocks in both
//!   the X and W kernels, and 32 row-chunks in the W kernel).
//! - `ragged`: `n_rows=37, d_model=300` — deliberately not a multiple of
//!   the W kernel's `ROWS_PER_CHUNK=16` or the X/W kernels' block size
//!   256, exercising the ragged last-chunk / last-column-block guards.

#![cfg(feature = "cuda")]

use pm_candle::CandleBackend;
use pm_core::{Ops, Param};
use pm_cuda::CudaBackend;

fn lcg_vec(seed: u64, n: usize, scale: f32, bias: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let r = ((state >> 41) as f32) / ((1u32 << 23) as f32);
            r * scale + bias
        })
        .collect()
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "max_abs_err: length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Runs `rmsnorm -> sum_all -> backward` on both backends with identical
/// data and asserts grad_x / grad_w agree within `tol` (absolute; inputs
/// are scaled so gradient magnitudes stay O(1e-2..1e0), matching the
/// project's fp32 1e-4 numerical-equivalence bar).
fn check_rmsnorm_backward_matches_candle(n_rows: usize, d_model: usize, tol: f32, tag: &str) {
    let x_data = lcg_vec(1001, n_rows * d_model, 0.5, -0.25);
    let w_data = lcg_vec(2002, d_model, 0.3, 0.9); // weight centred near 1.0

    // ---- CandleBackend (reference) ----
    let cpu = CandleBackend::new_cpu();
    let px_ref = cpu
        .param_from_slice_f32(&x_data, &[n_rows, d_model])
        .expect("cpu px");
    let pw_ref = cpu
        .param_from_slice_f32(&w_data, &[d_model])
        .expect("cpu pw");
    let y_ref = cpu
        .rmsnorm(px_ref.as_tensor(), pw_ref.as_tensor(), 1e-5)
        .expect("cpu rmsnorm");
    let loss_ref = cpu.sum_all(&y_ref).expect("cpu sum_all");
    let store_ref = cpu.backward(&loss_ref).expect("cpu backward");
    let gx_ref = cpu
        .to_vec_f32(
            &cpu.gradient(&store_ref, &px_ref)
                .expect("cpu grad_x call")
                .expect("cpu grad_x Some"),
        )
        .expect("cpu grad_x to_vec");
    let gw_ref = cpu
        .to_vec_f32(
            &cpu.gradient(&store_ref, &pw_ref)
                .expect("cpu grad_w call")
                .expect("cpu grad_w Some"),
        )
        .expect("cpu grad_w to_vec");

    // ---- CudaBackend (device under test) ----
    let bk = CudaBackend::new(0).expect("CUDA init");
    let px = bk
        .param_from_slice_f32(&x_data, &[n_rows, d_model])
        .expect("cuda px");
    let pw = bk
        .param_from_slice_f32(&w_data, &[d_model])
        .expect("cuda pw");
    let y = bk
        .rmsnorm(px.as_tensor(), pw.as_tensor(), 1e-5)
        .expect("cuda rmsnorm");
    let loss = bk.sum_all(&y).expect("cuda sum_all");
    let store = bk.backward(&loss).expect("cuda backward");
    let gx = bk
        .to_vec_f32(
            &bk.gradient(&store, &px)
                .expect("cuda grad_x call")
                .expect("cuda grad_x Some"),
        )
        .expect("cuda grad_x to_vec");
    let gw = bk
        .to_vec_f32(
            &bk.gradient(&store, &pw)
                .expect("cuda grad_w call")
                .expect("cuda grad_w Some"),
        )
        .expect("cuda grad_w to_vec");

    let err_x = max_abs_err(&gx, &gx_ref);
    let err_w = max_abs_err(&gw, &gw_ref);
    eprintln!(
        "rmsnorm_backward_geometry[{tag}] n_rows={n_rows} d_model={d_model}: \
         grad_x max_abs_err={err_x:.3e}  grad_w max_abs_err={err_w:.3e}"
    );
    assert!(
        err_x < tol,
        "[{tag}] grad_x vs CandleBackend: max_abs_err={err_x:.3e} exceeds {tol:.1e}"
    );
    assert!(
        err_w < tol,
        "[{tag}] grad_w vs CandleBackend: max_abs_err={err_w:.3e} exceeds {tol:.1e}"
    );
}

/// Production shape: n_rows=512 (T=512, B=1), d_model=768. Exercises the
/// X kernel's 512-block grid and the W kernel's 32 row-chunks (512 / 16)
/// over an exact 3-block-wide (768 / 256) column grid.
#[test]
fn rmsnorm_backward_matches_candle_prod_shape() {
    check_rmsnorm_backward_matches_candle(512, 768, 1e-4, "prod");
}

/// Ragged shape: n_rows=37 is not a multiple of the W kernel's
/// `ROWS_PER_CHUNK=16` (last chunk covers rows [32, 37)), and
/// d_model=300 is not a multiple of the 256-wide block used by both
/// kernels (last column block only has 44 active lanes/threads).
#[test]
fn rmsnorm_backward_matches_candle_ragged_shape() {
    check_rmsnorm_backward_matches_candle(37, 300, 1e-4, "ragged");
}
