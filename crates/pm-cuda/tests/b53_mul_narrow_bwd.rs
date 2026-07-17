//! Phase B'.3 wave-1 — numerical parity tests for the device-side `mul`
//! broadcast path (`CudaBackend::broadcast_mul_dev`) and the device-side
//! `narrow` backward (`kernels::narrow_backward_f32`).
//!
//! Both replace host round-trip fallbacks (`CudaBackend::broadcast_binary_op`
//! / the old `scatter_to_narrow` host loop) that measured ~1.3 ms/call in
//! training — see `docs/perf-log.md` B'.3. This file checks that the
//! device-kernel replacements are still numerically correct: every case
//! compares `CudaBackend` against `CandleBackend::new_cpu()` (fp32, tol
//! 1e-4), for both the forward value and the backward gradient
//! (CLAUDE.md invariant #3).
//!
//! Run with:
//!   cargo test -p pm-cuda --features cuda --test b53_mul_narrow_bwd -- --test-threads=1

#![cfg(feature = "cuda")]

use pm_candle::CandleBackend;
use pm_core::{Ops, Param, Tensor};
use pm_cuda::CudaBackend;

// ---- Helpers ----------------------------------------------------------------

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "max_abs_err: length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Deterministic pseudo-random data in `[-0.5, 0.5)` (same generator shape
/// as `matmul_broadcast.rs`, parameterised by `seed` so distinct operands
/// in the same test get independent-looking data).
fn filled(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed).wrapping_mul(2654435761)) & 0xffff) as f32 / 65536.0 - 0.5)
        .collect()
}

// ---- mul broadcast: forward + backward vs CandleBackend --------------------

/// `loss = sum((a*b)^2)` on both backends from identical data; compares
/// forward `a*b` and both input gradients. Squaring (rather than a bare
/// `sum(a*b)`) makes the gradient depend on the actual broadcast values at
/// each position, not just "scatter 1s" — a stronger check of the index
/// mapping in both the forward kernel and its VJP.
fn assert_mul_broadcast_matches_candle(a_shape: &[usize], b_shape: &[usize]) {
    assert_ne!(
        a_shape, b_shape,
        "test bug: shapes must differ to exercise the broadcast path"
    );
    let a_data = filled(a_shape.iter().product(), 101);
    let b_data = filled(b_shape.iter().product(), 202);

    // ---- CandleBackend (CPU) reference ----
    let cpu = CandleBackend::new_cpu();
    let pa_cpu = cpu.param_from_slice_f32(&a_data, a_shape).expect("pa_cpu");
    let pb_cpu = cpu.param_from_slice_f32(&b_data, b_shape).expect("pb_cpu");
    let y_cpu = cpu
        .mul(pa_cpu.as_tensor(), pb_cpu.as_tensor())
        .expect("mul cpu");
    let y2_cpu = cpu.mul(&y_cpu, &y_cpu).expect("square cpu");
    let loss_cpu = cpu.sum_all(&y2_cpu).expect("sum_all cpu");
    let y_ref = cpu.to_vec_f32(&y_cpu).expect("y_cpu vec");
    let store_cpu = cpu.backward(&loss_cpu).expect("backward cpu");
    let ga_ref = cpu
        .to_vec_f32(
            &cpu.gradient(&store_cpu, &pa_cpu)
                .expect("gradient(pa) call")
                .expect("Some(ga_cpu)"),
        )
        .expect("ga_cpu vec");
    let gb_ref = cpu
        .to_vec_f32(
            &cpu.gradient(&store_cpu, &pb_cpu)
                .expect("gradient(pb) call")
                .expect("Some(gb_cpu)"),
        )
        .expect("gb_cpu vec");

    // ---- CudaBackend ----
    let bk = CudaBackend::new(0).expect("CUDA init");
    let pa = bk.param_from_slice_f32(&a_data, a_shape).expect("pa");
    let pb = bk.param_from_slice_f32(&b_data, b_shape).expect("pb");
    let y = bk.mul(pa.as_tensor(), pb.as_tensor()).expect("mul cuda");
    let y2 = bk.mul(&y, &y).expect("square cuda");
    let loss = bk.sum_all(&y2).expect("sum_all cuda");

    assert_eq!(
        y.shape(),
        y_cpu.shape(),
        "fwd out shape a={a_shape:?} b={b_shape:?}"
    );
    let y_got = bk.to_vec_f32(&y).expect("y vec");
    let fwd_err = max_abs_err(&y_got, &y_ref);
    assert!(
        fwd_err < 1e-4,
        "mul fwd mismatch a={a_shape:?} b={b_shape:?}: max_abs_err={fwd_err:.2e}"
    );

    let store = bk.backward(&loss).expect("backward cuda");
    let ga = bk
        .gradient(&store, &pa)
        .expect("ga call")
        .expect("Some(ga)");
    let gb = bk
        .gradient(&store, &pb)
        .expect("gb call")
        .expect("Some(gb)");
    assert_eq!(ga.shape(), a_shape, "dA must match A's original shape");
    assert_eq!(gb.shape(), b_shape, "dB must match B's original shape");
    let ga_got = bk.to_vec_f32(&ga).expect("ga vec");
    let gb_got = bk.to_vec_f32(&gb).expect("gb vec");
    let ga_err = max_abs_err(&ga_got, &ga_ref);
    let gb_err = max_abs_err(&gb_got, &gb_ref);
    assert!(
        ga_err < 1e-4,
        "dA mismatch a={a_shape:?} b={b_shape:?}: max_abs_err={ga_err:.2e}"
    );
    assert!(
        gb_err < 1e-4,
        "dB mismatch a={a_shape:?} b={b_shape:?}: max_abs_err={gb_err:.2e}"
    );
}

#[test]
fn mul_broadcast_trailing_dim_matches_candle() {
    // Mirrors `Mamba2Block::forward`'s `x_dt = mul(x_ssm_4d, dt_4d)` /
    // `d_term = mul(d_r, x_ssm_4d)`: the smaller operand's *trailing* dim
    // is 1 (broadcasts P), big-operand-first argument order.
    assert_mul_broadcast_matches_candle(&[2, 3, 4, 5], &[2, 3, 4, 1]);
}

#[test]
fn mul_broadcast_leading_dims_matches_candle() {
    // Mirrors `a_bth = mul(a_r, dt)`: small-operand-first argument order,
    // broadcasting both leading dims (B and T).
    assert_mul_broadcast_matches_candle(&[1, 1, 4], &[2, 3, 4]);
}

#[test]
fn mul_broadcast_mixed_per_dim_matches_candle() {
    // Explicitly requested case: each operand broadcasts a *different*
    // axis simultaneously ([1,3] broadcasts dim0, [3,1] broadcasts dim1) —
    // out = [3,3]. This is the case `Ops::matmul`'s B'.2b fix explicitly
    // could *not* route to a single stride-0 batch; `broadcast_mul_dev`
    // must still get it right since it walks full per-dim strides.
    assert_mul_broadcast_matches_candle(&[1, 3], &[3, 1]);
}

#[test]
fn mul_broadcast_mixed_per_dim_rank3_matches_candle() {
    // Higher-rank per-dim-mixed variant: dim0 broadcasts on `a`, dim1
    // broadcasts on `b`, dim2 matches — exercises the general stride walk
    // at rank 3 (production shapes here are all rank 3-4).
    assert_mul_broadcast_matches_candle(&[2, 1, 3], &[1, 4, 3]);
}

#[test]
fn mul_broadcast_rank1_scalar_like_matches_candle() {
    // Degenerate rank-1 broadcast (a single scalar-ish operand against a
    // vector) — smallest possible `rank` the kernel's stride loop handles.
    assert_mul_broadcast_matches_candle(&[1], &[7]);
}

// ---- narrow backward: forward + backward vs CandleBackend ------------------

/// `loss = sum(narrow(x, dim, start, len)^2)` on both backends from
/// identical data; compares forward narrow output and the scattered input
/// gradient `dX` (shape = `orig_shape`, zero outside the narrowed window).
fn assert_narrow_backward_matches_candle(
    orig_shape: &[usize],
    dim: usize,
    start: usize,
    len: usize,
) {
    let n: usize = orig_shape.iter().product();
    let data = filled(n, 303 + start * 13 + len * 7);

    // ---- CandleBackend (CPU) reference ----
    let cpu = CandleBackend::new_cpu();
    let px_cpu = cpu.param_from_slice_f32(&data, orig_shape).expect("px_cpu");
    let y_cpu = cpu
        .narrow(px_cpu.as_tensor(), dim, start, len)
        .expect("narrow cpu");
    let y2_cpu = cpu.mul(&y_cpu, &y_cpu).expect("square cpu");
    let loss_cpu = cpu.sum_all(&y2_cpu).expect("sum_all cpu");
    let y_ref = cpu.to_vec_f32(&y_cpu).expect("y_cpu vec");
    let store_cpu = cpu.backward(&loss_cpu).expect("backward cpu");
    let gx_ref = cpu
        .to_vec_f32(
            &cpu.gradient(&store_cpu, &px_cpu)
                .expect("gradient(px) call")
                .expect("Some(gx_cpu)"),
        )
        .expect("gx_cpu vec");

    // ---- CudaBackend ----
    let bk = CudaBackend::new(0).expect("CUDA init");
    let px = bk.param_from_slice_f32(&data, orig_shape).expect("px");
    let y = bk
        .narrow(px.as_tensor(), dim, start, len)
        .expect("narrow cuda");
    let y2 = bk.mul(&y, &y).expect("square cuda");
    let loss = bk.sum_all(&y2).expect("sum_all cuda");

    assert_eq!(
        y.shape(),
        y_cpu.shape(),
        "fwd out shape orig={orig_shape:?} dim={dim} start={start} len={len}"
    );
    let y_got = bk.to_vec_f32(&y).expect("y vec");
    let fwd_err = max_abs_err(&y_got, &y_ref);
    assert!(
        fwd_err < 1e-4,
        "narrow fwd mismatch orig={orig_shape:?} dim={dim} start={start} len={len}: \
         max_abs_err={fwd_err:.2e}"
    );

    let store = bk.backward(&loss).expect("backward cuda");
    let gx = bk
        .gradient(&store, &px)
        .expect("gx call")
        .expect("Some(gx)");
    assert_eq!(
        gx.shape(),
        orig_shape,
        "dX must match X's original (pre-narrow) shape"
    );
    let gx_got = bk.to_vec_f32(&gx).expect("gx vec");
    let gx_err = max_abs_err(&gx_got, &gx_ref);
    assert!(
        gx_err < 1e-4,
        "dX mismatch orig={orig_shape:?} dim={dim} start={start} len={len}: \
         max_abs_err={gx_err:.2e}"
    );
}

#[test]
fn narrow_backward_last_dim_matches_candle() {
    // Generic last-dim narrow: `inner = orig_shape[dim+1..].product() == 1`
    // — the degenerate geometry the old host loop mishandled.
    assert_narrow_backward_matches_candle(&[2, 5, 7], 2, 2, 3);
}

#[test]
fn narrow_backward_production_geometry_matches_candle() {
    // `Mamba2Block::forward`'s `xbc = narrow(&xzd, 2, d_inner, xbc_dim)`
    // at the exact production feature width (`in_proj_dim = 1804`,
    // `xbc_dim = 1024` at `start = d_inner = 768`) with `T` shrunk to 8 for
    // test speed — same degenerate `inner == 1` geometry, real widths.
    assert_narrow_backward_matches_candle(&[1, 8, 1804], 2, 768, 1024);
}

#[test]
fn narrow_backward_non_last_dim_matches_candle() {
    // `inner > 1` (dim=1 of a rank-3 tensor) — confirms the shared kernel
    // still handles the non-degenerate geometry correctly, not just the
    // last-dim special case.
    assert_narrow_backward_matches_candle(&[2, 7, 5], 1, 2, 3);
}

#[test]
fn narrow_backward_dim0_matches_candle() {
    // `outer == 1` edge case (dim = 0: nothing precedes it).
    assert_narrow_backward_matches_candle(&[6, 4], 0, 1, 3);
}

#[test]
fn narrow_backward_full_window_matches_candle() {
    // start=0, len=full axis: every output element is inside the window
    // (the kernel's "outside window -> 0" branch is never taken here).
    assert_narrow_backward_matches_candle(&[3, 4], 1, 0, 4);
}
