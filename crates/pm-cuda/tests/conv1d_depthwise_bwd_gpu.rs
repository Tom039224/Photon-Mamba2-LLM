//! Numerical regression test for the depthwise conv1d backward GPU wiring
//! (Phase B'.2e): `conv1d_backward` now dispatches to the
//! `conv1d_depthwise_bwd_{x,w}_gpu` kernels (landed in commit 57d3719
//! but never wired — see `conv1d_backward_depthwise_gpu` in
//! `src/backend/ops_impl.rs` for why) instead of the host im2col+GEMM
//! loop, whenever `groups == c_in && c_in_per_group == 1 && k_size <=
//! 128` (the same guard the forward path already uses).
//!
//! Compares `CudaBackend` gradients against `CandleBackend` (native
//! Candle `conv1d`, a fully-differentiable op — unlike `rms_norm`, it is
//! not on the `_no_bwd` list in `CLAUDE.md`). Shapes are graduated from
//! tiny to production scale; the production-scale run is the first
//! exercise of this code path at the exact shape that previously hung
//! for 8h (see the doc comment above `conv1d_backward_depthwise_gpu`) —
//! run standalone with a bounded timeout, not as part of a big batch.

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

/// Depthwise conv1d (`groups = channels`, 1 filter/channel): builds x,
/// weight, bias as trainable params, runs `conv1d -> sum_all -> backward`
/// on both backends with identical data, and asserts grad_x / grad_w /
/// grad_bias agree within `tol`.
#[allow(clippy::too_many_arguments)]
fn check_depthwise_conv1d_backward_matches_candle(
    batch: usize,
    channels: usize,
    t_in: usize,
    k_size: usize,
    stride: usize,
    padding: usize,
    tol: f32,
    tag: &str,
) {
    let groups = channels;
    let x_data = lcg_vec(3001, batch * channels * t_in, 0.2, -0.1);
    let w_data = lcg_vec(4002, channels * k_size, 0.3, -0.1);
    let b_data = lcg_vec(5003, channels, 0.1, 0.0);
    let w_shape = [channels, 1, k_size];

    // ---- CandleBackend (reference) ----
    let cpu = CandleBackend::new_cpu();
    let px_ref = cpu
        .param_from_slice_f32(&x_data, &[batch, channels, t_in])
        .expect("cpu px");
    let pw_ref = cpu.param_from_slice_f32(&w_data, &w_shape).expect("cpu pw");
    let pb_ref = cpu
        .param_from_slice_f32(&b_data, &[channels])
        .expect("cpu pb");
    let y_ref = cpu
        .conv1d(
            px_ref.as_tensor(),
            pw_ref.as_tensor(),
            Some(pb_ref.as_tensor()),
            stride,
            padding,
            groups,
        )
        .expect("cpu conv1d");
    let loss_ref = cpu.sum_all(&y_ref).expect("cpu sum_all");
    let store_ref = cpu.backward(&loss_ref).expect("cpu backward");
    let gx_ref = cpu
        .to_vec_f32(
            &cpu.gradient(&store_ref, &px_ref)
                .expect("cpu gx call")
                .expect("cpu gx Some"),
        )
        .expect("cpu gx to_vec");
    let gw_ref = cpu
        .to_vec_f32(
            &cpu.gradient(&store_ref, &pw_ref)
                .expect("cpu gw call")
                .expect("cpu gw Some"),
        )
        .expect("cpu gw to_vec");
    let gb_ref = cpu
        .to_vec_f32(
            &cpu.gradient(&store_ref, &pb_ref)
                .expect("cpu gb call")
                .expect("cpu gb Some"),
        )
        .expect("cpu gb to_vec");

    // ---- CudaBackend (device under test — depthwise GPU backward path) ----
    let bk = CudaBackend::new(0).expect("CUDA init");
    let px = bk
        .param_from_slice_f32(&x_data, &[batch, channels, t_in])
        .expect("cuda px");
    let pw = bk.param_from_slice_f32(&w_data, &w_shape).expect("cuda pw");
    let pb = bk
        .param_from_slice_f32(&b_data, &[channels])
        .expect("cuda pb");
    let y = bk
        .conv1d(
            px.as_tensor(),
            pw.as_tensor(),
            Some(pb.as_tensor()),
            stride,
            padding,
            groups,
        )
        .expect("cuda conv1d");
    let loss = bk.sum_all(&y).expect("cuda sum_all");
    let store = bk.backward(&loss).expect("cuda backward");
    let gx = bk
        .to_vec_f32(
            &bk.gradient(&store, &px)
                .expect("cuda gx call")
                .expect("cuda gx Some"),
        )
        .expect("cuda gx to_vec");
    let gw = bk
        .to_vec_f32(
            &bk.gradient(&store, &pw)
                .expect("cuda gw call")
                .expect("cuda gw Some"),
        )
        .expect("cuda gw to_vec");
    let gb = bk
        .to_vec_f32(
            &bk.gradient(&store, &pb)
                .expect("cuda gb call")
                .expect("cuda gb Some"),
        )
        .expect("cuda gb to_vec");

    let err_x = max_abs_err(&gx, &gx_ref);
    let err_w = max_abs_err(&gw, &gw_ref);
    let err_b = max_abs_err(&gb, &gb_ref);
    eprintln!(
        "conv1d_depthwise_bwd_gpu[{tag}] batch={batch} channels={channels} t_in={t_in} \
         k={k_size}: grad_x max_abs_err={err_x:.3e}  grad_w max_abs_err={err_w:.3e}  \
         grad_b max_abs_err={err_b:.3e}"
    );
    assert!(
        err_x < tol,
        "[{tag}] grad_x vs CandleBackend: max_abs_err={err_x:.3e} exceeds {tol:.1e}"
    );
    assert!(
        err_w < tol,
        "[{tag}] grad_w vs CandleBackend: max_abs_err={err_w:.3e} exceeds {tol:.1e}"
    );
    assert!(
        err_b < tol,
        "[{tag}] grad_bias vs CandleBackend: max_abs_err={err_b:.3e} exceeds {tol:.1e}"
    );
}

/// Tiny scale — fast smoke gate before scaling up.
#[test]
fn conv1d_depthwise_bwd_tiny() {
    check_depthwise_conv1d_backward_matches_candle(2, 8, 10, 3, 1, 1, 1e-4, "tiny");
}

/// Medium scale, non-power-of-two channel count and asymmetric padding —
/// exercises the ragged edges of the grid before the full production run.
#[test]
fn conv1d_depthwise_bwd_medium() {
    check_depthwise_conv1d_backward_matches_candle(2, 100, 37, 4, 1, 2, 1e-4, "medium");
}

/// Production shape: batch=1, channels=1024 (xbc_dim = d_inner + 2*n_groups*
/// d_state = 768 + 2*128 for the 100M config), t_in=512, k_size=4 (d_conv),
/// stride=1, padding=3 (causal-style, matches Mamba2Block's conv1d call
/// pattern). This is the exact shape that hung the GPU for 8h in the
/// previous (reverted) wiring attempt.
/// Watchdog-wrapped: this exact shape hung the device for 8 hours before
/// the `bar.sync`-unroll fix, and a plain `#[test]` has no timeout — a
/// regression would block `cargo test` (and CI) forever instead of
/// failing. The body runs in a worker thread; if it does not finish
/// within the deadline the test panics immediately. (The worker thread
/// leaks on timeout — acceptable: the test binary is about to die
/// anyway, and a fast red X beats an eternal hang.)
#[test]
fn conv1d_depthwise_bwd_prod_shape() {
    const DEADLINE: std::time::Duration = std::time::Duration::from_secs(180);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        check_depthwise_conv1d_backward_matches_candle(1, 1024, 512, 4, 1, 3, 1e-4, "prod");
        let _ = tx.send(());
    });
    if rx.recv_timeout(DEADLINE).is_err() {
        panic!(
            "prod-shape depthwise conv1d backward exceeded {DEADLINE:?} — \
             bar.sync unroll hang regression? (see CLAUDE.md nvptx pitfall)"
        );
    }
}
