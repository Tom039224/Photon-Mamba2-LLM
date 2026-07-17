//! End-to-end autograd smoke test — B4.3c.
//!
//! Verifies that a small computation graph containing `ssd_scan` runs
//! `backward` without panicking, produces finite gradients for every
//! tracked parameter, and allows an SGD step to update the parameters.
//!
//! Test 1 (`ssd_scan_with_rmsnorm_e2e`) builds a minimal block:
//!   `rmsnorm(x_in, w) -> reshape -> ssd_scan -> sum_all`
//! with 5 trainable params (w_norm, x, a, b, c).  This exercises the
//! ssd_scan backward together with rmsnorm in a realistic graph.
//!
//! Test 2 (`ssd_scan_direct_e2e_no_nan`) uses just `ssd_scan -> sum_all`
//! with 4 params.

#![cfg(feature = "cuda")]

use pm_core::{Ops, Param};
use pm_cuda::CudaBackend;

// ---- helpers ----------------------------------------------------------------

/// Deterministic LCG -- same as ssd_parity.rs.
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

// ---- Test: ssd_scan_with_rmsnorm_e2e -----------------------------------------

/// Minimal E2E: rmsnorm -> reshape -> ssd_scan(x, a, b, c) -> sum_all.
///
/// This is a 5-param graph (x_in, w_norm, a, b, c).  Verifies:
/// - backward runs without panic
/// - all 5 params have finite grads
/// - SGD step succeeds
/// - second forward+backward also succeeds (no stale tape state)
#[test]
fn ssd_scan_with_rmsnorm_e2e() {
    let bk = CudaBackend::new(0).expect("CUDA init");

    let (b, t, h, p, n, q) = (1usize, 4, 2, 2, 2, 2);

    // x_in is (B, T, H*P) = (1, 4, 4).
    let x_in_data = lcg_vec(50, b * t * h * p, 0.3, -0.15);
    let w_norm_data = lcg_vec(51, h * p, 0.1, 0.9); // near 1 for stability
    let a_data = lcg_vec(52, b * t * h, 0.2, -0.1);
    let b_data = lcg_vec(53, b * t * h * n, 0.3, -0.15);
    let c_data = lcg_vec(54, b * t * h * n, 0.3, -0.15);

    let px_in = bk
        .param_from_slice_f32(&x_in_data, &[b, t, h * p])
        .expect("px_in");
    let pw_norm = bk
        .param_from_slice_f32(&w_norm_data, &[h * p])
        .expect("pw_norm");
    let pa = bk.param_from_slice_f32(&a_data, &[b, t, h]).expect("pa");
    let pb = bk.param_from_slice_f32(&b_data, &[b, t, h, n]).expect("pb");
    let pc = bk.param_from_slice_f32(&c_data, &[b, t, h, n]).expect("pc");

    // Forward: rmsnorm -> reshape -> ssd_scan.
    let x_norm = bk
        .rmsnorm(px_in.as_tensor(), pw_norm.as_tensor(), 1e-5)
        .expect("rmsnorm");
    let x_4d = bk.reshape(&x_norm, &[b, t, h, p]).expect("reshape to 4d");
    let y = bk
        .ssd_scan(&x_4d, pa.as_tensor(), pb.as_tensor(), pc.as_tensor(), q)
        .expect("ssd_scan");
    let loss = bk.sum_all(&y).expect("sum_all");

    // Backward.
    let store = bk.backward(&loss).expect("backward");
    assert_eq!(bk.tape_len(), 0, "tape not cleared after backward");

    // Check all params have finite grads.
    for (name, param_ref) in [
        ("x_in", &px_in),
        ("w_norm", &pw_norm),
        ("a", &pa),
        ("b", &pb),
        ("c", &pc),
    ] {
        let g = bk
            .gradient(&store, param_ref)
            .expect("gradient")
            .expect("Some");
        let gv = bk.to_vec_f32(&g).expect("to_vec");
        assert!(
            gv.iter().all(|v| v.is_finite()),
            "grad_{name} contains NaN/Inf"
        );
        let max_abs = gv.iter().map(|v| v.abs()).fold(0f32, f32::max);
        // M2: zero-grad degeneration check — at least one element must be non-trivial.
        assert!(
            max_abs > 1e-6,
            "grad_{name} appears to be zero (max_abs={max_abs:.3e})"
        );
        eprintln!("ssd_scan_with_rmsnorm_e2e: grad_{name} max_abs={max_abs:.3e}");
    }

    // SGD step.
    for param_ref in [&px_in, &pw_norm, &pa, &pb, &pc] {
        if let Some(g) = bk.gradient(&store, param_ref).expect("grad for sgd") {
            bk.sgd_step(param_ref, &g, 0.01).expect("sgd_step");
        }
    }

    // Second forward+backward (exercises fresh leaf re-registration post-SGD).
    let x_norm2 = bk
        .rmsnorm(px_in.as_tensor(), pw_norm.as_tensor(), 1e-5)
        .expect("second rmsnorm");
    let x_4d2 = bk.reshape(&x_norm2, &[b, t, h, p]).expect("reshape2");
    let y2 = bk
        .ssd_scan(&x_4d2, pa.as_tensor(), pb.as_tensor(), pc.as_tensor(), q)
        .expect("ssd_scan2");
    let loss2 = bk.sum_all(&y2).expect("sum_all2");
    let _store2 = bk.backward(&loss2).expect("second backward");
    assert_eq!(bk.tape_len(), 0, "tape not cleared after second backward");

    eprintln!("ssd_scan_with_rmsnorm_e2e: PASS");
}

// ---- Test: ssd_scan_direct_e2e_no_nan ----------------------------------------

/// Minimal E2E without the full block: just x -> ssd_scan -> sum_all -> backward.
/// Verifies that the backward produces exactly 4 param gradients (x, a, b, c).
#[test]
fn ssd_scan_direct_e2e_no_nan() {
    let (batch, t, h, p, n, q) = (2usize, 8, 2, 4, 4, 4);

    let bk = CudaBackend::new(0).expect("CUDA init");

    let x_data = lcg_vec(101, batch * t * h * p, 0.3, -0.15);
    let a_data = lcg_vec(102, batch * t * h, 0.2, -0.1);
    let b_data = lcg_vec(103, batch * t * h * n, 0.3, -0.15);
    let c_data = lcg_vec(104, batch * t * h * n, 0.3, -0.15);

    let px = bk
        .param_from_slice_f32(&x_data, &[batch, t, h, p])
        .expect("px");
    let pa = bk
        .param_from_slice_f32(&a_data, &[batch, t, h])
        .expect("pa");
    let pb = bk
        .param_from_slice_f32(&b_data, &[batch, t, h, n])
        .expect("pb");
    let pc = bk
        .param_from_slice_f32(&c_data, &[batch, t, h, n])
        .expect("pc");

    let y = bk
        .ssd_scan(
            px.as_tensor(),
            pa.as_tensor(),
            pb.as_tensor(),
            pc.as_tensor(),
            q,
        )
        .expect("ssd_scan");
    let loss = bk.sum_all(&y).expect("sum_all");
    let store = bk.backward(&loss).expect("backward");

    assert_eq!(bk.tape_len(), 0, "tape not cleared");

    for (name, param_ref) in [("x", &px), ("a", &pa), ("b", &pb), ("c", &pc)] {
        let g = bk
            .gradient(&store, param_ref)
            .expect("gradient")
            .expect("Some");
        let gv = bk.to_vec_f32(&g).expect("to_vec");
        assert!(
            gv.iter().all(|v| v.is_finite()),
            "grad_{name} contains NaN/Inf"
        );
        let max_abs = gv.iter().map(|v| v.abs()).fold(0f32, f32::max);
        // M2: zero-grad degeneration check — at least one element must be non-trivial.
        assert!(
            max_abs > 1e-6,
            "grad_{name} appears to be zero (max_abs={max_abs:.3e})"
        );
        eprintln!("ssd_scan_direct_e2e_no_nan: grad_{name} max_abs={max_abs:.3e}");
    }
}
