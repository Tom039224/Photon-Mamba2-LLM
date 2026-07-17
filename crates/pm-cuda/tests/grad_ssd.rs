//! Autograd correctness tests for `CudaBackend::ssd_scan` backward — B4.3c.
//!
//! Three tests:
//! 1. `ssd_scan_backward_matches_candle` — all four inputs are tracked params;
//!    compare autograd grads against pm-candle (CPU) with tol 1e-3.
//! 2. `ssd_scan_grad_x_only_via_fd` — only `x` is a param; check against
//!    finite-difference (tol 1e-3).
//! 3. `ssd_scan_block_full_seq` — block_len >= t (dense path); verify backward
//!    runs without panic/NaN and produces finite grads.

#![cfg(feature = "cuda")]

use pm_core::{Ops, Param};
use pm_cuda::CudaBackend;

// ---- helpers ----------------------------------------------------------------

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "max_abs_err: length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Deterministic LCG — matches the pattern in `ssd_parity.rs`.
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

/// Finite-difference gradient estimate for `loss = sum_all(f(x))`.
fn fd_grad<F>(x: &[f32], eps: f32, f: F) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let mut grad = vec![0.0f32; x.len()];
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    for i in 0..x.len() {
        xp[i] = x[i] + eps;
        xm[i] = x[i] - eps;
        grad[i] = (f(&xp) - f(&xm)) / (2.0 * eps);
        xp[i] = x[i];
        xm[i] = x[i];
    }
    grad
}

// ---- Test 1: ssd_scan_backward_matches_candle --------------------------------

/// All four inputs tracked; compare pm-cuda autograd grads with pm-candle
/// (which uses pure-Ops autograd via candle_core::Var).
#[test]
fn ssd_scan_backward_matches_candle() {
    // Small shape: B=1, T=8, H=2, P=4, N=4, block_len=4.
    let (b, t, h, p, n, q) = (1usize, 8, 2, 4, 4, 4);

    let x_data = lcg_vec(1, b * t * h * p, 0.5, -0.25);
    // Keep A small-negative so exp doesn't blow up.
    let a_data = lcg_vec(2, b * t * h, 0.3, -0.3);
    let b_data = lcg_vec(3, b * t * h * n, 0.5, -0.25);
    let c_data = lcg_vec(4, b * t * h * n, 0.5, -0.25);

    // ----- pm-cuda -----
    let cuda_bk = CudaBackend::new(0).expect("CUDA init");

    let px = cuda_bk
        .param_from_slice_f32(&x_data, &[b, t, h, p])
        .expect("param x");
    let pa = cuda_bk
        .param_from_slice_f32(&a_data, &[b, t, h])
        .expect("param a");
    let pb = cuda_bk
        .param_from_slice_f32(&b_data, &[b, t, h, n])
        .expect("param b");
    let pc = cuda_bk
        .param_from_slice_f32(&c_data, &[b, t, h, n])
        .expect("param c");

    let y = cuda_bk
        .ssd_scan(
            px.as_tensor(),
            pa.as_tensor(),
            pb.as_tensor(),
            pc.as_tensor(),
            q,
        )
        .expect("ssd_scan forward");
    let loss = cuda_bk.sum_all(&y).expect("sum_all");
    let store = cuda_bk.backward(&loss).expect("backward");

    let cuda_gx = cuda_bk
        .to_vec_f32(
            &cuda_bk
                .gradient(&store, &px)
                .expect("grad x")
                .expect("Some"),
        )
        .expect("to_vec gx");
    let cuda_ga = cuda_bk
        .to_vec_f32(
            &cuda_bk
                .gradient(&store, &pa)
                .expect("grad a")
                .expect("Some"),
        )
        .expect("to_vec ga");
    let cuda_gb = cuda_bk
        .to_vec_f32(
            &cuda_bk
                .gradient(&store, &pb)
                .expect("grad b")
                .expect("Some"),
        )
        .expect("to_vec gb");
    let cuda_gc = cuda_bk
        .to_vec_f32(
            &cuda_bk
                .gradient(&store, &pc)
                .expect("grad c")
                .expect("Some"),
        )
        .expect("to_vec gc");

    // ----- pm-candle (CPU, pure-Ops autograd reference) -----
    use pm_candle::CandleBackend;
    use pm_core::Ops as _;

    let candle_bk = CandleBackend::new_cpu();

    let cpx = candle_bk
        .param_from_slice_f32(&x_data, &[b, t, h, p])
        .expect("candle param x");
    let cpa = candle_bk
        .param_from_slice_f32(&a_data, &[b, t, h])
        .expect("candle param a");
    let cpb = candle_bk
        .param_from_slice_f32(&b_data, &[b, t, h, n])
        .expect("candle param b");
    let cpc = candle_bk
        .param_from_slice_f32(&c_data, &[b, t, h, n])
        .expect("candle param c");

    let cy = candle_bk
        .ssd_scan(
            cpx.as_tensor(),
            cpa.as_tensor(),
            cpb.as_tensor(),
            cpc.as_tensor(),
            q,
        )
        .expect("candle ssd_scan");
    let closs = candle_bk.sum_all(&cy).expect("candle sum_all");
    let cstore = candle_bk.backward(&closs).expect("candle backward");

    let candle_gx = candle_bk
        .to_vec_f32(
            &candle_bk
                .gradient(&cstore, &cpx)
                .expect("grad x")
                .expect("Some"),
        )
        .expect("candle gx vec");
    let candle_ga = candle_bk
        .to_vec_f32(
            &candle_bk
                .gradient(&cstore, &cpa)
                .expect("grad a")
                .expect("Some"),
        )
        .expect("candle ga vec");
    let candle_gb = candle_bk
        .to_vec_f32(
            &candle_bk
                .gradient(&cstore, &cpb)
                .expect("grad b")
                .expect("Some"),
        )
        .expect("candle gb vec");
    let candle_gc = candle_bk
        .to_vec_f32(
            &candle_bk
                .gradient(&cstore, &cpc)
                .expect("grad c")
                .expect("Some"),
        )
        .expect("candle gc vec");

    // Tolerance 1e-3: recompute path vs Candle may differ in op order.
    let tol = 1e-3f32;

    let err_x = max_abs_err(&cuda_gx, &candle_gx);
    let err_a = max_abs_err(&cuda_ga, &candle_ga);
    let err_b = max_abs_err(&cuda_gb, &candle_gb);
    let err_c = max_abs_err(&cuda_gc, &candle_gc);

    eprintln!(
        "ssd_scan_backward_matches_candle: err_x={:.2e} err_a={:.2e} err_b={:.2e} err_c={:.2e}",
        err_x, err_a, err_b, err_c
    );

    assert!(
        err_x < tol,
        "grad_x mismatch: max_abs_err={err_x:.4e} >= tol={tol}"
    );
    assert!(
        err_a < tol,
        "grad_a mismatch: max_abs_err={err_a:.4e} >= tol={tol}"
    );
    assert!(
        err_b < tol,
        "grad_b mismatch: max_abs_err={err_b:.4e} >= tol={tol}"
    );
    assert!(
        err_c < tol,
        "grad_c mismatch: max_abs_err={err_c:.4e} >= tol={tol}"
    );
}

// ---- Test 2: ssd_scan_grad_x_only_via_fd ------------------------------------

/// Only `x` is a tracked param; a/b/c are detached constants.
/// Compare grad_x against finite difference.
#[test]
fn ssd_scan_grad_x_only_via_fd() {
    let (b, t, h, p, n, q) = (1usize, 4, 1, 2, 2, 4);

    let x_data = lcg_vec(10, b * t * h * p, 0.4, -0.2);
    let a_data = lcg_vec(11, b * t * h, 0.2, -0.2);
    let b_data = lcg_vec(12, b * t * h * n, 0.4, -0.2);
    let c_data = lcg_vec(13, b * t * h * n, 0.4, -0.2);

    let bk = CudaBackend::new(0).expect("CUDA init");

    // Only x is a param; a/b/c are plain tensors.
    let px = bk
        .param_from_slice_f32(&x_data, &[b, t, h, p])
        .expect("param x");
    let a_t = bk.from_slice_f32(&a_data, &[b, t, h]).expect("a tensor");
    let b_t = bk.from_slice_f32(&b_data, &[b, t, h, n]).expect("b tensor");
    let c_t = bk.from_slice_f32(&c_data, &[b, t, h, n]).expect("c tensor");

    let y = bk
        .ssd_scan(px.as_tensor(), &a_t, &b_t, &c_t, q)
        .expect("ssd_scan");
    let loss = bk.sum_all(&y).expect("sum_all");
    let store = bk.backward(&loss).expect("backward");

    let cuda_gx = bk
        .to_vec_f32(&bk.gradient(&store, &px).expect("grad x").expect("Some"))
        .expect("to_vec");

    // Finite-difference reference using the scalar SSD reference.
    let fd = fd_grad(&x_data, 1e-3, |xv| {
        let y_ref =
            pm_core::mamba2::ssd_scan_naive_scalar(xv, &a_data, &b_data, &c_data, b, t, h, p, n);
        y_ref.iter().sum::<f32>()
    });

    let err = max_abs_err(&cuda_gx, &fd);
    eprintln!("ssd_scan_grad_x_only_via_fd: max_abs_err={err:.3e}");
    assert!(
        err < 1e-3,
        "grad_x FD mismatch: max_abs_err={err:.4e} >= 1e-3"
    );
}

// ---- Test 3: ssd_scan_block_full_seq ----------------------------------------

/// block_len >= t triggers the dense path in ssd_scan_ops_default.
/// Verify that backward runs without panic, all param grads are non-NaN,
/// and the tape is cleared afterwards.
#[test]
fn ssd_scan_block_full_seq() {
    let (b, t, h, p, n) = (1usize, 4, 1, 2, 2);
    // block_len = t or larger → dense path.
    let q = t; // exactly t

    let x_data = lcg_vec(20, b * t * h * p, 0.3, -0.15);
    let a_data = lcg_vec(21, b * t * h, 0.2, -0.1);
    let b_data = lcg_vec(22, b * t * h * n, 0.3, -0.15);
    let c_data = lcg_vec(23, b * t * h * n, 0.3, -0.15);

    let bk = CudaBackend::new(0).expect("CUDA init");

    let px = bk
        .param_from_slice_f32(&x_data, &[b, t, h, p])
        .expect("param x");
    let pa = bk
        .param_from_slice_f32(&a_data, &[b, t, h])
        .expect("param a");
    let pb = bk
        .param_from_slice_f32(&b_data, &[b, t, h, n])
        .expect("param b");
    let pc = bk
        .param_from_slice_f32(&c_data, &[b, t, h, n])
        .expect("param c");

    let y = bk
        .ssd_scan(
            px.as_tensor(),
            pa.as_tensor(),
            pb.as_tensor(),
            pc.as_tensor(),
            q,
        )
        .expect("ssd_scan dense path");
    let loss = bk.sum_all(&y).expect("sum_all");
    let store = bk.backward(&loss).expect("backward dense");

    // Tape must be cleared after backward.
    assert_eq!(bk.tape_len(), 0, "tape not cleared after backward");

    // All params must have finite grads.
    for (name, param) in [("x", &px), ("a", &pa), ("b", &pb), ("c", &pc)] {
        let g = bk.gradient(&store, param).expect("get grad").expect("Some");
        let gv = bk.to_vec_f32(&g).expect("to_vec");
        let all_finite = gv.iter().all(|v| v.is_finite());
        assert!(
            all_finite,
            "ssd_scan_block_full_seq: grad_{name} contains NaN/Inf"
        );
        eprintln!(
            "ssd_scan_block_full_seq: grad_{name} max_abs={:.3e}",
            gv.iter().map(|v| v.abs()).fold(0f32, f32::max)
        );
    }
}
