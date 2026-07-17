//! Numerical correctness tests for the B'.2b DIRECT-BROADCAST matmul path.
//!
//! Before B'.2b, any `needs_broadcast_a || needs_broadcast_b` triggered a
//! host round-trip (D2H → expand → H2D → gemm). For 1-vs-N batch broadcast
//! the fix falls through to `kernels::matmul_f32` directly, which handles it
//! via stride-0 in `gemm_strided_batched` — no host allocation.
//!
//! Test cases (both directions of 3D 1-vs-N broadcast):
//!   a=[4,3,6] @ b=[1,6,5] → out=[4,3,5]   (batch_b == 1)
//!   a=[1,3,6] @ b=[4,6,5] → out=[4,3,5]   (batch_a == 1)
//!
//! Forward: compared against Candle CPU reference (tol 1e-5).
//! Backward:
//!   grad_A and grad_B must both be non-zero and match the analytical
//!   expressions (tol 1e-4). For the b==1 case grad_B also verifies that
//!   `sum_to_shape` correctly reduces the expanded [4,6,5] gradient back to
//!   [1,6,5]: grad_B[0,k,n] = Σ_{batch,i} A[batch,i,k].

#![cfg(feature = "cuda")]

use pm_candle::CandleBackend;
use pm_core::{Ops, Param, Tensor};
use pm_cuda::CudaBackend;

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in max_abs_err");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn lcg(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map [0, u32::MAX] → [-1, 1]
            ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

// ── Forward ──────────────────────────────────────────────────────────────────

/// `a=[4,3,6] @ b=[1,6,5]` (batch_b == 1): DIRECT-BROADCAST path.
#[test]
fn direct_broadcast_b1_forward() {
    let cpu = CandleBackend::new_cpu();
    let bk = CudaBackend::new(0).expect("CUDA init");

    let a_data = lcg(4 * 3 * 6, 11);
    let b_data = lcg(1 * 6 * 5, 17);

    // Candle CPU reference.
    let a_ref = cpu.from_slice_f32(&a_data, &[4, 3, 6]).unwrap();
    let b_ref = cpu.from_slice_f32(&b_data, &[1, 6, 5]).unwrap();
    let c_ref = cpu.matmul(&a_ref, &b_ref).unwrap();
    let ref_vals = cpu.to_vec_f32(&c_ref).unwrap();

    // CUDA path (must take DIRECT-BROADCAST after B'.2b).
    let a_gpu = Ops::from_slice_f32(&bk, &a_data, &[4, 3, 6]).unwrap();
    let b_gpu = Ops::from_slice_f32(&bk, &b_data, &[1, 6, 5]).unwrap();
    let c_gpu = Ops::matmul(&bk, &a_gpu, &b_gpu).unwrap();
    assert_eq!(c_gpu.shape(), &[4, 3, 5], "shape mismatch");

    let gpu_vals = Ops::to_vec_f32(&bk, &c_gpu).unwrap();
    let err = max_abs_err(&gpu_vals, &ref_vals);
    assert!(
        err < 1e-5,
        "direct_broadcast_b1_forward: max_abs_err={err:.2e} (limit 1e-5)"
    );
}

/// `a=[1,3,6] @ b=[4,6,5]` (batch_a == 1): DIRECT-BROADCAST path.
#[test]
fn direct_broadcast_a1_forward() {
    let cpu = CandleBackend::new_cpu();
    let bk = CudaBackend::new(0).expect("CUDA init");

    let a_data = lcg(1 * 3 * 6, 31);
    let b_data = lcg(4 * 6 * 5, 37);

    let a_ref = cpu.from_slice_f32(&a_data, &[1, 3, 6]).unwrap();
    let b_ref = cpu.from_slice_f32(&b_data, &[4, 6, 5]).unwrap();
    let c_ref = cpu.matmul(&a_ref, &b_ref).unwrap();
    let ref_vals = cpu.to_vec_f32(&c_ref).unwrap();

    let a_gpu = Ops::from_slice_f32(&bk, &a_data, &[1, 3, 6]).unwrap();
    let b_gpu = Ops::from_slice_f32(&bk, &b_data, &[4, 6, 5]).unwrap();
    let c_gpu = Ops::matmul(&bk, &a_gpu, &b_gpu).unwrap();
    assert_eq!(c_gpu.shape(), &[4, 3, 5], "shape mismatch");

    let gpu_vals = Ops::to_vec_f32(&bk, &c_gpu).unwrap();
    let err = max_abs_err(&gpu_vals, &ref_vals);
    assert!(
        err < 1e-5,
        "direct_broadcast_a1_forward: max_abs_err={err:.2e} (limit 1e-5)"
    );
}

// ── Backward (batch_b == 1) ───────────────────────────────────────────────────
//
// `a=[4,3,6] @ b=[1,6,5]` — loss = Σ out
//
// Analytical gradients (g = ones):
//   ∂loss/∂A[batch,i,k] = Σ_n B[0,k,n]
//   ∂loss/∂B[0,k,n]     = Σ_{batch,i} A[batch,i,k]
//
// Note: ∂loss/∂B[0,k,n] is independent of n — the B gradient column is
// constant across output features, because each output position n receives
// the same unit upstream gradient.

/// grad_A for `a=[4,3,6] @ b=[1,6,5]`.
#[test]
fn direct_broadcast_b1_grad_a() {
    let bk = CudaBackend::new(0).expect("CUDA init");

    let a_data = lcg(4 * 3 * 6, 51);
    let b_data = lcg(1 * 6 * 5, 57);

    let pa = bk
        .param_from_slice_f32(&a_data, &[4, 3, 6])
        .expect("param_a");
    let b_t = Ops::from_slice_f32(&bk, &b_data, &[1, 6, 5]).expect("b");
    let out = Ops::matmul(&bk, pa.as_tensor(), &b_t).expect("matmul");
    assert_eq!(out.shape(), &[4, 3, 5]);
    let loss = Ops::sum_all(&bk, &out).expect("sum");
    let store = bk.backward(&loss).expect("backward");
    let ga = bk
        .gradient(&store, &pa)
        .expect("gradient")
        .expect("grad_A must be Some");
    assert_eq!(ga.shape(), &[4, 3, 6], "grad_A shape: {:?}", ga.shape());

    let got = Ops::to_vec_f32(&bk, &ga).expect("to_vec");

    // Σ_n B[0,k,n] for each k.
    let b_row_sums: Vec<f32> = (0..6)
        .map(|k| (0..5usize).map(|n| b_data[k * 5 + n]).sum::<f32>())
        .collect();
    let expected: Vec<f32> = (0..4 * 3 * 6).map(|idx| b_row_sums[idx % 6]).collect();

    let err = max_abs_err(&got, &expected);
    assert!(
        err < 1e-4,
        "direct_broadcast_b1 grad_A: max_err={err:.2e}\ngot={got:?}\nexp={expected:?}"
    );
}

/// grad_B for `a=[4,3,6] @ b=[1,6,5]` — verifies batch sum-reduce via
/// `sum_to_shape`: grad_B.shape must be [1,6,5].
#[test]
fn direct_broadcast_b1_grad_b() {
    let bk = CudaBackend::new(0).expect("CUDA init");

    let a_data = lcg(4 * 3 * 6, 61);
    let b_data = lcg(1 * 6 * 5, 67);

    let a_t = Ops::from_slice_f32(&bk, &a_data, &[4, 3, 6]).expect("a");
    let pb = bk
        .param_from_slice_f32(&b_data, &[1, 6, 5])
        .expect("param_b");
    let out = Ops::matmul(&bk, &a_t, pb.as_tensor()).expect("matmul");
    let loss = Ops::sum_all(&bk, &out).expect("sum");
    let store = bk.backward(&loss).expect("backward");
    let gb = bk
        .gradient(&store, &pb)
        .expect("gradient")
        .expect("grad_B must be Some");
    // sum_to_shape must have reduced [4,6,5] → [1,6,5].
    assert_eq!(gb.shape(), &[1, 6, 5], "grad_B shape: {:?}", gb.shape());

    let got = Ops::to_vec_f32(&bk, &gb).expect("to_vec");

    // Σ_{batch=0..4, i=0..3} A[batch,i,k] for each (k,n).
    let mut expected = vec![0.0f32; 1 * 6 * 5];
    for batch in 0..4usize {
        for i in 0..3usize {
            for k in 0..6usize {
                let v = a_data[batch * 3 * 6 + i * 6 + k];
                // grad_B[0,k,n] accumulates the same A value for every n.
                for n in 0..5usize {
                    expected[k * 5 + n] += v;
                }
            }
        }
    }
    let err = max_abs_err(&got, &expected);
    assert!(
        err < 1e-4,
        "direct_broadcast_b1 grad_B: max_err={err:.2e}\ngot={got:?}\nexp={expected:?}"
    );
}

// ── Backward (batch_a == 1) ───────────────────────────────────────────────────
//
// `a=[1,3,6] @ b=[4,6,5]` — loss = Σ out
//
// Analytical gradients (g = ones):
//   ∂loss/∂B[batch,k,n] = Σ_i A[0,i,k]
//   ∂loss/∂A[0,i,k]     = Σ_{batch,n} B[batch,k,n]  (batch-summed via sum_to_shape)

/// grad_B for `a=[1,3,6] @ b=[4,6,5]`.
#[test]
fn direct_broadcast_a1_grad_b() {
    let bk = CudaBackend::new(0).expect("CUDA init");

    let a_data = lcg(1 * 3 * 6, 71);
    let b_data = lcg(4 * 6 * 5, 77);

    let a_t = Ops::from_slice_f32(&bk, &a_data, &[1, 3, 6]).expect("a");
    let pb = bk
        .param_from_slice_f32(&b_data, &[4, 6, 5])
        .expect("param_b");
    let out = Ops::matmul(&bk, &a_t, pb.as_tensor()).expect("matmul");
    assert_eq!(out.shape(), &[4, 3, 5]);
    let loss = Ops::sum_all(&bk, &out).expect("sum");
    let store = bk.backward(&loss).expect("backward");
    let gb = bk
        .gradient(&store, &pb)
        .expect("gradient")
        .expect("grad_B must be Some");
    assert_eq!(gb.shape(), &[4, 6, 5], "grad_B shape: {:?}", gb.shape());

    let got = Ops::to_vec_f32(&bk, &gb).expect("to_vec");

    // Σ_i A[0,i,k] for each k — same value for every n.
    let a_col_sums: Vec<f32> = (0..6)
        .map(|k| (0..3usize).map(|i| a_data[i * 6 + k]).sum::<f32>())
        .collect();
    // grad_B[batch, k, n] = a_col_sums[k]  (independent of batch and n).
    let expected: Vec<f32> = (0..4 * 6 * 5)
        .map(|idx| {
            let k = (idx / 5) % 6;
            a_col_sums[k]
        })
        .collect();
    let err = max_abs_err(&got, &expected);
    assert!(
        err < 1e-4,
        "direct_broadcast_a1 grad_B: max_err={err:.2e}\ngot={got:?}\nexp={expected:?}"
    );
}

/// grad_A for `a=[1,3,6] @ b=[4,6,5]` — verifies batch sum-reduce:
/// grad_A.shape must be [1,3,6].
#[test]
fn direct_broadcast_a1_grad_a() {
    let bk = CudaBackend::new(0).expect("CUDA init");

    let a_data = lcg(1 * 3 * 6, 81);
    let b_data = lcg(4 * 6 * 5, 87);

    let pa = bk
        .param_from_slice_f32(&a_data, &[1, 3, 6])
        .expect("param_a");
    let b_t = Ops::from_slice_f32(&bk, &b_data, &[4, 6, 5]).expect("b");
    let out = Ops::matmul(&bk, pa.as_tensor(), &b_t).expect("matmul");
    let loss = Ops::sum_all(&bk, &out).expect("sum");
    let store = bk.backward(&loss).expect("backward");
    let ga = bk
        .gradient(&store, &pa)
        .expect("gradient")
        .expect("grad_A must be Some");
    // sum_to_shape must have reduced [4,3,6] → [1,3,6].
    assert_eq!(ga.shape(), &[1, 3, 6], "grad_A shape: {:?}", ga.shape());

    let got = Ops::to_vec_f32(&bk, &ga).expect("to_vec");

    // Σ_{batch=0..4, n=0..5} B[batch,k,n] for each (i,k).
    let mut expected = vec![0.0f32; 1 * 3 * 6];
    for i in 0..3usize {
        for k in 0..6usize {
            let mut acc = 0.0f32;
            for batch in 0..4usize {
                for n in 0..5usize {
                    acc += b_data[batch * 6 * 5 + k * 5 + n];
                }
            }
            // grad_A[0,i,k] = acc (same for every i since g=ones makes it
            // independent of i; the formula here confirms this numerically).
            expected[i * 6 + k] = acc;
        }
    }
    let err = max_abs_err(&got, &expected);
    assert!(
        err < 1e-4,
        "direct_broadcast_a1 grad_A: max_err={err:.2e}\ngot={got:?}\nexp={expected:?}"
    );
}
