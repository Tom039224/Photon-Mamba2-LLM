//! Autograd correctness tests for B4.3b algebra/shape/reduce backward.
//!
//! Each test verifies a VJP rule using finite-difference comparison.
//!
//! FD formula: `(f(x + eps) - f(x - eps)) / (2 * eps)`.
//! Tolerance: `1e-3` (adequate for f32 GPU kernels + FD noise).

#![cfg(feature = "cuda")]

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

/// Finite-difference gradient for a function that takes a flat f32 slice
/// and returns a scalar.
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

// ---- Test: matmul_grad_2d_matches_fd ----------------------------------------
// A (2×3), B (3×2): loss = sum(A @ B).
// grad_A = g @ B^T where g = ones(2,2) → each row of A gets sum of B cols.
//   grad_A[i,j] = sum_n B[j,n] = B[j,0] + B[j,1]
//   = [0.3, 0.7, 1.1, 0.3, 0.7, 1.1]
// grad_B = A^T @ g where g = ones(2,2) → each col of B gets sum of A rows.
//   grad_B[j,n] = sum_i A[i,j]
//   = [[5,5], [7,7], [9,9]]

#[test]
fn matmul_grad_2d_matches_fd() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
    let a_shape = [2, 3];
    let b_shape = [3, 2];

    // Autograd grad_A — compare against analytical
    {
        let bk2 = CudaBackend::new(0).expect("CUDA init");
        let pa = bk2
            .param_from_slice_f32(&a_data, &a_shape)
            .expect("param a");
        let b_t = bk2.from_slice_f32(&b_data, &b_shape).expect("b tensor");
        let out = bk2.matmul(pa.as_tensor(), &b_t).expect("matmul");
        let loss = bk2.sum_all(&out).expect("sum_all");
        let store = bk2.backward(&loss).expect("backward");
        let grad_a = bk2.gradient(&store, &pa).expect("ga").expect("Some");
        let got_ga = bk2.to_vec_f32(&grad_a).expect("to_vec");

        // Analytical: grad_A = ones(2,2) @ B^T, B^T[n,j] = B[j,n]
        // grad_A[i,j] = sum_n B[j,n] = B[j,0] + B[j,1]
        // Row 0 and row 1 of grad_A are identical.
        let expected_ga = [
            b_data[0] + b_data[1],
            b_data[2] + b_data[3],
            b_data[4] + b_data[5],
            b_data[0] + b_data[1],
            b_data[2] + b_data[3],
            b_data[4] + b_data[5],
        ];
        assert!(
            max_abs_err(&got_ga, &expected_ga) < 1e-4,
            "matmul_grad_2d: grad_A mismatch. got={got_ga:?} expected={expected_ga:?}"
        );
    }

    // Autograd grad_B — compare against analytical
    {
        let bk3 = CudaBackend::new(0).expect("CUDA init");
        let a_t = bk3.from_slice_f32(&a_data, &a_shape).expect("a tensor");
        let pb = bk3
            .param_from_slice_f32(&b_data, &b_shape)
            .expect("param b");
        let out = bk3.matmul(&a_t, pb.as_tensor()).expect("matmul");
        let loss = bk3.sum_all(&out).expect("sum_all");
        let store = bk3.backward(&loss).expect("backward");
        let grad_b = bk3.gradient(&store, &pb).expect("gb").expect("Some");
        let got_gb = bk3.to_vec_f32(&grad_b).expect("to_vec");

        // Analytical: grad_B = A^T @ ones(2,2), A^T[j,i] = A[i,j]
        // grad_B[j,n] = sum_i A[i,j] = A[0,j] + A[1,j] (same for each n)
        let expected_gb = [
            a_data[0] + a_data[3],
            a_data[0] + a_data[3], // j=0
            a_data[1] + a_data[4],
            a_data[1] + a_data[4], // j=1
            a_data[2] + a_data[5],
            a_data[2] + a_data[5], // j=2
        ];
        assert!(
            max_abs_err(&got_gb, &expected_gb) < 1e-4,
            "matmul_grad_2d: grad_B mismatch. got={got_gb:?} expected={expected_gb:?}"
        );
    }
    drop(bk);
}

/// Small host-side matmul for FD reference: A(M×K) @ B(K×N) → C(M×N).
#[allow(dead_code)]
fn ndarray_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += a[i * k + ki] * b[ki * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

// ---- Test: matmul_grad_3d_batched -------------------------------------------
// Batch=2, A(2×3×4), B(2×4×2): loss = sum(A@B).
// grad_A[b,i,j] = sum_n g[b,i,n] * B[b,j,n] = sum_n B[b,j,n] (g=ones).

#[test]
fn matmul_grad_3d_batched() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let batch = 2;
    let m = 3;
    let k = 4;
    let n = 2;
    let a_data: Vec<f32> = (0..batch * m * k).map(|i| (i + 1) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..batch * k * n).map(|i| (i + 1) as f32 * 0.05).collect();

    let pa = bk
        .param_from_slice_f32(&a_data, &[batch, m, k])
        .expect("pa");
    let b_t = bk.from_slice_f32(&b_data, &[batch, k, n]).expect("b");
    let out = bk.matmul(pa.as_tensor(), &b_t).expect("matmul");
    let loss = bk.sum_all(&out).expect("sum");
    let store = bk.backward(&loss).expect("bwd");
    let ga = bk.gradient(&store, &pa).expect("ga").expect("Some");
    let got = bk.to_vec_f32(&ga).expect("tv");

    // Analytical: grad_A[b,i,j] = sum_n B[b,j,n] (since g = ones(2,3,2))
    // = sum of row j of B[b] over its n=2 columns.
    let mut expected = vec![0.0f32; batch * m * k];
    for b_idx in 0..batch {
        for row_i in 0..m {
            for col_j in 0..k {
                // grad_A[b,row,col] = sum_n B[b,col,n]
                let b_base = b_idx * k * n + col_j * n;
                let val: f32 = (0..n).map(|ni| b_data[b_base + ni]).sum();
                expected[b_idx * m * k + row_i * k + col_j] = val;
            }
        }
    }
    assert!(
        max_abs_err(&got, &expected) < 1e-4,
        "matmul_3d: grad_A mismatch. max_err={} got={got:?} expected={expected:?}",
        max_abs_err(&got, &expected)
    );
}

// ---- Test: transpose_grad ---------------------------------------------------
// p (2×3), loss = sum(p^T * p^T) = sum(p^T * p^T).
// Simple test: p (3×2), transpose to (2×3), mul_scalar 2, loss = sum.
// grad_p = transpose(2 * ones(2×3), 0, 1) = 2 * ones(3×2).

#[test]
fn transpose_grad() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let p = bk.param_from_slice_f32(&data, &[3, 2]).expect("param");
    let pt = p.as_tensor();
    let t = bk.transpose(pt, 0, 1).expect("transpose"); // (2, 3)
    let y = bk.mul_scalar(&t, 2.0).expect("mul_scalar");
    let loss = bk.sum_all(&y).expect("sum");
    let store = bk.backward(&loss).expect("bwd");
    let grad = bk.gradient(&store, &p).expect("g").expect("Some");
    let got = bk.to_vec_f32(&grad).expect("tv");

    // Analytical: transpose is linear, grad is the transpose of g * 2.
    // g = ones(2,3), mul_scalar 2 → g_in = 2 * ones(2,3).
    // grad_p = transpose(g_in, 0, 1) = 2 * ones(3,2).
    let expected = [2.0f32; 6];
    assert!(
        max_abs_err(&got, &expected) < 1e-4,
        "transpose_grad: mismatch. got={got:?} expected={expected:?}"
    );
}

// ---- Test: narrow_grad -------------------------------------------------------
// p (6,), narrow [2..5] (len=3), mul_scalar 3, sum.
// grad_p[2..5] = 3, rest = 0.

#[test]
fn narrow_grad() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let p = bk.param_from_slice_f32(&data, &[6]).expect("param");
    let pt = p.as_tensor();
    let n = bk.narrow(pt, 0, 2, 3).expect("narrow"); // [3.0, 4.0, 5.0]
    let y = bk.mul_scalar(&n, 3.0).expect("mul");
    let loss = bk.sum_all(&y).expect("sum");
    let store = bk.backward(&loss).expect("bwd");
    let grad = bk.gradient(&store, &p).expect("g").expect("Some");
    let got = bk.to_vec_f32(&grad).expect("tv");

    let expected = [0.0f32, 0.0, 3.0, 3.0, 3.0, 0.0];
    assert!(
        max_abs_err(&got, &expected) < 1e-5,
        "narrow_grad: got={got:?} expected={expected:?}"
    );
}

// ---- Test: broadcast_as_grad ------------------------------------------------
// p (1, 3), broadcast to (2, 3), sum.
// grad_p = sum of rows = [2, 2, 2] summed to shape (1, 3).

#[test]
fn broadcast_as_grad() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let data = [1.0f32, 2.0, 3.0];
    let p = bk.param_from_slice_f32(&data, &[1, 3]).expect("param");
    let pt = p.as_tensor();
    let bc = bk.broadcast_as(pt, &[2, 3]).expect("broadcast");
    let loss = bk.sum_all(&bc).expect("sum");
    let store = bk.backward(&loss).expect("bwd");
    let grad = bk.gradient(&store, &p).expect("g").expect("Some");
    let got = bk.to_vec_f32(&grad).expect("tv");

    // grad shape is (1, 3) — each element is 2.0 (summed over 2 rows)
    let expected = [2.0f32, 2.0, 2.0];
    assert_eq!(got.len(), 3);
    assert!(
        max_abs_err(&got, &expected) < 1e-5,
        "broadcast_as_grad: got={got:?} expected={expected:?}"
    );
}

// ---- Test: concat_grad_dim0 -------------------------------------------------
// p1 (2,), p2 (3,) → concat dim=0 → (5,) → mul_scalar 2 → sum.
// grad_p1 = [2, 2], grad_p2 = [2, 2, 2].

#[test]
fn concat_grad_dim0() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let d1 = [1.0f32, 2.0];
    let d2 = [3.0f32, 4.0, 5.0];
    let p1 = bk.param_from_slice_f32(&d1, &[2]).expect("p1");
    let p2 = bk.param_from_slice_f32(&d2, &[3]).expect("p2");
    let cat = bk
        .concat(&[p1.as_tensor(), p2.as_tensor()], 0)
        .expect("concat");
    let y = bk.mul_scalar(&cat, 2.0).expect("mul");
    let loss = bk.sum_all(&y).expect("sum");
    let store = bk.backward(&loss).expect("bwd");

    let g1 = bk
        .to_vec_f32(&bk.gradient(&store, &p1).expect("g1").expect("Some"))
        .expect("tv");
    let g2 = bk
        .to_vec_f32(&bk.gradient(&store, &p2).expect("g2").expect("Some"))
        .expect("tv");

    assert!(
        max_abs_err(&g1, &[2.0f32, 2.0]) < 1e-5,
        "concat dim0 g1={g1:?}"
    );
    assert!(
        max_abs_err(&g2, &[2.0f32, 2.0, 2.0]) < 1e-5,
        "concat dim0 g2={g2:?}"
    );
}

// ---- Test: concat_grad_dim1 -------------------------------------------------
// p1 (2,2), p2 (2,3) → concat dim=1 → (2,5) → sum.
// grad_p1 = ones(2,2), grad_p2 = ones(2,3).

#[test]
fn concat_grad_dim1() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let d1 = [1.0f32; 4];
    let d2 = [1.0f32; 6];
    let p1 = bk.param_from_slice_f32(&d1, &[2, 2]).expect("p1");
    let p2 = bk.param_from_slice_f32(&d2, &[2, 3]).expect("p2");
    let cat = bk
        .concat(&[p1.as_tensor(), p2.as_tensor()], 1)
        .expect("concat");
    let loss = bk.sum_all(&cat).expect("sum");
    let store = bk.backward(&loss).expect("bwd");

    let g1 = bk
        .to_vec_f32(&bk.gradient(&store, &p1).expect("g1").expect("Some"))
        .expect("tv");
    let g2 = bk
        .to_vec_f32(&bk.gradient(&store, &p2).expect("g2").expect("Some"))
        .expect("tv");

    assert!(
        max_abs_err(&g1, &[1.0f32; 4]) < 1e-5,
        "concat dim1 g1={g1:?}"
    );
    assert!(
        max_abs_err(&g2, &[1.0f32; 6]) < 1e-5,
        "concat dim1 g2={g2:?}"
    );
}

// ---- Test: embedding_grad ---------------------------------------------------
// Small vocab=4, embed_dim=3. indices=[0,2]. FD on table rows 0 and 2.

#[test]
fn embedding_grad() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let vocab = 4usize;
    let emb_dim = 3usize;
    let table_data: Vec<f32> = (0..vocab * emb_dim).map(|i| (i + 1) as f32 * 0.1).collect();
    let indices_data = [0i64, 2];

    let pt = bk
        .param_from_slice_f32(&table_data, &[vocab, emb_dim])
        .expect("param table");
    let idx = bk.from_slice_i64(&indices_data, &[2]).expect("idx");
    let out = bk.embedding(pt.as_tensor(), &idx).expect("embedding");
    let loss = bk.sum_all(&out).expect("sum");
    let store = bk.backward(&loss).expect("bwd");
    let g = bk.gradient(&store, &pt).expect("g").expect("Some");
    let got = bk.to_vec_f32(&g).expect("tv");

    // Expected: rows 0 and 2 get gradient 1.0 per element; rows 1, 3 get 0.
    let mut expected = vec![0.0f32; vocab * emb_dim];
    for d in 0..emb_dim {
        expected[0 * emb_dim + d] += 1.0;
        expected[2 * emb_dim + d] += 1.0;
    }
    assert!(
        max_abs_err(&got, &expected) < 1e-5,
        "embedding_grad: got={got:?} expected={expected:?}"
    );
}

// ---- Test: gather_grad -------------------------------------------------------
// src (2, 4), indices (2, 2) selecting last dim, loss = sum(gather).
// grad_src via scatter_add.

#[test]
fn gather_grad() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let src_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let idx_data = [0i64, 2, 1, 3];

    let p = bk.param_from_slice_f32(&src_data, &[2, 4]).expect("p");
    let idx = bk.from_slice_i64(&idx_data, &[2, 2]).expect("idx");
    let out = bk.gather(p.as_tensor(), &idx, 1).expect("gather");
    let loss = bk.sum_all(&out).expect("sum");
    let store = bk.backward(&loss).expect("bwd");
    let grad = bk.gradient(&store, &p).expect("g").expect("Some");
    let got = bk.to_vec_f32(&grad).expect("tv");

    // Analytical: loss = sum of gathered elements.
    // Each element in src that appears in idx gets gradient 1.0.
    // idx = [[0,2],[1,3]] → src rows [0,1] are accessed at cols [0,2] and [1,3].
    // grad_src[row=0, col=0] = 1, [0,1] = 0, [0,2] = 1, [0,3] = 0
    // grad_src[row=1, col=0] = 0, [1,1] = 1, [1,2] = 0, [1,3] = 1
    let expected = [1.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    assert!(
        max_abs_err(&got, &expected) < 1e-5,
        "gather_grad: got={got:?} expected={expected:?}"
    );
}

// ---- Test: conv1d_regular_grad ----------------------------------------------
// B=1, C_in=2, T_in=5, C_out=3, K=2, stride=1, padding=0, groups=1.

#[test]
fn conv1d_regular_grad() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let b = 1;
    let c_in = 2;
    let t_in = 5;
    let c_out = 3;
    let k = 2;
    let stride = 1;
    let padding = 0;
    let groups = 1;

    let x_data: Vec<f32> = (0..b * c_in * t_in).map(|i| (i + 1) as f32 * 0.1).collect();
    let w_data: Vec<f32> = (0..c_out * (c_in / groups) * k)
        .map(|i| (i + 1) as f32 * 0.05)
        .collect();

    // grad_x
    {
        let bk2 = CudaBackend::new(0).expect("CUDA init");
        let px = bk2
            .param_from_slice_f32(&x_data, &[b, c_in, t_in])
            .expect("px");
        let w = bk2
            .from_slice_f32(&w_data, &[c_out, c_in / groups, k])
            .expect("w");
        let out = bk2
            .conv1d(px.as_tensor(), &w, None, stride, padding, groups)
            .expect("conv1d");
        let loss = bk2.sum_all(&out).expect("sum");
        let store = bk2.backward(&loss).expect("bwd");
        let gx = bk2.gradient(&store, &px).expect("gx").expect("Some");
        let got = bk2.to_vec_f32(&gx).expect("tv");

        // Use eps=1e-2 — conv output sum (~12 elements) causes f32 cancellation at small eps.
        let fd = fd_grad(&x_data, 1e-2, |xd| {
            let t_out = (t_in + 2 * padding - k) / stride + 1;
            conv1d_host(
                xd,
                &w_data,
                b,
                c_in,
                t_in,
                c_out,
                c_in / groups,
                k,
                stride,
                padding,
                groups,
                t_out,
            )
            .iter()
            .sum()
        });
        assert!(
            max_abs_err(&got, &fd) < 5e-3,
            "conv1d_regular grad_x: max_err={:.6}",
            max_abs_err(&got, &fd)
        );
    }

    // grad_w
    {
        let bk3 = CudaBackend::new(0).expect("CUDA init");
        let x = bk3.from_slice_f32(&x_data, &[b, c_in, t_in]).expect("x");
        let pw = bk3
            .param_from_slice_f32(&w_data, &[c_out, c_in / groups, k])
            .expect("pw");
        let out = bk3
            .conv1d(&x, pw.as_tensor(), None, stride, padding, groups)
            .expect("conv1d");
        let loss = bk3.sum_all(&out).expect("sum");
        let store = bk3.backward(&loss).expect("bwd");
        let gw = bk3.gradient(&store, &pw).expect("gw").expect("Some");
        let got = bk3.to_vec_f32(&gw).expect("tv");

        let fd = fd_grad(&w_data, 1e-2, |wd| {
            let t_out = (t_in + 2 * padding - k) / stride + 1;
            conv1d_host(
                &x_data,
                wd,
                b,
                c_in,
                t_in,
                c_out,
                c_in / groups,
                k,
                stride,
                padding,
                groups,
                t_out,
            )
            .iter()
            .sum()
        });
        assert!(
            max_abs_err(&got, &fd) < 5e-3,
            "conv1d_regular grad_w: max_err={:.6}",
            max_abs_err(&got, &fd)
        );
    }
    drop(bk);
}

// ---- Test: conv1d_depthwise_grad --------------------------------------------
// B=1, C_in=4, T_in=6, C_out=4, K=3, stride=1, padding=1, groups=4 (depthwise).

#[test]
fn conv1d_depthwise_grad() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let b = 1;
    let c_in = 4;
    let t_in = 6;
    let c_out = 4;
    let k = 3;
    let stride = 1;
    let padding = 1;
    let groups = 4;

    let x_data: Vec<f32> = (0..b * c_in * t_in).map(|i| (i + 1) as f32 * 0.1).collect();
    let c_in_per_group = c_in / groups;
    let w_data: Vec<f32> = (0..c_out * c_in_per_group * k)
        .map(|i| (i + 1) as f32 * 0.1)
        .collect();

    // grad_x
    {
        let bk2 = CudaBackend::new(0).expect("CUDA init");
        let px = bk2
            .param_from_slice_f32(&x_data, &[b, c_in, t_in])
            .expect("px");
        let w = bk2
            .from_slice_f32(&w_data, &[c_out, c_in_per_group, k])
            .expect("w");
        let out = bk2
            .conv1d(px.as_tensor(), &w, None, stride, padding, groups)
            .expect("conv1d");
        let loss = bk2.sum_all(&out).expect("sum");
        let store = bk2.backward(&loss).expect("bwd");
        let gx = bk2.gradient(&store, &px).expect("gx").expect("Some");
        let got = bk2.to_vec_f32(&gx).expect("tv");

        // Use eps=1e-2 — conv output sum causes f32 cancellation at small eps.
        let fd = fd_grad(&x_data, 1e-2, |xd| {
            let t_out = (t_in + 2 * padding - k) / stride + 1;
            conv1d_host(
                xd,
                &w_data,
                b,
                c_in,
                t_in,
                c_out,
                c_in_per_group,
                k,
                stride,
                padding,
                groups,
                t_out,
            )
            .iter()
            .sum()
        });
        assert!(
            max_abs_err(&got, &fd) < 5e-3,
            "conv1d_depthwise grad_x: max_err={:.6}",
            max_abs_err(&got, &fd)
        );
    }
    drop(bk);
}

/// Host-side conv1d for FD reference.
#[allow(clippy::too_many_arguments)]
fn conv1d_host(
    x: &[f32],
    w: &[f32],
    batch: usize,
    c_in: usize,
    t_in: usize,
    c_out: usize,
    c_in_per_group: usize,
    k_size: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    t_out: usize,
) -> Vec<f32> {
    let c_out_per_group = c_out / groups;
    let mut out = vec![0.0f32; batch * c_out * t_out];
    for b in 0..batch {
        for g in 0..groups {
            let c_in_g_start = g * c_in_per_group;
            let c_out_g_start = g * c_out_per_group;
            for co_g in 0..c_out_per_group {
                let co = c_out_g_start + co_g;
                for t_o in 0..t_out {
                    let mut acc = 0.0f32;
                    for ci_g in 0..c_in_per_group {
                        let ci = c_in_g_start + ci_g;
                        for ki in 0..k_size {
                            let t_i = (t_o * stride + ki) as i32 - padding as i32;
                            if t_i < 0 || t_i as usize >= t_in {
                                continue;
                            }
                            let x_val = x[b * c_in * t_in + ci * t_in + t_i as usize];
                            let w_val = w[co * c_in_per_group * k_size + ci_g * k_size + ki];
                            acc += x_val * w_val;
                        }
                    }
                    out[b * c_out * t_out + co * t_out + t_o] = acc;
                }
            }
        }
    }
    out
}

// ---- Test: log_softmax_grad -------------------------------------------------
// p (2, 4), log_softmax along dim=1, sum, backward.

#[test]
fn log_softmax_grad() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let data = [0.5f32, -0.3, 1.2, 0.8, -0.1, 0.9, 0.2, -0.5];
    let p = bk.param_from_slice_f32(&data, &[2, 4]).expect("p");
    let out = bk.log_softmax(p.as_tensor(), 1).expect("log_softmax");
    let loss = bk.sum_all(&out).expect("sum");
    let store = bk.backward(&loss).expect("bwd");
    let grad = bk.gradient(&store, &p).expect("g").expect("Some");
    let got = bk.to_vec_f32(&grad).expect("tv");

    // Use eps=1e-2 for FD to reduce cancellation errors.
    // log_softmax involves exp which amplifies FD noise at small eps.
    let fd = fd_grad(&data, 1e-2, |d| {
        // log_softmax along dim 1 of (2,4)
        let mut s = 0.0f32;
        for row in 0..2usize {
            let row_d = &d[row * 4..(row + 1) * 4];
            let max_v = row_d.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_e: f32 = row_d.iter().map(|&v| (v - max_v).exp()).sum();
            let log_sum = sum_e.ln();
            for &v in row_d {
                s += (v - max_v) - log_sum;
            }
        }
        s
    });
    assert!(
        max_abs_err(&got, &fd) < 5e-3,
        "log_softmax_grad: max_err={:.6}",
        max_abs_err(&got, &fd)
    );
}

// ---- Test: rmsnorm_grad -----------------------------------------------------
// x (2, 4), weight (4,), eps=1e-6. Check both grad_x and grad_w.

#[test]
fn rmsnorm_grad() {
    let _bk = CudaBackend::new(0).expect("CUDA init");
    let x_data = [0.5f32, -0.3, 1.2, 0.8, -0.1, 0.9, 0.2, -0.5];
    let w_data = [1.0f32, 0.5, 0.8, 1.2];
    let eps = 1e-6f32;

    // grad_x
    {
        let bk2 = CudaBackend::new(0).expect("CUDA init");
        let px = bk2.param_from_slice_f32(&x_data, &[2, 4]).expect("px");
        let w = bk2.from_slice_f32(&w_data, &[4]).expect("w");
        let out = bk2.rmsnorm(px.as_tensor(), &w, eps).expect("rmsnorm");
        let loss = bk2.sum_all(&out).expect("sum");
        let store = bk2.backward(&loss).expect("bwd");
        let gx = bk2.gradient(&store, &px).expect("gx").expect("Some");
        let got = bk2.to_vec_f32(&gx).expect("tv");

        // Use eps=1e-3 for FD — rmsnorm involves rsqrt which amplifies cancellation.
        let fd = fd_grad(&x_data, 1e-3, |xd| {
            rmsnorm_host(xd, &w_data, 2, 4, eps).iter().sum()
        });
        assert!(
            max_abs_err(&got, &fd) < 5e-3,
            "rmsnorm grad_x: max_err={:.6}",
            max_abs_err(&got, &fd)
        );
    }

    // grad_w
    {
        let bk3 = CudaBackend::new(0).expect("CUDA init");
        let x = bk3.from_slice_f32(&x_data, &[2, 4]).expect("x");
        let pw = bk3.param_from_slice_f32(&w_data, &[4]).expect("pw");
        let out = bk3.rmsnorm(&x, pw.as_tensor(), eps).expect("rmsnorm");
        let loss = bk3.sum_all(&out).expect("sum");
        let store = bk3.backward(&loss).expect("bwd");
        let gw = bk3.gradient(&store, &pw).expect("gw").expect("Some");
        let got = bk3.to_vec_f32(&gw).expect("tv");

        let fd = fd_grad(&w_data, 1e-3, |wd| {
            rmsnorm_host(&x_data, wd, 2, 4, eps).iter().sum()
        });
        assert!(
            max_abs_err(&got, &fd) < 5e-3,
            "rmsnorm grad_w: max_err={:.6}",
            max_abs_err(&got, &fd)
        );
    }
}

fn rmsnorm_host(x: &[f32], w: &[f32], n_rows: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; n_rows * d];
    for row in 0..n_rows {
        let base = row * d;
        let mean_sq: f32 = x[base..base + d].iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..d {
            out[base + i] = x[base + i] * w[i] * inv;
        }
    }
    out
}

// ---- Test: cumsum_grad_lastdim -----------------------------------------------
// p (2, 4), cumsum dim=1, loss=sum.
// grad = reverse_cumsum(ones(2,4), dim=1).

#[test]
fn cumsum_grad_lastdim() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let p = bk.param_from_slice_f32(&data, &[2, 4]).expect("p");
    let out = bk.cumsum(p.as_tensor(), 1).expect("cumsum");
    let loss = bk.sum_all(&out).expect("sum");
    let store = bk.backward(&loss).expect("bwd");
    let grad = bk.gradient(&store, &p).expect("g").expect("Some");
    let got = bk.to_vec_f32(&grad).expect("tv");

    // Analytical: loss = sum_all(cumsum(x, dim=1)).
    // d(loss)/d(x[row,i]) = number of positions j >= i in the row = (4 - i).
    // So grad = reverse_cumsum(ones) = [4,3,2,1, 4,3,2,1].
    let expected = [4.0f32, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0];
    assert!(
        max_abs_err(&got, &expected) < 1e-5,
        "cumsum_grad: got={got:?} expected={expected:?}"
    );
}

// ---- Test: mean_all_grad ---------------------------------------------------
// p (2, 3), mean_all, loss = mean_all * 2.
// grad_p = ones(2,3) * (1/6) * 2 = ones * 1/3.

#[test]
fn mean_all_grad() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let p = bk.param_from_slice_f32(&data, &[2, 3]).expect("p");
    let m = bk.mean_all(p.as_tensor()).expect("mean_all");
    // loss = mean * 2 (scalar)
    let loss = bk.mul_scalar(&m, 2.0).expect("mul_scalar");
    let store = bk.backward(&loss).expect("bwd");
    let grad = bk.gradient(&store, &p).expect("g").expect("Some");
    let got = bk.to_vec_f32(&grad).expect("tv");

    // grad_p = 2.0 / 6 = 1/3 for each element
    let expected = vec![2.0f32 / 6.0; 6];
    assert!(
        max_abs_err(&got, &expected) < 1e-5,
        "mean_all_grad: got={got:?} expected={expected:?}"
    );
}

// ---- Test: sum_all_grad -----------------------------------------------------
// p (3,), sum_all, scalar loss = sum * 3.
// grad_p = ones * 3.

#[test]
fn sum_all_grad() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let data = [1.0f32, 2.0, 3.0];
    let p = bk.param_from_slice_f32(&data, &[3]).expect("p");
    let s = bk.sum_all(p.as_tensor()).expect("sum_all");
    let loss = bk.mul_scalar(&s, 3.0).expect("mul");
    let store = bk.backward(&loss).expect("bwd");
    let grad = bk.gradient(&store, &p).expect("g").expect("Some");
    let got = bk.to_vec_f32(&grad).expect("tv");

    let expected = [3.0f32, 3.0, 3.0];
    assert!(
        max_abs_err(&got, &expected) < 1e-5,
        "sum_all_grad: got={got:?} expected={expected:?}"
    );
}

// ---- Test: conv1d_regular_grad_with_bias (M5) --------------------------------
// B=1, C_in=2, T_in=5, C_out=3, K=2, stride=1, padding=0, bias=Some.
// grad_bias[co] = sum_{b,t} grad_out[b, co, t].

#[test]
fn conv1d_regular_grad_with_bias() {
    let b = 1;
    let c_in = 2;
    let t_in = 5;
    let c_out = 3;
    let k = 2;
    let stride = 1;
    let padding = 0;
    let groups = 1;
    let t_out = (t_in + 2 * padding - k) / stride + 1; // = 4

    let x_data: Vec<f32> = (0..b * c_in * t_in).map(|i| (i + 1) as f32 * 0.1).collect();
    let w_data: Vec<f32> = (0..c_out * (c_in / groups) * k)
        .map(|i| (i + 1) as f32 * 0.05)
        .collect();
    let bias_data: Vec<f32> = (0..c_out).map(|i| (i as f32 + 1.0) * 0.1).collect();

    // grad_bias: loss = sum_all(conv1d(x, w, bias)), so grad_bias[co] = T_out = 4.
    {
        let bk = CudaBackend::new(0).expect("CUDA init");
        let x = bk.from_slice_f32(&x_data, &[b, c_in, t_in]).expect("x");
        let w = bk
            .from_slice_f32(&w_data, &[c_out, c_in / groups, k])
            .expect("w");
        let pb = bk.param_from_slice_f32(&bias_data, &[c_out]).expect("pb");
        let out = bk
            .conv1d(&x, &w, Some(pb.as_tensor()), stride, padding, groups)
            .expect("conv1d");
        let loss = bk.sum_all(&out).expect("sum");
        let store = bk.backward(&loss).expect("bwd");
        let gb = bk.gradient(&store, &pb).expect("gb").expect("Some");
        let got = bk.to_vec_f32(&gb).expect("tv");

        // Analytical: loss = sum of all output elements.
        // Each bias[co] appears once per (batch, t_out) position, so
        // grad_bias[co] = batch * t_out.
        let expected: Vec<f32> = vec![(b * t_out) as f32; c_out];
        assert!(
            max_abs_err(&got, &expected) < 1e-5,
            "conv1d_bias grad_b: got={got:?} expected={expected:?}"
        );
    }

    // grad_w with bias: bias does not affect grad_w; use FD to confirm.
    {
        let bk = CudaBackend::new(0).expect("CUDA init");
        let x = bk.from_slice_f32(&x_data, &[b, c_in, t_in]).expect("x");
        let pw = bk
            .param_from_slice_f32(&w_data, &[c_out, c_in / groups, k])
            .expect("pw");
        let bias = bk.from_slice_f32(&bias_data, &[c_out]).expect("bias");
        let out = bk
            .conv1d(&x, pw.as_tensor(), Some(&bias), stride, padding, groups)
            .expect("conv1d");
        let loss = bk.sum_all(&out).expect("sum");
        let store = bk.backward(&loss).expect("bwd");
        let gw = bk.gradient(&store, &pw).expect("gw").expect("Some");
        let got = bk.to_vec_f32(&gw).expect("tv");

        let fd = fd_grad(&w_data, 1e-2, |wd| {
            conv1d_host(
                &x_data,
                wd,
                b,
                c_in,
                t_in,
                c_out,
                c_in / groups,
                k,
                stride,
                padding,
                groups,
                t_out,
            )
            .iter()
            .sum::<f32>()
                + bias_data.iter().sum::<f32>() * (b * t_out) as f32
        });
        assert!(
            max_abs_err(&got, &fd) < 5e-3,
            "conv1d_bias grad_w: max_err={:.6}",
            max_abs_err(&got, &fd)
        );
    }
}

// ---- Test: conv1d_grouped_grad (M6) -----------------------------------------
// B=1, C_in=4, T_in=6, C_out=4, K=2, stride=1, padding=0, groups=2.
// (1 < groups < c_in case — neither regular nor depthwise)

#[test]
fn conv1d_grouped_grad() {
    let b = 1;
    let c_in = 4;
    let t_in = 6;
    let c_out = 4;
    let k = 2;
    let stride = 1;
    let padding = 0;
    let groups = 2;
    let c_in_per_group = c_in / groups; // = 2
    let t_out = (t_in + 2 * padding - k) / stride + 1; // = 5

    let x_data: Vec<f32> = (0..b * c_in * t_in).map(|i| (i + 1) as f32 * 0.1).collect();
    let w_data: Vec<f32> = (0..c_out * c_in_per_group * k)
        .map(|i| (i + 1) as f32 * 0.05)
        .collect();

    // grad_x
    {
        let bk = CudaBackend::new(0).expect("CUDA init");
        let px = bk
            .param_from_slice_f32(&x_data, &[b, c_in, t_in])
            .expect("px");
        let w = bk
            .from_slice_f32(&w_data, &[c_out, c_in_per_group, k])
            .expect("w");
        let out = bk
            .conv1d(px.as_tensor(), &w, None, stride, padding, groups)
            .expect("conv1d");
        let loss = bk.sum_all(&out).expect("sum");
        let store = bk.backward(&loss).expect("bwd");
        let gx = bk.gradient(&store, &px).expect("gx").expect("Some");
        let got = bk.to_vec_f32(&gx).expect("tv");

        let fd = fd_grad(&x_data, 1e-2, |xd| {
            conv1d_host(
                xd,
                &w_data,
                b,
                c_in,
                t_in,
                c_out,
                c_in_per_group,
                k,
                stride,
                padding,
                groups,
                t_out,
            )
            .iter()
            .sum()
        });
        assert!(
            max_abs_err(&got, &fd) < 5e-3,
            "conv1d_grouped grad_x: max_err={:.6}",
            max_abs_err(&got, &fd)
        );
    }

    // grad_w
    {
        let bk = CudaBackend::new(0).expect("CUDA init");
        let x = bk.from_slice_f32(&x_data, &[b, c_in, t_in]).expect("x");
        let pw = bk
            .param_from_slice_f32(&w_data, &[c_out, c_in_per_group, k])
            .expect("pw");
        let out = bk
            .conv1d(&x, pw.as_tensor(), None, stride, padding, groups)
            .expect("conv1d");
        let loss = bk.sum_all(&out).expect("sum");
        let store = bk.backward(&loss).expect("bwd");
        let gw = bk.gradient(&store, &pw).expect("gw").expect("Some");
        let got = bk.to_vec_f32(&gw).expect("tv");

        let fd = fd_grad(&w_data, 1e-2, |wd| {
            conv1d_host(
                &x_data,
                wd,
                b,
                c_in,
                t_in,
                c_out,
                c_in_per_group,
                k,
                stride,
                padding,
                groups,
                t_out,
            )
            .iter()
            .sum()
        });
        assert!(
            max_abs_err(&got, &fd) < 5e-3,
            "conv1d_grouped grad_w: max_err={:.6}",
            max_abs_err(&got, &fd)
        );
    }
}

// ---- Test: matmul_grad_3d_x_2d (B4.4a) ----------------------------------------
// A=[B,T,K], B=[K,M] — rank(b) < rank(a) so forward pads b to [1,K,M].
// loss = sum(A @ B).
// Analytical:
//   grad_A[b,t,k] = sum_m grad_y[b,t,m] * B[k,m] = sum_m B[k,m]  (g=ones)
//   grad_B[k,m]   = sum_{b,t} grad_y[b,t,m] * A[b,t,k] = sum_{b,t} A[b,t,k]

#[test]
fn matmul_grad_3d_x_2d() {
    let b_sz = 2usize;
    let t_sz = 3usize;
    let k_sz = 4usize;
    let m_sz = 5usize;

    // Deterministic input values (small integers scaled down).
    let a_data: Vec<f32> = (0..b_sz * t_sz * k_sz)
        .map(|i| (i + 1) as f32 * 0.1)
        .collect();
    let b_data: Vec<f32> = (0..k_sz * m_sz).map(|i| (i + 1) as f32 * 0.05).collect();

    // ---- grad_A ----------------------------------------------------------------
    {
        let bk = CudaBackend::new(0).expect("CUDA init");
        let pa = bk
            .param_from_slice_f32(&a_data, &[b_sz, t_sz, k_sz])
            .expect("pa");
        let b_t = bk.from_slice_f32(&b_data, &[k_sz, m_sz]).expect("b");
        // Forward: [B,T,K] @ [K,M] → [B,T,M].
        let out = bk.matmul(pa.as_tensor(), &b_t).expect("matmul 3dx2d");
        assert_eq!(
            out.shape(),
            &[b_sz, t_sz, m_sz],
            "forward shape wrong: {:?}",
            out.shape()
        );
        let loss = bk.sum_all(&out).expect("sum_all");
        let store = bk.backward(&loss).expect("backward");
        let ga = bk.gradient(&store, &pa).expect("ga").expect("Some(ga)");
        assert_eq!(
            ga.shape(),
            &[b_sz, t_sz, k_sz],
            "grad_A shape wrong: {:?}",
            ga.shape()
        );
        let got_ga = bk.to_vec_f32(&ga).expect("to_vec ga");

        // Analytical: grad_A[b,t,k] = sum_m B[k,m]  (g = ones, so just row-sum of B).
        let b_row_sums: Vec<f32> = (0..k_sz)
            .map(|k| (0..m_sz).map(|m| b_data[k * m_sz + m]).sum::<f32>())
            .collect();
        let expected_ga: Vec<f32> = (0..b_sz * t_sz * k_sz)
            .map(|idx| b_row_sums[idx % k_sz])
            .collect();

        let err = max_abs_err(&got_ga, &expected_ga);
        assert!(
            err < 1e-4,
            "matmul_3d_x_2d grad_A: max_err={err:.2e} got={got_ga:?} expected={expected_ga:?}"
        );
    }

    // ---- grad_B ----------------------------------------------------------------
    {
        let bk = CudaBackend::new(0).expect("CUDA init");
        let a_t = bk.from_slice_f32(&a_data, &[b_sz, t_sz, k_sz]).expect("a");
        let pb = bk.param_from_slice_f32(&b_data, &[k_sz, m_sz]).expect("pb");
        let out = bk.matmul(&a_t, pb.as_tensor()).expect("matmul 3dx2d pb");
        let loss = bk.sum_all(&out).expect("sum_all");
        let store = bk.backward(&loss).expect("backward");
        let gb = bk.gradient(&store, &pb).expect("gb").expect("Some(gb)");
        assert_eq!(
            gb.shape(),
            &[k_sz, m_sz],
            "grad_B shape wrong: {:?}",
            gb.shape()
        );
        let got_gb = bk.to_vec_f32(&gb).expect("to_vec gb");

        // Analytical: grad_B[k,m] = sum_{b,t} A[b,t,k]  (g = ones).
        let mut expected_gb = vec![0.0f32; k_sz * m_sz];
        for bk_idx in 0..b_sz {
            for ti in 0..t_sz {
                for ki in 0..k_sz {
                    let a_val = a_data[bk_idx * t_sz * k_sz + ti * k_sz + ki];
                    for mi in 0..m_sz {
                        expected_gb[ki * m_sz + mi] += a_val;
                    }
                }
            }
        }

        let err = max_abs_err(&got_gb, &expected_gb);
        assert!(
            err < 1e-3,
            "matmul_3d_x_2d grad_B: max_err={err:.2e} got={got_gb:?} expected={expected_gb:?}"
        );
    }
}

// ---- Test: add_bias_rank_reduce_grad (B4.4a) ------------------------------------
// x=[B,T,D], bias=[D]: bias is rank-1, x is rank-3.
// B4.4a fix: backward for bias must sum over [B,T] dims.
// Sub and Mul are tested in separate sub-tests for completeness.

#[test]
fn add_bias_rank_reduce_grad() {
    let b_sz = 2usize;
    let t_sz = 3usize;
    let d_sz = 4usize;

    let x_data: Vec<f32> = (0..b_sz * t_sz * d_sz)
        .map(|i| (i + 1) as f32 * 0.1)
        .collect();
    let bias_data: Vec<f32> = (0..d_sz).map(|i| (i + 1) as f32 * 0.2).collect();

    // Analytical: loss = sum(x + bias).
    // grad_x    = ones([B,T,D])
    // grad_bias = sum_{B,T} ones([B,T,D]) → each element = B*T.
    let bk = CudaBackend::new(0).expect("CUDA init");
    let x_t = bk.from_slice_f32(&x_data, &[b_sz, t_sz, d_sz]).expect("x");
    let pb = bk
        .param_from_slice_f32(&bias_data, &[d_sz])
        .expect("pb bias");
    let out = bk.add(&x_t, pb.as_tensor()).expect("add bias");
    assert_eq!(out.shape(), &[b_sz, t_sz, d_sz]);
    let loss = bk.sum_all(&out).expect("sum_all");
    let store = bk.backward(&loss).expect("backward");
    let g_bias = bk
        .gradient(&store, &pb)
        .expect("g_bias")
        .expect("Some(g_bias)");
    assert_eq!(
        g_bias.shape(),
        &[d_sz],
        "grad_bias shape wrong: {:?}",
        g_bias.shape()
    );
    let got = bk.to_vec_f32(&g_bias).expect("to_vec");

    // Each bias element receives gradient = B * T = 2 * 3 = 6.
    let expected = vec![(b_sz * t_sz) as f32; d_sz];
    let err = max_abs_err(&got, &expected);
    assert!(
        err < 1e-5,
        "add_bias_rank_reduce_grad: max_err={err:.2e} got={got:?} expected={expected:?}"
    );
}

#[test]
fn sub_bias_rank_reduce_grad() {
    // y = x - bias; loss = sum(y).
    // grad_bias = -B*T for each element (negative because sub).
    let b_sz = 2usize;
    let t_sz = 3usize;
    let d_sz = 4usize;
    let x_data: Vec<f32> = (0..b_sz * t_sz * d_sz)
        .map(|i| (i + 1) as f32 * 0.1)
        .collect();
    let bias_data: Vec<f32> = (0..d_sz).map(|i| (i + 1) as f32 * 0.2).collect();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let x_t = bk.from_slice_f32(&x_data, &[b_sz, t_sz, d_sz]).expect("x");
    let pb = bk
        .param_from_slice_f32(&bias_data, &[d_sz])
        .expect("pb bias");
    let out = bk.sub(&x_t, pb.as_tensor()).expect("sub bias");
    let loss = bk.sum_all(&out).expect("sum_all");
    let store = bk.backward(&loss).expect("backward");
    let g_bias = bk
        .gradient(&store, &pb)
        .expect("g_bias")
        .expect("Some(g_bias)");
    assert_eq!(g_bias.shape(), &[d_sz]);
    let got = bk.to_vec_f32(&g_bias).expect("to_vec");

    // grad_bias = -(B*T) per element.
    let expected = vec![-((b_sz * t_sz) as f32); d_sz];
    let err = max_abs_err(&got, &expected);
    assert!(
        err < 1e-5,
        "sub_bias_rank_reduce_grad: max_err={err:.2e} got={got:?} expected={expected:?}"
    );
}

#[test]
fn mul_bias_rank_reduce_grad() {
    // y = x * bias; loss = sum(y).
    // grad_bias[d] = sum_{B,T} x[b,t,d].
    let b_sz = 2usize;
    let t_sz = 3usize;
    let d_sz = 4usize;
    let x_data: Vec<f32> = (0..b_sz * t_sz * d_sz)
        .map(|i| (i + 1) as f32 * 0.1)
        .collect();
    let bias_data: Vec<f32> = (0..d_sz).map(|i| (i + 1) as f32 * 0.2).collect();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let x_t = bk.from_slice_f32(&x_data, &[b_sz, t_sz, d_sz]).expect("x");
    let pb = bk
        .param_from_slice_f32(&bias_data, &[d_sz])
        .expect("pb bias");
    let out = bk.mul(&x_t, pb.as_tensor()).expect("mul bias");
    let loss = bk.sum_all(&out).expect("sum_all");
    let store = bk.backward(&loss).expect("backward");
    let g_bias = bk
        .gradient(&store, &pb)
        .expect("g_bias")
        .expect("Some(g_bias)");
    assert_eq!(g_bias.shape(), &[d_sz]);
    let got = bk.to_vec_f32(&g_bias).expect("to_vec");

    // grad_bias[d] = sum_{b,t} x[b,t,d]  (g = ones).
    let mut expected = vec![0.0f32; d_sz];
    for bk_idx in 0..b_sz {
        for ti in 0..t_sz {
            for di in 0..d_sz {
                expected[di] += x_data[bk_idx * t_sz * d_sz + ti * d_sz + di];
            }
        }
    }

    let err = max_abs_err(&got, &expected);
    assert!(
        err < 1e-4,
        "mul_bias_rank_reduce_grad: max_err={err:.2e} got={got:?} expected={expected:?}"
    );
}

// ---- Test: matmul_grad_4d_broadcast_batch (L3) -------------------------------
// A=[2,1,4,3], B=[2,5,3,6] → out=[2,5,4,6].
// loss = sum_all(A @ B).
// grad_A.shape == [2,1,4,3] (broadcast axis sum-reduced).
// Verifies H1 fix: broadcast matmul path now pushes to tape.

#[test]
fn matmul_grad_4d_broadcast_batch() {
    // A: [2, 1, 4, 3], B: [2, 5, 3, 6] → out: [2, 5, 4, 6]
    let a_shape = [2usize, 1, 4, 3];
    let b_shape = [2usize, 5, 3, 6];
    let m = 4usize;
    let k = 3usize;
    let n = 6usize;

    // Small deterministic values to reduce FD noise.
    let a_data: Vec<f32> = (0..2 * 1 * m * k).map(|i| (i + 1) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..2 * 5 * k * n).map(|i| (i + 1) as f32 * 0.05).collect();

    // ---- grad_A ----------------------------------------------------------------
    {
        let bk = CudaBackend::new(0).expect("CUDA init");
        let pa = bk.param_from_slice_f32(&a_data, &a_shape).expect("pa");
        let b_t = bk.from_slice_f32(&b_data, &b_shape).expect("b");
        let out = bk.matmul(pa.as_tensor(), &b_t).expect("matmul");
        let loss = bk.sum_all(&out).expect("sum");
        let store = bk.backward(&loss).expect("bwd");
        let ga = bk.gradient(&store, &pa).expect("ga").expect("Some");
        let got_ga = bk.to_vec_f32(&ga).expect("tv");

        // Shape check: grad_A must match A's original shape.
        assert_eq!(
            ga.shape(),
            &a_shape,
            "grad_A shape mismatch: got {:?} want {:?}",
            ga.shape(),
            a_shape
        );

        // Analytical: loss = sum_{all} (A @ B).
        // grad_A[bi, 0, i, j] = sum_{bj=0..5} sum_{n=0..6} g[bi, bj, i, n] * B[bi, bj, j, n]
        // Since g = ones, grad_A[bi, 0, i, j] = sum_{bj} sum_n B[bi, bj, j, n].
        let mut expected_ga = vec![0.0f32; 2 * 1 * m * k];
        for bi in 0..2usize {
            for i in 0..m {
                for j in 0..k {
                    let mut acc = 0.0f32;
                    for bj in 0..5usize {
                        for nn in 0..n {
                            let b_idx = bi * 5 * k * n + bj * k * n + j * n + nn;
                            acc += b_data[b_idx];
                        }
                    }
                    expected_ga[bi * 1 * m * k + 0 * m * k + i * k + j] = acc;
                }
            }
        }
        assert!(
            max_abs_err(&got_ga, &expected_ga) < 1e-3,
            "matmul_4d_broadcast grad_A: max_err={:.6} got={got_ga:?} expected={expected_ga:?}",
            max_abs_err(&got_ga, &expected_ga)
        );
    }

    // ---- grad_B ----------------------------------------------------------------
    {
        let bk = CudaBackend::new(0).expect("CUDA init");
        let a_t = bk.from_slice_f32(&a_data, &a_shape).expect("a");
        let pb = bk.param_from_slice_f32(&b_data, &b_shape).expect("pb");
        let out = bk.matmul(&a_t, pb.as_tensor()).expect("matmul");
        let loss = bk.sum_all(&out).expect("sum");
        let store = bk.backward(&loss).expect("bwd");
        let gb = bk.gradient(&store, &pb).expect("gb").expect("Some");
        let got_gb = bk.to_vec_f32(&gb).expect("tv");

        // Shape check: grad_B must match B's original shape.
        assert_eq!(
            gb.shape(),
            &b_shape,
            "grad_B shape mismatch: got {:?} want {:?}",
            gb.shape(),
            b_shape
        );

        // Analytical: grad_B[bi, bj, j, n] = sum_i A[bi, 0, i, j]  (broadcast over bj).
        let mut expected_gb = vec![0.0f32; 2 * 5 * k * n];
        for bi in 0..2usize {
            for bj in 0..5usize {
                for j in 0..k {
                    let val: f32 = (0..m).map(|i| a_data[bi * m * k + i * k + j]).sum();
                    for nn in 0..n {
                        expected_gb[bi * 5 * k * n + bj * k * n + j * n + nn] = val;
                    }
                }
            }
        }
        assert!(
            max_abs_err(&got_gb, &expected_gb) < 1e-3,
            "matmul_4d_broadcast grad_B: max_err={:.6}",
            max_abs_err(&got_gb, &expected_gb)
        );
    }
}
