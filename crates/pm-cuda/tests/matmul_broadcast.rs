//! B'.2b — batched-broadcast matmul correctness (stride-0 direct path).
//!
//! `Ops::matmul` / `matmul_inner_fn` route 1-vs-N batch broadcasts (the
//! side needing broadcast has all leading dims == 1) straight to
//! `kernels::matmul_f32`'s stride-0 batched gemm instead of the host
//! materialisation path (which expanded the shared weight `n_chunks`-fold
//! on the CPU — ~190 ms/call for PHOTON level-1 shapes at T=512).
//! Per-dim mixed broadcasts (e.g. leading `[2,1] @ [1,4]`) must still take
//! the host path and stay numerically correct.
//!
//! Every case is checked against a naive CPU broadcast-matmul reference;
//! backward cases against analytical gradients (loss = sum_all(out)).

#![cfg(feature = "cuda")]

use pm_core::{Ops, Param, Tensor};
use pm_cuda::CudaBackend;

fn filled(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed).wrapping_mul(2654435761)) & 0xffff) as f32 / 65536.0 - 0.5)
        .collect()
}

/// Naive CPU reference: broadcast-aware batched matmul.
/// `b_shape` may have lower rank than `a_shape` (padded with leading 1s,
/// mirroring `Ops::matmul`). Returns (data, out_shape).
fn cpu_matmul_bcast(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape_raw: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    let rank = a_shape.len();
    let mut b_shape = vec![1usize; rank - b_shape_raw.len()];
    b_shape.extend_from_slice(b_shape_raw);

    let (m, k) = (a_shape[rank - 2], a_shape[rank - 1]);
    let n = b_shape[rank - 1];
    assert_eq!(b_shape[rank - 2], k);

    let lead_out: Vec<usize> = a_shape[..rank - 2]
        .iter()
        .zip(b_shape[..rank - 2].iter())
        .map(|(&da, &db)| da.max(db))
        .collect();
    let batch: usize = lead_out.iter().product();

    // flat batch index -> source flat batch index under broadcast
    let src_batch = |flat: usize, src_lead: &[usize]| -> usize {
        let mut rem = flat;
        let mut src = 0usize;
        for (d, (&_od, &sd)) in lead_out.iter().zip(src_lead.iter()).enumerate() {
            let stride: usize = lead_out[d + 1..].iter().product();
            let idx = rem / stride;
            rem %= stride;
            let sidx = if sd == 1 { 0 } else { idx };
            src = src * sd + sidx;
        }
        let _ = rem;
        src
    };

    let mut out = vec![0.0f32; batch * m * n];
    for p in 0..batch {
        let pa = src_batch(p, &a_shape[..rank - 2]);
        let pb = src_batch(p, &b_shape[..rank - 2]);
        for i in 0..m {
            for jn in 0..n {
                let mut acc = 0.0f32;
                for j in 0..k {
                    acc += a[pa * m * k + i * k + j] * b[pb * k * n + j * n + jn];
                }
                out[p * m * n + i * n + jn] = acc;
            }
        }
    }
    let mut out_shape = lead_out;
    out_shape.push(m);
    out_shape.push(n);
    (out, out_shape)
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn assert_fwd_matches(a_shape: &[usize], b_shape: &[usize], tol: f32) {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let a_data = filled(a_shape.iter().product(), 1);
    let b_data = filled(b_shape.iter().product(), 7);
    let a_t = Ops::from_slice_f32(&bk, &a_data, a_shape).expect("a");
    let b_t = Ops::from_slice_f32(&bk, &b_data, b_shape).expect("b");
    let out = bk.matmul(&a_t, &b_t).expect("matmul");
    let got = bk.to_vec_f32(&out).expect("to_vec");
    let (want, want_shape) = cpu_matmul_bcast(&a_data, a_shape, &b_data, b_shape);
    assert_eq!(out.shape(), &want_shape[..], "out shape");
    let err = max_abs_err(&got, &want);
    assert!(
        err < tol,
        "fwd mismatch a={a_shape:?} b={b_shape:?}: max_abs_err={err}"
    );
}

#[test]
fn fwd_batch_n_vs_1() {
    // level-1 pattern: activations batched, shared weight batch=1
    assert_fwd_matches(&[4, 3, 6], &[1, 6, 5], 1e-5);
}

#[test]
fn fwd_batch_1_vs_n() {
    assert_fwd_matches(&[1, 3, 6], &[4, 6, 5], 1e-5);
}

#[test]
fn fwd_rank2_weight_padding() {
    // rank-2 weight gets padded to [1, k, n] — production linear-layer form
    assert_fwd_matches(&[4, 3, 6], &[6, 5], 1e-5);
}

#[test]
fn fwd_mixed_broadcast_stays_on_host_path_and_correct() {
    // leading [2,1] @ [1,4] — not expressible as one flattened stride-0
    // batch; must take the host-expand path and still be correct.
    assert_fwd_matches(&[2, 1, 3, 6], &[1, 4, 6, 5], 1e-5);
}

#[test]
fn bwd_grads_flow_n_vs_1() {
    // loss = sum(matmul(A[4,3,6], B[1,6,5])):
    //   dA[p,i,j] = sum_n B[0,j,n]        (independent of p, i)
    //   dB[0,j,n] = sum_p sum_i A[p,i,j]  (batch-reduced via sum_to_shape)
    let bk = CudaBackend::new(0).expect("CUDA init");
    let (pb_dims, m, k, n) = (4usize, 3usize, 6usize, 5usize);
    let a_data = filled(pb_dims * m * k, 3);
    let b_data = filled(k * n, 11);

    let pa = bk
        .param_from_slice_f32(&a_data, &[pb_dims, m, k])
        .expect("pa");
    let pb = bk.param_from_slice_f32(&b_data, &[1, k, n]).expect("pb");
    let out = bk.matmul(pa.as_tensor(), pb.as_tensor()).expect("matmul");
    let loss = bk.sum_all(&out).expect("sum_all");
    let store = bk.backward(&loss).expect("backward");

    let ga = bk.gradient(&store, &pa).expect("ga").expect("Some(ga)");
    let got_ga = bk.to_vec_f32(&ga).expect("ga vec");
    let mut want_ga = vec![0.0f32; pb_dims * m * k];
    for p in 0..pb_dims {
        for i in 0..m {
            for j in 0..k {
                want_ga[p * m * k + i * k + j] = (0..n).map(|jn| b_data[j * n + jn]).sum();
            }
        }
    }
    let err_a = max_abs_err(&got_ga, &want_ga);
    assert!(err_a < 1e-4, "dA mismatch: {err_a}");

    let gb = bk.gradient(&store, &pb).expect("gb").expect("Some(gb)");
    assert_eq!(gb.shape(), &[1, k, n], "dB must reduce back to B's shape");
    let got_gb = bk.to_vec_f32(&gb).expect("gb vec");
    let mut want_gb = vec![0.0f32; k * n];
    for j in 0..k {
        let colsum: f32 = (0..pb_dims)
            .flat_map(|p| (0..m).map(move |i| (p, i)))
            .map(|(p, i)| a_data[p * m * k + i * k + j])
            .sum();
        for jn in 0..n {
            want_gb[j * n + jn] = colsum;
        }
    }
    let err_b = max_abs_err(&got_gb, &want_gb);
    assert!(err_b < 1e-4, "dB mismatch: {err_b}");
}

#[test]
fn production_shape_smoke_is_fast_and_finite() {
    // Real level-1 shape. With the old host-expand path this call cost
    // ~190 ms (release) / seconds (debug) — the 1 s bound fails loudly on a
    // regression to host materialisation while staying far above gemm cost
    // (~0.2 ms) plus cold-start noise.
    let bk = CudaBackend::new(0).expect("CUDA init");
    // warmup: tiny matmul to absorb module/handle cold start
    let wa = Ops::from_slice_f32(&bk, &filled(4, 1), &[1, 2, 2]).expect("wa");
    let wb = Ops::from_slice_f32(&bk, &filled(4, 2), &[2, 2]).expect("wb");
    let _ = bk.matmul(&wa, &wb).expect("warmup");

    let (nb, c, d, dp) = (128usize, 8usize, 768usize, 1804usize);
    let a_t = Ops::from_slice_f32(&bk, &filled(nb * c * d, 5), &[nb, c, d]).expect("a");
    let b_t = Ops::from_slice_f32(&bk, &filled(d * dp, 9), &[1, d, dp]).expect("b");
    let t0 = std::time::Instant::now();
    let out = bk.matmul(&a_t, &b_t).expect("matmul");
    let got = bk.to_vec_f32(&out).expect("to_vec");
    let dt = t0.elapsed();
    assert!(got.iter().all(|v| v.is_finite()), "non-finite output");
    assert_eq!(out.shape(), &[nb, c, dp]);
    assert!(
        dt.as_secs_f64() < 1.0,
        "production-shape matmul took {dt:?} — host-expand regression?"
    );
}
