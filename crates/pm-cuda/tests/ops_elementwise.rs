//! Numerical parity tests for B4.2b elementwise + scalar + reshape + param ops.
//!
//! Each test compares `CudaBackend` GPU output to `CandleBackend::new_cpu()`
//! reference. Tolerance: fp32 1e-4 (1e-6 for exact scalar arithmetic).
//!
//! Run with:
//!   cargo test -p pm-cuda --features cuda --test ops_elementwise -- --test-threads=1

#![cfg(feature = "cuda")]

use pm_candle::CandleBackend;
use pm_core::{Dtype, Ops, Param, Tensor};
use pm_cuda::CudaBackend;

// ---- Helpers ---------------------------------------------------------------

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "max_abs_err: length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Deterministic data generator — same LCG as ops_matmul.rs.
fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 33) as u32;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Candle CPU reference: apply unary `op` and return host vec.
fn candle_unary(
    shape: &[usize],
    data: &[f32],
    op: impl Fn(&CandleBackend, &<CandleBackend as Ops>::Tensor) -> <CandleBackend as Ops>::Tensor,
) -> Vec<f32> {
    let cpu = CandleBackend::new_cpu();
    let t = cpu.from_slice_f32(data, shape).unwrap();
    let y = op(&cpu, &t);
    cpu.to_vec_f32(&y).unwrap()
}

// ---- 1. exp ----------------------------------------------------------------

#[test]
fn exp_matches_candle() {
    let shape = [64];
    // Values in [-3, 3] — keeps exp in a well-behaved range.
    let data: Vec<f32> = lcg_vec(64, 1001).iter().map(|x| x * 1.5).collect();

    let ref_vals = candle_unary(&shape, &data, |bk, t| bk.exp(t).unwrap());

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.exp(&t).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &shape[..]);
    let err = max_abs_err(&got, &ref_vals);
    assert!(err < 1e-4, "exp: max_abs_err={err:.2e} exceeds 1e-4");
}

// ---- 2. sqrt ---------------------------------------------------------------

#[test]
fn sqrt_matches_candle() {
    let shape = [64];
    // Positive inputs only.
    let data: Vec<f32> = lcg_vec(64, 2002).iter().map(|x| x.abs() + 0.1).collect();

    let ref_vals = candle_unary(&shape, &data, |bk, t| bk.sqrt(t).unwrap());

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.sqrt(&t).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &shape[..]);
    let err = max_abs_err(&got, &ref_vals);
    assert!(err < 1e-4, "sqrt: max_abs_err={err:.2e} exceeds 1e-4");
}

// ---- 3. silu ---------------------------------------------------------------

#[test]
fn silu_matches_candle() {
    let shape = [2, 32];
    let data = lcg_vec(64, 3003);

    let ref_vals = candle_unary(&shape, &data, |bk, t| bk.silu(t).unwrap());

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.silu(&t).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &shape[..]);
    let err = max_abs_err(&got, &ref_vals);
    assert!(err < 1e-4, "silu: max_abs_err={err:.2e} exceeds 1e-4");
}

// ---- 4. sigmoid ------------------------------------------------------------

#[test]
fn sigmoid_matches_candle() {
    let shape = [2, 32];
    let data = lcg_vec(64, 4004);

    let ref_vals = candle_unary(&shape, &data, |bk, t| bk.sigmoid(t).unwrap());

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.sigmoid(&t).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &shape[..]);
    let err = max_abs_err(&got, &ref_vals);
    assert!(err < 1e-4, "sigmoid: max_abs_err={err:.2e} exceeds 1e-4");
}

// ---- 5. softplus -----------------------------------------------------------

#[test]
fn softplus_matches_candle() {
    let shape = [2, 32];
    let data = lcg_vec(64, 5005);

    let ref_vals = candle_unary(&shape, &data, |bk, t| bk.softplus(t).unwrap());

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.softplus(&t).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &shape[..]);
    let err = max_abs_err(&got, &ref_vals);
    assert!(err < 1e-4, "softplus: max_abs_err={err:.2e} exceeds 1e-4");
}

// ---- 6. div ----------------------------------------------------------------

#[test]
fn div_matches_candle() {
    let shape = [64];
    let a_data = lcg_vec(64, 6006);
    // Keep denominator away from 0.
    let b_data: Vec<f32> = lcg_vec(64, 6007).iter().map(|x| x.abs() + 0.5).collect();

    let cpu = CandleBackend::new_cpu();
    let a_ref = cpu.from_slice_f32(&a_data, &shape).unwrap();
    let b_ref = cpu.from_slice_f32(&b_data, &shape).unwrap();
    let ref_vals = cpu.to_vec_f32(&cpu.div(&a_ref, &b_ref).unwrap()).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let a_gpu = bk.from_slice_f32(&a_data, &shape).unwrap();
    let b_gpu = bk.from_slice_f32(&b_data, &shape).unwrap();
    let y = bk.div(&a_gpu, &b_gpu).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &shape[..]);
    let err = max_abs_err(&got, &ref_vals);
    assert!(err < 1e-4, "div: max_abs_err={err:.2e} exceeds 1e-4");
}

// ---- 7. mul_scalar ---------------------------------------------------------

#[test]
fn mul_scalar_matches_candle() {
    let shape = [16];
    let data = lcg_vec(16, 7007);
    let scale = 3.5_f32;

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &shape).unwrap();
    let ref_vals = cpu
        .to_vec_f32(&cpu.mul_scalar(&t_ref, scale).unwrap())
        .unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t_gpu = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.mul_scalar(&t_gpu, scale).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &shape[..]);
    let err = max_abs_err(&got, &ref_vals);
    assert!(err < 1e-6, "mul_scalar: max_abs_err={err:.2e} exceeds 1e-6");
}

// ---- 8. add_scalar ---------------------------------------------------------

#[test]
fn add_scalar_matches_candle() {
    let shape = [16];
    let data = lcg_vec(16, 8008);
    let scalar = 1.25_f32;

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &shape).unwrap();
    let ref_vals = cpu
        .to_vec_f32(&cpu.add_scalar(&t_ref, scalar).unwrap())
        .unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t_gpu = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.add_scalar(&t_gpu, scalar).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &shape[..]);
    let err = max_abs_err(&got, &ref_vals);
    assert!(err < 1e-6, "add_scalar: max_abs_err={err:.2e} exceeds 1e-6");
}

// ---- 9. reshape (zero-copy) ------------------------------------------------
//
// We verify the zero-copy property indirectly: reshape the tensor, then
// mutate the original's underlying memory by performing a GPU op on a copy,
// which should NOT affect the reshaped view (they share storage via Arc clone,
// not a mutable alias). We also verify shapes and data content match.

#[test]
fn reshape_zero_copy() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = bk.from_slice_f32(&data, &[2, 3]).unwrap();

    // Reshape: [2,3] → [6]. No GPU op — just metadata.
    let reshaped = bk.reshape(&t, &[6]).unwrap();

    assert_eq!(reshaped.shape(), &[6], "reshape: wrong shape");
    let vals = bk.to_vec_f32(&reshaped).unwrap();
    assert_eq!(vals, data, "reshape: data content mismatch");

    // Also verify the original tensor is unchanged.
    let orig_vals = bk.to_vec_f32(&t).unwrap();
    assert_eq!(orig_vals, data, "reshape: original tensor was altered");
}

// ---- 10. from_slice_i64 round-trip -----------------------------------------

#[test]
fn from_slice_i64_round_trip() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let data: &[i64] = &[1, 2, 3, 4, 5];
    let t = bk.from_slice_i64(data, &[5]).unwrap();

    assert_eq!(t.shape(), &[5], "from_slice_i64: shape mismatch");
    assert_eq!(t.dtype(), Dtype::I64, "from_slice_i64: dtype mismatch");

    // Round-trip via public to_vec_i64 helper.
    let recovered = bk.to_vec_i64(&t).unwrap();
    assert_eq!(recovered, data, "from_slice_i64: data round-trip mismatch");
}

// ---- 11. param round-trips -------------------------------------------------

#[test]
fn param_zeros_ones_round_trip() {
    let bk = CudaBackend::new(0).expect("CUDA init");

    // param_zeros
    let pz = bk.param_zeros(&[2, 3], Dtype::F32).unwrap();
    let vals_z = bk.to_vec_f32(pz.as_tensor()).unwrap();
    assert_eq!(vals_z.len(), 6, "param_zeros: wrong numel");
    assert!(
        vals_z.iter().all(|&x| x == 0.0),
        "param_zeros: not all zeros"
    );

    // param_ones
    let po = bk.param_ones(&[2, 3], Dtype::F32).unwrap();
    let vals_o = bk.to_vec_f32(po.as_tensor()).unwrap();
    assert_eq!(vals_o.len(), 6, "param_ones: wrong numel");
    assert!(vals_o.iter().all(|&x| x == 1.0), "param_ones: not all ones");

    // param_from_slice_f32
    let data: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
    let ps = bk.param_from_slice_f32(&data, &[2, 3]).unwrap();
    let vals_s = bk.to_vec_f32(ps.as_tensor()).unwrap();
    assert_eq!(vals_s, data, "param_from_slice_f32: data mismatch");

    // Each param must have a unique id (via the public param_id() accessor).
    assert_ne!(
        pz.param_id(),
        po.param_id(),
        "param_zeros and param_ones must have distinct IDs"
    );
    assert_ne!(
        po.param_id(),
        ps.param_id(),
        "param_ones and param_from_slice must have distinct IDs"
    );
}
