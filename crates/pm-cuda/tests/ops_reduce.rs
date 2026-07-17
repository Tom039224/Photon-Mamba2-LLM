//! Numerical parity tests for B4.2c reduction ops.
//!
//! Each test compares `CudaBackend` GPU output to `CandleBackend::new_cpu()`.
//!
//! Run with:
//!   cargo test -p pm-cuda --features cuda --test ops_reduce -- --test-threads=1

#![cfg(feature = "cuda")]

use pm_candle::CandleBackend;
use pm_core::{Ops, Tensor};
use pm_cuda::CudaBackend;

// ---- Helpers ----------------------------------------------------------------

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "max_abs_err: length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

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

// ---- 1. mean_all_matches_candle --------------------------------------------

#[test]
fn mean_all_matches_candle() {
    // Use integer-valued data so the reference is exact.
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let shape = [4usize, 4usize];

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &shape).unwrap();
    let y_ref = cpu.mean_all(&t_ref).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.mean_all(&t).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(got.len(), 1, "mean_all: result must be scalar");
    let err = max_abs_err(&got, &ref_vals);
    assert!(err < 1e-6, "mean_all: max_abs_err={err:.2e} exceeds 1e-6");
}

// ---- 2. sum_all_matches_candle ---------------------------------------------

#[test]
fn sum_all_matches_candle() {
    // Integer-valued data for exact comparison.
    let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let shape = [4usize, 5usize];

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &shape).unwrap();
    let y_ref = cpu.sum_all(&t_ref).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.sum_all(&t).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(got.len(), 1, "sum_all: result must be scalar");
    let err = max_abs_err(&got, &ref_vals);
    assert!(err < 1e-6, "sum_all: max_abs_err={err:.2e} exceeds 1e-6");
}

// ---- 3. cumsum_lastdim_matches_candle --------------------------------------

#[test]
fn cumsum_lastdim_matches_candle() {
    let shape = [3usize, 8usize];
    let data = lcg_vec(24, 3003);

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &shape).unwrap();
    let y_ref = cpu.cumsum(&t_ref, 1).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.cumsum(&t, 1).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), shape, "cumsum_lastdim: wrong output shape");
    let err = max_abs_err(&got, &ref_vals);
    // cumsum accumulates floating-point additions; 1e-5 is comfortably tight.
    assert!(
        err < 1e-5,
        "cumsum_lastdim: max_abs_err={err:.2e} exceeds 1e-5"
    );
}

// ---- 4. cumsum_non_lastdim_matches_candle ----------------------------------

#[test]
fn cumsum_non_lastdim_matches_candle() {
    // dim=0 triggers the transpose path.
    let shape = [6usize, 4usize];
    let data = lcg_vec(24, 4004);

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &shape).unwrap();
    let y_ref = cpu.cumsum(&t_ref, 0).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.cumsum(&t, 0).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), shape, "cumsum_non_lastdim: wrong output shape");
    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-5,
        "cumsum_non_lastdim: max_abs_err={err:.2e} exceeds 1e-5"
    );
}

// ---- 5. rmsnorm_matches_candle ---------------------------------------------

#[test]
fn rmsnorm_matches_candle() {
    let shape = [4usize, 128usize];
    let data = lcg_vec(4 * 128, 5005);
    let weight_data = vec![1.0f32; 128];
    let eps = 1e-5_f32;

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &shape).unwrap();
    let w_ref = cpu.from_slice_f32(&weight_data, &[128]).unwrap();
    let y_ref = cpu.rmsnorm(&t_ref, &w_ref, eps).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let w = bk.from_slice_f32(&weight_data, &[128]).unwrap();
    let y = bk.rmsnorm(&t, &w, eps).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &shape[..], "rmsnorm: wrong output shape");
    let err = max_abs_err(&got, &ref_vals);
    assert!(err < 1e-4, "rmsnorm: max_abs_err={err:.2e} exceeds 1e-4");
}

// ---- 6. log_softmax_lastdim_matches_candle ---------------------------------

#[test]
fn log_softmax_lastdim_matches_candle() {
    let shape = [8usize, 32usize];
    let data = lcg_vec(256, 6006);

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &shape).unwrap();
    let y_ref = cpu.log_softmax(&t_ref, 1).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.log_softmax(&t, 1).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(
        y.shape(),
        &shape[..],
        "log_softmax_lastdim: wrong output shape"
    );
    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-4,
        "log_softmax_lastdim: max_abs_err={err:.2e} exceeds 1e-4"
    );
}

// ---- 7. log_softmax_non_lastdim_matches_candle -----------------------------

#[test]
fn log_softmax_non_lastdim_matches_candle() {
    // dim=0 triggers the transpose path.
    let shape = [8usize, 16usize];
    let data = lcg_vec(128, 7007);

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &shape).unwrap();
    let y_ref = cpu.log_softmax(&t_ref, 0).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.log_softmax(&t, 0).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(
        y.shape(),
        &shape[..],
        "log_softmax_non_lastdim: wrong output shape"
    );
    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-4,
        "log_softmax_non_lastdim: max_abs_err={err:.2e} exceeds 1e-4"
    );
}
