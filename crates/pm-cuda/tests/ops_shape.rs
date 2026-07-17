//! Numerical parity tests for B4.2c shape copy ops.
//!
//! Tests compare `CudaBackend` GPU output to `CandleBackend::new_cpu()`.
//! Tolerance: fp32 1e-4 for general ops, 1e-6 for exact copies.
//!
//! Run with:
//!   cargo test -p pm-cuda --features cuda --test ops_shape -- --test-threads=1

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

// ---- 1. transpose_2d_matches_candle ----------------------------------------

#[test]
fn transpose_2d_matches_candle() {
    let shape = [3usize, 4usize];
    let data = lcg_vec(12, 1001);

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &shape).unwrap();
    let y_ref = cpu.transpose(&t_ref, 0, 1).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.transpose(&t, 0, 1).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &[4, 3], "transpose_2d: wrong output shape");
    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-6,
        "transpose_2d: max_abs_err={err:.2e} exceeds 1e-6"
    );
}

// ---- 2. transpose_3d_matches_candle ----------------------------------------

#[test]
fn transpose_3d_matches_candle() {
    let shape = [2usize, 3usize, 4usize];
    let data = lcg_vec(24, 2002);

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &shape).unwrap();
    // Transpose dims 1 and 2: (2,3,4) → (2,4,3)
    let y_ref = cpu.transpose(&t_ref, 1, 2).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.transpose(&t, 1, 2).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &[2, 4, 3], "transpose_3d: wrong output shape");
    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-6,
        "transpose_3d: max_abs_err={err:.2e} exceeds 1e-6"
    );
}

// ---- 3. narrow_dim0_matches_candle -----------------------------------------

#[test]
fn narrow_dim0_matches_candle() {
    let shape = [6usize, 4usize];
    let data = lcg_vec(24, 3003);

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &shape).unwrap();
    // narrow dim=0, start=1, len=3 → shape (3,4)
    let y_ref = cpu.narrow(&t_ref, 0, 1, 3).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.narrow(&t, 0, 1, 3).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &[3, 4], "narrow_dim0: wrong output shape");
    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-6,
        "narrow_dim0: max_abs_err={err:.2e} exceeds 1e-6"
    );
}

// ---- 4. narrow_dim1_matches_candle -----------------------------------------

#[test]
fn narrow_dim1_matches_candle() {
    let shape = [4usize, 8usize];
    let data = lcg_vec(32, 4004);

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &shape).unwrap();
    // narrow dim=1, start=2, len=5 → shape (4,5)
    let y_ref = cpu.narrow(&t_ref, 1, 2, 5).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &shape).unwrap();
    let y = bk.narrow(&t, 1, 2, 5).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &[4, 5], "narrow_dim1: wrong output shape");
    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-6,
        "narrow_dim1: max_abs_err={err:.2e} exceeds 1e-6"
    );
}

// ---- 5. broadcast_as_1_to_n ------------------------------------------------

#[test]
fn broadcast_as_1_to_n() {
    // [1, 4] → [3, 4]
    let src_shape = [1usize, 4usize];
    let data = lcg_vec(4, 5005);

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &src_shape).unwrap();
    let y_ref = cpu.broadcast_as(&t_ref, &[3, 4]).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &src_shape).unwrap();
    let y = bk.broadcast_as(&t, &[3, 4]).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(
        y.shape(),
        &[3, 4],
        "broadcast_as_1_to_n: wrong output shape"
    );
    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-6,
        "broadcast_as_1_to_n: max_abs_err={err:.2e} exceeds 1e-6"
    );
}

// ---- 6. broadcast_as_inner -------------------------------------------------

#[test]
fn broadcast_as_inner() {
    // [3, 1, 4] → [3, 2, 4]
    let src_shape = [3usize, 1usize, 4usize];
    let data = lcg_vec(12, 6006);

    let cpu = CandleBackend::new_cpu();
    let t_ref = cpu.from_slice_f32(&data, &src_shape).unwrap();
    let y_ref = cpu.broadcast_as(&t_ref, &[3, 2, 4]).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &src_shape).unwrap();
    let y = bk.broadcast_as(&t, &[3, 2, 4]).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(
        y.shape(),
        &[3, 2, 4],
        "broadcast_as_inner: wrong output shape"
    );
    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-6,
        "broadcast_as_inner: max_abs_err={err:.2e} exceeds 1e-6"
    );
}

// ---- 7. concat_dim0 --------------------------------------------------------

#[test]
fn concat_dim0() {
    // 2 × (2, 3) → (4, 3)
    let data_a = lcg_vec(6, 7007);
    let data_b = lcg_vec(6, 7008);

    let cpu = CandleBackend::new_cpu();
    let a_ref = cpu.from_slice_f32(&data_a, &[2, 3]).unwrap();
    let b_ref = cpu.from_slice_f32(&data_b, &[2, 3]).unwrap();
    let y_ref = cpu.concat(&[&a_ref, &b_ref], 0).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let a = bk.from_slice_f32(&data_a, &[2, 3]).unwrap();
    let b = bk.from_slice_f32(&data_b, &[2, 3]).unwrap();
    let y = bk.concat(&[&a, &b], 0).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &[4, 3], "concat_dim0: wrong output shape");
    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-6,
        "concat_dim0: max_abs_err={err:.2e} exceeds 1e-6"
    );
}

// ---- 8. concat_dim1 --------------------------------------------------------

#[test]
fn concat_dim1() {
    // 2 × (2, 3) → (2, 6)
    let data_a = lcg_vec(6, 8008);
    let data_b = lcg_vec(6, 8009);

    let cpu = CandleBackend::new_cpu();
    let a_ref = cpu.from_slice_f32(&data_a, &[2, 3]).unwrap();
    let b_ref = cpu.from_slice_f32(&data_b, &[2, 3]).unwrap();
    let y_ref = cpu.concat(&[&a_ref, &b_ref], 1).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let a = bk.from_slice_f32(&data_a, &[2, 3]).unwrap();
    let b = bk.from_slice_f32(&data_b, &[2, 3]).unwrap();
    let y = bk.concat(&[&a, &b], 1).unwrap();
    let got = bk.to_vec_f32(&y).unwrap();

    assert_eq!(y.shape(), &[2, 6], "concat_dim1: wrong output shape");
    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-6,
        "concat_dim1: max_abs_err={err:.2e} exceeds 1e-6"
    );
}
