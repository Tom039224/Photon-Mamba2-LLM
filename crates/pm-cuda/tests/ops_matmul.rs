//! Numerical parity tests: `CudaBackend::matmul` vs `CandleBackend::new_cpu()`.
//!
//! B4.2a — validates cuBLAS matmul implementation.
//!
//! Tests run sequentially (`--test-threads=1`) to avoid contention with
//! any concurrent B3 training on the GPU.

#![cfg(feature = "cuda")]

use pm_candle::CandleBackend;
use pm_core::{Ops, Tensor};
use pm_cuda::CudaBackend;

/// Maximum absolute element-wise error between two f32 slices.
fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "max_abs_err: length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Simple deterministic data generator using a linear-congruential RNG.
fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map to [-1, 1]
            let bits = (state >> 33) as u32;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// 1. 2D square matmul: (4,4) @ (4,4) → (4,4)
#[test]
fn matmul_2d_square() {
    let cpu = CandleBackend::new_cpu();
    let bk = CudaBackend::new(0).expect("CUDA init");

    let a_data = lcg_vec(4 * 4, 42);
    let b_data = lcg_vec(4 * 4, 137);

    let a_ref = cpu.from_slice_f32(&a_data, &[4, 4]).unwrap();
    let b_ref = cpu.from_slice_f32(&b_data, &[4, 4]).unwrap();
    let c_ref = cpu.matmul(&a_ref, &b_ref).unwrap();
    let ref_vals = cpu.to_vec_f32(&c_ref).unwrap();

    let a_gpu = bk.from_slice_f32(&a_data, &[4, 4]).unwrap();
    let b_gpu = bk.from_slice_f32(&b_data, &[4, 4]).unwrap();
    let c_gpu = bk.matmul(&a_gpu, &b_gpu).unwrap();
    let gpu_vals = bk.to_vec_f32(&c_gpu).unwrap();

    assert_eq!(
        gpu_vals.len(),
        ref_vals.len(),
        "matmul_2d_square: output length mismatch"
    );
    let err = max_abs_err(&gpu_vals, &ref_vals);
    assert!(
        err < 1e-4,
        "matmul_2d_square: max_abs_err={err} exceeds 1e-4"
    );
}

/// 2. 2D rectangular matmul: (2,3) @ (3,4) → (2,4)
#[test]
fn matmul_2d_rect() {
    let cpu = CandleBackend::new_cpu();
    let bk = CudaBackend::new(0).expect("CUDA init");

    let a_data = lcg_vec(2 * 3, 11);
    let b_data = lcg_vec(3 * 4, 17);

    let a_ref = cpu.from_slice_f32(&a_data, &[2, 3]).unwrap();
    let b_ref = cpu.from_slice_f32(&b_data, &[3, 4]).unwrap();
    let c_ref = cpu.matmul(&a_ref, &b_ref).unwrap();
    let ref_vals = cpu.to_vec_f32(&c_ref).unwrap();

    let a_gpu = bk.from_slice_f32(&a_data, &[2, 3]).unwrap();
    let b_gpu = bk.from_slice_f32(&b_data, &[3, 4]).unwrap();
    let c_gpu = bk.matmul(&a_gpu, &b_gpu).unwrap();
    let gpu_vals = bk.to_vec_f32(&c_gpu).unwrap();

    assert_eq!(c_gpu.shape(), &[2, 4]);
    let err = max_abs_err(&gpu_vals, &ref_vals);
    assert!(err < 1e-4, "matmul_2d_rect: max_abs_err={err} exceeds 1e-4");
}

/// 3. 3D batched matmul: (2,2,3) @ (2,3,4) → (2,2,4)
#[test]
fn matmul_batched_3d() {
    let cpu = CandleBackend::new_cpu();
    let bk = CudaBackend::new(0).expect("CUDA init");

    let a_data = lcg_vec(2 * 2 * 3, 99);
    let b_data = lcg_vec(2 * 3 * 4, 101);

    let a_ref = cpu.from_slice_f32(&a_data, &[2, 2, 3]).unwrap();
    let b_ref = cpu.from_slice_f32(&b_data, &[2, 3, 4]).unwrap();
    let c_ref = cpu.matmul(&a_ref, &b_ref).unwrap();
    let ref_vals = cpu.to_vec_f32(&c_ref).unwrap();

    let a_gpu = bk.from_slice_f32(&a_data, &[2, 2, 3]).unwrap();
    let b_gpu = bk.from_slice_f32(&b_data, &[2, 3, 4]).unwrap();
    let c_gpu = bk.matmul(&a_gpu, &b_gpu).unwrap();
    let gpu_vals = bk.to_vec_f32(&c_gpu).unwrap();

    assert_eq!(c_gpu.shape(), &[2, 2, 4]);
    let err = max_abs_err(&gpu_vals, &ref_vals);
    assert!(
        err < 1e-4,
        "matmul_batched_3d: max_abs_err={err} exceeds 1e-4"
    );
}

/// 4. 4D broadcast-batched matmul: (1,3,2,3) @ (4,3,3,5) → (4,3,2,5)
///
/// The first batch dim of `a` is 1, which should broadcast against `b`'s 4.
/// The second batch dim is 3 for both — no broadcast there.
#[test]
fn matmul_batched_broadcast_4d() {
    let cpu = CandleBackend::new_cpu();
    let bk = CudaBackend::new(0).expect("CUDA init");

    // a: (1, 3, 2, 3)
    let a_data = lcg_vec(3 * 2 * 3, 200);
    // b: (4, 3, 3, 5)
    let b_data = lcg_vec(4 * 3 * 3 * 5, 201);

    let a_ref = cpu.from_slice_f32(&a_data, &[1, 3, 2, 3]).unwrap();
    let b_ref = cpu.from_slice_f32(&b_data, &[4, 3, 3, 5]).unwrap();
    let c_ref = cpu.matmul(&a_ref, &b_ref).unwrap();
    let ref_vals = cpu.to_vec_f32(&c_ref).unwrap();

    let a_gpu = bk.from_slice_f32(&a_data, &[1, 3, 2, 3]).unwrap();
    let b_gpu = bk.from_slice_f32(&b_data, &[4, 3, 3, 5]).unwrap();
    let c_gpu = bk.matmul(&a_gpu, &b_gpu).unwrap();
    let gpu_vals = bk.to_vec_f32(&c_gpu).unwrap();

    assert_eq!(c_gpu.shape(), &[4, 3, 2, 5]);
    let err = max_abs_err(&gpu_vals, &ref_vals);
    assert!(
        err < 1e-4,
        "matmul_batched_broadcast_4d: max_abs_err={err} exceeds 1e-4"
    );
}
