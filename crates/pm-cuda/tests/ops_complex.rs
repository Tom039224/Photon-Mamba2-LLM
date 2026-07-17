//! B4.2d parity tests: embedding / gather / conv1d / ssd_scan.
//!
//! Each test compares `CudaBackend` GPU output against `CandleBackend::new_cpu()`
//! reference. Tolerance: fp32 1e-4 (matching the B4 parity budget).
//!
//! Run with:
//!   cargo test -p pm-cuda --features cuda --test ops_complex -- --test-threads=1

#![cfg(feature = "cuda")]

use pm_candle::CandleBackend;
use pm_core::{Ops, Tensor};
use pm_cuda::CudaBackend;

// ---- Helpers ---------------------------------------------------------------

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

// ---- 1. embedding ----------------------------------------------------------

#[test]
fn embedding_matches_candle() {
    // vocab=100, dim=32, indices shape (2, 3).
    let vocab = 100usize;
    let dim = 32usize;
    let idx_data: &[i64] = &[0, 5, 99, 10, 20, 30];
    let idx_shape = &[2usize, 3];
    let table_data: Vec<f32> = lcg_vec(vocab * dim, 1001);

    // Reference: Candle CPU.
    let cpu = CandleBackend::new_cpu();
    let table_ref = cpu.from_slice_f32(&table_data, &[vocab, dim]).unwrap();
    let idx_ref = cpu.from_slice_i64(idx_data, idx_shape).unwrap();
    let y_ref = cpu.embedding(&table_ref, &idx_ref).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();
    assert_eq!(y_ref.shape(), &[2, 3, dim]);

    // CUDA.
    let bk = CudaBackend::new(0).expect("CUDA init");
    let table_gpu = bk.from_slice_f32(&table_data, &[vocab, dim]).unwrap();
    let idx_gpu = bk.from_slice_i64(idx_data, idx_shape).unwrap();
    let y_gpu = bk.embedding(&table_gpu, &idx_gpu).unwrap();
    assert_eq!(y_gpu.shape(), &[2, 3, dim]);
    let got = bk.to_vec_f32(&y_gpu).unwrap();

    let err = max_abs_err(&got, &ref_vals);
    assert!(err < 1e-4, "embedding: max_abs_err={err:.2e} exceeds 1e-4");
}

// ---- 2. gather (last-dim) --------------------------------------------------

#[test]
fn gather_lastdim_matches_candle() {
    // src shape (2, 4), indices shape (2, 2), dim=1 (last dim for rank-2).
    let src_data: &[f32] = &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let src_shape = &[2usize, 4];
    // For dim=1 (last dim): pick column indices per row.
    let idx_data: &[i64] = &[3, 0, 2, 1];
    let idx_shape = &[2usize, 2];

    // Reference: Candle CPU.
    let cpu = CandleBackend::new_cpu();
    let src_ref = cpu.from_slice_f32(src_data, src_shape).unwrap();
    let idx_ref = cpu.from_slice_i64(idx_data, idx_shape).unwrap();
    let y_ref = cpu.gather(&src_ref, &idx_ref, 1).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();
    assert_eq!(y_ref.shape(), &[2, 2]);

    // CUDA.
    let bk = CudaBackend::new(0).expect("CUDA init");
    let src_gpu = bk.from_slice_f32(src_data, src_shape).unwrap();
    let idx_gpu = bk.from_slice_i64(idx_data, idx_shape).unwrap();
    let y_gpu = bk.gather(&src_gpu, &idx_gpu, 1).unwrap();
    assert_eq!(y_gpu.shape(), &[2, 2]);
    let got = bk.to_vec_f32(&y_gpu).unwrap();

    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-4,
        "gather_lastdim: max_abs_err={err:.2e} exceeds 1e-4"
    );
}

// ---- 3. conv1d depthwise ---------------------------------------------------

#[test]
fn conv1d_depthwise_matches_candle() {
    // B=1, C_in=C_out=2, T=4, K=1, groups=2 (depthwise identity).
    let x_data: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x_shape = &[1usize, 2, 4]; // (B=1, C=2, T=4)
                                   // weight (C_out=2, C_in/groups=1, K=1) all-ones → identity.
    let w_data: &[f32] = &[1.0, 1.0];
    let w_shape = &[2usize, 1, 1];

    // Reference: Candle CPU.
    let cpu = CandleBackend::new_cpu();
    let x_ref = cpu.from_slice_f32(x_data, x_shape).unwrap();
    let w_ref = cpu.from_slice_f32(w_data, w_shape).unwrap();
    let y_ref = cpu.conv1d(&x_ref, &w_ref, None, 1, 0, 2).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();
    assert_eq!(y_ref.shape(), &[1, 2, 4]);

    // CUDA.
    let bk = CudaBackend::new(0).expect("CUDA init");
    let x_gpu = bk.from_slice_f32(x_data, x_shape).unwrap();
    let w_gpu = bk.from_slice_f32(w_data, w_shape).unwrap();
    let y_gpu = bk.conv1d(&x_gpu, &w_gpu, None, 1, 0, 2).unwrap();
    assert_eq!(y_gpu.shape(), &[1, 2, 4]);
    let got = bk.to_vec_f32(&y_gpu).unwrap();

    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-4,
        "conv1d_depthwise: max_abs_err={err:.2e} exceeds 1e-4"
    );
}

// ---- 4. conv1d regular -----------------------------------------------------

#[test]
fn conv1d_regular_matches_candle() {
    // B=1, C_in=4, C_out=8, T=8, K=3, stride=1, padding=1, groups=1.
    let b = 1usize;
    let c_in = 4usize;
    let c_out = 8usize;
    let t = 8usize;
    let k = 3usize;
    let x_data = lcg_vec(b * c_in * t, 2001);
    let w_data = lcg_vec(c_out * c_in * k, 2002);
    // t_out = (8 + 2*1 - 3) / 1 + 1 = 8
    let t_out = (t + 2 * 1 - k) / 1 + 1;

    // Reference: Candle CPU.
    let cpu = CandleBackend::new_cpu();
    let x_ref = cpu.from_slice_f32(&x_data, &[b, c_in, t]).unwrap();
    let w_ref = cpu.from_slice_f32(&w_data, &[c_out, c_in, k]).unwrap();
    let y_ref = cpu.conv1d(&x_ref, &w_ref, None, 1, 1, 1).unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();
    assert_eq!(y_ref.shape(), &[b, c_out, t_out]);

    // CUDA.
    let bk = CudaBackend::new(0).expect("CUDA init");
    let x_gpu = bk.from_slice_f32(&x_data, &[b, c_in, t]).unwrap();
    let w_gpu = bk.from_slice_f32(&w_data, &[c_out, c_in, k]).unwrap();
    let y_gpu = bk.conv1d(&x_gpu, &w_gpu, None, 1, 1, 1).unwrap();
    assert_eq!(y_gpu.shape(), &[b, c_out, t_out]);
    let got = bk.to_vec_f32(&y_gpu).unwrap();

    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-4,
        "conv1d_regular: max_abs_err={err:.2e} exceeds 1e-4"
    );
}

// ---- 5. ssd_scan -----------------------------------------------------------

#[test]
fn ssd_scan_matches_candle() {
    // B=1, T=8, H=2, P=4, N=4, block_len=4 (T must be multiple of block_len).
    let (batch, t_len, n_heads, p_dim, n_dim, block_len) = (1usize, 8, 2, 4, 4, 4);

    let x_data = lcg_vec(batch * t_len * n_heads * p_dim, 3001);
    // A: small negative values so exp stays well-behaved.
    let a_data: Vec<f32> = lcg_vec(batch * t_len * n_heads, 3002)
        .iter()
        .map(|v| v * 0.1 - 0.1)
        .collect();
    let b_data = lcg_vec(batch * t_len * n_heads * n_dim, 3003);
    let c_data = lcg_vec(batch * t_len * n_heads * n_dim, 3004);

    // Reference: Candle CPU (pure-Ops ssd_scan).
    let cpu = CandleBackend::new_cpu();
    let x_ref = cpu
        .from_slice_f32(&x_data, &[batch, t_len, n_heads, p_dim])
        .unwrap();
    let a_ref = cpu
        .from_slice_f32(&a_data, &[batch, t_len, n_heads])
        .unwrap();
    let b_ref = cpu
        .from_slice_f32(&b_data, &[batch, t_len, n_heads, n_dim])
        .unwrap();
    let c_ref = cpu
        .from_slice_f32(&c_data, &[batch, t_len, n_heads, n_dim])
        .unwrap();
    let y_ref = cpu
        .ssd_scan(&x_ref, &a_ref, &b_ref, &c_ref, block_len)
        .unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();
    assert_eq!(y_ref.shape(), &[batch, t_len, n_heads, p_dim]);

    // CUDA.
    let bk = CudaBackend::new(0).expect("CUDA init");
    let x_gpu = bk
        .from_slice_f32(&x_data, &[batch, t_len, n_heads, p_dim])
        .unwrap();
    let a_gpu = bk
        .from_slice_f32(&a_data, &[batch, t_len, n_heads])
        .unwrap();
    let b_gpu = bk
        .from_slice_f32(&b_data, &[batch, t_len, n_heads, n_dim])
        .unwrap();
    let c_gpu = bk
        .from_slice_f32(&c_data, &[batch, t_len, n_heads, n_dim])
        .unwrap();
    let y_gpu = bk
        .ssd_scan(&x_gpu, &a_gpu, &b_gpu, &c_gpu, block_len)
        .unwrap();
    assert_eq!(y_gpu.shape(), &[batch, t_len, n_heads, p_dim]);
    let got = bk.to_vec_f32(&y_gpu).unwrap();

    let err = max_abs_err(&got, &ref_vals);
    assert!(err < 1e-4, "ssd_scan: max_abs_err={err:.2e} exceeds 1e-4");
}

// ---- 6. ssd_scan fallback when t_len % block_len != 0 (B4.4a) ---------------
//
// t_len=5, block_len=4  → 5 % 4 = 1 ≠ 0, so the PTX kernel cannot run and
// CudaBackend must fall back to pm_core::mamba2::ssd_scan_ops_default.
// We verify parity against CandleBackend (which always uses the pure-Ops path).

#[test]
fn ssd_scan_fallback_non_multiple_matches_candle() {
    let (batch, t_len, n_heads, p_dim, n_dim, block_len) = (1usize, 5, 2, 4, 4, 4);
    // Precondition: t_len % block_len != 0  (5 % 4 = 1).
    assert_ne!(
        t_len % block_len,
        0,
        "test invariant: t_len must NOT be a multiple of block_len"
    );

    let x_data = lcg_vec(batch * t_len * n_heads * p_dim, 4001);
    let a_data: Vec<f32> = lcg_vec(batch * t_len * n_heads, 4002)
        .iter()
        .map(|v| v * 0.1 - 0.1)
        .collect();
    let b_data = lcg_vec(batch * t_len * n_heads * n_dim, 4003);
    let c_data = lcg_vec(batch * t_len * n_heads * n_dim, 4004);

    // Reference: CandleBackend CPU (pure-Ops path).
    let cpu = CandleBackend::new_cpu();
    let x_ref = cpu
        .from_slice_f32(&x_data, &[batch, t_len, n_heads, p_dim])
        .unwrap();
    let a_ref = cpu
        .from_slice_f32(&a_data, &[batch, t_len, n_heads])
        .unwrap();
    let b_ref = cpu
        .from_slice_f32(&b_data, &[batch, t_len, n_heads, n_dim])
        .unwrap();
    let c_ref = cpu
        .from_slice_f32(&c_data, &[batch, t_len, n_heads, n_dim])
        .unwrap();
    let y_ref = cpu
        .ssd_scan(&x_ref, &a_ref, &b_ref, &c_ref, block_len)
        .unwrap();
    let ref_vals = cpu.to_vec_f32(&y_ref).unwrap();
    assert_eq!(y_ref.shape(), &[batch, t_len, n_heads, p_dim]);

    // CUDA — must route through the ssd_scan_ops_default fallback.
    let bk = CudaBackend::new(0).expect("CUDA init");
    let x_gpu = bk
        .from_slice_f32(&x_data, &[batch, t_len, n_heads, p_dim])
        .unwrap();
    let a_gpu = bk
        .from_slice_f32(&a_data, &[batch, t_len, n_heads])
        .unwrap();
    let b_gpu = bk
        .from_slice_f32(&b_data, &[batch, t_len, n_heads, n_dim])
        .unwrap();
    let c_gpu = bk
        .from_slice_f32(&c_data, &[batch, t_len, n_heads, n_dim])
        .unwrap();
    let y_gpu = bk
        .ssd_scan(&x_gpu, &a_gpu, &b_gpu, &c_gpu, block_len)
        .expect("ssd_scan fallback (t_len not multiple of block_len)");
    assert_eq!(y_gpu.shape(), &[batch, t_len, n_heads, p_dim]);
    let got = bk.to_vec_f32(&y_gpu).unwrap();

    let err = max_abs_err(&got, &ref_vals);
    assert!(
        err < 1e-4,
        "ssd_scan_fallback_non_multiple: max_abs_err={err:.2e} exceeds 1e-4"
    );
}
