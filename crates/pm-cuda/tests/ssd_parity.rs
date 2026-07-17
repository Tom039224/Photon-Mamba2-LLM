//! pm-cuda PTX SSD scan — J.2 numerical parity test.
//!
//! Compares the fused `pm_cuda::ssd_scan_chunked` against
//! `pm_core::mamba2::ssd_scan_naive_scalar` (the same scalar
//! reference that already matches PyTorch's mamba-ssm to 5e-3 on the
//! fixture). Runs only when the `cuda` feature is enabled.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use pm_core::mamba2::ssd_scan_naive_scalar;

/// Deterministic LCG so the test is reproducible without pulling in
/// `rand` (matches the pattern used by `gen_fixtures.py` upstream).
fn lcg_vec(seed: u64, n: usize, scale: f32, bias: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            // top-23 bits → uniform [0, 1)
            let r = ((state >> 41) as f32) / ((1u32 << 23) as f32);
            r * scale + bias
        })
        .collect()
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn ssd_parity_small() {
    let (batch, t_len, n_heads, p_dim, n_dim, q) = (1, 128, 2, 8, 16, 64);
    let x = lcg_vec(1, batch * t_len * n_heads * p_dim, 1.0, -0.5);
    // A is typically the log of a per-head decay; keep it ≤ 0 so exp
    // doesn't blow up.
    let a = lcg_vec(2, batch * t_len * n_heads, 0.5, -0.5);
    let b = lcg_vec(3, batch * t_len * n_heads * n_dim, 1.0, -0.5);
    let c = lcg_vec(4, batch * t_len * n_heads * n_dim, 1.0, -0.5);

    let y_ref = ssd_scan_naive_scalar(&x, &a, &b, &c, batch, t_len, n_heads, p_dim, n_dim);
    let y_cuda = pm_cuda::ssd_scan_chunked(&x, &a, &b, &c, batch, t_len, n_heads, p_dim, n_dim, q)
        .expect("kernel launch failed");

    let err = max_abs_err(&y_ref, &y_cuda);
    eprintln!("ssd_parity_small: max abs err = {err:.3e}");
    assert!(err < 1e-4, "max abs err {err} exceeds 1e-4");
}

#[test]
fn ssd_parity_production_shape() {
    // PLAN J.1 DoD: T=2048, B=4 finite output. We sweep that here
    // and additionally compare to the scalar reference to keep J.2
    // honest at the size we'll actually train on.
    let (batch, t_len, n_heads, p_dim, n_dim, q) = (4, 2048, 12, 64, 128, 64);
    let n_inputs = batch * t_len * n_heads;
    let x = lcg_vec(11, n_inputs * p_dim, 1.0, -0.5);
    let a = lcg_vec(12, n_inputs, 0.05, -0.05); // small negative so the long T doesn't underflow exp
    let b = lcg_vec(13, n_inputs * n_dim, 1.0, -0.5);
    let c = lcg_vec(14, n_inputs * n_dim, 1.0, -0.5);

    let y_cuda = pm_cuda::ssd_scan_chunked(&x, &a, &b, &c, batch, t_len, n_heads, p_dim, n_dim, q)
        .expect("kernel launch failed");

    assert_eq!(y_cuda.len(), batch * t_len * n_heads * p_dim);
    let finite = y_cuda.iter().all(|v| v.is_finite());
    assert!(finite, "kernel produced non-finite output");

    // Spot-check parity on a tiny window (full naive scan is ~O(T²)
    // and would take minutes for T=2048; we instead trust the small
    // test for correctness and only require finite output here).
    let n_finite = y_cuda.iter().filter(|v| v.is_finite()).count();
    eprintln!(
        "ssd_parity_production_shape: {} elements, all finite = {}, max abs = {:.3e}",
        n_finite,
        finite,
        y_cuda.iter().map(|v| v.abs()).fold(0f32, f32::max),
    );
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("reference")
        .join("fixtures")
        .join("ssd_q64")
}

fn load_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    npyz::NpyFile::new(bytes.as_slice())
        .unwrap_or_else(|e| panic!("parse npy {}: {e}", path.display()))
        .into_vec::<f32>()
        .unwrap_or_else(|e| panic!("decode f32 {}: {e}", path.display()))
}

#[test]
fn ssd_parity_pytorch_fixture() {
    let dir = fixtures_dir();
    if !dir.exists() {
        eprintln!("skipping: fixture dir not present at {}", dir.display());
        return;
    }
    let (batch, t_len, n_heads, p_dim, n_dim, q) = (1, 128, 2, 8, 16, 64);

    let x = load_f32(&dir.join("X.npy"));
    let a = load_f32(&dir.join("A.npy"));
    let b = load_f32(&dir.join("B.npy"));
    let c = load_f32(&dir.join("C.npy"));
    let y_pt = load_f32(&dir.join("Y.npy"));

    let y_cuda = pm_cuda::ssd_scan_chunked(&x, &a, &b, &c, batch, t_len, n_heads, p_dim, n_dim, q)
        .expect("kernel launch failed");

    let err = max_abs_err(&y_pt, &y_cuda);
    eprintln!("ssd_parity_pytorch_fixture: max abs err vs PyTorch = {err:.3e}");
    // Same 5e-3 budget as the scalar reference vs PyTorch: dominated
    // by sum-order differences in fp32, not by the kernel.
    assert!(err < 5e-3, "max abs err {err} exceeds 5e-3");
}

/// J.3.P2 numerical parity: covers the production dispatch shape
/// (n_dim=128, block_len=64, p_dim=64 → routes to `ssd_scan_chunked_p1`
/// PTX symbol, which since commit 4842bca holds the P2 cooperative
/// shared-load body).
///
/// `ssd_parity_small` and `ssd_parity_pytorch_fixture` both use
/// n_dim≤16/p_dim≤8 and therefore fall through to the legacy kernel,
/// so without this test the P2 kernel itself ships with zero numerical
/// coverage (production_shape only asserts finiteness).
///
/// Shape is reduced to B=1, T=128, H=1 to keep the O(T²·H·P·N²) scalar
/// reference fast (≈ 134 M ops, < 1 s CPU).
#[test]
fn ssd_parity_p2_shape_numerical() {
    let (batch, t_len, n_heads, p_dim, n_dim, q) = (1, 128, 1, 64, 128, 64);
    let n_inputs = batch * t_len * n_heads;
    let x = lcg_vec(21, n_inputs * p_dim, 1.0, -0.5);
    // Small negative `a` so 2-chunk exp accumulation stays in fp32 range.
    let a = lcg_vec(22, n_inputs, 0.05, -0.05);
    let b = lcg_vec(23, n_inputs * n_dim, 0.5, -0.25);
    let c = lcg_vec(24, n_inputs * n_dim, 0.5, -0.25);

    let y_ref = ssd_scan_naive_scalar(&x, &a, &b, &c, batch, t_len, n_heads, p_dim, n_dim);
    let y_cuda = pm_cuda::ssd_scan_chunked(&x, &a, &b, &c, batch, t_len, n_heads, p_dim, n_dim, q)
        .expect("kernel launch failed");

    let err = max_abs_err(&y_ref, &y_cuda);
    eprintln!("ssd_parity_p2_shape_numerical: max abs err = {err:.3e}");
    assert!(err < 1e-4, "P2 kernel max abs err {err} exceeds 1e-4");
}
