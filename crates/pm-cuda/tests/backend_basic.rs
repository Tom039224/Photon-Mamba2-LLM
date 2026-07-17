//! pm-cuda native backend (B4.1) — round-trip + elementwise smoke.
//!
//! Verifies `CudaBackend::{zeros, from_slice_f32, to_vec_f32, add,
//! sub, mul, neg}` on RTX 5070 with shapes the future Ops trait impl
//! will rely on.

#![cfg(feature = "cuda")]

use pm_cuda::CudaBackend;

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn zeros_round_trip() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.zeros(&[3, 5]).unwrap();
    let v = bk.to_vec_f32(&t).unwrap();
    assert_eq!(v.len(), 15);
    assert!(v.iter().all(|&x| x == 0.0));
}

#[test]
fn from_slice_round_trip() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let host: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 1.0).collect();
    let t = bk.from_slice_f32(&host, &[3, 4]).unwrap();
    let v = bk.to_vec_f32(&t).unwrap();
    assert_eq!(v, host);
}

#[test]
fn elementwise_add_matches_host() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let n = 1024;
    let a_host: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.01).collect();
    let a = bk.from_slice_f32(&a_host, &[n]).unwrap();
    let b = bk.from_slice_f32(&b_host, &[n]).unwrap();
    let c = bk.add(&a, &b).unwrap();
    let got = bk.to_vec_f32(&c).unwrap();
    let want: Vec<f32> = a_host
        .iter()
        .zip(b_host.iter())
        .map(|(x, y)| x + y)
        .collect();
    assert!(max_abs_err(&got, &want) < 1e-6);
}

#[test]
fn elementwise_sub_mul_neg() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let a_host: Vec<f32> = (0..64).map(|i| i as f32 * 0.25).collect();
    let b_host: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
    let a = bk.from_slice_f32(&a_host, &[64]).unwrap();
    let b = bk.from_slice_f32(&b_host, &[64]).unwrap();

    let s = bk.to_vec_f32(&bk.sub(&a, &b).unwrap()).unwrap();
    let m = bk.to_vec_f32(&bk.mul(&a, &b).unwrap()).unwrap();
    let n = bk.to_vec_f32(&bk.neg(&a).unwrap()).unwrap();

    let s_ref: Vec<f32> = a_host.iter().zip(&b_host).map(|(x, y)| x - y).collect();
    let m_ref: Vec<f32> = a_host.iter().zip(&b_host).map(|(x, y)| x * y).collect();
    let n_ref: Vec<f32> = a_host.iter().map(|x| -x).collect();

    assert!(max_abs_err(&s, &s_ref) < 1e-6);
    assert!(max_abs_err(&m, &m_ref) < 1e-6);
    assert!(max_abs_err(&n, &n_ref) < 1e-6);
}
