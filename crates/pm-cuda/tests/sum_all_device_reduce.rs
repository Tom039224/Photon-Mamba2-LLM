//! Numerical regression test for the device-side `sum_all`/`mean_all`
//! reduce (Phase B'.2f): `CudaBackend::sum_all`/`mean_all` now run a
//! deterministic two-pass device reduce (`kernels::sum_all_f32` — grid-
//! stride partial sums, then a fixed-order final reduction, no atomics)
//! instead of a D2H → host `iter().sum()` → H2D round trip.
//!
//! `ops_reduce.rs::{sum_all,mean_all}_matches_candle` already covers a
//! small fixed shape (4×128 = 512 elements); this file specifically
//! targets the **size-dependent** part of the two-pass design (the
//! `n_chunks = ceil(sqrt(numel))` chunk count, and whether `numel` evenly
//! divides `n_chunks`), across shapes from a few elements up to ~1e6.

#![cfg(feature = "cuda")]

use pm_core::Ops;
use pm_cuda::CudaBackend;

fn lcg_vec(seed: u64, n: usize, scale: f32, bias: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let r = ((state >> 41) as f32) / ((1u32 << 23) as f32);
            r * scale + bias
        })
        .collect()
}

/// Checks `sum_all` and `mean_all` against an **f64-accumulated** host
/// reference, for a tensor of `n` elements.
///
/// A naive sequential **f32** host sum is not a trustworthy ground truth
/// at n ~ 1e6: sequential summation has O(n·eps) worst-case error growth,
/// while the device's two-pass chunked/tree reduce has only O(log n·eps)
/// — so at large n the device result is typically *more* accurate than a
/// naive f32 host sum, not less. Comparing both against an f64 reference
/// (whose own rounding error is negligible at f32 magnitudes) avoids
/// conflating "device reduce disagrees with a naive host sum" with "device
/// reduce is wrong".
fn check_sum_and_mean_at_size(n: usize, seed: u64) {
    let data = lcg_vec(seed, n, 2.0, -1.0);
    let host_sum_f64: f64 = data.iter().map(|&v| v as f64).sum();
    let host_mean_f64 = host_sum_f64 / n as f64;

    let bk = CudaBackend::new(0).expect("CUDA init");
    let t = bk.from_slice_f32(&data, &[n]).expect("from_slice_f32");

    let sum_t = bk.sum_all(&t).expect("sum_all");
    let sum_got = bk.to_vec_f32(&sum_t).expect("sum to_vec")[0];
    let mean_t = bk.mean_all(&t).expect("mean_all");
    let mean_got = bk.to_vec_f32(&mean_t).expect("mean to_vec")[0];

    // Relative tolerance (against the f64 reference) with an absolute
    // floor for values that land near zero by chance.
    let rel_tol = 1e-5_f64;
    let sum_denom = host_sum_f64.abs().max(1e-6);
    let sum_rel_err = (sum_got as f64 - host_sum_f64).abs() / sum_denom;
    let mean_denom = host_mean_f64.abs().max(1e-6);
    let mean_rel_err = (mean_got as f64 - host_mean_f64).abs() / mean_denom;

    eprintln!(
        "sum_all_device_reduce[n={n}]: sum f64_ref={host_sum_f64:.6e} got={sum_got:.6e} \
         rel_err={sum_rel_err:.3e} | mean f64_ref={host_mean_f64:.6e} got={mean_got:.6e} \
         rel_err={mean_rel_err:.3e}"
    );
    assert!(
        sum_rel_err < rel_tol,
        "n={n}: sum_all rel_err={sum_rel_err:.3e} exceeds {rel_tol:.1e} \
         (f64_ref={host_sum_f64}, got={sum_got})"
    );
    assert!(
        mean_rel_err < rel_tol,
        "n={n}: mean_all rel_err={mean_rel_err:.3e} exceeds {rel_tol:.1e} \
         (f64_ref={host_mean_f64}, got={mean_got})"
    );
}

#[test]
fn sum_all_matches_host_across_sizes() {
    // Deliberately includes: n=1 (n_chunks clamps to 1); n smaller than a
    // perfect square (ragged last grid-stride visit); n exactly a perfect
    // square (n_chunks*n_chunks == n, no raggedness at all); a mid-size
    // shape typical of a per-token loss (T=512); and a ~1e6-element shape
    // that forces genuinely multi-block grids in both reduce passes.
    let sizes = [1usize, 2, 7, 64, 100, 512, 4096, 40_000, 1_048_576];
    for (i, &n) in sizes.iter().enumerate() {
        check_sum_and_mean_at_size(n, 7000 + i as u64);
    }
}

/// Same check at a large size with a different random seed/scale, to
/// avoid over-fitting to one particular data distribution.
#[test]
fn sum_all_matches_host_large_alt_distribution() {
    check_sum_and_mean_at_size(1_000_003, 99); // prime-ish size, not a perfect square
}
