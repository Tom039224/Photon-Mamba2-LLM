//! Blackwell (RTX 5070, sm_120) + CUDA 13.3 smoke test.
//!
//! Build with the `cuda` feature to target the GPU:
//! ```
//! cargo run --release -p pm-candle --example smoke --features cuda
//! ```
//! Without the feature, this falls back to CPU so the example still
//! compiles on machines without CUDA.
//!
//! What it checks:
//! 1. Device init does not panic.
//! 2. A small (256x256) f32 matmul produces a finite result.
//! 3. RMSNorm runs end-to-end.
//!
//! If any of these fail on Blackwell, log the error in
//! `docs/blackwell-notes.md` and decide whether to (a) wait for a Candle
//! fix, (b) fork Candle for sm_120, or (c) bring forward the Phase 2
//! cudarc backend (per PLAN risk R1).

use pm_backend::Backend;
use pm_candle::CandleBackend;
use pm_core::{Dtype, Ops, Tensor};

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "cuda")]
    let backend = {
        eprintln!("smoke: trying CUDA device 0...");
        CandleBackend::new_cuda(0)?
    };
    #[cfg(not(feature = "cuda"))]
    let backend = {
        eprintln!("smoke: cuda feature disabled; using CPU");
        CandleBackend::new_cpu()
    };

    eprintln!("smoke: backend device_kind = {:?}", backend.device_kind());

    // 256x256 f32 matmul.
    let n: usize = 256;
    let data: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.01).collect();
    let a = backend.from_slice_f32(&data, &[n, n])?;
    let b = backend.from_slice_f32(&data, &[n, n])?;
    let c = backend.matmul(&a, &b)?;
    assert_eq!(c.shape(), &[n, n]);
    let host = backend.to_vec_f32(&c)?;
    let any_finite = host.iter().take(16).all(|x| x.is_finite());
    assert!(any_finite, "smoke matmul produced non-finite values");
    eprintln!("smoke: matmul OK (first value = {})", host[0]);

    // RMSNorm.
    let x = backend.from_slice_f32(&data[..n], &[1, n])?;
    let w = backend.ones(&[n], Dtype::F32)?;
    let y = backend.rmsnorm(&x, &w, 1e-6)?;
    assert_eq!(y.shape(), &[1, n]);
    eprintln!("smoke: rmsnorm OK");

    eprintln!("smoke: all checks passed");
    Ok(())
}
