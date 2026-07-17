//! PLAN J.3 — forward-only SSD scan throughput benchmark.
//!
//! Times `pm_cuda::ssd_scan_chunked_with_context` (the fused PTX
//! kernel, K.1A's eventual forward path) against
//! `CandleBackend::ssd_scan` (the pure-Ops chunked scan that drives
//! Phase 1's autograd). Both are warmed up and the GPU is synced at
//! the end of each timed pass so we measure real device time, not
//! async launch latency.
//!
//! Run with:
//! ```text
//! cargo run -p pm-cuda --example bench_ssd --features cuda --release
//! ```

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use pm_candle::CandleBackend;
use pm_core::Ops;

const WARMUP: usize = 3;
const ITERS: usize = 10;

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

#[allow(clippy::too_many_arguments)]
fn bench_pm_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    shape: (usize, usize, usize, usize, usize, usize),
) -> anyhow::Result<f64> {
    let (batch, t_len, n_heads, p_dim, n_dim, q) = shape;

    for _ in 0..WARMUP {
        pm_cuda::ssd_scan_chunked_with_context(
            ctx, stream, x, a, b, c, y, batch, t_len, n_heads, p_dim, n_dim, q,
        )?;
    }
    stream.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        pm_cuda::ssd_scan_chunked_with_context(
            ctx, stream, x, a, b, c, y, batch, t_len, n_heads, p_dim, n_dim, q,
        )?;
    }
    stream.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() / ITERS as f64)
}

fn bench_pm_candle(
    bk: &CandleBackend,
    x: &<CandleBackend as Ops>::Tensor,
    a: &<CandleBackend as Ops>::Tensor,
    b: &<CandleBackend as Ops>::Tensor,
    c: &<CandleBackend as Ops>::Tensor,
    q: usize,
) -> anyhow::Result<f64> {
    for _ in 0..WARMUP {
        let _ = bk.ssd_scan(x, a, b, c, q)?;
    }
    bk.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = bk.ssd_scan(x, a, b, c, q)?;
    }
    bk.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() / ITERS as f64)
}

fn main() -> anyhow::Result<()> {
    // 100M-config production shape (configs/photon_mamba_100m.toml). The
    // dense ssd_scan_ops_default OOMs here on 12 GB VRAM, but the chunked
    // form (the actual Candle dispatch when block_len < t) does fit.
    let (batch, t_len, n_heads, p_dim, n_dim, q) = (4, 2048, 12, 64, 128, 64);
    let shape = (batch, t_len, n_heads, p_dim, n_dim, q);

    let n_x = batch * t_len * n_heads * p_dim;
    let n_a = batch * t_len * n_heads;
    let n_bn = batch * t_len * n_heads * n_dim;
    let x_host = lcg_vec(11, n_x, 1.0, -0.5);
    let a_host = lcg_vec(12, n_a, 0.05, -0.05);
    let b_host = lcg_vec(13, n_bn, 1.0, -0.5);
    let c_host = lcg_vec(14, n_bn, 1.0, -0.5);

    // I/O footprint (theoretical, fp32 row-major):
    //   x + y         : 2 · B·T·H·P · 4
    //   B + C         : 2 · B·T·H·N · 4
    //   A             :     B·T·H   · 4
    let io_bytes = 2 * n_x * 4 + 2 * n_bn * 4 + n_a * 4;

    println!(
        "shape: B={batch} T={t_len} H={n_heads} P={p_dim} N={n_dim} Q={q}  (warmup={WARMUP}, iters={ITERS})",
    );
    println!(
        "theoretical I/O footprint: {:.1} MB",
        io_bytes as f64 / 1.0e6
    );

    // ---- pm-cuda PTX path ----
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let (free_pre, total) = ctx.mem_get_info()?;
    let x_dev = stream.clone_htod(&x_host)?;
    let a_dev = stream.clone_htod(&a_host)?;
    let b_dev = stream.clone_htod(&b_host)?;
    let c_dev = stream.clone_htod(&c_host)?;
    let mut y_dev = stream.alloc_zeros::<f32>(n_x)?;
    stream.synchronize()?;
    let (free_alloc, _) = ctx.mem_get_info()?;

    let t_cuda = bench_pm_cuda(
        &ctx, &stream, &x_dev, &a_dev, &b_dev, &c_dev, &mut y_dev, shape,
    )?;
    let (free_post, _) = ctx.mem_get_info()?;
    println!(
        "pm-cuda (PTX, on-device tensors):     {:>9.3} ms/iter  ({:>7.1} iter/s)",
        t_cuda * 1e3,
        1.0 / t_cuda,
    );
    println!(
        "  device VRAM total = {:.2} GB,  free before alloc = {:.2} GB,",
        total as f64 / 1.0e9,
        free_pre as f64 / 1.0e9,
    );
    println!(
        "  free after I/O alloc = {:.2} GB  (allocated = {:.1} MB),",
        free_alloc as f64 / 1.0e9,
        (free_pre - free_alloc) as f64 / 1.0e6,
    );
    println!(
        "  free after {ITERS} iters  = {:.2} GB  (peak working set = {:.1} MB)",
        free_post as f64 / 1.0e9,
        (free_pre - free_post) as f64 / 1.0e6,
    );
    drop((x_dev, a_dev, b_dev, c_dev, y_dev));

    // ---- pm-candle pure-Ops path ----
    let bk = CandleBackend::new_cuda(0)?;
    let x_t = bk.from_slice_f32(&x_host, &[batch, t_len, n_heads, p_dim])?;
    let a_t = bk.from_slice_f32(&a_host, &[batch, t_len, n_heads])?;
    let b_t = bk.from_slice_f32(&b_host, &[batch, t_len, n_heads, n_dim])?;
    let c_t = bk.from_slice_f32(&c_host, &[batch, t_len, n_heads, n_dim])?;

    let t_candle = bench_pm_candle(&bk, &x_t, &a_t, &b_t, &c_t, q)?;
    println!(
        "pm-candle (pure-Ops chunked, on-dev): {:>9.3} ms/iter  ({:>7.1} iter/s)",
        t_candle * 1e3,
        1.0 / t_candle,
    );

    let ratio = t_candle / t_cuda;
    println!("\nPTX speedup vs pure-Ops chunked: {ratio:.2}x");
    if ratio >= 5.0 {
        println!("✅ J.3 DoD (≥ 5x throughput) met");
    } else {
        println!("⚠️  J.3 DoD (≥ 5x throughput) NOT met yet — optimisation needed");
    }

    Ok(())
}
