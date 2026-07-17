//! B4.4e — Per-op wall-time profiler for CudaBackend.
//!
//! Times each key `Ops::*` call at production shapes (PhotonMamba 102 M,
//! B=1, T=512, d_model=768, n_heads=12, d_head=64, d_state=128).
//!
//! GPU stream is synchronized BEFORE and AFTER each timed call so the
//! measured time is real device compute + any host round-trips, not just
//! async launch latency.
//!
//! Run with:
//! ```text
//! cargo run -p pm-cuda --example op_profile --features cuda --release
//! ```
//!
//! Output is a Markdown table suitable for pasting into docs/b4-4e-op-profile.md.

use std::time::Instant;

use pm_core::{Ops, Param};
use pm_cuda::CudaBackend;

// ---- Production shapes -------------------------------------------------------
const B: usize = 1;
const T: usize = 512;
const D_MODEL: usize = 768;
const N_HEADS: usize = 12;
const D_HEAD: usize = 64;
const D_STATE: usize = 128;
const N_GROUPS: usize = 1;
const D_CONV: usize = 4;
const D_INNER: usize = N_HEADS * D_HEAD; // 768
const XBC_DIM: usize = D_INNER + 2 * N_GROUPS * D_STATE; // 1024
const IN_PROJ_DIM: usize = D_INNER + XBC_DIM + N_HEADS; // 1804
const Q: usize = 64; // block_len
const VOCAB: usize = 50257;

/// Pseudo-random f32 vector.
fn lcg_vec(seed: u64, n: usize, scale: f32, bias: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let r = ((state >> 41) as f32) / ((1u32 << 23) as f32);
            r * scale + bias
        })
        .collect()
}

/// Time one op invocation: sync GPU, call `f`, sync GPU again, return ms.
fn time_ms<F: FnOnce() -> anyhow::Result<()>>(bk: &CudaBackend, f: F) -> anyhow::Result<f64> {
    bk.stream().synchronize()?;
    let t = Instant::now();
    f()?;
    bk.stream().synchronize()?;
    Ok(t.elapsed().as_secs_f64() * 1_000.0)
}

/// Average of N timed calls (1 warmup, then N measured).
fn bench<F: FnMut() -> anyhow::Result<()>>(
    bk: &CudaBackend,
    mut f: F,
    n: usize,
) -> anyhow::Result<f64> {
    // warmup
    f()?;
    bk.stream().synchronize()?;
    // timed
    let mut total = 0.0f64;
    for _ in 0..n {
        total += time_ms(bk, &mut f)?;
    }
    Ok(total / n as f64)
}

fn main() -> anyhow::Result<()> {
    let bk = CudaBackend::new(0)?;

    println!("# B4.4e CudaBackend per-op wall-time profile");
    println!("# Config: B={B} T={T} d_model={D_MODEL} n_heads={N_HEADS} d_head={D_HEAD} d_state={D_STATE} n_groups={N_GROUPS} d_conv={D_CONV} Q={Q}");
    println!();

    const N_ITERS: usize = 5;

    let mut rows: Vec<(String, f64, String)> = Vec::new();

    // ---- 1. matmul in_proj: [1,T,D_MODEL] × [D_MODEL, IN_PROJ_DIM] ----------
    {
        let a_data = lcg_vec(1, B * T * D_MODEL, 0.1, 0.0);
        let b_data = lcg_vec(2, D_MODEL * IN_PROJ_DIM, 0.02, 0.0);
        let a = bk.from_slice_f32(&a_data, &[B, T, D_MODEL])?;
        let b = bk.from_slice_f32(&b_data, &[D_MODEL, IN_PROJ_DIM])?;
        let ms = bench(
            &bk,
            || {
                bk.matmul(&a, &b)?;
                Ok(())
            },
            N_ITERS,
        )?;
        rows.push((
            "matmul in_proj [1,512,768]×[768,1804]".into(),
            ms,
            format!("cuBLAS gemm M={T} N={IN_PROJ_DIM} K={D_MODEL} batch=1"),
        ));
    }

    // ---- 2. matmul out_proj: [1,T,D_INNER] × [D_INNER, D_MODEL] -------------
    {
        let a_data = lcg_vec(3, B * T * D_INNER, 0.1, 0.0);
        let b_data = lcg_vec(4, D_INNER * D_MODEL, 0.02, 0.0);
        let a = bk.from_slice_f32(&a_data, &[B, T, D_INNER])?;
        let b = bk.from_slice_f32(&b_data, &[D_INNER, D_MODEL])?;
        let ms = bench(
            &bk,
            || {
                bk.matmul(&a, &b)?;
                Ok(())
            },
            N_ITERS,
        )?;
        rows.push((
            "matmul out_proj [1,512,768]×[768,768]".into(),
            ms,
            format!("cuBLAS gemm M={T} N={D_MODEL} K={D_INNER} batch=1"),
        ));
    }

    // ---- 3. conv1d depthwise: [1,XBC_DIM,T] groups=XBC_DIM K=4 -------------
    {
        let x_data = lcg_vec(5, B * XBC_DIM * T, 0.1, 0.0);
        let w_data = lcg_vec(6, XBC_DIM * D_CONV, 0.02, 0.0);
        let b_data = lcg_vec(7, XBC_DIM, 0.01, 0.0);
        let x = bk.from_slice_f32(&x_data, &[B, XBC_DIM, T])?;
        let w = bk.from_slice_f32(&w_data, &[XBC_DIM, 1, D_CONV])?;
        let bias = bk.from_slice_f32(&b_data, &[XBC_DIM])?;
        let ms = bench(
            &bk,
            || {
                bk.conv1d(&x, &w, Some(&bias), 1, D_CONV - 1, XBC_DIM)?;
                Ok(())
            },
            N_ITERS,
        )?;
        rows.push((
            format!("conv1d depthwise [1,{XBC_DIM},{T}] K={D_CONV} groups={XBC_DIM}"),
            ms,
            "im2col+{XBC_DIM}×tiny-cuBLAS+D2H/H2D loop".to_string(),
        ));
    }

    // ---- 4. broadcast mul: [1,T,N_HEADS,D_HEAD] × [1,T,N_HEADS,1] ----------
    {
        let a_data = lcg_vec(8, B * T * N_HEADS * D_HEAD, 0.1, 0.0);
        let b_data = lcg_vec(9, B * T * N_HEADS, 0.1, 0.5);
        let a = bk.from_slice_f32(&a_data, &[B, T, N_HEADS, D_HEAD])?;
        let b = bk.from_slice_f32(&b_data, &[B, T, N_HEADS, 1])?;
        let ms = bench(
            &bk,
            || {
                bk.mul(&a, &b)?;
                Ok(())
            },
            N_ITERS,
        )?;
        rows.push((
            "broadcast mul [1,512,12,64]×[1,512,12,1]".to_string(),
            ms,
            "broadcast_binary_op: D2H+CPU+H2D; numel=393K".to_string(),
        ));
    }

    // ---- 5. broadcast mul: [1,1,N_HEADS,1] × [1,T,N_HEADS,D_HEAD] ----------
    {
        let a_data = lcg_vec(10, N_HEADS, -0.5, -0.1);
        let b_data = lcg_vec(11, B * T * N_HEADS * D_HEAD, 0.1, 0.0);
        let a = bk.from_slice_f32(&a_data, &[1, 1, N_HEADS, 1])?;
        let b = bk.from_slice_f32(&b_data, &[B, T, N_HEADS, D_HEAD])?;
        let ms = bench(
            &bk,
            || {
                bk.mul(&a, &b)?;
                Ok(())
            },
            N_ITERS,
        )?;
        rows.push((
            "broadcast mul [1,1,12,1]×[1,512,12,64]".to_string(),
            ms,
            "broadcast_binary_op: D2H+CPU+H2D; numel=393K".to_string(),
        ));
    }

    // ---- 6. broadcast add: [1,T,N_HEADS] + [1,1,N_HEADS] -------------------
    {
        let a_data = lcg_vec(12, B * T * N_HEADS, 0.1, 0.0);
        let b_data = lcg_vec(13, N_HEADS, 0.01, 0.0);
        let a = bk.from_slice_f32(&a_data, &[B, T, N_HEADS])?;
        let b = bk.from_slice_f32(&b_data, &[1, 1, N_HEADS])?;
        let ms = bench(
            &bk,
            || {
                bk.add(&a, &b)?;
                Ok(())
            },
            N_ITERS,
        )?;
        rows.push((
            "broadcast add [1,512,12]+[1,1,12]".to_string(),
            ms,
            "broadcast_binary_op: D2H+CPU+H2D; numel=6K".to_string(),
        ));
    }

    // ---- 7. ssd_scan PTX P2: B=1 T=512 H=12 P=64 N=128 Q=64 ----------------
    {
        let x_data = lcg_vec(14, B * T * N_HEADS * D_HEAD, 0.1, 0.0);
        let a_data = lcg_vec(15, B * T * N_HEADS, -0.1, -0.05);
        let b_data = lcg_vec(16, B * T * N_HEADS * D_STATE, 0.1, 0.0);
        let c_data = lcg_vec(17, B * T * N_HEADS * D_STATE, 0.1, 0.0);
        let x = bk.from_slice_f32(&x_data, &[B, T, N_HEADS, D_HEAD])?;
        let a = bk.from_slice_f32(&a_data, &[B, T, N_HEADS])?;
        let b_t = bk.from_slice_f32(&b_data, &[B, T, N_HEADS, D_STATE])?;
        let c = bk.from_slice_f32(&c_data, &[B, T, N_HEADS, D_STATE])?;
        let ms = bench(
            &bk,
            || {
                bk.ssd_scan(&x, &a, &b_t, &c, Q)?;
                Ok(())
            },
            N_ITERS,
        )?;
        rows.push((
            "ssd_scan PTX-P2 [1,512,12,64] Q=64 N=128".to_string(),
            ms,
            "P2 cooperative kernel: 12 blocks @ 2 warps".to_string(),
        ));
    }

    // ---- 8. rmsnorm: [1,T,D_MODEL] ------------------------------------------
    {
        let x_data = lcg_vec(18, B * T * D_MODEL, 0.1, 0.0);
        let w_data = vec![1.0f32; D_MODEL];
        let x = bk.from_slice_f32(&x_data, &[B, T, D_MODEL])?;
        let w = bk.from_slice_f32(&w_data, &[D_MODEL])?;
        let ms = bench(
            &bk,
            || {
                bk.rmsnorm(&x, &w, 1e-5)?;
                Ok(())
            },
            N_ITERS,
        )?;
        rows.push((
            "rmsnorm [1,512,768]".to_string(),
            ms,
            "PTX kernel: 512 rows × 768 cols".to_string(),
        ));
    }

    // ---- 9. log_softmax: [1,T,VOCAB] ----------------------------------------
    {
        let x_data = lcg_vec(19, B * T * VOCAB, 0.1, 0.0);
        let x = bk.from_slice_f32(&x_data, &[B, T, VOCAB])?;
        let ms = bench(
            &bk,
            || {
                bk.log_softmax(&x, 2)?;
                Ok(())
            },
            N_ITERS,
        )?;
        rows.push((
            format!("log_softmax [1,512,{VOCAB}]"),
            ms,
            "PTX kernel: 512 rows × 50257 cols".to_string(),
        ));
    }

    // ---- 10. gather: [1,T,VOCAB] indices [1,T] → [1,T,1] -------------------
    {
        let logits = lcg_vec(20, B * T * VOCAB, 0.1, 0.0);
        let idx_raw: Vec<i64> = (0..B * T).map(|i| (i % VOCAB) as i64).collect();
        let idx_shape_data: Vec<i64> = idx_raw.to_vec();
        let x = bk.from_slice_f32(&logits, &[B * T, VOCAB])?;
        let idx = bk.from_slice_i64(&idx_shape_data, &[B * T, 1])?;
        let ms = bench(
            &bk,
            || {
                bk.gather(&x, &idx, 1)?;
                Ok(())
            },
            N_ITERS,
        )?;
        rows.push((
            "gather [512,50257] dim=1 indices [512,1]".to_string(),
            ms,
            "PTX gather_lastdim kernel".to_string(),
        ));
    }

    // ---- 11. sum_all: [1,512] scalar reduce (loss final step) ---------------
    {
        let x_data = lcg_vec(21, B * T, 0.1, 0.0);
        let x = bk.from_slice_f32(&x_data, &[B, T])?;
        let ms = bench(
            &bk,
            || {
                bk.sum_all(&x)?;
                Ok(())
            },
            N_ITERS,
        )?;
        rows.push((
            "sum_all [1,512]".to_string(),
            ms,
            "D2H → CPU reduce → H2D (host round-trip)".to_string(),
        ));
    }

    // ---- 12. mean_all: [1,512] scalar reduce --------------------------------
    {
        let x_data = lcg_vec(22, B * T, 0.1, 0.0);
        let x = bk.from_slice_f32(&x_data, &[B, T])?;
        let ms = bench(
            &bk,
            || {
                bk.mean_all(&x)?;
                Ok(())
            },
            N_ITERS,
        )?;
        rows.push((
            "mean_all [1,512]".to_string(),
            ms,
            "D2H → CPU reduce → H2D (host round-trip)".to_string(),
        ));
    }

    // ---- 13. concat [1,T,D_INNER] × 8 along dim 1 (ssd chunked) -------------
    {
        // Simulating chunked ssd_scan_ops_default concatenation
        let chunk = B * (T / Q) * N_HEADS * D_HEAD; // per-chunk y_intra
        let chunks: Vec<_> = (0..8usize)
            .map(|i| {
                let d = lcg_vec(30 + i as u64, chunk, 0.1, 0.0);
                bk.from_slice_f32(&d, &[B, T / Q, N_HEADS, D_HEAD]).unwrap()
            })
            .collect();
        let chunks_ref: Vec<_> = chunks.iter().collect();
        let ms = bench(
            &bk,
            || {
                bk.concat(&chunks_ref, 1)?;
                Ok(())
            },
            N_ITERS,
        )?;
        rows.push((
            "concat ×8 [1,8,12,64] dim=1".to_string(),
            ms,
            "host gather D2H×8 + H2D (used in ssd_chunked)".to_string(),
        ));
    }

    // ---- Print table --------------------------------------------------------
    println!("| Op | Avg ms ({N_ITERS} iters) | Note |");
    println!("|---|---:|---|");
    let step_estimate: f64;
    {
        // Per-step estimate for 20 Mamba2 blocks:
        // matmul in_proj×20 + out_proj×20 + conv1d×20 + 4 broadcasts×20 +
        // ssd_scan×20 + rmsnorm×20 + log_softmax + gather + sum_all
        // (backward roughly doubles compute for matmul + ssd + broadcast)
        let in_proj_ms = rows[0].1;
        let out_proj_ms = rows[1].1;
        let conv1d_ms = rows[2].1;
        let bcast_mul_large_ms = rows[3].1; // mul [1,512,12,64]×[1,512,12,1]
        let bcast_mul_small_ms = rows[4].1; // mul [1,1,12,1]×[1,512,12,64]
        let bcast_add_ms = rows[5].1; // add [1,512,12]+[1,1,12]
        let ssd_ms = rows[6].1;
        let rmsnorm_ms = rows[7].1;

        // Forward pass:
        let fwd_matmul = (in_proj_ms + out_proj_ms) * 20.0;
        let fwd_conv = conv1d_ms * 20.0;
        let fwd_bcast = (bcast_mul_large_ms * 2.0 + bcast_mul_small_ms + bcast_add_ms) * 20.0;
        let fwd_ssd = ssd_ms * 20.0;
        let fwd_rmsnorm = rmsnorm_ms * 20.0;

        // Backward (rough factor): matmul×2, conv_bwd×1, ssd_ops_default×1
        // ssd_ops_default backward ≈ 3× ssd_scan (2 matmuls per chunk per VJP)
        let bwd_matmul = fwd_matmul * 2.0;
        let bwd_conv = conv1d_ms * 20.0; // conv_bwd uses host CPU, similar cost
        let bwd_ssd = ssd_ms * 20.0 * 3.0; // sub-walk generates many matmuls

        step_estimate = fwd_matmul
            + fwd_conv
            + fwd_bcast
            + fwd_ssd
            + fwd_rmsnorm
            + bwd_matmul
            + bwd_conv
            + bwd_ssd;

        println!(
            "| **forward matmul ×40 (in+out, 20 blocks)** | **{:.1}** | |",
            fwd_matmul
        );
        println!("| **forward conv1d ×20** | **{:.1}** | |", fwd_conv);
        println!(
            "| **forward broadcast_mul ×40 (large) + ×20 (small) + ×20 (add)** | **{:.1}** | |",
            fwd_bcast
        );
        println!("| **forward ssd_scan ×20** | **{:.1}** | |", fwd_ssd);
        println!("| **backward matmul ×80** | **{:.1}** | |", bwd_matmul);
        println!(
            "| **backward conv1d ×20 (host GEMM)** | **{:.1}** | |",
            bwd_conv
        );
        println!(
            "| **backward ssd_scan (sub-walk)** | **{:.1}** | |",
            bwd_ssd
        );
        println!("| **ESTIMATED step total** | **{:.0}** | |", step_estimate);
    }
    println!();
    println!("## Raw per-op timings");
    println!("| Op | Avg ms | Note |");
    println!("|---|---:|---|");
    for (name, ms, note) in &rows {
        println!("| {name} | {ms:.2} | {note} |");
    }
    println!();
    println!(
        "Measured step ≈ 19600 ms; estimated above ≈ {:.0} ms",
        step_estimate
    );

    // ---- ssd_scan backward (sub-walk timing) --------------------------------
    println!();
    println!("## ssd_scan backward via sub-walk (T=512, 1 call)");
    {
        // Create x,a,b,c as params so ssd_scan records them on tape
        let x_data = lcg_vec(14, B * T * N_HEADS * D_HEAD, 0.1, 0.0);
        let a_data = lcg_vec(15, B * T * N_HEADS, -0.1, -0.05);
        let b_data = lcg_vec(16, B * T * N_HEADS * D_STATE, 0.1, 0.0);
        let c_data = lcg_vec(17, B * T * N_HEADS * D_STATE, 0.1, 0.0);
        let xp = bk.param_from_slice_f32(&x_data, &[B, T, N_HEADS, D_HEAD])?;
        let ap = bk.param_from_slice_f32(&a_data, &[B, T, N_HEADS])?;
        let bp = bk.param_from_slice_f32(&b_data, &[B, T, N_HEADS, D_STATE])?;
        let cp = bk.param_from_slice_f32(&c_data, &[B, T, N_HEADS, D_STATE])?;

        // Warmup pass
        let y_w = bk.ssd_scan(
            xp.as_tensor(),
            ap.as_tensor(),
            bp.as_tensor(),
            cp.as_tensor(),
            Q,
        )?;
        let s_w = bk.sum_all(&y_w)?;
        let _gs_w = bk.backward(&s_w)?;
        bk.stream().synchronize()?;

        // Timed: forward (PTX) + backward (sub-walk with ops_default)
        let xp2 = bk.param_from_slice_f32(&x_data, &[B, T, N_HEADS, D_HEAD])?;
        let ap2 = bk.param_from_slice_f32(&a_data, &[B, T, N_HEADS])?;
        let bp2 = bk.param_from_slice_f32(&b_data, &[B, T, N_HEADS, D_STATE])?;
        let cp2 = bk.param_from_slice_f32(&c_data, &[B, T, N_HEADS, D_STATE])?;
        bk.stream().synchronize()?;

        let t_start = Instant::now();
        let y = bk.ssd_scan(
            xp2.as_tensor(),
            ap2.as_tensor(),
            bp2.as_tensor(),
            cp2.as_tensor(),
            Q,
        )?;
        let s = bk.sum_all(&y)?;
        let _gs = bk.backward(&s)?;
        bk.stream().synchronize()?;
        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        let bwd_ms = total_ms - 53.46; // subtract measured forward time
        println!("| ssd_scan fwd+bwd (1 call, T={T}) | {total_ms:.1} ms | fwd={:.1} bwd≈{bwd_ms:.1} ms |", 53.46f64);
        println!(
            "| ssd_scan bwd ×10 (T=512 blocks) | {:.0} ms | sub-walk via ops_default |",
            bwd_ms * 10.0
        );
    }

    // ---- conv1d backward specifically ----------------------------------------
    println!();
    println!("## conv1d backward only (T=512)");
    {
        let x_data = lcg_vec(5, B * XBC_DIM * T, 0.1, 0.0);
        let w_data = lcg_vec(6, XBC_DIM * D_CONV, 0.02, 0.0);
        let b_data = lcg_vec(7, XBC_DIM, 0.01, 0.0);
        let xp = bk.param_from_slice_f32(&x_data, &[B, XBC_DIM, T])?;
        let wp = bk.param_from_slice_f32(&w_data, &[XBC_DIM, 1, D_CONV])?;
        let bp = bk.param_from_slice_f32(&b_data, &[XBC_DIM])?;

        // Forward + backward
        let out = bk.conv1d(
            xp.as_tensor(),
            wp.as_tensor(),
            Some(bp.as_tensor()),
            1,
            D_CONV - 1,
            XBC_DIM,
        )?;
        let loss = bk.sum_all(&out)?;
        bk.stream().synchronize()?;
        let t_start = Instant::now();
        let _gs = bk.backward(&loss)?;
        bk.stream().synchronize()?;
        let bwd_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        println!("| conv1d backward only [1,{XBC_DIM},{T}] groups={XBC_DIM} | {bwd_ms:.1} ms | host-GEMM path |");
        bk.reset_tape()?;
    }

    Ok(())
}
