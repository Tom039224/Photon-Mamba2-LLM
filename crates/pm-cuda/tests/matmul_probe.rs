//! Phase B'.2a matmul split-probe (PLAN.md B'.2 ①).
//!
//! Decomposes the ~63 ms/call forward `matmul` observed by the B'.1
//! per-op profile (docs/perf-log.md 2026-07-04 §2 "open question") into:
//!   raw cuBLAS gemm_v2 (2D) vs gemm_strided_batched (rank-3, batch=1)
//!   vs `alloc_zeros` vs the full `Ops::matmul` wrapper (tape included),
//! at the production shapes of the 102M model
//! (in_proj-like 512x768@768x3340, out_proj-like 512x1536@1536x768).
//!
//! "lat" = per-call latency with a stream sync after every call — the same
//! measurement the B'.1 profiler makes. "amortized" = N launches then one
//! sync, i.e. what a purely async pipeline would cost.
//!
//! Not a correctness test; excluded from normal runs. Execute manually:
//!   PATH=/opt/cuda/bin:$PATH CUDA_HOME=/opt/cuda \
//!     cargo test -p pm-cuda --features cuda --test matmul_probe -- --ignored --nocapture

#![cfg(feature = "cuda")]

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaSlice, CudaStream};
use pm_core::Ops;
use pm_cuda::backend::kernels;
use pm_cuda::CudaBackend;
use std::sync::Arc;
use std::time::Instant;

fn filled(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i.wrapping_mul(2654435761)) & 0xffff) as f32 / 65536.0 - 0.5)
        .collect()
}

fn lat(label: &str, stream: &Arc<CudaStream>, iters: usize, warmup: usize, mut f: impl FnMut()) {
    for _ in 0..warmup {
        f();
    }
    stream.synchronize().expect("sync");
    let mut total = 0.0f64;
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        stream.synchronize().expect("sync");
        total += t0.elapsed().as_secs_f64();
    }
    println!(
        "{label:<56} {:>12.1} us/call  (lat, n={iters})",
        total * 1e6 / iters as f64
    );
}

fn thr(label: &str, stream: &Arc<CudaStream>, iters: usize, warmup: usize, mut f: impl FnMut()) {
    for _ in 0..warmup {
        f();
    }
    stream.synchronize().expect("sync");
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    stream.synchronize().expect("sync");
    println!(
        "{label:<56} {:>12.1} us/call  (amortized, n={iters})",
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    );
}

#[test]
#[ignore = "manual B'.2a probe — run with --ignored --nocapture"]
fn matmul_split_probe() {
    let bk = CudaBackend::new(0).expect("cuda init");
    let stream = bk.stream().clone();
    let cublas = Arc::new(CudaBlas::new(stream.clone()).expect("cublas init"));

    // ---------- shape S1: in_proj-like (512,768)@(768,3340) ----------
    let (m, k, n) = (512usize, 768usize, 3340usize);
    let a_host = filled(m * k);
    let b_host = filled(k * n);
    let a_dev: CudaSlice<f32> = stream.clone_htod(&a_host).expect("htod a");
    let b_dev: CudaSlice<f32> = stream.clone_htod(&b_host).expect("htod b");
    let mut out_dev: CudaSlice<f32> = stream.alloc_zeros::<f32>(m * n).expect("alloc out");

    // cold first calls (cuBLAS heuristics / JIT, per API entry point)
    let t0 = Instant::now();
    kernels::matmul_f32(&cublas, &a_dev, &[m, k], &b_dev, &[k, n], &mut out_dev).expect("2d");
    stream.synchronize().expect("sync");
    println!(
        "{:<56} {:>12.1} us  (COLD first call)",
        "raw gemm_v2 2D S1",
        t0.elapsed().as_secs_f64() * 1e6
    );
    let t0 = Instant::now();
    kernels::matmul_f32(
        &cublas,
        &a_dev,
        &[1, m, k],
        &b_dev,
        &[1, k, n],
        &mut out_dev,
    )
    .expect("3d");
    stream.synchronize().expect("sync");
    println!(
        "{:<56} {:>12.1} us  (COLD first call)",
        "raw strided_batched b=1 S1",
        t0.elapsed().as_secs_f64() * 1e6
    );

    lat(
        "raw gemm_v2 2D  [512,768]@[768,3340]",
        &stream,
        40,
        8,
        || {
            kernels::matmul_f32(&cublas, &a_dev, &[m, k], &b_dev, &[k, n], &mut out_dev)
                .expect("2d");
        },
    );
    thr(
        "raw gemm_v2 2D  [512,768]@[768,3340]",
        &stream,
        40,
        8,
        || {
            kernels::matmul_f32(&cublas, &a_dev, &[m, k], &b_dev, &[k, n], &mut out_dev)
                .expect("2d");
        },
    );
    lat(
        "raw strided_batched b=1  [1,512,768]@[1,768,3340]",
        &stream,
        40,
        8,
        || {
            kernels::matmul_f32(
                &cublas,
                &a_dev,
                &[1, m, k],
                &b_dev,
                &[1, k, n],
                &mut out_dev,
            )
            .expect("3d");
        },
    );
    thr(
        "raw strided_batched b=1  [1,512,768]@[1,768,3340]",
        &stream,
        40,
        8,
        || {
            kernels::matmul_f32(
                &cublas,
                &a_dev,
                &[1, m, k],
                &b_dev,
                &[1, k, n],
                &mut out_dev,
            )
            .expect("3d");
        },
    );

    // ---------- allocator ----------
    lat("alloc_zeros(512*3340) + drop", &stream, 40, 8, || {
        let x = stream.alloc_zeros::<f32>(m * n).expect("alloc");
        drop(x);
    });

    // ---------- Ops-level (tape active, same shapes) ----------
    let a_t3 = Ops::from_slice_f32(&bk, &a_host, &[1, m, k]).expect("a_t3");
    let b_t2 = Ops::from_slice_f32(&bk, &b_host, &[k, n]).expect("b_t2");
    lat(
        "Ops::matmul rank3xrank2 (padded -> strided b=1)",
        &stream,
        40,
        8,
        || {
            let _o = Ops::matmul(&bk, &a_t3, &b_t2).expect("mm3");
        },
    );
    let a_t2 = Ops::from_slice_f32(&bk, &a_host, &[m, k]).expect("a_t2");
    lat(
        "Ops::matmul rank2xrank2 (gemm_v2 path)",
        &stream,
        40,
        8,
        || {
            let _o = Ops::matmul(&bk, &a_t2, &b_t2).expect("mm2");
        },
    );

    // ---------- shape S2: out_proj-like (512,1536)@(1536,768) ----------
    let (m2, k2, n2) = (512usize, 1536usize, 768usize);
    let a2_dev: CudaSlice<f32> = stream.clone_htod(&filled(m2 * k2)).expect("htod a2");
    let b2_dev: CudaSlice<f32> = stream.clone_htod(&filled(k2 * n2)).expect("htod b2");
    let mut out2: CudaSlice<f32> = stream.alloc_zeros::<f32>(m2 * n2).expect("alloc out2");
    lat(
        "raw gemm_v2 2D  [512,1536]@[1536,768]",
        &stream,
        40,
        8,
        || {
            kernels::matmul_f32(&cublas, &a2_dev, &[m2, k2], &b2_dev, &[k2, n2], &mut out2)
                .expect("2d s2");
        },
    );
    lat(
        "raw strided_batched b=1  [1,512,1536]@[1,1536,768]",
        &stream,
        40,
        8,
        || {
            kernels::matmul_f32(
                &cublas,
                &a2_dev,
                &[1, m2, k2],
                &b2_dev,
                &[1, k2, n2],
                &mut out2,
            )
            .expect("3d s2");
        },
    );

    // ---------- shape S3: T=2048 scaling check ----------
    let (m3, k3, n3) = (2048usize, 768usize, 3340usize);
    let a3_dev: CudaSlice<f32> = stream.clone_htod(&filled(m3 * k3)).expect("htod a3");
    let b3_dev: CudaSlice<f32> = stream.clone_htod(&filled(k3 * n3)).expect("htod b3");
    let mut out3: CudaSlice<f32> = stream.alloc_zeros::<f32>(m3 * n3).expect("alloc out3");
    lat(
        "raw gemm_v2 2D  [2048,768]@[768,3340]",
        &stream,
        10,
        3,
        || {
            kernels::matmul_f32(&cublas, &a3_dev, &[m3, k3], &b3_dev, &[k3, n3], &mut out3)
                .expect("2d s3");
        },
    );
    lat(
        "raw strided_batched b=1  [1,2048,768]@[1,768,3340]",
        &stream,
        10,
        3,
        || {
            kernels::matmul_f32(
                &cublas,
                &a3_dev,
                &[1, m3, k3],
                &b3_dev,
                &[1, k3, n3],
                &mut out3,
            )
            .expect("3d s3");
        },
    );
    drop(out3);
    drop(a3_dev);
    drop(b3_dev);

    // ---------- context (1): pool pressure + fragmentation ----------
    // Training holds thousands of live buffers (~5-8 GB) in the
    // stream-ordered pool with a churn pattern. Reproduce, then re-measure
    // the same S1 gemm (operands re-allocated post-churn, like training).
    let sizes = [
        1_000_000usize,
        3_000_000,
        700_000,
        5_000_000,
        130_000,
        2_500_000,
        60_000,
    ];
    let mut hold: Vec<CudaSlice<f32>> = Vec::new();
    let mut live: usize = 0;
    let mut i = 0usize;
    while live < 1_200_000_000 / 4 {
        // ~1.2 GB... grown below to ~5 GB
        let n_e = sizes[i % sizes.len()];
        let buf = stream.alloc_zeros::<f32>(n_e).expect("churn alloc");
        if i.is_multiple_of(3) {
            drop(buf);
        } else {
            live += n_e;
            hold.push(buf);
        }
        i += 1;
    }
    // free every 4th held buffer to fragment, then grow to ~5 GB live
    let mut j = 0usize;
    hold.retain(|_| {
        j += 1;
        !j.is_multiple_of(4)
    });
    while live < 5_000_000_000 / 4 {
        let n_e = sizes[i % sizes.len()];
        live += n_e;
        hold.push(stream.alloc_zeros::<f32>(n_e).expect("grow alloc"));
        i += 1;
    }
    stream.synchronize().expect("sync");
    println!(
        "--- pool now ~{:.1} GB live, fragmented ---",
        live as f64 * 4.0 / 1e9
    );

    let a_p: CudaSlice<f32> = stream.clone_htod(&a_host).expect("htod a_p");
    let b_p: CudaSlice<f32> = stream.clone_htod(&b_host).expect("htod b_p");
    let mut out_p: CudaSlice<f32> = stream.alloc_zeros::<f32>(m * n).expect("alloc out_p");
    lat(
        "raw gemm_v2 2D S1  (under pool pressure)",
        &stream,
        40,
        8,
        || {
            kernels::matmul_f32(&cublas, &a_p, &[m, k], &b_p, &[k, n], &mut out_p).expect("2d p");
        },
    );
    lat(
        "alloc_zeros(512*3340)+drop  (under pool pressure)",
        &stream,
        40,
        8,
        || {
            let x = stream.alloc_zeros::<f32>(m * n).expect("alloc p");
            drop(x);
        },
    );
    lat(
        "Ops::matmul rank3xrank2  (under pool pressure)",
        &stream,
        40,
        8,
        || {
            let _o = Ops::matmul(&bk, &a_t3, &b_t2).expect("mm3 p");
        },
    );

    // ---------- context (2): shape-mix heuristic-cache thrash ----------
    // Training cycles ~50+ distinct gemm shapes per step. Cycle 64 distinct
    // shapes round-robin and report the average per-call latency once every
    // shape has been seen (steady state for any per-shape cache).
    let shapes: Vec<(usize, usize, usize)> = (0..64)
        .map(|s| (256 + 8 * (s % 16), 768usize, 512 + 16 * (s % 32)))
        .collect();
    let bufs: Vec<(CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>)> = shapes
        .iter()
        .map(|&(sm, sk, sn)| {
            (
                stream.clone_htod(&filled(sm * sk)).expect("a s"),
                stream.clone_htod(&filled(sk * sn)).expect("b s"),
                stream.alloc_zeros::<f32>(sm * sn).expect("o s"),
            )
        })
        .collect();
    // warmup: every shape once (cold pass, excluded)
    let mut bufs = bufs;
    for (idx, &(sm, sk, sn)) in shapes.iter().enumerate() {
        let (ref a_s, ref b_s, ref mut o_s) = bufs[idx];
        kernels::matmul_f32(&cublas, a_s, &[sm, sk], b_s, &[sk, sn], o_s).expect("warm s");
    }
    stream.synchronize().expect("sync");
    let rounds = 4usize;
    let t0 = Instant::now();
    for _ in 0..rounds {
        for (idx, &(sm, sk, sn)) in shapes.iter().enumerate() {
            let (ref a_s, ref b_s, ref mut o_s) = bufs[idx];
            kernels::matmul_f32(&cublas, a_s, &[sm, sk], b_s, &[sk, sn], o_s).expect("mix s");
            stream.synchronize().expect("sync");
        }
    }
    println!(
        "{:<56} {:>12.1} us/call  (lat, 64 shapes x {rounds} rounds)",
        "raw gemm_v2 2D  cycling 64 distinct shapes",
        t0.elapsed().as_secs_f64() * 1e6 / (rounds * shapes.len()) as f64
    );

    drop(bufs);
    drop(hold);
}
