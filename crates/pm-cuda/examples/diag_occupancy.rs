//! Phase B'.3 wave-2 diagnostic — occupancy calculator for
//! `ssd_scan_chunked_p1` (the J.3.P2 SSD scan kernel).
//!
//! Not a benchmark. Answers "how many blocks of this kernel are
//! simultaneously resident per SM, and what fraction of the SM's warp
//! slots does that use" using only CUDA **driver-API introspection**
//! (`cuFuncGetAttribute`, `cuDeviceGetAttribute`,
//! `cuOccupancyMaxActiveBlocksPerMultiprocessor`) — no Nsight Compute /
//! perf-counter permission required. This is the numeric backing for the
//! occupancy-bound hypothesis in `docs/perf-log.md`'s B'.3 wave-2 entry.
//!
//! Diagnostic-only artifact for that investigation; safe to delete once
//! the numbers it prints have been transcribed into the report, or keep
//! it around as a reusable occupancy probe for future kernel work (P3
//! rewrite, other production-shape kernels).
//!
//! Run with:
//! ```text
//! cargo run -p pm-cuda --example diag_occupancy --features cuda --release
//! ```

use cudarc::driver::sys::{CUdevice_attribute_enum as DAttr, CUfunction_attribute_enum as FAttr};
use cudarc::driver::CudaContext;
use cudarc::nvrtc::Ptx;

/// Must match `ssd.rs::KERNEL_P1_NAME`.
const KERNEL_NAME: &str = "ssd_scan_chunked_p1";
/// Must match `ssd.rs::P1_SMEM_BYTES` (100 352 B dynamic shared memory).
const P1_SMEM_BYTES: i32 = 100_352;
/// Must match `ssd.rs::P1_P_DIM` (threads per block = p_dim).
const BLOCK_SIZE: u32 = 64;
/// Production `n_heads` (`configs/photon_mamba_100m.toml`).
const H: usize = 12;

fn main() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0)?;

    let sm_count = ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let max_threads_per_sm =
        ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?;
    let max_threads_per_block = ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?;
    let smem_per_block_default =
        ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?;
    let smem_per_block_optin =
        ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?;
    let smem_per_sm = ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)?;
    let regs_per_block_max = ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)?;
    let regs_per_sm = ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)?;
    let max_blocks_per_sm_hw = ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR)?;
    let warp_size = ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_WARP_SIZE)?;
    let cc_major = ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let cc_minor = ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    let clock_khz = ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)?;
    let mem_clock_khz = ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)?;
    let bus_width_bits = ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?;
    let l2_bytes = ctx.attribute(DAttr::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?;

    println!("=== Device: sm_{cc_major}{cc_minor} ===");
    println!("SM count                             : {sm_count}");
    println!("max threads / SM                     : {max_threads_per_sm}");
    println!("max threads / block                  : {max_threads_per_block}");
    println!("warp size                            : {warp_size}");
    println!("shared mem / block (default, 48KB)   : {smem_per_block_default} B");
    println!("shared mem / block (opt-in max)      : {smem_per_block_optin} B");
    println!("shared mem / SM (hard cap)           : {smem_per_sm} B");
    println!("registers / block (max)              : {regs_per_block_max}");
    println!("registers / SM                       : {regs_per_sm}");
    println!("max resident blocks / SM (hw limit)  : {max_blocks_per_sm_hw}");
    println!(
        "boost clock (reported max)           : {:.3} GHz",
        clock_khz as f64 / 1e6
    );
    println!(
        "memory clock (pin rate, as reported) : {:.3} GHz",
        mem_clock_khz as f64 / 1e6
    );
    println!("memory bus width                     : {bus_width_bits} bits");
    println!("L2 cache                             : {} KB", l2_bytes / 1024);

    // Sanity-check bandwidth estimate: driver reports the raw pin clock;
    // GDDR6/7 transfer 2 bits/pin/Hz (DDR), so peak = 2 * clock * (bus/8).
    // Cross-check against the vendor's advertised spec, not authoritative.
    let peak_bw_gbs = 2.0 * (mem_clock_khz as f64 * 1000.0) * (bus_width_bits as f64 / 8.0) / 1e9;
    println!(
        "peak mem bandwidth (2*clock*width/8) : {peak_bw_gbs:.1} GB/s (sanity check vs. vendor spec)"
    );

    println!();
    println!("=== Kernel: {KERNEL_NAME} ===");
    let module = ctx.load_module(Ptx::from_src(pm_cuda::KERNEL_PTX))?;
    let func = module.load_function(KERNEL_NAME)?;

    // Mirror production launch config (ssd.rs sets this before every
    // launch) so the queried attributes reflect the real runtime state.
    func.set_attribute(
        FAttr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        P1_SMEM_BYTES,
    )?;

    let num_regs = func.num_regs()?;
    let local_bytes = func.local_size_bytes()?;
    let static_smem = func.shared_size_bytes()?;
    let max_tpb = func.max_threads_per_block()?;

    println!("registers / thread                   : {num_regs}");
    println!("local mem / thread (spill)           : {local_bytes} B  (0 = no spill)");
    println!("static shared mem                    : {static_smem} B");
    println!("dynamic shared mem (this launch cfg) : {P1_SMEM_BYTES} B");
    println!(
        "total shared mem / block             : {} B",
        static_smem + P1_SMEM_BYTES
    );
    println!("max threads/block (function limit)   : {max_tpb}");
    println!(
        "launch config block size             : {BLOCK_SIZE} threads ({} warps)",
        BLOCK_SIZE / warp_size as u32
    );

    // ---- Hand-computed occupancy bounds (cross-check only) ----
    //
    // These ignore hardware allocation-granularity rounding (e.g.
    // registers are handed out in fixed-size chunks per warp/block,
    // shared memory in fixed-size pages) — the driver's own calculator
    // below is authoritative; this is just to show the dominant term by
    // hand, matching the format the report asks for.
    let smem_total_per_block = i64::from(static_smem + P1_SMEM_BYTES);
    let blocks_by_smem = i64::from(smem_per_sm) / smem_total_per_block.max(1);
    let blocks_by_regs = if num_regs > 0 {
        i64::from(regs_per_sm) / (i64::from(num_regs) * i64::from(BLOCK_SIZE))
    } else {
        i64::MAX
    };
    let blocks_by_threads = i64::from(max_threads_per_sm) / i64::from(BLOCK_SIZE);
    let blocks_by_hw_cap = i64::from(max_blocks_per_sm_hw);
    let hand_blocks_per_sm = [blocks_by_smem, blocks_by_regs, blocks_by_threads, blocks_by_hw_cap]
        .into_iter()
        .min()
        .unwrap_or(0);

    println!();
    println!("--- Hand-computed occupancy bounds (before driver granularity rules) ---");
    println!(
        "blocks/SM limited by shared mem      : {blocks_by_smem}   ({smem_per_sm} / {smem_total_per_block})"
    );
    println!(
        "blocks/SM limited by registers       : {blocks_by_regs}   ({regs_per_sm} / ({num_regs} * {BLOCK_SIZE}))"
    );
    println!("blocks/SM limited by max threads     : {blocks_by_threads}");
    println!("blocks/SM limited by hw block cap    : {blocks_by_hw_cap}");
    println!("=> hand-computed blocks/SM (min)     : {hand_blocks_per_sm}");

    // ---- Driver's own occupancy calculator (authoritative) ----
    let driver_blocks_per_sm = func.occupancy_max_active_blocks_per_multiprocessor(
        BLOCK_SIZE,
        P1_SMEM_BYTES as usize,
        None,
    )?;
    let warps_per_block = i64::from(BLOCK_SIZE) / i64::from(warp_size);
    let max_warps_per_sm = i64::from(max_threads_per_sm) / i64::from(warp_size);
    let active_warps = i64::from(driver_blocks_per_sm) * warps_per_block;
    let occupancy_pct = 100.0 * active_warps as f64 / max_warps_per_sm as f64;

    println!();
    println!("--- Driver occupancy calculator (cuOccupancyMaxActiveBlocksPerMultiprocessor) ---");
    println!("blocks/SM (driver, authoritative)    : {driver_blocks_per_sm}");
    println!("warps/block                          : {warps_per_block}");
    println!("active warps/SM                      : {active_warps} / {max_warps_per_sm} max");
    println!("theoretical occupancy                : {occupancy_pct:.1}%");

    // ---- Grid-fill numbers at production B values (grid = B*H blocks) ----
    println!();
    println!("--- Grid fill, one block per (b,h), H={H} ---");
    for &b in &[1usize, 2, 4, 8, 16] {
        let n_blocks = b * H;
        let sms_touched = n_blocks.min(sm_count as usize);
        let avg_blocks_per_sm =
            (n_blocks as f64 / sm_count as f64).min(driver_blocks_per_sm as f64);
        let occ_at_b = 100.0 * (avg_blocks_per_sm * warps_per_block as f64) / max_warps_per_sm as f64;
        println!(
            "B={b:<2} -> grid={n_blocks:<3} blocks | SMs touched={sms_touched:>2}/{sm_count} | avg blocks/SM={avg_blocks_per_sm:.2} (cap {driver_blocks_per_sm}) | occupancy~={occ_at_b:.1}%"
        );
    }

    Ok(())
}
