//! Host-side wrapper for the fused Mamba2 chunked SSD scan kernel
//! (`pm_cuda_kernel::ssd_scan_chunked`).
//!
//! Two entry points:
//! - [`ssd_scan_chunked`] — convenience: takes `&[f32]` slices, copies
//!   them to the GPU, runs the kernel, copies back. Used by parity
//!   tests and the eventual Candle `custom_op` bridge in K.1A.
//! - [`ssd_scan_chunked_with_context`] — production: caller supplies
//!   the loaded module + `&CudaStream` and on-device slices, no
//!   round-trips. Used by Group L once we wire pm-candle dispatch.

use std::sync::Arc;

use cudarc::driver::{
    sys::CUfunction_attribute_enum, CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

use crate::module::load_kernel_function;
use crate::CudaError;

/// `n_dim` (= d_state) upper bound the PTX kernel was compiled for.
pub const MAX_N_DIM: usize = 128;

/// `block_len` (= Q) upper bound the PTX kernel was compiled for.
pub const MAX_BLOCK_LEN: usize = 128;

/// Kernel name for the original (fallback) SSD scan.
const KERNEL_NAME: &str = "ssd_scan_chunked";

/// Kernel name for the J.3.P2 cooperative shared-load SSD scan.
/// The PTX symbol retains the `_p1` suffix for binary compatibility with
/// the dispatch guard below; the kernel body has been replaced by the P2
/// implementation.
const KERNEL_P1_NAME: &str = "ssd_scan_chunked_p1";

// P2 fixed shape constants (must match the kernel's compile-time values).
const P1_N_DIM: usize = 128;
const P1_BLOCK_LEN: usize = 64;
const P1_P_DIM: usize = 64;
const P1_N_PAD: usize = P1_N_DIM + 1; // = 129, bank-conflict padding for s_hstate

// Dynamic shared memory required by the P2 kernel (100 352 bytes ≈ 98 KB):
//
//   Layout (N_PAD = N+1 = 129 for s_hstate bank-conflict elimination):
//     s_a_cum[Q]          = Q_FIXED floats                    =      256 B
//     s_b[Q*N]            = Q_FIXED * N_DIM floats            =   32 768 B
//     s_c[Q*N]            = Q_FIXED * N_DIM floats            =   32 768 B
//     s_hstate[P*N_PAD]   = P_DIM * (N_DIM + 1) floats       =   33 024 B
//     s_decay_end[Q]      = Q_FIXED floats                    =      256 B
//     s_bc_row_A[Q]       = Q_FIXED floats                    =      256 B
//     s_bc_row_B[Q]       = Q_FIXED floats                    =      256 B
//     s_decay_row_A[Q]    = Q_FIXED floats                    =      256 B
//     s_decay_row_B[Q]    = Q_FIXED floats                    =      256 B
//     s_decay_start[Q]    = Q_FIXED floats                    =      256 B
//
//   HSTATE_END = Q + 2*Q*N + P*N_PAD = 64 + 16384 + 8256 = 24704 floats
//   Total floats = HSTATE_END + 6*Q = 24704 + 384 = 25088
//   Total bytes  = 25088 * 4 = 100 352 B
//
// RTX 5070 (sm_120) default sharedMemPerBlock = 49 152 B (48 KB),
// sharedMemPerBlockOptin = 101 376 B (≈ 99 KB).
// 100 352 > 49 152 → cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) MUST
// be called.  100 352 < 101 376 → fits in the opt-in limit.
const P1_SMEM_BYTES: u32 =
    ((P1_BLOCK_LEN + 2 * P1_BLOCK_LEN * P1_N_DIM + P1_P_DIM * P1_N_PAD + 6 * P1_BLOCK_LEN) * 4)
        as u32; // = (64 + 16384 + 8256 + 384) * 4 = 25088 * 4 = 100 352

/// Convenience wrapper around the PTX kernel that round-trips through
/// the GPU. Allocates a fresh `CudaContext` if `ctx` is `None`.
///
/// # Errors
/// Returns `CudaError` on driver / module-load / launch failure.
///
/// # Panics
/// Panics if shape constraints are violated (`n_dim > MAX_N_DIM`,
/// `block_len > MAX_BLOCK_LEN`, `t_len % block_len != 0`, slice
/// lengths inconsistent with declared shape).
#[allow(clippy::too_many_arguments)]
pub fn ssd_scan_chunked(
    x: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    batch: usize,
    t_len: usize,
    n_heads: usize,
    p_dim: usize,
    n_dim: usize,
    block_len: usize,
) -> Result<Vec<f32>, CudaError> {
    assert!(
        n_dim <= MAX_N_DIM,
        "n_dim={n_dim} exceeds kernel MAX_N_DIM={MAX_N_DIM}"
    );
    assert!(
        block_len <= MAX_BLOCK_LEN,
        "block_len={block_len} exceeds kernel MAX_BLOCK_LEN={MAX_BLOCK_LEN}"
    );
    assert!(
        block_len > 0 && t_len.is_multiple_of(block_len),
        "t_len={t_len} must be a positive multiple of block_len={block_len}"
    );
    assert_eq!(x.len(), batch * t_len * n_heads * p_dim, "x size mismatch");
    assert_eq!(a.len(), batch * t_len * n_heads, "a size mismatch");
    assert_eq!(b.len(), batch * t_len * n_heads * n_dim, "b size mismatch");
    assert_eq!(c.len(), batch * t_len * n_heads * n_dim, "c size mismatch");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let x_dev = stream.clone_htod(x)?;
    let a_dev = stream.clone_htod(a)?;
    let b_dev = stream.clone_htod(b)?;
    let c_dev = stream.clone_htod(c)?;
    let mut y_dev = stream.alloc_zeros::<f32>(batch * t_len * n_heads * p_dim)?;

    ssd_scan_chunked_with_context(
        &ctx, &stream, &x_dev, &a_dev, &b_dev, &c_dev, &mut y_dev, batch, t_len, n_heads, p_dim,
        n_dim, block_len,
    )?;

    Ok(stream.clone_dtoh(&y_dev)?)
}

/// In-place variant: takes already-allocated device tensors. The
/// caller owns the output buffer.
///
/// Dispatches to the J.3.P2 cooperative-shared-load kernel
/// (`ssd_scan_chunked_p1` symbol, replaced body) when the shape matches
/// the production configuration (`n_dim = 128`, `block_len = 64`,
/// `p_dim = 64`).
/// For all other shapes (e.g. the parity-test small cases) falls back
/// to the original `ssd_scan_chunked` kernel which is correct for any
/// `n_dim ≤ 128`, `block_len ≤ 128`.
///
/// # Errors
///
/// Returns `CudaError::Shape` when `block_len == 0`, when `t_len == 0`,
/// or when `t_len` is not a multiple of `block_len` (both kernels rely
/// on `t_len / block_len` integer division).
///
/// # Preconditions (caller-side)
///
/// - `n_dim ≤ MAX_N_DIM` (128), `block_len ≤ MAX_BLOCK_LEN` (128).
/// - On-device slices `x` / `b` / `c` / `y` were allocated by the
///   caller at lengths matching `(batch, t_len, n_heads, p_dim, n_dim)`.
/// - `a` is `(batch, t_len, n_heads)` elements.
#[allow(clippy::too_many_arguments)]
pub fn ssd_scan_chunked_with_context(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    batch: usize,
    t_len: usize,
    n_heads: usize,
    p_dim: usize,
    n_dim: usize,
    block_len: usize,
) -> Result<(), CudaError> {
    // H-1 (review of 6a1f89f): both kernels compute n_chunks = t_len / Q via
    // integer division.  Without this guard, callers of `with_context` (the
    // production path; pm-candle dispatch will land here at Group L) would
    // silently drop the last `t_len % block_len` tokens — wrong outputs, no
    // error.  The convenience wrapper `ssd_scan_chunked` has its own assert,
    // but that does not protect direct `with_context` callers.
    if block_len == 0 {
        return Err(CudaError::Shape("block_len must be positive".into()));
    }
    if t_len == 0 || !t_len.is_multiple_of(block_len) {
        return Err(CudaError::Shape(
            format!("t_len ({t_len}) must be a positive multiple of block_len ({block_len})")
                .into(),
        ));
    }

    let batch_u =
        u32::try_from(batch).map_err(|_| CudaError::Shape("batch overflows u32".into()))?;
    let t_len_u =
        u32::try_from(t_len).map_err(|_| CudaError::Shape("t_len overflows u32".into()))?;
    let n_heads_u =
        u32::try_from(n_heads).map_err(|_| CudaError::Shape("n_heads overflows u32".into()))?;
    let p_dim_u =
        u32::try_from(p_dim).map_err(|_| CudaError::Shape("p_dim overflows u32".into()))?;
    let n_dim_u =
        u32::try_from(n_dim).map_err(|_| CudaError::Shape("n_dim overflows u32".into()))?;
    let block_len_u =
        u32::try_from(block_len).map_err(|_| CudaError::Shape("block_len overflows u32".into()))?;

    // Dispatch: use the P1 shared-memory kernel for the production shape,
    // fall back to the original register-spill kernel for other shapes
    // (e.g. ssd_parity_small / ssd_parity_pytorch_fixture tests).
    let use_p1 = n_dim == P1_N_DIM && block_len == P1_BLOCK_LEN && p_dim == P1_P_DIM;

    if use_p1 {
        let func = load_kernel_function(ctx, KERNEL_P1_NAME)?;

        // P2 requires 100 352 bytes of dynamic shared memory (100 096 in the
        // previous N_PAD version), which exceeds the CUDA default limit of
        // 49 152 B per block.  The opt-in limit on RTX 5070 (sm_120) is
        // 101 376 B, so we set CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        // before the first launch.  Re-setting on every call is cheap — it is
        // a driver-side property on the CUfunction handle, not per-launch.
        //
        // SAFETY: `func` is a valid loaded kernel function; P1_SMEM_BYTES =
        // 100 352 B is within Blackwell's opt-in limit (101 376 B).
        // `set_attribute` is an FFI call to `cuFuncSetAttribute` which is safe
        // to call from the host at any time after module load.
        func.set_attribute(
            CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            P1_SMEM_BYTES as i32,
        )
        .map_err(|e| {
            CudaError::Internal(format!(
                "cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES={P1_SMEM_BYTES} failed \
                 (hardware opt-in shared-mem limit likely exceeded; \
                  Blackwell sm_120 maximum is 101376 B): {e}"
            ))
        })?;

        // P2: one block per (b, h), p_dim threads per block.
        let n_blocks = u32::try_from(batch * n_heads)
            .map_err(|_| CudaError::Shape("batch * n_heads overflows u32".into()))?;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (P1_P_DIM as u32, 1, 1),
            shared_mem_bytes: P1_SMEM_BYTES,
        };
        let mut launch = stream.launch_builder(&func);
        launch.arg(x);
        launch.arg(a);
        launch.arg(b);
        launch.arg(c);
        launch.arg(y);
        launch.arg(&batch_u);
        launch.arg(&t_len_u);
        launch.arg(&n_heads_u);
        launch.arg(&p_dim_u);
        launch.arg(&n_dim_u);
        launch.arg(&block_len_u);

        // SAFETY: P2 kernel contract —
        //   - p_dim == 64, n_dim == 128, block_len == 64 guaranteed by
        //     `use_p1` guard above (matches kernel's compile-time const P/N/Q),
        //   - t_len is a positive multiple of block_len, guarded at function top,
        //   - MAX_DYNAMIC_SHARED_SIZE_BYTES set to P1_SMEM_BYTES (100 352 B) above;
        //     driver allows up to sharedMemPerBlockOptin (101 376 B on RTX 5070
        //     sm_120); 100 352 < 101 376,
        //   - on-device slices were sized by the caller per the declared
        //     (batch, t_len, n_heads, p_dim, n_dim) shape — same contract as
        //     the legacy fallback path below.
        unsafe { launch.launch(cfg) }?;
    } else {
        let func = load_kernel_function(ctx, KERNEL_NAME)?;
        // Fallback: one thread per (b, h, p).  Register pressure is
        // high (≈256 float locals) so cap to 64 threads/block to avoid
        // CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES on Blackwell.
        let n_threads = u32::try_from(batch * n_heads * p_dim)
            .map_err(|_| CudaError::Shape("batch * n_heads * p_dim overflows u32".into()))?;
        const THREADS_PER_BLOCK: u32 = 64;
        let cfg = LaunchConfig {
            grid_dim: (n_threads.div_ceil(THREADS_PER_BLOCK), 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&func);
        launch.arg(x);
        launch.arg(a);
        launch.arg(b);
        launch.arg(c);
        launch.arg(y);
        launch.arg(&batch_u);
        launch.arg(&t_len_u);
        launch.arg(&n_heads_u);
        launch.arg(&p_dim_u);
        launch.arg(&n_dim_u);
        launch.arg(&block_len_u);

        // SAFETY: shape arguments validated in the convenience wrapper;
        // on-device buffers came from the caller with matching sizes.
        unsafe { launch.launch(cfg) }?;
    }
    Ok(())
}
