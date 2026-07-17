//! Host-side launchers for the elementwise PTX kernels in
//! `pm-cuda-kernel`, and cuBLAS matrix multiply helpers.
//!
//! B4.1: add/sub/mul/neg elementwise ops.
//! B4.2a: `matmul_f32` via cuBLAS `gemm` / `gemm_strided_batched`.

use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::CudaError;

const ELEMENT_THREADS_PER_BLOCK: u32 = 256;

fn launch_cfg(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(ELEMENT_THREADS_PER_BLOCK), 1, 1),
        block_dim: (ELEMENT_THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Convert a `usize` element count to `u32` for the kernel's grid
/// arithmetic, returning a recoverable `CudaError::Shape` instead of
/// panicking on overflow (the launcher is called from `Ops`-impl
/// methods that already return `Result`).
fn u32_len(n: usize, op: &'static str) -> Result<u32, CudaError> {
    u32::try_from(n)
        .map_err(|_| CudaError::Shape(format!("{op}: tensor length {n} exceeds u32::MAX")))
}

/// Shape-mismatch check used by every binary launcher (lhs/rhs/out
/// must all have the same length). Returns `Err(Shape(..))` rather
/// than panicking so callers see a recoverable error.
fn check_binary_lengths(
    n: usize,
    rhs: usize,
    out: usize,
    op: &'static str,
) -> Result<(), CudaError> {
    if n != rhs {
        return Err(CudaError::Shape(format!(
            "{op}: lhs/rhs length mismatch ({n} vs {rhs})"
        )));
    }
    if n != out {
        return Err(CudaError::Shape(format!(
            "{op}: out length mismatch ({n} vs {out})"
        )));
    }
    Ok(())
}

/// Shape-mismatch check for unary/scalar launchers (input length must
/// match output length).
fn check_unary_lengths(n: usize, out: usize, op: &'static str) -> Result<(), CudaError> {
    if n != out {
        return Err(CudaError::Shape(format!(
            "{op}: out length mismatch ({n} vs {out})"
        )));
    }
    Ok(())
}

/// Last-dim extractor that returns `Err(Shape)` on empty shape — used
/// by row-major reductions (cumsum/rmsnorm/log_softmax) so callers
/// don't have to `.unwrap()` after an empty check.
fn last_dim_or_err(shape: &[usize], op: &'static str) -> Result<usize, CudaError> {
    shape
        .last()
        .copied()
        .ok_or_else(|| CudaError::Shape(format!("{op}: empty shape")))
}

/// `c[i] = a[i] + b[i]` for i in 0..n.
pub fn add_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    let n = a.len();
    assert_eq!(n, b.len(), "add_f32: lhs/rhs length mismatch");
    assert_eq!(n, c.len(), "add_f32: out length mismatch");
    let func = crate::module::load_kernel_function(ctx, "add_f32")?;
    let n_u32 = u32::try_from(n).expect("elementwise length overflow");
    let mut launch = stream.launch_builder(&func);
    launch.arg(a);
    launch.arg(b);
    launch.arg(c);
    launch.arg(&n_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

/// `c[i] = a[i] - b[i]`.
pub fn sub_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    let n = a.len();
    assert_eq!(n, b.len(), "sub_f32: lhs/rhs length mismatch");
    assert_eq!(n, c.len(), "sub_f32: out length mismatch");
    let func = crate::module::load_kernel_function(ctx, "sub_f32")?;
    let n_u32 = u32::try_from(n).expect("elementwise length overflow");
    let mut launch = stream.launch_builder(&func);
    launch.arg(a);
    launch.arg(b);
    launch.arg(c);
    launch.arg(&n_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

/// `c[i] = a[i] * b[i]`.
pub fn mul_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    let n = a.len();
    assert_eq!(n, b.len(), "mul_f32: lhs/rhs length mismatch");
    assert_eq!(n, c.len(), "mul_f32: out length mismatch");
    // The PTX symbol is `mul` (the I.3 smoke kernel double-duties).
    let func = crate::module::load_kernel_function(ctx, "mul")?;
    let n_u32 = u32::try_from(n).expect("elementwise length overflow");
    let mut launch = stream.launch_builder(&func);
    launch.arg(a);
    launch.arg(b);
    launch.arg(c);
    launch.arg(&n_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

/// `c[i] = -a[i]`.
pub fn neg_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    let n = a.len();
    assert_eq!(n, c.len(), "neg_f32: out length mismatch");
    let func = crate::module::load_kernel_function(ctx, "neg_f32")?;
    let n_u32 = u32::try_from(n).expect("elementwise length overflow");
    let mut launch = stream.launch_builder(&func);
    launch.arg(a);
    launch.arg(c);
    launch.arg(&n_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

// ---- B4.2b unary / scalar elementwise launchers --------------------------

fn launch_unary(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &'static str,
    a: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    let n = a.len();
    check_unary_lengths(n, c.len(), kernel)?;
    let func = crate::module::load_kernel_function(ctx, kernel)?;
    let n_u32 = u32_len(n, kernel)?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(a);
    launch.arg(c);
    launch.arg(&n_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

fn launch_binary(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &'static str,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    let n = a.len();
    check_binary_lengths(n, b.len(), c.len(), kernel)?;
    let func = crate::module::load_kernel_function(ctx, kernel)?;
    let n_u32 = u32_len(n, kernel)?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(a);
    launch.arg(b);
    launch.arg(c);
    launch.arg(&n_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

fn launch_unary_scalar(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kernel: &'static str,
    a: &CudaSlice<f32>,
    scalar: f32,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    let n = a.len();
    check_unary_lengths(n, c.len(), kernel)?;
    let func = crate::module::load_kernel_function(ctx, kernel)?;
    let n_u32 = u32_len(n, kernel)?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(a);
    launch.arg(&scalar);
    launch.arg(c);
    launch.arg(&n_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

/// `c[i] = exp(a[i])`.
pub fn exp_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    launch_unary(ctx, stream, "exp_f32", a, c)
}

/// `c[i] = sqrt(a[i])`.
pub fn sqrt_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    launch_unary(ctx, stream, "sqrt_f32", a, c)
}

/// `c[i] = a[i] / b[i]`.
pub fn div_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    launch_binary(ctx, stream, "div_f32", a, b, c)
}

/// `c[i] = a[i] * scalar`.
pub fn mul_scalar_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    scalar: f32,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    launch_unary_scalar(ctx, stream, "mul_scalar_f32", a, scalar, c)
}

/// `c[i] = a[i] + scalar`.
pub fn add_scalar_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    scalar: f32,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    launch_unary_scalar(ctx, stream, "add_scalar_f32", a, scalar, c)
}

/// `c[i] = sigmoid(a[i])`.
pub fn sigmoid_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    launch_unary(ctx, stream, "sigmoid_f32", a, c)
}

/// `c[i] = silu(a[i]) = a[i] * sigmoid(a[i])`.
pub fn silu_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    launch_unary(ctx, stream, "silu_f32", a, c)
}

/// `c[i] = softplus(a[i]) = log(1 + exp(a[i]))` (numerically stable).
pub fn softplus_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    c: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    launch_unary(ctx, stream, "softplus_f32", a, c)
}

// ---- B4.2c shape copy launchers -------------------------------------------

/// Transpose a 2D tensor: `(rows, cols)` → `(cols, rows)`.
pub fn transpose_2d_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    rows: usize,
    cols: usize,
) -> Result<(), CudaError> {
    let n = rows * cols;
    check_unary_lengths(n, dst.len(), "transpose_2d_f32")?;
    let func = crate::module::load_kernel_function(ctx, "transpose_2d_f32")?;
    let n_u32 = u32_len(n, "transpose_2d_f32")?;
    let rows_u32 = u32_len(rows, "transpose_2d_f32")?;
    let cols_u32 = u32_len(cols, "transpose_2d_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(src);
    launch.arg(dst);
    launch.arg(&rows_u32);
    launch.arg(&cols_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

/// ND transpose using pre-computed strides arrays.
///
/// `in_strides` and `out_strides` are host-side `Vec<u32>` computed by
/// the caller. They are uploaded to device for the kernel.
pub fn transpose_nd_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    in_strides: &[u32],
    out_strides: &[u32],
) -> Result<(), CudaError> {
    let n = src.len();
    check_unary_lengths(n, dst.len(), "transpose_nd_f32")?;
    if in_strides.len() != out_strides.len() {
        return Err(CudaError::Shape(format!(
            "transpose_nd_f32: in_strides.len()={} != out_strides.len()={}",
            in_strides.len(),
            out_strides.len()
        )));
    }
    let rank = in_strides.len();
    if rank > 8 {
        return Err(CudaError::Shape(format!(
            "transpose_nd_f32: rank {rank} > 8 (max supported)"
        )));
    }
    let func = crate::module::load_kernel_function(ctx, "transpose_nd_f32")?;
    let n_u32 = u32_len(n, "transpose_nd_f32")?;
    let rank_u32 = rank as u32;
    let in_strides_dev: CudaSlice<u32> = stream.clone_htod(in_strides)?;
    let out_strides_dev: CudaSlice<u32> = stream.clone_htod(out_strides)?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(src);
    launch.arg(dst);
    launch.arg(&in_strides_dev);
    launch.arg(&out_strides_dev);
    launch.arg(&rank_u32);
    launch.arg(&n_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

/// Narrow copy: copy `len` elements from `start` along `axis`.
///
/// `src_shape` is the full shape of the source tensor.
/// `axis`, `start`, `len` specify the narrow operation.
pub fn narrow_copy_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_shape: &[usize],
    axis: usize,
    start: usize,
    len: usize,
) -> Result<(), CudaError> {
    // outer_count = product of dims before axis
    let outer_count: usize = src_shape[..axis].iter().product();
    let axis_len_src = src_shape[axis];
    // inner_len = product of dims after axis
    let inner_len: usize = src_shape[axis + 1..].iter().product();
    let n = outer_count * len * inner_len;
    if dst.len() != n {
        return Err(CudaError::Shape(format!(
            "narrow_copy_f32: dst len {} != expected {n}",
            dst.len()
        )));
    }
    let func = crate::module::load_kernel_function(ctx, "narrow_copy_f32")?;
    let n_u32 = u32_len(n, "narrow_copy_f32")?;
    let outer_u32 = u32_len(outer_count, "narrow_copy_f32")?;
    let axis_src_u32 = u32_len(axis_len_src, "narrow_copy_f32")?;
    let axis_dst_u32 = u32_len(len, "narrow_copy_f32")?;
    let offset_u32 = u32_len(start, "narrow_copy_f32")?;
    let inner_u32 = u32_len(inner_len, "narrow_copy_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(src);
    launch.arg(dst);
    launch.arg(&outer_u32);
    launch.arg(&axis_src_u32);
    launch.arg(&axis_dst_u32);
    launch.arg(&offset_u32);
    launch.arg(&inner_u32);
    launch.arg(&n_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

/// `narrow`'s backward (VJP): scatter `grad_out` (shape = `orig_shape` with
/// `dim` replaced by `len`) into a freshly-written `orig_shape` tensor,
/// zero outside the `[start, start+len)` window along `dim`. One kernel
/// launch, one thread per **output** (`orig_shape`-sized) element — see
/// `narrow_backward_f32` in `pm-cuda-kernel` for the inverse-of-`narrow`
/// index mapping. `dst` does not need to be pre-zeroed: every element is
/// written exactly once (either gathered from `grad_out` or `0.0`).
pub fn narrow_backward_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    grad_out: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    orig_shape: &[usize],
    dim: usize,
    start: usize,
    len: usize,
) -> Result<(), CudaError> {
    let outer_count: usize = orig_shape[..dim].iter().product();
    let axis_len_orig = orig_shape[dim];
    let inner_len: usize = orig_shape[dim + 1..].iter().product();
    let n = outer_count * axis_len_orig * inner_len;
    if dst.len() != n {
        return Err(CudaError::Shape(format!(
            "narrow_backward_f32: dst len {} != expected {n}",
            dst.len()
        )));
    }
    let expected_grad_len = outer_count * len * inner_len;
    if grad_out.len() != expected_grad_len {
        return Err(CudaError::Shape(format!(
            "narrow_backward_f32: grad_out len {} != expected {expected_grad_len}",
            grad_out.len()
        )));
    }
    let func = crate::module::load_kernel_function(ctx, "narrow_backward_f32")?;
    let n_u32 = u32_len(n, "narrow_backward_f32")?;
    let axis_orig_u32 = u32_len(axis_len_orig, "narrow_backward_f32")?;
    let axis_grad_u32 = u32_len(len, "narrow_backward_f32")?;
    let start_u32 = u32_len(start, "narrow_backward_f32")?;
    let inner_u32 = u32_len(inner_len, "narrow_backward_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(grad_out);
    launch.arg(dst);
    launch.arg(&axis_orig_u32);
    launch.arg(&axis_grad_u32);
    launch.arg(&start_u32);
    launch.arg(&inner_u32);
    launch.arg(&n_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

/// Row-major broadcast strides for one operand: the natural row-major
/// stride where `shape[i] != 1`, else `0` (a broadcast dim always reads
/// index 0 along that axis). Shared by `broadcast_copy_f32` and
/// `broadcast_mul_f32` so the two binary-broadcast launchers can't drift.
fn broadcast_strides(shape: &[usize], rank: usize) -> Vec<u32> {
    let mut nat = vec![1u32; rank];
    for i in (0..rank - 1).rev() {
        nat[i] = nat[i + 1] * shape[i + 1] as u32;
    }
    let mut s = vec![0u32; rank];
    for i in 0..rank {
        s[i] = if shape[i] == 1 { 0 } else { nat[i] };
    }
    s
}

/// Broadcast copy: expand `src` with shape `src_shape` to `dst_shape`.
///
/// Size-1 dimensions in `src_shape` are broadcast; all other dimensions
/// must match. Host computes the stride arrays.
pub fn broadcast_copy_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_shape: &[usize],
    dst_shape: &[usize],
) -> Result<(), CudaError> {
    let rank = dst_shape.len();
    if src_shape.len() != rank {
        return Err(CudaError::Shape(format!(
            "broadcast_copy_f32: rank mismatch src={} dst={rank}",
            src_shape.len()
        )));
    }
    if rank > 8 {
        return Err(CudaError::Shape(format!(
            "broadcast_copy_f32: rank {rank} > 8"
        )));
    }
    let n: usize = dst_shape.iter().product();
    if dst.len() != n {
        return Err(CudaError::Shape(format!(
            "broadcast_copy_f32: dst len {} != {n}",
            dst.len()
        )));
    }
    // Compute out_strides for dst (row-major).
    let mut out_strides = vec![1u32; rank];
    for i in (0..rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * dst_shape[i + 1] as u32;
    }
    let src_strides = broadcast_strides(src_shape, rank);
    let func = crate::module::load_kernel_function(ctx, "broadcast_copy_f32")?;
    let n_u32 = u32_len(n, "broadcast_copy_f32")?;
    let rank_u32 = rank as u32;
    let src_strides_dev: CudaSlice<u32> = stream.clone_htod(&src_strides)?;
    let out_strides_dev: CudaSlice<u32> = stream.clone_htod(&out_strides)?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(src);
    launch.arg(dst);
    launch.arg(&src_strides_dev);
    launch.arg(&out_strides_dev);
    launch.arg(&rank_u32);
    launch.arg(&n_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

/// Broadcast-aware elementwise multiply: `dst = a * b` with NumPy-style
/// broadcasting, entirely on device (no host round trip). `a_shape` and
/// `b_shape` must already be rank-padded to `dst_shape`'s rank (leading
/// size-1 dims added by the caller — same convention `broadcast_copy_f32`
/// uses for its single operand).
///
/// Replaces `CudaBackend::broadcast_binary_op`'s host round trip
/// (`clone_dtoh` × 2 → host divmod loop → `clone_htod`) on `mul`'s
/// broadcast path (B'.3, `docs/perf-log.md`): measured ~1.3 ms/call in
/// training, with the host-side scalar divmod loop — not the PCIe
/// transfers — as the dominant cost.
pub fn broadcast_mul_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    a_shape: &[usize],
    b_shape: &[usize],
    dst_shape: &[usize],
) -> Result<(), CudaError> {
    let rank = dst_shape.len();
    if a_shape.len() != rank || b_shape.len() != rank {
        return Err(CudaError::Shape(format!(
            "broadcast_mul_f32: rank mismatch a={} b={} dst={rank}",
            a_shape.len(),
            b_shape.len()
        )));
    }
    if rank > 8 {
        return Err(CudaError::Shape(format!(
            "broadcast_mul_f32: rank {rank} > 8"
        )));
    }
    let n: usize = dst_shape.iter().product();
    if dst.len() != n {
        return Err(CudaError::Shape(format!(
            "broadcast_mul_f32: dst len {} != {n}",
            dst.len()
        )));
    }
    // Validate operand backing lengths too (the CudaTensor::new
    // debug_assert is a no-op in release; a mismatch here would silently
    // read OOB via the broadcast strides).
    let a_n: usize = a_shape.iter().product();
    let b_n: usize = b_shape.iter().product();
    if a.len() != a_n {
        return Err(CudaError::Shape(format!(
            "broadcast_mul_f32: a len {} != a_shape product {a_n}",
            a.len()
        )));
    }
    if b.len() != b_n {
        return Err(CudaError::Shape(format!(
            "broadcast_mul_f32: b len {} != b_shape product {b_n}",
            b.len()
        )));
    }
    let mut out_strides = vec![1u32; rank];
    for i in (0..rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * dst_shape[i + 1] as u32;
    }
    let a_strides = broadcast_strides(a_shape, rank);
    let b_strides = broadcast_strides(b_shape, rank);

    let func = crate::module::load_kernel_function(ctx, "broadcast_mul_f32")?;
    let n_u32 = u32_len(n, "broadcast_mul_f32")?;
    let rank_u32 = rank as u32;
    let a_strides_dev: CudaSlice<u32> = stream.clone_htod(&a_strides)?;
    let b_strides_dev: CudaSlice<u32> = stream.clone_htod(&b_strides)?;
    let out_strides_dev: CudaSlice<u32> = stream.clone_htod(&out_strides)?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(a);
    launch.arg(b);
    launch.arg(dst);
    launch.arg(&a_strides_dev);
    launch.arg(&b_strides_dev);
    launch.arg(&out_strides_dev);
    launch.arg(&rank_u32);
    launch.arg(&n_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

// ---- B4.2c reduction launchers --------------------------------------------

/// Inclusive cumulative sum along the last dimension.
///
/// `shape` is the full shape of `src` (and `dst`).
pub fn cumsum_lastdim_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    shape: &[usize],
) -> Result<(), CudaError> {
    let n = src.len();
    check_unary_lengths(n, dst.len(), "cumsum_lastdim_f32")?;
    let last_dim = last_dim_or_err(shape, "cumsum_lastdim_f32")?;
    let n_rows = n / last_dim;
    let func = crate::module::load_kernel_function(ctx, "cumsum_lastdim_f32")?;
    let n_rows_u32 = u32_len(n_rows, "cumsum_lastdim_f32")?;
    let last_dim_u32 = u32_len(last_dim, "cumsum_lastdim_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(src);
    launch.arg(dst);
    launch.arg(&n_rows_u32);
    launch.arg(&last_dim_u32);
    unsafe { launch.launch(launch_cfg(n_rows_u32)) }?;
    Ok(())
}

/// RMSNorm: `out[row, d] = x[row, d] * weight[d] * rsqrt(mean(x[row]^2) + eps)`.
///
/// `shape` is the shape of `x`. `weight` has shape `(shape.last(),)`.
pub fn rmsnorm_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    shape: &[usize],
    eps: f32,
) -> Result<(), CudaError> {
    let n = x.len();
    check_unary_lengths(n, out.len(), "rmsnorm_f32")?;
    let d_model = last_dim_or_err(shape, "rmsnorm_f32")?;
    if weight.len() != d_model {
        return Err(CudaError::Shape(format!(
            "rmsnorm_f32: weight.len()={} != d_model={d_model}",
            weight.len()
        )));
    }
    let n_rows = n / d_model;
    let func = crate::module::load_kernel_function(ctx, "rmsnorm_f32")?;
    let n_rows_u32 = u32_len(n_rows, "rmsnorm_f32")?;
    let d_model_u32 = u32_len(d_model, "rmsnorm_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(x);
    launch.arg(weight);
    launch.arg(out);
    launch.arg(&n_rows_u32);
    launch.arg(&d_model_u32);
    launch.arg(&eps);
    unsafe { launch.launch(launch_cfg(n_rows_u32)) }?;
    Ok(())
}

/// log_softmax along the last dimension.
///
/// `shape` is the full shape of `src`.
pub fn log_softmax_lastdim_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    shape: &[usize],
) -> Result<(), CudaError> {
    let n = src.len();
    check_unary_lengths(n, dst.len(), "log_softmax_lastdim_f32")?;
    let last_dim = last_dim_or_err(shape, "log_softmax_lastdim_f32")?;
    let n_rows = n / last_dim;
    let func = crate::module::load_kernel_function(ctx, "log_softmax_lastdim_f32")?;
    let n_rows_u32 = u32_len(n_rows, "log_softmax_lastdim_f32")?;
    let last_dim_u32 = u32_len(last_dim, "log_softmax_lastdim_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(src);
    launch.arg(dst);
    launch.arg(&n_rows_u32);
    launch.arg(&last_dim_u32);
    unsafe { launch.launch(launch_cfg(n_rows_u32)) }?;
    Ok(())
}

// ---- B4.2d indexing / convolution launchers --------------------------------

/// Embedding lookup: `out[i * D + d] = table[indices[i] * D + d]`.
///
/// `table`: `(V, D)` f32 slice.
/// `indices`: flat i64 slice of length `indices_len`.
/// `out`: pre-allocated f32 slice of length `indices_len * embed_dim`.
pub fn embedding_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    table: &CudaSlice<f32>,
    indices: &CudaSlice<i64>,
    out: &mut CudaSlice<f32>,
    indices_len: usize,
    embed_dim: usize,
    vocab_size: usize,
) -> Result<(), CudaError> {
    let n = indices_len * embed_dim;
    if out.len() != n {
        return Err(CudaError::Shape(format!(
            "embedding_f32: out len {} != expected {n}",
            out.len()
        )));
    }
    let func = crate::module::load_kernel_function(ctx, "embedding_f32")?;
    let n_u32 = u32_len(n, "embedding_f32")?;
    let indices_len_u32 = u32_len(indices_len, "embedding_f32")?;
    let embed_dim_u32 = u32_len(embed_dim, "embedding_f32")?;
    let vocab_size_u32 = u32_len(vocab_size, "embedding_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(table);
    launch.arg(indices);
    launch.arg(out);
    launch.arg(&indices_len_u32);
    launch.arg(&embed_dim_u32);
    launch.arg(&vocab_size_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

/// Gather along the last dimension of `src` using i64 `indices`.
///
/// `src`:      `(..., last_dim_src)`
/// `indices`:  `(..., last_dim_idx)`
/// `out`:      `(..., last_dim_idx)` — pre-allocated.
pub fn gather_lastdim_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    indices: &CudaSlice<i64>,
    out: &mut CudaSlice<f32>,
    last_dim_src: usize,
    last_dim_idx: usize,
) -> Result<(), CudaError> {
    let n = out.len();
    if indices.len() != n {
        return Err(CudaError::Shape(format!(
            "gather_lastdim_f32: indices.len()={} != out.len()={n}",
            indices.len()
        )));
    }
    let func = crate::module::load_kernel_function(ctx, "gather_lastdim_f32")?;
    let n_u32 = u32_len(n, "gather_lastdim_f32")?;
    let last_src_u32 = u32_len(last_dim_src, "gather_lastdim_f32")?;
    let last_idx_u32 = u32_len(last_dim_idx, "gather_lastdim_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(src);
    launch.arg(indices);
    launch.arg(out);
    launch.arg(&n_u32);
    launch.arg(&last_src_u32);
    launch.arg(&last_idx_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

/// im2col for 1D convolution: `(B, C_in, T_in)` → `(B * T_out, C_in * K)`.
///
/// `T_out = (T_in + 2 * padding - k_size) / stride + 1`.
/// `out` must be pre-allocated with `B * T_out * C_in * K` elements.
#[allow(clippy::too_many_arguments)]
pub fn im2col_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    batch: usize,
    c_in: usize,
    t_in: usize,
    t_out: usize,
    k_size: usize,
    stride: usize,
    padding: usize,
) -> Result<(), CudaError> {
    let n = batch * t_out * c_in * k_size;
    if dst.len() != n {
        return Err(CudaError::Shape(format!(
            "im2col_f32: dst len {} != expected {n}",
            dst.len()
        )));
    }
    let func = crate::module::load_kernel_function(ctx, "im2col_f32")?;
    let n_u32 = u32_len(n, "im2col_f32")?;
    let batch_u32 = u32_len(batch, "im2col_f32")?;
    let c_in_u32 = u32_len(c_in, "im2col_f32")?;
    let t_in_u32 = u32_len(t_in, "im2col_f32")?;
    let t_out_u32 = u32_len(t_out, "im2col_f32")?;
    let k_u32 = u32_len(k_size, "im2col_f32")?;
    let stride_u32 = u32_len(stride, "im2col_f32")?;
    let padding_u32 = u32_len(padding, "im2col_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(src);
    launch.arg(dst);
    launch.arg(&batch_u32);
    launch.arg(&c_in_u32);
    launch.arg(&t_in_u32);
    launch.arg(&t_out_u32);
    launch.arg(&k_u32);
    launch.arg(&stride_u32);
    launch.arg(&padding_u32);
    launch.arg(&n_u32);
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

/// 1D grouped convolution using im2col + cuBLAS GEMM.
///
/// `x`:      `(B, C_in, T_in)` f32.
/// `weight`: `(C_out, C_in / groups, K)` f32.
/// `bias`:   optional `(C_out,)` f32.
/// `stride`, `padding`, `groups`: standard conv1d parameters.
/// Returns `(B, C_out, T_out)`.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    cublas: &Arc<CudaBlas>,
    x: &CudaSlice<f32>,
    x_shape: &[usize], // [B, C_in, T_in]
    weight: &CudaSlice<f32>,
    weight_shape: &[usize], // [C_out, C_in_per_group, K]
    bias: Option<&CudaSlice<f32>>,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Result<CudaSlice<f32>, CudaError> {
    if x_shape.len() != 3 {
        return Err(CudaError::Shape(
            "conv1d_f32: x must be rank-3 (B, C_in, T_in)".to_string(),
        ));
    }
    if weight_shape.len() != 3 {
        return Err(CudaError::Shape(
            "conv1d_f32: weight must be rank-3 (C_out, C_in/groups, K)".to_string(),
        ));
    }
    let batch = x_shape[0];
    let c_in = x_shape[1];
    let t_in = x_shape[2];
    let c_out = weight_shape[0];
    let c_in_per_group = weight_shape[1];
    let k_size = weight_shape[2];

    if c_in != c_in_per_group * groups {
        return Err(CudaError::Shape(format!(
            "conv1d_f32: C_in={c_in} != C_in_per_group={c_in_per_group} * groups={groups}"
        )));
    }
    if c_out % groups != 0 {
        return Err(CudaError::Shape(format!(
            "conv1d_f32: C_out={c_out} must be divisible by groups={groups}"
        )));
    }
    let c_out_per_group = c_out / groups;

    let t_out = (t_in + 2 * padding - k_size) / stride + 1;

    // ---- B4.4e-conv1d fast path: depthwise (groups == c_in, 1 filter/channel) ----
    //
    // Replace the im2col + 1024-iteration host GEMM loop with a single GPU
    // kernel launch.  Production shape [1,1024,512] w [1024,1,4] groups=1024
    // dropped from 26.62 ms → <1 ms with this path.
    //
    // Constraint: K ≤ 128 (the kernel uses threads 0..K-1 to load the weight
    // strip into shared memory).  For K > 128 fall through to the generic path.
    if groups == c_in && c_in_per_group == 1 && k_size <= 128 {
        let mut y = stream.alloc_zeros::<f32>(batch * c_out * t_out)?;
        conv1d_depthwise_fwd_gpu(
            ctx, stream, x, weight, bias, &mut y, batch, c_in, t_in, t_out, k_size, stride, padding,
        )?;
        return Ok(y);
    }

    // ---- Step 1: im2col for the full input  --------------------------------
    // Shape: (B * T_out, C_in * K)
    let im2col_rows = batch * t_out;
    let im2col_cols = c_in * k_size;
    let mut col_buf = stream.alloc_zeros::<f32>(im2col_rows * im2col_cols)?;
    im2col_f32(
        ctx,
        stream,
        x,
        &mut col_buf,
        batch,
        c_in,
        t_in,
        t_out,
        k_size,
        stride,
        padding,
    )?;

    // ---- Step 2: per-group GEMM + assemble output --------------------------
    // weight slice per group: (C_out/g, C_in/g * K)
    // col slice per group: (B * T_out, C_in/g * K)  → from col_buf with stride
    // We work group-by-group on host-copied data (B4.2 correctness tier).
    let col_host = stream.clone_dtoh(&col_buf)?;
    let weight_host = stream.clone_dtoh(weight)?;

    // out_host: (B, C_out, T_out) row-major
    let mut out_host = vec![0.0f32; batch * c_out * t_out];

    for g in 0..groups {
        let c_in_g_start = g * c_in_per_group;
        let c_in_g_k = c_in_per_group * k_size;
        let c_out_g_start = g * c_out_per_group;

        // Build col_g: (B * T_out, C_in_per_group * K) from col_buf columns.
        // col_buf row-major: (B * T_out, C_in * K).
        // Our group's columns are [g * c_in_per_group * K .. (g+1) * c_in_per_group * K).
        let col_offset = c_in_g_start * k_size;
        let mut col_g = vec![0.0f32; im2col_rows * c_in_g_k];
        for r in 0..im2col_rows {
            let src_base = r * im2col_cols + col_offset;
            let dst_base = r * c_in_g_k;
            col_g[dst_base..dst_base + c_in_g_k]
                .copy_from_slice(&col_host[src_base..src_base + c_in_g_k]);
        }

        // weight_g: (C_out_per_group, C_in_per_group * K)
        let w_offset = c_out_g_start * c_in_g_k;
        let w_g = &weight_host[w_offset..w_offset + c_out_per_group * c_in_g_k];

        // GEMM: out_g = weight_g @ col_g^T  →  (C_out/g, B * T_out)
        // Then reshape to (B, C_out/g, T_out) and scatter into out_host.
        //
        // Using cuBLAS: we need weight_g (C_out/g × CK) × col_g^T (CK × B*T_out)
        // = (C_out/g, B*T_out).
        // Row-major C = A @ B:  cuBLAS sees column-major so use (B^T @ A^T).
        let col_g_dev = stream.clone_htod(&col_g)?;
        let w_g_dev = stream.clone_htod(w_g)?;
        // out_g_dev: (C_out/g, B * T_out)
        let mut out_g_dev = stream.alloc_zeros::<f32>(c_out_per_group * im2col_rows)?;

        // We want: out_g (C_out/g, B*T_out) = weight_g (C_out/g, CK) @ col_g^T (CK, B*T_out)
        // i.e. A shape (C_out/g, CK), B shape (CK, B*T_out) — but col_g is (B*T_out, CK).
        // So B = col_g^T. In row-major terms: matmul A [M×K] × B [K×N] → C [M×N].
        // M = C_out/g, K = c_in_g_k, N = B*T_out.
        // Call matmul_f32 with a_shape=[M,K] and b_shape=[K,N].
        // col_g is (B*T_out, CK) = (N, K) so we need to transpose it → (K, N).
        // Easier: use the weight as A [M,K] and note that col_g is (N,K) so
        // pass it transposed as B [K,N]. But matmul_f32 doesn't support transA/transB.
        //
        // Alternative: cuBLAS directly with transB=T.
        // We use the identity C = A @ B^T where A=weight_g (M×K), B=col_g (N×K).
        // In cuBLAS column-major notation with the swap trick:
        //   C^T (N×M) = col_g (N×K) @ weight_g^T (K×M)
        //   → gemm(col_g, weight_g, OP_N, OP_T) with m=M, n=N, k=K.
        {
            use cudarc::cublas::sys::cublasOperation_t;
            use cudarc::cublas::{Gemm, GemmConfig};
            let m = c_out_per_group as i32;
            let n_dim = im2col_rows as i32;
            let k = c_in_g_k as i32;
            // cuBLAS column-major trick for C_row_major = A_row_major @ B_row_major^T:
            // out_g is (M, N), col_g is (N, K), weight_g is (M, K).
            // We want out_g = weight_g @ col_g^T.
            // cuBLAS(col-major) call: C = alpha * op(A) * op(B) + beta * C
            // Swap roles: C^T (N×M) = col_g (N×K, K-major) * weight_g^T (K×M, M-major).
            // gemm(transa=N, transb=T, m=N, n=M, k=K, lda=K, ldb=K, ldc=N, A=col_g, B=weight_g)
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n_dim, // N (B*T_out) — left of C^T in col-major
                n: m,     // M (C_out/g) — right of C^T
                k,
                alpha: 1.0f32,
                lda: k, // leading dim of col_g (row-major K cols → K)
                ldb: k, // leading dim of weight_g (row-major K cols → K)
                beta: 0.0f32,
                ldc: n_dim, // leading dim of out_g^T (row-major N cols → N)
            };
            // SAFETY: sizes and device pointers validated above.
            unsafe { cublas.gemm(cfg, &col_g_dev, &w_g_dev, &mut out_g_dev) }?;
        }

        // Bring out_g back to host: shape (C_out/g, B * T_out).
        let out_g_host = stream.clone_dtoh(&out_g_dev)?;

        // Scatter into out_host (B, C_out, T_out).
        // out_g layout: row c_out_g (0..C_out/g), col bt (0..B*T_out).
        // out_host[b, c_out_g_start + cg, t] = out_g[cg, b * T_out + t].
        for cg in 0..c_out_per_group {
            for b_idx in 0..batch {
                for t in 0..t_out {
                    let src_idx = cg * im2col_rows + b_idx * t_out + t;
                    let dst_idx = b_idx * (c_out * t_out) + (c_out_g_start + cg) * t_out + t;
                    out_host[dst_idx] = out_g_host[src_idx];
                }
            }
        }
    }

    // ---- Step 3: optional bias addition ------------------------------------
    // bias shape (C_out,); broadcast over (B, C_out, T_out).
    if let Some(bias_slice) = bias {
        if bias_slice.len() != c_out {
            return Err(CudaError::Shape(format!(
                "conv1d_f32: bias.len()={} != C_out={c_out}",
                bias_slice.len()
            )));
        }
        let bias_host = stream.clone_dtoh(bias_slice)?;
        for b_idx in 0..batch {
            for co in 0..c_out {
                let base = b_idx * c_out * t_out + co * t_out;
                for t in 0..t_out {
                    out_host[base + t] += bias_host[co];
                }
            }
        }
    }

    Ok(stream.clone_htod(&out_host)?)
}

// ---- B4.3b backward launchers ------------------------------------------------

/// Scatter-add for embedding backward.
///
/// `table_grad` must be pre-zeroed and have `(vocab_size, embed_dim)` elements.
/// Launches one thread per `(index, dim)` pair.
///
/// # Safety
/// `table_grad` is written via PTX `atom.global.add.f32`; concurrent threads
/// accumulating into the same row are serialised by the GPU.
#[allow(clippy::too_many_arguments)]
pub fn scatter_add_embedding_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    table_grad: &mut CudaSlice<f32>,
    indices: &CudaSlice<i64>,
    grad_out: &CudaSlice<f32>,
    indices_len: usize,
    embed_dim: usize,
    vocab_size: usize,
) -> Result<(), CudaError> {
    let n = indices_len * embed_dim;
    let func = crate::module::load_kernel_function(ctx, "scatter_add_embedding_f32")?;
    let n_u32 = u32_len(n, "scatter_add_embedding_f32")?;
    let indices_len_u32 = u32_len(indices_len, "scatter_add_embedding_f32")?;
    let embed_dim_u32 = u32_len(embed_dim, "scatter_add_embedding_f32")?;
    let vocab_size_u32 = u32_len(vocab_size, "scatter_add_embedding_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(table_grad);
    launch.arg(indices);
    launch.arg(grad_out);
    launch.arg(&indices_len_u32);
    launch.arg(&embed_dim_u32);
    launch.arg(&vocab_size_u32);
    // SAFETY: all device pointers are valid; atomic ops handle concurrency.
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

/// Scatter-add for gather backward (last dim only).
///
/// `x_grad` must be pre-zeroed.
pub fn scatter_add_gather_lastdim_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x_grad: &mut CudaSlice<f32>,
    indices: &CudaSlice<i64>,
    grad_out: &CudaSlice<f32>,
    last_dim_src: usize,
    last_dim_idx: usize,
) -> Result<(), CudaError> {
    let n = grad_out.len();
    let func = crate::module::load_kernel_function(ctx, "scatter_add_gather_lastdim_f32")?;
    let n_u32 = u32_len(n, "scatter_add_gather_lastdim_f32")?;
    let last_src_u32 = u32_len(last_dim_src, "scatter_add_gather_lastdim_f32")?;
    let last_idx_u32 = u32_len(last_dim_idx, "scatter_add_gather_lastdim_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(x_grad);
    launch.arg(indices);
    launch.arg(grad_out);
    launch.arg(&n_u32);
    launch.arg(&last_src_u32);
    launch.arg(&last_idx_u32);
    // SAFETY: atomic add; all pointers valid.
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

/// col2im backward for 1D convolution (GPU path).
///
/// Reserved for B4.5 GPU col2im path; current `conv1d_backward` uses a CPU
/// scatter instead.  Retained here to avoid re-deriving the launch logic.
///
/// `x_grad` must be pre-zeroed.
/// `n` = `batch * t_out * c_in * k_size` (total col buffer elements).
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn col2im_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x_grad: &mut CudaSlice<f32>,
    col_grad: &CudaSlice<f32>,
    batch: usize,
    c_in: usize,
    t_in: usize,
    t_out: usize,
    k_size: usize,
    stride: usize,
    padding: usize,
) -> Result<(), CudaError> {
    let n = batch * t_out * c_in * k_size;
    if col_grad.len() != n {
        return Err(CudaError::Shape(format!(
            "col2im_f32: col_grad len {} != expected {n}",
            col_grad.len()
        )));
    }
    let func = crate::module::load_kernel_function(ctx, "col2im_f32")?;
    let n_u32 = u32_len(n, "col2im_f32")?;
    let batch_u32 = u32_len(batch, "col2im_f32")?;
    let c_in_u32 = u32_len(c_in, "col2im_f32")?;
    let t_in_u32 = u32_len(t_in, "col2im_f32")?;
    let t_out_u32 = u32_len(t_out, "col2im_f32")?;
    let k_u32 = u32_len(k_size, "col2im_f32")?;
    let stride_u32 = u32_len(stride, "col2im_f32")?;
    let padding_u32 = u32_len(padding, "col2im_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(x_grad);
    launch.arg(col_grad);
    launch.arg(&batch_u32);
    launch.arg(&c_in_u32);
    launch.arg(&t_in_u32);
    launch.arg(&t_out_u32);
    launch.arg(&k_u32);
    launch.arg(&stride_u32);
    launch.arg(&padding_u32);
    launch.arg(&n_u32);
    // SAFETY: atomic add; all pointers valid.
    unsafe { launch.launch(launch_cfg(n_u32)) }?;
    Ok(())
}

// ---- B4.4e-conv1d  depthwise GPU kernel launchers --------------------------

/// Depthwise 1-D convolution forward (groups == channels, K ≤ 128).
///
/// B4.4e-conv1d — GPU replacement for the 1024-iteration host GEMM loop.
///
/// Writes into the pre-allocated (zeroed) `y` slice.
/// `bias` is optional; when `None`, a 1-element dummy allocation is used
/// and the kernel guards the read with `has_bias == 0`.
///
/// # Errors
/// Returns `CudaError::Shape` when `k_size > 128`.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_depthwise_fwd_gpu(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    bias: Option<&CudaSlice<f32>>,
    y: &mut CudaSlice<f32>,
    batch: usize,
    channels: usize,
    t_in: usize,
    t_out: usize,
    k_size: usize,
    stride: usize,
    padding: usize,
) -> Result<(), CudaError> {
    if k_size > 128 {
        return Err(CudaError::Shape(format!(
            "conv1d_depthwise_fwd_gpu: k_size={k_size} > 128 (kernel limit)"
        )));
    }
    const THREADS_PER_BLOCK: u32 = 128;
    let n_blocks = u32_len(batch * channels, "conv1d_depthwise_fwd_gpu")?;
    let batch_u = u32_len(batch, "conv1d_depthwise_fwd_gpu")?;
    let channels_u = u32_len(channels, "conv1d_depthwise_fwd_gpu")?;
    let t_in_u = u32_len(t_in, "conv1d_depthwise_fwd_gpu")?;
    let t_out_u = u32_len(t_out, "conv1d_depthwise_fwd_gpu")?;
    let k_size_u = u32_len(k_size, "conv1d_depthwise_fwd_gpu")?;
    let stride_u = u32_len(stride, "conv1d_depthwise_fwd_gpu")?;
    let padding_u = u32_len(padding, "conv1d_depthwise_fwd_gpu")?;

    // Allocate a 1-element dummy when no bias; the kernel checks has_bias before
    // reading bias_ptr, so the dummy is never actually dereferenced.
    let dummy_bias;
    let (bias_slice, has_bias_u): (&CudaSlice<f32>, u32) = match bias {
        Some(b) => (b, 1u32),
        None => {
            dummy_bias = stream.alloc_zeros::<f32>(1)?;
            (&dummy_bias, 0u32)
        }
    };

    let func = crate::module::load_kernel_function(ctx, "depthwise_conv1d_fwd_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        // Shared memory: K floats for the per-channel weight strip.
        shared_mem_bytes: k_size_u * 4,
    };
    let mut launch = stream.launch_builder(&func);
    launch.arg(x);
    launch.arg(w);
    launch.arg(bias_slice);
    launch.arg(y);
    launch.arg(&batch_u);
    launch.arg(&channels_u);
    launch.arg(&t_in_u);
    launch.arg(&t_out_u);
    launch.arg(&k_size_u);
    launch.arg(&stride_u);
    launch.arg(&padding_u);
    launch.arg(&has_bias_u);
    // SAFETY: all device slices are valid; shared_mem_bytes = k_size*4 covers
    // the K-element weight strip loaded by threads 0..k_size-1.
    // K ≤ 128 = THREADS_PER_BLOCK, validated above.
    unsafe { launch.launch(cfg) }?;
    Ok(())
}

/// Depthwise 1-D convolution backward w.r.t. input (groups == channels).
///
/// B4.4e-conv1d. Writes into the pre-allocated (zeroed) `grad_x` slice.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_depthwise_bwd_x_gpu(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    grad_out: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    grad_x: &mut CudaSlice<f32>,
    batch: usize,
    channels: usize,
    t_in: usize,
    t_out: usize,
    k_size: usize,
    stride: usize,
    padding: usize,
) -> Result<(), CudaError> {
    if k_size > 128 {
        return Err(CudaError::Shape(format!(
            "conv1d_depthwise_bwd_x_gpu: k_size={k_size} > 128 (kernel limit)"
        )));
    }
    const THREADS_PER_BLOCK: u32 = 128;
    let n_blocks = u32_len(batch * channels, "conv1d_depthwise_bwd_x_gpu")?;
    let batch_u = u32_len(batch, "conv1d_depthwise_bwd_x_gpu")?;
    let channels_u = u32_len(channels, "conv1d_depthwise_bwd_x_gpu")?;
    let t_in_u = u32_len(t_in, "conv1d_depthwise_bwd_x_gpu")?;
    let t_out_u = u32_len(t_out, "conv1d_depthwise_bwd_x_gpu")?;
    let k_size_u = u32_len(k_size, "conv1d_depthwise_bwd_x_gpu")?;
    let stride_u = u32_len(stride, "conv1d_depthwise_bwd_x_gpu")?;
    let padding_u = u32_len(padding, "conv1d_depthwise_bwd_x_gpu")?;

    let func = crate::module::load_kernel_function(ctx, "depthwise_conv1d_bwd_x_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        // Shared memory: K floats for the per-channel weight strip.
        shared_mem_bytes: k_size_u * 4,
    };
    let mut launch = stream.launch_builder(&func);
    launch.arg(grad_out);
    launch.arg(w);
    launch.arg(grad_x);
    launch.arg(&batch_u);
    launch.arg(&channels_u);
    launch.arg(&t_in_u);
    launch.arg(&t_out_u);
    launch.arg(&k_size_u);
    launch.arg(&stride_u);
    launch.arg(&padding_u);
    // SAFETY: device slices valid; shared_mem_bytes = k_size*4 ≤ 512 B.
    // K ≤ 128 validated above.
    unsafe { launch.launch(cfg) }?;
    Ok(())
}

/// Depthwise 1-D convolution backward w.r.t. weight (groups == channels).
///
/// B4.4e-conv1d. Grid is `(channels, k_size, 1)`.
/// Writes into the pre-allocated (zeroed) `grad_w` slice via parallel reduction.
/// `grad_w` layout: `(C, 1, K)` flat = `C * K` elements.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_depthwise_bwd_w_gpu(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    grad_out: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    grad_w: &mut CudaSlice<f32>,
    batch: usize,
    channels: usize,
    t_in: usize,
    t_out: usize,
    k_size: usize,
    stride: usize,
    padding: usize,
) -> Result<(), CudaError> {
    const THREADS_W: u32 = 128;
    // Shared memory: THREADS_W floats for the parallel reduction per block.
    const SMEM_BYTES: u32 = THREADS_W * 4; // 512 bytes

    let channels_u = u32_len(channels, "conv1d_depthwise_bwd_w_gpu")?;
    let k_size_u = u32_len(k_size, "conv1d_depthwise_bwd_w_gpu")?;
    let batch_u = u32_len(batch, "conv1d_depthwise_bwd_w_gpu")?;
    let t_in_u = u32_len(t_in, "conv1d_depthwise_bwd_w_gpu")?;
    let t_out_u = u32_len(t_out, "conv1d_depthwise_bwd_w_gpu")?;
    let stride_u = u32_len(stride, "conv1d_depthwise_bwd_w_gpu")?;
    let padding_u = u32_len(padding, "conv1d_depthwise_bwd_w_gpu")?;

    let func = crate::module::load_kernel_function(ctx, "depthwise_conv1d_bwd_w_f32")?;
    let cfg = LaunchConfig {
        // One block per (channel, kernel_position) pair.
        grid_dim: (channels_u, k_size_u, 1),
        block_dim: (THREADS_W, 1, 1),
        shared_mem_bytes: SMEM_BYTES,
    };
    let mut launch = stream.launch_builder(&func);
    launch.arg(grad_out);
    launch.arg(x);
    launch.arg(grad_w);
    launch.arg(&batch_u);
    launch.arg(&channels_u);
    launch.arg(&t_in_u);
    launch.arg(&t_out_u);
    launch.arg(&k_size_u);
    launch.arg(&stride_u);
    launch.arg(&padding_u);
    // SAFETY: device slices valid; SMEM_BYTES = 512 B well within hardware limit.
    // grid_dim.y = k_size so blockIdx.y < k_size always.
    unsafe { launch.launch(cfg) }?;
    Ok(())
}

/// RMSNorm backward — compute grad_x.
///
/// **Geometry**: one block per row, `RMSNORM_BWD_X_BLOCK` threads
/// cooperating via a strided loop + shared-memory tree reduction over
/// `d_model` (coalesced reads/writes, full-SM occupancy instead of the
/// old one-thread-per-row scheme — see the PTX kernel docstring for the
/// per-row VJP formula). `RMSNORM_BWD_X_BLOCK` must stay a power of two;
/// the kernel's tree reduction assumes it.
///
/// `grad_x` must be pre-allocated with the same length as `x`.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_backward_x_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    grad_x: &mut CudaSlice<f32>,
    grad_out: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    shape: &[usize], // shape of x (and grad_out)
    eps: f32,
) -> Result<(), CudaError> {
    const RMSNORM_BWD_X_BLOCK: u32 = 256;

    let n = x.len();
    check_unary_lengths(n, grad_x.len(), "rmsnorm_backward_x_f32")?;
    check_unary_lengths(n, grad_out.len(), "rmsnorm_backward_x_f32")?;
    let d_model = last_dim_or_err(shape, "rmsnorm_backward_x_f32")?;
    let n_rows = n / d_model;
    let func = crate::module::load_kernel_function(ctx, "rmsnorm_backward_x_f32")?;
    let n_rows_u32 = u32_len(n_rows, "rmsnorm_backward_x_f32")?;
    let d_model_u32 = u32_len(d_model, "rmsnorm_backward_x_f32")?;
    let cfg = LaunchConfig {
        // One block per row; the kernel covers d_model via a strided
        // loop, so grid.x = n_rows exactly (not ceil(numel / block)).
        grid_dim: (n_rows_u32, 1, 1),
        block_dim: (RMSNORM_BWD_X_BLOCK, 1, 1),
        // Two f32 partial-sum arrays (sum_sq, dy_dot), one slot/thread.
        shared_mem_bytes: 2 * RMSNORM_BWD_X_BLOCK * 4,
    };
    let mut launch = stream.launch_builder(&func);
    launch.arg(grad_x);
    launch.arg(grad_out);
    launch.arg(x);
    launch.arg(weight);
    launch.arg(&n_rows_u32);
    launch.arg(&d_model_u32);
    launch.arg(&eps);
    // SAFETY: sizes validated above; no aliasing; shared_mem_bytes covers
    // the kernel's `2 * block_dim.x` f32 reduction buffers.
    unsafe { launch.launch(cfg) }?;
    Ok(())
}

/// RMSNorm backward — compute grad_weight.
///
/// **Two-stage device reduction**, replacing a former one-thread-per-
/// column kernel that used only `d_model` threads total regardless of
/// `n_rows` (768 at production scale): stage 1
/// (`rmsnorm_backward_w_partial_f32`) computes per-row-chunk partial
/// sums into a `(n_chunks, d_model)` buffer using `d_model * n_chunks`
/// threads; stage 2 (`reduce_sum_dim_keepdim_f32`) reduces the chunk
/// axis with a fixed left-to-right sum (no atomics), so the result is
/// bit-for-bit deterministic across runs.
///
/// `grad_w` must be pre-allocated with `d_model` elements.
pub fn rmsnorm_backward_w_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    grad_w: &mut CudaSlice<f32>,
    grad_out: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    shape: &[usize], // shape of x
    eps: f32,
) -> Result<(), CudaError> {
    // Rows visited per thread within one chunk. Smaller = more parallelism
    // but a bigger (n_chunks, d_model) partial buffer + reduction pass.
    const ROWS_PER_CHUNK: usize = 16;
    const BLOCK: u32 = 256;

    let n = x.len();
    check_unary_lengths(n, grad_out.len(), "rmsnorm_backward_w_f32")?;
    let d_model = last_dim_or_err(shape, "rmsnorm_backward_w_f32")?;
    if grad_w.len() != d_model {
        return Err(CudaError::Shape(format!(
            "rmsnorm_backward_w_f32: grad_w.len()={} != d_model={d_model}",
            grad_w.len()
        )));
    }
    let n_rows = n / d_model;
    let n_chunks = n_rows.div_ceil(ROWS_PER_CHUNK).max(1);

    let mut partial = stream.alloc_zeros::<f32>(n_chunks * d_model)?;
    let func = crate::module::load_kernel_function(ctx, "rmsnorm_backward_w_partial_f32")?;
    let n_rows_u32 = u32_len(n_rows, "rmsnorm_backward_w_f32")?;
    let d_model_u32 = u32_len(d_model, "rmsnorm_backward_w_f32")?;
    let rows_per_chunk_u32 = u32_len(ROWS_PER_CHUNK, "rmsnorm_backward_w_f32")?;
    let n_chunks_u32 = u32_len(n_chunks, "rmsnorm_backward_w_f32")?;
    let grid_x = d_model_u32.div_ceil(BLOCK);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, n_chunks_u32, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(&func);
    launch.arg(&mut partial);
    launch.arg(grad_out);
    launch.arg(x);
    launch.arg(&n_rows_u32);
    launch.arg(&d_model_u32);
    launch.arg(&rows_per_chunk_u32);
    launch.arg(&n_chunks_u32);
    launch.arg(&eps);
    // SAFETY: grid covers every (column, chunk) pair; the kernel guards
    // col >= d_model and chunk >= n_chunks internally for ragged edges.
    unsafe { launch.launch(cfg) }?;

    // Stage 2: fixed-order (deterministic) reduction over the chunk axis.
    // Logical shape (outer=1, axis_len=n_chunks, inner=d_model).
    reduce_sum_dim_keepdim_f32(ctx, stream, &partial, grad_w, 1, n_chunks, d_model)?;
    Ok(())
}

/// Reverse inclusive cumsum along the last dimension (VJP of cumsum).
pub fn reverse_cumsum_lastdim_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    shape: &[usize],
) -> Result<(), CudaError> {
    let n = src.len();
    check_unary_lengths(n, dst.len(), "reverse_cumsum_lastdim_f32")?;
    let last_dim = last_dim_or_err(shape, "reverse_cumsum_lastdim_f32")?;
    let n_rows = n / last_dim;
    let func = crate::module::load_kernel_function(ctx, "reverse_cumsum_lastdim_f32")?;
    let n_rows_u32 = u32_len(n_rows, "reverse_cumsum_lastdim_f32")?;
    let last_dim_u32 = u32_len(last_dim, "reverse_cumsum_lastdim_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(src);
    launch.arg(dst);
    launch.arg(&n_rows_u32);
    launch.arg(&last_dim_u32);
    // SAFETY: sizes validated above.
    unsafe { launch.launch(launch_cfg(n_rows_u32)) }?;
    Ok(())
}

/// Sum-reduce along a single axis with keepdim=true.
///
/// `outer` = product of dims before `axis`.
/// `axis_len` = size of the axis being reduced.
/// `inner` = product of dims after `axis`.
/// Output has shape `[outer, 1, inner]` (= outer * inner elements).
pub fn reduce_sum_dim_keepdim_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    outer: usize,
    axis_len: usize,
    inner: usize,
) -> Result<(), CudaError> {
    let n_out = outer * inner;
    if dst.len() != n_out {
        return Err(CudaError::Shape(format!(
            "reduce_sum_dim_keepdim_f32: dst len {} != outer*inner={n_out}",
            dst.len()
        )));
    }
    let func = crate::module::load_kernel_function(ctx, "reduce_sum_dim_keepdim_f32")?;
    let n_out_u32 = u32_len(n_out, "reduce_sum_dim_keepdim_f32")?;
    let outer_u32 = u32_len(outer, "reduce_sum_dim_keepdim_f32")?;
    let axis_u32 = u32_len(axis_len, "reduce_sum_dim_keepdim_f32")?;
    let inner_u32 = u32_len(inner, "reduce_sum_dim_keepdim_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(src);
    launch.arg(dst);
    launch.arg(&outer_u32);
    launch.arg(&axis_u32);
    launch.arg(&inner_u32);
    // SAFETY: sizes validated above.
    unsafe { launch.launch(launch_cfg(n_out_u32)) }?;
    Ok(())
}

/// Pass 1 of the two-pass `sum_all_f32` device reduce — see
/// `sum_reduce_partial_f32` in the PTX kernel crate for the exact
/// (fixed-order, non-atomic) grid-stride summation.
///
/// `partial` must be pre-allocated with `n_chunks` elements.
pub fn sum_reduce_partial_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    partial: &mut CudaSlice<f32>,
    numel: usize,
    n_chunks: usize,
) -> Result<(), CudaError> {
    if partial.len() != n_chunks {
        return Err(CudaError::Shape(format!(
            "sum_reduce_partial_f32: partial len {} != n_chunks={n_chunks}",
            partial.len()
        )));
    }
    let func = crate::module::load_kernel_function(ctx, "sum_reduce_partial_f32")?;
    let numel_u32 = u32_len(numel, "sum_reduce_partial_f32")?;
    let n_chunks_u32 = u32_len(n_chunks, "sum_reduce_partial_f32")?;
    let mut launch = stream.launch_builder(&func);
    launch.arg(src);
    launch.arg(partial);
    launch.arg(&numel_u32);
    launch.arg(&n_chunks_u32);
    // SAFETY: sizes validated above; the kernel's grid-stride loop is
    // bounds-checked against `numel` internally (no OOB reads regardless
    // of whether `n_chunks` evenly divides `numel`).
    unsafe { launch.launch(launch_cfg(n_chunks_u32)) }?;
    Ok(())
}

/// Full-tensor sum reduction to a single scalar — fully device-side and
/// deterministic (B'.2f). Replaces the D2H → host `iter().sum()` → H2D
/// path that dominated `sum_all`/`mean_all`, and (via their VJPs) a
/// meaningful share of grad-norm wall time.
///
/// Two-level reduction, balanced around `sqrt(numel)` chunks so both
/// passes do roughly `sqrt(numel)` sequential work per thread: pass 1
/// (`sum_reduce_partial_f32`) computes `n_chunks` partial sums in
/// parallel; pass 2 (`reduce_sum_dim_keepdim_f32`) reduces those to one
/// scalar with a single fixed-order sequential pass (no atomics either
/// stage, so results reproduce bit-for-bit across runs).
pub fn sum_all_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
) -> Result<CudaSlice<f32>, CudaError> {
    let numel = src.len();
    if numel == 0 {
        return Err(CudaError::Shape("sum_all_f32: empty tensor".to_string()));
    }
    let n_chunks = ((numel as f64).sqrt().ceil() as usize).clamp(1, numel);
    let mut partial = stream.alloc_zeros::<f32>(n_chunks)?;
    sum_reduce_partial_f32(ctx, stream, src, &mut partial, numel, n_chunks)?;
    let mut out = stream.alloc_zeros::<f32>(1)?;
    reduce_sum_dim_keepdim_f32(ctx, stream, &partial, &mut out, 1, n_chunks, 1)?;
    Ok(out)
}

// ---- cuBLAS matrix multiply -----------------------------------------------

/// Batched row-major matrix multiply using cuBLAS.
///
/// `a` has shape `(batch, M, K)` (or `(M, K)` for 2D),
/// `b` has shape `(batch, K, N)` (or `(K, N)` for 2D),
/// result has shape `(batch, M, N)` (or `(M, N)`).
///
/// Broadcasting: if `a` has batch_size 1 and `b` has batch_size B > 1
/// (or vice versa), stride_a (or stride_b) is set to 0 so cuBLAS
/// repeats the same matrix for every batch slot.
///
/// **cuBLAS column-major trick**: cuBLAS is column-major. To compute
/// the row-major `C = A @ B`, we exploit the identity:
///   `C^T = B^T @ A^T`
/// In cuBLAS terms (column-major): calling `gemm(B, A)` with both ops
/// set to `N` gives exactly this. Arguments:
///   `transa = N, transb = N, m = N, n = M, k = K,
///    lda = N, ldb = K, ldc = N, a_arg = B_dev, b_arg = A_dev`
pub fn matmul_f32(
    cublas: &Arc<CudaBlas>,
    a: &CudaSlice<f32>,
    a_shape: &[usize], // full shape, e.g. [B1, ..., M, K]
    b: &CudaSlice<f32>,
    b_shape: &[usize], // full shape, e.g. [B2, ..., K, N]
    out: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    let rank = a_shape.len();
    if rank < 2 || b_shape.len() != rank {
        return Err(CudaError::Shape(format!(
            "matmul: a/b must have same rank ≥ 2, got a.rank={rank} b.rank={}",
            b_shape.len()
        )));
    }

    let m = a_shape[rank - 2];
    let k = a_shape[rank - 1];
    let k2 = b_shape[rank - 2];
    let n = b_shape[rank - 1];
    if k != k2 {
        return Err(CudaError::Shape(format!(
            "matmul: inner dims must match: a.K={k} vs b.K={k2}"
        )));
    }

    if rank == 2 {
        // Simple 2D case: single sgemm call.
        // cuBLAS column-major: C = A @ B  ⟺  gemm(B, A) with N,N
        // m_blas = N, n_blas = M, k_blas = K
        // lda = N (leading dim of B in row-major = num cols = N)
        // ldb = K (leading dim of A in row-major = num cols = K)
        // ldc = N
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };
        // SAFETY: dimensions and device pointers are validated above.
        unsafe { cublas.gemm(cfg, b, a, out) }?;
    } else {
        // Batched case: flatten all leading dims into a single batch.
        // Compute batch_a, batch_b independently (for broadcast support).
        let batch_a: usize = a_shape[..rank - 2].iter().product();
        let batch_b: usize = b_shape[..rank - 2].iter().product();
        // Only equal sizes or 1-vs-N are supported at this layer; the
        // general numpy broadcast is materialised by the caller in
        // ops_impl.rs before reaching us. Guard so future direct
        // callers don't read OOB by mistake.
        if batch_a != batch_b && batch_a != 1 && batch_b != 1 {
            return Err(CudaError::Shape(format!(
                "matmul: batched leading-dim sizes must be equal or one of them 1, got {batch_a} vs {batch_b}"
            )));
        }
        let batch = batch_a.max(batch_b);

        // Stride of 0 = broadcast: cuBLAS reuses the same matrix for
        // every batch index, implementing the broadcasting rule.
        let stride_b_val: i64 = if batch_b == 1 { 0 } else { (k * n) as i64 };
        let stride_a_val: i64 = if batch_a == 1 { 0 } else { (m * k) as i64 };
        let stride_c: i64 = (m * n) as i64;

        // cuBLAS column-major trick: gemm_strided_batched(B, A)
        // m_blas = N, n_blas = M, k_blas = K
        // stride for a_blas (= b_row_major) = stride_b_val (K*N)
        // stride for b_blas (= a_row_major) = stride_a_val (M*K)
        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0f32,
                lda: n as i32,
                ldb: k as i32,
                beta: 0.0f32,
                ldc: n as i32,
            },
            batch_size: batch as i32,
            stride_a: stride_b_val, // a_blas = B_row_major
            stride_b: stride_a_val, // b_blas = A_row_major
            stride_c,
        };
        // SAFETY: dimensions, strides, and device pointers are validated above.
        unsafe { cublas.gemm_strided_batched(cfg, b, a, out) }?;
    }

    // No explicit sync: cuBLAS handle is bound to the backend's single
    // stream so subsequent ops on that stream see the result in order.
    // Callers that move data back to host go through clone_dtoh, which
    // syncs on the same stream.
    Ok(())
}
