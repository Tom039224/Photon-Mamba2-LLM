//! Compiled PTX produced by `kernel/` and embedded at build time.

/// PTX source for every kernel exported by `pm_cuda_kernel` (the
/// `kernel/` subcrate compiled with `--target nvptx64-nvidia-cuda`).
///
/// Load it once per `CudaContext`:
///
/// ```ignore
/// use cudarc::{driver::CudaContext, nvrtc::Ptx};
/// let ctx = CudaContext::new(0)?;
/// let module = ctx.load_module(Ptx::from_src(pm_cuda::KERNEL_PTX))?;
/// let mul = module.load_function("mul")?;
/// ```
pub const KERNEL_PTX: &str = include_str!(env!("KERNEL_PTX"));
