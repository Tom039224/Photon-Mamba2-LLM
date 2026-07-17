//! Process-global cache of the loaded PTX module.
//!
//! Every host launcher in this crate (ssd_scan, backend kernels) needs
//! `CudaModule::load_function(name)`, so we parse `KERNEL_PTX` once and
//! reuse the resulting `Arc<CudaModule>`. Lock-free via `AtomicPtr`.

use core::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule},
    nvrtc::Ptx,
};

use crate::{CudaError, KERNEL_PTX};

static MODULE_CACHE: AtomicPtr<CudaModule> = AtomicPtr::new(core::ptr::null_mut());

fn module_for(ctx: &Arc<CudaContext>) -> Result<Arc<CudaModule>, CudaError> {
    let cached = MODULE_CACHE.load(Ordering::Acquire);
    if !cached.is_null() {
        // SAFETY: any non-null entry was published by a successful
        // `Arc::into_raw` below; cloning the Arc bumps its refcount.
        let arc = unsafe { Arc::from_raw(cached) };
        let out = arc.clone();
        // Don't drop the cache's strong reference.
        let _ = Arc::into_raw(arc);
        return Ok(out);
    }
    let module = ctx.load_module(Ptx::from_src(KERNEL_PTX))?;
    let raw = Arc::into_raw(module.clone()) as *mut CudaModule;
    match MODULE_CACHE.compare_exchange(
        core::ptr::null_mut(),
        raw,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => Ok(module),
        Err(existing) => {
            // Lost the race; drop our copy and use the published one.
            // SAFETY: raw was produced from Arc::into_raw above.
            drop(unsafe { Arc::from_raw(raw) });
            // SAFETY: existing was published by another caller.
            let arc = unsafe { Arc::from_raw(existing) };
            let out = arc.clone();
            let _ = Arc::into_raw(arc);
            Ok(out)
        }
    }
}

/// Look up a named kernel from the bundled `pm-cuda-kernel` PTX,
/// loading the module on the first call (process-global cache).
pub(crate) fn load_kernel_function(
    ctx: &Arc<CudaContext>,
    name: &'static str,
) -> Result<CudaFunction, CudaError> {
    let module = module_for(ctx)?;
    module
        .load_function(name)
        .map_err(|_| CudaError::KernelNotFound(name))
}
