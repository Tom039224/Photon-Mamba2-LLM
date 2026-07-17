//! pm-cuda — CUDA backend for the Photon × Mamba2 LLM workspace (Phase 2).
//!
//! Hosts the cudarc-based device runtime and PTX kernels (compiled in
//! `kernel/`). Forward implementations of `pm_core::Ops` — notably the
//! fused SSD scan from Mamba2 — are added in Group J/K.
//!
//! When the `cuda` feature is disabled, the crate is an empty stub so
//! the workspace builds on machines without CUDA.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

#[cfg(feature = "cuda")]
mod ptx;

#[cfg(feature = "cuda")]
pub use ptx::*;

#[cfg(feature = "cuda")]
mod error;

#[cfg(feature = "cuda")]
pub use error::CudaError;

#[cfg(feature = "cuda")]
mod module;

#[cfg(feature = "cuda")]
mod ssd;

#[cfg(feature = "cuda")]
pub use ssd::{ssd_scan_chunked, ssd_scan_chunked_with_context, MAX_BLOCK_LEN, MAX_N_DIM};

#[cfg(feature = "cuda")]
pub mod backend;

#[cfg(feature = "cuda")]
pub use backend::{CudaBackend, CudaGradStore, CudaParam, CudaTensor, ParamId};

/// Env-gated per-op wall-time profiler (Phase B'.1b, `PLAN.md` "Phase B'").
/// Disabled by default (`PM_CUDA_PROFILE=1` to enable) — see module docs.
#[cfg(feature = "cuda")]
pub mod profiler;
