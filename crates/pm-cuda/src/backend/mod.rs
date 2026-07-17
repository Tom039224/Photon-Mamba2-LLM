//! pm-cuda native backend — Candle 置換 (B4 計画、`docs/b4-design.md`)。
//!
//! 現状 (B4.3a): autograd tape + simple ops backward 完成。
//! - [`CudaTensor`]: F32/I64 dual-storage variant + autograd `NodeId`
//! - [`CudaBackend`]: cuBLAS handle + param-id allocator + `Arc<Mutex<Tape>>`
//! - [`CudaParam`]: `Arc<ParamInner>` + `UnsafeCell` for in-place SGD
//! - [`CudaGradStore`]: `HashMap<ParamId, CudaTensor>` gradient store
//! - `impl pm_core::Ops for CudaBackend`: all 42 methods implemented
//!
//! `pm-train` / `pm-infer` の差し替えは B4.4 で扱う。

pub mod grad;
pub mod kernels;
pub mod ops_impl;
pub mod param;
pub mod tape;
pub mod tensor;

pub use grad::CudaGradStore;
pub use param::{CudaParam, ParamId};
pub use tape::{NodeId, Tape};
pub use tensor::CudaTensor;

mod backend_impl;
pub use backend_impl::CudaBackend;
