//! Backend trait layered on top of `pm-core::Ops`.
//!
//! A concrete backend (e.g. `pm-candle::CandleBackend`) implements both
//! `pm_core::Ops` and `Backend`. Phase 2 will add `pm-cuda` and Phase 3
//! `pm-tt`; the rest of the workspace should not need to change.

#![forbid(unsafe_code)]

pub mod backend;
pub mod device;

pub use backend::Backend;
pub use device::DeviceKind;

pub use pm_core::Ops;
