//! Mamba2 building blocks.
//!
//! Phase 1 lands a pure-scalar reference of the SSD chunked scan so the
//! rest of the architecture (PHOTON encoder / decoder) can be built and
//! tested. Performance-oriented implementations live in backend crates
//! (`pm-candle::Ops::ssd_scan`, eventually `pm-cuda::Ops::ssd_scan`).

pub mod block;
pub mod ssd;
pub mod ssd_ops;
pub mod state;

pub use block::{Mamba2Block, Mamba2Config};
pub use ssd::ssd_scan_naive_scalar;
pub use ssd_ops::ssd_scan_ops_default;
pub use state::Mamba2State;
