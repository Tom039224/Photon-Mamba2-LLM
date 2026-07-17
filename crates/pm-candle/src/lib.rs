//! Candle backend for Photon x Mamba2 LLM (Phase 1).
//!
//! Implements `pm_core::Ops` and `pm_backend::Backend` on top of
//! `candle-core`. Phase 2 will add `pm-cuda` which targets cudarc directly;
//! Phase 3 will add `pm-tt` for Tenstorrent. Both must not require changes
//! to model code in `pm-core`.

#![forbid(unsafe_code)]

mod backend;
mod dtype;
mod error;
mod ops_impl;
mod param;
mod tensor;

pub use backend::CandleBackend;
pub use error::Error;
pub use param::CandleParam;
pub use tensor::CandleTensor;
