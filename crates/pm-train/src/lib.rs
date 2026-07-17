//! Photon × Mamba2 training crate.
//!
//! Backend-agnostic optimisers, losses, gradient clipping, checkpoint
//! save/load, and the high-level `Trainer`. All public types are
//! generic over `pm_core::Ops`, so adding a new backend (Phase 2
//! cudarc, Phase 3 Tenstorrent) involves no edits here.

#![forbid(unsafe_code)]

pub mod checkpoint;
pub mod clip;
pub mod loss;
pub mod optim;
pub mod trainer;

pub use checkpoint::{load as load_checkpoint, save as save_checkpoint};
pub use clip::clip_grad_norm;
pub use loss::{
    cross_entropy_loss, fused_cross_entropy_injected, fused_photon_loss_injected, photon_loss,
    recursive_consistency_loss, LossComponents, PhotonLossReport,
};
pub use optim::{AdamW, AdamWConfig, Optimizer, Sgd};
pub use trainer::{StepReport, Trainer};

/// Error wrapper that distinguishes backend (`O::Error`) failures from
/// checkpoint / safetensors plumbing failures.
#[derive(Debug, thiserror::Error)]
pub enum TrainError<E: std::error::Error + Send + Sync + 'static> {
    #[error("backend: {0}")]
    Backend(#[source] E),

    #[error("safetensors: {0}")]
    Safetensors(String),
}
