//! Photon x Mamba2 LLM core library.
//!
//! Backend-agnostic abstractions: `Dtype`, `Shape`, `Tensor`, `Ops`, `Module`.
//! Concrete tensor / device / kernel code lives in backend crates
//! (`pm-candle`, `pm-cuda`, `pm-tt`); `pm-core` only sees them through traits.
//!
//! This crate must not depend on any backend-specific crate. The
//! `pm-core-no-backend-deps` CI job enforces this.

#![forbid(unsafe_code)]

pub mod checkpoint;
pub mod dtype;
pub mod flat;
pub mod loss;
pub mod mamba2;
pub mod model;
pub mod module;
pub mod nn;
pub mod ops;
pub mod param;
pub mod photon;
pub mod shape;
pub mod tensor;

pub use checkpoint::{
    checkpoint_backward, forward_checkpointed, CheckpointSegment, CheckpointState,
};
pub use dtype::Dtype;
pub use flat::FlatMamba;
pub use loss::fused_cross_entropy_tiled;
pub use model::{PhotonForwardOutput, PhotonHiddenOutput, PhotonMamba};
pub use module::Module;
pub use ops::Ops;
pub use param::{Param, Parameterized};
pub use shape::Shape;
pub use tensor::Tensor;
