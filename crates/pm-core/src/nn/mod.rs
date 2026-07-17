//! Reusable neural-network primitives.
//!
//! `Linear` and `Embedding` are the building blocks used throughout
//! `pm-core::photon`. They mirror their PyTorch counterparts but stay
//! backend-agnostic: only `Ops`/`Module<O>` are touched.
//!
//! Conventions:
//! - `Linear::weight` has shape `(in_features, out_features)`, matching
//!   the existing `Mamba2Block` projection layout (`x @ W`, not `W @ x`).
//!   This is opposite to PyTorch's `nn.Linear` (which stores `(out, in)`)
//!   and is intentional: it avoids a transpose on every forward.
//! - `Embedding::weight` has shape `(num_embeddings, embedding_dim)`.

pub mod embedding;
pub mod linear;

pub use embedding::Embedding;
pub use linear::Linear;
