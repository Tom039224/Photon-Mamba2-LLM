//! PHOTON architecture layers.
//!
//! Phase 1 / Group D builds these incrementally:
//! - D.1 `embedding` (this file): `TokenEmbedding` + `RotaryEmbedding`
//! - D.2 `chunker`: `ContextChunker` (level (l-1) → level l)
//! - D.3 `encoder`: `ContextEncoder` (Mamba2 stack, causal)
//! - D.4 `converter`: `ContextConverter` (1 latent → R_l latents)
//! - D.5 `decoder`: `ChunkLocalDecoder` (chunk-internal Mamba2 decoder)
//! - D.6/D.7 `HierarchicalEncoder`, `HierarchicalDecoder` (L=2 stacks)
//! - D.8 `model`: `PhotonMamba` assembly
//!
//! Paper reference: `Papers/Photon/main.tex` §2.1–2.3.

pub mod chunker;
pub mod converter;
pub mod decode_state;
pub mod decoder;
pub mod embedding;
pub mod encoder;
pub mod hierarchical_decoder;
pub mod hierarchical_encoder;

pub use chunker::ContextChunker;
pub use converter::ContextConverter;
pub use decode_state::PhotonDecodeState;
pub use decoder::ChunkLocalDecoder;
pub use embedding::{RotaryEmbedding, TokenEmbedding, ROPE_DEFAULT_BASE};
pub use encoder::{ContextEncoder, ContextEncoderState};
pub use hierarchical_decoder::{DecoderLevel, HierarchicalDecoder};
pub use hierarchical_encoder::{EncodedHierarchy, HierarchicalEncoder, HierarchicalLevel};
