//! Data pipeline (E.2 / E.3 / E.4).
//!
//! Phase-1 scope: plain-text file → BPE encode → fixed-length packed
//! sequences whose length is a multiple of `chunk_size^(L-1)` (the
//! PhotonMamba forward requirement).
//!
//! Streaming directly from FineWeb-Edu `.parquet` shards (the original
//! E.2 wording) is deferred to Phase 1.5 — the immediate target is
//! getting a real-data signal into `pm train`, and a plain UTF-8 text
//! file works for that and is unit-testable. The trait surface here is
//! intentionally minimal so a parquet backend can drop in later.

#![forbid(unsafe_code)]

pub mod packing;
pub mod text_source;

pub use packing::PackedBatcher;
pub use text_source::TextFileSource;

#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] pm_tokenizer::TokenizerError),
    #[error("config: {0}")]
    Config(String),
}
