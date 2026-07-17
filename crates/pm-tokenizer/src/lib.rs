//! Thin wrapper around HuggingFace `tokenizers` (E.1).
//!
//! Phase-1 reuses the GPT-2 BPE tokenizer.json shipped on the Hub. The
//! caller hands us a path; we expose `encode(text) -> Vec<i64>` and
//! `decode(ids) -> String`. SentencePiece / training of a custom
//! tokenizer is out of scope (Phase 1.5).

#![forbid(unsafe_code)]

use std::path::Path;

use tokenizers::Tokenizer;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("load: {0}")]
    Load(String),
    #[error("encode: {0}")]
    Encode(String),
    #[error("decode: {0}")]
    Decode(String),
}

pub struct BpeTokenizer {
    inner: Tokenizer,
    vocab_size: usize,
}

impl BpeTokenizer {
    /// Load a HuggingFace `tokenizer.json` from disk. The file format is
    /// the one produced by `AutoTokenizer.save_pretrained(...)` and is
    /// what `tokenizers` natively reads.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_file(path.as_ref())
            .map_err(|e| TokenizerError::Load(format!("{e}")))?;
        let vocab_size = inner.get_vocab_size(true);
        Ok(Self { inner, vocab_size })
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>, TokenizerError> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::Encode(format!("{e}")))?;
        Ok(enc.get_ids().iter().map(|&u| u as i64).collect())
    }

    pub fn decode(&self, ids: &[i64], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        let u32_ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.inner
            .decode(&u32_ids, skip_special_tokens)
            .map_err(|e| TokenizerError::Decode(format!("{e}")))
    }
}
