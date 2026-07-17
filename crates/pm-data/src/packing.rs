//! Pack token-id streams into fixed-shape (batch_size, seq_len) blocks.
//!
//! `seq_len` must be a multiple of `chunk_size^(L-1)` so that the
//! PhotonMamba forward accepts the resulting batch. Padding fills any
//! short tail at end-of-stream so the consumer always sees a full
//! tensor.

use pm_tokenizer::BpeTokenizer;

use crate::{DataError, TextFileSource};

#[derive(Debug, Clone)]
pub struct PackedBatcher {
    pub batch_size: usize,
    pub seq_len: usize,
    pub pad_token_id: i64,
}

impl PackedBatcher {
    /// `seq_len` is asserted at construction to be a positive multiple
    /// of `chunk_product`. Returning a `DataError` keeps the error path
    /// uniform with the rest of pm-data.
    pub fn new(
        batch_size: usize,
        seq_len: usize,
        chunk_product: usize,
        pad_token_id: i64,
    ) -> Result<Self, DataError> {
        if batch_size == 0 || seq_len == 0 {
            return Err(DataError::Config(
                "batch_size and seq_len must be > 0".into(),
            ));
        }
        if !seq_len.is_multiple_of(chunk_product) {
            return Err(DataError::Config(format!(
                "seq_len ({seq_len}) must be a multiple of chunk_product ({chunk_product})"
            )));
        }
        Ok(Self {
            batch_size,
            seq_len,
            pad_token_id,
        })
    }

    /// Build `(ids, targets)` of shape `(batch_size, seq_len)` each.
    /// Targets are the input shifted left by one position, with the last
    /// slot set to `pad_token_id`. Reads up to `batch_size * (seq_len + 1)`
    /// tokens from the source.
    ///
    /// Returns `Ok(None)` once the source is fully drained.
    #[allow(clippy::type_complexity)]
    pub fn next_batch(
        &self,
        source: &mut TextFileSource,
        tokenizer: &BpeTokenizer,
    ) -> Result<Option<(Vec<i64>, Vec<i64>)>, DataError> {
        let needed = self.batch_size * (self.seq_len + 1);
        let mut raw = source.read_n(needed, tokenizer)?;
        if raw.is_empty() {
            return Ok(None);
        }
        if raw.len() < needed {
            raw.resize(needed, self.pad_token_id);
        }

        let mut ids = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(self.batch_size * self.seq_len);
        for b in 0..self.batch_size {
            let base = b * (self.seq_len + 1);
            ids.extend_from_slice(&raw[base..base + self.seq_len]);
            targets.extend_from_slice(&raw[base + 1..base + 1 + self.seq_len]);
        }
        Ok(Some((ids, targets)))
    }
}
