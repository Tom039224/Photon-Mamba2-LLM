//! Plain UTF-8 text file → stream of token ids.
//!
//! Reads line by line and tokenises with the supplied [`BpeTokenizer`].
//! A configurable separator token (BOS by default = id 0) is inserted
//! between documents so the packer doesn't accidentally let context
//! bleed across document boundaries.

use std::io::{BufRead, BufReader};
use std::path::Path;

use pm_tokenizer::BpeTokenizer;

use crate::DataError;

pub struct TextFileSource {
    reader: BufReader<std::fs::File>,
    separator: i64,
    buffer: Vec<i64>,
    eof: bool,
}

impl TextFileSource {
    pub fn open<P: AsRef<Path>>(path: P, separator: i64) -> Result<Self, DataError> {
        let file = std::fs::File::open(path.as_ref())?;
        Ok(Self {
            reader: BufReader::new(file),
            separator,
            buffer: Vec::new(),
            eof: false,
        })
    }

    /// Yield the next `n` token ids. Returns fewer than `n` only when
    /// the file is exhausted and the internal buffer is drained.
    pub fn read_n(&mut self, n: usize, tokenizer: &BpeTokenizer) -> Result<Vec<i64>, DataError> {
        while self.buffer.len() < n && !self.eof {
            let mut line = String::new();
            let read = self.reader.read_line(&mut line)?;
            if read == 0 {
                self.eof = true;
                break;
            }
            let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
            if trimmed.is_empty() {
                continue;
            }
            let ids = tokenizer.encode(trimmed, false)?;
            self.buffer.extend(ids);
            self.buffer.push(self.separator);
        }
        let take = n.min(self.buffer.len());
        let out: Vec<i64> = self.buffer.drain(..take).collect();
        Ok(out)
    }

    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.eof && self.buffer.is_empty()
    }
}
