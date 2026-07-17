//! PHOTON hierarchical encoder (D.6).
//!
//! Bottom-up stack of `L` levels. At every level we run a
//! [`ContextEncoder`] (Mamba2 residual stack) over the current stream;
//! every level except the top then runs a [`ContextChunker`] to produce
//! the coarser stream that feeds the next level.
//!
//! Paper reference: `Papers/Photon/main.tex` §2.1.

use crate::photon::{ContextChunker, ContextEncoder};
use crate::{Ops, Parameterized};

pub struct HierarchicalLevel<O: Ops> {
    pub encoder: ContextEncoder<O>,
    /// `None` for the top level.
    pub chunker: Option<ContextChunker<O>>,
}

pub struct HierarchicalEncoder<O: Ops> {
    pub levels: Vec<HierarchicalLevel<O>>,
}

/// Per-level intermediate tensors produced by [`HierarchicalEncoder::encode`].
pub struct EncodedHierarchy<O: Ops> {
    /// `encoded[l]` is the output of `levels[l].encoder`. Length `L`.
    pub encoded: Vec<O::Tensor>,
    /// `chunked[l]` is the output of `levels[l].chunker`. Length `L - 1`.
    pub chunked: Vec<O::Tensor>,
}

impl<O: Ops> HierarchicalEncoder<O> {
    pub fn from_levels(levels: Vec<HierarchicalLevel<O>>) -> Self {
        assert!(
            !levels.is_empty(),
            "HierarchicalEncoder needs at least 1 level"
        );
        let n = levels.len();
        for (i, lvl) in levels.iter().enumerate() {
            let is_top = i + 1 == n;
            assert_eq!(
                lvl.chunker.is_some(),
                !is_top,
                "level {i}: chunker present iff not top level (is_top={is_top})"
            );
        }
        Self { levels }
    }

    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Run the bottom-up pass. `x` is the level-0 stream.
    pub fn encode(&self, ops: &O, x: &O::Tensor) -> Result<EncodedHierarchy<O>, O::Error> {
        use crate::Module;
        let n = self.levels.len();
        let mut encoded = Vec::with_capacity(n);
        let mut chunked = Vec::with_capacity(n.saturating_sub(1));

        let e0 = self.levels[0].encoder.forward(ops, x)?;
        encoded.push(e0);

        for l in 0..n {
            if let Some(ch) = &self.levels[l].chunker {
                let next_stream = ch.forward(ops, &encoded[l])?;
                chunked.push(next_stream);
                let next_l = l + 1;
                if next_l < n {
                    let e_next = self.levels[next_l].encoder.forward(ops, &chunked[l])?;
                    encoded.push(e_next);
                }
            }
        }

        Ok(EncodedHierarchy { encoded, chunked })
    }
}

impl<O: Ops> Parameterized<O> for HierarchicalLevel<O> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>) {
        self.encoder.append_params(out);
        if let Some(ch) = &self.chunker {
            ch.append_params(out);
        }
    }
}

impl<O: Ops> Parameterized<O> for HierarchicalEncoder<O> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>) {
        for lvl in &self.levels {
            lvl.append_params(out);
        }
    }
}
