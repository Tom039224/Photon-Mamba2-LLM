//! Bottom-up context chunker for PHOTON.
//!
//! Maps a stream at level `l-1` to a coarser stream at level `l` by
//! concatenating every `c` consecutive vectors and projecting back to
//! `d_out`:
//!
//! ```text
//! x:  (B, T,        d_in)         where T % c == 0
//! r:  (B, T / c,    c * d_in)     reshape (concatenate chunks)
//! y:  (B, T / c,    d_out)        y = r @ W  + b
//! ```
//!
//! Paper reference: `Papers/Photon/main.tex` §2.1 (hierarchical encoder,
//! "context chunker"). Chunk size `c` corresponds to `C_l` in the paper.

use crate::nn::Linear;
use crate::{Module, Ops, Parameterized, Tensor};

pub struct ContextChunker<O: Ops> {
    pub proj: Linear<O>,
    pub chunk_size: usize, // C_l
    pub d_in: usize,
    pub d_out: usize,
}

impl<O: Ops> ContextChunker<O> {
    pub fn from_constants(
        ops: &O,
        d_in: usize,
        d_out: usize,
        chunk_size: usize,
        weight_scale: f32,
    ) -> Result<Self, O::Error> {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        let proj = Linear::from_constants(ops, chunk_size * d_in, d_out, true, weight_scale)?;
        Ok(Self {
            proj,
            chunk_size,
            d_in,
            d_out,
        })
    }

    pub fn from_linear(d_in: usize, d_out: usize, chunk_size: usize, proj: Linear<O>) -> Self {
        Self {
            proj,
            chunk_size,
            d_in,
            d_out,
        }
    }
}

impl<O: Ops> Module<O> for ContextChunker<O> {
    fn forward(&self, ops: &O, x: &O::Tensor) -> Result<O::Tensor, O::Error> {
        let shape = x.shape();
        assert_eq!(shape.len(), 3, "ContextChunker expects (B, T, d_in)");
        let (b, t, d) = (shape[0], shape[1], shape[2]);
        assert_eq!(
            d, self.d_in,
            "d_in mismatch: got {d}, expected {}",
            self.d_in
        );
        assert!(
            t.is_multiple_of(self.chunk_size),
            "T={t} must be a multiple of chunk_size={}",
            self.chunk_size
        );

        let n_chunks = t / self.chunk_size;
        let flat = ops.reshape(x, &[b, n_chunks, self.chunk_size * d])?;
        self.proj.forward(ops, &flat)
    }
}

impl<O: Ops> Parameterized<O> for ContextChunker<O> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>) {
        self.proj.append_params(out);
    }
}
