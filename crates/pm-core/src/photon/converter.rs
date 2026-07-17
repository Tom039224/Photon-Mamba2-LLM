//! Top-down context converter for PHOTON.
//!
//! Inverse of [`ContextChunker`](super::ContextChunker): turns one latent
//! at level `l` into `r` latents at level `l-1`. Phase 1 uses a simple
//! linear "depth-to-time" expansion:
//!
//! ```text
//! x:  (B, S,     d_in)
//! w:  (d_in, r * d_out)
//! y0: (B, S,     r * d_out)        y0 = x @ w  + b
//! y:  (B, S * r, d_out)            reshape: split last dim, flatten S,r
//! ```
//!
//! A learned `starting_latent` of shape `(d_out,)` is held alongside the
//! projection. PHOTON uses it as the seed for the chunk-local decoder
//! when no higher-level context is available (autoregressive bootstrap).
//! It is **not** prepended in this layer's `forward`; doing so belongs to
//! the `HierarchicalDecoder` once it knows the generation cursor (D.7).
//!
//! Paper reference: `Papers/Photon/main.tex` §2.2 (top-down decoder).

use crate::nn::Linear;
use crate::{Module, Ops, Parameterized, Tensor};

pub struct ContextConverter<O: Ops> {
    pub proj: Linear<O>,
    pub starting_latent: O::Param, // (d_out,)
    pub expansion: usize,          // R_l
    pub d_in: usize,
    pub d_out: usize,
}

impl<O: Ops> ContextConverter<O> {
    pub fn from_constants(
        ops: &O,
        d_in: usize,
        d_out: usize,
        expansion: usize,
        weight_scale: f32,
    ) -> Result<Self, O::Error> {
        assert!(expansion > 0, "expansion must be > 0");
        let proj = Linear::from_constants(ops, d_in, expansion * d_out, true, weight_scale)?;
        let starting_latent = ops.param_from_slice_f32(&vec![0.0; d_out], &[d_out])?;
        Ok(Self {
            proj,
            starting_latent,
            expansion,
            d_in,
            d_out,
        })
    }

    pub fn from_parts(
        d_in: usize,
        d_out: usize,
        expansion: usize,
        proj: Linear<O>,
        starting_latent: O::Param,
    ) -> Self {
        Self {
            proj,
            starting_latent,
            expansion,
            d_in,
            d_out,
        }
    }
}

impl<O: Ops> Module<O> for ContextConverter<O> {
    fn forward(&self, ops: &O, x: &O::Tensor) -> Result<O::Tensor, O::Error> {
        let shape = x.shape();
        assert_eq!(shape.len(), 3, "ContextConverter expects (B, S, d_in)");
        let (b, s, d) = (shape[0], shape[1], shape[2]);
        assert_eq!(
            d, self.d_in,
            "d_in mismatch: got {d}, expected {}",
            self.d_in
        );

        let projected = self.proj.forward(ops, x)?;
        ops.reshape(&projected, &[b, s * self.expansion, self.d_out])
    }
}

impl<O: Ops> Parameterized<O> for ContextConverter<O> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>) {
        self.proj.append_params(out);
        out.push(&self.starting_latent);
    }
}
