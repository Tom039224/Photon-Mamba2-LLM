//! Chunk-local Mamba2 decoder (PHOTON D.5).
//!
//! Operates on chunks of length `r + c` where `r = R_l` is the number of
//! context latents fed down from level `l+1` (via `ContextConverter`)
//! and `c = C_l` is the number of level-`l` positions predicted inside
//! the chunk.
//!
//! Training-time parallelism: every chunk's bounded context is
//! independent of every other chunk (the higher level already conveys
//! cross-chunk dependencies via the converter), so the chunk axis is
//! folded into the batch axis for one big parallel forward:
//!
//! ```text
//! x:    (B, S,    r + c, d)
//! flat: (B * S,   r + c, d)        reshape
//! flat: (B * S,   r + c, d)        Mamba2 residual stack (causal)
//! y:    (B, S,    r + c, d)        reshape back
//! ```
//!
//! Returning the full `(r + c)` positions keeps this layer composable;
//! `HierarchicalDecoder` (D.7) slices the trailing `c` and flattens
//! chunks into the level-`l` output stream.
//!
//! Paper reference: `Papers/Photon/main.tex` §2.2 (chunk-local decoder).

use crate::checkpoint::{forward_checkpointed, CheckpointState};
use crate::mamba2::Mamba2Block;
use crate::{Module, Ops, Parameterized, Tensor};

pub struct ChunkLocalDecoder<O: Ops> {
    pub layers: Vec<Mamba2Block<O>>,
    pub r_l: usize,
    pub c_l: usize,
}

impl<O: Ops> ChunkLocalDecoder<O> {
    pub fn from_layers(layers: Vec<Mamba2Block<O>>, r_l: usize, c_l: usize) -> Self {
        assert!(
            !layers.is_empty(),
            "ChunkLocalDecoder needs at least 1 layer"
        );
        assert!(r_l > 0, "r_l (latent context) must be > 0");
        assert!(c_l > 0, "c_l (predicted positions) must be > 0");
        let d_model = layers[0].config.d_model;
        for (i, l) in layers.iter().enumerate() {
            assert_eq!(
                l.config.d_model, d_model,
                "ChunkLocalDecoder layer {i} d_model mismatch"
            );
        }
        Self { layers, r_l, c_l }
    }

    pub fn bounded_context_len(&self) -> usize {
        self.r_l + self.c_l
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

impl<O: Ops> Module<O> for ChunkLocalDecoder<O> {
    /// `x`: `(B, S, r + c, d)` — chunks already assembled by the caller.
    /// Returns `(B, S, r + c, d)`.
    fn forward(&self, ops: &O, x: &O::Tensor) -> Result<O::Tensor, O::Error> {
        let shape = x.shape();
        assert_eq!(shape.len(), 4, "ChunkLocalDecoder expects (B, S, r+c, d)");
        let (b, s, t, d) = (shape[0], shape[1], shape[2], shape[3]);
        let expected_t = self.bounded_context_len();
        assert_eq!(
            t, expected_t,
            "ChunkLocalDecoder: T={t} must equal r_l + c_l = {expected_t}"
        );

        let flat = ops.reshape(x, &[b * s, t, d])?;

        let first_out = self.layers[0].forward(ops, &flat)?;
        let mut h = ops.add(&flat, &first_out)?;
        for layer in &self.layers[1..] {
            let y = layer.forward(ops, &h)?;
            h = ops.add(&h, &y)?;
        }

        ops.reshape(&h, &[b, s, t, d])
    }
}

impl<O: Ops> Parameterized<O> for ChunkLocalDecoder<O> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>) {
        for layer in &self.layers {
            layer.append_params(out);
        }
    }
}

impl<O: Ops> ChunkLocalDecoder<O>
where
    O::Tensor: Clone,
    O::Param: Clone,
{
    /// Same as [`Module::forward`] but each Mamba2 block in the stack
    /// is wrapped in a checkpoint segment. The `(B, S, r+c, d) →
    /// (B*S, r+c, d)` reshape stays outside the checkpoint so the
    /// per-block forwards see the same flattened shape they would
    /// without checkpointing.
    pub fn forward_checkpointed(
        &self,
        ops: &O,
        x: &O::Tensor,
        cp: &mut CheckpointState<O>,
        block_id_offset: usize,
    ) -> Result<O::Tensor, O::Error> {
        let shape = x.shape();
        assert_eq!(shape.len(), 4, "ChunkLocalDecoder expects (B, S, r+c, d)");
        let (b, s, t, d) = (shape[0], shape[1], shape[2], shape[3]);
        let expected_t = self.bounded_context_len();
        assert_eq!(
            t, expected_t,
            "ChunkLocalDecoder: T={t} must equal r_l + c_l = {expected_t}"
        );

        let flat = ops.reshape(x, &[b * s, t, d])?;

        let first_delta = forward_checkpointed(ops, &self.layers[0], block_id_offset, &flat, cp)?;
        let mut h = ops.add(&flat, &first_delta)?;
        for (i, layer) in self.layers[1..].iter().enumerate() {
            let delta = forward_checkpointed(ops, layer, block_id_offset + 1 + i, &h, cp)?;
            h = ops.add(&h, &delta)?;
        }

        ops.reshape(&h, &[b, s, t, d])
    }
}
