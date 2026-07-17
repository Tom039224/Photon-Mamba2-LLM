//! Causal residual stack of `Mamba2Block` layers.
//!
//! Used by `HierarchicalEncoder` at every level of the bottom-up
//! encoder. Each layer is wrapped in a simple residual connection:
//!
//! ```text
//! h_0 = x
//! h_l = h_{l-1} + Mamba2Block_l(h_{l-1})
//! ```
//!
//! `Mamba2Block` already contains its own gated RMSNorm before the
//! output projection, so no outer pre-norm is needed at this level. The
//! Mamba2 SSD recurrence is intrinsically causal, which means the stack
//! inherits causality without any masking.
//!
//! Paper reference: `Papers/Photon/main.tex` §2.1 (context encoder).

use crate::checkpoint::{forward_checkpointed, CheckpointState};
use crate::mamba2::{Mamba2Block, Mamba2State};
use crate::{Module, Ops, Parameterized};

pub struct ContextEncoder<O: Ops> {
    pub layers: Vec<Mamba2Block<O>>,
}

/// Per-layer recurrent state for [`ContextEncoder::step`] (Phase C:
/// O(1)-memory recurrent decode). One [`Mamba2State`] per layer, in
/// the same order as `ContextEncoder::layers`.
pub struct ContextEncoderState<O: Ops> {
    pub layers: Vec<Mamba2State<O>>,
}

impl<O: Ops> ContextEncoder<O> {
    pub fn from_layers(layers: Vec<Mamba2Block<O>>) -> Self {
        assert!(!layers.is_empty(), "ContextEncoder needs at least 1 layer");
        let d_model = layers[0].config.d_model;
        for (i, l) in layers.iter().enumerate() {
            assert_eq!(
                l.config.d_model, d_model,
                "ContextEncoder layer {i} d_model mismatch: got {}, expected {d_model}",
                l.config.d_model
            );
        }
        Self { layers }
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

impl<O: Ops> Module<O> for ContextEncoder<O> {
    fn forward(&self, ops: &O, x: &O::Tensor) -> Result<O::Tensor, O::Error> {
        let first_out = self.layers[0].forward(ops, x)?;
        let mut h = ops.add(x, &first_out)?;
        for layer in &self.layers[1..] {
            let y = layer.forward(ops, &h)?;
            h = ops.add(&h, &y)?;
        }
        Ok(h)
    }
}

impl<O: Ops> ContextEncoder<O>
where
    O::Tensor: Clone,
    O::Param: Clone,
{
    /// Same as [`Module::forward`] but each Mamba2 block is wrapped in a
    /// checkpoint segment. The residual `add`s stay in the main
    /// autograd tape so the residual stream's gradient flows through
    /// `cp.segments[i].boundary` as expected.
    pub fn forward_checkpointed(
        &self,
        ops: &O,
        x: &O::Tensor,
        cp: &mut CheckpointState<O>,
        block_id_offset: usize,
    ) -> Result<O::Tensor, O::Error> {
        let first_delta = forward_checkpointed(ops, &self.layers[0], block_id_offset, x, cp)?;
        let mut h = ops.add(x, &first_delta)?;
        for (i, layer) in self.layers[1..].iter().enumerate() {
            let delta = forward_checkpointed(ops, layer, block_id_offset + 1 + i, &h, cp)?;
            h = ops.add(&h, &delta)?;
        }
        Ok(h)
    }
}

impl<O: Ops> Parameterized<O> for ContextEncoder<O> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>) {
        for layer in &self.layers {
            layer.append_params(out);
        }
    }
}

// -------- Phase C: O(1)-memory recurrent decode --------

impl<O: Ops> ContextEncoder<O> {
    /// All-zero state for [`step`](Self::step): one fresh
    /// [`Mamba2State`] per layer.
    pub fn zero_state(&self, ops: &O, batch: usize) -> Result<ContextEncoderState<O>, O::Error> {
        let layers = self
            .layers
            .iter()
            .map(|l| Mamba2State::zeros(ops, &l.config, batch))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ContextEncoderState { layers })
    }
}

impl<O: Ops> ContextEncoder<O>
where
    O::Tensor: Clone,
{
    /// One-token step through every layer, mirroring [`Module::forward`]'s
    /// residual recursion (`h_l = h_{l-1} + Mamba2Block_l(h_{l-1})`) but
    /// advancing each layer's [`Mamba2State`] instead of re-scanning the
    /// whole sequence. Functional / non-mutating, for the same reason
    /// as [`Mamba2Block::step`]: `state` is borrowed, a fresh
    /// `ContextEncoderState` is returned.
    pub fn step(
        &self,
        ops: &O,
        x_t: &O::Tensor,
        state: &ContextEncoderState<O>,
    ) -> Result<(O::Tensor, ContextEncoderState<O>), O::Error> {
        assert_eq!(
            state.layers.len(),
            self.layers.len(),
            "ContextEncoderState layer count mismatch: got {}, expected {}",
            state.layers.len(),
            self.layers.len()
        );
        let mut new_states = Vec::with_capacity(self.layers.len());
        let (y0, st0) = self.layers[0].step(ops, x_t, &state.layers[0])?;
        new_states.push(st0);
        let mut h = ops.add(x_t, &y0)?;
        for (layer, st) in self.layers[1..].iter().zip(state.layers[1..].iter()) {
            let (y, st_new) = layer.step(ops, &h, st)?;
            new_states.push(st_new);
            h = ops.add(&h, &y)?;
        }
        Ok((h, ContextEncoderState { layers: new_states }))
    }
}
