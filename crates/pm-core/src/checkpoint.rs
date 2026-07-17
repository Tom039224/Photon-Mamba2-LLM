//! Activation checkpointing (F.6).
//!
//! Phase-1 strategy: per-`Mamba2Block` checkpointing. Each call to
//! [`forward_checkpointed`] runs a block's forward, wraps the output as
//! a fresh trainable [`Param`] (a "boundary"), records `(input, block_id,
//! boundary)` into the state, and drops the raw output so its autograd
//! tape — including the 7× `(B,H,T,T)` decay/bc/score intermediates
//! that dominate the SSD scan's memory footprint — gets garbage-collected.
//!
//! After the main `loss.backward()`, [`checkpoint_backward`] walks the
//! recorded segments in reverse:
//! 1. Pull the gradient that flowed into the boundary (i.e. `dy`).
//! 2. Recompute the block forward from the saved input with autograd on.
//! 3. Backward through `phantom_loss = sum(recomputed_output * dy)`.
//! 4. Merge the inner backward's grad store into the main one.
//!
//! Memory: peak forward+backward is *per-block* instead of *per-stack*,
//! at the cost of one extra forward per checkpointed block during
//! backward. For the 102.7 M PhotonMamba this drops the 30 blocks ×
//! ~170 MB intermediates from "all alive at once" to "one block alive
//! at a time" — the difference between OOM and B=4+ at T=512.

use crate::{Module, Ops, Param};

pub struct CheckpointSegment<O: Ops> {
    pub saved_input: O::Tensor,
    pub boundary: O::Param,
    pub block_id: usize,
}

pub struct CheckpointState<O: Ops> {
    pub segments: Vec<CheckpointSegment<O>>,
}

impl<O: Ops> Default for CheckpointState<O> {
    fn default() -> Self {
        Self {
            segments: Vec::new(),
        }
    }
}

impl<O: Ops> CheckpointState<O> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

/// Forward through `block` with a checkpoint boundary at the output.
///
/// The raw forward output is wrapped into a fresh `Param` and the
/// original `Tensor` (along with its autograd tape) is dropped before
/// returning, so the block's intermediate activations no longer
/// occupy device memory. The caller passes a stable `block_id` that
/// the recompute callback in [`checkpoint_backward`] uses to look the
/// block up again.
pub fn forward_checkpointed<O, M>(
    ops: &O,
    block: &M,
    block_id: usize,
    input: &O::Tensor,
    state: &mut CheckpointState<O>,
) -> Result<O::Tensor, O::Error>
where
    O: Ops,
    M: Module<O>,
    O::Tensor: Clone,
    O::Param: Clone,
{
    let raw_output = block.forward(ops, input)?;
    let boundary = ops.param_from_tensor(&raw_output)?;
    // Save (input, boundary, block_id). `input.clone()` is a cheap
    // metadata clone on backends where Tensor is Arc-backed (true for
    // CandleTensor).
    state.segments.push(CheckpointSegment {
        saved_input: input.clone(),
        boundary: boundary.clone(),
        block_id,
    });
    // Hand back a Tensor view of the boundary param. Downstream ops
    // build their autograd tape against this Var, so the main backward
    // will deposit a gradient at the boundary's id for us to pick up.
    let out = boundary.as_tensor().clone();
    // `raw_output` drops here. Its autograd parents (the block's
    // intermediates) lose their last strong ref and get freed.
    drop(raw_output);
    let _ = boundary;
    Ok(out)
}

/// Propagate gradients backward through every recorded checkpoint
/// segment. Each segment's block is re-run via `recompute_block`,
/// which the caller supplies because pm-core doesn't know how the
/// `block_id` numbers map onto the user's model tree.
///
/// `grad_store` should be the result of the main `loss.backward()`.
/// On return it contains gradients for every original parameter that
/// the checkpointed segments touch.
pub fn checkpoint_backward<O, F>(
    ops: &O,
    state: CheckpointState<O>,
    grad_store: &mut O::GradStore,
    mut recompute_block: F,
) -> Result<(), O::Error>
where
    O: Ops,
    F: FnMut(&O, usize, &O::Tensor) -> Result<O::Tensor, O::Error>,
{
    for seg in state.segments.into_iter().rev() {
        let Some(dy) = ops.gradient(grad_store, &seg.boundary)? else {
            // Nothing flowed through this segment — skip it.
            continue;
        };
        // Recompute the segment forward (autograd is on, since the
        // block's own Params are still Vars).
        let recomp = recompute_block(ops, seg.block_id, &seg.saved_input)?;
        // Phantom scalar loss whose grad w.r.t. each Param is dy · ∂recomp/∂Param.
        let weighted = ops.mul(&recomp, &dy)?;
        let phantom = ops.sum_all(&weighted)?;
        let inner_grads = ops.backward(&phantom)?;
        ops.merge_grad_stores(grad_store, inner_grads)?;
    }
    Ok(())
}
