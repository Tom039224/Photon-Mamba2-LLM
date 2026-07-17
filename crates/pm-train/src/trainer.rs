//! High-level training loop wrapper.
//!
//! The Trainer holds an optimiser and (optionally) a gradient-clipping
//! threshold. Each `step` does:
//!   1. `forward_fn(ops)` — produces a scalar loss
//!   2. `ops.backward(&loss)`
//!   3. (optional) `clip_grad_norm(...)` — F.5
//!   4. `optimiser.step(ops, &params, &grads)`
//!
//! The caller supplies `forward_fn`, which keeps the Trainer generic
//! over both the model architecture and the loss function.

use pm_core::Ops;

use crate::clip::clip_grad_norm;
use crate::optim::Optimizer;

pub struct Trainer<O: Ops, Opt: Optimizer<O>> {
    pub optimiser: Opt,
    /// Maximum global L2-norm. `None` disables clipping.
    pub max_grad_norm: Option<f32>,
    _ops_marker: std::marker::PhantomData<fn(&O)>,
}

/// One-step telemetry.
#[derive(Debug, Clone, Copy)]
pub struct StepReport {
    pub loss: f32,
    /// Pre-clip global gradient norm. `None` when clipping is disabled.
    pub grad_norm: Option<f32>,
}

impl<O: Ops, Opt: Optimizer<O>> Trainer<O, Opt> {
    #[must_use]
    pub fn new(optimiser: Opt) -> Self {
        Self {
            optimiser,
            max_grad_norm: None,
            _ops_marker: std::marker::PhantomData,
        }
    }

    /// Builder-style setter for the global gradient-norm cap.
    #[must_use]
    pub fn with_clip(mut self, max_norm: f32) -> Self {
        self.max_grad_norm = Some(max_norm);
        self
    }

    /// Run one training step. Returns a [`StepReport`] with the loss
    /// (and pre-clip gradient norm when clipping is enabled).
    pub fn step<F>(
        &mut self,
        ops: &O,
        params: &[&O::Param],
        forward_fn: F,
    ) -> Result<StepReport, O::Error>
    where
        F: FnOnce(&O) -> Result<O::Tensor, O::Error>,
    {
        let loss = forward_fn(ops)?;
        let loss_val = scalar_to_f32(ops, &loss)?;
        let mut grads = ops.backward(&loss)?;
        let grad_norm = match self.max_grad_norm {
            Some(max_norm) => Some(clip_grad_norm(ops, params, &mut grads, max_norm)?),
            None => None,
        };
        self.optimiser.step(ops, params, &grads)?;
        Ok(StepReport {
            loss: loss_val,
            grad_norm,
        })
    }

    /// Convenience: just the loss, no telemetry.
    pub fn step_loss<F>(
        &mut self,
        ops: &O,
        params: &[&O::Param],
        forward_fn: F,
    ) -> Result<f32, O::Error>
    where
        F: FnOnce(&O) -> Result<O::Tensor, O::Error>,
    {
        Ok(self.step(ops, params, forward_fn)?.loss)
    }

    /// Like [`step`], but the caller is responsible for both the
    /// forward AND the backward (returning the populated `GradStore`).
    /// Used by activation-checkpointed training where the backward
    /// involves an extra pass through `checkpoint_backward` before the
    /// optimiser runs.
    pub fn step_with_grads<F>(
        &mut self,
        ops: &O,
        params: &[&O::Param],
        forward_and_backward_fn: F,
    ) -> Result<StepReport, O::Error>
    where
        F: FnOnce(&O) -> Result<(O::Tensor, O::GradStore), O::Error>,
    {
        let (loss, mut grads) = forward_and_backward_fn(ops)?;
        let loss_val = scalar_to_f32(ops, &loss)?;
        let grad_norm = match self.max_grad_norm {
            Some(max_norm) => Some(crate::clip::clip_grad_norm(
                ops, params, &mut grads, max_norm,
            )?),
            None => None,
        };
        self.optimiser.step(ops, params, &grads)?;
        Ok(StepReport {
            loss: loss_val,
            grad_norm,
        })
    }
}

fn scalar_to_f32<O: Ops>(ops: &O, t: &O::Tensor) -> Result<f32, O::Error> {
    let v = ops.to_vec_f32(t)?;
    assert_eq!(v.len(), 1, "expected scalar loss, got {} elements", v.len());
    Ok(v[0])
}
