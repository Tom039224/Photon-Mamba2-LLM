//! Trainable parameter abstraction.
//!
//! Backends construct concrete [`Param`] values (e.g. `pm-candle::CandleParam`
//! wraps a `candle_core::Var`). Modules in `pm-core` hold parameters via
//! the trait, so the `pm-core` code stays free of backend-specific types
//! (CLAUDE.md invariant 1).
//!
//! Read access goes through [`Param::as_tensor`]; the optimizer applies
//! updates via [`Ops::sgd_step`](crate::Ops::sgd_step) on the param
//! directly.

use crate::Tensor;

/// A trainable parameter.
///
/// Neither `Send` nor `Sync` is required.  Training is single-threaded
/// (CLAUDE.md): `forward` → `backward` → optimizer-step never overlap on
/// different threads, so backends are free to use interior-mutable
/// storage (`UnsafeCell`) without exposing UB.  If multi-thread training
/// is added later (B4.4+), individual backends may opt in by switching
/// to `Mutex<CudaTensor>` (or similar) — that is a backend-local
/// concern, not a trait-level one.
pub trait Param {
    type Tensor: Tensor;

    /// Read-only view of the parameter's current value, suitable for
    /// passing to `Ops::matmul`, `Ops::add`, etc. inside `forward`.
    fn as_tensor(&self) -> &Self::Tensor;
}

/// Modules that own trainable [`Param`]s implement this so the Trainer
/// can collect them into a single flat list for the optimizer.
///
/// Uses an append-into-buffer pattern so the model tree is walked once
/// at training-init time, not once per step.
pub trait Parameterized<O: crate::Ops> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>);

    fn collect_params(&self) -> Vec<&O::Param> {
        let mut v = Vec::new();
        self.append_params(&mut v);
        v
    }
}
