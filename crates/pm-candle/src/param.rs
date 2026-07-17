//! Candle-backed trainable parameter.
//!
//! Wraps `candle_core::Var` (a `Tensor` with mutable storage tracked by
//! autograd) and exposes a `CandleTensor` view via `pm_core::Param`.
//!
//! `candle_core::Var::set` takes `&self` because the storage lives
//! behind an `Arc<RwLock<...>>` — all existing handles see updates in
//! place. We therefore cache the `CandleTensor` view once at
//! construction and never need to rebuild it on `assign`/`set`.

use crate::CandleTensor;
use pm_core::Param;

/// Trainable parameter held on a Candle device.
#[derive(Clone, Debug)]
pub struct CandleParam {
    pub(crate) var: candle_core::Var,
    /// `CandleTensor` view of the `Var`. Shares storage with `var`, so
    /// reads through `as_tensor()` reflect the current value even after
    /// the optimiser writes via `var.set(...)`.
    pub(crate) tensor_view: CandleTensor,
}

impl CandleParam {
    pub(crate) fn from_tensor(t: candle_core::Tensor) -> Result<Self, crate::Error> {
        let var = candle_core::Var::from_tensor(&t)?;
        let view = CandleTensor::new(var.as_tensor().clone())?;
        Ok(Self {
            var,
            tensor_view: view,
        })
    }

    /// Replace the parameter's tensor value in place (same shape required).
    pub(crate) fn assign(&self, value: &candle_core::Tensor) -> Result<(), crate::Error> {
        self.var.set(value)?;
        Ok(())
    }
}

impl Param for CandleParam {
    type Tensor = CandleTensor;

    fn as_tensor(&self) -> &Self::Tensor {
        &self.tensor_view
    }
}
