use crate::dtype::from_candle;
use pm_core::{Dtype, Tensor};

/// Backend tensor: a thin newtype around `candle_core::Tensor`.
///
/// We expose `inner()` to ops implementations within this crate only;
/// downstream code never sees the inner type.
#[derive(Clone, Debug)]
pub struct CandleTensor {
    inner: candle_core::Tensor,
    dtype: Dtype,
    shape: Vec<usize>,
}

impl CandleTensor {
    pub(crate) fn new(inner: candle_core::Tensor) -> Result<Self, crate::Error> {
        let dtype = from_candle(inner.dtype())?;
        let shape = inner.dims().to_vec();
        Ok(Self {
            inner,
            dtype,
            shape,
        })
    }

    pub(crate) fn inner(&self) -> &candle_core::Tensor {
        &self.inner
    }

    /// Consume self and return the underlying Candle tensor. Used by
    /// `set_gradient` to hand ownership of a fresh tensor into the
    /// `GradStore`.
    pub(crate) fn into_inner(self) -> candle_core::Tensor {
        self.inner
    }
}

impl Tensor for CandleTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> Dtype {
        self.dtype
    }
}
