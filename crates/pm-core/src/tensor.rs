//! Backend-agnostic tensor trait.
//!
//! Concrete tensor structs live in backend crates (`pm-candle::CandleTensor`,
//! `pm-cuda::CudaTensor`, etc.). `pm-core` only sees them as `impl Tensor`.

use crate::Dtype;

/// Read-only view into a tensor's metadata.
///
/// Backends implement this for their tensor type. Mutation and arithmetic
/// go through `Ops` instead, so `Tensor` itself stays cheap to pass around
/// and easy to mock in tests.
pub trait Tensor: Send + Sync {
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> Dtype;

    #[inline]
    fn rank(&self) -> usize {
        self.shape().len()
    }

    #[inline]
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }
}
