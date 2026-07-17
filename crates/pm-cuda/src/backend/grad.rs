//! Gradient store for `CudaBackend`.
//!
//! B4.3a: replaces the empty skeleton with a real `HashMap`-backed store.

use std::collections::HashMap;

use super::param::ParamId;
use super::CudaTensor;

/// Maps each trainable `ParamId` to its accumulated gradient tensor.
///
/// Returned by `CudaBackend::backward()` and consumed by the optimiser.
/// The gradient tensors have the same shape as the corresponding parameter.
pub struct CudaGradStore {
    pub(crate) grads: HashMap<ParamId, CudaTensor>,
}

impl CudaGradStore {
    /// Construct an empty store.
    pub(crate) fn new() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }
}
