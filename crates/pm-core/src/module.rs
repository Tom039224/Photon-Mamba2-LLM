//! Backend-agnostic module trait.
//!
//! Every PhotonMamba layer (`Mamba2Block`, `ContextEncoder`, etc.)
//! implements this trait. `O: Ops` is the only way these layers reach
//! the backend.

use crate::Ops;

pub trait Module<O: Ops> {
    fn forward(&self, ops: &O, input: &O::Tensor) -> Result<O::Tensor, O::Error>;
}
