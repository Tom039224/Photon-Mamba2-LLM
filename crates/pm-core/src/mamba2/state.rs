//! Fixed-size recurrent state for `Mamba2Block::step` (memory-efficiency
//! plan, Phase C: O(1)-memory recurrent decode).
//!
//! `Mamba2Block::forward` re-derives everything from scratch for a
//! whole `(B, T, D)` sequence every call — the right shape for
//! teacher-forced training, but `O(T²)`-ish per generated token if
//! reused naively for autoregressive decoding (padded re-forward,
//! `pm-infer::Generator`). [`Mamba2State`] instead carries exactly the
//! two pieces of state Mamba2's SSD recurrence needs across time steps,
//! so `Mamba2Block::step` can advance one token at a fixed cost:
//!
//! - `ssm_state`: the running SSM hidden state (Mamba2 §2.1 Eq.
//!   `h_t = A h_{t-1} + B x_t`, specialised to Mamba2's scalar-per-head
//!   `A` and the SSD chunked-scan's discretisation — see
//!   `Mamba2Block::step`'s docstring for the exact derivation).
//! - `conv_window`: the trailing `d_conv - 1` pre-conv `xBC` columns,
//!   replacing `forward`'s `conv1d(padding = d_conv - 1)` + crop with
//!   an explicit rolling history buffer that produces the identical
//!   causal window one column at a time.
//!
//! Always `F32`, matching `Ops::ssd_scan`'s dtype invariant
//! (`mamba2::ssd_ops` module docs): the recurrence exponentiates a
//! per-step decay, which is exactly the computation that invariant
//! says must not run in `bf16`.

use crate::mamba2::block::Mamba2Config;
use crate::{Dtype, Ops};

/// Per-`Mamba2Block` recurrent state carried token-to-token by
/// [`Mamba2Block::step`](crate::mamba2::Mamba2Block::step).
pub struct Mamba2State<O: Ops> {
    /// Running SSM hidden state `h`, shape `(B, H, N, P)`.
    pub ssm_state: O::Tensor,
    /// Last `d_conv - 1` pre-conv `xBC` columns, shape
    /// `(B, xbc_dim, d_conv - 1)`.
    pub conv_window: O::Tensor,
}

impl<O: Ops> Mamba2State<O> {
    /// All-zero state for a fresh generation: `h_{-1} = 0` (no prior
    /// SSM history) and a zero conv history — the same implicit
    /// zero-padding `Mamba2Block::forward`'s causal
    /// `conv1d(padding = d_conv - 1)` uses at the first `d_conv - 1`
    /// positions of any sequence.
    pub fn zeros(ops: &O, cfg: &Mamba2Config, batch: usize) -> Result<Self, O::Error> {
        Ok(Self {
            ssm_state: ops.zeros(&[batch, cfg.n_heads, cfg.d_state, cfg.d_head], Dtype::F32)?,
            conv_window: ops.zeros(&[batch, cfg.xbc_dim(), cfg.d_conv - 1], Dtype::F32)?,
        })
    }
}
