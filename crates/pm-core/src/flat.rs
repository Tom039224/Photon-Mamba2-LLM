//! Flat Mamba2 language model — Phase D.2 baseline.
//!
//! Non-hierarchical counterpart of [`PhotonMamba`]: a single residual stack
//! of Mamba2 blocks (`ContextEncoder`) with a tied token embedding + lm_head,
//! and nothing else. Used as the "same param count, no hierarchy" baseline
//! to measure the value of PHOTON's hierarchical structure (PLAN.md Phase D.2).
//!
//! ## Architecture
//!
//! ```text
//! ids (B, T)
//!   └─ TokenEmbedding   →  (B, T, D)        [tied lm_head weight]
//!        └─ ContextEncoder (N Mamba2Blocks)  →  (B, T, D)   Mamba2 §1 standard LM stack
//!             └─ lm_head_logits              →  (B, T, V)   hidden @ embed_wt.T
//! ```
//!
//! The lm_head has no separate final RMSNorm. This mirrors `PhotonMamba`'s
//! training-path convention: `predicted[0]` enters the tied lm_head without
//! a final norm (PLAN.md Phase D.2 fairness note: keeping the two models
//! structurally symmetric at the lm_head boundary). See `docs/deviations.md`
//! for the deliberate deviation from the norm-before-lm_head convention used
//! in some Mamba2 reference implementations.
//!
//! ## Forward paths
//!
//! | Method | Returns | Use |
//! |--------|---------|-----|
//! | `forward_hidden` | `(B,T,D)` hidden | training via `fused_cross_entropy` |
//! | `forward` | `(B,T,V)` logits | eval / HellaSwag scoring |
//! | `forward_checkpointed_hidden` | `((B,T,D), CheckpointState)` | ckpt training |
//! | `recompute_block` | `(B,T,D)` | checkpoint_backward callback |
//!
//! `decode` (prefill / commit / predict_logits) is **not** implemented — the
//! O(1) recurrent decode path is PHOTON-specific (D.2 scope boundary).
//!
//! ## Parameter layout (via `Parameterized`)
//!
//! `embed.weight` first, then `trunk.layers[0..N]` in forward order. This
//! matches `PhotonMamba`'s convention (embed → encoder → decoder), keeping
//! safetensors checkpoints structurally comparable.
//!
//! ## Paper references
//!
//! - Mamba2 §1 (standard language-model construction: residual Mamba2 stack)
//! - PLAN.md Phase D.2 (comparison baseline design)

use crate::checkpoint::CheckpointState;
use crate::photon::{ContextEncoder, TokenEmbedding};
use crate::{Dtype, Ops, Parameterized};

/// Flat (non-hierarchical) Mamba2 LM — Phase D.2 PHOTON-comparison baseline.
///
/// See module-level documentation for architecture details and usage.
pub struct FlatMamba<O: Ops> {
    pub embed: TokenEmbedding<O>,
    /// The single residual stack of N Mamba2 blocks.
    pub trunk: ContextEncoder<O>,
    /// Forward-pass compute dtype. `F32` by default. Set via
    /// [`with_compute_dtype`](FlatMamba::with_compute_dtype).
    pub compute_dtype: Dtype,
}

impl<O: Ops> FlatMamba<O> {
    /// Construct from a pre-built embedding and encoder trunk.
    /// `compute_dtype` defaults to `Dtype::F32`.
    pub fn new(embed: TokenEmbedding<O>, trunk: ContextEncoder<O>) -> Self {
        Self {
            embed,
            trunk,
            compute_dtype: Dtype::F32,
        }
    }

    /// Opt into a non-default forward-pass compute dtype (e.g. `BF16` for
    /// half-precision activation tape). Matches `PhotonMamba::with_compute_dtype`.
    #[must_use]
    pub fn with_compute_dtype(mut self, dtype: Dtype) -> Self {
        self.compute_dtype = dtype;
        self
    }

    /// Run the full forward pass up to (but not including) the lm_head.
    ///
    /// Returns the trunk's output `(B, T, D)`. The training loop calls this
    /// and then runs `Ops::fused_cross_entropy` on the result, avoiding the
    /// full `(B, T, V)` logit materialisation.
    ///
    /// Mamba2 §1 (residual Mamba2 stack forward). PLAN.md D.2a.
    pub fn forward_hidden(&self, ops: &O, ids: &O::Tensor) -> Result<O::Tensor, O::Error> {
        use crate::Module;
        let x_native = self.embed.forward(ops, ids)?;
        let x = ops.to_dtype(&x_native, self.compute_dtype)?;
        self.trunk.forward(ops, &x)
    }

    /// Full forward pass including the tied lm_head matmul.
    ///
    /// Returns `(B, T, V)` logits. Used by HellaSwag evaluation (log-prob
    /// scoring) and any other eval path that needs the full vocabulary
    /// distribution.
    ///
    /// No separate final RMSNorm is applied before the lm_head — see the
    /// module-level "lm_head design" note.
    pub fn forward(&self, ops: &O, ids: &O::Tensor) -> Result<O::Tensor, O::Error> {
        let hidden = self.forward_hidden(ops, ids)?;
        self.embed.lm_head_logits(ops, &hidden)
    }
}

impl<O: Ops> FlatMamba<O>
where
    O::Tensor: Clone,
    O::Param: Clone,
{
    /// Same as [`forward_hidden`](Self::forward_hidden) but each Mamba2Block
    /// is wrapped in a checkpoint segment. Returns the trunk output **plus**
    /// a `CheckpointState` that the caller must hand to
    /// `pm_core::checkpoint_backward` after computing the loss gradient.
    ///
    /// Mirrors `PhotonMamba::forward_checkpointed_no_lm_head` for structural
    /// parity. PLAN.md D.2a, activation checkpointing path.
    pub fn forward_checkpointed_hidden(
        &self,
        ops: &O,
        ids: &O::Tensor,
    ) -> Result<(O::Tensor, CheckpointState<O>), O::Error> {
        use crate::Module;
        let mut cp = CheckpointState::new();

        let x_native = self.embed.forward(ops, ids)?;
        let x = ops.to_dtype(&x_native, self.compute_dtype)?;

        // Delegate to ContextEncoder::forward_checkpointed; block_id_offset=0
        // because this model has a single flat stack (no hierarchy, so ids
        // are simply 0..N-1).
        let hidden = self.trunk.forward_checkpointed(ops, &x, &mut cp, 0)?;
        Ok((hidden, cp))
    }

    /// Recompute the `block_id`-th Mamba2Block forward during
    /// `checkpoint_backward`. Panics if `block_id` is out of range (same
    /// contract as `PhotonMamba::recompute_block`).
    ///
    /// `block_id` matches the numbering used by `forward_checkpointed_hidden`:
    /// blocks are numbered `0..trunk.n_layers()` in forward order.
    pub fn recompute_block(
        &self,
        ops: &O,
        block_id: usize,
        input: &O::Tensor,
    ) -> Result<O::Tensor, O::Error> {
        use crate::Module;
        let n = self.trunk.n_layers();
        if block_id < n {
            return self.trunk.layers[block_id].forward(ops, input);
        }
        panic!(
            "FlatMamba::recompute_block: block_id {block_id} out of range \
             (trunk has {n} layers)"
        )
    }
}

impl<O: Ops> Parameterized<O> for FlatMamba<O> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>) {
        // embed first, then trunk — mirrors PhotonMamba::append_params order
        // (embed → encoder → decoder) so checkpoint files are structurally
        // comparable across model types.
        self.embed.append_params(out);
        self.trunk.append_params(out);
    }
}
