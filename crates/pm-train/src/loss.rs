//! Losses for PHOTON training.
//!
//! - [`cross_entropy_loss`] — token-level next-token CE (PHOTON §2.3
//!   Eq. (10), `L_token`).
//! - [`fused_cross_entropy_injected`] — memory-bounded version of the
//!   above: same value/gradient, never materialises `(B,T,vocab)`.
//! - [`recursive_consistency_loss`] — F.2, cosine distance between the
//!   decoder's level-l prediction and the encoder's level-l target
//!   (PHOTON §2.3 Eq. (11), `L_rec`).
//! - [`photon_loss`] — F.3, the full PHOTON objective (PHOTON §2.3
//!   Eq. (9)): `L = L_token + α · L_rec`. Materialises `(B,T,vocab)` —
//!   reference/testing use only, **not** the production training path.
//! - [`fused_photon_loss_injected`] — Phase D.1: the fused counterpart
//!   of `photon_loss` actually wired into `pm-cli::train_cmd` — same
//!   value/gradient as `photon_loss`, never materialises
//!   `(B,T,vocab)`, and is exactly zero-cost when `α = 0`. Returns a
//!   [`PhotonLossReport`] rather than a bare tensor: Phase B'.3 needs
//!   the CE (`L_token`) and `L_rec` components separately loggable —
//!   they used to be indistinguishable inside the single α-weighted
//!   `total` (`docs/perf-log.md` 2026-07-05, D.2b's α=0.3 run had no
//!   way to recover the CE-alone trajectory from the log).
//!
//! All losses are backend-agnostic (`O: Ops`) and return scalar tensors
//! (or, for `fused_photon_loss_injected`, a small tensor-carrying report
//! struct) that the optimiser can `.backward()`.

use pm_core::{Dtype, Ops, Param, Tensor};

/// Token-level cross-entropy.
///
/// `logits`: `(B, T, V)` float. `targets`: `(B, T)` i64 — class index per
/// position. Returns a scalar tensor (mean over all positions).
///
/// Upcasts `logits` to `F32` before the softmax reduction (a no-op when
/// `logits` is already `F32`). Under the bf16 mixed-precision compute
/// path (memory-efficiency plan Phase A2) `logits` may be `bf16` — the
/// `log_softmax` max-subtract/exp/sum/log chain needs fp32 headroom,
/// and the loss itself must stay fp32 (CLAUDE.md invariant #3).
pub fn cross_entropy_loss<O: Ops>(
    ops: &O,
    logits: &O::Tensor,
    targets: &O::Tensor,
) -> Result<O::Tensor, O::Error> {
    let l_shape = logits.shape();
    assert_eq!(l_shape.len(), 3, "cross_entropy: logits must be (B, T, V)");
    let (b, t, _v) = (l_shape[0], l_shape[1], l_shape[2]);

    let t_shape = targets.shape();
    assert_eq!(
        t_shape,
        &[b, t],
        "cross_entropy: targets must be (B, T) matching logits"
    );

    let logits_f32 = ops.to_dtype(logits, Dtype::F32)?;
    // log p(token | context) — shape (B, T, V).
    let log_p = ops.log_softmax(&logits_f32, 2)?;

    // Gather along last dim: pick log_p[..., targets[..,]].
    // Candle expects index tensor to have the same rank as input, with
    // the gathered dim's size = output's gathered-dim size. Here we want
    // output shape (B, T, 1), so reshape targets to (B, T, 1).
    let targets_3d = ops.reshape(targets, &[b, t, 1])?;
    let picked = ops.gather(&log_p, &targets_3d, 2)?; // (B, T, 1)

    // NLL = -mean(picked).
    let neg = ops.neg(&picked)?;
    ops.mean_all(&neg)
}

/// Memory-bounded next-token cross-entropy: computes the loss and injects
/// its gradient into the real autograd graph **without** ever
/// materialising the full `(B, T, vocab)` logits (memory-efficiency
/// plan: fused/tiled cross-entropy). Use in place of
/// `cross_entropy_loss(ops, &lm_head_logits(hidden), targets)` — the
/// caller must have used `PhotonMamba::forward_no_lm_head` /
/// `forward_checkpointed_no_lm_head` (not `forward`/`forward_checkpointed`)
/// to get `hidden` in the first place, or the `(B, T, vocab)` tensor this
/// function avoids would already have been materialised upstream.
///
/// `hidden`: the decoder's pre-lm_head output (`predicted[0]`), still
/// carrying its real autograd ancestry (the whole encoder/decoder).
/// `embed_weight`: the tied embedding table (`TokenEmbedding::weight`),
/// used for both input-embedding lookup *and* the lm_head — its total
/// gradient has contributions from both. `tile_rows`: row-tile width
/// for `Ops::fused_cross_entropy`'s internal tiling (memory/host-
/// roundtrip-count trade-off).
///
/// Returns `(loss, grads)`, ready to hand to `Trainer::step_with_grads`
/// (optionally after merging in `checkpoint_backward`'s contributions,
/// same as the pre-existing checkpointed path did with the old
/// `ops.backward(&cross_entropy_loss(...))`).
///
/// ## Why this is correct: the phantom-loss trick
///
/// `Ops::fused_cross_entropy` returns `grad_hidden = ∂loss/∂hidden` and
/// `grad_table = ∂loss/∂embed_weight` (holding the other fixed) as
/// **detached** tensors — values with no autograd ancestry (see
/// `Ops::detach`'s doc comment for why that is load-bearing for memory,
/// not just correctness). To route them back through the real graph we
/// build a scalar
///
/// ```text
/// phantom = sum_all(hidden ⊙ grad_hidden) + sum_all(embed_weight ⊙ grad_table)
/// ```
///
/// and run `Ops::backward` on it once. For any parameter `θ` upstream of
/// `hidden` (any Mamba2 block, chunker, converter, …), `∂phantom/∂θ =
/// grad_hidden · ∂hidden/∂θ = (∂loss/∂hidden) · ∂hidden/∂θ = ∂loss/∂θ`
/// by the chain rule — exactly the gradient a plain
/// `cross_entropy_loss(lm_head_logits(hidden), targets).backward()`
/// would have produced. For `θ = embed_weight` specifically, which is
/// reachable from `phantom` via *two* paths (the input-embedding lookup
/// inside `hidden`'s ancestry, and the explicit second `sum_all` term),
/// the backward pass sums both contributions automatically (multi-path
/// gradient accumulation is a basic property of both backends'
/// `Ops::backward`) — reproducing the untied-then-summed gradient a
/// tied embedding table needs. This is the same "detach + phantom loss"
/// mechanism `pm-core::checkpoint::checkpoint_backward` already uses to
/// reconnect a recomputed segment's gradient to its boundary `Param`.
pub fn fused_cross_entropy_injected<O: Ops>(
    ops: &O,
    hidden: &O::Tensor,
    embed_weight: &O::Param,
    targets: &O::Tensor,
    tile_rows: usize,
) -> Result<(O::Tensor, O::GradStore), O::Error> {
    let table = embed_weight.as_tensor();
    let (loss, grad_hidden, grad_table) =
        ops.fused_cross_entropy(hidden, table, targets, tile_rows)?;

    let hidden_term = ops.sum_all(&ops.mul(hidden, &grad_hidden)?)?;
    let table_term = ops.sum_all(&ops.mul(table, &grad_table)?)?;
    let phantom = ops.add(&hidden_term, &table_term)?;

    let grads = ops.backward(&phantom)?;
    Ok((loss, grads))
}

/// Cosine distance per token, averaged over all positions and levels.
///
/// `predicted[l]` and `targets[l]` must have the same shape `(B, T_l, D_l)`.
/// `1 - cos_sim` along the feature dim, then mean-reduce.
///
/// PHOTON §2.3 Eq. (11) (`eq:recursive-loss`):
/// `L_rec = Σ_{l=1}^{L} Σ_{g=1}^{M_l} D(X̂^{(l-1)}_{I_g^{(l)}}, X^{(l-1)}_{I_g^{(l)}})`,
/// with `D` the cosine distance, "computed for each position and
/// averaged" (paper text right after the equation) — hence the
/// per-level `mean_all` below rather than a sum, and the final
/// `1/n_levels` so multi-level configs stay comparable in magnitude to
/// the paper's single implicit average over `l`.
///
/// `targets[l]` (`X^{(l-1)}`, the real encoder output) is **not**
/// stop-gradiented: this mirrors [`photon_loss`], the pre-existing
/// (F.2/F.3) reference this function was written for, which never
/// detaches its `encoded_targets` argument either. `docs/deviations.md`
/// P.4 fixes *which* tensor is the target (the encoder's per-level
/// *output*, not its input) but is silent on stop-gradient; see the
/// addendum there for why this implementation keeps both sides
/// differentiable.
pub fn recursive_consistency_loss<O: Ops>(
    ops: &O,
    predicted: &[O::Tensor],
    targets: &[O::Tensor],
) -> Result<O::Tensor, O::Error> {
    assert_eq!(
        predicted.len(),
        targets.len(),
        "consistency: predicted and targets must have same number of levels"
    );
    assert!(
        !predicted.is_empty(),
        "consistency: at least 1 level required"
    );

    let mut total: Option<O::Tensor> = None;
    let n_levels = predicted.len();
    for (p, q) in predicted.iter().zip(targets.iter()) {
        assert_eq!(
            p.shape(),
            q.shape(),
            "consistency: level shape mismatch ({:?} vs {:?})",
            p.shape(),
            q.shape()
        );
        // numerator = sum(p * q) over feature dim
        let dot = ops.mul(p, q)?;
        let dot_shape = dot.shape().to_vec();
        let d = dot_shape[dot_shape.len() - 1];
        let dot_sum = sum_last_dim(ops, &dot, &dot_shape, d)?;

        let p_sq = ops.mul(p, p)?;
        let q_sq = ops.mul(q, q)?;
        let p_norm_sq = sum_last_dim(ops, &p_sq, &dot_shape, d)?;
        let q_norm_sq = sum_last_dim(ops, &q_sq, &dot_shape, d)?;
        let p_norm = ops.sqrt(&ops.add_scalar(&p_norm_sq, 1e-12)?)?;
        let q_norm = ops.sqrt(&ops.add_scalar(&q_norm_sq, 1e-12)?)?;
        let denom = ops.mul(&p_norm, &q_norm)?;

        let cos = ops.div(&dot_sum, &denom)?;
        let dist = ops.mul_scalar(&ops.add_scalar(&ops.neg(&cos)?, 1.0)?, 1.0)?;
        let level_loss = ops.mean_all(&dist)?;
        total = Some(match total {
            Some(t) => ops.add(&t, &level_loss)?,
            None => level_loss,
        });
    }
    let sum = total.expect("at least one level");
    // Normalise by number of levels so the magnitude is comparable across configs.
    ops.mul_scalar(&sum, 1.0 / n_levels as f32)
}

/// Sum over the trailing axis: `(..., D) -> (...)` via `(..., D) @ ones((D, 1))`.
/// We need this because `Ops` doesn't (yet) expose `sum(dim)`.
fn sum_last_dim<O: Ops>(
    ops: &O,
    x: &O::Tensor,
    shape: &[usize],
    d: usize,
) -> Result<O::Tensor, O::Error> {
    let ones = ops.from_slice_f32(&vec![1.0_f32; d], &[d, 1])?;
    let y = ops.matmul(x, &ones)?; // (..., 1)
                                   // Drop trailing 1 by reshape.
    let out_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
    ops.reshape(&y, &out_shape)
}

/// PHOTON total loss (PHOTON §2.3 Eq. (9)): `L = L_token + α · L_rec`.
///
/// `alpha` is `0.0` in the paper's own main results (isolates gains
/// from the hierarchy/bounded-decoding architecture); `α≈0.3` is the
/// paper's appendix ablation optimum for downstream zero-shot accuracy
/// (`Papers/Photon/main.tex`, "Ablations over Strength of Recursive
/// Loss"; Phase D.1 ablation, `PLAN.md`).
///
/// Materialises `logits` (`(B,T,vocab)`) via `cross_entropy_loss` —
/// this is the *reference* implementation (used by tests to validate
/// [`fused_photon_loss_injected`]'s gradients), not the training-loop
/// path: at `seq_len = 2048` the `(B,T,vocab)` tensor alone is the
/// dominant activation-memory cost (memory-efficiency plan,
/// `docs/perf-log.md` 2026-07-03). `pm-cli::train_cmd` calls
/// `fused_photon_loss_injected` instead.
pub fn photon_loss<O: Ops>(
    ops: &O,
    logits: &O::Tensor,
    targets: &O::Tensor,
    predicted: &[O::Tensor],
    encoded_targets: &[O::Tensor],
    alpha: f32,
) -> Result<O::Tensor, O::Error> {
    let ce = cross_entropy_loss(ops, logits, targets)?;
    if alpha == 0.0 || predicted.is_empty() {
        return Ok(ce);
    }
    let rec = recursive_consistency_loss(ops, predicted, encoded_targets)?;
    let weighted = ops.mul_scalar(&rec, alpha)?;
    ops.add(&ce, &weighted)
}

/// `ce`/`lrec` component values behind [`PhotonLossReport::components`]
/// (Phase B'.3, `docs/perf-log.md` 2026-07-05's CE/L_rec log-separation
/// TODO). Both are the *raw*, unweighted values read straight off the
/// tensors `PhotonLossReport::total` is built from via `Ops::to_vec_f32`
/// — plain `f32`, not tensors, since they exist purely for logging and
/// are never fed back into the graph. `total ≈ ce + alpha * lrec`
/// (PHOTON §2.3 Eq. (9)).
#[derive(Debug, Clone, Copy)]
pub struct LossComponents {
    /// `L_token` (Eq. (10)) — the next-token cross-entropy value.
    pub ce: f32,
    /// `L_rec` (Eq. (11)) — the *unweighted* recursive-consistency
    /// value (i.e. not yet multiplied by `alpha`).
    pub lrec: f32,
}

/// [`fused_photon_loss_injected`]'s return value: the backprop-ready
/// scalar (`total`, identical to what this function returned before
/// Phase B'.3 introduced this struct) plus, when `alpha > 0`, the
/// individual CE/`L_rec` values needed to log them separately (see
/// [`LossComponents`]).
///
/// `components` is `None` exactly on the zero-cost `alpha == 0.0` (or
/// empty `predicted`) path: no recursive-consistency term was computed,
/// so `total` already *is* CE and there is nothing extra to report.
pub struct PhotonLossReport<O: Ops> {
    pub total: O::Tensor,
    pub components: Option<LossComponents>,
}

/// Fused/tiled counterpart of [`photon_loss`] — Phase D.1: wires the
/// recursive-consistency loss (PHOTON §2.3 Eq. (11)) into the same
/// `(B,T,vocab)`-avoiding path [`fused_cross_entropy_injected`]
/// already provides for the next-token term (Eq. (10)), so the total
/// objective (Eq. (9)) can be trained at `seq_len = 2048` without
/// regressing to the naive-CE memory profile.
///
/// - `hidden`, `embed_weight`, `ce_targets`, `tile_rows`: exactly
///   [`fused_cross_entropy_injected`]'s arguments.
/// - `predicted`, `encoded_targets`: exactly [`recursive_consistency_loss`]'s
///   arguments. Callers pass `hidden_output.predicted` and
///   `&hidden_output.encoded.encoded[..hidden_output.predicted.len()]`
///   (`pm_core::model::PhotonHiddenOutput`); `hidden` must be
///   `&predicted[0]` (asserted below).
/// - `alpha`: Eq. (9)'s α. `0.0` is the paper's own main-result value.
///
/// ## Zero-cost at α = 0 (memory-efficiency plan D.1 requirement)
///
/// `alpha == 0.0` (or an empty `predicted`) calls
/// `fused_cross_entropy_injected(ops, hidden, embed_weight, ce_targets,
/// tile_rows)` **directly** — literally the same function call, not a
/// re-derivation of it — and wraps its `(loss, grads)` pair as
/// `(PhotonLossReport { total: loss, components: None }, grads)`. That
/// wrapping is a plain struct construction around the exact values
/// `fused_cross_entropy_injected` produced: no new tensor, no extra
/// device op, no `Ops::to_vec_f32` — so the pre-D.1 training graph
/// (step time, peak memory, every intermediate value) is reproduced
/// bit-for-bit, and `PhotonLossReport::components` being `None` costs
/// nothing to construct (Phase B'.3's CE/L_rec split, see
/// [`LossComponents`], is entirely paid for by the α>0 branch below).
/// `predicted`/`encoded_targets` are never read on this path. Callers
/// may pass them unconditionally regardless of `alpha` (as
/// `pm-cli::train_cmd` does): `PhotonHiddenOutput::predicted` and
/// `::encoded` are ordinary fields of the model's forward output,
/// already materialised as byproducts of computing `hidden` itself —
/// slicing them costs nothing (no clone of tensor storage, no device
/// op) whether or not this function goes on to use them.
///
/// ## One backward call, not two
///
/// The α>0 path folds *both* loss terms into one phantom scalar —
/// `combined = phantom_ce + α·L_rec` (see [`fused_cross_entropy_injected`]'s
/// doc comment for what `phantom_ce` is and why it reproduces `∂CE/∂θ`)
/// — before the *one* `Ops::backward` call. Multi-path gradient
/// accumulation then sums the CE and recursive-consistency
/// contributions for every parameter both terms share as an ancestor.
/// In this L=2 hierarchy that is *every* trainable parameter:
/// `predicted[0]`'s own forward ancestry already runs through every
/// encoder level (`PhotonMamba::forward_no_lm_head` builds
/// `predicted[0]` from `converter(encoded.encoded[1])`, and
/// `encoded.encoded[1]` from `encoder_level1(chunker(encoded.encoded[0]))`
/// — see `pm-core::photon::hierarchical_decoder`), so CE's gradient
/// already reaches encoder params without any help from `L_rec`; see
/// `nonzero_alpha_changes_gradient_at_encoder_only_params` in
/// `pm-train/tests/recursive_consistency_wiring.rs`, which checks the
/// α-dependent *delta* at an encoder-only parameter to confirm
/// `L_rec`'s contribution is real, not just CE's.
///
/// A single combined backward also matters for activation
/// checkpointing (`PhotonMamba::forward_checkpointed_no_lm_head`): the
/// returned `grads` already has the combined (CE + α·L_rec) value at
/// every checkpoint-boundary `Param`, so the caller's usual *one*
/// `pm_core::checkpoint_backward` call correctly recomputes every
/// touched block exactly once. Two independent `backward` calls (one
/// per loss term) would instead double the recompute cost and require
/// hand-accumulating two partial grad stores.
pub fn fused_photon_loss_injected<O: Ops>(
    ops: &O,
    hidden: &O::Tensor,
    embed_weight: &O::Param,
    ce_targets: &O::Tensor,
    tile_rows: usize,
    predicted: &[O::Tensor],
    encoded_targets: &[O::Tensor],
    alpha: f32,
) -> Result<(PhotonLossReport<O>, O::GradStore), O::Error> {
    if alpha == 0.0 || predicted.is_empty() {
        let (loss, grads) =
            fused_cross_entropy_injected(ops, hidden, embed_weight, ce_targets, tile_rows)?;
        return Ok((
            PhotonLossReport {
                total: loss,
                components: None,
            },
            grads,
        ));
    }
    assert_eq!(
        predicted.len(),
        encoded_targets.len(),
        "fused_photon_loss_injected: predicted and encoded_targets must have \
         the same number of levels"
    );
    assert_eq!(
        predicted[0].shape(),
        hidden.shape(),
        "fused_photon_loss_injected: predicted[0] must be the same tensor as \
         `hidden` (PhotonHiddenOutput::predicted[0], see PhotonMamba::forward_no_lm_head)"
    );

    let table = embed_weight.as_tensor();
    let (loss, grad_hidden, grad_table) =
        ops.fused_cross_entropy(hidden, table, ce_targets, tile_rows)?;

    let hidden_term = ops.sum_all(&ops.mul(hidden, &grad_hidden)?)?;
    let table_term = ops.sum_all(&ops.mul(table, &grad_table)?)?;
    let ce_phantom = ops.add(&hidden_term, &table_term)?;

    let rec = recursive_consistency_loss(ops, predicted, encoded_targets)?;
    let weighted_rec = ops.mul_scalar(&rec, alpha)?;

    let combined = ops.add(&ce_phantom, &weighted_rec)?;
    let grads = ops.backward(&combined)?;

    // Reported value: CE (already the true loss value — `fused_cross_
    // entropy`'s whole point is computing it without an autograd
    // graph) plus α·L_rec's own value. `total_loss` itself is never
    // backward()'d — only read via `Ops::to_vec_f32` for logging — so
    // it carrying `weighted_rec`'s ancestry (unlike `loss`) is harmless.
    let total_loss = ops.add(&loss, &weighted_rec)?;

    // Phase B'.3: split components for logging, gated by the same
    // `alpha > 0` branch this whole block already lives behind — the
    // two extra `Ops::to_vec_f32` device syncs below are the *only*
    // place this function ever performs one on the α>0 path beyond
    // what `fused_cross_entropy_injected` itself already required, and
    // they never happen at all on the α=0 path above.
    let ce = to_scalar_f32(ops, &loss)?;
    let lrec = to_scalar_f32(ops, &rec)?;

    Ok((
        PhotonLossReport {
            total: total_loss,
            components: Some(LossComponents { ce, lrec }),
        },
        grads,
    ))
}

/// Read a scalar tensor back to a plain `f32`. Used only by
/// [`fused_photon_loss_injected`]'s `alpha > 0` branch to populate
/// [`PhotonLossReport::components`] — the device sync this performs
/// (`Ops::to_vec_f32`) is exactly the "extra GPU sync ... only when
/// alpha>0" cost documented there.
fn to_scalar_f32<O: Ops>(ops: &O, t: &O::Tensor) -> Result<f32, O::Error> {
    let v = ops.to_vec_f32(t)?;
    // assert (not debug_assert): in release this would otherwise silently
    // return v[0] for a mis-shaped multi-element tensor. Matches
    // `trainer.rs::scalar_to_f32`'s strictness.
    assert_eq!(
        v.len(),
        1,
        "to_scalar_f32: expected a scalar tensor, got {} elements",
        v.len()
    );
    Ok(v[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use pm_candle::CandleBackend;

    /// PHOTON §2.3, the sentence right after Eq. (11): `D` is the
    /// cosine distance `(1 - Cosine Similarity)`, "computed for each
    /// position and averaged". Hand-computed reference:
    ///
    /// Level 0 (B=1,T=2,D=2): p=[[1,0],[0,1]], q=[[1,0],[1,0]].
    ///   pos 0: cos([1,0],[1,0]) = 1 -> dist = 0
    ///   pos 1: cos([0,1],[1,0]) = 0 -> dist = 1
    ///   level-0 mean = (0 + 1) / 2 = 0.5
    /// Level 1 (B=1,T=1,D=3): p=q=[1,1,1] (identical) -> cos=1 -> dist=0
    ///   level-1 mean = 0.0
    /// total = (0.5 + 0.0) / 2 levels = 0.25
    #[test]
    fn recursive_consistency_loss_matches_hand_computed_cosine_distance() {
        let bk = CandleBackend::new_cpu();
        let p0 = bk
            .from_slice_f32(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2])
            .unwrap();
        let q0 = bk
            .from_slice_f32(&[1.0, 0.0, 1.0, 0.0], &[1, 2, 2])
            .unwrap();
        let p1 = bk.from_slice_f32(&[1.0, 1.0, 1.0], &[1, 1, 3]).unwrap();
        let q1 = bk.from_slice_f32(&[1.0, 1.0, 1.0], &[1, 1, 3]).unwrap();

        let loss = recursive_consistency_loss(&bk, &[p0, p1], &[q0, q1]).unwrap();
        let v = bk.to_vec_f32(&loss).unwrap();
        assert_eq!(v.len(), 1);
        assert!((v[0] - 0.25).abs() < 1e-5, "expected 0.25, got {}", v[0]);
    }

    /// Degenerate case folded into the same hand-computed check: an
    /// exactly-matching predicted/target pair must contribute zero
    /// distance (cos_sim = 1 exactly, not just approximately, since
    /// both operands are bit-identical).
    #[test]
    fn recursive_consistency_loss_is_zero_when_predicted_equals_target() {
        let bk = CandleBackend::new_cpu();
        let p = bk
            .from_slice_f32(&[0.3, -1.2, 4.0, 0.1, 2.0, -3.0], &[1, 2, 3])
            .unwrap();
        let loss = recursive_consistency_loss(&bk, &[p.clone()], &[p]).unwrap();
        let v = bk.to_vec_f32(&loss).unwrap();
        assert!(
            v[0].abs() < 1e-5,
            "identical predicted/target should give ~0 distance, got {}",
            v[0]
        );
    }
}
