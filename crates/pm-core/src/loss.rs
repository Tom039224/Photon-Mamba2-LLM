//! Backend-agnostic reference tiling for [`Ops::fused_cross_entropy`].
//!
//! Memory-efficiency plan — fused/tiled cross-entropy. Mirrors the
//! project's established pattern for "big op with a pure-`Ops` default"
//! (`pm-core::mamba2::ssd_scan_ops_default` is the precedent `ssd_scan`
//! follows): backends implement `Ops::fused_cross_entropy` by delegating
//! to [`fused_cross_entropy_tiled`] here unless/until they need a true
//! fused kernel (PLAN.md Phase 2.5 / M.3).
//!
//! ## Algorithm
//!
//! For row-tile `h_tile: (rows, D)` (a slice of the flattened `(N, D)`
//! hidden state, `N = B·T`) and `w: (V, D)` the tied embedding table:
//!
//! ```text
//! logits_tile = h_tile @ w^T                          (rows, V)
//! log_p_tile  = log_softmax(logits_tile, dim=-1)       (rows, V)
//! nll_tile    = -sum(log_p_tile[row, target[row]])     scalar, accumulated
//!
//! softmax_tile      = exp(log_p_tile)                              (rows, V)
//! grad_logits_tile  = softmax_tile - one_hot(target_tile, V)        (rows, V)
//! grad_h_tile       = grad_logits_tile @ w                          (rows, D)
//! grad_w           += grad_logits_tile^T @ h_tile                   (V, D), accumulated
//! ```
//!
//! `grad_logits_tile = softmax(logits) - one_hot(target)` is the
//! standard cross-entropy VJP (softmax's Jacobian contracted with the
//! `-log p[target]` loss). `grad_h_tile`'s one-hot term is rewritten as
//! `one_hot(target_tile) @ w == embedding(w, target_tile)` (`Ops::
//! embedding`'s forward) to avoid building a second `(rows, V)` matrix
//! for that specific term; `grad_w`'s one-hot term has no such
//! shortcut (it is a scatter-add `Ops` doesn't expose), so
//! `one_hot_tile` is built directly — this is still only a *transient*
//! `(rows, V)` buffer, not a new memory-scaling axis.
//!
//! ## Memory bound: every op in the loop is on a *detached* operand
//!
//! `hidden`/`table` are [`Ops::detach`]ed once, up front, and every
//! per-tile accumulator (`total_nll`, `grad_table_accum`, each
//! `grad_h_tile`) is re-detached immediately after each `Ops::add`. This
//! is not optional bookkeeping — it is what makes the `(rows, V)`
//! intermediates *transient* rather than retained:
//!
//! - Backends with a shared/global autograd tape (`pm-cuda`): every op
//!   gates tape-recording on "does an operand carry an autograd
//!   ancestor" (`node_id().is_some()` in pm-cuda's terms). Starting from
//!   fully detached operands means **zero** tape entries are pushed for
//!   the whole loop, so there is nothing for a later `Ops::backward`
//!   call to hold alive.
//! - Backends with a per-tensor DAG (`pm-candle`): a tensor's ancestors
//!   stay reachable (`Arc`-held) for as long as the tensor itself is
//!   reachable, *regardless* of whether anyone ever calls `backward` on
//!   it. Without re-detaching the running accumulators, the final
//!   `loss`/`grad_table` would transitively reference *every* tile's
//!   `(rows, V)` logits via the chain of `Ops::add` calls — silently
//!   reproducing the full `(B, T, V)` residency this module exists to
//!   avoid, even though nothing was ever "tracked" in the pm-cuda sense.
//!
//! `Ops::backward` is never called inside this loop (the gradient is
//! analytic, not autograd-derived), so neither backend's "backward
//! clears/walks a big structure" behaviour is ever triggered here.

use crate::{Dtype, Ops};

/// See the module docs. `tile_rows` bounds peak transient memory to
/// `O(tile_rows · V)`; callers pick it to trade host-roundtrip count
/// (one per tile, for the target slice + one-hot upload) against peak
/// memory. Must be `> 0`.
///
/// Returns `(loss, grad_hidden, grad_table)` — see [`Ops::
/// fused_cross_entropy`]'s doc comment.
#[allow(clippy::type_complexity)]
pub fn fused_cross_entropy_tiled<O: Ops>(
    ops: &O,
    hidden: &O::Tensor,
    table: &O::Tensor,
    targets: &O::Tensor,
    tile_rows: usize,
) -> Result<(O::Tensor, O::Tensor, O::Tensor), O::Error> {
    use crate::Tensor as _;

    assert!(tile_rows > 0, "fused_cross_entropy: tile_rows must be > 0");
    let h_shape = hidden.shape().to_vec();
    assert!(
        h_shape.len() >= 2,
        "fused_cross_entropy: hidden must be rank >= 2 (..., D), got {h_shape:?}"
    );
    let d = h_shape[h_shape.len() - 1];
    let n: usize = h_shape[..h_shape.len() - 1].iter().product();
    assert!(n > 0, "fused_cross_entropy: hidden has zero rows");

    let t_shape = table.shape();
    assert_eq!(
        t_shape.len(),
        2,
        "fused_cross_entropy: table must be (V, D), got {t_shape:?}"
    );
    let v = t_shape[0];
    assert_eq!(
        t_shape[1], d,
        "fused_cross_entropy: table's D ({}) must match hidden's D ({d})",
        t_shape[1]
    );
    assert_eq!(
        targets.numel(),
        n,
        "fused_cross_entropy: targets element count ({}) must match hidden's row count ({n})",
        targets.numel()
    );

    let cdt = hidden.dtype();

    // ---- detach `hidden`/`table` *before* any other op touches them ----
    // (not just before the tile loop): on a shared/global-tape backend,
    // detaching only *after* a `reshape`/`to_dtype` still lets that one
    // call see the real (tracked) input and push a tape entry first —
    // harmless in isolation (`Reshape`/no-op-`to_dtype` entries carry no
    // `(rows, V)`-sized payload), but avoidable, and it breaks the
    // "everything from here down is untracked" invariant this loop
    // otherwise gets to rely on. Detaching first means every following
    // op's *input* already has no autograd ancestor, so — per every
    // backend's `grad_enabled() && operand.node_id().is_some()`-style
    // gate — none of them record anything either; no further per-call
    // `Ops::detach` is needed until the tile loop's own accumulators.
    let hidden_det = ops.detach(hidden)?; // (…,D) cdt
    let hidden_det_flat = ops.reshape(&hidden_det, &[n, d])?; // (N,D) cdt
    let hidden_det_f32 = ops.to_dtype(&hidden_det_flat, Dtype::F32)?; // (N,D) f32

    let table_det = ops.detach(table)?; // (V,D) native (fp32) dtype
    let table_det_cdt = ops.to_dtype(&table_det, cdt)?; // (V,D) cdt
    let table_det_cdt_t = ops.transpose(&table_det_cdt, 0, 1)?; // (D,V) cdt
    let table_det_f32 = ops.to_dtype(&table_det, Dtype::F32)?; // (V,D) f32

    // Targets are tiny (N elements) — one host round-trip up front, then
    // slice/reupload per tile. Avoids needing `Ops::narrow`/`Ops::
    // one_hot` on integer tensors (pm-cuda's `narrow` is F32-only today).
    let targets_flat = ops.reshape(targets, &[n])?;
    let targets_host = ops.to_vec_i64(&targets_flat)?;

    let mut total_nll: Option<O::Tensor> = None;
    let mut grad_table_accum: Option<O::Tensor> = None;
    let mut grad_hidden_tiles: Vec<O::Tensor> = Vec::with_capacity(n.div_ceil(tile_rows));

    let mut start = 0usize;
    while start < n {
        let len = tile_rows.min(n - start);
        let tgt_slice = &targets_host[start..start + len];

        let h_tile_cdt = ops.narrow(&hidden_det_flat, 0, start, len)?; // (rows,D) cdt
        let h_tile_f32 = ops.narrow(&hidden_det_f32, 0, start, len)?; // (rows,D) f32
        let tgt_tile = ops.from_slice_i64(tgt_slice, &[len])?; // (rows,) i64, fresh (untracked)

        // ---- forward: per-row NLL --------------------------------------
        let logits_tile_cdt = ops.matmul(&h_tile_cdt, &table_det_cdt_t)?; // (rows,V) cdt
        let logits_tile_f32 = ops.to_dtype(&logits_tile_cdt, Dtype::F32)?; // (rows,V) f32
        let log_p_tile = ops.log_softmax(&logits_tile_f32, 1)?; // (rows,V) f32
        let tgt_tile_2d = ops.reshape(&tgt_tile, &[len, 1])?;
        let picked = ops.gather(&log_p_tile, &tgt_tile_2d, 1)?; // (rows,1)
        let nll_tile = ops.neg(&ops.sum_all(&picked)?)?; // scalar

        let nll_next = match total_nll {
            Some(t) => ops.add(&t, &nll_tile)?,
            None => nll_tile,
        };
        total_nll = Some(ops.detach(&nll_next)?);

        // ---- backward: analytic softmax-minus-one-hot -------------------
        let softmax_tile = ops.exp(&log_p_tile)?; // (rows,V) f32 == softmax(logits)
        let onehot_tile = one_hot_from_host_targets(ops, tgt_slice, v)?; // (rows,V) f32
        let grad_logits_tile = ops.sub(&softmax_tile, &onehot_tile)?; // (rows,V) f32

        // grad_h_tile = grad_logits_tile @ w  (== (softmax - one_hot) @ w;
        // the one_hot @ w term equals embedding(w, target) but computing
        // it this way needs no extra Ops call since onehot_tile already
        // exists for the grad_w term below).
        let grad_h_tile_f32 = ops.matmul(&grad_logits_tile, &table_det_f32)?; // (rows,D) f32
        let grad_h_tile_cdt = ops.to_dtype(&grad_h_tile_f32, cdt)?;
        grad_hidden_tiles.push(ops.detach(&grad_h_tile_cdt)?);

        // grad_w += grad_logits_tile^T @ h_tile
        let grad_logits_tile_t = ops.transpose(&grad_logits_tile, 0, 1)?; // (V,rows) f32
        let grad_table_tile = ops.matmul(&grad_logits_tile_t, &h_tile_f32)?; // (V,D) f32
        let grad_table_next = match grad_table_accum {
            Some(t) => ops.add(&t, &grad_table_tile)?,
            None => grad_table_tile,
        };
        grad_table_accum = Some(ops.detach(&grad_table_next)?);

        start += len;
    }

    // SAFETY (logic, not memory): `n > 0` (asserted above) guarantees the
    // `while start < n` loop above ran at least once, so both `Option`s
    // are `Some`.
    let total_nll = total_nll.expect("fused_cross_entropy: n > 0 guarantees >= 1 tile");
    let grad_table_accum =
        grad_table_accum.expect("fused_cross_entropy: n > 0 guarantees >= 1 tile");

    let inv_n = 1.0f32 / n as f32;
    let loss = ops.mul_scalar(&total_nll, inv_n)?;
    let grad_table = ops.mul_scalar(&grad_table_accum, inv_n)?;

    let grad_hidden_tile_refs: Vec<&O::Tensor> = grad_hidden_tiles.iter().collect();
    let grad_hidden_flat = ops.concat(&grad_hidden_tile_refs, 0)?; // (N,D) cdt
    let grad_hidden_flat_scaled = ops.mul_scalar(&grad_hidden_flat, inv_n)?;
    let grad_hidden = ops.reshape(&grad_hidden_flat_scaled, &h_shape)?;

    Ok((loss, grad_hidden, grad_table))
}

/// Build a `(rows, num_classes)` one-hot `F32` matrix from host-side
/// target indices. `rows == targets.len()`.
fn one_hot_from_host_targets<O: Ops>(
    ops: &O,
    targets: &[i64],
    num_classes: usize,
) -> Result<O::Tensor, O::Error> {
    let rows = targets.len();
    let mut data = vec![0f32; rows * num_classes];
    for (r, &tgt) in targets.iter().enumerate() {
        assert!(
            tgt >= 0 && (tgt as usize) < num_classes,
            "fused_cross_entropy: target index {tgt} out of range [0, {num_classes})"
        );
        data[r * num_classes + tgt as usize] = 1.0;
    }
    ops.from_slice_f32(&data, &[rows, num_classes])
}
