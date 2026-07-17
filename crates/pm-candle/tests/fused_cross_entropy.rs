//! Loss + gradient parity for `Ops::fused_cross_entropy` (memory-
//! efficiency plan: fused/tiled cross-entropy) against the pre-existing
//! unfused path: `TokenEmbedding::lm_head_logits` (materialises the full
//! `(rows, V)` logits) + `pm_train::cross_entropy_loss` (full
//! `log_softmax` + `gather`).
//!
//! TDD gate (task spec): loss within 1e-5, grads within 1e-4, both on
//! CPU fp32.

use pm_candle::CandleBackend;
use pm_core::photon::TokenEmbedding;
use pm_core::{Ops, Param};
use pm_train::cross_entropy_loss;

type Bk = CandleBackend;

fn lcg_vec(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as u32 as f32 / u32::MAX as f32 - 0.5) * scale
        })
        .collect()
}

fn lcg_targets(seed: u64, n: usize, vocab: usize) -> Vec<i64> {
    let mut state = seed.wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % vocab as u64) as i64
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// `(backend, hidden_param (B,T,D), table_param (V,D), targets (B,T) i64)`.
fn setup(
    b: usize,
    t: usize,
    d: usize,
    v: usize,
) -> (
    Bk,
    <Bk as Ops>::Param,
    <Bk as Ops>::Param,
    <Bk as Ops>::Tensor,
) {
    let bk = CandleBackend::new_cpu();
    let n = b * t;
    let hidden = bk
        .param_from_slice_f32(&lcg_vec(1, n * d, 0.5), &[b, t, d])
        .unwrap();
    let table = bk
        .param_from_slice_f32(&lcg_vec(2, v * d, 0.5), &[v, d])
        .unwrap();
    let targets = bk.from_slice_i64(&lcg_targets(3, n, v), &[b, t]).unwrap();
    (bk, hidden, table, targets)
}

/// Reference: full `(rows, V)` logits via the existing tied-embedding
/// lm_head, then the existing (non-tiled) `cross_entropy_loss`.
fn reference_loss(
    bk: &Bk,
    hidden: &<Bk as Ops>::Param,
    table: &<Bk as Ops>::Param,
    targets: &<Bk as Ops>::Tensor,
    v: usize,
    d: usize,
) -> <Bk as Ops>::Tensor {
    let embed = TokenEmbedding::from_param(v, d, table.clone());
    let logits = embed.lm_head_logits(bk, hidden.as_tensor()).unwrap();
    cross_entropy_loss(bk, &logits, targets).unwrap()
}

#[test]
fn loss_matches_unfused_cross_entropy() {
    let (b, t, d, v) = (2usize, 5usize, 6usize, 11usize);
    let (bk, hidden, table, targets) = setup(b, t, d, v);

    let loss_ref = reference_loss(&bk, &hidden, &table, &targets, v, d);
    let loss_ref_v = bk.to_vec_f32(&loss_ref).unwrap()[0];

    // 10 rows, tile_rows=3 -> ragged tiles (3,3,3,1): also exercises the
    // last-tile-shorter-than-tile_rows path.
    let (loss_fused, _gh, _gt) = bk
        .fused_cross_entropy(hidden.as_tensor(), table.as_tensor(), &targets, 3)
        .unwrap();
    let loss_fused_v = bk.to_vec_f32(&loss_fused).unwrap()[0];

    assert!(
        (loss_ref_v - loss_fused_v).abs() < 1e-5,
        "loss mismatch: ref={loss_ref_v:.8} fused={loss_fused_v:.8} \
         diff={:.3e}",
        (loss_ref_v - loss_fused_v).abs()
    );
}

#[test]
fn grads_match_unfused_backward() {
    let (b, t, d, v) = (2usize, 5usize, 6usize, 11usize);
    let (bk, hidden, table, targets) = setup(b, t, d, v);

    // Reference: full unfused path, ordinary autograd backward.
    let loss_ref = reference_loss(&bk, &hidden, &table, &targets, v, d);
    let grads_ref = bk.backward(&loss_ref).unwrap();
    let grad_hidden_ref = bk
        .to_vec_f32(&bk.gradient(&grads_ref, &hidden).unwrap().unwrap())
        .unwrap();
    let grad_table_ref = bk
        .to_vec_f32(&bk.gradient(&grads_ref, &table).unwrap().unwrap())
        .unwrap();

    // Fused: analytic tiled path, same ragged tiling as the loss test.
    let (_loss_fused, grad_hidden_fused, grad_table_fused) = bk
        .fused_cross_entropy(hidden.as_tensor(), table.as_tensor(), &targets, 3)
        .unwrap();
    let grad_hidden_fused = bk.to_vec_f32(&grad_hidden_fused).unwrap();
    let grad_table_fused = bk.to_vec_f32(&grad_table_fused).unwrap();

    let dh = max_abs_diff(&grad_hidden_ref, &grad_hidden_fused);
    let dt = max_abs_diff(&grad_table_ref, &grad_table_fused);
    assert!(dh < 1e-4, "grad_hidden max|Δ| = {dh:.3e} (budget 1e-4)");
    assert!(dt < 1e-4, "grad_table max|Δ| = {dt:.3e} (budget 1e-4)");
}

#[test]
fn tile_rows_choice_does_not_change_result() {
    // Same inputs, three different tile_rows (a 1-row tile, a ragged
    // tile, and a single all-rows tile) must produce the same loss and
    // gradients — a tiling-boundary regression guard independent of the
    // "vs unfused" comparison above.
    let (b, t, d, v) = (1usize, 7usize, 4usize, 9usize);
    let (bk, hidden, table, targets) = setup(b, t, d, v);
    let n = b * t;

    let mut losses = Vec::new();
    let mut gh_all = Vec::new();
    let mut gt_all = Vec::new();
    for tile_rows in [1usize, 3, n] {
        let (loss, gh, gt) = bk
            .fused_cross_entropy(hidden.as_tensor(), table.as_tensor(), &targets, tile_rows)
            .unwrap();
        losses.push(bk.to_vec_f32(&loss).unwrap()[0]);
        gh_all.push(bk.to_vec_f32(&gh).unwrap());
        gt_all.push(bk.to_vec_f32(&gt).unwrap());
    }
    for i in 1..losses.len() {
        assert!(
            (losses[0] - losses[i]).abs() < 1e-5,
            "loss differs across tile_rows: {} vs {}",
            losses[0],
            losses[i]
        );
        assert!(
            max_abs_diff(&gh_all[0], &gh_all[i]) < 1e-5,
            "grad_hidden differs across tile_rows choices"
        );
        assert!(
            max_abs_diff(&gt_all[0], &gt_all[i]) < 1e-5,
            "grad_table differs across tile_rows choices"
        );
    }
}

#[test]
fn detach_strips_ancestry_but_keeps_value() {
    // `Ops::detach` itself: value-preserving, and the result must not
    // extend the *reachable* backward graph of anything built from it
    // (checked indirectly: backward through a detached-derived tensor
    // must not deposit a gradient on the original param).
    let bk = CandleBackend::new_cpu();
    let p = bk
        .param_from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .unwrap();
    let detached = bk.detach(p.as_tensor()).unwrap();
    assert_eq!(
        bk.to_vec_f32(&detached).unwrap(),
        bk.to_vec_f32(p.as_tensor()).unwrap()
    );

    let y = bk.mul_scalar(&detached, 3.0).unwrap();
    let loss = bk.sum_all(&y).unwrap();
    let grads = bk.backward(&loss).unwrap();
    assert!(
        bk.gradient(&grads, &p).unwrap().is_none(),
        "detach did not sever ancestry: original param still received a gradient"
    );
}
