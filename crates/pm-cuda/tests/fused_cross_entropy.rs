//! `CudaBackend::fused_cross_entropy` loss + gradient parity against the
//! unfused path (`TokenEmbedding::lm_head_logits` + `cross_entropy_loss`)
//! — the `pm-cuda` counterpart to `pm-candle/tests/fused_cross_entropy.rs`.
//!
//! `pm-cuda` has a shared/global autograd tape (unlike Candle's
//! per-tensor DAG) that only clears on `Ops::backward` — this test is
//! also the empirical check that `Ops::fused_cross_entropy`'s tiling
//! loop, built entirely from `Ops::detach`ed operands, never grows that
//! tape (every op here gates tape-recording on `node_id().is_some()`;
//! see `crates/pm-core/src/loss.rs`'s module docs).

#![cfg(feature = "cuda")]

use pm_core::photon::TokenEmbedding;
use pm_core::{Ops, Param};
use pm_cuda::CudaBackend;
use pm_train::cross_entropy_loss;

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
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn loss_and_grads_match_unfused_cross_entropy() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let (b, t, d, v) = (2usize, 5usize, 6usize, 11usize);
    let n = b * t;

    let hidden = bk
        .param_from_slice_f32(&lcg_vec(1, n * d, 0.5), &[b, t, d])
        .unwrap();
    let table = bk
        .param_from_slice_f32(&lcg_vec(2, v * d, 0.5), &[v, d])
        .unwrap();
    let targets = bk.from_slice_i64(&lcg_targets(3, n, v), &[b, t]).unwrap();

    // ---- reference: full (rows, V) logits, unfused cross_entropy_loss ----
    let embed = TokenEmbedding::from_param(v, d, table.clone());
    let logits = embed.lm_head_logits(&bk, hidden.as_tensor()).unwrap();
    let loss_ref = cross_entropy_loss(&bk, &logits, &targets).unwrap();
    let loss_ref_v = bk.to_vec_f32(&loss_ref).unwrap()[0];
    let grads_ref = bk.backward(&loss_ref).unwrap();
    let grad_hidden_ref = bk
        .to_vec_f32(&bk.gradient(&grads_ref, &hidden).unwrap().unwrap())
        .unwrap();
    let grad_table_ref = bk
        .to_vec_f32(&bk.gradient(&grads_ref, &table).unwrap().unwrap())
        .unwrap();

    // ---- fused: tiled, ragged tile_rows=3 (10 rows -> 3,3,3,1) ----------
    let (loss_fused, grad_hidden_fused, grad_table_fused) = bk
        .fused_cross_entropy(hidden.as_tensor(), table.as_tensor(), &targets, 3)
        .unwrap();
    let loss_fused_v = bk.to_vec_f32(&loss_fused).unwrap()[0];
    let grad_hidden_fused = bk.to_vec_f32(&grad_hidden_fused).unwrap();
    let grad_table_fused = bk.to_vec_f32(&grad_table_fused).unwrap();

    assert!(
        (loss_ref_v - loss_fused_v).abs() < 1e-5,
        "loss mismatch: ref={loss_ref_v:.8} fused={loss_fused_v:.8}"
    );
    let dh = max_abs_diff(&grad_hidden_ref, &grad_hidden_fused);
    let dt = max_abs_diff(&grad_table_ref, &grad_table_fused);
    assert!(dh < 1e-4, "grad_hidden max|Δ| = {dh:.3e} (budget 1e-4)");
    assert!(dt < 1e-4, "grad_table max|Δ| = {dt:.3e} (budget 1e-4)");
}

/// The tape-safety property this module's doc comment claims: running
/// `fused_cross_entropy` does not grow `CudaBackend`'s shared autograd
/// tape at all (every operand it touches is `Ops::detach`ed, so no op
/// records a `TapeOp` entry). Verified by building an *unrelated*
/// pending tracked computation first, running `fused_cross_entropy`,
/// and then successfully backpropagating the original computation — if
/// `fused_cross_entropy` had pushed tape entries and something later
/// called `Ops::backward` on them, the shared tape would already be a
/// mess; here we instead directly assert the tape length is unchanged.
#[test]
fn fused_cross_entropy_does_not_grow_the_shared_tape() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let (b, t, d, v) = (1usize, 4usize, 3usize, 5usize);
    let n = b * t;

    // An unrelated tracked computation left pending (mimics `hidden`'s
    // own not-yet-backpropagated ancestry during real training).
    let x = bk
        .param_from_slice_f32(&lcg_vec(9, 4, 0.3), &[2, 2])
        .unwrap();
    let _pending = bk.mul_scalar(x.as_tensor(), 2.0).unwrap();

    let hidden = bk
        .param_from_slice_f32(&lcg_vec(1, n * d, 0.5), &[b, t, d])
        .unwrap();
    let table = bk
        .param_from_slice_f32(&lcg_vec(2, v * d, 0.5), &[v, d])
        .unwrap();
    let targets = bk.from_slice_i64(&lcg_targets(3, n, v), &[b, t]).unwrap();

    let tape_len_before = bk.tape_len();
    let _ = bk
        .fused_cross_entropy(hidden.as_tensor(), table.as_tensor(), &targets, 2)
        .unwrap();
    let tape_len_after = bk.tape_len();

    assert_eq!(
        tape_len_before,
        tape_len_after,
        "fused_cross_entropy pushed {} tape entries — the tiling loop is not \
         fully detached, so `(rows, V)` intermediates would stay resident \
         until the next Ops::backward call",
        tape_len_after - tape_len_before
    );
}
