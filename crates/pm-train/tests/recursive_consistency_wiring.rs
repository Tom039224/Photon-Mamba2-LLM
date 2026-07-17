//! Phase D.1: `fused_photon_loss_injected` wiring correctness.
//!
//! `crates/pm-train/src/loss.rs::recursive_consistency_loss` (F.2) and
//! `photon_loss` (F.3) were implemented but never called from the
//! training loop. This file is the integration-level gate for wiring
//! them in via a fused-CE-compatible path, covering the two properties
//! CLAUDE.md's review checklist calls out explicitly:
//! - "forward だけでなく backward / grad の検証もある" — every test
//!   below compares full parameter gradients, not just losses.
//! - the memory-efficiency plan's D.1 zero-cost-at-α=0 requirement.
//!
//! Three gates:
//! 1. `alpha == 0.0` is (numerically) identical to the pre-D.1
//!    `fused_cross_entropy_injected` path, both loss and every grad —
//!    turning the feature "off" must not perturb existing training.
//! 2. `alpha > 0.0` matches the naive reference `photon_loss`
//!    (materialises `(B,T,vocab)`, plain `Ops::backward`) across every
//!    trainable parameter, with and without activation checkpointing —
//!    this is what proves the "one combined backward, not two" design
//!    (`fused_photon_loss_injected`'s doc comment) is correct under
//!    both training paths `pm-cli::train_cmd` actually uses.
//! 3. Turning `alpha` on changes the gradient at an **encoder-only**
//!    parameter (not just decoder/converter/embed) — i.e. `L_rec`'s
//!    own backward pass, not just `L_token`'s, reaches the encoder.

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{
    ChunkLocalDecoder, ContextChunker, ContextConverter, ContextEncoder, DecoderLevel,
    HierarchicalDecoder, HierarchicalEncoder, HierarchicalLevel, TokenEmbedding,
};
use pm_core::{Ops, Param, Parameterized, PhotonMamba, Tensor};
use pm_train::{fused_cross_entropy_injected, fused_photon_loss_injected, photon_loss};

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

fn seed_params(bk: &CandleBackend, params: &[&<CandleBackend as Ops>::Param], seed: u64) {
    for (k, p) in params.iter().enumerate() {
        let shape = p.as_tensor().shape().to_vec();
        let n: usize = shape.iter().product();
        let t = bk
            .from_slice_f32(&lcg_vec(seed.wrapping_add(k as u64 * 101), n, 0.1), &shape)
            .unwrap();
        bk.assign(p, &t).unwrap();
    }
}

fn mk_block(bk: &CandleBackend, d_model: usize) -> Mamba2Block<CandleBackend> {
    let cfg = Mamba2Config {
        d_model,
        d_state: 8,
        d_head: 8,
        n_heads: d_model / 8,
        n_groups: 1,
        d_conv: 4,
        block_len: 4,
        rmsnorm_eps: 1e-5,
    };
    Mamba2Block::from_constants(bk, cfg, 0.05).unwrap()
}

fn build_toy_model(bk: &CandleBackend) -> PhotonMamba<CandleBackend> {
    let (vocab, d_model, chunk, n_layers) = (64usize, 32usize, 4usize, 2usize);
    let embed = TokenEmbedding::from_constants(bk, vocab, d_model, 0.05).unwrap();
    let lvl0 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(
            (0..n_layers).map(|_| mk_block(bk, d_model)).collect(),
        ),
        chunker: Some(ContextChunker::from_constants(bk, d_model, d_model, chunk, 0.05).unwrap()),
    };
    let lvl1 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(
            (0..n_layers).map(|_| mk_block(bk, d_model)).collect(),
        ),
        chunker: None,
    };
    let encoder = HierarchicalEncoder::from_levels(vec![lvl0, lvl1]);
    let conv = ContextConverter::from_constants(bk, d_model, d_model, chunk, 0.05).unwrap();
    let dec = ChunkLocalDecoder::from_layers(
        (0..n_layers).map(|_| mk_block(bk, d_model)).collect(),
        chunk,
        chunk,
    );
    let decoder = HierarchicalDecoder::from_levels(vec![DecoderLevel::new(conv, dec)]);
    PhotonMamba::new(embed, encoder, decoder)
}

fn ids_targets(b: usize, t: usize, vocab: usize) -> (Vec<i64>, Vec<i64>) {
    let ids: Vec<i64> = (0..b * t)
        .map(|i| (i as i64 * 7 + 3) % vocab as i64)
        .collect();
    let mut tgt = vec![0i64; b * t];
    for bi in 0..b {
        for ti in 0..t {
            tgt[bi * t + ti] = ids[bi * t + (ti + 1) % t];
        }
    }
    (ids, tgt)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Collect every param's gradient (as `Option<Vec<f32>>`, `None` for
/// params the backward pass never touched) in `params`' order.
fn collect_grads(
    bk: &CandleBackend,
    params: &[&<CandleBackend as Ops>::Param],
    grads: &<CandleBackend as Ops>::GradStore,
) -> Vec<Option<Vec<f32>>> {
    params
        .iter()
        .map(|p| {
            bk.gradient(grads, p)
                .unwrap()
                .map(|g| bk.to_vec_f32(&g).unwrap())
        })
        .collect()
}

fn assert_all_grads_match(plain: &[Option<Vec<f32>>], fused: &[Option<Vec<f32>>], tol: f32) {
    assert_eq!(plain.len(), fused.len());
    let mut compared = 0usize;
    let mut worst = 0f32;
    for (i, (a, b)) in plain.iter().zip(fused.iter()).enumerate() {
        match (a, b) {
            (Some(ga), Some(gb)) => {
                assert_eq!(ga.len(), gb.len(), "param[{i}] grad length mismatch");
                let d = max_abs_diff(ga, gb);
                worst = worst.max(d);
                compared += 1;
                assert!(
                    d < tol,
                    "param[{i}]: fused grad differs from naive reference grad by \
                     {d:.3e} (budget {tol:.0e})"
                );
            }
            (None, None) => {}
            (Some(_), None) => panic!("param[{i}]: naive path had a grad, fused path did not"),
            (None, Some(_)) => panic!("param[{i}]: fused path had a grad, naive path did not"),
        }
    }
    assert!(compared > 0, "no params were compared — test is vacuous");
    eprintln!("compared {compared} params, worst max|Δ| = {worst:.3e}");
}

/// Gate 1: `alpha == 0.0` must reproduce `fused_cross_entropy_injected`
/// exactly — turning the feature "off" is not allowed to perturb the
/// pre-D.1 training graph (memory-efficiency plan D.1).
#[test]
fn alpha_zero_matches_fused_cross_entropy_injected_exactly() {
    let bk = CandleBackend::new_cpu();
    let (b, t) = (2, 16);
    let (ids_v, tgt_v) = ids_targets(b, t, 64);
    let ids = bk.from_slice_i64(&ids_v, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&tgt_v, &[b, t]).unwrap();

    let (ce_loss, ce_grads) = {
        let model = build_toy_model(&bk);
        let params = model.collect_params();
        seed_params(&bk, &params, 999);
        let hidden = model.forward_no_lm_head(&bk, &ids).unwrap();
        let (loss, grads) = fused_cross_entropy_injected(
            &bk,
            &hidden.predicted[0],
            &model.embed.weight,
            &targets,
            5,
        )
        .unwrap();
        (
            bk.to_vec_f32(&loss).unwrap()[0],
            collect_grads(&bk, &params, &grads),
        )
    };

    // Same seed, same inputs — only difference is going through
    // `fused_photon_loss_injected` with alpha=0.0 (and, deliberately,
    // *non-empty* predicted/encoded_targets slices) instead of calling
    // `fused_cross_entropy_injected` directly, to prove the alpha==0.0
    // guard really does bypass everything below it.
    let (photon_loss_v, photon_grads, photon_components) = {
        let model = build_toy_model(&bk);
        let params = model.collect_params();
        seed_params(&bk, &params, 999);
        let hidden = model.forward_no_lm_head(&bk, &ids).unwrap();
        let n_dec = hidden.predicted.len();
        let (report, grads) = fused_photon_loss_injected(
            &bk,
            &hidden.predicted[0],
            &model.embed.weight,
            &targets,
            5,
            &hidden.predicted,
            &hidden.encoded.encoded[..n_dec],
            0.0,
        )
        .unwrap();
        (
            bk.to_vec_f32(&report.total).unwrap()[0],
            collect_grads(&bk, &params, &grads),
            report.components,
        )
    };

    assert!(
        (ce_loss - photon_loss_v).abs() < 1e-6,
        "alpha=0 loss must match fused_cross_entropy_injected: ce={ce_loss:.8} photon={photon_loss_v:.8}"
    );
    // Phase B'.3: the zero-cost α=0 branch must never populate
    // `components` — no `L_rec` term was computed, so there is nothing
    // to report beyond `total` (already asserted above to equal CE).
    assert!(
        photon_components.is_none(),
        "alpha=0 must leave PhotonLossReport::components as None, got {photon_components:?}"
    );
    assert_all_grads_match(&ce_grads, &photon_grads, 1e-6);
}

/// Gate 2a: `alpha > 0.0`, no checkpointing — matches the naive
/// `photon_loss` reference (materialised logits + plain backward)
/// across every parameter.
#[test]
fn alpha_positive_matches_naive_photon_loss() {
    let bk = CandleBackend::new_cpu();
    let (b, t) = (2, 16);
    let (ids_v, tgt_v) = ids_targets(b, t, 64);
    let ids = bk.from_slice_i64(&ids_v, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&tgt_v, &[b, t]).unwrap();
    let alpha = 0.3f32;

    let (naive_loss, naive_grads) = {
        let model = build_toy_model(&bk);
        let params = model.collect_params();
        seed_params(&bk, &params, 2024);
        let out = model.forward(&bk, &ids).unwrap();
        let n_dec = out.predicted.len();
        let loss = photon_loss(
            &bk,
            &out.logits,
            &targets,
            &out.predicted,
            &out.encoded.encoded[..n_dec],
            alpha,
        )
        .unwrap();
        let loss_v = bk.to_vec_f32(&loss).unwrap()[0];
        let grads = bk.backward(&loss).unwrap();
        (loss_v, collect_grads(&bk, &params, &grads))
    };

    let (fused_loss, fused_grads) = {
        let model = build_toy_model(&bk);
        let params = model.collect_params();
        seed_params(&bk, &params, 2024);
        let hidden = model.forward_no_lm_head(&bk, &ids).unwrap();
        let n_dec = hidden.predicted.len();
        let (report, grads) = fused_photon_loss_injected(
            &bk,
            &hidden.predicted[0],
            &model.embed.weight,
            &targets,
            5,
            &hidden.predicted,
            &hidden.encoded.encoded[..n_dec],
            alpha,
        )
        .unwrap();
        let loss_v = bk.to_vec_f32(&report.total).unwrap()[0];
        (loss_v, collect_grads(&bk, &params, &grads))
    };

    assert!(
        (naive_loss - fused_loss).abs() < 1e-5,
        "loss mismatch: naive={naive_loss:.8} fused={fused_loss:.8}"
    );
    assert_all_grads_match(&naive_grads, &fused_grads, 1e-4);
}

/// Gate 2b: same as above, but the fused side goes through
/// `forward_checkpointed_no_lm_head` + `checkpoint_backward` — proves
/// the recursive-consistency term composes correctly with activation
/// checkpointing (not just the CE term, which `fused_cross_entropy_
/// injection.rs`'s equivalent test already covers).
#[test]
fn alpha_positive_matches_naive_photon_loss_with_checkpointing() {
    let bk = CandleBackend::new_cpu();
    let (b, t) = (2, 16);
    let (ids_v, tgt_v) = ids_targets(b, t, 64);
    let ids = bk.from_slice_i64(&ids_v, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&tgt_v, &[b, t]).unwrap();
    let alpha = 0.3f32;

    let (naive_loss, naive_grads) = {
        let model = build_toy_model(&bk);
        let params = model.collect_params();
        seed_params(&bk, &params, 555);
        let out = model.forward(&bk, &ids).unwrap();
        let n_dec = out.predicted.len();
        let loss = photon_loss(
            &bk,
            &out.logits,
            &targets,
            &out.predicted,
            &out.encoded.encoded[..n_dec],
            alpha,
        )
        .unwrap();
        let loss_v = bk.to_vec_f32(&loss).unwrap()[0];
        let grads = bk.backward(&loss).unwrap();
        (loss_v, collect_grads(&bk, &params, &grads))
    };

    let (ckpt_loss, ckpt_grads) = {
        let model = build_toy_model(&bk);
        let params = model.collect_params();
        seed_params(&bk, &params, 555);
        let (hidden, cp) = model.forward_checkpointed_no_lm_head(&bk, &ids).unwrap();
        let n_dec = hidden.predicted.len();
        let (report, mut grads) = fused_photon_loss_injected(
            &bk,
            &hidden.predicted[0],
            &model.embed.weight,
            &targets,
            5,
            &hidden.predicted,
            &hidden.encoded.encoded[..n_dec],
            alpha,
        )
        .unwrap();
        pm_core::checkpoint_backward(&bk, cp, &mut grads, |o, id, x| {
            model.recompute_block(o, id, x)
        })
        .unwrap();
        let loss_v = bk.to_vec_f32(&report.total).unwrap()[0];
        (loss_v, collect_grads(&bk, &params, &grads))
    };

    assert!(
        (naive_loss - ckpt_loss).abs() < 1e-5,
        "loss mismatch: naive={naive_loss:.8} fused+ckpt={ckpt_loss:.8}"
    );
    assert_all_grads_match(&naive_grads, &ckpt_grads, 1e-4);
}

/// Gate 3: turning `alpha` on must change the gradient at an
/// **encoder**-only parameter, not just decoder/converter/embed ones.
///
/// `predicted[0]`'s own forward ancestry already runs through every
/// encoder level (see `fused_photon_loss_injected`'s doc comment), so
/// CE *alone* (alpha=0) already deposits a nonzero gradient at every
/// encoder parameter — a bare "grad is nonzero at alpha=0.3" check
/// would not prove `L_rec` contributed anything. This test instead
/// diffs the *same* encoder-level-0 parameter's gradient between
/// alpha=0.0 and alpha=0.3 runs (same model, same seed, same batch —
/// the only difference is `alpha`), so any nonzero delta is
/// attributable to `L_rec`'s own backward pass.
#[test]
fn nonzero_alpha_changes_gradient_at_encoder_only_params() {
    let bk = CandleBackend::new_cpu();
    let (b, t) = (2, 16);
    let (ids_v, tgt_v) = ids_targets(b, t, 64);
    let ids = bk.from_slice_i64(&ids_v, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&tgt_v, &[b, t]).unwrap();

    let model = build_toy_model(&bk);
    let params = model.collect_params();
    seed_params(&bk, &params, 3333);

    let mut encoder_l0_params: Vec<&<CandleBackend as Ops>::Param> = Vec::new();
    model.encoder.levels[0]
        .encoder
        .append_params(&mut encoder_l0_params);
    assert!(
        !encoder_l0_params.is_empty(),
        "test setup: encoder level 0 must have trainable params"
    );

    let run_alpha = |alpha: f32| -> (f32, Vec<Option<Vec<f32>>>) {
        let hidden = model.forward_no_lm_head(&bk, &ids).unwrap();
        let n_dec = hidden.predicted.len();
        let (report, grads) = fused_photon_loss_injected(
            &bk,
            &hidden.predicted[0],
            &model.embed.weight,
            &targets,
            5,
            &hidden.predicted,
            &hidden.encoded.encoded[..n_dec],
            alpha,
        )
        .unwrap();
        let loss_v = bk.to_vec_f32(&report.total).unwrap()[0];
        (loss_v, collect_grads(&bk, &encoder_l0_params, &grads))
    };

    let (loss0, grads0) = run_alpha(0.0);
    let (loss3, grads3) = run_alpha(0.3);

    assert!(
        (loss3 - loss0).abs() > 1e-6,
        "alpha=0.3 total loss should differ from alpha=0.0 (loss0={loss0:.6}, loss3={loss3:.6})"
    );

    let mut max_delta = 0f32;
    for (a, b) in grads0.iter().zip(grads3.iter()) {
        if let (Some(ga), Some(gb)) = (a, b) {
            max_delta = max_delta.max(max_abs_diff(ga, gb));
        }
    }
    assert!(
        max_delta > 1e-6,
        "expected L_rec to change at least one encoder-level-0 parameter's \
         gradient between alpha=0.0 and alpha=0.3, but max|Δ| = {max_delta:.3e}"
    );
    eprintln!(
        "encoder level-0 params: loss delta = {:.6}, max grad delta (alpha 0 -> 0.3) = {max_delta:.3e}",
        (loss3 - loss0).abs()
    );
}

/// Gate 4 (Phase B'.3): `PhotonLossReport::components` — when `alpha > 0`
/// it must be populated, and `total ≈ ce + alpha * lrec` (PHOTON §2.3
/// Eq. (9)). Closes `docs/perf-log.md` 2026-07-05's D.2b TODO: before
/// this, an α=0.3 run's logged `loss` was the CE+α·L_rec composite with
/// no way to recover the CE-alone trajectory from the log.
#[test]
fn fused_photon_loss_injected_reports_ce_and_lrec_components() {
    let bk = CandleBackend::new_cpu();
    let (b, t) = (2, 16);
    let (ids_v, tgt_v) = ids_targets(b, t, 64);
    let ids = bk.from_slice_i64(&ids_v, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&tgt_v, &[b, t]).unwrap();
    let alpha = 0.3f32;

    let model = build_toy_model(&bk);
    let params = model.collect_params();
    seed_params(&bk, &params, 7777);
    let hidden = model.forward_no_lm_head(&bk, &ids).unwrap();
    let n_dec = hidden.predicted.len();
    let (report, _grads) = fused_photon_loss_injected(
        &bk,
        &hidden.predicted[0],
        &model.embed.weight,
        &targets,
        5,
        &hidden.predicted,
        &hidden.encoded.encoded[..n_dec],
        alpha,
    )
    .unwrap();

    let total_v = bk.to_vec_f32(&report.total).unwrap()[0];
    let components = report
        .components
        .expect("alpha>0 must populate PhotonLossReport::components");
    let recombined = components.ce + alpha * components.lrec;
    assert!(
        (total_v - recombined).abs() < 1e-5,
        "total ({total_v}) should equal ce + alpha*lrec ({recombined}): \
         ce={} lrec={} alpha={alpha}",
        components.ce,
        components.lrec
    );
    // Sanity: lrec (L_rec, PHOTON §2.3 Eq. (11)) is a mean cosine
    // distance in [0, 2], not itself already alpha-weighted.
    assert!(
        (0.0..=2.0).contains(&components.lrec),
        "lrec out of the expected cosine-distance range: {}",
        components.lrec
    );
}

/// Companion to the above: `alpha == 0.0` must leave `components` as
/// `None` — the zero-cost branch never computes `L_rec`, so there is
/// nothing to report. (Also asserted inline in
/// `alpha_zero_matches_fused_cross_entropy_injected_exactly`; this test
/// isolates just that one property for a clearer failure message.)
#[test]
fn fused_photon_loss_injected_components_is_none_at_alpha_zero() {
    let bk = CandleBackend::new_cpu();
    let (b, t) = (2, 16);
    let (ids_v, tgt_v) = ids_targets(b, t, 64);
    let ids = bk.from_slice_i64(&ids_v, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&tgt_v, &[b, t]).unwrap();

    let model = build_toy_model(&bk);
    let params = model.collect_params();
    seed_params(&bk, &params, 7777);
    let hidden = model.forward_no_lm_head(&bk, &ids).unwrap();
    let n_dec = hidden.predicted.len();
    let (report, _grads) = fused_photon_loss_injected(
        &bk,
        &hidden.predicted[0],
        &model.embed.weight,
        &targets,
        5,
        &hidden.predicted,
        &hidden.encoded.encoded[..n_dec],
        0.0,
    )
    .unwrap();

    assert!(
        report.components.is_none(),
        "alpha=0 must not populate PhotonLossReport::components, got {:?}",
        report.components
    );
}

/// The zero-cost branch is `alpha == 0.0 || predicted.is_empty()` (an
/// OR — see `fused_photon_loss_injected`'s doc). The alpha=0 leg is
/// covered above; this covers the *other* leg: even with `alpha > 0`,
/// an empty `predicted`/`encoded_targets` (no hierarchy to penalise)
/// must take the literal-delegation path and leave `components` None.
#[test]
fn fused_photon_loss_injected_components_is_none_when_predicted_empty() {
    let bk = CandleBackend::new_cpu();
    let (b, t) = (2, 16);
    let (ids_v, tgt_v) = ids_targets(b, t, 64);
    let ids = bk.from_slice_i64(&ids_v, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&tgt_v, &[b, t]).unwrap();

    let model = build_toy_model(&bk);
    let params = model.collect_params();
    seed_params(&bk, &params, 7777);
    let hidden = model.forward_no_lm_head(&bk, &ids).unwrap();

    // alpha = 0.3 (> 0) but empty consistency slices: still zero-cost.
    let empty: &[<CandleBackend as Ops>::Tensor] = &[];
    let (report, _grads) = fused_photon_loss_injected(
        &bk,
        &hidden.predicted[0],
        &model.embed.weight,
        &targets,
        5,
        empty,
        empty,
        0.3,
    )
    .unwrap();

    assert!(
        report.components.is_none(),
        "empty predicted must not populate components even at alpha>0, got {:?}",
        report.components
    );
}
