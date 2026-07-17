//! End-to-end grad parity for `fused_cross_entropy_injected` against the
//! pre-existing unfused training path, across **every** trainable
//! parameter of a real (small) `PhotonMamba` — not just `hidden`/
//! `embed_weight` in isolation (`pm-candle/tests/fused_cross_entropy.rs`
//! covers that narrower case).
//!
//! This is the integration-level correctness gate for the "phantom
//! loss" injection (`sum_all(hidden*grad_hidden) + sum_all(table*
//! grad_table)` then one `Ops::backward`): it must reproduce gradients
//! for every Mamba2 block / chunker / converter parameter reachable
//! through `hidden`'s real ancestry, *and* correctly sum
//! `embed_weight`'s two contributions (input-embedding lookup +
//! lm_head).
//!
//! Dims mirror `pm-candle/tests/checkpoint_grad_parity.rs` (known to
//! avoid the separate, pre-existing Candle conv1d-backward underflow —
//! `docs/perf-log.md` / `crates/pm-candle/src/../conv.rs:46` — small T
//! not a clean multiple of block_len can trigger it).

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{
    ChunkLocalDecoder, ContextChunker, ContextConverter, ContextEncoder, DecoderLevel,
    HierarchicalDecoder, HierarchicalEncoder, HierarchicalLevel, TokenEmbedding,
};
use pm_core::{Ops, Param, Parameterized, PhotonMamba, Tensor};
use pm_train::{cross_entropy_loss, fused_cross_entropy_injected};

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
                    "param[{i}]: fused-injected grad differs from plain unfused grad by \
                     {d:.3e} (budget {tol:.0e})"
                );
            }
            (None, None) => {}
            (Some(_), None) => panic!("param[{i}]: plain path had a grad, fused path did not"),
            (None, Some(_)) => panic!("param[{i}]: fused path had a grad, plain path did not"),
        }
    }
    assert!(compared > 0, "no params were compared — test is vacuous");
    eprintln!("compared {compared} params, worst max|Δ| = {worst:.3e}");
}

#[test]
fn fused_injected_grads_match_plain_unfused_grads() {
    let bk = CandleBackend::new_cpu();
    let (b, t) = (2, 16);
    let (ids_v, tgt_v) = ids_targets(b, t, 64);
    let ids = bk.from_slice_i64(&ids_v, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&tgt_v, &[b, t]).unwrap();

    // --- plain: model.forward + cross_entropy_loss(out.logits) --------
    let (plain_loss, plain_grads) = {
        let model = build_toy_model(&bk);
        let params = model.collect_params();
        seed_params(&bk, &params, 4242);
        let out = model.forward(&bk, &ids).unwrap();
        let loss = cross_entropy_loss(&bk, &out.logits, &targets).unwrap();
        let loss_v = bk.to_vec_f32(&loss).unwrap()[0];
        let grads = bk.backward(&loss).unwrap();
        (loss_v, collect_grads(&bk, &params, &grads))
    };

    // --- fused: forward_no_lm_head + fused_cross_entropy_injected -----
    let (fused_loss, fused_grads) = {
        let model = build_toy_model(&bk);
        let params = model.collect_params();
        seed_params(&bk, &params, 4242); // same seed -> identical weights
        let hidden = model.forward_no_lm_head(&bk, &ids).unwrap();
        let (loss, grads) = fused_cross_entropy_injected(
            &bk,
            &hidden.predicted[0],
            &model.embed.weight,
            &targets,
            5,
        )
        .unwrap();
        let loss_v = bk.to_vec_f32(&loss).unwrap()[0];
        (loss_v, collect_grads(&bk, &params, &grads))
    };

    assert!(
        (plain_loss - fused_loss).abs() < 1e-5,
        "loss mismatch: plain={plain_loss:.8} fused={fused_loss:.8}"
    );
    assert_all_grads_match(&plain_grads, &fused_grads, 1e-4);
}

#[test]
fn fused_injected_grads_match_plain_unfused_grads_with_checkpointing() {
    let bk = CandleBackend::new_cpu();
    let (b, t) = (2, 16);
    let (ids_v, tgt_v) = ids_targets(b, t, 64);
    let ids = bk.from_slice_i64(&ids_v, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&tgt_v, &[b, t]).unwrap();

    // --- plain (no checkpointing) reference, as above ------------------
    let (plain_loss, plain_grads) = {
        let model = build_toy_model(&bk);
        let params = model.collect_params();
        seed_params(&bk, &params, 777);
        let out = model.forward(&bk, &ids).unwrap();
        let loss = cross_entropy_loss(&bk, &out.logits, &targets).unwrap();
        let loss_v = bk.to_vec_f32(&loss).unwrap()[0];
        let grads = bk.backward(&loss).unwrap();
        (loss_v, collect_grads(&bk, &params, &grads))
    };

    // --- fused + activation-checkpointed -------------------------------
    let (ckpt_loss, ckpt_grads) = {
        let model = build_toy_model(&bk);
        let params = model.collect_params();
        seed_params(&bk, &params, 777);
        let (hidden, cp) = model.forward_checkpointed_no_lm_head(&bk, &ids).unwrap();
        let (loss, mut grads) = fused_cross_entropy_injected(
            &bk,
            &hidden.predicted[0],
            &model.embed.weight,
            &targets,
            6,
        )
        .unwrap();
        pm_core::checkpoint_backward(&bk, cp, &mut grads, |o, id, x| {
            model.recompute_block(o, id, x)
        })
        .unwrap();
        let loss_v = bk.to_vec_f32(&loss).unwrap()[0];
        (loss_v, collect_grads(&bk, &params, &grads))
    };

    assert!(
        (plain_loss - ckpt_loss).abs() < 1e-5,
        "loss mismatch: plain={plain_loss:.8} fused+ckpt={ckpt_loss:.8}"
    );
    assert_all_grads_match(&plain_grads, &ckpt_grads, 1e-4);
}
