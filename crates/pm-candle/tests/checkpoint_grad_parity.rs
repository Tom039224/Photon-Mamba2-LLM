//! Activation-checkpointing correctness: checkpointed gradients must equal
//! plain (non-checkpointed) gradients, parameter-for-parameter.
//!
//! This guards the B0 fix (`ops_impl.rs::merge_grad_stores` now `.detach()`s
//! gradients as they enter the persistent store, so `checkpoint_backward`
//! stops pinning all 30 blocks' forward tapes and OOMing at B=1 T=512).
//! The detach must change *memory only*, never gradient *values* — this test
//! is the invariance guard. It also fills a pre-existing coverage gap: no
//! test previously asserted the checkpointed path produces the same grads as
//! the plain path (the bf16 checkpointed test only checks finiteness).
//!
//! Runs on the CPU backend (fp32) so it guards the fix in default CI.

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{
    ChunkLocalDecoder, ContextChunker, ContextConverter, ContextEncoder, DecoderLevel,
    HierarchicalDecoder, HierarchicalEncoder, HierarchicalLevel, TokenEmbedding,
};
use pm_core::{Ops, Param, Parameterized, PhotonMamba, Tensor};
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

#[test]
fn checkpointed_grads_match_plain_grads() {
    let bk = CandleBackend::new_cpu();
    let (b, t) = (2, 16);
    let (ids_v, tgt_v) = ids_targets(b, t, 64);
    let ids = bk.from_slice_i64(&ids_v, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&tgt_v, &[b, t]).unwrap();

    // --- plain backward ---
    let plain = {
        let model = build_toy_model(&bk);
        let params = model.collect_params();
        seed_params(&bk, &params, 4242);
        let out = model.forward(&bk, &ids).unwrap();
        let loss = cross_entropy_loss(&bk, &out.logits, &targets).unwrap();
        let grads = bk.backward(&loss).unwrap();
        params
            .iter()
            .map(|p| {
                bk.gradient(&grads, p)
                    .unwrap()
                    .map(|g| bk.to_vec_f32(&g).unwrap())
            })
            .collect::<Vec<_>>()
    };

    // --- checkpointed backward (same seed → same weights) ---
    let ckpt = {
        let model = build_toy_model(&bk);
        let params = model.collect_params();
        seed_params(&bk, &params, 4242);
        let (out, cp) = model.forward_checkpointed(&bk, &ids).unwrap();
        let loss = cross_entropy_loss(&bk, &out.logits, &targets).unwrap();
        let mut grads = bk.backward(&loss).unwrap();
        pm_core::checkpoint_backward(&bk, cp, &mut grads, |o, id, x| {
            model.recompute_block(o, id, x)
        })
        .unwrap();
        params
            .iter()
            .map(|p| {
                bk.gradient(&grads, p)
                    .unwrap()
                    .map(|g| bk.to_vec_f32(&g).unwrap())
            })
            .collect::<Vec<_>>()
    };

    assert_eq!(plain.len(), ckpt.len());
    let mut compared = 0usize;
    let mut worst = 0f32;
    for (i, (a, c)) in plain.iter().zip(ckpt.iter()).enumerate() {
        match (a, c) {
            (Some(ga), Some(gc)) => {
                assert_eq!(ga.len(), gc.len(), "param[{i}] grad length mismatch");
                let max_abs: f32 = ga
                    .iter()
                    .zip(gc.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                worst = worst.max(max_abs);
                compared += 1;
                assert!(
                    max_abs < 1e-5,
                    "param[{i}]: checkpointed grad differs from plain by {max_abs:.3e} (>1e-5) \
                     — the merge_grad_stores detach changed gradient VALUES, not just memory"
                );
            }
            (None, None) => {}
            _ => panic!("param[{i}]: grad presence differs between plain and checkpointed paths"),
        }
    }
    eprintln!(
        "checkpointed_grads_match_plain_grads: compared {compared} params, worst |Δ| = {worst:.3e}"
    );
    assert!(
        compared > 20,
        "expected all block params compared, got {compared}"
    );
}
