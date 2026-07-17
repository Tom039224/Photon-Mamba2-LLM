//! Phase C (memory-efficiency plan) acceptance gate (invariant #3 in
//! CLAUDE.md terms): `StatefulGenerator` (O(1)-memory recurrent decode)
//! must match `Generator` (padded re-forward) token-for-token under
//! greedy sampling, on *any* weights — HierGen is equivalent to
//! re-forward by construction (Phase C spec). Checked both at raw init
//! and after a few real AdamW training steps, since training is exactly
//! the kind of "any weights" this must hold for.

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{
    ChunkLocalDecoder, ContextChunker, ContextConverter, ContextEncoder, DecoderLevel,
    HierarchicalDecoder, HierarchicalEncoder, HierarchicalLevel, TokenEmbedding,
};
use pm_core::{Ops, Param, Parameterized, PhotonMamba, Tensor};
use pm_infer::{GenerateConfig, Generator, Sampler, StatefulGenerator};
use pm_train::{cross_entropy_loss, AdamW, AdamWConfig, Trainer};

// ---- tiny model dims (same family as pm-cli/tests/backend_parity.rs) ----
const VOCAB: usize = 128;
const D_MODEL: usize = 32;
const D_STATE: usize = 16;
const D_HEAD: usize = 8;
const N_HEADS: usize = D_MODEL / D_HEAD; // 4
const N_GROUPS: usize = 1;
const D_CONV: usize = 4;
const BLOCK_LEN: usize = 8;
const RMSNORM_EPS: f32 = 1e-5;
const N_LAYERS: usize = 2;
const CHUNK_SIZE: usize = 4;
const INIT_SCALE: f32 = 0.05;
const N_NEW_TOKENS: usize = 64;

fn lcg_vec(seed: u64, n: usize, scale: f32, bias: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let r = ((state >> 41) as f32) / ((1u32 << 23) as f32);
            r * scale + bias
        })
        .collect()
}

fn seed_params(bk: &CandleBackend, params: &[&<CandleBackend as Ops>::Param], seed: u64) {
    for (i, p) in params.iter().enumerate() {
        let shape = p.as_tensor().shape().to_vec();
        let n: usize = shape.iter().product();
        let data = lcg_vec(seed.wrapping_add(i as u64 * 1_000_003), n, 0.2, -0.1);
        let t = bk.from_slice_f32(&data, &shape).unwrap();
        bk.assign(p, &t).unwrap();
    }
}

fn make_model(bk: &CandleBackend) -> PhotonMamba<CandleBackend> {
    let m2cfg = Mamba2Config {
        d_model: D_MODEL,
        d_state: D_STATE,
        d_head: D_HEAD,
        n_heads: N_HEADS,
        n_groups: N_GROUPS,
        d_conv: D_CONV,
        block_len: BLOCK_LEN,
        rmsnorm_eps: RMSNORM_EPS,
    };
    let mk_block = || Mamba2Block::from_constants(bk, m2cfg.clone(), INIT_SCALE).unwrap();

    let embed = TokenEmbedding::from_constants(bk, VOCAB, D_MODEL, INIT_SCALE).unwrap();
    let lvl0 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers((0..N_LAYERS).map(|_| mk_block()).collect()),
        chunker: Some(
            ContextChunker::from_constants(bk, D_MODEL, D_MODEL, CHUNK_SIZE, INIT_SCALE).unwrap(),
        ),
    };
    let lvl1 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers((0..N_LAYERS).map(|_| mk_block()).collect()),
        chunker: None,
    };
    let encoder = HierarchicalEncoder::from_levels(vec![lvl0, lvl1]);

    let conv =
        ContextConverter::from_constants(bk, D_MODEL, D_MODEL, CHUNK_SIZE, INIT_SCALE).unwrap();
    let dec_stack = ChunkLocalDecoder::from_layers(
        (0..N_LAYERS).map(|_| mk_block()).collect(),
        CHUNK_SIZE,
        CHUNK_SIZE,
    );
    let decoder = HierarchicalDecoder::from_levels(vec![DecoderLevel::new(conv, dec_stack)]);

    PhotonMamba::new(embed, encoder, decoder)
}

/// Compare `Generator` (padded re-forward) against `StatefulGenerator`
/// (O(1)-memory recurrent decode) under greedy sampling: must be
/// token-for-token identical.
fn assert_greedy_matches(model: &PhotonMamba<CandleBackend>, bk: &CandleBackend, prompt: &[i64]) {
    let chunk_product = CHUNK_SIZE; // L=2 => chunk_size^(L-1) = chunk_size
    let cfg = GenerateConfig {
        max_new_tokens: N_NEW_TOKENS,
        chunk_product,
        vocab_size: VOCAB,
        pad_token_id: 0,
        seed: 0, // unused by greedy sampling
    };

    let reforward = Generator::new(model, cfg.clone(), Sampler::greedy());
    let stateful = StatefulGenerator::new(model, cfg, Sampler::greedy());

    let out_reforward = reforward.generate(bk, prompt).unwrap();
    let out_stateful = stateful.generate(bk, prompt).unwrap();

    assert_eq!(out_reforward.len(), prompt.len() + N_NEW_TOKENS);
    assert_eq!(out_stateful.len(), prompt.len() + N_NEW_TOKENS);
    assert_eq!(
        out_reforward, out_stateful,
        "greedy StatefulGenerator diverged from Generator (padded re-forward):\n\
         reforward = {out_reforward:?}\n\
         stateful  = {out_stateful:?}"
    );
}

#[test]
fn greedy_stateful_matches_reforward_on_init() {
    let bk = CandleBackend::new_cpu();
    let model = make_model(&bk);
    seed_params(&bk, &model.collect_params(), 4242);

    let prompt: Vec<i64> = vec![1, 2, 3, 4, 5];
    assert_greedy_matches(&model, &bk, &prompt);
}

#[test]
fn greedy_stateful_matches_reforward_after_training() {
    let bk = CandleBackend::new_cpu();
    let model = make_model(&bk);
    let params = model.collect_params();
    seed_params(&bk, &params, 777);

    // A handful of real AdamW steps on a fixed random batch — "any
    // weights" per the Phase C acceptance criterion must include
    // trained ones, not just init.
    let seq_len = 32; // multiple of chunk_product
    let batch = 1;
    let raw = lcg_vec(99, batch * seq_len, 1.0, 0.0);
    let ids_data: Vec<i64> = raw
        .iter()
        .map(|&r| (r * VOCAB as f32) as i64 % VOCAB as i64)
        .collect();
    let mut targets_data = vec![0i64; batch * seq_len];
    for t in 0..seq_len {
        targets_data[t] = ids_data[(t + 1) % seq_len];
    }
    let ids = bk.from_slice_i64(&ids_data, &[batch, seq_len]).unwrap();
    let targets = bk.from_slice_i64(&targets_data, &[batch, seq_len]).unwrap();

    let optim = AdamW::new(
        &bk,
        &params,
        AdamWConfig {
            lr: 1e-3,
            ..AdamWConfig::default()
        },
    )
    .unwrap();
    let mut trainer = Trainer::new(optim);
    let mut losses = Vec::new();
    for _ in 0..5 {
        let loss = trainer
            .step_loss(&bk, &params, |o| {
                let out = model.forward(o, &ids)?;
                cross_entropy_loss(o, &out.logits, &targets)
            })
            .unwrap();
        losses.push(loss);
    }
    eprintln!("greedy_stateful_matches_reforward_after_training: losses={losses:?}");
    assert!(losses.iter().all(|l| l.is_finite()));

    let prompt: Vec<i64> = vec![1, 2, 3, 4, 5];
    assert_greedy_matches(&model, &bk, &prompt);
}
