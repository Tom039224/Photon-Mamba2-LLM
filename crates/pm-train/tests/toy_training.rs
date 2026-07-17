//! F.4 toy training smoke test.
//!
//! Builds a tiny PhotonMamba, drives it for 10 AdamW steps on random
//! token ids with next-token cross-entropy, and asserts the loss
//! strictly decreases between the first and last step. This is the
//! end-to-end proof that:
//! - autograd reaches every trainable param (via `Ops::Param` + the
//!   pure-Ops ssd_scan landed in 79b6265),
//! - the optimiser actually updates them (via `Ops::assign`),
//! - the forward / backward / step loop is plumbed correctly.

use pm_candle::{CandleBackend, CandleParam};
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{
    ChunkLocalDecoder, ContextChunker, ContextConverter, ContextEncoder, DecoderLevel,
    HierarchicalDecoder, HierarchicalEncoder, HierarchicalLevel, TokenEmbedding,
};
use pm_core::{Ops, Param, Parameterized, PhotonMamba, Tensor};
use pm_train::{cross_entropy_loss, AdamW, AdamWConfig, Trainer};

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

fn build_toy_model(
    bk: &CandleBackend,
    vocab: usize,
    d: usize,
    c: usize,
) -> PhotonMamba<CandleBackend> {
    let embed = TokenEmbedding::from_constants(bk, vocab, d, 0.05).unwrap();

    let lvl0 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(vec![mk_block(bk, d)]),
        chunker: Some(ContextChunker::from_constants(bk, d, d, c, 0.05).unwrap()),
    };
    let lvl1 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(vec![mk_block(bk, d)]),
        chunker: None,
    };
    let encoder = HierarchicalEncoder::from_levels(vec![lvl0, lvl1]);

    let conv = ContextConverter::from_constants(bk, d, d, c, 0.05).unwrap();
    let dec_stack = ChunkLocalDecoder::from_layers(vec![mk_block(bk, d)], c, c);
    let decoder = HierarchicalDecoder::from_levels(vec![DecoderLevel::new(conv, dec_stack)]);

    PhotonMamba::new(embed, encoder, decoder)
}

/// Deterministic LCG-based random init. The `from_constants` factories
/// fill every weight with the same value, which is symmetric and
/// degenerate (every neuron computes the same thing). Perturbing breaks
/// that so the optimiser has a non-zero direction to descend.
fn perturb_params(bk: &CandleBackend, params: &[&CandleParam], seed: u64) -> anyhow::Result<()> {
    let mut state = seed.wrapping_add(1);
    for p in params {
        let shape = p.as_tensor().shape().to_vec();
        let n: usize = shape.iter().product();
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 33) as u32;
            // Map to [-0.05, 0.05]
            let x = (bits as f32 / u32::MAX as f32 - 0.5) * 0.1;
            data.push(x);
        }
        let t = bk.from_slice_f32(&data, &shape)?;
        bk.assign(p, &t)?;
    }
    Ok(())
}

#[test]
fn photon_mamba_toy_loss_decreases_over_10_adamw_steps() -> anyhow::Result<()> {
    let bk = CandleBackend::new_cpu();
    let vocab = 16;
    let d = 16;
    let c = 4;
    let model = build_toy_model(&bk, vocab, d, c);

    let params = model.collect_params();
    eprintln!("toy model parameter count: {}", params.len());
    perturb_params(&bk, &params, 0xDEAD_BEEF_C0FE_F00D)?;

    let optim = AdamW::new(
        &bk,
        &params,
        AdamWConfig {
            lr: 5e-3,
            ..Default::default()
        },
    )?;
    let mut trainer = Trainer::new(optim);

    // t=16 (not 8): with chunk_size 4, the level-1 stream is t/4 tokens
    // long, and Candle CPU's conv1d *backward* (grad_x via
    // conv_transpose1d) underflows its l_out computation when that
    // stream is shorter than the causal padding window (d_conv=4,
    // padding=3). t=8 → level-1 length 2 panics; t=16 → length 4 is the
    // smallest safe multiple. This was masked until the 2026-07-03
    // rms_norm_slow fix — before it, backward never reached the conv
    // (see CLAUDE.md "Candle の no-backward op").
    let (b, t) = (1, 16);
    let mut ids_data = vec![0i64; b * t];
    let mut rng_state: u64 = 0xCAFE_BABE_1234_5678;
    for slot in &mut ids_data {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *slot = (rng_state >> 33) as i64 % vocab as i64;
    }

    let ids = bk.from_slice_i64(&ids_data, &[b, t])?;
    // Target = next-token rotation: targets[t] = ids[(t+1) % T].
    let mut tgt_data = vec![0i64; b * t];
    for bi in 0..b {
        for ti in 0..t {
            tgt_data[bi * t + ti] = ids_data[bi * t + (ti + 1) % t];
        }
    }
    let targets = bk.from_slice_i64(&tgt_data, &[b, t])?;

    let mut losses = Vec::with_capacity(10);
    for step in 0..10 {
        let report = trainer.step(&bk, &params, |ops| {
            let out = model.forward(ops, &ids)?;
            cross_entropy_loss(ops, &out.logits, &targets)
        })?;
        eprintln!("step {step}: loss = {:.6}", report.loss);
        losses.push(report.loss);
    }

    let first = losses[0];
    let last = *losses.last().unwrap();
    assert!(
        first.is_finite() && last.is_finite(),
        "loss went non-finite: first={first}, last={last}"
    );
    assert!(
        last < first - 1e-3,
        "loss did not decrease meaningfully over 10 steps \
         (first={first:.6}, last={last:.6})"
    );

    Ok(())
}
