//! Smoke tests for Generator + score_continuation against a tiny
//! random-init PhotonMamba.

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{
    ChunkLocalDecoder, ContextChunker, ContextConverter, ContextEncoder, DecoderLevel,
    HierarchicalDecoder, HierarchicalEncoder, HierarchicalLevel, TokenEmbedding,
};
use pm_core::PhotonMamba;
use pm_infer::{score_continuation, GenerateConfig, Generator, Sampler};

fn build_toy_model(
    bk: &CandleBackend,
    vocab: usize,
    d: usize,
    c: usize,
) -> PhotonMamba<CandleBackend> {
    let mk_block = || {
        Mamba2Block::from_constants(
            bk,
            Mamba2Config {
                d_model: d,
                d_state: 8,
                d_head: 8,
                n_heads: d / 8,
                n_groups: 1,
                d_conv: 4,
                block_len: 4,
                rmsnorm_eps: 1e-5,
            },
            0.05,
        )
        .unwrap()
    };
    let embed = TokenEmbedding::from_constants(bk, vocab, d, 0.05).unwrap();
    let lvl0 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(vec![mk_block()]),
        chunker: Some(ContextChunker::from_constants(bk, d, d, c, 0.05).unwrap()),
    };
    let lvl1 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(vec![mk_block()]),
        chunker: None,
    };
    let encoder = HierarchicalEncoder::from_levels(vec![lvl0, lvl1]);
    let conv = ContextConverter::from_constants(bk, d, d, c, 0.05).unwrap();
    let dec_stack = ChunkLocalDecoder::from_layers(vec![mk_block()], c, c);
    let decoder = HierarchicalDecoder::from_levels(vec![DecoderLevel::new(conv, dec_stack)]);
    PhotonMamba::new(embed, encoder, decoder)
}

#[test]
fn generator_produces_expected_length_and_in_vocab_ids() -> anyhow::Result<()> {
    let bk = CandleBackend::new_cpu();
    let vocab = 16;
    let d = 16;
    let c = 4;
    let model = build_toy_model(&bk, vocab, d, c);

    let cfg = GenerateConfig {
        max_new_tokens: 5,
        chunk_product: c, // L=2 → chunk_product = c^(L-1) = c
        vocab_size: vocab,
        pad_token_id: 0,
        seed: 42,
    };
    let gen = Generator::new(&model, cfg.clone(), Sampler::greedy());
    let prompt = vec![1i64, 2, 3];
    let out = gen.generate(&bk, &prompt)?;
    assert_eq!(out.len(), prompt.len() + cfg.max_new_tokens);
    for &id in &out {
        assert!(id >= 0 && (id as usize) < vocab);
    }
    Ok(())
}

#[test]
fn greedy_generation_is_deterministic() -> anyhow::Result<()> {
    let bk = CandleBackend::new_cpu();
    let vocab = 8;
    let model = build_toy_model(&bk, vocab, 16, 4);
    let cfg = GenerateConfig {
        max_new_tokens: 4,
        chunk_product: 4,
        vocab_size: vocab,
        pad_token_id: 0,
        seed: 1234,
    };
    let gen = Generator::new(&model, cfg.clone(), Sampler::greedy());
    let prompt = vec![1i64, 2, 3, 4];
    let a = gen.generate(&bk, &prompt)?;
    let b = gen.generate(&bk, &prompt)?;
    assert_eq!(a, b);
    Ok(())
}

#[test]
fn score_continuation_is_finite() -> anyhow::Result<()> {
    let bk = CandleBackend::new_cpu();
    let vocab = 8;
    let model = build_toy_model(&bk, vocab, 16, 4);
    let ctx = vec![1i64, 2, 3, 4];
    let cont = vec![5i64, 6];
    let (sum, mean) = score_continuation(&bk, &model, 4, 0, &ctx, &cont)?;
    assert!(sum.is_finite() && mean.is_finite());
    assert!(mean <= 0.0, "log-prob must be ≤ 0, got {mean}");
    Ok(())
}
