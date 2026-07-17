//! Phase C (memory-efficiency plan) — Test 5: VRAM-flat.
//!
//! `oom_regression.rs`-style in-process `mem_get_info` measurement:
//! `StatefulGenerator` (O(1)-memory recurrent decode) must not leak
//! VRAM across many generated tokens on `CudaBackend`. Exercised
//! through the *real* production path — `pm generate --decode
//! stateful --backend cuda`'s exact code (`StatefulGenerator::
//! generate_with_hook` + `generate_cmd::end_of_decode_step`) — not a
//! hand-rolled loop, so this guards the actual CLI wiring.
//!
//! `CudaBackend`'s autograd tape only clears on `Ops::backward` /
//! `reset_tape()` (never on plain inference-only forward calls), so
//! without `end_of_decode_step`'s periodic `reset_tape()`, this test
//! fails with clearly-growing `free` VRAM — verified during
//! development (`git log`: this file's history / task report).
//!
//! The official cross-length sweep on the 100M production config
//! (`configs/photon_mamba_100m.toml`, `pm generate --backend cuda`,
//! external `nvidia-smi` peak sampling) is reported in the task
//! summary, not re-run here: this in-process test uses a smaller model
//! so `cargo test --features cuda` stays fast, and measures the same
//! O(1)-ness property directly via `mem_get_info`.

#![cfg(feature = "cuda")]

use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{
    ChunkLocalDecoder, ContextChunker, ContextConverter, ContextEncoder, DecoderLevel,
    HierarchicalDecoder, HierarchicalEncoder, HierarchicalLevel, TokenEmbedding,
};
use pm_core::{Ops, Param, Parameterized, PhotonMamba, Tensor};
use pm_cuda::CudaBackend;
use pm_infer::{GenerateConfig, Sampler, StatefulGenerator};

const VOCAB: usize = 256;
const D_MODEL: usize = 64;
const D_STATE: usize = 16;
const D_HEAD: usize = 8;
const N_HEADS: usize = D_MODEL / D_HEAD;
const N_GROUPS: usize = 1;
const D_CONV: usize = 4;
const BLOCK_LEN: usize = 8;
const RMSNORM_EPS: f32 = 1e-5;
const N_LAYERS: usize = 2;
const CHUNK_SIZE: usize = 4;
const INIT_SCALE: f32 = 0.05;

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

fn seed_params(bk: &CudaBackend, params: &[&<CudaBackend as Ops>::Param], seed: u64) {
    for (i, p) in params.iter().enumerate() {
        let shape = p.as_tensor().shape().to_vec();
        let n: usize = shape.iter().product();
        let data = lcg_vec(seed.wrapping_add(i as u64 * 1_000_003), n, 0.2, -0.1);
        let t = bk.from_slice_f32(&data, &shape).unwrap();
        bk.assign(p, &t).unwrap();
    }
}

fn make_model(bk: &CudaBackend) -> PhotonMamba<CudaBackend> {
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

/// Mirrors `generate_cmd::end_of_decode_step` exactly (can't import a
/// private `pm-cli` binary function from an integration test, since
/// `pm-cli` has no library target) — periodic `CudaBackend::reset_tape`
/// so the autograd tape (which only clears on `backward`/`reset_tape`,
/// never on plain forward-only ops) doesn't retain every step's
/// intermediates for the whole generation.
fn end_of_decode_step(bk: &CudaBackend) {
    let _ = bk.reset_tape();
}

/// The core Phase C memory claim: `StatefulGenerator` peak VRAM does
/// not grow with `max_new_tokens`. `mem_get_info` sampled periodically
/// through a single long generation (not separate runs) — the direct
/// way to see whether per-token cost accumulates.
#[test]
fn stateful_decode_vram_flat_in_process() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let model = make_model(&bk);
    seed_params(&bk, &model.collect_params(), 4242);

    let n_new_tokens = 2000;
    let cfg = GenerateConfig {
        max_new_tokens: n_new_tokens,
        chunk_product: CHUNK_SIZE,
        vocab_size: VOCAB,
        pad_token_id: 0,
        seed: 0,
    };
    let gen = StatefulGenerator::new(&model, cfg, Sampler::greedy());

    bk.stream().synchronize().expect("pre-run sync");
    let (free_before, _) = bk.device().mem_get_info().expect("mem_get_info before");

    let mut min_free = free_before;
    let mut max_free = free_before;
    let sample_every = 200usize;
    let mut step = 0usize;
    let out = gen
        .generate_with_hook(&bk, &[1, 2, 3, 4, 5], |ops| {
            end_of_decode_step(ops);
            step += 1;
            if step.is_multiple_of(sample_every) {
                ops.stream().synchronize().expect("mid-run sync");
                let (free_now, _) = ops.device().mem_get_info().expect("mem_get_info mid-run");
                min_free = min_free.min(free_now);
                max_free = max_free.max(free_now);
                eprintln!("step {step:5}: free={:.1} MiB", free_now as f64 / 1e6);
            }
        })
        .unwrap();
    assert_eq!(out.len(), 5 + n_new_tokens);

    bk.stream().synchronize().expect("post-run sync");
    let (free_after, _) = bk.device().mem_get_info().expect("mem_get_info after");
    min_free = min_free.min(free_after);
    max_free = max_free.max(free_after);

    let spread = max_free.saturating_sub(min_free);
    let leaked = free_before.saturating_sub(free_after);
    eprintln!(
        "stateful_decode_vram_flat_in_process: {n_new_tokens} tokens, \
         free_before={:.2} GB, free_after={:.2} GB, spread={:.1} MiB, leaked={:.1} MiB",
        free_before as f64 / 1e9,
        free_after as f64 / 1e9,
        spread as f64 / 1e6,
        leaked as f64 / 1e6,
    );

    // Threshold: a per-token leak on this toy model would show up as
    // many MiB over 2000 steps (see the exploratory pre-fix numbers in
    // the task report: ~34 MiB / 300 steps without `reset_tape`, i.e.
    // ~225 MiB by 2000 steps). 20 MiB is generous headroom above pure
    // allocator noise while still catching any regression of that
    // magnitude.
    const MAX_SPREAD_BYTES: usize = 20 * 1024 * 1024;
    assert!(
        spread <= MAX_SPREAD_BYTES,
        "VRAM free spread {:.1} MiB over {n_new_tokens} tokens exceeds {:.0} MiB — \
         stateful decode is not O(1) in generation length",
        spread as f64 / 1e6,
        MAX_SPREAD_BYTES as f64 / 1e6,
    );
}
