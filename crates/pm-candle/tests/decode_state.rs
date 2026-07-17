//! Phase C (memory-efficiency plan): O(1)-memory recurrent decode.
//!
//! TDD order (each gate before the next — see the Phase C spec):
//! 1. `step_parity_*`      — `Mamba2Block::step × T == Mamba2Block::forward`.
//! 2. `encoder_step_*`     — `ContextEncoder::step × T == ContextEncoder::forward`.
//! 3. `prefill_predict_*`  — `PhotonMamba::{prefill, predict_logits}` ==
//!    `PhotonMamba::forward` on the pad-completed prompt (the pad-completion
//!    crux: `predict_logits` must reproduce `Generator::step`'s whole-sequence
//!    padding one token early, non-destructively).
//!
//! All comparisons run on the Candle CPU backend, fp32, tolerance 1e-4
//! (CLAUDE.md invariant #3).

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config, Mamba2State};
use pm_core::photon::{
    ChunkLocalDecoder, ContextChunker, ContextConverter, ContextEncoder, DecoderLevel,
    HierarchicalDecoder, HierarchicalEncoder, HierarchicalLevel, TokenEmbedding,
};
use pm_core::{Module, Ops, Param, Parameterized, PhotonMamba, Tensor};

// ---- shared helpers --------------------------------------------------------

/// Deterministic LCG — matches the pattern used throughout the workspace
/// (e.g. `pm-cli/tests/backend_parity.rs`, `pm-candle/tests/rmsnorm_backward.rs`).
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

/// Overwrite every parameter with deterministic LCG values — breaks
/// `from_constants`' uniform-fill symmetry so a broken `step()` can't
/// accidentally agree with `forward()` by degenerate cancellation.
fn seed_params(bk: &CandleBackend, params: &[&<CandleBackend as Ops>::Param], seed: u64) {
    for (i, p) in params.iter().enumerate() {
        let shape = p.as_tensor().shape().to_vec();
        let n: usize = shape.iter().product();
        let data = lcg_vec(seed.wrapping_add(i as u64 * 1_000_003), n, 0.2, -0.1);
        let t = bk.from_slice_f32(&data, &shape).unwrap();
        bk.assign(p, &t).unwrap();
    }
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

// ---- Test 1: Mamba2Block::step × T == Mamba2Block::forward (GATE) ---------

fn run_step_parity(cfg: Mamba2Config, t_len: usize, seed: u64) -> f32 {
    let bk = CandleBackend::new_cpu();
    let block = Mamba2Block::from_constants(&bk, cfg.clone(), 0.05).unwrap();
    seed_params(&bk, &block.collect_params(), seed);

    let b = 1;
    let x_data = lcg_vec(seed.wrapping_add(999), b * t_len * cfg.d_model, 1.0, 0.0);
    let x = bk
        .from_slice_f32(&x_data, &[b, t_len, cfg.d_model])
        .unwrap();

    let y_ref = block.forward(&bk, &x).unwrap();
    let y_ref_host = bk.to_vec_f32(&y_ref).unwrap();

    let mut state = Mamba2State::zeros(&bk, &cfg, b).unwrap();
    let mut y_stepped_host = Vec::with_capacity(b * t_len * cfg.d_model);
    for t in 0..t_len {
        let x_t = bk.narrow(&x, 1, t, 1).unwrap();
        let (y_t, next_state) = block.step(&bk, &x_t, &state).unwrap();
        y_stepped_host.extend(bk.to_vec_f32(&y_t).unwrap());
        state = next_state;
    }

    let err = max_abs_err(&y_ref_host, &y_stepped_host);
    eprintln!(
        "step_parity: T={t_len} n_groups={} max_abs_err={err:.3e}",
        cfg.n_groups
    );
    err
}

#[test]
fn step_parity_matches_forward_small_t() {
    let cfg = Mamba2Config {
        d_model: 32,
        d_state: 16,
        d_head: 8,
        n_heads: 4,
        n_groups: 1,
        d_conv: 4,
        block_len: 4,
        rmsnorm_eps: 1e-5,
    };
    let err = run_step_parity(cfg, 8, 11);
    assert!(err < 1e-4, "max abs err {err:.3e} exceeds 1e-4");
}

#[test]
fn step_parity_matches_forward_n_groups_equals_n_heads() {
    let cfg = Mamba2Config {
        d_model: 32,
        d_state: 16,
        d_head: 8,
        n_heads: 4,
        n_groups: 4, // == n_heads: exercises the no-broadcast path
        d_conv: 4,
        block_len: 4,
        rmsnorm_eps: 1e-5,
    };
    let err = run_step_parity(cfg, 12, 23);
    assert!(err < 1e-4, "max abs err {err:.3e} exceeds 1e-4");
}

/// The GATE test: T=512 with a production-shaped `block_len=64` (8 chunks),
/// so `forward` exercises the chunked SSD path (`ssd_scan_chunked`), whose
/// batched-matmul summation order differs from `step`'s 512 sequential
/// fp32 accumulations. Checks that fp32 rounding does not drift the two
/// paths apart over many steps.
#[test]
fn step_parity_matches_forward_t512_drift() {
    let cfg = Mamba2Config {
        d_model: 32,
        d_state: 16,
        d_head: 8,
        n_heads: 4,
        n_groups: 1,
        d_conv: 4,
        block_len: 64,
        rmsnorm_eps: 1e-5,
    };
    let err = run_step_parity(cfg, 512, 7);
    assert!(err < 1e-4, "max abs err {err:.3e} exceeds 1e-4 at T=512");
}

// ---- Test 2: ContextEncoder::step × T == ContextEncoder::forward ----------

fn mk_block(bk: &CandleBackend, d_model: usize) -> Mamba2Block<CandleBackend> {
    Mamba2Block::from_constants(
        bk,
        Mamba2Config {
            d_model,
            d_state: 8,
            d_head: 8,
            n_heads: d_model / 8,
            n_groups: 1,
            d_conv: 4,
            block_len: 8,
            rmsnorm_eps: 1e-5,
        },
        0.05,
    )
    .unwrap()
}

#[test]
fn encoder_step_matches_forward() {
    let bk = CandleBackend::new_cpu();
    let d_model = 24;
    let n_layers = 3;
    let layers: Vec<_> = (0..n_layers).map(|_| mk_block(&bk, d_model)).collect();
    let encoder = ContextEncoder::from_layers(layers);
    seed_params(&bk, &encoder.collect_params(), 101);

    let (b, t) = (1, 64);
    let x_data = lcg_vec(202, b * t * d_model, 1.0, 0.0);
    let x = bk.from_slice_f32(&x_data, &[b, t, d_model]).unwrap();

    let y_ref = encoder.forward(&bk, &x).unwrap();
    let y_ref_host = bk.to_vec_f32(&y_ref).unwrap();

    let mut state = encoder.zero_state(&bk, b).unwrap();
    let mut y_stepped_host = Vec::with_capacity(b * t * d_model);
    for t_idx in 0..t {
        let x_t = bk.narrow(&x, 1, t_idx, 1).unwrap();
        let (y_t, next_state) = encoder.step(&bk, &x_t, &state).unwrap();
        y_stepped_host.extend(bk.to_vec_f32(&y_t).unwrap());
        state = next_state;
    }

    let err = max_abs_err(&y_ref_host, &y_stepped_host);
    eprintln!("encoder_step_parity: max_abs_err={err:.3e}");
    assert!(err < 1e-4, "max abs err {err:.3e} exceeds 1e-4");
}

// ---- Test 3: prefill + predict_logits == forward(pad_up(prompt)) ----------
// THE pad-completion crux test.

struct ToyDims {
    vocab: usize,
    d_model: usize,
    chunk_size: usize,
    n_layers_per_level: usize,
}

fn build_toy_photon_mamba(bk: &CandleBackend, dims: &ToyDims) -> PhotonMamba<CandleBackend> {
    let mk = || mk_block(bk, dims.d_model);
    let embed = TokenEmbedding::from_constants(bk, dims.vocab, dims.d_model, 0.05).unwrap();

    let lvl0 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers((0..dims.n_layers_per_level).map(|_| mk()).collect()),
        chunker: Some(
            ContextChunker::from_constants(bk, dims.d_model, dims.d_model, dims.chunk_size, 0.05)
                .unwrap(),
        ),
    };
    let lvl1 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers((0..dims.n_layers_per_level).map(|_| mk()).collect()),
        chunker: None,
    };
    let encoder = HierarchicalEncoder::from_levels(vec![lvl0, lvl1]);

    let conv =
        ContextConverter::from_constants(bk, dims.d_model, dims.d_model, dims.chunk_size, 0.05)
            .unwrap();
    let dec_stack = ChunkLocalDecoder::from_layers(
        (0..dims.n_layers_per_level).map(|_| mk()).collect(),
        dims.chunk_size,
        dims.chunk_size,
    );
    let decoder = HierarchicalDecoder::from_levels(vec![DecoderLevel::new(conv, dec_stack)]);

    PhotonMamba::new(embed, encoder, decoder)
}

fn pad_up(n: usize, multiple: usize) -> usize {
    if n.is_multiple_of(multiple) {
        n.max(multiple)
    } else {
        ((n / multiple) + 1) * multiple
    }
}

#[test]
fn prefill_predict_matches_forward_pad_completion() {
    let bk = CandleBackend::new_cpu();
    let dims = ToyDims {
        vocab: 64,
        d_model: 16,
        chunk_size: 4,
        n_layers_per_level: 2,
    };
    let model = build_toy_photon_mamba(&bk, &dims);
    seed_params(&bk, &model.collect_params(), 4242);

    let pad_token_id: i64 = 0;
    // Covers: mid chunk 0 (k=0, j=0..2), end of chunk 0 (k=0,j=3), start
    // of chunk 1 (k=1,j=0 — exercises `prev_chunk_embeds`), mid/end of
    // chunk 1, and into chunk 2 and 3.
    let prompt_lens = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17];

    // Same token stream for every prefix length so the comparison is
    // apples-to-apples (a fixed "generation so far").
    let max_len = *prompt_lens.iter().max().unwrap();
    let all_ids: Vec<i64> = (0..max_len)
        .map(|i| ((i * 7 + 3) % dims.vocab) as i64)
        .collect();

    let mut worst = 0f32;
    for &len in &prompt_lens {
        let prompt = &all_ids[..len];

        // Reference: forward() on the pad-completed prompt.
        let padded_len = pad_up(len, dims.chunk_size);
        let mut buf = vec![pad_token_id; padded_len];
        buf[..len].copy_from_slice(prompt);
        let ids_padded = bk.from_slice_i64(&buf, &[1, padded_len]).unwrap();
        let out = model.forward(&bk, &ids_padded).unwrap();
        let logits_ref_row = bk.narrow(&out.logits, 1, len - 1, 1).unwrap();
        let logits_ref = bk.to_vec_f32(&logits_ref_row).unwrap();

        // Stateful: prefill + predict_logits.
        let state = model.prefill(&bk, prompt, pad_token_id).unwrap();
        let logits_stateful_t = model.predict_logits(&bk, &state).unwrap();
        assert_eq!(logits_stateful_t.shape(), &[1, 1, dims.vocab]);
        let logits_stateful = bk.to_vec_f32(&logits_stateful_t).unwrap();

        let err = max_abs_err(&logits_ref, &logits_stateful);
        eprintln!("prefill_predict: len={len} padded_len={padded_len} max_abs_err={err:.3e}");
        worst = worst.max(err);
        assert!(
            err < 1e-4,
            "len={len}: max abs err {err:.3e} exceeds 1e-4 (padded_len={padded_len})"
        );
    }
    eprintln!("prefill_predict_matches_forward_pad_completion: worst max_abs_err={worst:.3e}");
}
