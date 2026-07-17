//! B4.4b — Candle (CPU) vs CudaBackend 10-step training loss parity.
//!
//! Runs a tiny PhotonMamba model (d_model=64, n_levels=2,
//! n_layers_per_level=1, seq_len=32) for 10 steps with each backend,
//! same seed, fixed batch, and asserts |Δloss| < 1e-2 per step.
//!
//! Build: `cargo test -p pm-cli --features cuda --test backend_parity`

#![cfg(feature = "cuda")]

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{
    ChunkLocalDecoder, ContextChunker, ContextConverter, ContextEncoder, DecoderLevel,
    HierarchicalDecoder, HierarchicalEncoder, HierarchicalLevel, TokenEmbedding,
};
use pm_core::{Ops, Param, Parameterized, PhotonMamba, Tensor};
use pm_cuda::CudaBackend;
use pm_train::{cross_entropy_loss, AdamW, AdamWConfig, Trainer};

// ── Tiny model dimensions ─────────────────────────────────────────────────────
const VOCAB: usize = 256;
const D_MODEL: usize = 64;
const D_STATE: usize = 16;
const D_HEAD: usize = 8;
const N_HEADS: usize = D_MODEL / D_HEAD; // 8
const N_GROUPS: usize = 1;
const D_CONV: usize = 4;
const BLOCK_LEN: usize = 4;
const RMSNORM_EPS: f32 = 1e-5;
const N_LAYERS: usize = 1;
const CHUNK_SIZE: usize = 4;
const SEQ_LEN: usize = 32; // must be divisible by CHUNK_SIZE^(N_LEVELS-1) = 4
const BATCH: usize = 1;
const INIT_SCALE: f32 = 0.05;
const N_STEPS: usize = 10;

/// Deterministic LCG — matches the pattern used in ssd_parity.rs.
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

/// Overwrite every parameter with deterministic LCG values.
/// Same algorithm as `perturb_params` in `train_cmd.rs`.
fn seed_params<O: Ops>(ops: &O, params: &[&O::Param], seed: u64) {
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
            let x = (bits as f32 / u32::MAX as f32 - 0.5) * 0.1;
            data.push(x);
        }
        let t = ops.from_slice_f32(&data, &shape).unwrap();
        ops.assign(p, &t).unwrap();
    }
}

/// Build a tiny PhotonMamba model, generic over any backend.
fn make_tiny_model<O: Ops>(ops: &O) -> PhotonMamba<O> {
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
    let mk_block = || Mamba2Block::from_constants(ops, m2cfg.clone(), INIT_SCALE).unwrap();

    let embed = TokenEmbedding::from_constants(ops, VOCAB, D_MODEL, INIT_SCALE).unwrap();

    let lvl0 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers((0..N_LAYERS).map(|_| mk_block()).collect()),
        chunker: Some(
            ContextChunker::from_constants(ops, D_MODEL, D_MODEL, CHUNK_SIZE, INIT_SCALE).unwrap(),
        ),
    };
    let lvl1 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers((0..N_LAYERS).map(|_| mk_block()).collect()),
        chunker: None,
    };
    let encoder = HierarchicalEncoder::from_levels(vec![lvl0, lvl1]);

    let conv =
        ContextConverter::from_constants(ops, D_MODEL, D_MODEL, CHUNK_SIZE, INIT_SCALE).unwrap();
    let dec_stack = ChunkLocalDecoder::from_layers(
        (0..N_LAYERS).map(|_| mk_block()).collect(),
        CHUNK_SIZE,
        CHUNK_SIZE,
    );
    let decoder = HierarchicalDecoder::from_levels(vec![DecoderLevel::new(conv, dec_stack)]);

    PhotonMamba::new(embed, encoder, decoder)
}

/// Build a fixed deterministic (ids, targets) batch, shared across steps.
/// Token ids in [0, VOCAB). Targets are shifted-by-one next-token ids.
fn fixed_batch(batch_seed: u64) -> (Vec<i64>, Vec<i64>) {
    let n = BATCH * SEQ_LEN;
    let raw = lcg_vec(batch_seed, n, 1.0, 0.0);
    let ids: Vec<i64> = raw
        .iter()
        .map(|&r| (r * VOCAB as f32) as i64 % VOCAB as i64)
        .collect();
    let mut targets = vec![0i64; n];
    for bi in 0..BATCH {
        for ti in 0..SEQ_LEN {
            targets[bi * SEQ_LEN + ti] = ids[bi * SEQ_LEN + (ti + 1) % SEQ_LEN];
        }
    }
    (ids, targets)
}

/// Run N_STEPS of AdamW training and return per-step losses.
fn run_n_steps<O: Ops>(ops: O) -> Vec<f32> {
    // Build model and seed params identically.
    let model = make_tiny_model(&ops);
    let params = model.collect_params();
    seed_params(&ops, &params, 42);

    // Build fixed batch (same tokens for every step).
    let (ids_data, targets_data) = fixed_batch(7);
    let ids = ops.from_slice_i64(&ids_data, &[BATCH, SEQ_LEN]).unwrap();
    let targets = ops
        .from_slice_i64(&targets_data, &[BATCH, SEQ_LEN])
        .unwrap();

    // AdamW with small lr so floating-point order differences don't amplify.
    let optim = AdamW::new(
        &ops,
        &params,
        AdamWConfig {
            lr: 1e-3,
            ..AdamWConfig::default()
        },
    )
    .unwrap();
    let mut trainer = Trainer::new(optim);

    let mut losses = Vec::with_capacity(N_STEPS);
    for _ in 0..N_STEPS {
        let loss = trainer
            .step_loss(&ops, &params, |o| {
                let out = model.forward(o, &ids)?;
                cross_entropy_loss(o, &out.logits, &targets)
            })
            .unwrap();
        losses.push(loss);
    }
    losses
}

#[test]
fn backend_parity_candle_vs_cuda_10_step() {
    let losses_candle = run_n_steps(CandleBackend::new_cpu());
    let losses_cuda = run_n_steps(CudaBackend::new(0).unwrap());

    let mut max_abs = 0.0f32;
    for (i, (&lc, &lu)) in losses_candle.iter().zip(losses_cuda.iter()).enumerate() {
        let abs = (lc - lu).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        eprintln!("step {i:2}: candle={lc:.6}  cuda={lu:.6}  |Δ|={abs:.3e}");
    }
    eprintln!("max |Δ| over {N_STEPS} steps = {max_abs:.3e}");

    // Tolerance: 1e-2.
    // fp32 + cuBLAS vs Candle can diverge due to fma-order differences in
    // matmul (cuBLAS uses tensor-core accumulators; Candle uses sequential
    // fp32 dots). Empirically the per-step diff stays well below 1e-2 on
    // this tiny model. If this assertion ever fails, check whether a new op
    // introduced a non-deterministic path (e.g. atomics in backward) before
    // widening the tolerance.
    assert!(
        max_abs < 1e-2,
        "max step loss differs by {max_abs:.3e} which exceeds 1e-2; \
         check op-level parity (seed_params, fixed batch, Adam state)"
    );
}
