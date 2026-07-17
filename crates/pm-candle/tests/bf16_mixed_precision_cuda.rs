//! Phase A2: bf16 mixed-precision forward compute — model-level
//! numerical parity + grad-flow tests (memory-efficiency plan,
//! `docs/perf-log.md` 2026-07-03 entry,
//! `plans/fancy-enchanting-lamport.md` Phase A2).
//!
//! Requires `--features cuda` (`cargo test -p pm-candle --features cuda
//! --test bf16_mixed_precision_cuda`): Candle 0.11's CPU backend
//! restricts `matmul` to `F16 | F32 | F64` (bf16 unsupported); the CUDA
//! backend supports bf16 matmul via cuBLAS. See
//! `bf16_mixed_precision.rs`'s module docs for the full explanation.
//! This matches the real deployment target anyway — the OOM problem
//! this feature addresses only exists on GPU.
//!
//! Strategy under test ("fp32 islands"): `PhotonMamba::compute_dtype =
//! BF16` flows the big `(B,T,·)` activations and the `in_proj`/
//! `out_proj` matmuls in bf16, while `softplus(dt)`, `exp(a_log)`,
//! `rmsnorm`, `conv1d`, and the entire `ssd_scan` call stay fp32
//! internally (see `Mamba2Block::forward` docs). Parameters stay fp32
//! in storage; the optimizer is unaffected. The default
//! (`compute_dtype = F32`, i.e. every other existing test in this
//! crate, which never opts into bf16) must stay bit-identical — that
//! is covered by the full existing suite continuing to pass unchanged.
//!
//! CLAUDE.md invariant #3: fp16 tolerance is 1e-2 (max relative error).
//!
//! ### A note on grad-flow coverage
//!
//! Historical (fixed): `candle_nn::ops::rms_norm` (candle-nn 0.11.0) is
//! built via `Tensor::apply_op2_no_bwd` (`BackpropOp::none()`) — **no
//! backward**. While `pm-candle` used it, gradients reached only
//! `out_proj_weight` per `Mamba2Block` + the tied embedding; every
//! pre-norm parameter got `None` and was skipped by the optimiser.
//! `ops_impl.rs::rmsnorm` now uses the differentiable `rms_norm_slow`,
//! so **all** parameters receive gradients — see the dedicated
//! regression tests in `tests/rmsnorm_backward.rs`
//! (`mamba2_block_backward_reaches_all_params`, plus an FD grad-check).
//!
//! The grad-flow tests below therefore now exercise every parameter in
//! both dtypes. They assert (a) backward runs with no dtype-mismatch
//! error under bf16 (the actual risk *this* change introduces), (b)
//! grads are finite, and (c) bf16 does not change *which* params get a
//! gradient relative to fp32 (the presence sets match — now all-present
//! rather than the old buggy subset).

#![cfg(feature = "cuda")]

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{
    ChunkLocalDecoder, ContextChunker, ContextConverter, ContextEncoder, DecoderLevel,
    HierarchicalDecoder, HierarchicalEncoder, HierarchicalLevel, TokenEmbedding,
};
use pm_core::{Dtype, Module, Ops, Param, Parameterized, PhotonMamba, Tensor};
use pm_train::cross_entropy_loss;

/// fp16/bf16 numerical-parity budget for FORWARD outputs (CLAUDE.md
/// invariant #3).
const REL_TOL: f32 = 1e-2;

/// Looser budget for GRADIENT parity. A parameter gradient flowing back
/// through bf16-stored activations accumulates more rounding than the
/// forward pass — the bf16-rounded activation feeds every downstream
/// backward reduction. Measured on this tiny block (fp32 vs bf16 grad,
/// zero-thresholded rel err): `in_proj_weight` 1.17e-2, `conv1d_weight`
/// 7.7e-3, `conv1d_bias` 3.4e-2 (a per-channel bias = a sum over all
/// B*T positions, so it magnifies input rounding most). All abs errors
/// are < 1.3e-3. This is normal bf16 mixed-precision behaviour, not a
/// VJP bug — a real backward bug (wrong axis / formula) would be orders
/// of magnitude larger or non-finite. The budget guards against that
/// while accepting bf16 reality; keeping these ops in bf16 is the whole
/// point (memory). Forward/loss parity stays at the tight `REL_TOL`.
const GRAD_REL_TOL: f32 = 5e-2;

// ---------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------

fn cuda() -> CandleBackend {
    CandleBackend::new_cuda(0).expect("RTX 5070 must be available for this test (CLAUDE.md env)")
}

/// Deterministic LCG, matching `pm-cli::train_cmd::perturb_params` /
/// `pm-cli/tests/backend_parity.rs::seed_params` so weight symmetry is
/// broken the same way the rest of the codebase already relies on for
/// gradient-check tests.
fn lcg_vec(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 33) as u32;
            (bits as f32 / u32::MAX as f32 - 0.5) * scale
        })
        .collect()
}

fn seed_params(ops: &CandleBackend, params: &[&<CandleBackend as Ops>::Param], seed: u64) {
    for p in params {
        let shape = p.as_tensor().shape().to_vec();
        let n: usize = shape.iter().product();
        let data = lcg_vec(seed, n, 0.1);
        let t = ops.from_slice_f32(&data, &shape).unwrap();
        ops.assign(p, &t).unwrap();
    }
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in comparison");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Max relative error, skipping elements where both `a[i]` and `b[i]`
/// are near zero — same convention as
/// `pm-cuda/tests/b44f_op_parity_prod_shape.rs::max_rel_err`. A pure
/// relative metric blows up at zero-crossings (e.g. an
/// `out_proj_weight` gradient row that is a near-cancelling sum over
/// `normed` — the same "cancellation noise" the B4.4f T5 FD-eps fix
/// (`b5895c8`) hit), which is not a real precision problem.
fn max_rel_err(a: &[f32], b: &[f32], zero_thresh: f32) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in comparison");
    a.iter()
        .zip(b.iter())
        .filter(|(&x, &y)| x.abs() > zero_thresh || y.abs() > zero_thresh)
        .map(|(&x, &y)| (x - y).abs() / x.abs().max(y.abs()))
        .fold(0f32, f32::max)
}

fn tiny_mamba2_cfg() -> Mamba2Config {
    Mamba2Config {
        d_model: 32,
        d_state: 16,
        d_head: 8,
        n_heads: 4, // d_inner = 32
        n_groups: 1,
        d_conv: 4,
        block_len: 8,
        rmsnorm_eps: 1e-5,
    }
}

struct ToyDims {
    vocab: usize,
    d_model: usize,
    chunk_size: usize,
    n_layers_per_level: usize,
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

/// Same 2-level toy-model shape as `photon_model.rs::build_toy_model`.
fn build_toy_model(bk: &CandleBackend, dims: &ToyDims) -> PhotonMamba<CandleBackend> {
    let embed = TokenEmbedding::from_constants(bk, dims.vocab, dims.d_model, 0.05).unwrap();

    let lvl0 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(
            (0..dims.n_layers_per_level)
                .map(|_| mk_block(bk, dims.d_model))
                .collect(),
        ),
        chunker: Some(
            ContextChunker::from_constants(bk, dims.d_model, dims.d_model, dims.chunk_size, 0.05)
                .unwrap(),
        ),
    };
    let lvl1 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(
            (0..dims.n_layers_per_level)
                .map(|_| mk_block(bk, dims.d_model))
                .collect(),
        ),
        chunker: None,
    };
    let encoder = HierarchicalEncoder::from_levels(vec![lvl0, lvl1]);

    let conv =
        ContextConverter::from_constants(bk, dims.d_model, dims.d_model, dims.chunk_size, 0.05)
            .unwrap();
    let dec_stack = ChunkLocalDecoder::from_layers(
        (0..dims.n_layers_per_level)
            .map(|_| mk_block(bk, dims.d_model))
            .collect(),
        dims.chunk_size,
        dims.chunk_size,
    );
    let decoder = HierarchicalDecoder::from_levels(vec![DecoderLevel::new(conv, dec_stack)]);

    PhotonMamba::new(embed, encoder, decoder)
}

/// Deterministic `(ids, targets)` batch — targets are shifted-by-one
/// next-token ids within each row (matches
/// `pm-cli::train_cmd::next_random_batch`).
fn make_ids_targets(b: usize, t: usize, vocab: usize) -> (Vec<i64>, Vec<i64>) {
    let ids: Vec<i64> = (0..b * t)
        .map(|i| (i as i64 * 7 + 3) % vocab as i64)
        .collect();
    let mut targets = vec![0i64; b * t];
    for bi in 0..b {
        for ti in 0..t {
            targets[bi * t + ti] = ids[bi * t + (ti + 1) % t];
        }
    }
    (ids, targets)
}

// ---------------------------------------------------------------------
// Test 1: numerical parity — Mamba2Block forward, fp32 vs bf16.
// ---------------------------------------------------------------------

#[test]
fn mamba2_block_bf16_forward_matches_fp32() {
    let bk = cuda();
    let cfg = tiny_mamba2_cfg();
    let block = Mamba2Block::from_constants(&bk, cfg.clone(), 0.1).unwrap();
    seed_params(&bk, &block.collect_params(), 1234);

    let (b, t) = (2, 16);
    let x_data: Vec<f32> = (0..b * t * cfg.d_model)
        .map(|i| (i as f32 * 0.037).sin() * 0.5)
        .collect();
    let x_f32 = bk.from_slice_f32(&x_data, &[b, t, cfg.d_model]).unwrap();
    let x_bf16 = bk.to_dtype(&x_f32, Dtype::BF16).unwrap();

    let y_f32 = block.forward(&bk, &x_f32).unwrap();
    let y_bf16 = block.forward(&bk, &x_bf16).unwrap();
    assert_eq!(y_f32.dtype(), Dtype::F32);
    assert_eq!(
        y_bf16.dtype(),
        Dtype::BF16,
        "bf16 input must produce a bf16 block output"
    );
    assert_eq!(y_f32.shape(), y_bf16.shape());

    let v_f32 = bk.to_vec_f32(&y_f32).unwrap();
    let v_bf16 = bk.to_vec_f32(&y_bf16).unwrap();
    assert!(
        v_f32.iter().all(|v| v.is_finite()),
        "fp32 forward produced non-finite output"
    );
    assert!(
        v_bf16.iter().all(|v| v.is_finite()),
        "bf16 forward produced non-finite output"
    );

    let abs_err = max_abs_err(&v_f32, &v_bf16);
    let rel_err = max_rel_err(&v_f32, &v_bf16, REL_TOL);
    eprintln!(
        "mamba2_block_bf16_forward_matches_fp32: max abs err = {abs_err:.4e}, max rel err (|v|>{REL_TOL:e}) = {rel_err:.4e}"
    );
    assert!(
        abs_err < REL_TOL,
        "max abs err {abs_err:.4e} exceeds fp16 budget {REL_TOL:e}"
    );
    assert!(
        rel_err < REL_TOL,
        "max rel err {rel_err:.4e} exceeds fp16 budget {REL_TOL:e}"
    );
}

// ---------------------------------------------------------------------
// Test 2: grad flow (backward) — Mamba2Block, fp32 vs bf16.
// ---------------------------------------------------------------------

#[test]
fn mamba2_block_bf16_backward_matches_fp32_grad_presence_and_is_finite() {
    let bk = cuda();
    let cfg = tiny_mamba2_cfg();

    let run = |dtype: Dtype| -> (f32, Vec<Option<Vec<f32>>>) {
        let block = Mamba2Block::from_constants(&bk, cfg.clone(), 0.1).unwrap();
        seed_params(&bk, &block.collect_params(), 1234);

        let (b, t) = (2, 16);
        let x_data: Vec<f32> = (0..b * t * cfg.d_model)
            .map(|i| (i as f32 * 0.037).sin() * 0.5)
            .collect();
        let x_f32 = bk.from_slice_f32(&x_data, &[b, t, cfg.d_model]).unwrap();
        let x = bk.to_dtype(&x_f32, dtype).unwrap();

        let y = block.forward(&bk, &x).unwrap();
        let loss = bk.sum_all(&y).unwrap();
        let loss_val = bk.to_vec_f32(&loss).unwrap()[0];
        let grads = bk.backward(&loss).unwrap();

        let params = block.collect_params();
        let g = params
            .iter()
            .map(|p| {
                bk.gradient(&grads, p)
                    .unwrap()
                    .map(|t| bk.to_vec_f32(&t).unwrap())
            })
            .collect();
        (loss_val, g)
    };

    let (loss_f32, g_f32) = run(Dtype::F32);
    let (loss_bf16, g_bf16) = run(Dtype::BF16);

    let names = [
        "in_proj_weight",
        "conv1d_weight",
        "conv1d_bias",
        "a_log",
        "d_skip",
        "dt_bias",
        "norm_weight",
        "out_proj_weight",
    ];
    assert_eq!(g_f32.len(), names.len());

    let mut n_with_grad = 0usize;
    for (name, (a, b)) in names.iter().zip(g_f32.iter().zip(g_bf16.iter())) {
        match (a, b) {
            (Some(ga), Some(gb)) => {
                n_with_grad += 1;
                assert!(
                    ga.iter().all(|v| v.is_finite()),
                    "{name}: fp32 grad has non-finite entries"
                );
                assert!(
                    gb.iter().all(|v| v.is_finite()),
                    "{name}: bf16 grad has non-finite entries"
                );
                // Combined abs/rel check (matches
                // `pm-cuda/tests/b44f_op_parity_prod_shape.rs`
                // convention): a pure relative metric blows up on
                // near-cancelling gradient sums (e.g. `out_proj_weight`
                // rows = a sum over `normed` values across B*T
                // positions), which is noise, not a real precision bug
                // — same class of issue as the B4.4f T5 FD-eps
                // recalibration (`b5895c8`).
                let abs_err = max_abs_err(ga, gb);
                let rel_err = max_rel_err(ga, gb, REL_TOL);
                eprintln!("{name}: grad max abs err = {abs_err:.4e}, max rel err = {rel_err:.4e}");
                assert!(
                    abs_err < REL_TOL,
                    "{name}: grad max abs err {abs_err:.4e} exceeds fp16 budget {REL_TOL:e}"
                );
                assert!(
                    rel_err < GRAD_REL_TOL,
                    "{name}: grad max rel err {rel_err:.4e} exceeds bf16 grad budget {GRAD_REL_TOL:e}"
                );
            }
            (None, None) => {
                // Pre-existing rms_norm-no-backward gap (see module
                // docs) — consistent absence in both dtypes is fine.
            }
            (ga, gb) => panic!(
                "{name}: gradient presence differs between fp32 ({}) and bf16 ({}) — the \
                 bf16 fp32-island casts introduced a NEW autograd break",
                ga.is_some(),
                gb.is_some()
            ),
        }
    }
    assert!(
        n_with_grad > 0,
        "no parameter received a gradient in either dtype — vacuous test"
    );

    let loss_err = (loss_f32 - loss_bf16).abs() / loss_f32.abs().max(1e-6);
    assert!(
        loss_err < REL_TOL,
        "loss fp32={loss_f32} vs bf16={loss_bf16}, rel err {loss_err:.4e} exceeds {REL_TOL:e}"
    );
}

// ---------------------------------------------------------------------
// Test 3: numerical parity — full PhotonMamba forward + CE loss.
// ---------------------------------------------------------------------

#[test]
fn photon_mamba_bf16_forward_and_loss_match_fp32() {
    let bk = cuda();
    let dims = ToyDims {
        vocab: 64,
        d_model: 32,
        chunk_size: 4,
        n_layers_per_level: 2,
    };

    let model_f32 = build_toy_model(&bk, &dims);
    seed_params(&bk, &model_f32.collect_params(), 777);

    let model_bf16 = build_toy_model(&bk, &dims).with_compute_dtype(Dtype::BF16);
    seed_params(&bk, &model_bf16.collect_params(), 777);

    let (b, t) = (2, 16);
    let (ids_data, targets_data) = make_ids_targets(b, t, dims.vocab);
    let ids = bk.from_slice_i64(&ids_data, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&targets_data, &[b, t]).unwrap();

    let out_f32 = model_f32.forward(&bk, &ids).unwrap();
    let out_bf16 = model_bf16.forward(&bk, &ids).unwrap();
    assert_eq!(out_f32.logits.dtype(), Dtype::F32);
    assert_eq!(
        out_bf16.logits.dtype(),
        Dtype::BF16,
        "logits must flow in the ambient compute dtype"
    );

    let logits_f32 = bk.to_vec_f32(&out_f32.logits).unwrap();
    let logits_bf16 = bk.to_vec_f32(&out_bf16.logits).unwrap();
    assert!(logits_f32.iter().all(|v| v.is_finite()));
    assert!(
        logits_bf16.iter().all(|v| v.is_finite()),
        "bf16 logits non-finite"
    );
    let logit_abs_err = max_abs_err(&logits_f32, &logits_bf16);
    let logit_rel_err = max_rel_err(&logits_f32, &logits_bf16, REL_TOL);
    eprintln!(
        "photon_mamba_bf16_forward_and_loss_match_fp32: logits max abs err = {logit_abs_err:.4e}, max rel err = {logit_rel_err:.4e}"
    );
    assert!(
        logit_abs_err < REL_TOL,
        "logits max abs err {logit_abs_err:.4e} exceeds {REL_TOL:e}"
    );
    assert!(
        logit_rel_err < REL_TOL,
        "logits max rel err {logit_rel_err:.4e} exceeds {REL_TOL:e}"
    );

    let loss_f32 = cross_entropy_loss(&bk, &out_f32.logits, &targets).unwrap();
    let loss_bf16 = cross_entropy_loss(&bk, &out_bf16.logits, &targets).unwrap();
    assert_eq!(loss_f32.dtype(), Dtype::F32, "loss must stay fp32");
    assert_eq!(
        loss_bf16.dtype(),
        Dtype::F32,
        "loss must stay fp32 even under bf16 compute"
    );

    let lv_f32 = bk.to_vec_f32(&loss_f32).unwrap()[0];
    let lv_bf16 = bk.to_vec_f32(&loss_bf16).unwrap()[0];
    assert!(lv_f32.is_finite() && lv_bf16.is_finite());
    let loss_err = (lv_f32 - lv_bf16).abs() / lv_f32.abs().max(1e-6);
    eprintln!("photon_mamba_bf16_forward_and_loss_match_fp32: loss fp32={lv_f32} bf16={lv_bf16} rel_err={loss_err:.4e}");
    assert!(
        loss_err < REL_TOL,
        "loss rel err {loss_err:.4e} exceeds {REL_TOL:e}"
    );
}

// ---------------------------------------------------------------------
// Test 4: grad flow (backward) — full PhotonMamba, fp32 vs bf16.
// ---------------------------------------------------------------------

#[test]
fn photon_mamba_bf16_backward_grad_finite_and_matches_fp32_presence() {
    let bk = cuda();
    let dims = ToyDims {
        vocab: 64,
        d_model: 32,
        chunk_size: 4,
        n_layers_per_level: 2,
    };
    let (b, t) = (2, 16);
    let (ids_data, targets_data) = make_ids_targets(b, t, dims.vocab);
    let ids = bk.from_slice_i64(&ids_data, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&targets_data, &[b, t]).unwrap();

    let run = |dtype: Dtype| -> Vec<Option<Vec<f32>>> {
        let model = build_toy_model(&bk, &dims).with_compute_dtype(dtype);
        let params = model.collect_params();
        seed_params(&bk, &params, 999);

        let out = model.forward(&bk, &ids).unwrap();
        let loss = cross_entropy_loss(&bk, &out.logits, &targets).unwrap();
        let grads = bk.backward(&loss).unwrap();

        params
            .iter()
            .map(|p| {
                bk.gradient(&grads, p)
                    .unwrap()
                    .map(|t| bk.to_vec_f32(&t).unwrap())
            })
            .collect()
    };

    let g_f32 = run(Dtype::F32);
    let g_bf16 = run(Dtype::BF16);
    assert_eq!(g_f32.len(), g_bf16.len());

    let mut n_with_grad = 0usize;
    for (i, (a, gb)) in g_f32.iter().zip(g_bf16.iter()).enumerate() {
        match (a, gb) {
            (Some(ga), Some(gb)) => {
                n_with_grad += 1;
                assert!(
                    ga.iter().all(|v| v.is_finite()),
                    "param[{i}]: fp32 grad has non-finite entries"
                );
                assert!(
                    gb.iter().all(|v| v.is_finite()),
                    "param[{i}]: bf16 grad has non-finite entries"
                );
            }
            (None, None) => {}
            _ => panic!(
                "param[{i}]: gradient presence differs between fp32 ({}) and bf16 ({}) — the \
                 bf16 fp32-island casts introduced a NEW autograd break",
                a.is_some(),
                gb.is_some()
            ),
        }
    }
    assert!(
        n_with_grad > 0,
        "no parameter received a gradient in either dtype — vacuous test"
    );
    eprintln!(
        "photon_mamba_bf16_backward_grad_finite_and_matches_fp32_presence: {n_with_grad}/{} params received a gradient",
        g_f32.len()
    );
}

// ---------------------------------------------------------------------
// Test 5: bf16 + activation checkpointing together (the mem-probe
// scenario) — exercises the `forward_checkpointed` code path, which
// duplicates the seed-cast logic from `hierarchical_decoder.rs`.
// ---------------------------------------------------------------------

#[test]
fn photon_mamba_bf16_checkpointed_forward_and_backward_smoke() {
    let bk = cuda();
    let dims = ToyDims {
        vocab: 64,
        d_model: 32,
        chunk_size: 4,
        n_layers_per_level: 2,
    };
    let model = build_toy_model(&bk, &dims).with_compute_dtype(Dtype::BF16);
    let params = model.collect_params();
    seed_params(&bk, &params, 555);

    let (b, t) = (2, 16);
    let (ids_data, targets_data) = make_ids_targets(b, t, dims.vocab);
    let ids = bk.from_slice_i64(&ids_data, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&targets_data, &[b, t]).unwrap();

    let (out, cp) = model.forward_checkpointed(&bk, &ids).unwrap();
    assert_eq!(out.logits.dtype(), Dtype::BF16);
    let logits_v = bk.to_vec_f32(&out.logits).unwrap();
    assert!(
        logits_v.iter().all(|v| v.is_finite()),
        "checkpointed bf16 forward non-finite"
    );

    let loss = cross_entropy_loss(&bk, &out.logits, &targets).unwrap();
    assert_eq!(loss.dtype(), Dtype::F32);
    let mut grads = bk.backward(&loss).unwrap();
    pm_core::checkpoint_backward(&bk, cp, &mut grads, |o, id, x| {
        model.recompute_block(o, id, x)
    })
    .unwrap();

    let mut n_with_grad = 0usize;
    for p in &params {
        if let Some(g) = bk.gradient(&grads, p).unwrap() {
            n_with_grad += 1;
            let g_host = bk.to_vec_f32(&g).unwrap();
            assert!(
                g_host.iter().all(|v| v.is_finite()),
                "non-finite grad under bf16+checkpointing"
            );
        }
    }
    assert!(
        n_with_grad > 0,
        "no parameter received a gradient — vacuous test"
    );
}
