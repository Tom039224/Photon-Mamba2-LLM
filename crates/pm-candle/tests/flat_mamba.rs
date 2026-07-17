//! Phase D.2a: FlatMamba baseline tests.
//!
//! Seven gates (per PLAN.md D.2, Phase D.2a spec):
//!
//! 1. `param_parity_tiny` — verifies the analytical param parity logic on a
//!    small model (fast, always runs).
//! 2. `param_parity_100m` — builds the real 102M PHOTON and the 32-layer flat
//!    model and verifies |flat-photon|/photon ≤ 0.02. (#[ignore] — heavy)
//! 3. `forward_shape` — tiny flat: ids(B,T) → hidden(B,T,D), logits(B,T,V).
//! 4. `all_params_receive_gradient` — fused CE backward reaches every param.
//! 5. `ckpt_parity` — no-ckpt vs ckpt grads agree to 1e-4.
//! 6. `save_load_roundtrip` — safetensors checkpoint round-trip is bit-exact.
//! 7. `alpha_positive_rejected_for_flat` — a flat + alpha>0 Config fails
//!    validation (tested separately in pm-cli/tests).
//!
//! Paper reference: Mamba2 §1-2 (standard LM construction: residual stack of
//! Mamba2 blocks, embedding lookup + tied lm_head). PLAN.md Phase D.2.

use pm_candle::CandleBackend;
use pm_core::flat::FlatMamba;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{ContextEncoder, TokenEmbedding};
use pm_core::{checkpoint_backward, Ops, Param, Parameterized, Tensor};
use pm_train::{fused_cross_entropy_injected, load_checkpoint, save_checkpoint};

// ─── helpers ────────────────────────────────────────────────────────────────

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

fn block_params(cfg: &Mamba2Config) -> usize {
    let d_inner = cfg.n_heads * cfg.d_head;
    let xbc_dim = d_inner + 2 * cfg.n_groups * cfg.d_state;
    let in_proj_dim = d_inner + xbc_dim + cfg.n_heads;
    // in_proj + conv1d_w + conv1d_b + a_log + d_skip + dt_bias + norm_w + out_proj
    cfg.d_model * in_proj_dim
        + xbc_dim * cfg.d_conv
        + xbc_dim
        + cfg.n_heads
        + cfg.n_heads
        + cfg.n_heads
        + d_inner
        + d_inner * cfg.d_model
}

fn build_tiny_flat(bk: &CandleBackend, n_layers: usize) -> FlatMamba<CandleBackend> {
    let (vocab, d_model) = (32usize, 16usize);
    let embed = TokenEmbedding::from_constants(bk, vocab, d_model, 0.05).unwrap();
    let layers: Vec<Mamba2Block<CandleBackend>> =
        (0..n_layers).map(|_| mk_block(bk, d_model)).collect();
    let trunk = ContextEncoder::from_layers(layers);
    FlatMamba::new(embed, trunk)
}

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

// ─── test 1a: param parity logic on tiny model ──────────────────────────────

/// Verify that flat N=3 and flat N=4 differ by exactly one block's param count.
/// This exercises the analytical formula without building 102M models.
///
/// Mamba2 §1 (standard LM, residual stack of Mamba2 blocks).
/// PLAN.md Phase D.2a, gate 1.
#[test]
fn param_parity_tiny() {
    let bk = CandleBackend::new_cpu();
    let (vocab, d_model) = (32usize, 16usize);

    let flat3 = build_tiny_flat(&bk, 3);
    let flat4 = build_tiny_flat(&bk, 4);

    let count3: usize = flat3
        .collect_params()
        .iter()
        .map(|p| p.as_tensor().numel())
        .sum();
    let count4: usize = flat4
        .collect_params()
        .iter()
        .map(|p| p.as_tensor().numel())
        .sum();

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
    let expected_block_params = block_params(&cfg);
    let embed_params = vocab * d_model;

    assert_eq!(
        count3,
        embed_params + 3 * expected_block_params,
        "flat N=3 param count mismatch"
    );
    assert_eq!(
        count4 - count3,
        expected_block_params,
        "flat N=4 - flat N=3 must equal exactly one block's params"
    );
    eprintln!(
        "param_parity_tiny: flat N=3 = {count3}, flat N=4 = {count4}, \
         block_params = {expected_block_params}, embed = {embed_params}"
    );
}

// ─── test 1b: 100M param parity (heavy — #[ignore] in CI) ──────────────────

/// Build the real 102M PHOTON and the 32-layer flat baseline; verify
/// |flat − photon| / photon ≤ 0.02 (PLAN.md Phase D.2a spec).
///
/// Expected (computed analytically from photon_mamba_100m.toml):
///   PHOTON ≈ 102_757_176 (38.60M embed + 59.44M blocks + 4.72M chunker/converter)
///   Flat N=32 ≈ 101_996_416 (38.60M embed + 63.40M blocks)
///   Deviation ≈ 0.74 %
///
/// Run with: cargo test -p pm-candle -- --ignored param_parity_100m
#[test]
#[ignore]
fn param_parity_100m() {
    use pm_core::mamba2::Mamba2Config;
    use pm_core::photon::{
        ChunkLocalDecoder, ContextChunker, ContextConverter, ContextEncoder, DecoderLevel,
        HierarchicalDecoder, HierarchicalEncoder, HierarchicalLevel, TokenEmbedding,
    };
    use pm_core::PhotonMamba;

    let bk = CandleBackend::new_cpu();
    let vocab = 50257usize;
    let d_model = 768usize;
    let d_state = 128usize;
    let d_head = 64usize;
    let n_heads = 12usize;
    let n_groups = 1usize;
    let d_conv = 4usize;
    let block_len = 64usize;
    let rmsnorm_eps = 1e-5f32;
    let n_layers_per_level = 10usize;
    let chunk_size = 4usize;
    let init_scale = 0.02f32;

    let m2_cfg = Mamba2Config {
        d_model,
        d_state,
        d_head,
        n_heads,
        n_groups,
        d_conv,
        block_len,
        rmsnorm_eps,
    };
    let mk_layers = |n: usize| -> Vec<Mamba2Block<CandleBackend>> {
        (0..n)
            .map(|_| Mamba2Block::from_constants(&bk, m2_cfg.clone(), init_scale).unwrap())
            .collect()
    };

    // PHOTON
    let embed_p = TokenEmbedding::from_constants(&bk, vocab, d_model, init_scale).unwrap();
    let lvl0 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(mk_layers(n_layers_per_level)),
        chunker: Some(
            ContextChunker::from_constants(&bk, d_model, d_model, chunk_size, init_scale).unwrap(),
        ),
    };
    let lvl1 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(mk_layers(n_layers_per_level)),
        chunker: None,
    };
    let encoder = HierarchicalEncoder::from_levels(vec![lvl0, lvl1]);
    let conv =
        ContextConverter::from_constants(&bk, d_model, d_model, chunk_size, init_scale).unwrap();
    let dec = ChunkLocalDecoder::from_layers(mk_layers(n_layers_per_level), chunk_size, chunk_size);
    let decoder = HierarchicalDecoder::from_levels(vec![DecoderLevel::new(conv, dec)]);
    let photon = PhotonMamba::new(embed_p, encoder, decoder);
    let photon_count: usize = photon
        .collect_params()
        .iter()
        .map(|p| p.as_tensor().numel())
        .sum();

    // Flat N=32
    let flat_n = 32usize;
    let embed_f = TokenEmbedding::from_constants(&bk, vocab, d_model, init_scale).unwrap();
    let trunk = ContextEncoder::from_layers(mk_layers(flat_n));
    let flat = FlatMamba::new(embed_f, trunk);
    let flat_count: usize = flat
        .collect_params()
        .iter()
        .map(|p| p.as_tensor().numel())
        .sum();

    let deviation = (flat_count as f64 - photon_count as f64).abs() / photon_count as f64;
    eprintln!(
        "param_parity_100m: photon = {photon_count}, flat N=32 = {flat_count}, \
         deviation = {:.4}%",
        deviation * 100.0
    );
    assert!(
        deviation <= 0.02,
        "flat N=32 param count deviates {:.4}% from PHOTON (budget 2%); \
         photon={photon_count}, flat={flat_count}",
        deviation * 100.0
    );
}

// ─── test 2: forward shape ───────────────────────────────────────────────────

/// Verify ids(B,T) → hidden(B,T,D) and logits(B,T,V).
///
/// Mamba2 standard LM forward. PLAN.md Phase D.2a, gate 2.
#[test]
fn forward_shape() {
    let bk = CandleBackend::new_cpu();
    let (vocab, d_model, b, t) = (32usize, 16usize, 2usize, 8usize);
    let model = build_tiny_flat(&bk, 2);

    let ids_data: Vec<i64> = (0..b * t).map(|i| (i as i64) % vocab as i64).collect();
    let ids = bk.from_slice_i64(&ids_data, &[b, t]).unwrap();

    // hidden shape
    let hidden = model.forward_hidden(&bk, &ids).unwrap();
    assert_eq!(
        hidden.shape(),
        &[b, t, d_model],
        "forward_hidden must return (B,T,D)"
    );
    let hv = bk.to_vec_f32(&hidden).unwrap();
    assert!(
        hv.iter().all(|x| x.is_finite()),
        "hidden contains non-finite values"
    );

    // logits shape
    let logits = model.forward(&bk, &ids).unwrap();
    assert_eq!(
        logits.shape(),
        &[b, t, vocab],
        "forward must return (B,T,V) logits"
    );
    let lv = bk.to_vec_f32(&logits).unwrap();
    assert!(
        lv.iter().all(|x| x.is_finite()),
        "logits contain non-finite values"
    );
    let max_abs = lv.iter().map(|x| x.abs()).fold(0f32, f32::max);
    assert!(max_abs > 0.0, "logits collapsed to zero");
}

// ─── test 3: all params receive gradients ───────────────────────────────────

/// After a fused CE backward, every param must have a Some, finite gradient.
///
/// This is the flat equivalent of CLAUDE.md's "forward だけでなく backward /
/// grad の検証もある" invariant. PLAN.md Phase D.2a, gate 3.
#[test]
fn all_params_receive_gradient() {
    let bk = CandleBackend::new_cpu();
    let (vocab, b, t) = (32usize, 2usize, 8usize);
    let model = build_tiny_flat(&bk, 2);
    let params = model.collect_params();
    seed_params(&bk, &params, 777);

    let (ids_v, tgt_v) = ids_targets(b, t, vocab);
    let ids = bk.from_slice_i64(&ids_v, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&tgt_v, &[b, t]).unwrap();

    let hidden = model.forward_hidden(&bk, &ids).unwrap();
    let (loss, grads) =
        fused_cross_entropy_injected(&bk, &hidden, &model.embed.weight, &targets, 4).unwrap();

    let loss_val = bk.to_vec_f32(&loss).unwrap()[0];
    assert!(loss_val.is_finite(), "loss must be finite, got {loss_val}");

    let mut missing = 0usize;
    let mut non_finite = 0usize;
    for (i, p) in params.iter().enumerate() {
        match bk.gradient(&grads, p).unwrap() {
            None => {
                eprintln!("param[{i}] shape={:?}: no gradient", p.as_tensor().shape());
                missing += 1;
            }
            Some(g) => {
                let gv = bk.to_vec_f32(&g).unwrap();
                if gv.iter().any(|x| !x.is_finite()) {
                    eprintln!(
                        "param[{i}] shape={:?}: non-finite gradient",
                        p.as_tensor().shape()
                    );
                    non_finite += 1;
                }
            }
        }
    }
    assert_eq!(
        missing, 0,
        "{missing} param(s) had no gradient after flat CE backward"
    );
    assert_eq!(
        non_finite, 0,
        "{non_finite} param(s) had non-finite gradient"
    );
    eprintln!(
        "all_params_receive_gradient: loss = {loss_val:.4}, {} params all have finite grads",
        params.len()
    );
}

// ─── test 4: checkpoint parity ───────────────────────────────────────────────

/// No-ckpt and ckpt gradients must agree to 1e-4 (fp32, CLAUDE.md invariant #3).
///
/// PLAN.md Phase D.2a, gate 4. Pattern mirrors
/// `pm-candle/tests/checkpoint_grad_parity.rs::checkpointed_grads_match_plain_grads`.
#[test]
fn ckpt_parity() {
    let bk = CandleBackend::new_cpu();
    let (vocab, b, t) = (32usize, 2usize, 8usize);
    let (ids_v, tgt_v) = ids_targets(b, t, vocab);
    let ids = bk.from_slice_i64(&ids_v, &[b, t]).unwrap();
    let targets = bk.from_slice_i64(&tgt_v, &[b, t]).unwrap();
    let n_layers = 3usize;

    let plain_grads = {
        let model = build_tiny_flat(&bk, n_layers);
        let params = model.collect_params();
        seed_params(&bk, &params, 4242);
        let hidden = model.forward_hidden(&bk, &ids).unwrap();
        let (_loss, grads) =
            fused_cross_entropy_injected(&bk, &hidden, &model.embed.weight, &targets, 4).unwrap();
        params
            .iter()
            .map(|p| {
                bk.gradient(&grads, p)
                    .unwrap()
                    .map(|g| bk.to_vec_f32(&g).unwrap())
            })
            .collect::<Vec<_>>()
    };

    let ckpt_grads = {
        let model = build_tiny_flat(&bk, n_layers);
        let params = model.collect_params();
        seed_params(&bk, &params, 4242);
        let (hidden, cp) = model.forward_checkpointed_hidden(&bk, &ids).unwrap();
        let (_loss, mut grads) =
            fused_cross_entropy_injected(&bk, &hidden, &model.embed.weight, &targets, 4).unwrap();
        checkpoint_backward(&bk, cp, &mut grads, |o, id, x| {
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

    assert_eq!(plain_grads.len(), ckpt_grads.len());
    let mut compared = 0usize;
    let mut worst = 0f32;
    for (i, (a, c)) in plain_grads.iter().zip(ckpt_grads.iter()).enumerate() {
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
                    max_abs < 1e-4,
                    "param[{i}]: ckpt grad differs from plain by {max_abs:.3e} (budget 1e-4); \
                     PHOTON §2.1, PLAN.md D.2a"
                );
            }
            (None, None) => {}
            _ => panic!("param[{i}]: grad presence differs between plain and ckpt"),
        }
    }
    eprintln!("ckpt_parity: compared {compared} params, worst |Δ| = {worst:.3e} (budget 1e-4)");
    assert!(
        compared >= n_layers * 8,
        "expected at least all block params compared"
    );
}

// ─── test 5: save / load roundtrip ──────────────────────────────────────────

/// Flat checkpoint written by `save_checkpoint` then read by `load_checkpoint`
/// must recover every parameter value to within 1e-6 (fp32 safetensors precision).
///
/// PLAN.md Phase D.2a, gate 5. Pattern mirrors
/// `pm-train/tests/checkpoint_roundtrip.rs::save_then_load_recovers_param_values`.
#[test]
fn save_load_roundtrip() -> anyhow::Result<()> {
    let bk = CandleBackend::new_cpu();

    let src = build_tiny_flat(&bk, 2);
    let src_params = src.collect_params();
    seed_params(&bk, &src_params, 999);

    let mut path = std::env::temp_dir();
    path.push(format!(
        "pm_flat_roundtrip_{}.safetensors",
        std::process::id()
    ));
    save_checkpoint(&bk, &src_params, &path)?;

    let dst = build_tiny_flat(&bk, 2);
    let dst_params = dst.collect_params();
    assert_eq!(
        src_params.len(),
        dst_params.len(),
        "src/dst must have same param count"
    );
    load_checkpoint(&bk, &dst_params, &path)?;

    for (i, (s, d)) in src_params.iter().zip(dst_params.iter()).enumerate() {
        let sv = bk.to_vec_f32(s.as_tensor())?;
        let dv = bk.to_vec_f32(d.as_tensor())?;
        assert_eq!(sv.len(), dv.len(), "param[{i}] shape mismatch after load");
        for (j, (sx, dx)) in sv.iter().zip(dv.iter()).enumerate() {
            assert!(
                (sx - dx).abs() < 1e-6,
                "param[{i}][{j}]: load mismatch src={sx}, dst={dx}"
            );
        }
    }

    std::fs::remove_file(&path).ok();
    Ok(())
}
