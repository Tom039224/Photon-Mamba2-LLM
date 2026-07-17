//! `pm train --config <path>` subcommand.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use pm_backend::Backend;
use pm_candle::CandleBackend;
use pm_core::{FlatMamba, Ops, Parameterized, PhotonMamba};
use pm_train::{
    fused_cross_entropy_injected, fused_photon_loss_injected, save_checkpoint, AdamW, AdamWConfig,
    LossComponents, PhotonLossReport, Trainer,
};

use crate::config::Config;
use crate::model_build::{build_flat_mamba, build_photon_mamba, validate_flat_config_alpha};
use crate::BackendKind;

/// Row-tile width for `Ops::fused_cross_entropy` (memory-efficiency
/// plan: fused/tiled cross-entropy, `docs/perf-log.md` 2026-07-03).
/// Bounds the transient `(tile_rows, vocab_size)` logits/softmax buffers
/// to ~52 MB at `vocab_size=50257` (f32) regardless of `seq_len` — the
/// lever that gets `seq_len=2048` training under the 12 GB budget.
const CE_TILE_ROWS: usize = 256;

#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Path to a TOML config file.
    #[arg(long)]
    pub config: std::path::PathBuf,
    /// Which compute backend to use.
    #[arg(long, value_enum, default_value_t = BackendKind::Candle)]
    pub backend: BackendKind,
}

pub fn run(args: TrainArgs) -> Result<()> {
    match args.backend {
        BackendKind::Candle => {
            let cfg = Config::load(&args.config)
                .with_context(|| format!("loading {}", args.config.display()))?;
            let bk = build_candle_backend(&cfg.runtime.device)?;
            run_inner(args, bk, cfg)
        }
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => {
            let cfg = Config::load(&args.config)
                .with_context(|| format!("loading {}", args.config.display()))?;
            let bk = pm_cuda::CudaBackend::new(0)
                .map_err(|e| anyhow::anyhow!("CudaBackend::new(0): {e}"))?;
            let result = run_inner(args, bk, cfg);
            // Phase B'.1b: env-gated per-op wall-time profiler
            // (`PM_CUDA_PROFILE=1`); `report()` is `None` unless enabled,
            // so this is a no-op on every normal (non-profiled) run.
            // Printed regardless of `result` so a run that errors out
            // partway through (e.g. OOM at some step) still surfaces
            // whatever was recorded up to that point.
            if let Some(report) = pm_cuda::profiler::report() {
                eprintln!("{report}");
            }
            result
        }
    }
}

// ─── model dispatch ──────────────────────────────────────────────────────────

/// Unified wrapper over the two supported model architectures.
///
/// `AnyModel` is pm-cli local — it exists only to unify the training loop
/// dispatch without duplicating the step logic or touching pm-train/pm-infer.
/// Flat and PHOTON have different forward signatures, so the two loss
/// paths (ckpt / no-ckpt) are encapsulated as methods here.
///
/// PLAN.md Phase D.2a.
enum AnyModel<O: Ops>
where
    O::Tensor: Clone,
    O::Param: Clone,
{
    Photon(PhotonMamba<O>),
    Flat(FlatMamba<O>),
}

impl<O: Ops> AnyModel<O>
where
    O::Tensor: Clone,
    O::Param: Clone,
{
    fn collect_params(&self) -> Vec<&O::Param> {
        match self {
            AnyModel::Photon(m) => m.collect_params(),
            AnyModel::Flat(m) => m.collect_params(),
        }
    }

    /// No-checkpointing forward+loss step.
    ///
    /// - PHOTON: uses `fused_photon_loss_injected` (supports α>0 consistency;
    ///   returns `LossComponents` — Phase B'.3 — whenever alpha>0 actually
    ///   computed `L_rec`).
    /// - Flat: calls `fused_cross_entropy_injected` directly (no hierarchy,
    ///   α must be 0 — validated at config-load time via
    ///   `validate_flat_config_alpha`) and wraps the result with
    ///   `components: None` so both arms share one return type.
    fn forward_loss_no_ckpt(
        &self,
        ops: &O,
        ids: &O::Tensor,
        targets: &O::Tensor,
        tile_rows: usize,
        alpha: f32,
    ) -> Result<(PhotonLossReport<O>, O::GradStore), O::Error> {
        match self {
            AnyModel::Photon(model) => {
                let hidden = model.forward_no_lm_head(ops, ids)?;
                let n_dec = hidden.predicted.len();
                fused_photon_loss_injected(
                    ops,
                    &hidden.predicted[0],
                    &model.embed.weight,
                    targets,
                    tile_rows,
                    &hidden.predicted,
                    &hidden.encoded.encoded[..n_dec],
                    alpha,
                )
            }
            AnyModel::Flat(model) => {
                // α=0 is enforced at config validation (`validate_flat_
                // config_alpha`); flat has no hierarchy to penalise.
                debug_assert!(
                    alpha == 0.0,
                    "flat model reached the loss path with α={alpha}; \
                     validate_flat_config_alpha must run first"
                );
                let hidden = model.forward_hidden(ops, ids)?;
                let (loss, grads) = fused_cross_entropy_injected(
                    ops,
                    &hidden,
                    &model.embed.weight,
                    targets,
                    tile_rows,
                )?;
                Ok((
                    PhotonLossReport {
                        total: loss,
                        components: None,
                    },
                    grads,
                ))
            }
        }
    }

    /// Activation-checkpointing forward+loss step.
    ///
    /// Same split as `forward_loss_no_ckpt` but each Mamba2Block's forward
    /// is wrapped in a checkpoint segment, so `checkpoint_backward` handles
    /// the recompute pass after the main backward.
    fn forward_loss_ckpt(
        &self,
        ops: &O,
        ids: &O::Tensor,
        targets: &O::Tensor,
        tile_rows: usize,
        alpha: f32,
    ) -> Result<(PhotonLossReport<O>, O::GradStore), O::Error> {
        match self {
            AnyModel::Photon(model) => {
                let (hidden, cp) = model.forward_checkpointed_no_lm_head(ops, ids)?;
                let n_dec = hidden.predicted.len();
                let (report, mut grads) = fused_photon_loss_injected(
                    ops,
                    &hidden.predicted[0],
                    &model.embed.weight,
                    targets,
                    tile_rows,
                    &hidden.predicted,
                    &hidden.encoded.encoded[..n_dec],
                    alpha,
                )?;
                pm_core::checkpoint_backward(ops, cp, &mut grads, |o, id, x| {
                    model.recompute_block(o, id, x)
                })?;
                Ok((report, grads))
            }
            AnyModel::Flat(model) => {
                debug_assert!(
                    alpha == 0.0,
                    "flat model reached the loss path with α={alpha}; \
                     validate_flat_config_alpha must run first"
                );
                let (hidden, cp) = model.forward_checkpointed_hidden(ops, ids)?;
                let (loss, mut grads) = fused_cross_entropy_injected(
                    ops,
                    &hidden,
                    &model.embed.weight,
                    targets,
                    tile_rows,
                )?;
                pm_core::checkpoint_backward(ops, cp, &mut grads, |o, id, x| {
                    model.recompute_block(o, id, x)
                })?;
                Ok((
                    PhotonLossReport {
                        total: loss,
                        components: None,
                    },
                    grads,
                ))
            }
        }
    }
}

// ─── inner training loop ─────────────────────────────────────────────────────

fn run_inner<O>(args: TrainArgs, bk: O, cfg: Config) -> Result<()>
where
    O: Ops + Backend,
    O::Tensor: Clone,
    O::Param: Clone,
{
    let log_every = cfg.train.log_every.max(1);
    eprintln!(
        "pm train: backend = {:?}, config = {}, model_type = {}, compute_dtype = {}",
        bk.device_kind(),
        args.config.display(),
        cfg.model.model_type,
        cfg.model.compute_dtype
    );

    // Validate flat + alpha combination before building anything.
    validate_flat_config_alpha(&cfg.model.model_type, cfg.train.consistency_alpha)
        .with_context(|| format!("config {}", args.config.display()))?;

    eprintln!("pm train: building model");
    let model: AnyModel<O> = match cfg.model.model_type.as_str() {
        "photon" => AnyModel::Photon(build_photon_mamba(&bk, &cfg.model)?),
        "flat" => AnyModel::Flat(build_flat_mamba(&bk, &cfg.model)?),
        other => anyhow::bail!("unknown model_type {other:?}; expected \"photon\" or \"flat\""),
    };

    let params = model.collect_params();
    eprintln!(
        "pm train: model has {} trainable params, total elements = {}",
        params.len(),
        total_param_count::<O>(&params)
    );

    if cfg.train.init_perturb_seed != 0 {
        perturb_params(&bk, &params, cfg.train.init_perturb_seed)?;
    }

    let optim = AdamW::new(
        &bk,
        &params,
        AdamWConfig {
            lr: cfg.train.lr,
            beta1: cfg.train.beta1,
            beta2: cfg.train.beta2,
            eps: cfg.train.eps,
            weight_decay: cfg.train.weight_decay,
        },
    )?;
    let mut trainer = Trainer::new(optim);
    if let Some(max_norm) = cfg.train.max_grad_norm {
        if max_norm > 0.0 {
            trainer = trainer.with_clip(max_norm);
        }
    }

    // chunk_product for PHOTON (controls seq_len alignment + batcher stride).
    // For flat, chunk_product = 1 (no alignment needed); the batcher just
    // gets stride=1, which is a valid no-op.
    let chunk_product = if cfg.model.model_type == "flat" {
        1usize
    } else {
        cfg.model.chunk_size.pow(cfg.model.n_levels as u32 - 1)
    };

    if cfg.model.model_type == "photon" {
        anyhow::ensure!(
            cfg.train.seq_len.is_multiple_of(chunk_product),
            "seq_len ({}) must be a multiple of chunk_size^{} = {}",
            cfg.train.seq_len,
            cfg.model.n_levels - 1,
            chunk_product
        );
    }

    // Two data paths: real tokenised text when `cfg.train.text_data` +
    // `cfg.train.tokenizer` are set, else random batches.
    let mut real_data = if !cfg.train.text_data.is_empty() {
        if cfg.train.tokenizer.is_empty() {
            anyhow::bail!("text_data set but tokenizer is not");
        }
        let tk = pm_tokenizer::BpeTokenizer::from_file(&cfg.train.tokenizer)
            .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
        let source = pm_data::TextFileSource::open(&cfg.train.text_data, cfg.train.doc_sep_id)
            .map_err(|e| anyhow::anyhow!("text source: {e}"))?;
        let batcher =
            pm_data::PackedBatcher::new(cfg.train.batch_size, cfg.train.seq_len, chunk_product, 0)
                .map_err(|e| anyhow::anyhow!("packed batcher: {e}"))?;
        Some((tk, source, batcher))
    } else {
        None
    };

    let mut rng = cfg.train.seed;
    if rng == 0 {
        rng = 0xCAFE_BABE_1234_5678;
    }

    // Fixed overfit batch — only used when real_data is None AND
    // fresh_batch_per_step is false.
    let toy_fixed_batch = if real_data.is_none() && !cfg.train.fresh_batch_per_step {
        let (ids_data, targets_data) = next_random_batch(
            &mut rng,
            cfg.train.batch_size,
            cfg.train.seq_len,
            cfg.model.vocab_size,
        );
        Some((
            bk.from_slice_i64(&ids_data, &[cfg.train.batch_size, cfg.train.seq_len])?,
            bk.from_slice_i64(&targets_data, &[cfg.train.batch_size, cfg.train.seq_len])?,
        ))
    } else {
        None
    };

    // Ctrl+C cleanup: finish the current step, save, then exit.
    let should_stop = Arc::new(AtomicBool::new(false));
    {
        let flag = should_stop.clone();
        let _ = ctrlc::set_handler(move || {
            eprintln!("\npm train: Ctrl+C received — finishing current step then saving + exiting");
            flag.store(true, Ordering::SeqCst);
        });
    }

    let save_path = cfg.train.save_path.trim().to_owned();
    let save_every = cfg.train.save_every;
    let max_wall = cfg.train.max_wall_time_seconds.filter(|t| *t > 0.0);

    let save_if_path = |params: &[&O::Param], reason: &str| -> Result<()> {
        if save_path.is_empty() {
            return Ok(());
        }
        let tmp = format!("{save_path}.tmp");
        save_checkpoint(&bk, params, &tmp).map_err(|e| anyhow::anyhow!("save_checkpoint: {e}"))?;
        std::fs::rename(&tmp, &save_path)
            .with_context(|| format!("rename {tmp} -> {save_path}"))?;
        eprintln!("pm train: {reason} checkpoint written to {save_path}");
        Ok(())
    };

    let alpha = cfg.train.consistency_alpha;
    let use_ckpt = cfg.train.activation_checkpointing;

    let start = Instant::now();
    let mut step = 0usize;
    let mut stopped_reason = "completed";
    while step < cfg.train.n_steps {
        if should_stop.load(Ordering::SeqCst) {
            stopped_reason = "Ctrl+C";
            break;
        }
        if let Some(max) = max_wall {
            if start.elapsed().as_secs_f32() > max {
                stopped_reason = "wall-time cap";
                break;
            }
        }

        let (ids, targets) = if let Some((tk, source, batcher)) = real_data.as_mut() {
            let Some((ids_v, tgt_v)) = batcher
                .next_batch(source, tk)
                .map_err(|e| anyhow::anyhow!("next_batch: {e}"))?
            else {
                stopped_reason = "data exhausted";
                break;
            };
            (
                bk.from_slice_i64(&ids_v, &[cfg.train.batch_size, cfg.train.seq_len])?,
                bk.from_slice_i64(&tgt_v, &[cfg.train.batch_size, cfg.train.seq_len])?,
            )
        } else if let Some((ids, tgt)) = &toy_fixed_batch {
            (ids.clone(), tgt.clone())
        } else {
            // fresh_batch_per_step path
            let (ids_data, targets_data) = next_random_batch(
                &mut rng,
                cfg.train.batch_size,
                cfg.train.seq_len,
                cfg.model.vocab_size,
            );
            (
                bk.from_slice_i64(&ids_data, &[cfg.train.batch_size, cfg.train.seq_len])?,
                bk.from_slice_i64(&targets_data, &[cfg.train.batch_size, cfg.train.seq_len])?,
            )
        };

        // The training loop dispatch is unified: `AnyModel::forward_loss_{no_}ckpt`
        // handles both PHOTON and flat models. PHOTON uses
        // `fused_photon_loss_injected` (supports α>0 consistency); flat uses
        // `fused_cross_entropy_injected` (α=0 enforced at config validation).
        //
        // `Trainer::step_with_grads` is fixed to `(O::Tensor, O::GradStore)`
        // closures — it reads `O::Tensor` straight into `StepReport::loss`
        // via `Ops::to_vec_f32` — so `PhotonLossReport::components` (Phase
        // B'.3's CE/L_rec split) can't ride through its return value.
        // Stash it from inside the closure instead; `step_with_grads` calls
        // its closure exactly once, so the mutable capture is released by
        // the time `loss_components` is read below.
        let mut loss_components: Option<LossComponents> = None;
        let report = if use_ckpt {
            trainer.step_with_grads(&bk, &params, |ops| {
                let (loss_report, grads) =
                    model.forward_loss_ckpt(ops, &ids, &targets, CE_TILE_ROWS, alpha)?;
                loss_components = loss_report.components;
                Ok((loss_report.total, grads))
            })?
        } else {
            trainer.step_with_grads(&bk, &params, |ops| {
                let (loss_report, grads) =
                    model.forward_loss_no_ckpt(ops, &ids, &targets, CE_TILE_ROWS, alpha)?;
                loss_components = loss_report.components;
                Ok((loss_report.total, grads))
            })?
        };

        if step.is_multiple_of(log_every) || step + 1 == cfg.train.n_steps {
            let elapsed = start.elapsed().as_secs_f32();
            let gnorm = report
                .grad_norm
                .map(|g| format!(", grad_norm = {g:.3}"))
                .unwrap_or_default();
            // `loss_components` is `None` whenever `alpha == 0.0` (the
            // default, and the paper's own main-result setting) — this
            // branch must reproduce the pre-Phase-B'.3 line byte-for-byte,
            // since `docs/perf-log.md`'s grad_norm fingerprints grep it.
            match loss_components {
                Some(c) => eprintln!(
                    "step {step:>6}: loss = {:.4}, ce = {:.4}, lrec = {:.4}{gnorm}, elapsed = {:.1}s",
                    report.loss, c.ce, c.lrec, elapsed
                ),
                None => eprintln!(
                    "step {step:>6}: loss = {:.4}{gnorm}, elapsed = {:.1}s",
                    report.loss, elapsed
                ),
            }
        }

        if save_every > 0 && step > 0 && step.is_multiple_of(save_every) {
            save_if_path(&params, &format!("step {step}"))?;
        }

        step += 1;
    }

    save_if_path(&params, &format!("final ({stopped_reason})"))?;
    eprintln!(
        "pm train: done — {} step(s) completed in {:.1}s ({stopped_reason})",
        step,
        start.elapsed().as_secs_f32()
    );
    Ok(())
}

fn total_param_count<O: Ops>(params: &[&O::Param]) -> usize {
    use pm_core::{Param, Tensor};
    params.iter().map(|p| p.as_tensor().numel()).sum()
}

fn build_candle_backend(device: &str) -> Result<CandleBackend> {
    match device {
        "cpu" => Ok(CandleBackend::new_cpu()),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Ok(CandleBackend::new_cuda(0)?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!(
                    "pm train: `cuda` requested but binary was built without --features cuda; \
                     falling back to CPU"
                );
                Ok(CandleBackend::new_cpu())
            }
        }
        other => anyhow::bail!("unknown device {other:?}; expected \"cpu\" or \"cuda\""),
    }
}

fn perturb_params<O: Ops>(bk: &O, params: &[&O::Param], seed: u64) -> Result<()> {
    use pm_core::{Param, Tensor};
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
        let t = bk.from_slice_f32(&data, &shape)?;
        bk.assign(p, &t)?;
    }
    Ok(())
}

/// Generate a `(batch_size, seq_len)` block of random token ids plus a
/// shifted-by-one target tensor (next-token prediction). All values
/// in `[0, vocab)`.
fn next_random_batch(rng: &mut u64, b: usize, t: usize, vocab: usize) -> (Vec<i64>, Vec<i64>) {
    let n = b * t;
    let mut ids = vec![0i64; n];
    for slot in &mut ids {
        *rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *slot = (*rng >> 33) as i64 % vocab as i64;
    }
    let mut targets = vec![0i64; n];
    for bi in 0..b {
        for ti in 0..t {
            targets[bi * t + ti] = ids[bi * t + (ti + 1) % t];
        }
    }
    (ids, targets)
}
