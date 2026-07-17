//! `pm eval hellaswag` subcommand (H.3).
//!
//! Reads a JSONL file where each line is
//! `{ "context": [int], "endings": [[int]; 4], "label": int }`
//! — already tokenised. Real tokeniser plumbing (Group E) plus the
//! HellaSwag download / tokenisation script (H.1) feed into this.
//!
//! Output: per-item prediction (optional, behind `--verbose`) plus the
//! final accuracy and a summary line that the `phase1-report.md` text
//! can paste verbatim.

use std::io::BufRead;

use anyhow::{Context, Result};
use clap::Args;
use pm_backend::Backend;
use pm_candle::CandleBackend;
use pm_core::{FlatMamba, Ops, Parameterized, Tensor};
use pm_infer::{eval::score_hellaswag, HellaSwagItem};
use pm_train::load_checkpoint;

use crate::config::Config;
use crate::model_build::{build_flat_mamba, build_photon_mamba};
use crate::BackendKind;

#[derive(Args, Debug)]
pub struct EvalArgs {
    #[arg(long)]
    pub config: std::path::PathBuf,
    #[arg(long)]
    pub model: Option<std::path::PathBuf>,
    /// JSONL file: `{context: [int], endings: [[int]; 4], label: int}` per line.
    #[arg(long)]
    pub data: std::path::PathBuf,
    /// Cap on items read (0 = all).
    #[arg(long, default_value_t = 0)]
    pub limit: usize,
    #[arg(long, default_value_t = false)]
    pub verbose: bool,
    /// Which compute backend to use.
    #[arg(long, value_enum, default_value_t = BackendKind::Candle)]
    pub backend: BackendKind,
}

#[derive(serde::Deserialize)]
struct JsonItem {
    context: Vec<i64>,
    endings: [Vec<i64>; 4],
    label: usize,
}

pub fn run(args: EvalArgs) -> Result<()> {
    match args.backend {
        BackendKind::Candle => {
            let cfg = Config::load(&args.config)
                .with_context(|| format!("loading {}", args.config.display()))?;
            let bk = build_candle_backend(&cfg.runtime.device)?;
            run_inner(args, bk)
        }
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => {
            let bk = pm_cuda::CudaBackend::new(0)
                .map_err(|e| anyhow::anyhow!("CudaBackend::new(0): {e}"))?;
            run_inner(args, bk)
        }
    }
}

fn run_inner<O: Ops + Backend>(args: EvalArgs, bk: O) -> Result<()> {
    let cfg =
        Config::load(&args.config).with_context(|| format!("loading {}", args.config.display()))?;

    // Dispatch to the flat-model scoring path when model_type = "flat".
    // `score_hellaswag` (from pm-infer) is not generic — it takes
    // `&PhotonMamba<O>` — so we use a local helper for FlatMamba. Neither
    // `pm-train` nor `pm-infer` are touched. PLAN.md Phase D.2a.
    match cfg.model.model_type.as_str() {
        "flat" => {
            let model = build_flat_mamba(&bk, &cfg.model)?;
            let params = model.collect_params();
            if let Some(path) = &args.model {
                load_checkpoint(&bk, &params, path)
                    .map_err(|e| anyhow::anyhow!("load_checkpoint: {e}"))?;
            } else {
                eprintln!("pm eval: WARNING no --model; using init weights");
            }
            run_hellaswag_flat(&bk, &model, &args)
        }
        _ => {
            let model = build_photon_mamba(&bk, &cfg.model)?;
            let params = model.collect_params();
            if let Some(path) = &args.model {
                load_checkpoint(&bk, &params, path)
                    .map_err(|e| anyhow::anyhow!("load_checkpoint: {e}"))?;
            } else {
                eprintln!("pm eval: WARNING no --model; using init weights");
            }
            let chunk_product = cfg.model.chunk_size.pow(cfg.model.n_levels as u32 - 1);
            run_hellaswag_photon(&bk, &model, chunk_product, &args)
        }
    }
}

/// HellaSwag scoring loop for [`PhotonMamba`] — delegates to the
/// pm-infer `score_hellaswag` function (takes `&PhotonMamba<O>`).
fn run_hellaswag_photon<O: Ops + Backend>(
    bk: &O,
    model: &pm_core::PhotonMamba<O>,
    chunk_product: usize,
    args: &EvalArgs,
) -> Result<()> {
    run_hellaswag_loop(args, |item| {
        // no_grad_scope: eval never calls backward, so without this the
        // tape-based backends would retain every item's forward
        // activations until OOM (D.2b post-mortem, 2026-07-05).
        let result = bk
            .no_grad_scope(|| score_hellaswag(bk, model, chunk_product, 0, item))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok((result.predicted, result.scores))
    })
}

/// HellaSwag scoring loop for [`FlatMamba`].
///
/// `score_hellaswag` (pm-infer) takes `&PhotonMamba<O>` and cannot be
/// reused here without modifying pm-infer (CLAUDE.md invariant: pm-train /
/// pm-infer must not be modified to add a new model type). This local
/// helper reimplements length-normalised continuation scoring using
/// `FlatMamba::forward` — no chunk-product alignment needed (flat has no
/// hierarchy). Mamba2 §1 (standard LM stack, scored by next-token log-prob).
fn run_hellaswag_flat<O: Ops + Backend>(
    bk: &O,
    model: &FlatMamba<O>,
    args: &EvalArgs,
) -> Result<()> {
    run_hellaswag_loop(args, |item| {
        // Same no_grad_scope rationale as run_hellaswag_photon.
        bk.no_grad_scope(|| score_flat_item(bk, model, item))
            .map_err(|e| anyhow::anyhow!("{e}"))
    })
}

/// Score all 4 endings of one HellaSwag item with `FlatMamba` and return
/// `(best index, per-ending mean log-probs)` — the same contract as
/// `pm_infer::eval::HellaSwagResult` so `--verbose` output is uniform
/// across model types.
fn score_flat_item<O: Ops>(
    ops: &O,
    model: &FlatMamba<O>,
    item: &HellaSwagItem,
) -> Result<(usize, [f32; 4]), O::Error> {
    let mut scores = [f32::NEG_INFINITY; 4];
    let mut best_i = 0;
    let mut best_score = f32::NEG_INFINITY;
    for (i, ending) in item.endings.iter().enumerate() {
        let mean = score_flat_continuation(ops, model, &item.context, ending)?;
        if let Some(slot) = scores.get_mut(i) {
            *slot = mean;
        }
        if mean > best_score {
            best_score = mean;
            best_i = i;
        }
    }
    Ok((best_i, scores))
}

/// Length-normalised log-probability of `continuation` given `context`
/// using the flat model's full `(B,T,V)` logit output.
///
/// No chunk-product padding is needed — the flat model has no alignment
/// constraint beyond the block_len used by the SSD scan, which operates
/// on a sequence of any positive length. Mamba2 §1.
///
/// DRIFT TWIN: this is a near-verbatim port of
/// `pm_infer::eval::score_continuation` (off-by-one alignment, mean
/// length-normalisation, per-token scoring loop). It is duplicated
/// rather than shared because pm-infer must not be modified to
/// accommodate a new model type (CLAUDE.md invariant #2). If you fix a
/// scoring bug in either copy, apply it to the other.
fn score_flat_continuation<O: Ops>(
    ops: &O,
    model: &FlatMamba<O>,
    context: &[i64],
    continuation: &[i64],
) -> Result<f32, O::Error> {
    assert!(!continuation.is_empty(), "continuation must not be empty");
    assert!(
        !context.is_empty(),
        "context must not be empty (start = context.len() - 1 would underflow)"
    );
    let total_len = context.len() + continuation.len();
    let mut buf = vec![0i64; total_len];
    buf[..context.len()].copy_from_slice(context);
    buf[context.len()..].copy_from_slice(continuation);

    let ids = ops.from_slice_i64(&buf, &[1, total_len])?;
    // FlatMamba::forward returns (1, T, V) logits directly.
    let logits = model.forward(ops, &ids)?;
    let v = logits.shape()[2];

    // logits[t] predicts ids[t+1]. First continuation token is at
    // position context.len() in buf, predicted by logits[context.len()-1].
    let start = context.len() - 1;
    let mut sum = 0.0f32;
    for k in 0..continuation.len() {
        let pos = start + k;
        let row = ops.narrow(&logits, 1, pos, 1)?; // (1, 1, V)
        let logp = ops.log_softmax(&row, 2)?;
        let host: Vec<f32> = ops.to_vec_f32(&logp)?;
        assert_eq!(host.len(), v);
        sum += host[continuation[k] as usize];
    }
    Ok(sum / continuation.len() as f32)
}

/// Shared loop that reads JSONL items, calls `score_fn`, tallies accuracy.
/// `score_fn` returns `(predicted index, per-ending scores)`.
fn run_hellaswag_loop<F>(args: &EvalArgs, mut score_fn: F) -> Result<()>
where
    F: FnMut(&HellaSwagItem) -> Result<(usize, [f32; 4])>,
{
    let file = std::fs::File::open(&args.data)
        .with_context(|| format!("opening {}", args.data.display()))?;
    let reader = std::io::BufReader::new(file);

    let mut total = 0usize;
    let mut correct = 0usize;
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let raw: JsonItem =
            serde_json::from_str(&line).with_context(|| format!("parsing line {i}"))?;
        let item = HellaSwagItem {
            context: raw.context,
            endings: raw.endings,
            label: raw.label,
        };
        let (predicted, scores) = score_fn(&item)?;
        total += 1;
        if predicted == item.label {
            correct += 1;
        }
        if args.verbose {
            eprintln!(
                "[{i:>5}] pred={predicted} label={} scores={scores:?}",
                item.label
            );
        }
        if args.limit > 0 && total >= args.limit {
            break;
        }
    }

    let accuracy = if total == 0 {
        0.0
    } else {
        correct as f32 / total as f32
    };
    println!(
        "hellaswag: correct = {correct} / {total}, accuracy = {:.4}",
        accuracy
    );
    Ok(())
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
                eprintln!("pm eval: cuda requested but binary built without --features cuda; CPU fallback");
                Ok(CandleBackend::new_cpu())
            }
        }
        other => anyhow::bail!("unknown device {other:?}"),
    }
}
