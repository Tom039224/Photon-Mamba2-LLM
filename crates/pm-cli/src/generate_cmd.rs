//! `pm generate` subcommand (G.5).
//!
//! Tokeniser integration lands with Group E. Until then, `--prompt-ids`
//! accepts a comma-separated list of token IDs and the output is also
//! printed as IDs. Once we have GPT-2 BPE wired up, `--prompt` becomes
//! a string flag and decoded text is printed alongside.

use anyhow::{Context, Result};
use clap::Args;
use pm_backend::Backend;
use pm_candle::CandleBackend;
use pm_core::{Ops, Parameterized};
use pm_infer::{GenerateConfig, Generator, Sampler, StatefulGenerator};
use pm_train::load_checkpoint;

use crate::config::Config;
use crate::model_build::build_photon_mamba;
use crate::BackendKind;

/// Decoding strategy (memory-efficiency plan, Phase C).
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecodeMode {
    /// O(1)-memory recurrent decode (`StatefulGenerator`). Default.
    #[default]
    Stateful,
    /// Padded re-forward baseline (`Generator`) — O(T²)-ish per step;
    /// kept for comparison / fallback.
    Reforward,
}

#[derive(Args, Debug)]
pub struct GenerateArgs {
    /// Path to the model config used to build the architecture
    /// skeleton (same TOML that `pm train` consumed).
    #[arg(long)]
    pub config: std::path::PathBuf,
    /// Path to a `.safetensors` checkpoint. Optional — without it the
    /// model uses the (untrained) init weights from the config, which
    /// is only useful for plumbing smokes.
    #[arg(long)]
    pub model: Option<std::path::PathBuf>,
    /// Comma-separated list of integer token IDs. Mutually exclusive
    /// with `--prompt`.
    #[arg(long, conflicts_with = "prompt")]
    pub prompt_ids: Option<String>,
    /// Free-text prompt. Requires `--tokenizer`.
    #[arg(long, requires = "tokenizer")]
    pub prompt: Option<String>,
    /// Path to a HuggingFace `tokenizer.json`. Required when `--prompt`
    /// (text) is used, optional otherwise (lets us decode the generated
    /// ids back to text).
    #[arg(long)]
    pub tokenizer: Option<std::path::PathBuf>,
    /// How many tokens to generate.
    #[arg(long, default_value_t = 32)]
    pub max_new_tokens: usize,
    /// Sampling temperature. 0.0 ⇒ greedy.
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,
    /// Top-k filtering.
    #[arg(long)]
    pub top_k: Option<usize>,
    /// Top-p (nucleus) filtering.
    #[arg(long)]
    pub top_p: Option<f32>,
    /// Sampler RNG seed.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Which compute backend to use.
    #[arg(long, value_enum, default_value_t = BackendKind::Candle)]
    pub backend: BackendKind,
    /// Decoding strategy: `stateful` (O(1) memory, default) or
    /// `reforward` (padded re-forward baseline).
    #[arg(long, value_enum, default_value_t = DecodeMode::Stateful)]
    pub decode: DecodeMode,
}

pub fn run(args: GenerateArgs) -> Result<()> {
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

fn run_inner<O: Ops + Backend + 'static>(args: GenerateArgs, bk: O) -> Result<()>
where
    O::Tensor: Clone,
{
    let cfg =
        Config::load(&args.config).with_context(|| format!("loading {}", args.config.display()))?;

    // The stateful / reforward generators in pm-infer are built around
    // `PhotonMamba`'s O(1) decode-state infrastructure. A flat model
    // evaluation (D.2 comparison) scores perplexity / HellaSwag accuracy
    // via `pm eval hellaswag`, not token-by-token generation. PLAN.md D.2a.
    if cfg.model.model_type == "flat" {
        anyhow::bail!(
            "pm generate does not support model_type = \"flat\": the flat baseline \
             has no O(1) recurrent decode state (that infrastructure is PHOTON-specific). \
             Use `pm eval hellaswag` to score a flat model checkpoint."
        );
    }

    let model = build_photon_mamba(&bk, &cfg.model)?;
    let params = model.collect_params();

    if let Some(path) = &args.model {
        eprintln!("pm generate: loading checkpoint {}", path.display());
        load_checkpoint(&bk, &params, path).map_err(|e| anyhow::anyhow!("load_checkpoint: {e}"))?;
    } else {
        eprintln!("pm generate: WARNING no --model checkpoint, using init weights");
    }

    // Load tokenizer if provided (required for --prompt text).
    let tokenizer = match &args.tokenizer {
        Some(p) => Some(
            pm_tokenizer::BpeTokenizer::from_file(p)
                .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?,
        ),
        None => None,
    };

    let prompt_ids: Vec<i64> = if let Some(text) = &args.prompt {
        let tk = tokenizer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--prompt requires --tokenizer"))?;
        tk.encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?
    } else if let Some(ids) = &args.prompt_ids {
        ids.split(',')
            .map(|s| s.trim().parse::<i64>())
            .collect::<Result<_, _>>()
            .context("parsing --prompt-ids")?
    } else {
        anyhow::bail!("either --prompt or --prompt-ids is required");
    };
    anyhow::ensure!(!prompt_ids.is_empty(), "prompt must contain at least 1 id");

    let chunk_product = cfg.model.chunk_size.pow(cfg.model.n_levels as u32 - 1);
    let sampler = Sampler {
        temperature: if args.temperature == 0.0 {
            1.0
        } else {
            args.temperature
        },
        top_k: args.top_k,
        top_p: args.top_p,
        greedy: args.temperature == 0.0,
    };
    let gen_cfg = GenerateConfig {
        max_new_tokens: args.max_new_tokens,
        chunk_product,
        vocab_size: cfg.model.vocab_size,
        pad_token_id: 0,
        seed: args.seed,
    };
    let out = match args.decode {
        DecodeMode::Stateful => {
            let gen = StatefulGenerator::new(&model, gen_cfg, sampler);
            gen.generate_with_hook(&bk, &prompt_ids, end_of_decode_step)?
        }
        DecodeMode::Reforward => {
            let gen = Generator::new(&model, gen_cfg, sampler);
            gen.generate(&bk, &prompt_ids)?
        }
    };
    if let Some(tk) = &tokenizer {
        match tk.decode(&out, true) {
            Ok(text) => println!("{text}"),
            Err(e) => {
                eprintln!("pm generate: decode failed ({e}); printing ids");
                let comma_sep: Vec<String> = out.iter().map(|i| i.to_string()).collect();
                println!("{}", comma_sep.join(","));
            }
        }
    } else {
        let comma_sep: Vec<String> = out.iter().map(|i| i.to_string()).collect();
        println!("{}", comma_sep.join(","));
    }
    Ok(())
}

/// Best-effort, backend-specific per-token decode housekeeping (Phase
/// C), passed to `StatefulGenerator::generate_with_hook`.
///
/// `pm_cuda::CudaBackend` records every op (even pure-inference,
/// non-`backward`-bound ones) on an autograd tape that is only ever
/// cleared by `Ops::backward` or the CUDA-specific `reset_tape()` —
/// there is no generic `Ops` operation for "this graph is inference-only,
/// don't bother tracking it". Without a periodic reset, a many-token
/// `--backend cuda` decode would grow VRAM roughly linearly in
/// `max_new_tokens` even though `pm-core`'s `PhotonDecodeState` itself
/// is fixed-size (measured directly in
/// `pm-cli/tests/decode_vram_flat.rs`). Calling `reset_tape()` is safe
/// here because decode never calls `Ops::backward`, so there is no
/// gradient computation whose tape entries this could discard.
///
/// Implemented with a runtime `Any` downcast instead of a new trait so
/// this stays entirely local to `pm-cli`: `pm-core` / `pm-infer` never
/// learn `CudaBackend` exists (CLAUDE.md invariant #2 — backend
/// addition must be trait-implementation-only there), and a backend
/// that doesn't need this kind of hygiene (or doesn't exist yet, e.g.
/// Tenstorrent) just doesn't match and this is a no-op.
fn end_of_decode_step<O: Ops + 'static>(bk: &O) {
    #[cfg(feature = "cuda")]
    {
        use std::any::Any;
        if let Some(cuda_bk) = (bk as &dyn Any).downcast_ref::<pm_cuda::CudaBackend>() {
            let _ = cuda_bk.reset_tape();
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = bk;
    }
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
                    "pm generate: cuda requested but binary built without --features cuda; \
                     falling back to CPU"
                );
                Ok(CandleBackend::new_cpu())
            }
        }
        other => anyhow::bail!("unknown device {other:?}"),
    }
}
