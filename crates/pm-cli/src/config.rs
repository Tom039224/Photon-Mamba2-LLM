//! TOML config schema.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub model: ModelConfig,
    pub train: TrainConfig,
    #[serde(default)]
    pub runtime: RuntimeConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_state: usize,
    pub d_head: usize,
    pub n_heads: usize,
    pub n_groups: usize,
    pub d_conv: usize,
    pub block_len: usize,
    pub rmsnorm_eps: f32,
    /// Mamba2 layers per encoder level (and per decoder level).
    pub n_layers_per_level: usize,
    /// Number of hierarchical encoder levels (L). Phase 1 only supports L=2.
    pub n_levels: usize,
    /// Chunk size = converter expansion R = chunker C (P.5 in deviations).
    pub chunk_size: usize,
    /// Initial uniform scale applied to all weight tensors.
    pub init_scale: f32,
    /// Forward-pass compute dtype: `"f32"` (default) or `"bf16"`.
    /// `"bf16"` halves training activation-tape memory by flowing the
    /// big `(B,T,·)` activations and the `in_proj`/`out_proj` matmuls
    /// in bf16, while numerically-sensitive sub-computations (SSD
    /// scan, softplus/exp, rmsnorm, cross-entropy) stay fp32
    /// internally (memory-efficiency plan Phase A2). Parameters and
    /// the optimizer state are unaffected — always fp32.
    #[serde(default = "default_compute_dtype")]
    pub compute_dtype: String,
    /// Model architecture selector. `"photon"` (default) or `"flat"`.
    ///
    /// `"photon"` builds the full PHOTON hierarchical encoder+decoder
    /// (`n_levels`, `n_layers_per_level`, `chunk_size`).
    /// `"flat"` builds the non-hierarchical Mamba2 LM baseline (D.2a):
    /// `flat_n_layers` Mamba2 blocks in a single residual stack, same
    /// embedding/lm_head as PHOTON, no chunker/converter/decoder.
    ///
    /// When `"flat"`, the PHOTON-specific fields (`n_levels`,
    /// `n_layers_per_level`, `chunk_size`) must still appear in the TOML
    /// (they are required schema fields) but are silently ignored.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Total number of Mamba2 layers in the flat trunk.
    /// Required when `model_type = "flat"`, ignored when `"photon"`.
    #[serde(default)]
    pub flat_n_layers: Option<usize>,
}

fn default_compute_dtype() -> String {
    "f32".to_string()
}

fn default_model_type() -> String {
    "photon".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainConfig {
    pub seq_len: usize,
    pub batch_size: usize,
    pub n_steps: usize,
    pub lr: f32,
    #[serde(default = "default_beta1")]
    pub beta1: f32,
    #[serde(default = "default_beta2")]
    pub beta2: f32,
    #[serde(default = "default_eps")]
    pub eps: f32,
    #[serde(default)]
    pub weight_decay: f32,
    /// Global L2-norm clip. `None`/0 disables.
    #[serde(default)]
    pub max_grad_norm: Option<f32>,
    /// Seed for random-id generation in the toy data loop.
    #[serde(default)]
    pub seed: u64,
    /// Seed for parameter perturbation (breaks the symmetry of
    /// `from_constants`-style init). 0 disables.
    #[serde(default)]
    pub init_perturb_seed: u64,
    /// Save the final checkpoint to this path (relative to CWD). Empty disables.
    #[serde(default)]
    pub save_path: String,
    /// Log a loss line every `log_every` steps. `0` defaults to 1.
    #[serde(default)]
    pub log_every: usize,
    /// Path to a UTF-8 text file. When set together with `tokenizer`,
    /// the trainer streams real tokens instead of random ones.
    #[serde(default)]
    pub text_data: String,
    /// Path to a HuggingFace `tokenizer.json`. Required when
    /// `text_data` is set.
    #[serde(default)]
    pub tokenizer: String,
    /// Token id inserted between documents in the text stream.
    /// Defaults to 0 (typically GPT-2 BOS).
    #[serde(default)]
    pub doc_sep_id: i64,
    /// Save a checkpoint every `save_every` steps (in addition to the
    /// final save). 0 disables periodic saves. Requires `save_path`.
    #[serde(default)]
    pub save_every: usize,
    /// Hard wall-time cap. Training stops cleanly (with a final save)
    /// once this many seconds have elapsed. `None` / 0 disables.
    #[serde(default)]
    pub max_wall_time_seconds: Option<f32>,
    /// When true, generate a fresh random batch every step instead of
    /// overfitting on a single fixed batch. Ignored when `text_data`
    /// is set (real data always uses fresh batches).
    #[serde(default)]
    pub fresh_batch_per_step: bool,
    /// Use per-`Mamba2Block` activation checkpointing (F.6). Trades a
    /// recompute pass during backward for much lower peak memory —
    /// required for `batch_size > 1` at `seq_len = 512`.
    #[serde(default)]
    pub activation_checkpointing: bool,
    /// Weight α on the recursive-consistency auxiliary loss (Phase
    /// D.1; PHOTON §2.3 Eq. (9), `L = L_token + α·L_rec`). `0.0`
    /// (default, so existing config files are unaffected) reproduces
    /// training exactly as before this option existed — see
    /// `pm_train::loss::fused_photon_loss_injected`'s "Zero-cost at
    /// α = 0" doc section. The paper's own main-result runs also use
    /// α=0 (isolates the hierarchy/bounded-decoding gains); α≈0.3 is
    /// their appendix ablation optimum for downstream zero-shot
    /// accuracy.
    #[serde(default)]
    pub consistency_alpha: f32,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RuntimeConfig {
    /// `"cpu"` (default) or `"cuda"`. When `"cuda"` is requested but the
    /// binary was built without `--features cuda`, falls back to CPU.
    #[serde(default = "default_device")]
    pub device: String,
}

const fn default_beta1() -> f32 {
    0.9
}
const fn default_beta2() -> f32 {
    0.999
}
const fn default_eps() -> f32 {
    1e-8
}
fn default_device() -> String {
    "cpu".to_string()
}

impl Config {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let cfg: Config = toml::from_str(&s)?;
        Ok(cfg)
    }
}
