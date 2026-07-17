//! Build `PhotonMamba<O>` or `FlatMamba<O>` from a `ModelConfig`.

use anyhow::Result;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::photon::{
    ChunkLocalDecoder, ContextChunker, ContextConverter, ContextEncoder, DecoderLevel,
    HierarchicalDecoder, HierarchicalEncoder, HierarchicalLevel, TokenEmbedding,
};
use pm_core::{Dtype, FlatMamba, Ops, PhotonMamba};

use crate::config::ModelConfig;

/// Parse the `[model].compute_dtype` TOML string into a `pm_core::Dtype`.
///
/// Only `"f32"` and `"bf16"` are accepted. `Dtype::F16` exists but is
/// rejected here — the bf16 mixed-precision path (memory-efficiency
/// plan Phase A2) has only been validated for bf16's wider exponent
/// range; f16's narrower range against the SSD scan's dynamic range is
/// untested.
fn parse_compute_dtype(s: &str) -> Result<Dtype> {
    match s {
        "f32" => Ok(Dtype::F32),
        "bf16" => Ok(Dtype::BF16),
        other => anyhow::bail!("unknown compute_dtype {other:?}; expected \"f32\" or \"bf16\""),
    }
}

pub fn build_photon_mamba<O: Ops>(bk: &O, cfg: &ModelConfig) -> Result<PhotonMamba<O>> {
    anyhow::ensure!(
        cfg.n_levels == 2,
        "Phase 1 only supports n_levels=2 (got {})",
        cfg.n_levels
    );
    let compute_dtype = parse_compute_dtype(&cfg.compute_dtype)?;

    let m2_cfg = Mamba2Config {
        d_model: cfg.d_model,
        d_state: cfg.d_state,
        d_head: cfg.d_head,
        n_heads: cfg.n_heads,
        n_groups: cfg.n_groups,
        d_conv: cfg.d_conv,
        block_len: cfg.block_len,
        rmsnorm_eps: cfg.rmsnorm_eps,
    };

    let mk_layers = |n: usize| -> Result<Vec<Mamba2Block<O>>> {
        (0..n)
            .map(|_| {
                Ok(Mamba2Block::from_constants(
                    bk,
                    m2_cfg.clone(),
                    cfg.init_scale,
                )?)
            })
            .collect()
    };

    let embed = TokenEmbedding::from_constants(bk, cfg.vocab_size, cfg.d_model, cfg.init_scale)?;

    let lvl0 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(mk_layers(cfg.n_layers_per_level)?),
        chunker: Some(ContextChunker::from_constants(
            bk,
            cfg.d_model,
            cfg.d_model,
            cfg.chunk_size,
            cfg.init_scale,
        )?),
    };
    let lvl1 = HierarchicalLevel {
        encoder: ContextEncoder::from_layers(mk_layers(cfg.n_layers_per_level)?),
        chunker: None,
    };
    let encoder = HierarchicalEncoder::from_levels(vec![lvl0, lvl1]);

    let conv = ContextConverter::from_constants(
        bk,
        cfg.d_model,
        cfg.d_model,
        cfg.chunk_size,
        cfg.init_scale,
    )?;
    let dec_stack = ChunkLocalDecoder::from_layers(
        mk_layers(cfg.n_layers_per_level)?,
        cfg.chunk_size,
        cfg.chunk_size,
    );
    let decoder = HierarchicalDecoder::from_levels(vec![DecoderLevel::new(conv, dec_stack)]);

    Ok(PhotonMamba::new(embed, encoder, decoder).with_compute_dtype(compute_dtype))
}

/// Build a [`FlatMamba`] from a `ModelConfig` with `model_type = "flat"`.
///
/// Reads: `vocab_size`, `d_model`, `d_state`, `d_head`, `n_heads`,
/// `n_groups`, `d_conv`, `block_len`, `rmsnorm_eps`, `init_scale`,
/// `compute_dtype`, and `flat_n_layers`.
///
/// Silently ignores `n_levels`, `n_layers_per_level`, and `chunk_size`
/// (present in TOML for schema compatibility, not meaningful for a flat model).
///
/// Errors if `flat_n_layers` is `None` or if `compute_dtype` is unknown.
///
/// PLAN.md Phase D.2a.
pub fn build_flat_mamba<O: Ops>(bk: &O, cfg: &ModelConfig) -> Result<FlatMamba<O>> {
    let flat_n_layers = cfg.flat_n_layers.ok_or_else(|| {
        anyhow::anyhow!(
            "[model].flat_n_layers is required when model_type = \"flat\" but was not set"
        )
    })?;
    anyhow::ensure!(
        flat_n_layers > 0,
        "flat_n_layers must be > 0, got {flat_n_layers}"
    );

    let compute_dtype = parse_compute_dtype(&cfg.compute_dtype)?;

    let m2_cfg = Mamba2Config {
        d_model: cfg.d_model,
        d_state: cfg.d_state,
        d_head: cfg.d_head,
        n_heads: cfg.n_heads,
        n_groups: cfg.n_groups,
        d_conv: cfg.d_conv,
        block_len: cfg.block_len,
        rmsnorm_eps: cfg.rmsnorm_eps,
    };

    // Reuse the same mk_layers pattern as build_photon_mamba.
    let mk_layers = |n: usize| -> Result<Vec<Mamba2Block<O>>> {
        (0..n)
            .map(|_| {
                Ok(Mamba2Block::from_constants(
                    bk,
                    m2_cfg.clone(),
                    cfg.init_scale,
                )?)
            })
            .collect()
    };

    let embed = TokenEmbedding::from_constants(bk, cfg.vocab_size, cfg.d_model, cfg.init_scale)?;
    let trunk = ContextEncoder::from_layers(mk_layers(flat_n_layers)?);

    Ok(FlatMamba::new(embed, trunk).with_compute_dtype(compute_dtype))
}

/// Validate that a flat config does not request a consistency loss.
/// Returns `Err` if `model_type == "flat"` and `consistency_alpha > 0.0`.
///
/// The recursive-consistency loss (PHOTON §2.3 Eq. (11)) requires a
/// hierarchical encoder with multiple levels — a flat model has no
/// hierarchy, so α > 0 is nonsensical and is rejected at config validation
/// time rather than silently becoming a no-op.
pub fn validate_flat_config_alpha(model_type: &str, consistency_alpha: f32) -> anyhow::Result<()> {
    if model_type == "flat" && consistency_alpha > 0.0 {
        anyhow::bail!(
            "flat baseline has no hierarchy; consistency_alpha must be 0 \
             (got {consistency_alpha:.4}). Use model_type = \"photon\" for the \
             recursive-consistency loss (PLAN.md Phase D.1/D.2)"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PLAN.md Phase D.2a: `model_type="flat"` with `consistency_alpha>0`
    /// must be rejected at config-validation time, not silently allowed to
    /// become a no-op inside `fused_photon_loss_injected`. The flat model
    /// has no hierarchical encoder levels, so α>0 is nonsensical.
    #[test]
    fn flat_with_positive_alpha_is_rejected() {
        let err = validate_flat_config_alpha("flat", 0.3).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("consistency_alpha must be 0"),
            "expected rejection message, got: {msg}"
        );
    }

    /// α=0 must always pass, regardless of model type.
    #[test]
    fn flat_with_alpha_zero_is_accepted() {
        validate_flat_config_alpha("flat", 0.0).expect("α=0 must be allowed for flat");
    }

    /// PHOTON (hierarchy present) may use any α ≥ 0.
    #[test]
    fn photon_with_positive_alpha_is_accepted() {
        validate_flat_config_alpha("photon", 0.3).expect("α>0 is valid for photon");
    }
}
