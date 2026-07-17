//! F.8 checkpoint save/load round-trip.

use pm_candle::CandleBackend;
use pm_core::mamba2::{Mamba2Block, Mamba2Config};
use pm_core::{Ops, Param, Parameterized};
use pm_train::{load_checkpoint, save_checkpoint};
use std::path::PathBuf;

fn tmp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pm_train_test_{}_{}.safetensors",
        name,
        std::process::id()
    ));
    p
}

#[test]
fn save_then_load_recovers_param_values() -> anyhow::Result<()> {
    let bk = CandleBackend::new_cpu();
    let cfg = Mamba2Config {
        d_model: 16,
        d_state: 8,
        d_head: 8,
        n_heads: 2,
        n_groups: 1,
        d_conv: 4,
        block_len: 4,
        rmsnorm_eps: 1e-5,
    };

    // Source model with weight_scale=0.123 (a non-default value we'll
    // verify came back exactly).
    let src = Mamba2Block::from_constants(&bk, cfg.clone(), 0.123)?;
    let src_params = src.collect_params();
    let path = tmp_path("mamba2_block");
    save_checkpoint(&bk, &src_params, &path)?;

    // Destination model with weight_scale=0.999 — different on every weight.
    let dst = Mamba2Block::from_constants(&bk, cfg, 0.999)?;
    let dst_params = dst.collect_params();
    load_checkpoint(&bk, &dst_params, &path)?;

    // Every dst param's tensor must equal the corresponding src param's tensor.
    for (s, d) in src_params.iter().zip(dst_params.iter()) {
        let sv = bk.to_vec_f32(s.as_tensor())?;
        let dv = bk.to_vec_f32(d.as_tensor())?;
        assert_eq!(sv.len(), dv.len(), "shape mismatch after load");
        for (sx, dx) in sv.iter().zip(dv.iter()) {
            assert!(
                (sx - dx).abs() < 1e-6,
                "value mismatch after load: src={sx}, dst={dx}"
            );
        }
    }

    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn load_rejects_param_count_mismatch() -> anyhow::Result<()> {
    let bk = CandleBackend::new_cpu();
    let cfg = Mamba2Config {
        d_model: 16,
        d_state: 8,
        d_head: 8,
        n_heads: 2,
        n_groups: 1,
        d_conv: 4,
        block_len: 4,
        rmsnorm_eps: 1e-5,
    };
    let src = Mamba2Block::from_constants(&bk, cfg.clone(), 0.1)?;
    let path = tmp_path("count_mismatch");
    save_checkpoint(&bk, &src.collect_params(), &path)?;

    // Load into a *shorter* param list than we saved.
    let short = bk.param_from_slice_f32(&[0.0; 4], &[4])?;
    let short_params: Vec<&pm_candle::CandleParam> = vec![&short];
    let err = load_checkpoint(&bk, &short_params, &path);
    assert!(err.is_err(), "expected error on param count mismatch");

    std::fs::remove_file(&path).ok();
    Ok(())
}
