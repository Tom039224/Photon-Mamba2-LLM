//! B4.4f-T5: production-shape Mamba2Block finite-difference gradient check.
//!
//! T4 (in b44f_investigation.rs) used tiny dims (d_model=8) that kept all
//! matrix products well below the size that triggers cuBLAS tensor-core paths.
//! T4 also forced n_groups == n_heads to avoid the `broadcast_as` path.
//!
//! T5 addresses both gaps:
//! - **Production dims**: d_model=768, n_heads=12, d_head=64, d_state=128,
//!   n_groups=1 (MIS shared — triggers `broadcast_as`), d_conv=4, block_len=64.
//! - **B=1, T=64** — exactly 1 SSD chunk, keeping FD cost manageable (~80
//!   forwards × ~100 ms = ~8 s) while testing every op in the block.
//!
//! For each of the 8 trainable params, 5 element indices (uniformly spread) are
//! perturbed by ±eps (central difference) and the resulting FD estimate is
//! compared to the analytical gradient from `CudaBackend::backward`.
//!
//! ## Interpretation guide
//!
//! | rel_err  | verdict                                                     |
//! |----------|-------------------------------------------------------------|
//! | < 5 %    | VJP is individually clean at production scale               |
//! | 5–10 %   | marginal; warrants a focused investigation                  |
//! | > 10 %   | smoking gun — backward op for this param is buggy           |
//!
//! If **all params** show `rel_err < 5 %`, the 13 % grad_norm delta is
//! compound/accumulative across the 20 blocks, not a single-op bug.
//!
//! This test **does not assert failure** on rel_err.  It is a diagnostic
//! probe: run with `-- --nocapture` to see the full per-element table.  Only
//! NaN / Inf in any backward output is a hard failure.
//!
//! ## Parameter → op mapping
//!
//! | Param name       | Affected backward op(s)          |
//! |------------------|----------------------------------|
//! | in_proj_weight   | MatMul VJP (grad_W, grad_X)      |
//! | conv1d_weight    | Conv1d VJP (CPU im2col+GEMM)     |
//! | conv1d_bias      | Conv1d VJP                       |
//! | a_log            | Exp + Neg + SsdScan VJP          |
//! | d_skip           | Mul (skip-connection) VJP        |
//! | dt_bias          | Softplus + Add VJP               |
//! | norm_weight      | RmsNorm VJP                      |
//! | out_proj_weight  | MatMul VJP (output projection)   |

#![cfg(feature = "cuda")]

use pm_core::{
    mamba2::{Mamba2Block, Mamba2Config},
    Module, Ops, Param, Tensor,
};
use pm_cuda::{CudaBackend, CudaParam, CudaTensor};

// ---------------------------------------------------------------------------
// LCG deterministic random data
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Block construction at production dims (n_groups=1, triggers broadcast_as)
// ---------------------------------------------------------------------------

/// Build a `Mamba2Block<CudaBackend>` at production dims with LCG-random weights.
///
/// n_groups=1 is the production MIS configuration — it forces `broadcast_as`
/// inside `Mamba2Block::forward`, which T4 explicitly avoided.
fn make_prod_block(
    bk: &CudaBackend,
    cfg: &Mamba2Config,
    seed: u64,
) -> Result<Mamba2Block<CudaBackend>, pm_cuda::CudaError> {
    let d = cfg.d_model;
    let d_inner = cfg.d_inner();
    let xbc = cfg.xbc_dim();
    let in_p = cfg.in_proj_dim();
    let h = cfg.n_heads;
    let dc = cfg.d_conv;

    let mk = |s: u64, shape: &[usize], scale: f32, bias: f32| {
        let n: usize = shape.iter().product();
        bk.param_from_slice_f32(&lcg_vec(s, n, scale, bias), shape)
    };

    Ok(Mamba2Block {
        config: cfg.clone(),
        // Small but non-zero weights to keep activations in a sane range.
        in_proj_weight: mk(seed, &[d, in_p], 0.05, -0.025)?,
        conv1d_weight: mk(seed + 100, &[xbc, 1, dc], 0.05, -0.025)?,
        conv1d_bias: mk(seed + 200, &[xbc], 0.02, -0.01)?,
        a_log: mk(seed + 300, &[h], 0.20, -0.10)?,
        d_skip: mk(seed + 400, &[h], 0.20, -0.10)?,
        dt_bias: mk(seed + 500, &[h], 0.20, -0.10)?,
        // norm_weight initialised ~1 to mimic real training start.
        norm_weight: mk(seed + 600, &[d_inner], 0.10, 0.95)?,
        out_proj_weight: mk(seed + 700, &[d_inner, d], 0.05, -0.025)?,
    })
}

// ---------------------------------------------------------------------------
// Central-difference FD scalar
// ---------------------------------------------------------------------------

/// Central-difference FD estimate for element `idx` of `param`.
///
/// Perturbs param[idx] by ±eps, re-runs `block.forward` + `sum_all` each time
/// (no backward), and returns `(loss_plus - loss_minus) / (2·eps)`.
///
/// The param is restored to its original data after each call.
fn fd_scalar(
    bk: &CudaBackend,
    block: &Mamba2Block<CudaBackend>,
    x: &CudaTensor,
    param: &CudaParam,
    orig: &[f32],
    shape: &[usize],
    idx: usize,
    eps: f32,
) -> f32 {
    let eval = |data: Vec<f32>| -> f32 {
        let t = bk.from_slice_f32(&data, shape).expect("fd tensor");
        bk.assign(param, &t).expect("fd assign");
        let y = block.forward(bk, x).expect("fd forward");
        let loss_t = bk.sum_all(&y).expect("fd sum_all");
        let v = bk.to_vec_f32(&loss_t).expect("fd to_vec")[0];
        bk.reset_tape().expect("fd reset_tape");
        v
    };

    let mut p_plus = orig.to_vec();
    p_plus[idx] += eps;
    let loss_plus = eval(p_plus);

    let mut p_minus = orig.to_vec();
    p_minus[idx] -= eps;
    let loss_minus = eval(p_minus);

    // Restore original parameter value.
    let t_orig = bk.from_slice_f32(orig, shape).expect("fd restore");
    bk.assign(param, &t_orig).expect("fd restore assign");
    bk.reset_tape().expect("fd restore reset");

    (loss_plus - loss_minus) / (2.0 * eps)
}

// ---------------------------------------------------------------------------
// T5 test
// ---------------------------------------------------------------------------

/// Production-scale Mamba2Block FD gradient check.
///
/// `n_groups=1` uses `broadcast_as` (the path T4 skipped).  All matmuls are
/// 768×1804 and 768×768 — large enough to trigger cuBLAS tensor-core dispatch
/// if any tensor-core path is active.
///
/// Run with `--nocapture` for the full per-element table.
#[test]
fn b44f_t5_prod_block_fd_grad() {
    // Production config — matches configs/photon_mamba_100m.toml per block.
    let cfg = Mamba2Config {
        d_model: 768,
        d_state: 128,
        d_head: 64,
        n_heads: 12,
        n_groups: 1, // MIS shared; triggers broadcast_as in forward
        d_conv: 4,
        block_len: 64,
        rmsnorm_eps: 1e-5,
    };

    // T=64 = exactly 1 SSD chunk → inter-chunk accumulation is zero.
    // This isolates individual-op VJP errors from cross-chunk drift.
    let (b, t) = (1usize, 64usize);
    let d = cfg.d_model;

    let bk = CudaBackend::new(0).expect("CudaBackend::new(0)");
    let block = make_prod_block(&bk, &cfg, 5000).expect("make_prod_block");

    // Fixed input in (-0.25, 0.25) — small enough that activations stay finite.
    let x_data = lcg_vec(9999, b * t * d, 0.5, -0.25);
    let x = bk
        .from_slice_f32(&x_data, &[b, t, d])
        .expect("input tensor");

    // ── Phase 1: analytical backward ─────────────────────────────────────────
    let y = block.forward(&bk, &x).expect("analytical forward");
    let loss = bk.sum_all(&y).expect("sum_all");
    let store = bk.backward(&loss).expect("backward");

    // Ensure the forward output is sane before probing gradients.
    let y_data = bk.to_vec_f32(&y).expect("y to_vec");
    assert!(
        y_data.iter().all(|v| v.is_finite()),
        "forward output contains NaN/Inf — weights or input need rescaling"
    );

    // Param list mirrors Mamba2Block::append_params order.
    let params: &[(&str, &CudaParam)] = &[
        ("in_proj_weight", &block.in_proj_weight),
        ("conv1d_weight", &block.conv1d_weight),
        ("conv1d_bias", &block.conv1d_bias),
        ("a_log", &block.a_log),
        ("d_skip", &block.d_skip),
        ("dt_bias", &block.dt_bias),
        ("norm_weight", &block.norm_weight),
        ("out_proj_weight", &block.out_proj_weight),
    ];

    // Gather analytical gradients from the store.
    let ana_grads: Vec<Option<Vec<f32>>> = params
        .iter()
        .map(|&(_, p)| {
            bk.gradient(&store, p)
                .expect("gradient() call")
                .map(|g| bk.to_vec_f32(&g).expect("grad to_vec"))
        })
        .collect();

    // ── Phase 2: FD probes ───────────────────────────────────────────────────
    // eps calibration: at production dims (d_model=768) the loss magnitude is
    // large enough that (loss(w+eps) - loss(w-eps)) with eps=1e-3 loses most
    // of its significant fp32 digits to cancellation — an eps-sweep isolated
    // probe (in the follow-up to T5) showed norm_weight max_rel_err going
    // 39 % → 37 % → 13 % → 3.8 % → 0.8 % as eps was raised 1e-4 → 1e-2. So the
    // original T5 "SMOKING GUN" verdict for norm_weight and conv1d_weight at
    // eps=1e-3 was FD cancellation noise, not a real VJP bug. Use eps=1e-2
    // here and treat >5 % as genuinely suspicious.
    let eps = 1e-2_f32;
    let n_probes = 5usize;

    eprintln!();
    eprintln!("B4.4f-T5  production-shape Mamba2Block FD gradient check");
    eprintln!(
        "  config : d_model={d} n_heads=12 d_head=64 d_state=128 \
         n_groups=1 d_conv=4 block_len=64"
    );
    eprintln!("  input  : B={b} T={t}  loss=sum_all(output)  eps={eps:.0e}");
    eprintln!("  probes : {n_probes} per param (uniformly spread)");
    eprintln!("  n_groups=1 → broadcast_as path active (gap vs T4 which used n_groups=2)");
    eprintln!();

    let mut global_max_rel: f32 = 0.0;
    let mut global_worst_param = "none";
    let mut global_worst_idx: usize = 0;
    let mut global_worst_rel: f32 = 0.0;

    for (&(name, param), ana_opt) in params.iter().zip(ana_grads.iter()) {
        let orig = bk.to_vec_f32(param.as_tensor()).expect("orig to_vec");
        let shape = param.as_tensor().shape().to_vec();
        let n = orig.len();

        // Uniformly spread n_probes indices across [0, n-1].
        // Deduplication via BTreeSet handles tiny params (e.g. a_log has 12 elements).
        let probe_indices: Vec<usize> = {
            let mut set = std::collections::BTreeSet::new();
            for k in 0..n_probes {
                let idx = if n == 1 {
                    0
                } else {
                    k * (n - 1) / (n_probes - 1)
                };
                set.insert(idx.min(n - 1));
            }
            set.into_iter().collect()
        };

        let n_probed = probe_indices.len();
        eprintln!("  param={name:<18}  shape={shape:?}  n_probed={n_probed}");
        eprintln!(
            "    {:>6}  {:>14}  {:>14}  {:>12}  {:>12}",
            "idx", "analytic", "fd", "abs_err", "rel_err%"
        );

        let mut param_max_rel: f32 = 0.0;
        let mut param_worst_idx: usize = 0;

        for idx in probe_indices {
            let fd = fd_scalar(&bk, &block, &x, param, &orig, &shape, idx, eps);

            let (ana, abs_err, rel_err) = match ana_opt {
                None => {
                    // Param not reachable from loss — FD should also be ≈ 0.
                    let abs_err = fd.abs();
                    let rel_err = abs_err / (fd.abs().max(1e-8));
                    (f32::NAN, abs_err, rel_err)
                }
                Some(v) => {
                    let a = v[idx];
                    let abs_err = (a - fd).abs();
                    // Relative error denominator: max(|a|, |fd|, eps_floor).
                    let denom = a.abs().max(fd.abs()).max(1e-8);
                    let rel_err = abs_err / denom;
                    (a, abs_err, rel_err)
                }
            };

            let ana_display = if ana.is_nan() { 0.0 } else { ana };
            eprintln!(
                "    {:>6}  {:>14.6e}  {:>14.6e}  {:>12.4e}  {:>11.3}%",
                idx,
                ana_display,
                fd,
                abs_err,
                rel_err * 100.0
            );

            // Track worst for this param.
            if ana.is_finite() && rel_err > param_max_rel {
                param_max_rel = rel_err;
                param_worst_idx = idx;
            }
            // Track global worst.
            if ana.is_finite() && rel_err > global_max_rel {
                global_max_rel = rel_err;
                global_worst_param = name;
                global_worst_idx = idx;
                global_worst_rel = rel_err;
            }

            // Hard failure only on NaN/Inf (indicates a backward kernel crash).
            assert!(
                rel_err.is_finite(),
                "rel_err is NaN/Inf for param={name} idx={idx} — backward kernel produced invalid output"
            );
        }

        let verdict = if param_max_rel > 0.10 {
            "*** SMOKING GUN (> 10%) — VJP likely buggy"
        } else if param_max_rel > 0.05 {
            "** MARGINAL (> 5%)     — warrants investigation"
        } else {
            "OK (< 5%)"
        };
        eprintln!(
            "    → param max_rel_err = {:.3}%  worst_idx={}  {}",
            param_max_rel * 100.0,
            param_worst_idx,
            verdict
        );
        eprintln!();
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    eprintln!("  ════════════════════════════════════════════════");
    eprintln!("  SUMMARY: max_rel_err = {:.3}%", global_max_rel * 100.0);
    eprintln!(
        "           worst param = {} [idx={}]  rel={:.3}%",
        global_worst_param,
        global_worst_idx,
        global_worst_rel * 100.0,
    );
    eprintln!();

    if global_max_rel > 0.10 {
        eprintln!(
            "  VERDICT: SPECIFIC VJP BUG — param '{}' exceeds 10% \
             rel_err at production dims. Fix the backward op for that parameter.",
            global_worst_param
        );
    } else if global_max_rel > 0.05 {
        eprintln!(
            "  VERDICT: MARGINAL — max rel_err {:.1}% is elevated. \
             Could be compound truncation or a weak single-op error. \
             Probe more indices or reduce eps to distinguish.",
            global_max_rel * 100.0
        );
    } else {
        eprintln!("  VERDICT: ALL VJPs INDIVIDUALLY CLEAN (< 5% at production scale).");
        eprintln!("           The 13% grad_norm delta is COMPOUND across the 20 blocks,");
        eprintln!("           not a single-op bug. Root fix: eliminate ULP accumulation");
        eprintln!("           (e.g. bf16 accumulation, fused block backward, or");
        eprintln!("           Kahan compensation in the gradient norm sum).");
    }
    eprintln!("  ════════════════════════════════════════════════");
}
