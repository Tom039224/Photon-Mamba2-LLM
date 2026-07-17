//! B4.4f root-cause investigation: Jacobian mismatch between PTX P2 forward
//! and ops_default backward recompute at production shape.
//!
//! Two experiments:
//!
//! * **T1** — forward output drift: max abs diff between PTX `ssd_scan_chunked`
//!   and `ssd_scan_ops_default` at (B=1, T=512, H=12, P=64, N=128, Q=64).
//!
//! * **T2** — gradient Jacobian mismatch: compare grad_{x,a,b,c} from
//!   Path A (PTX fwd + ops_default bwd via `ssd_scan_backward_vjp`) versus
//!   Path B (ops_default fwd + ops_default bwd, Candle-equivalent).
//!
//! If Hypothesis 1 (Jacobian mismatch) is the root cause, Path A grads will
//! differ measurably from Path B grads and the discrepancy should scale with T.

#![cfg(feature = "cuda")]

use pm_core::{
    mamba2::{ssd_scan_ops_default, Mamba2Block, Mamba2Config},
    Module, Ops, Param, Tensor,
};
use pm_cuda::{CudaBackend, CudaParam, CudaTensor};

// ---------------------------------------------------------------------------
// Helpers
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

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn mean_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// Run one ssd_scan + backward and return (grad_x, grad_a, grad_b, grad_c).
// `use_ptx = true`  → CudaBackend::ssd_scan (PTX forward + ops_default backward via VJP).
// `use_ptx = false` → ssd_scan_ops_default (ops_default forward AND backward).
#[allow(clippy::too_many_arguments)]
fn run_backward(
    bk: &CudaBackend,
    x_data: &[f32],
    a_data: &[f32],
    b_data: &[f32],
    c_data: &[f32],
    bat: usize,
    t: usize,
    h: usize,
    p: usize,
    n: usize,
    q: usize,
    use_ptx: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let px = bk
        .param_from_slice_f32(x_data, &[bat, t, h, p])
        .expect("param x");
    let pa = bk
        .param_from_slice_f32(a_data, &[bat, t, h])
        .expect("param a");
    let pb = bk
        .param_from_slice_f32(b_data, &[bat, t, h, n])
        .expect("param b");
    let pc = bk
        .param_from_slice_f32(c_data, &[bat, t, h, n])
        .expect("param c");

    let y = if use_ptx {
        bk.ssd_scan(
            px.as_tensor(),
            pa.as_tensor(),
            pb.as_tensor(),
            pc.as_tensor(),
            q,
        )
        .expect("ssd_scan PTX")
    } else {
        ssd_scan_ops_default(
            bk,
            px.as_tensor(),
            pa.as_tensor(),
            pb.as_tensor(),
            pc.as_tensor(),
            q,
        )
        .expect("ssd_scan ops_default")
    };

    let loss = bk.sum_all(&y).expect("sum_all");
    let store = bk.backward(&loss).expect("backward");

    let gx = bk
        .to_vec_f32(
            &bk.gradient(&store, &px)
                .expect("grad x ok")
                .expect("grad x Some"),
        )
        .expect("gx vec");
    let ga = bk
        .to_vec_f32(
            &bk.gradient(&store, &pa)
                .expect("grad a ok")
                .expect("grad a Some"),
        )
        .expect("ga vec");
    let gb = bk
        .to_vec_f32(
            &bk.gradient(&store, &pb)
                .expect("grad b ok")
                .expect("grad b Some"),
        )
        .expect("gb vec");
    let gc = bk
        .to_vec_f32(
            &bk.gradient(&store, &pc)
                .expect("grad c ok")
                .expect("grad c Some"),
        )
        .expect("gc vec");
    (gx, ga, gb, gc)
}

// ---------------------------------------------------------------------------
// T1: forward output drift PTX vs ops_default at production shape
// ---------------------------------------------------------------------------

/// Compare the PTX P2 forward output with the ops_default forward output at
/// the production training shape (B=1, T=512, H=12, P=64, N=128, Q=64).
///
/// This shape routes to `ssd_scan_chunked_p1` (n_dim=128, block_len=64,
/// p_dim=64) per the dispatch in `crates/pm-cuda/src/ssd.rs`.
///
/// Expected: if Hypothesis 1 (Jacobian mismatch) is correct, the forward
/// outputs differ by more than the 1e-4 bound of `ssd_parity_p2_shape_numerical`
/// (which only uses T=128, H=1).
#[test]
fn b44f_t1_forward_drift_production_shape() {
    let (bat, t, h, p, n, q) = (1usize, 512, 12, 64, 128, 64);
    let ni = bat * t * h;

    let x_h = lcg_vec(101, ni * p, 1.0, -0.5);
    // A small and negative so exp stays in range over T=512 steps.
    let a_h = lcg_vec(102, ni, 0.05, -0.05);
    let b_h = lcg_vec(103, ni * n, 0.5, -0.25);
    let c_h = lcg_vec(104, ni * n, 0.5, -0.25);

    // PTX path: round-trips host→device→host.
    let y_ptx = pm_cuda::ssd_scan_chunked(&x_h, &a_h, &b_h, &c_h, bat, t, h, p, n, q)
        .expect("PTX kernel launch");

    // ops_default path via CudaBackend primitive ops (no grad tracking needed).
    let bk = CudaBackend::new(0).expect("CudaBackend::new(0)");
    let y_ops = {
        let x_t = bk.from_slice_f32(&x_h, &[bat, t, h, p]).expect("x tensor");
        let a_t = bk.from_slice_f32(&a_h, &[bat, t, h]).expect("a tensor");
        let b_t = bk.from_slice_f32(&b_h, &[bat, t, h, n]).expect("b tensor");
        let c_t = bk.from_slice_f32(&c_h, &[bat, t, h, n]).expect("c tensor");
        let y = ssd_scan_ops_default(&bk, &x_t, &a_t, &b_t, &c_t, q).expect("ops_default fwd");
        bk.to_vec_f32(&y).expect("to_vec y")
    };

    let max_err = max_abs_err(&y_ptx, &y_ops);
    let mean_err = mean_abs_err(&y_ptx, &y_ops);
    let n_exceed_1e4 = y_ptx
        .iter()
        .zip(y_ops.iter())
        .filter(|&(a, b)| (a - b).abs() > 1e-4)
        .count();
    let total = y_ptx.len();

    eprintln!("B4.4f-T1 fwd PTX vs ops_default (B={bat},T={t},H={h},P={p},N={n},Q={q}):");
    eprintln!("  total elements   = {total}");
    eprintln!("  max_abs_err      = {max_err:.3e}");
    eprintln!("  mean_abs_err     = {mean_err:.3e}");
    eprintln!(
        "  elements > 1e-4  = {n_exceed_1e4} / {total}  ({:.2}%)",
        n_exceed_1e4 as f32 / total as f32 * 100.0
    );

    // Not asserting a tight bound — this test is diagnostic.
    // Hypothesis: max_err > 1e-4 (i.e., exceeds the bound in ssd_parity_p2_shape_numerical).
    assert!(max_err.is_finite(), "forward max_abs_err is NaN/Inf");
}

// ---------------------------------------------------------------------------
// T2: gradient Jacobian mismatch PTX vs ops_default at production shape
// ---------------------------------------------------------------------------

/// Compare gradients from Path A (PTX fwd + ops_default bwd) vs Path B
/// (ops_default fwd + ops_default bwd) at production shape.
///
/// Path A is the actual production no-ckpt code path. Path B is the reference
/// (Candle-equivalent). If the Jacobian-mismatch hypothesis is correct,
/// the grad norms from Path A should exceed Path B, and the excess should
/// correlate with the 13% step-0 grad_norm delta observed in B4.4f.
#[test]
fn b44f_t2_grad_jacobian_mismatch_production_shape() {
    let (bat, t, h, p, n, q) = (1usize, 512, 12, 64, 128, 64);
    let ni = bat * t * h;

    let x_h = lcg_vec(201, ni * p, 1.0, -0.5);
    let a_h = lcg_vec(202, ni, 0.05, -0.05);
    let b_h = lcg_vec(203, ni * n, 0.5, -0.25);
    let c_h = lcg_vec(204, ni * n, 0.5, -0.25);

    let bk = CudaBackend::new(0).expect("CudaBackend::new(0)");

    // Path A: PTX forward → ops_default backward (production no-ckpt path).
    let (gx_a, ga_a, gb_a, gc_a) =
        run_backward(&bk, &x_h, &a_h, &b_h, &c_h, bat, t, h, p, n, q, true);

    // Path B: ops_default forward → ops_default backward (Candle-equivalent reference).
    let (gx_b, ga_b, gb_b, gc_b) =
        run_backward(&bk, &x_h, &a_h, &b_h, &c_h, bat, t, h, p, n, q, false);

    macro_rules! report {
        ($name:expr, $a:expr, $b:expr) => {{
            let norm_a = l2_norm(&$a);
            let norm_b = l2_norm(&$b);
            let max_err = max_abs_err(&$a, &$b);
            let mean_err = mean_abs_err(&$a, &$b);
            let pct = (norm_a - norm_b).abs() / norm_b.max(1e-30) * 100.0;
            eprintln!(
                "  grad_{}: ||A||={:.4e}  ||B||={:.4e}  Δnorm={:.2}%  \
                 max_err={:.3e}  mean_err={:.3e}",
                $name, norm_a, norm_b, pct, max_err, mean_err
            );
            assert!(max_err.is_finite(), "grad_{} max_err is NaN/Inf", $name);
        }};
    }

    eprintln!("B4.4f-T2 grad Jacobian mismatch (B={bat},T={t},H={h},P={p},N={n},Q={q}):");
    eprintln!("  Path A = PTX fwd + ops_default bwd  (production no-ckpt)");
    eprintln!("  Path B = ops_default fwd + ops_default bwd  (Candle-equivalent)");
    report!("x", gx_a, gx_b);
    report!("a", ga_a, ga_b);
    report!("b", gb_a, gb_b);
    report!("c", gc_a, gc_b);
}

// ---------------------------------------------------------------------------
// T3: forward drift scaling with T (T=128 vs T=512 comparison)
// ---------------------------------------------------------------------------

/// Run T1 at T=128 (2 chunks) and T=512 (8 chunks) to see if drift scales
/// with sequence length / number of chunks.  Supports the inter-chunk
/// h-state accumulation hypothesis.
#[test]
fn b44f_t3_forward_drift_scaling_with_t() {
    for &t_len in &[128usize, 512] {
        let (bat, h, p, n, q) = (1usize, 12, 64, 128, 64);
        let ni = bat * t_len * h;

        let x_h = lcg_vec(301, ni * p, 1.0, -0.5);
        let a_h = lcg_vec(302, ni, 0.05, -0.05);
        let b_h = lcg_vec(303, ni * n, 0.5, -0.25);
        let c_h = lcg_vec(304, ni * n, 0.5, -0.25);

        let y_ptx = pm_cuda::ssd_scan_chunked(&x_h, &a_h, &b_h, &c_h, bat, t_len, h, p, n, q)
            .expect("PTX launch");

        let bk = CudaBackend::new(0).expect("CudaBackend");
        let y_ops = {
            let x_t = bk.from_slice_f32(&x_h, &[bat, t_len, h, p]).expect("x");
            let a_t = bk.from_slice_f32(&a_h, &[bat, t_len, h]).expect("a");
            let b_t = bk.from_slice_f32(&b_h, &[bat, t_len, h, n]).expect("b");
            let c_t = bk.from_slice_f32(&c_h, &[bat, t_len, h, n]).expect("c");
            let y = ssd_scan_ops_default(&bk, &x_t, &a_t, &b_t, &c_t, q).expect("ops_default");
            bk.to_vec_f32(&y).expect("to_vec y")
        };

        let max_err = max_abs_err(&y_ptx, &y_ops);
        let mean_err = mean_abs_err(&y_ptx, &y_ops);
        let n_chunks = t_len / q;
        eprintln!(
            "B4.4f-T3 drift at T={t_len} ({n_chunks} chunks): \
             max_err={max_err:.3e}  mean_err={mean_err:.3e}"
        );
        assert!(max_err.is_finite(), "T={t_len} max_err is NaN/Inf");
    }
}

// ---------------------------------------------------------------------------
// T4 helpers — FD gradient check for a full Mamba2Block
// ---------------------------------------------------------------------------

/// Create a `Mamba2Block<CudaBackend>` with LCG-random weights.
///
/// Uses tiny dims so the block runs on the ssd_scan FALLBACK path
/// (not the PTX P1/P2 kernel), which matches the ops_default reference.
/// `n_groups == n_heads` avoids the `broadcast_as` path.
fn make_test_block(
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

    // Each param gets a different seed offset to avoid identical rows/columns.
    let mk = |s: u64, shape: &[usize], scale: f32, bias: f32| {
        let n: usize = shape.iter().product();
        bk.param_from_slice_f32(&lcg_vec(s, n, scale, bias), shape)
    };

    Ok(Mamba2Block {
        config: cfg.clone(),
        in_proj_weight: mk(seed, &[d, in_p], 0.05, -0.025)?,
        conv1d_weight: mk(seed + 100, &[xbc, 1, dc], 0.05, -0.025)?,
        conv1d_bias: mk(seed + 200, &[xbc], 0.02, -0.01)?,
        a_log: mk(seed + 300, &[h], 0.20, -0.10)?,
        d_skip: mk(seed + 400, &[h], 0.20, -0.10)?,
        dt_bias: mk(seed + 500, &[h], 0.20, -0.10)?,
        norm_weight: mk(seed + 600, &[d_inner], 0.10, 0.95)?, // initialised ~1
        out_proj_weight: mk(seed + 700, &[d_inner, d], 0.05, -0.025)?,
    })
}

/// Compute the central-difference FD gradient for one element of one param.
///
/// Perturbs `param[idx]` by ±eps, runs the full block forward (tape is
/// discarded with `reset_tape()` — no backward), and returns
/// `(loss_plus − loss_minus) / (2·eps)`.
///
/// The param is restored to its original data before returning.
#[allow(clippy::too_many_arguments)]
fn fd_scalar(
    bk: &CudaBackend,
    block: &Mamba2Block<CudaBackend>,
    x: &CudaTensor,
    param: &CudaParam,
    orig_data: &[f32],
    shape: &[usize],
    idx: usize,
    eps: f32,
) -> f32 {
    // Helper: run forward with the param set to `data`, return scalar loss.
    let eval = |data: Vec<f32>| -> f32 {
        let t = bk.from_slice_f32(&data, shape).expect("fd tensor");
        bk.assign(param, &t).expect("fd assign");
        let y = block.forward(bk, x).expect("fd forward");
        let loss_t = bk.sum_all(&y).expect("fd sum_all");
        let val = bk.to_vec_f32(&loss_t).expect("fd to_vec")[0];
        bk.reset_tape().expect("fd reset_tape");
        val
    };

    let mut p_plus = orig_data.to_vec();
    p_plus[idx] += eps;
    let loss_plus = eval(p_plus);

    let mut p_minus = orig_data.to_vec();
    p_minus[idx] -= eps;
    let loss_minus = eval(p_minus);

    // Restore original so subsequent calls see unmodified params.
    let t_orig = bk
        .from_slice_f32(orig_data, shape)
        .expect("fd restore tensor");
    bk.assign(param, &t_orig).expect("fd restore assign");
    bk.reset_tape().expect("fd restore reset_tape");

    (loss_plus - loss_minus) / (2.0 * eps)
}

// ---------------------------------------------------------------------------
// T4: FD gradient check — Mamba2Block at small scale
// ---------------------------------------------------------------------------

/// Finite-difference gradient check for all 8 trainable parameters of a
/// tiny Mamba2Block on CudaBackend.
///
/// Setup (fallback ssd_scan path, n_groups==n_heads avoids broadcast_as):
///   d_model=8, n_heads=2, d_head=4, d_state=4, n_groups=2,
///   d_conv=2, block_len=4, B=1, T=4.
///
/// For each param, 3 element indices are probed.  The analytical grad
/// (from `backward`) is compared to the central-difference FD estimate.
///
/// Expected if backward is correct: max relative error ≲ 1 %
/// (O(eps²) truncation, eps = 1e-3).
///
/// A relative error ≫ 1 % for a specific param identifies the buggy op.
/// This is a *diagnostic* test — it reports findings but only hard-fails
/// on NaN/Inf (which would indicate a crash in the backward kernel).
#[test]
fn b44f_t4_fd_grad_check_mamba2block() {
    let cfg = Mamba2Config {
        d_model: 8,
        d_state: 4,
        d_head: 4,
        n_heads: 2,
        n_groups: 2, // == n_heads → skips broadcast_as
        d_conv: 2,
        block_len: 4,
        rmsnorm_eps: 1e-5,
    };

    let bk = CudaBackend::new(0).expect("CudaBackend::new(0)");
    let block = make_test_block(&bk, &cfg, 500).expect("make_test_block");

    // Input: [B=1, T=4, D=8], small random values in (−0.25, 0.25).
    let x_data = lcg_vec(9901, 4 * 8, 0.5, -0.25);
    let x = bk.from_slice_f32(&x_data, &[1, 4, 8]).expect("x tensor");

    // ── Phase 1: analytical backward ────────────────────────────────────────
    let y = block.forward(&bk, &x).expect("analytical forward");
    let loss = bk.sum_all(&y).expect("sum_all");
    let store = bk.backward(&loss).expect("backward");
    // Tape is now empty; ParamIds in `store` are stable.

    // Collect (name, analytical_grad_vec) for each param.
    let param_entries: &[(&str, &CudaParam)] = &[
        ("in_proj_weight", &block.in_proj_weight),
        ("conv1d_weight", &block.conv1d_weight),
        ("conv1d_bias", &block.conv1d_bias),
        ("a_log", &block.a_log),
        ("d_skip", &block.d_skip),
        ("dt_bias", &block.dt_bias),
        ("norm_weight", &block.norm_weight),
        ("out_proj_weight", &block.out_proj_weight),
    ];

    let ana_grads: Vec<Option<Vec<f32>>> = param_entries
        .iter()
        .map(|&(_, p)| {
            bk.gradient(&store, p)
                .expect("gradient() call")
                .map(|g| bk.to_vec_f32(&g).expect("grad to_vec"))
        })
        .collect();

    // ── Phase 2: central-difference FD for each param ───────────────────────
    let eps = 1e-3_f32;

    eprintln!("B4.4f-T4 FD gradient check (Mamba2Block, CudaBackend fallback path):");
    eprintln!("  dims: d_model=8 n_heads=2 d_head=4 d_state=4 n_groups=2 d_conv=2 block_len=4");
    eprintln!("  B=1, T=4,  loss=sum_all(output),  eps={eps:.0e}");
    eprintln!(
        "  {:>20}  {:>5}  {:>12}  {:>12}  {:>10}",
        "param", "idx", "ana_grad", "fd_grad", "rel_err%"
    );

    let mut max_rel_err = 0f32;
    let mut worst_param = "none";
    let mut worst_idx = 0usize;
    let mut worst_rel = 0f32;

    for (&(name, param), ana_opt) in param_entries.iter().zip(ana_grads.iter()) {
        let orig_data = bk.to_vec_f32(param.as_tensor()).expect("param to_vec");
        let shape = param.as_tensor().shape().to_vec();
        let n = orig_data.len();

        // Three spread-out indices: 0, n/3, 2n/3.
        let indices = [0_usize, (n / 3).max(1).min(n - 1), (2 * n / 3).min(n - 1)];

        for &idx in &indices {
            let fd = fd_scalar(&bk, &block, &x, param, &orig_data, &shape, idx, eps);

            let (ana, rel) = match ana_opt {
                None => {
                    // Param not reachable from loss — FD should be ≈ 0 too.
                    let rel = fd.abs() / (fd.abs().max(1e-8));
                    (f32::NAN, rel)
                }
                Some(v) => {
                    let a = v[idx];
                    let rel = (a - fd).abs() / (a.abs().max(fd.abs()).max(1e-8));
                    (a, rel)
                }
            };

            eprintln!(
                "  {:>20}  {:>5}  {:>12.4e}  {:>12.4e}  {:>9.2}%",
                name,
                idx,
                if ana.is_nan() { 0.0 } else { ana },
                fd,
                rel * 100.0
            );

            if ana.is_finite() && rel > max_rel_err {
                max_rel_err = rel;
                worst_param = name;
                worst_idx = idx;
                worst_rel = rel;
            }
        }
    }

    eprintln!(
        "  max rel_err = {:.2}%  worst: {}[{}] rel={:.2}%",
        max_rel_err * 100.0,
        worst_param,
        worst_idx,
        worst_rel * 100.0,
    );
    eprintln!("  NOTE: rel_err > 10% for a param → its backward op is suspect");

    // Hard-fail only on NaN/Inf: indicates a kernel crash or division by zero
    // in the backward pass.  The rel_err threshold is intentionally absent —
    // this is a diagnostic probe whose output drives the investigation report.
    assert!(
        max_rel_err.is_finite(),
        "max_rel_err is NaN or Inf — some backward kernel produced invalid output"
    );
}
