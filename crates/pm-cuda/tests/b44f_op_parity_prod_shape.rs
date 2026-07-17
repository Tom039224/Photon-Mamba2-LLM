//! B4.4f-followup: per-op numerical parity between CudaBackend and CandleBackend
//! at production shape.
//!
//! H6 (cuBLAS TF32) was disproven — `cublasSgemm_v2` does not use TF32 on
//! sm_120. The 13 % grad_norm delta between CudaBackend no-ckpt (5.162) and
//! Candle (4.585) therefore lives in a **non-matmul op**. This file isolates
//! each candidate op (H7–H11) at production shape and reports element-wise
//! error between CudaBackend (GPU) and CandleBackend (CPU, IEEE 754 reference).
//!
//! A 6th test exercises the backward of `mul` with saved tensors
//! (`h_saved_tensor_probe`): if the tape saves a tensor with wrong content,
//! `grad_a` from CudaBackend will diverge from CandleBackend even though the
//! forward is bit-identical.
//!
//! **None of these tests assert failure.** They are diagnostic probes. Run with
//! `-- --nocapture --test-threads=1` to see the output tables.
//!
//! Reference shape: 102 M PhotonMamba, batch=1, seq=512, d_model=768,
//! n_heads=12, d_head=64, d_state=128, d_conv=4, vocab=50257.

#![cfg(feature = "cuda")]

use pm_candle::CandleBackend;
use pm_core::{Ops, Param, Tensor};
use pm_cuda::CudaBackend;

// ---------------------------------------------------------------------------
// LCG random data generator — identical seed on both backends for parity.
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

/// Same as `lcg_vec` but returns `i64` in `[lo, hi)`.
fn lcg_vec_i64(seed: u64, n: usize, lo: i64, hi: i64) -> Vec<i64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let r = (state >> 41) % ((hi - lo) as u64);
            lo + r as i64
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Error statistics helpers
// ---------------------------------------------------------------------------

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn mean_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32
}

/// Max relative error, skipping elements where both a[i] and b[i] are near zero.
fn max_rel_err(a: &[f32], b: &[f32], zero_thresh: f32) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .filter(|(&x, &y)| x.abs() > zero_thresh || y.abs() > zero_thresh)
        .map(|(&x, &y)| {
            let denom = x.abs().max(y.abs());
            (x - y).abs() / denom
        })
        .fold(0f32, f32::max)
}

/// Number of elements where |a[i] - b[i]| > thresh.
fn n_exceed(a: &[f32], b: &[f32], thresh: f32) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(&x, &y)| (x - y).abs() > thresh)
        .count()
}

/// Print a standard error table row and return `(max_abs_err, max_rel_err)`.
fn print_stats(label: &str, cuda: &[f32], candle: &[f32]) -> (f32, f32, usize) {
    let mae = max_abs_err(cuda, candle);
    let mean = mean_abs_err(cuda, candle);
    let mre = max_rel_err(cuda, candle, 1e-6);
    let n = n_exceed(cuda, candle, 1e-3);
    let total = cuda.len();
    eprintln!(
        "  {label:<40} max_abs={mae:.3e}  mean_abs={mean:.3e}  \
         max_rel={mre:.3e}  n>1e-3={n}/{total}",
    );
    (mae, mre, n)
}

// ---------------------------------------------------------------------------
// H7: broadcast mul at production shape — exercises `broadcast_binary_op`
//     (D2H scalar loop H2D) vs Candle's on-device broadcast.
// ---------------------------------------------------------------------------

/// Production shapes for `dt · x` and similar Hadamard products inside
/// Mamba2 SSD:
///   Sub-case A: `[1,512,12,64] × [1,512,12,1]`  (broadcast over P=64 dim)
///   Sub-case B: `[1,1,12,1]   × [1,512,12,64]`  (broadcast over T and P)
///
/// If `broadcast_binary_op` has a stride error, max_rel > 0.01 here.
#[test]
fn h7_broadcast_mul_prod_shape() {
    let cuda_bk = CudaBackend::new(0).expect("CudaBackend::new(0)");
    let candle_bk = CandleBackend::new_cpu();

    eprintln!("B4.4f-H7 broadcast mul at production shape:");

    // Sub-case A: [1,512,12,64] × [1,512,12,1]
    {
        let (b, t, h, p) = (1usize, 512, 12, 64);
        let a_data = lcg_vec(701, b * t * h * p, 1.0, -0.5);
        let b_data = lcg_vec(702, b * t * h, 1.0, -0.5);

        let a_cuda = cuda_bk
            .from_slice_f32(&a_data, &[b, t, h, p])
            .expect("cuda a");
        let b_cuda = cuda_bk
            .from_slice_f32(&b_data, &[b, t, h, 1])
            .expect("cuda b");
        let y_cuda = cuda_bk.mul(&a_cuda, &b_cuda).expect("cuda mul A");
        let y_cuda_vec = cuda_bk.to_vec_f32(&y_cuda).expect("cuda to_vec A");

        let a_candle = candle_bk
            .from_slice_f32(&a_data, &[b, t, h, p])
            .expect("candle a");
        let b_candle = candle_bk
            .from_slice_f32(&b_data, &[b, t, h, 1])
            .expect("candle b");
        let y_candle = candle_bk.mul(&a_candle, &b_candle).expect("candle mul A");
        let y_candle_vec = candle_bk.to_vec_f32(&y_candle).expect("candle to_vec A");

        print_stats(
            &format!("A: [{b},{t},{h},{p}] × [{b},{t},{h},1]"),
            &y_cuda_vec,
            &y_candle_vec,
        );
        assert!(
            max_abs_err(&y_cuda_vec, &y_candle_vec).is_finite(),
            "H7-A max_abs_err NaN/Inf"
        );
    }

    // Sub-case B: [1,1,12,1] × [1,512,12,64]
    {
        let (b, t, h, p) = (1usize, 512, 12, 64);
        let a_data = lcg_vec(703, h, 1.0, -0.5);
        let b_data = lcg_vec(704, b * t * h * p, 1.0, -0.5);

        let a_cuda = cuda_bk
            .from_slice_f32(&a_data, &[1, 1, h, 1])
            .expect("cuda a B");
        let b_cuda = cuda_bk
            .from_slice_f32(&b_data, &[b, t, h, p])
            .expect("cuda b B");
        let y_cuda = cuda_bk.mul(&a_cuda, &b_cuda).expect("cuda mul B");
        let y_cuda_vec = cuda_bk.to_vec_f32(&y_cuda).expect("cuda to_vec B");

        let a_candle = candle_bk
            .from_slice_f32(&a_data, &[1, 1, h, 1])
            .expect("candle a B");
        let b_candle = candle_bk
            .from_slice_f32(&b_data, &[b, t, h, p])
            .expect("candle b B");
        let y_candle = candle_bk.mul(&a_candle, &b_candle).expect("candle mul B");
        let y_candle_vec = candle_bk.to_vec_f32(&y_candle).expect("candle to_vec B");

        print_stats(
            &format!("B: [1,1,{h},1] × [{b},{t},{h},{p}]"),
            &y_cuda_vec,
            &y_candle_vec,
        );
        assert!(
            max_abs_err(&y_cuda_vec, &y_candle_vec).is_finite(),
            "H7-B max_abs_err NaN/Inf"
        );
    }
}

// ---------------------------------------------------------------------------
// H8: RMSNorm at production shape — PTX reduction kernel vs Candle cuBLAS norm
// ---------------------------------------------------------------------------

/// `x=[1,512,768]` normalised over last dim, weight `[768]`.
/// Each RMSNorm in the 20 Mamba2 blocks sees this shape.
/// If the PTX kernel has a fp reduction bias vs Candle's two-pass reduce,
/// max_rel_err will be >1e-3 here.
#[test]
fn h8_rmsnorm_prod_shape() {
    let cuda_bk = CudaBackend::new(0).expect("CudaBackend::new(0)");
    let candle_bk = CandleBackend::new_cpu();

    let (b, t, d) = (1usize, 512, 768);
    let eps = 1e-5_f32;

    let x_data = lcg_vec(801, b * t * d, 1.0, -0.5);
    // Weight initialised near 1 (typical RMSNorm init).
    let w_data = lcg_vec(802, d, 0.1, 0.95);

    let x_cuda = cuda_bk.from_slice_f32(&x_data, &[b, t, d]).expect("cuda x");
    let w_cuda = cuda_bk.from_slice_f32(&w_data, &[d]).expect("cuda w");
    let y_cuda = cuda_bk
        .rmsnorm(&x_cuda, &w_cuda, eps)
        .expect("cuda rmsnorm");
    let y_cuda_vec = cuda_bk.to_vec_f32(&y_cuda).expect("cuda to_vec");

    let x_candle = candle_bk
        .from_slice_f32(&x_data, &[b, t, d])
        .expect("candle x");
    let w_candle = candle_bk.from_slice_f32(&w_data, &[d]).expect("candle w");
    let y_candle = candle_bk
        .rmsnorm(&x_candle, &w_candle, eps)
        .expect("candle rmsnorm");
    let y_candle_vec = candle_bk.to_vec_f32(&y_candle).expect("candle to_vec");

    eprintln!("B4.4f-H8 RMSNorm (B={b},T={t},D={d},eps={eps:.0e}):");
    print_stats(
        &format!("rmsnorm [{b},{t},{d}] w [{d}]"),
        &y_cuda_vec,
        &y_candle_vec,
    );

    assert!(
        max_abs_err(&y_cuda_vec, &y_candle_vec).is_finite(),
        "H8 max_abs_err NaN/Inf"
    );
}

// ---------------------------------------------------------------------------
// H9: log_softmax at production vocab size — PTX lastdim kernel vs Candle
// ---------------------------------------------------------------------------

/// `x=[1,512,50257]` at `dim=2`. Exercises the `log_softmax_lastdim_f32`
/// kernel at full vocab. If the PTX kernel uses a naïve one-pass max+sum
/// that differs from Candle's numerically-stable two-pass form, max_rel
/// will be large for tokens near the max logit.
#[test]
fn h9_log_softmax_prod_shape() {
    let cuda_bk = CudaBackend::new(0).expect("CudaBackend::new(0)");
    let candle_bk = CandleBackend::new_cpu();

    let (b, t, v) = (1usize, 512, 50257);
    // Logits in (-3, 3) typical range.
    let x_data = lcg_vec(901, b * t * v, 6.0, -3.0);

    let x_cuda = cuda_bk.from_slice_f32(&x_data, &[b, t, v]).expect("cuda x");
    let y_cuda = cuda_bk.log_softmax(&x_cuda, 2).expect("cuda log_softmax");
    let y_cuda_vec = cuda_bk.to_vec_f32(&y_cuda).expect("cuda to_vec");

    let x_candle = candle_bk
        .from_slice_f32(&x_data, &[b, t, v])
        .expect("candle x");
    let y_candle = candle_bk
        .log_softmax(&x_candle, 2)
        .expect("candle log_softmax");
    let y_candle_vec = candle_bk.to_vec_f32(&y_candle).expect("candle to_vec");

    eprintln!("B4.4f-H9 log_softmax (B={b},T={t},V={v},dim=2):");
    print_stats(
        &format!("log_softmax [{b},{t},{v}] dim=2"),
        &y_cuda_vec,
        &y_candle_vec,
    );

    assert!(
        max_abs_err(&y_cuda_vec, &y_candle_vec).is_finite(),
        "H9 max_abs_err NaN/Inf"
    );
}

// ---------------------------------------------------------------------------
// H10: gather / embedding at production shape
//   Sub-case A: `Ops::embedding` — token embed table [50257,768], ids [1,512]
//   Sub-case B: `Ops::gather`    — NLL loss gather, logprobs [1,512,50257],
//                                   ids [1,512,1] i64, dim=2
// ---------------------------------------------------------------------------

/// The model uses `Ops::embedding` for token lookup and `Ops::gather` for
/// NLL loss. H10 exercises both at production shape.
///
/// Note: the task specification describes this as "`Ops::gather` (embedding
/// lookup)" but the model code uses `Ops::embedding` for table lookup and
/// `Ops::gather` only in the cross-entropy loss path. Both are tested here.
#[test]
fn h10_gather_prod_shape() {
    let cuda_bk = CudaBackend::new(0).expect("CudaBackend::new(0)");
    let candle_bk = CandleBackend::new_cpu();

    let (v, d_embed) = (50257usize, 768usize);
    let (b, t) = (1usize, 512usize);

    eprintln!("B4.4f-H10 gather/embedding at production shape:");

    // Sub-case A: Ops::embedding — table [V,D], indices [B,T] i64
    {
        let table_data = lcg_vec(1001, v * d_embed, 0.1, -0.05);
        let idx_data = lcg_vec_i64(1002, b * t, 0, v as i64);

        let table_cuda = cuda_bk
            .from_slice_f32(&table_data, &[v, d_embed])
            .expect("cuda table");
        let idx_cuda = cuda_bk
            .from_slice_i64(&idx_data, &[b, t])
            .expect("cuda idx");
        let y_cuda = cuda_bk
            .embedding(&table_cuda, &idx_cuda)
            .expect("cuda embedding");
        let y_cuda_vec = cuda_bk.to_vec_f32(&y_cuda).expect("cuda to_vec emb");

        let table_candle = candle_bk
            .from_slice_f32(&table_data, &[v, d_embed])
            .expect("candle table");
        let idx_candle = candle_bk
            .from_slice_i64(&idx_data, &[b, t])
            .expect("candle idx");
        let y_candle = candle_bk
            .embedding(&table_candle, &idx_candle)
            .expect("candle embedding");
        let y_candle_vec = candle_bk.to_vec_f32(&y_candle).expect("candle to_vec emb");

        print_stats(
            &format!("A: embedding [{v},{d_embed}] ids [{b},{t}]"),
            &y_cuda_vec,
            &y_candle_vec,
        );
        assert!(
            max_abs_err(&y_cuda_vec, &y_candle_vec).is_finite(),
            "H10-A max_abs_err NaN/Inf"
        );
    }

    // Sub-case B: Ops::gather — NLL loss pattern
    //   logprobs [B,T,V], ids [B,T,1] i64, dim=2  → output [B,T,1]
    {
        // Only use a small vocab slice to keep memory manageable for the [B,T,V] tensor.
        let v_sub = 50257usize; // full vocab but only B=1,T=512
        let logp_data = lcg_vec(1003, b * t * v_sub, 2.0, -10.0); // logprobs ~ (−12,−8)
        let idx_data = lcg_vec_i64(1004, b * t, 0, v_sub as i64);

        let logp_cuda = cuda_bk
            .from_slice_f32(&logp_data, &[b, t, v_sub])
            .expect("cuda logp");
        // Reshape idx to [B, T, 1] as required by gather(dim=2)
        let idx_3d_data: Vec<i64> = idx_data.clone();
        let idx_cuda = cuda_bk
            .from_slice_i64(&idx_3d_data, &[b, t, 1])
            .expect("cuda idx gather");
        let y_cuda = cuda_bk
            .gather(&logp_cuda, &idx_cuda, 2)
            .expect("cuda gather");
        let y_cuda_vec = cuda_bk.to_vec_f32(&y_cuda).expect("cuda to_vec gather");

        let logp_candle = candle_bk
            .from_slice_f32(&logp_data, &[b, t, v_sub])
            .expect("candle logp");
        let idx_candle = candle_bk
            .from_slice_i64(&idx_3d_data, &[b, t, 1])
            .expect("candle idx gather");
        let y_candle = candle_bk
            .gather(&logp_candle, &idx_candle, 2)
            .expect("candle gather");
        let y_candle_vec = candle_bk
            .to_vec_f32(&y_candle)
            .expect("candle to_vec gather");

        print_stats(
            &format!("B: gather [{b},{t},{v_sub}] ids [{b},{t},1] dim=2"),
            &y_cuda_vec,
            &y_candle_vec,
        );
        assert!(
            max_abs_err(&y_cuda_vec, &y_candle_vec).is_finite(),
            "H10-B max_abs_err NaN/Inf",
        );
    }
}

// ---------------------------------------------------------------------------
// H11: depthwise conv1d at production shape — CPU im2col+GEMM path in CudaBackend
// ---------------------------------------------------------------------------

/// Input `[1, 1024, 512]`, weight `[1024, 1, 4]`, groups=1024 (depthwise).
/// This is the `conv1d` op inside each Mamba2 `xbc_dim` projection at
/// production seq_len=512 and `d_conv=4`. CudaBackend's `conv1d_f32`
/// performs D2H im2col → CPU SGEMM → H2D. T4 showed 2.87% error at tiny
/// scale; this test checks whether the error grows at production scale.
#[test]
fn h11_conv1d_depthwise_prod_shape() {
    let cuda_bk = CudaBackend::new(0).expect("CudaBackend::new(0)");
    let candle_bk = CandleBackend::new_cpu();

    // input [B=1, C=1024, T=512], weight [C_out=1024, 1, K=4], groups=1024
    let (b, c, t_in) = (1usize, 1024, 512);
    let k = 4usize;
    let padding = k - 1; // causal padding to keep T_out = T_in
    let stride = 1;
    let groups = c;

    let x_data = lcg_vec(1101, b * c * t_in, 1.0, -0.5);
    let w_data = lcg_vec(1102, c * k, 0.1, -0.05);

    let x_cuda = cuda_bk
        .from_slice_f32(&x_data, &[b, c, t_in])
        .expect("cuda x");
    let w_cuda = cuda_bk.from_slice_f32(&w_data, &[c, 1, k]).expect("cuda w");
    let y_cuda = cuda_bk
        .conv1d(&x_cuda, &w_cuda, None, stride, padding, groups)
        .expect("cuda conv1d");
    // Output shape [B, C, T_out]; truncate to T_in for causal comparison.
    let t_out_cuda = y_cuda.shape()[2];
    let y_cuda_full = cuda_bk.to_vec_f32(&y_cuda).expect("cuda to_vec");
    // Keep only the first T_in positions (causal window).
    let y_cuda_vec: Vec<f32> = y_cuda_full
        .chunks(t_out_cuda)
        .flat_map(|ch| ch.iter().take(t_in).copied())
        .collect();

    let x_candle = candle_bk
        .from_slice_f32(&x_data, &[b, c, t_in])
        .expect("candle x");
    let w_candle = candle_bk
        .from_slice_f32(&w_data, &[c, 1, k])
        .expect("candle w");
    let y_candle = candle_bk
        .conv1d(&x_candle, &w_candle, None, stride, padding, groups)
        .expect("candle conv1d");
    let t_out_candle = y_candle.shape()[2];
    let y_candle_full = candle_bk.to_vec_f32(&y_candle).expect("candle to_vec");
    let y_candle_vec: Vec<f32> = y_candle_full
        .chunks(t_out_candle)
        .flat_map(|ch| ch.iter().take(t_in).copied())
        .collect();

    eprintln!(
        "B4.4f-H11 depthwise conv1d (B={b},C={c},T={t_in},K={k},groups={groups},pad={padding}):"
    );
    eprintln!(
        "  t_out_cuda={t_out_cuda}  t_out_candle={t_out_candle}  \
         comparing first {t_in} positions per channel"
    );
    print_stats(
        &format!("conv1d [{b},{c},{t_in}] w [{c},1,{k}] g={groups}"),
        &y_cuda_vec,
        &y_candle_vec,
    );

    assert!(
        max_abs_err(&y_cuda_vec, &y_candle_vec).is_finite(),
        "H11 max_abs_err NaN/Inf"
    );
}

// ---------------------------------------------------------------------------
// H_saved_tensor_probe: backward through `mul` — exercises the tape's
// saved-tensor path at production shape.
//
// If `TapeOp::Mul { lhs_val, rhs_val }` stores a tensor with wrong content
// (e.g. stale data after a fused kernel overwrote the buffer in-place), the
// CudaBackend gradient will diverge from CandleBackend's Autograd result.
//
// Sub-case A: same-shape — `[1,512,12,64] × [1,512,12,64]` (no broadcast)
// Sub-case B: broadcast   — `[1,512,12,1] × [1,512,12,64]` (broadcast P dim)
// ---------------------------------------------------------------------------

/// Run `loss = sum_all(mul(a, b))` on both backends, call backward, and
/// compare `grad_a` and `grad_b`.
///
/// Expected (analytically):
///   grad_a[i] = b[i]   (or sum-reduced to a's shape for broadcast)
///   grad_b[i] = a[i]   (or sum-reduced to b's shape for broadcast)
///
/// Any deviation > 1 % relative is the smoking gun for the "saved tensor is
/// subtly wrong" hypothesis and pins the root cause of the 13 % grad_norm delta.
#[test]
fn h_saved_tensor_probe() {
    eprintln!("B4.4f-H_saved_tensor_probe: mul backward — saved-tensor parity:");

    let (b, t, h, p) = (1usize, 512, 12, 64);

    // Sub-case A: same-shape [1,512,12,64] × [1,512,12,64]
    {
        let a_data = lcg_vec(2001, b * t * h * p, 1.0, -0.5);
        let b_data = lcg_vec(2002, b * t * h * p, 1.0, -0.5);

        // CudaBackend backward.
        let cuda_bk = CudaBackend::new(0).expect("cuda");
        let a_cuda = cuda_bk
            .param_from_slice_f32(&a_data, &[b, t, h, p])
            .expect("cuda param a");
        let b_cuda = cuda_bk
            .param_from_slice_f32(&b_data, &[b, t, h, p])
            .expect("cuda param b");
        let prod_cuda = cuda_bk
            .mul(a_cuda.as_tensor(), b_cuda.as_tensor())
            .expect("cuda mul");
        let loss_cuda = cuda_bk.sum_all(&prod_cuda).expect("cuda sum_all");
        let store_cuda = cuda_bk.backward(&loss_cuda).expect("cuda backward");
        let ga_cuda = cuda_bk
            .gradient(&store_cuda, &a_cuda)
            .expect("cuda grad_a call")
            .expect("cuda grad_a Some");
        let gb_cuda = cuda_bk
            .gradient(&store_cuda, &b_cuda)
            .expect("cuda grad_b call")
            .expect("cuda grad_b Some");
        let ga_cuda_vec = cuda_bk.to_vec_f32(&ga_cuda).expect("ga cuda vec");
        let gb_cuda_vec = cuda_bk.to_vec_f32(&gb_cuda).expect("gb cuda vec");

        // CandleBackend backward.
        let candle_bk = CandleBackend::new_cpu();
        let a_candle = candle_bk
            .param_from_slice_f32(&a_data, &[b, t, h, p])
            .expect("candle param a");
        let b_candle = candle_bk
            .param_from_slice_f32(&b_data, &[b, t, h, p])
            .expect("candle param b");
        let prod_candle = candle_bk
            .mul(a_candle.as_tensor(), b_candle.as_tensor())
            .expect("candle mul");
        let loss_candle = candle_bk.sum_all(&prod_candle).expect("candle sum_all");
        let store_candle = candle_bk.backward(&loss_candle).expect("candle backward");
        let ga_candle = candle_bk
            .gradient(&store_candle, &a_candle)
            .expect("candle grad_a call")
            .expect("candle grad_a Some");
        let gb_candle = candle_bk
            .gradient(&store_candle, &b_candle)
            .expect("candle grad_b call")
            .expect("candle grad_b Some");
        let ga_candle_vec = candle_bk.to_vec_f32(&ga_candle).expect("ga candle vec");
        let gb_candle_vec = candle_bk.to_vec_f32(&gb_candle).expect("gb candle vec");

        eprintln!("  Sub-case A (same-shape [{b},{t},{h},{p}] × [{b},{t},{h},{p}]):");
        let (mae_a, mre_a, n_a) = print_stats(
            "    grad_a  (should equal b_data)",
            &ga_cuda_vec,
            &ga_candle_vec,
        );
        let (mae_b, mre_b, n_b) = print_stats(
            "    grad_b  (should equal a_data)",
            &gb_cuda_vec,
            &gb_candle_vec,
        );

        // Cross-check: grad_a from CUDA should ≈ b_data (the reference).
        let ref_err_a = max_abs_err(&ga_cuda_vec, &b_data);
        let ref_err_b = max_abs_err(&gb_cuda_vec, &a_data);
        eprintln!("    CUDA grad_a vs b_data (reference): max_abs={ref_err_a:.3e}");
        eprintln!("    CUDA grad_b vs a_data (reference): max_abs={ref_err_b:.3e}");

        if mre_a > 0.01 || mre_b > 0.01 {
            eprintln!(
                "  *** A CULPRIT CANDIDATE: same-shape mul backward max_rel={:.3e}/{:.3e}",
                mre_a, mre_b
            );
        }

        assert!(mae_a.is_finite() && mae_b.is_finite(), "H_saved A NaN/Inf");
        let _ = (mae_a, mae_b, mre_a, mre_b, n_a, n_b); // suppress unused warnings
    }

    // Sub-case B: broadcast [1,512,12,1] × [1,512,12,64]
    {
        let a_data = lcg_vec(2003, b * t * h, 1.0, -0.5);
        let b_data = lcg_vec(2004, b * t * h * p, 1.0, -0.5);

        // CudaBackend backward.
        let cuda_bk = CudaBackend::new(0).expect("cuda B");
        let a_cuda = cuda_bk
            .param_from_slice_f32(&a_data, &[b, t, h, 1])
            .expect("cuda param a B");
        let b_cuda = cuda_bk
            .param_from_slice_f32(&b_data, &[b, t, h, p])
            .expect("cuda param b B");
        let prod_cuda = cuda_bk
            .mul(a_cuda.as_tensor(), b_cuda.as_tensor())
            .expect("cuda mul B");
        let loss_cuda = cuda_bk.sum_all(&prod_cuda).expect("cuda sum_all B");
        let store_cuda = cuda_bk.backward(&loss_cuda).expect("cuda backward B");
        let ga_cuda = cuda_bk
            .gradient(&store_cuda, &a_cuda)
            .expect("cuda grad_a B call")
            .expect("cuda grad_a B Some");
        let gb_cuda = cuda_bk
            .gradient(&store_cuda, &b_cuda)
            .expect("cuda grad_b B call")
            .expect("cuda grad_b B Some");
        let ga_cuda_vec = cuda_bk.to_vec_f32(&ga_cuda).expect("ga cuda vec B");
        let gb_cuda_vec = cuda_bk.to_vec_f32(&gb_cuda).expect("gb cuda vec B");

        // CandleBackend backward.
        let candle_bk = CandleBackend::new_cpu();
        let a_candle = candle_bk
            .param_from_slice_f32(&a_data, &[b, t, h, 1])
            .expect("candle param a B");
        let b_candle = candle_bk
            .param_from_slice_f32(&b_data, &[b, t, h, p])
            .expect("candle param b B");
        let prod_candle = candle_bk
            .mul(a_candle.as_tensor(), b_candle.as_tensor())
            .expect("candle mul B");
        let loss_candle = candle_bk.sum_all(&prod_candle).expect("candle sum_all B");
        let store_candle = candle_bk.backward(&loss_candle).expect("candle backward B");
        let ga_candle = candle_bk
            .gradient(&store_candle, &a_candle)
            .expect("candle grad_a B call")
            .expect("candle grad_a B Some");
        let gb_candle = candle_bk
            .gradient(&store_candle, &b_candle)
            .expect("candle grad_b B call")
            .expect("candle grad_b B Some");
        let ga_candle_vec = candle_bk.to_vec_f32(&ga_candle).expect("ga candle vec B");
        let gb_candle_vec = candle_bk.to_vec_f32(&gb_candle).expect("gb candle vec B");

        eprintln!("  Sub-case B (broadcast [{b},{t},{h},1] × [{b},{t},{h},{p}]):");
        let (mae_a, mre_a, n_a) = print_stats(
            "    grad_a  (should be sum-reduced b_data)",
            &ga_cuda_vec,
            &ga_candle_vec,
        );
        let (mae_b, mre_b, n_b) = print_stats(
            "    grad_b  (should equal a_data broadcast)",
            &gb_cuda_vec,
            &gb_candle_vec,
        );

        if mre_a > 0.01 || mre_b > 0.01 {
            eprintln!(
                "  *** A CULPRIT CANDIDATE: broadcast mul backward max_rel={:.3e}/{:.3e}",
                mre_a, mre_b
            );
        }

        assert!(mae_a.is_finite() && mae_b.is_finite(), "H_saved B NaN/Inf");
        let _ = (mae_a, mae_b, mre_a, mre_b, n_a, n_b);
    }
}
