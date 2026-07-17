//! Phase A2: bf16 mixed-precision forward compute — dtype-mechanics
//! sanity checks that don't require a GPU (memory-efficiency plan,
//! `docs/perf-log.md` 2026-07-03 entry,
//! `plans/fancy-enchanting-lamport.md` Phase A2).
//!
//! **The model-level numerical-parity and grad-flow tests for this
//! feature live in `bf16_mixed_precision_cuda.rs`, gated behind
//! `--features cuda`.** That split is not stylistic: Candle 0.11's CPU
//! backend explicitly restricts `matmul` to `F16 | F32 | F64`
//! (`cpu_backend/mod.rs::MatMul::f`, `Err(UnsupportedDTypeForOp(BF16,
//! "matmul"))`) — bf16 matmul only exists on the CUDA backend, via
//! cuBLAS (`cuda_backend/mod.rs`, `gemm_strided_batched_bf16`). Since
//! `Mamba2Block::forward`'s `in_proj`/`out_proj` matmuls (and
//! `Linear`/`lm_head`) run in the ambient compute dtype, any test that
//! builds a real block/model and runs it with `compute_dtype = BF16`
//! needs the CUDA backend. This matches the actual deployment target
//! anyway (`pm train --backend candle` with `device = "cuda"` on the
//! RTX 5070) — the memory problem this feature solves only exists on
//! GPU, CPU RAM was never the bottleneck.
//!
//! This file only exercises `Ops::to_dtype` itself, which is dtype-cast
//! only (no matmul), and does work on CPU.

use pm_candle::CandleBackend;
use pm_core::{Dtype, Ops, Tensor};

/// fp16/bf16 numerical-parity budget (CLAUDE.md invariant #3).
const REL_TOL: f32 = 1e-2;

fn max_abs_rel_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in comparison");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let denom = x.abs().max(y.abs()).max(1e-6);
            (x - y).abs() / denom
        })
        .fold(0f32, f32::max)
}

#[test]
fn to_dtype_same_dtype_is_value_preserving_noop() {
    let bk = CandleBackend::new_cpu();
    let x = bk.from_slice_f32(&[1.0, -2.5, 3.75], &[3]).unwrap();
    let y = bk.to_dtype(&x, Dtype::F32).unwrap();
    assert_eq!(y.dtype(), Dtype::F32);
    assert_eq!(bk.to_vec_f32(&x).unwrap(), bk.to_vec_f32(&y).unwrap());
}

#[test]
fn to_dtype_f32_bf16_roundtrip_within_bf16_precision() {
    let bk = CandleBackend::new_cpu();
    let data = vec![1.0f32, -2.5, 3.75, 0.1, 100.25, -0.003];
    let x = bk.from_slice_f32(&data, &[data.len()]).unwrap();
    let bf16 = bk.to_dtype(&x, Dtype::BF16).unwrap();
    assert_eq!(bf16.dtype(), Dtype::BF16);
    assert_eq!(bf16.shape(), x.shape());
    let back = bk.to_dtype(&bf16, Dtype::F32).unwrap();
    let v = bk.to_vec_f32(&back).unwrap();
    let err = max_abs_rel_err(&data, &v);
    assert!(
        err < REL_TOL,
        "f32->bf16->f32 roundtrip rel err {err:.4e} exceeds {REL_TOL:e}"
    );
}
