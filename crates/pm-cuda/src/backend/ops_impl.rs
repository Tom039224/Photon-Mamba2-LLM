//! `impl pm_core::Ops for CudaBackend` — B4 scaffolding.
//!
//! B4.2a lands:
//!   - Basic construction: zeros, ones, from_slice_f32, from_slice_i64, to_vec_f32
//!   - Elementwise: add, sub, mul, neg
//!   - matmul (cuBLAS)
//!   - param skeleton: param_from_slice_f32 / param_zeros / param_ones
//!
//! B4.3a adds:
//!   - Tape recording for 13 forward ops
//!   - `backward()` with VJP for all recorded ops
//!   - `gradient`, `set_gradient`, `param_from_tensor`, `merge_grad_stores`,
//!     `sgd_step`, `assign` — all previously `unimplemented!("B4.3: …")`

use std::collections::HashMap;
use std::sync::Arc;

use pm_core::{Dtype, Ops, Param, Tensor};

use super::grad::CudaGradStore;
use super::param::CudaParam;
use super::tape::{grad_enabled, no_grad, NodeId, SsdScanData, Tape, TapeOp};
use super::{CudaBackend, CudaTensor};
use crate::CudaError;

// ---- Helper: broadcast-expand batch dims ----------------------------------

/// Expand `src` from `src_batch_dims` shape to `dst_batch_dims` shape (host side).
///
/// Both `src` and `dst` are treated as flat arrays of `(batch, slice_len)` elements.
/// Size-1 dimensions in `src_batch_dims` broadcast against larger dims in `dst_batch_dims`.
/// Other dimensions must match.
fn broadcast_expand_batch(
    src: &[f32],
    src_batch: &[usize],
    dst_batch: &[usize],
    slice_len: usize,
    dst: &mut [f32],
) -> Result<(), CudaError> {
    if src_batch.len() != dst_batch.len() {
        return Err(CudaError::Shape(format!(
            "broadcast_expand_batch: rank mismatch ({} vs {})",
            src_batch.len(),
            dst_batch.len()
        )));
    }
    let total_dst: usize = dst_batch.iter().product();
    debug_assert_eq!(dst.len(), total_dst * slice_len);

    for dst_batch_idx in 0..total_dst {
        // Convert dst_batch_idx into multi-index, then map to src flat index.
        let mut remaining = dst_batch_idx;
        let mut src_batch_idx = 0usize;
        let mut src_stride = 1usize;
        // Process dims in reverse (innermost first) to compute src_batch_idx.
        for dim_i in (0..dst_batch.len()).rev() {
            let dst_d = dst_batch[dim_i];
            let src_d = src_batch[dim_i];
            let coord = remaining % dst_d;
            remaining /= dst_d;
            // Broadcast: if src_d == 1, use coord 0; otherwise use coord.
            let src_coord = if src_d == 1 { 0 } else { coord };
            src_batch_idx += src_coord * src_stride;
            src_stride *= src_d;
        }
        let dst_off = dst_batch_idx * slice_len;
        let src_off = src_batch_idx * slice_len;
        dst[dst_off..dst_off + slice_len].copy_from_slice(&src[src_off..src_off + slice_len]);
    }
    Ok(())
}

// ---- Tape recording helper -----------------------------------------------

/// Push a `TapeOp` onto the shared tape and return the new `NodeId`.
///
/// Acquires the tape lock.  Panics if the lock is poisoned — poisoning
/// is only possible if a prior call to `backward` panicked, which is
/// itself a programming error; aborting the current operation via panic
/// is the correct response in that situation.
///
/// # SAFETY
/// Lock poisoning only occurs after a panic inside a `MutexGuard` scope.
/// The only `MutexGuard` scope in this crate that can panic is
/// `binary_op`'s shape-mismatch `assert_eq!`.  Once a panic has occurred
/// the process is unwinding anyway; panicking here as well is safe.
fn tape_push(tape: &std::sync::Mutex<Tape>, op: TapeOp) -> NodeId {
    tape.lock()
        .expect("autograd tape lock poisoned — prior panic in backward?")
        .push(op)
}

// ---- Leaf registration helper --------------------------------------------

/// Register `tensor` as a fresh leaf on the tape and return a new
/// `CudaTensor` stamped with the resulting `NodeId` *and* a [`LeafOrigin`]
/// back-reference (`origin`).
///
/// Used by `param_from_slice_f32`, `param_zeros`, `param_ones`,
/// `param_from_tensor`, `sgd_step`, and `assign` to ensure that every
/// parameter has a current leaf entry after creation or update so that
/// the next `backward` can route gradients correctly.
///
/// The `origin` stamp (not just `node_id`) is what makes this survive
/// `pm_core::checkpoint_backward`'s multiple `Ops::backward` calls per
/// training step: any later clone of the returned tensor (e.g. a
/// checkpoint boundary threaded into the next segment's `saved_input`, or
/// a `CudaParam`'s value read after an intervening `backward()` cleared
/// the tape) re-resolves to a fresh, current-generation `Leaf` entry via
/// `LeafOrigin::resolve` instead of dangling. See `tape.rs`'s docs.
fn register_leaf(
    tape: &std::sync::Arc<std::sync::Mutex<Tape>>,
    tensor: CudaTensor,
    param_id: super::param::ParamId,
) -> CudaTensor {
    let nid = {
        let mut t = tape
            .lock()
            .expect("autograd tape lock poisoned — prior panic in backward?");
        let nid = t.push(TapeOp::Leaf { param_id });
        t.leaf_by_param.insert(param_id, nid);
        nid
    };
    tensor
        .with_node_id(nid)
        .with_origin(std::sync::Arc::new(super::tape::LeafOrigin {
            tape: tape.clone(),
            param_id,
        }))
}

// ---- Accumulate gradient helper ------------------------------------------

/// Accumulate `contrib` into `grads[nid]`.
///
/// If `nid` already has an entry the gradients are added element-wise
/// via `add_inner` directly, bypassing the tape without requiring
/// `no_grad`.  If there is no entry the contrib is inserted directly.
fn accumulate(
    bk: &CudaBackend,
    grads: &mut HashMap<NodeId, CudaTensor>,
    nid: NodeId,
    contrib: CudaTensor,
) -> Result<(), CudaError> {
    match grads.remove(&nid) {
        None => {
            grads.insert(nid, contrib);
        }
        Some(existing) => {
            let sum = no_grad(|| bk.add_inner(&existing, &contrib))?;
            grads.insert(nid, sum);
        }
    }
    Ok(())
}

impl Ops for CudaBackend {
    type Tensor = CudaTensor;
    type Error = CudaError;
    type Param = CudaParam;
    type GradStore = CudaGradStore;

    // ---- Construction -------------------------------------------------------

    fn zeros(&self, shape: &[usize], dtype: Dtype) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("zeros", &self.stream);
        match dtype {
            Dtype::F32 => self.zeros_f32(shape),
            _other => Err(CudaError::Unsupported(
                "zeros: only Dtype::F32 is supported in B4; bf16/f16/i64 land later",
            )),
        }
    }

    fn ones(&self, shape: &[usize], dtype: Dtype) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("ones", &self.stream);
        match dtype {
            Dtype::F32 => {
                let n = shape.iter().product::<usize>();
                let host = vec![1.0f32; n];
                let storage = self.stream.clone_htod(&host)?;
                Ok(CudaTensor::new(Arc::new(storage), shape.to_vec()))
            }
            _ => Err(CudaError::Unsupported(
                "ones: only Dtype::F32 supported in B4",
            )),
        }
    }

    fn from_slice_f32(&self, data: &[f32], shape: &[usize]) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("from_slice_f32", &self.stream);
        self.from_slice_f32_inner(data, shape)
    }

    fn from_slice_i64(&self, data: &[i64], shape: &[usize]) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("from_slice_i64", &self.stream);
        let n = shape.iter().product::<usize>();
        if data.len() != n {
            return Err(CudaError::Shape(format!(
                "from_slice_i64: data len {} != shape product {n}",
                data.len()
            )));
        }
        let storage = self.stream.clone_htod(data)?;
        Ok(CudaTensor::new_i64(Arc::new(storage), shape.to_vec()))
    }

    fn to_vec_f32(&self, x: &CudaTensor) -> Result<Vec<f32>, CudaError> {
        let _t = crate::profiler::timer("to_vec_f32", &self.stream);
        self.to_vec_f32_inner(x)
    }

    fn to_vec_i64(&self, x: &CudaTensor) -> Result<Vec<i64>, CudaError> {
        let _t = crate::profiler::timer("to_vec_i64", &self.stream);
        Ok(self.stream.clone_dtoh(x.storage_i64().as_ref())?)
    }

    // ---- Dtype conversion -----------------------------------------------------

    /// `CudaStorage` only has `F32`/`I64` variants today (no bf16/f16
    /// buffer or cast kernel exists yet), so this only supports the
    /// identity cast. That is exactly the case every current caller
    /// hits: `pm-core` model code (e.g. `Mamba2Block::forward`) calls
    /// `to_dtype` unconditionally at fp32-island boundaries, and since
    /// `pm-cuda` never produces non-F32 float tensors, `t.dtype()` is
    /// always `dtype` already — this is a cheap `Arc` clone that
    /// preserves `node_id` so it is transparent to the autograd tape,
    /// mirroring `CandleBackend::to_dtype`'s same-dtype fast path. A
    /// genuine bf16 cast (kernel + `CudaStorage::BF16`) is out of scope
    /// for the bf16 mixed-precision plan (Phase A2, Candle-only for
    /// now); see `docs/perf-log.md` 2026-07-03.
    fn to_dtype(&self, t: &CudaTensor, dtype: Dtype) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("to_dtype", &self.stream);
        if t.dtype() == dtype {
            return Ok(t.clone());
        }
        Err(CudaError::Unsupported(
            "to_dtype: pm-cuda only stores Dtype::F32 (and I64 for indices) today; \
             the bf16 mixed-precision compute path is Candle-only for now",
        ))
    }

    // ---- Element-wise -------------------------------------------------------

    fn add(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("add", &self.stream);
        let lhs_shape = a.shape().to_vec();
        let rhs_shape = b.shape().to_vec();
        // H1: same-shape fast path — use PTX kernel directly, skip host round-trip.
        let out = if a.shape() == b.shape() {
            self.add_inner(a, b)?
        } else {
            self.broadcast_binary_op(a, b, |x, y| x + y)?
        };
        if grad_enabled() && (a.node_id().is_some() || b.node_id().is_some()) {
            let nid = tape_push(
                &self.tape,
                TapeOp::Add {
                    lhs: a.node_id(),
                    rhs: b.node_id(),
                    lhs_shape,
                    rhs_shape,
                },
            );
            Ok(out.with_node_id(nid))
        } else {
            Ok(out)
        }
    }

    fn sub(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("sub", &self.stream);
        let lhs_shape = a.shape().to_vec();
        let rhs_shape = b.shape().to_vec();
        // H1: same-shape fast path — use PTX kernel directly, skip host round-trip.
        let out = if a.shape() == b.shape() {
            self.sub_inner(a, b)?
        } else {
            self.broadcast_binary_op(a, b, |x, y| x - y)?
        };
        if grad_enabled() && (a.node_id().is_some() || b.node_id().is_some()) {
            let nid = tape_push(
                &self.tape,
                TapeOp::Sub {
                    lhs: a.node_id(),
                    rhs: b.node_id(),
                    lhs_shape,
                    rhs_shape,
                },
            );
            Ok(out.with_node_id(nid))
        } else {
            Ok(out)
        }
    }

    fn mul(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("mul", &self.stream);
        // H1: same-shape fast path — use PTX kernel directly, skip host round-trip.
        // B'.3: the broadcast path also stays on-device now (`broadcast_mul_dev`)
        // — see its doc comment / docs/perf-log.md for the host round-trip this
        // replaced (measured ~1.3 ms/call on the Mamba2Block `x_dt`/`d_term`/
        // `a_bth` broadcast multiplies).
        let out = if a.shape() == b.shape() {
            self.mul_inner(a, b)?
        } else {
            self.broadcast_mul_dev(a, b)?
        };
        if grad_enabled() && (a.node_id().is_some() || b.node_id().is_some()) {
            let nid = tape_push(
                &self.tape,
                TapeOp::Mul {
                    lhs: a.node_id(),
                    rhs: b.node_id(),
                    lhs_val: a.clone(),
                    rhs_val: b.clone(),
                },
            );
            Ok(out.with_node_id(nid))
        } else {
            Ok(out)
        }
    }

    fn neg(&self, a: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("neg", &self.stream);
        let out = self.neg_inner(a)?;
        if grad_enabled() && a.node_id().is_some() {
            let nid = tape_push(&self.tape, TapeOp::Neg { input: a.node_id() });
            Ok(out.with_node_id(nid))
        } else {
            Ok(out)
        }
    }

    // ---- Linear algebra -----------------------------------------------------

    /// Batched matrix multiply. Handles 2D and ND (batched) cases via cuBLAS.
    ///
    /// Broadcasting rule: leading dims (all except last 2) follow numpy/torch
    /// broadcast semantics (size-1 dims broadcast). When broadcast is needed,
    /// the tensor data is materialised on host and re-uploaded — this matches
    /// Candle's `broadcast_matmul` strategy and is fine for B4.2 correctness;
    /// performance optimisation is a later concern.
    ///
    /// cuBLAS column-major trick: `C = A @ B` row-major is computed as
    /// `gemm(B, A)` with `CUBLAS_OP_N` on both sides. See `kernels::matmul_f32`.
    fn matmul(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("matmul", &self.stream);
        // B'.2a diagnostic (`PM_CUDA_MATMUL_TRACE=1`): per-call sub-op wall
        // marks (alloc / gemm / tape+rest) with stream syncs at each mark.
        // Zero cost when the env var is unset.
        let trace_t0 = if mm_trace_enabled() {
            let _ = self.stream.synchronize();
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut mm_marks = [0.0f64; 2];
        let a_shape = a.shape().to_vec();
        let rank = a_shape.len();
        if rank < 2 {
            return Err(CudaError::Unsupported(
                "matmul: inputs must be at least rank-2",
            ));
        }

        // NumPy / PyTorch broadcast-matmul rule: if b has fewer dims than a,
        // pad b's shape with leading 1s so both have the same rank.
        // e.g. a=[batch, seq, d_in], b=[d_in, d_out] → b_padded=[1, d_in, d_out].
        // The existing broadcast path handles the expansion on the batch dims.
        let b_padded_shape: Vec<usize> = if b.shape().len() < rank {
            let n_pad = rank - b.shape().len();
            let mut s = vec![1usize; n_pad];
            s.extend_from_slice(b.shape());
            s
        } else {
            b.shape().to_vec()
        };
        let b_shape = b_padded_shape;

        if b_shape.len() != rank {
            return Err(CudaError::Shape(format!(
                "matmul: rank mismatch: a.rank={rank}, b.rank={}",
                b_shape.len()
            )));
        }

        let m = a_shape[rank - 2];
        let k = a_shape[rank - 1];
        let n = b_shape[rank - 1];

        // Compute output shape: broadcast leading dims element-wise.
        // numpy rule — pair must be equal, or one of them must be 1.
        // Any other case is an unbroadcastable shape and must error out
        // *before* allocation, otherwise we hand cuBLAS a buffer with the
        // wrong size and the asserts inside the kernel only fire after
        // memory has been allocated and copied.
        let out_shape = if rank == 2 {
            vec![m, n]
        } else {
            let mut s: Vec<usize> = Vec::with_capacity(rank);
            for (&da, &db) in a_shape[..rank - 2].iter().zip(b_shape[..rank - 2].iter()) {
                if da == db {
                    s.push(da);
                } else if da == 1 {
                    s.push(db);
                } else if db == 1 {
                    s.push(da);
                } else {
                    return Err(CudaError::Shape(format!(
                        "matmul: leading dims not broadcastable: {da} vs {db} (a={a_shape:?}, b={b_shape:?})"
                    )));
                }
            }
            s.push(m);
            s.push(n);
            s
        };

        // Check if broadcast materialisation is needed (any leading dim differs).
        let needs_broadcast_a = rank > 2 && a_shape[..rank - 2] != out_shape[..rank - 2];
        let needs_broadcast_b = rank > 2 && b_shape[..rank - 2] != out_shape[..rank - 2];

        // 1-vs-N batch broadcast (B'.2b): when the side that needs
        // broadcasting has ALL leading dims == 1, `kernels::matmul_f32`
        // handles the expansion natively with a stride-0 batched gemm — no
        // host materialisation. This is the PHOTON level-1 hot path
        // ([n_chunks, chunk, d] @ [1, d, d']); the host path used to expand
        // the shared weight n_chunks-fold on the CPU (~190 ms/call at T=512).
        // Per-dim mixed broadcasts (e.g. [1,3,..] @ [3,1,..]) cannot be
        // expressed as a single flattened stride-0 batch and stay on the
        // host path.
        let batch_a: usize = a_shape[..rank - 2].iter().product();
        let batch_b: usize = b_shape[..rank - 2].iter().product();
        let stride0_broadcast_ok =
            (!needs_broadcast_a || batch_a == 1) && (!needs_broadcast_b || batch_b == 1);

        if (needs_broadcast_a || needs_broadcast_b) && stride0_broadcast_ok {
            if trace_t0.is_some() {
                eprintln!("[mm-trace] DIRECT-BROADCAST (stride-0) a={a_shape:?} b={b_shape:?}");
            }
            // Fall through to the fast path below: `kernels::matmul_f32`
            // flattens the leading dims and uses stride 0 for the batch-1
            // side; the tape push saves the original (unexpanded) tensors,
            // and the backward arm's `sum_to_shape` reduces grads back.
        } else if needs_broadcast_a || needs_broadcast_b {
            if trace_t0.is_some() {
                eprintln!("[mm-trace] HOST-EXPAND branch a={a_shape:?} b={b_shape:?}");
            }
            // Materialise on host: D2H, expand, H2D, then call matmul on expanded tensors.
            // This matches Candle's broadcast_matmul strategy.
            //
            // Tape note: we push TapeOp::MatMul with the *original* (unexpanded) a/b tensors.
            // The backward arm uses sum_to_shape to reduce grad_A / grad_B back to the
            // original shapes — so saving the unexpanded tensors here is correct.
            let a_host = self.stream.clone_dtoh(a.storage().as_ref())?;
            let b_host = self.stream.clone_dtoh(b.storage().as_ref())?;

            let batch_out: usize = out_shape[..rank - 2].iter().product();

            // Expand A to out_batch × M × K
            let mut a_expanded = vec![0.0f32; batch_out * m * k];
            broadcast_expand_batch(
                &a_host,
                &a_shape[..rank - 2],
                &out_shape[..rank - 2],
                m * k,
                &mut a_expanded,
            )?;

            // Expand B to out_batch × K × N
            let mut b_expanded = vec![0.0f32; batch_out * k * n];
            broadcast_expand_batch(
                &b_host,
                &b_shape[..rank - 2],
                &out_shape[..rank - 2],
                k * n,
                &mut b_expanded,
            )?;

            // Build contiguous shapes for kernel call
            let a_flat_shape = [batch_out, m, k];
            let b_flat_shape = [batch_out, k, n];

            let a_dev = self.stream.clone_htod(&a_expanded)?;
            let b_dev = self.stream.clone_htod(&b_expanded)?;
            let out_n = out_shape.iter().product::<usize>();
            let mut out_storage = self.stream.alloc_zeros::<f32>(out_n)?;

            super::kernels::matmul_f32(
                &self.cublas,
                &a_dev,
                &a_flat_shape,
                &b_dev,
                &b_flat_shape,
                &mut out_storage,
            )?;

            let out_t = CudaTensor::new(Arc::new(out_storage), out_shape);
            // Push to tape with original (unexpanded) tensors; backward uses sum_to_shape
            // to reduce grad_A / grad_B from the broadcast output shape back to the
            // original input shapes.
            if grad_enabled() && (a.node_id().is_some() || b.node_id().is_some()) {
                let nid = tape_push(
                    &self.tape,
                    TapeOp::MatMul {
                        a: a.node_id(),
                        b: b.node_id(),
                        a_val: a.clone(),
                        b_val: b.clone(),
                    },
                );
                return Ok(out_t.with_node_id(nid));
            }
            return Ok(out_t);
        }

        let out_n = out_shape.iter().product::<usize>();
        let mut out_storage = self.stream.alloc_zeros::<f32>(out_n)?;
        if let Some(t0) = trace_t0 {
            let _ = self.stream.synchronize();
            mm_marks[0] = t0.elapsed().as_secs_f64();
        }

        super::kernels::matmul_f32(
            &self.cublas,
            a.storage().as_ref(),
            &a_shape,
            b.storage().as_ref(),
            &b_shape,
            &mut out_storage,
        )?;
        if let Some(t0) = trace_t0 {
            let _ = self.stream.synchronize();
            mm_marks[1] = t0.elapsed().as_secs_f64();
        }

        let out_t = CudaTensor::new(Arc::new(out_storage), out_shape);
        let out_t = if grad_enabled() && (a.node_id().is_some() || b.node_id().is_some()) {
            let nid = tape_push(
                &self.tape,
                TapeOp::MatMul {
                    a: a.node_id(),
                    b: b.node_id(),
                    a_val: a.clone(),
                    b_val: b.clone(),
                },
            );
            out_t.with_node_id(nid)
        } else {
            out_t
        };
        if let Some(t0) = trace_t0 {
            let total = t0.elapsed().as_secs_f64();
            eprintln!(
                "[mm-trace] dev a={a_shape:?} b={b_shape:?} alloc_us={:.0} gemm_us={:.0} rest_us={:.0}",
                mm_marks[0] * 1e6,
                (mm_marks[1] - mm_marks[0]) * 1e6,
                (total - mm_marks[1]) * 1e6
            );
        }
        Ok(out_t)
    }

    // ---- Activations / Normalisation ----------------------------------------

    fn rmsnorm(
        &self,
        x: &CudaTensor,
        weight: &CudaTensor,
        eps: f32,
    ) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("rmsnorm", &self.stream);
        let shape = x.shape().to_vec();
        let n = x.numel();
        let mut out = self.stream.alloc_zeros::<f32>(n)?;
        super::kernels::rmsnorm_f32(
            &self.ctx,
            &self.stream,
            x.storage().as_ref(),
            weight.storage().as_ref(),
            &mut out,
            &shape,
            eps,
        )?;
        let out_t = CudaTensor::new(Arc::new(out), shape);
        if grad_enabled() && (x.node_id().is_some() || weight.node_id().is_some()) {
            let nid = tape_push(
                &self.tape,
                TapeOp::RmsNorm {
                    x: x.node_id(),
                    w: weight.node_id(),
                    x_val: x.clone(),
                    w_val: weight.clone(),
                    eps,
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    fn silu(&self, x: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("silu", &self.stream);
        let n = x.numel();
        let mut out = self.stream.alloc_zeros::<f32>(n)?;
        super::kernels::silu_f32(&self.ctx, &self.stream, x.storage().as_ref(), &mut out)?;
        let out_t = CudaTensor::new(Arc::new(out), x.shape().to_vec());
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::Silu {
                    input: x.node_id(),
                    input_val: x.clone(),
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    fn softplus(&self, x: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("softplus", &self.stream);
        let n = x.numel();
        let mut out = self.stream.alloc_zeros::<f32>(n)?;
        super::kernels::softplus_f32(&self.ctx, &self.stream, x.storage().as_ref(), &mut out)?;
        let out_t = CudaTensor::new(Arc::new(out), x.shape().to_vec());
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::Softplus {
                    input: x.node_id(),
                    input_val: x.clone(),
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    fn sigmoid(&self, x: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("sigmoid", &self.stream);
        let n = x.numel();
        let mut out = self.stream.alloc_zeros::<f32>(n)?;
        super::kernels::sigmoid_f32(&self.ctx, &self.stream, x.storage().as_ref(), &mut out)?;
        let out_t = CudaTensor::new(Arc::new(out), x.shape().to_vec());
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::Sigmoid {
                    input: x.node_id(),
                    out_val: out_t.clone(),
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    // ---- Convolution --------------------------------------------------------

    fn conv1d(
        &self,
        x: &CudaTensor,
        weight: &CudaTensor,
        bias: Option<&CudaTensor>,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("conv1d", &self.stream);
        let x_shape = x.shape().to_vec();
        let w_shape = weight.shape().to_vec();
        if x_shape.len() != 3 {
            return Err(CudaError::Shape(
                "conv1d: x must be rank-3 (B, C_in, T_in)".to_string(),
            ));
        }
        if w_shape.len() != 3 {
            return Err(CudaError::Shape(
                "conv1d: weight must be rank-3 (C_out, C_in/groups, K)".to_string(),
            ));
        }
        let batch = x_shape[0];
        let c_out = w_shape[0];
        let k_size = w_shape[2];
        let t_in = x_shape[2];
        if stride == 0 {
            return Err(CudaError::Shape("conv1d: stride must be >= 1".to_string()));
        }
        // Guard the t_out subtraction below. In release mode the
        // unchecked `t_in + 2*padding - k_size` wraps to a huge usize
        // when the kernel is wider than the padded input, which
        // silently corrupts the alloc_zeros size that follows.
        if t_in + 2 * padding < k_size {
            return Err(CudaError::Shape(format!(
                "conv1d: t_in={t_in} + 2·padding={padding} = {} < k_size={k_size}",
                t_in + 2 * padding
            )));
        }
        let t_out = (t_in + 2 * padding - k_size) / stride + 1;
        let out_shape = vec![batch, c_out, t_out];

        let bias_slice = bias.map(|b| b.storage().as_ref());
        let out_storage = super::kernels::conv1d_f32(
            &self.ctx,
            &self.stream,
            &self.cublas,
            x.storage().as_ref(),
            &x_shape,
            weight.storage().as_ref(),
            &w_shape,
            bias_slice,
            stride,
            padding,
            groups,
        )?;
        let out_t = CudaTensor::new(Arc::new(out_storage), out_shape);
        if grad_enabled()
            && (x.node_id().is_some()
                || weight.node_id().is_some()
                || bias.and_then(|b| b.node_id()).is_some())
        {
            let bias_node = bias.map(|b| b.node_id());
            let nid = tape_push(
                &self.tape,
                TapeOp::Conv1d {
                    x: x.node_id(),
                    w: weight.node_id(),
                    bias_node,
                    x_val: x.clone(),
                    w_val: weight.clone(),
                    stride,
                    padding,
                    groups,
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    // ---- Indexing / reduction -----------------------------------------------

    fn cumsum(&self, x: &CudaTensor, dim: usize) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("cumsum", &self.stream);
        let shape = x.shape().to_vec();
        let rank = shape.len();
        if rank == 0 {
            return Err(CudaError::Shape("cumsum: empty shape".to_string()));
        }
        let last_dim = rank - 1;
        let out_t = if dim == last_dim {
            // Fast path: last dimension — launch directly.
            let n = x.numel();
            let mut out = self.stream.alloc_zeros::<f32>(n)?;
            super::kernels::cumsum_lastdim_f32(
                &self.ctx,
                &self.stream,
                x.storage().as_ref(),
                &mut out,
                &shape,
            )?;
            CudaTensor::new(Arc::new(out), shape.clone())
        } else {
            // Non-last dim: transpose so target dim becomes last, run kernel,
            // then transpose back.
            let x_t = <Self as pm_core::Ops>::transpose(self, x, dim, last_dim)?;
            let x_t_shape = x_t.shape().to_vec();
            let n = x_t.numel();
            let mut out_t = self.stream.alloc_zeros::<f32>(n)?;
            super::kernels::cumsum_lastdim_f32(
                &self.ctx,
                &self.stream,
                x_t.storage().as_ref(),
                &mut out_t,
                &x_t_shape,
            )?;
            let y_t = CudaTensor::new(Arc::new(out_t), x_t_shape);
            <Self as pm_core::Ops>::transpose(self, &y_t, dim, last_dim)?
        };
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::Cumsum {
                    input: x.node_id(),
                    dim,
                    input_shape: shape,
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    fn narrow(
        &self,
        x: &CudaTensor,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("narrow", &self.stream);
        let src_shape = x.shape().to_vec();
        let rank = src_shape.len();
        if dim >= rank {
            return Err(CudaError::Shape(format!(
                "narrow: dim {dim} out of range for rank {rank}"
            )));
        }
        if start + len > src_shape[dim] {
            return Err(CudaError::Shape(format!(
                "narrow: start({start})+len({len}) > dim size {}",
                src_shape[dim]
            )));
        }
        let mut dst_shape = src_shape.clone();
        dst_shape[dim] = len;
        let n_out: usize = dst_shape.iter().product();
        let mut out = self.stream.alloc_zeros::<f32>(n_out)?;
        super::kernels::narrow_copy_f32(
            &self.ctx,
            &self.stream,
            x.storage().as_ref(),
            &mut out,
            &src_shape,
            dim,
            start,
            len,
        )?;
        let out_t = CudaTensor::new(Arc::new(out), dst_shape);
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::Narrow {
                    input: x.node_id(),
                    dim,
                    start,
                    len,
                    orig_shape: src_shape,
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    // ---- Shape manipulation -------------------------------------------------

    fn reshape(&self, x: &CudaTensor, shape: &[usize]) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("reshape", &self.stream);
        let old_numel: usize = x.shape().iter().product();
        let new_numel: usize = shape.iter().product();
        if old_numel != new_numel {
            return Err(CudaError::Shape(format!(
                "reshape: numel mismatch {old_numel} -> {new_numel} (new shape: {shape:?})"
            )));
        }
        // `with_shape` drops node_id intentionally; we stamp a fresh one below.
        let out = x.with_shape(shape.to_vec());
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::Reshape {
                    input: x.node_id(),
                    orig_shape: x.shape().to_vec(),
                },
            );
            Ok(out.with_node_id(nid))
        } else {
            Ok(out)
        }
    }

    fn transpose(
        &self,
        x: &CudaTensor,
        dim_a: usize,
        dim_b: usize,
    ) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("transpose", &self.stream);
        let src_shape = x.shape().to_vec();
        let rank = src_shape.len();
        if dim_a >= rank || dim_b >= rank {
            return Err(CudaError::Shape(format!(
                "transpose: dim_a={dim_a} or dim_b={dim_b} out of range for rank {rank}"
            )));
        }
        if dim_a == dim_b {
            // No-op: return a copy with same shape.
            let data = self.stream.clone_dtoh(x.storage().as_ref())?;
            let out = self.stream.clone_htod(&data)?;
            return Ok(CudaTensor::new(Arc::new(out), src_shape));
        }

        // Compute output shape.
        let mut dst_shape = src_shape.clone();
        dst_shape.swap(dim_a, dim_b);
        let n: usize = src_shape.iter().product();
        let mut out = self.stream.alloc_zeros::<f32>(n)?;

        if rank == 2 {
            // Fast path: 2D kernel.
            super::kernels::transpose_2d_f32(
                &self.ctx,
                &self.stream,
                x.storage().as_ref(),
                &mut out,
                src_shape[0],
                src_shape[1],
            )?;
        } else {
            // ND: compute in_strides (in output coord order) and out_strides.
            //
            // out_strides: row-major strides of the output tensor.
            // in_strides[d]: stride in the source tensor for the axis that
            //   maps to output axis d. For most dims this is src_strides[d];
            //   for dim_a and dim_b the axes are swapped.
            //
            // src row-major strides:
            let mut src_strides = vec![1u32; rank];
            for i in (0..rank - 1).rev() {
                src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as u32;
            }
            // out row-major strides:
            let mut out_strides = vec![1u32; rank];
            for i in (0..rank - 1).rev() {
                out_strides[i] = out_strides[i + 1] * dst_shape[i + 1] as u32;
            }
            // in_strides in output coordinate order: dim_a and dim_b are swapped.
            let mut in_strides_out_order = src_strides.clone();
            in_strides_out_order.swap(dim_a, dim_b);

            super::kernels::transpose_nd_f32(
                &self.ctx,
                &self.stream,
                x.storage().as_ref(),
                &mut out,
                &in_strides_out_order,
                &out_strides,
            )?;
        }
        let out_t = CudaTensor::new(Arc::new(out), dst_shape);
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::Transpose {
                    input: x.node_id(),
                    dim_a,
                    dim_b,
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    /// Broadcast `x` to `shape` using numpy-style broadcasting rules.
    ///
    /// **Note:** rank-extension broadcasting is not supported; `x` must already
    /// have the same rank as `shape`.  Callers that need to add leading size-1
    /// dims (e.g. `[3]` → `[2, 3]`) must explicitly reshape `x` to the full
    /// rank (padding with size-1 dims) before calling this function.
    fn broadcast_as(&self, x: &CudaTensor, shape: &[usize]) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("broadcast_as", &self.stream);
        let src_shape = x.shape().to_vec();
        let dst_shape = shape.to_vec();
        let rank = dst_shape.len();
        if src_shape.len() != rank {
            return Err(CudaError::Shape(format!(
                "broadcast_as: rank mismatch src={} dst={rank}",
                src_shape.len()
            )));
        }
        // Validate: each src dim must be either 1 or equal to dst dim.
        for (i, (&s, &d)) in src_shape.iter().zip(dst_shape.iter()).enumerate() {
            if s != 1 && s != d {
                return Err(CudaError::Shape(format!(
                    "broadcast_as: dim {i} src={s} cannot broadcast to dst={d}"
                )));
            }
        }
        let n_out: usize = dst_shape.iter().product();
        let mut out = self.stream.alloc_zeros::<f32>(n_out)?;
        super::kernels::broadcast_copy_f32(
            &self.ctx,
            &self.stream,
            x.storage().as_ref(),
            &mut out,
            &src_shape,
            &dst_shape,
        )?;
        let out_t = CudaTensor::new(Arc::new(out), dst_shape.clone());
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::BroadcastAs {
                    input: x.node_id(),
                    orig_shape: src_shape,
                    target_shape: dst_shape,
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    fn concat(&self, tensors: &[&CudaTensor], dim: usize) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("concat", &self.stream);
        if tensors.is_empty() {
            return Err(CudaError::Shape("concat: empty tensor list".to_string()));
        }
        let first_shape = tensors[0].shape().to_vec();
        let rank = first_shape.len();
        if dim >= rank {
            return Err(CudaError::Shape(format!(
                "concat: dim {dim} out of range for rank {rank}"
            )));
        }
        // Validate all tensors have matching shape except on `dim`.
        for (i, t) in tensors.iter().enumerate().skip(1) {
            let s = t.shape();
            if s.len() != rank {
                return Err(CudaError::Shape(format!(
                    "concat: tensor {i} rank {} != {rank}",
                    s.len()
                )));
            }
            for (d, (&a, &b)) in first_shape.iter().zip(s.iter()).enumerate() {
                if d != dim && a != b {
                    return Err(CudaError::Shape(format!(
                        "concat: tensor {i} dim {d} size {b} != {a}"
                    )));
                }
            }
        }
        // Compute output shape.
        let mut out_shape = first_shape.clone();
        out_shape[dim] = tensors.iter().map(|t| t.shape()[dim]).sum();
        let n_out: usize = out_shape.iter().product();

        // Perform concat via memcpy.
        // For dim=0: simple sequential copy of each tensor's full data.
        // For dim>0: copy row by row (strided copy).
        //
        // We materialise on host for simplicity — this is correct and
        // B4.5 can replace with a proper GPU concat kernel if needed.
        let mut host_out: Vec<f32> = vec![0.0f32; n_out];

        // Use host-side gather: D2H each input then fill output.
        let outer: usize = out_shape[..dim].iter().product();
        let inner: usize = out_shape[dim + 1..].iter().product();
        let out_axis_stride = inner;
        let out_outer_stride = out_shape[dim] * inner;

        let mut axis_offset = 0usize;
        for t in tensors.iter() {
            let t_host = self.stream.clone_dtoh(t.storage().as_ref())?;
            let t_dim_len = t.shape()[dim];
            let t_outer_stride = t_dim_len * inner;
            for o in 0..outer {
                for a in 0..t_dim_len {
                    let src_base = o * t_outer_stride + a * inner;
                    let dst_base = o * out_outer_stride + (axis_offset + a) * out_axis_stride;
                    host_out[dst_base..dst_base + inner]
                        .copy_from_slice(&t_host[src_base..src_base + inner]);
                }
            }
            axis_offset += t_dim_len;
        }

        let out = self.stream.clone_htod(&host_out)?;
        let out_t = CudaTensor::new(Arc::new(out), out_shape);

        // Record tape entry if any input is tracked.
        let any_tracked = tensors.iter().any(|t| t.node_id().is_some());
        if grad_enabled() && any_tracked {
            let inputs: Vec<Option<NodeId>> = tensors.iter().map(|t| t.node_id()).collect();
            let input_shapes: Vec<Vec<usize>> =
                tensors.iter().map(|t| t.shape().to_vec()).collect();
            let nid = tape_push(
                &self.tape,
                TapeOp::Concat {
                    inputs,
                    dim,
                    input_shapes,
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    // ---- Embedding ----------------------------------------------------------

    fn embedding(&self, table: &CudaTensor, indices: &CudaTensor) -> Result<CudaTensor, CudaError> {
        // Phase B'.1b: the profiler's fwd/bwd/post phase state machine
        // treats this as the first op of every forward pass — see
        // `crate::profiler`'s module docs — so its own `timer()` call is
        // load-bearing beyond just recording `embedding`'s own cost.
        let _t = crate::profiler::timer("embedding", &self.stream);
        let table_shape = table.shape().to_vec();
        if table_shape.len() != 2 {
            return Err(CudaError::Shape(
                "embedding: table must be rank-2 (V, D)".to_string(),
            ));
        }
        let vocab_size = table_shape[0];
        let embed_dim = table_shape[1];
        let idx_shape = indices.shape().to_vec();
        let indices_len: usize = idx_shape.iter().product();

        // Output shape: indices.shape ++ [embed_dim]
        let mut out_shape = idx_shape.clone();
        out_shape.push(embed_dim);
        let n_out = indices_len * embed_dim;
        let mut out = self.stream.alloc_zeros::<f32>(n_out)?;

        super::kernels::embedding_f32(
            &self.ctx,
            &self.stream,
            table.storage().as_ref(),
            indices.storage_i64().as_ref(),
            &mut out,
            indices_len,
            embed_dim,
            vocab_size,
        )?;
        let out_t = CudaTensor::new(Arc::new(out), out_shape);
        if grad_enabled() && table.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::Embedding {
                    table: table.node_id(),
                    indices_val: indices.clone(),
                    table_shape: table_shape.clone(),
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    // ---- Element-wise transcendental ----------------------------------------

    fn exp(&self, x: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("exp", &self.stream);
        let n = x.numel();
        let mut out = self.stream.alloc_zeros::<f32>(n)?;
        super::kernels::exp_f32(&self.ctx, &self.stream, x.storage().as_ref(), &mut out)?;
        let out_t = CudaTensor::new(Arc::new(out), x.shape().to_vec());
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::Exp {
                    input: x.node_id(),
                    out_val: out_t.clone(),
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    // ---- Trainable parameters -----------------------------------------------

    fn param_from_slice_f32(&self, data: &[f32], shape: &[usize]) -> Result<CudaParam, CudaError> {
        let _t = crate::profiler::timer("param_from_slice_f32", &self.stream);
        let id = self.alloc_param_id();
        let t = self.from_slice_f32_inner(data, shape)?;
        let tensor = if grad_enabled() {
            register_leaf(&self.tape, t, id)
        } else {
            t
        };
        Ok(CudaParam::new(tensor, id))
    }

    fn param_zeros(&self, shape: &[usize], dtype: Dtype) -> Result<CudaParam, CudaError> {
        let _t = crate::profiler::timer("param_zeros", &self.stream);
        let id = self.alloc_param_id();
        let t = <Self as Ops>::zeros(self, shape, dtype)?;
        let tensor = if grad_enabled() {
            register_leaf(&self.tape, t, id)
        } else {
            t
        };
        Ok(CudaParam::new(tensor, id))
    }

    fn param_ones(&self, shape: &[usize], dtype: Dtype) -> Result<CudaParam, CudaError> {
        let _t = crate::profiler::timer("param_ones", &self.stream);
        let id = self.alloc_param_id();
        let t = <Self as Ops>::ones(self, shape, dtype)?;
        let tensor = if grad_enabled() {
            register_leaf(&self.tape, t, id)
        } else {
            t
        };
        Ok(CudaParam::new(tensor, id))
    }

    // ---- Autograd (B4.3a) ---------------------------------------------------

    /// Reverse-mode autodiff.
    ///
    /// Walk the tape in reverse, accumulating VJP contributions.
    /// All VJP arithmetic runs inside `no_grad` to avoid re-recording.
    ///
    /// After completing successfully the tape is automatically cleared.
    /// The next `forward` pass starts from a clean tape and any
    /// `sgd_step`/`assign` calls will push fresh `Leaf` entries.
    ///
    /// References: Mamba2 §4 (SSD gradient); standard textbook VJPs.
    fn backward(&self, loss: &CudaTensor) -> Result<CudaGradStore, CudaError> {
        // Phase B'.1b: dedicated guard (bypasses the shared nesting-depth
        // counter so the per-VJP `vjp:<Variant>` timers in `apply_vjp`
        // still fire — see `crate::profiler::enter_backward`'s doc
        // comment) that also drives the fwd/bwd/post phase state machine:
        // flips it to `bwd` now, and to `post` when `_bwd_total` drops at
        // the end of this call (every exit path, including the early
        // `Ok(CudaGradStore::new())` return below).
        let _bwd_total = crate::profiler::enter_backward(&self.stream);
        // loss must be a scalar (shape [1] or []).
        let numel = loss.numel();
        if numel != 1 {
            return Err(CudaError::Shape(format!(
                "backward: loss must be a scalar (numel=1), got shape {:?} (numel={numel})",
                loss.shape()
            )));
        }

        let root_nid = match loss.node_id() {
            Some(nid) => nid,
            None => {
                // Loss is detached — nothing to differentiate.
                return Ok(CudaGradStore::new());
            }
        };

        // Snapshot tape entries + current generation under lock, then release.
        // `generation` is needed to construct `NodeId`s below — see
        // `tape.rs`'s `NodeId` doc for why the generation tag (not just the
        // index) matters for checkpointed backward correctness.
        let (n_entries, generation) = {
            let t = self
                .tape
                .lock()
                .map_err(|_| CudaError::Internal("tape lock poisoned".into()))?;
            (t.entries.len(), t.generation)
        };

        // grads[nid] = accumulated gradient for node `nid`.
        let mut grads: HashMap<NodeId, CudaTensor> = HashMap::new();

        // Seed: d(loss)/d(loss) = 1 (scalar).
        let seed = no_grad(|| {
            let host = vec![1.0f32];
            self.stream
                .clone_htod(&host)
                .map(|s| CudaTensor::new(Arc::new(s), loss.shape().to_vec()))
        })?;
        grads.insert(root_nid, seed);

        // param_id → accumulated gradient (collected at Leaf nodes).
        let mut param_grads: HashMap<super::param::ParamId, CudaTensor> = HashMap::new();

        // Traverse tape in reverse.
        for idx in (0..n_entries).rev() {
            let nid = NodeId {
                generation,
                index: idx as u32,
            };
            let grad_out = match grads.remove(&nid) {
                Some(g) => g,
                None => continue, // not on any path from loss → skip
            };

            // We need to read the TapeOp for this node.  We snapshot the
            // shape/variant under lock to avoid holding the lock across
            // potentially expensive GPU operations.
            //
            // The tape is only appended during forward; we never remove
            // entries, so `idx` is always a valid index.
            //
            // We clone the minimal data we need out of the TapeOp.
            let op_snapshot = {
                let tape = self
                    .tape
                    .lock()
                    .map_err(|_| CudaError::Internal("tape lock poisoned".into()))?;
                // Clone just what we need per variant.
                match tape.get(nid) {
                    TapeOp::Leaf { param_id } => BackwardOp::Leaf {
                        param_id: *param_id,
                    },
                    TapeOp::Add {
                        lhs,
                        rhs,
                        lhs_shape,
                        rhs_shape,
                    } => BackwardOp::Add {
                        lhs: *lhs,
                        rhs: *rhs,
                        lhs_shape: lhs_shape.clone(),
                        rhs_shape: rhs_shape.clone(),
                    },
                    TapeOp::Sub {
                        lhs,
                        rhs,
                        lhs_shape,
                        rhs_shape,
                    } => BackwardOp::Sub {
                        lhs: *lhs,
                        rhs: *rhs,
                        lhs_shape: lhs_shape.clone(),
                        rhs_shape: rhs_shape.clone(),
                    },
                    TapeOp::Mul {
                        lhs,
                        rhs,
                        lhs_val,
                        rhs_val,
                    } => BackwardOp::Mul {
                        lhs: *lhs,
                        rhs: *rhs,
                        lhs_val: lhs_val.clone(),
                        rhs_val: rhs_val.clone(),
                    },
                    TapeOp::Neg { input } => BackwardOp::Neg { input: *input },
                    TapeOp::Reshape { input, orig_shape } => BackwardOp::Reshape {
                        input: *input,
                        orig_shape: orig_shape.clone(),
                    },
                    TapeOp::MulScalar { input, scale } => BackwardOp::MulScalar {
                        input: *input,
                        scale: *scale,
                    },
                    TapeOp::AddScalar { input } => BackwardOp::AddScalar { input: *input },
                    TapeOp::Sqrt { input, out_val } => BackwardOp::Sqrt {
                        input: *input,
                        out_val: out_val.clone(),
                    },
                    TapeOp::Div {
                        lhs,
                        rhs,
                        rhs_val,
                        out_val,
                    } => BackwardOp::Div {
                        lhs: *lhs,
                        rhs: *rhs,
                        rhs_val: rhs_val.clone(),
                        out_val: out_val.clone(),
                    },
                    TapeOp::Exp { input, out_val } => BackwardOp::Exp {
                        input: *input,
                        out_val: out_val.clone(),
                    },
                    TapeOp::Silu { input, input_val } => BackwardOp::Silu {
                        input: *input,
                        input_val: input_val.clone(),
                    },
                    TapeOp::Sigmoid { input, out_val } => BackwardOp::Sigmoid {
                        input: *input,
                        out_val: out_val.clone(),
                    },
                    TapeOp::Softplus { input, input_val } => BackwardOp::Softplus {
                        input: *input,
                        input_val: input_val.clone(),
                    },
                    // ---- B4.3b ops ------------------------------------------------
                    TapeOp::MatMul { a, b, a_val, b_val } => BackwardOp::MatMul {
                        a: *a,
                        b: *b,
                        a_val: a_val.clone(),
                        b_val: b_val.clone(),
                    },
                    TapeOp::Transpose {
                        input,
                        dim_a,
                        dim_b,
                    } => BackwardOp::Transpose {
                        input: *input,
                        dim_a: *dim_a,
                        dim_b: *dim_b,
                    },
                    TapeOp::Narrow {
                        input,
                        dim,
                        start,
                        len: _,
                        orig_shape,
                    } => BackwardOp::Narrow {
                        input: *input,
                        dim: *dim,
                        start: *start,
                        orig_shape: orig_shape.clone(),
                    },
                    TapeOp::BroadcastAs {
                        input,
                        orig_shape,
                        target_shape: _,
                    } => BackwardOp::BroadcastAs {
                        input: *input,
                        orig_shape: orig_shape.clone(),
                    },
                    TapeOp::Concat {
                        inputs,
                        dim,
                        input_shapes,
                    } => BackwardOp::Concat {
                        inputs: inputs.clone(),
                        dim: *dim,
                        input_shapes: input_shapes.clone(),
                    },
                    TapeOp::Embedding {
                        table,
                        indices_val,
                        table_shape,
                    } => BackwardOp::Embedding {
                        table: *table,
                        indices_val: indices_val.clone(),
                        table_shape: table_shape.clone(),
                    },
                    TapeOp::Gather {
                        input,
                        indices_val,
                        dim,
                        input_shape,
                    } => BackwardOp::Gather {
                        input: *input,
                        indices_val: indices_val.clone(),
                        dim: *dim,
                        input_shape: input_shape.clone(),
                    },
                    TapeOp::Conv1d {
                        x,
                        w,
                        bias_node,
                        x_val,
                        w_val,
                        stride,
                        padding,
                        groups,
                    } => BackwardOp::Conv1d {
                        x: *x,
                        w: *w,
                        bias_node: *bias_node,
                        x_val: x_val.clone(),
                        w_val: w_val.clone(),
                        stride: *stride,
                        padding: *padding,
                        groups: *groups,
                    },
                    TapeOp::LogSoftmax {
                        input,
                        dim,
                        out_val,
                    } => BackwardOp::LogSoftmax {
                        input: *input,
                        dim: *dim,
                        out_val: out_val.clone(),
                    },
                    TapeOp::RmsNorm {
                        x,
                        w,
                        x_val,
                        w_val,
                        eps,
                    } => BackwardOp::RmsNorm {
                        x: *x,
                        w: *w,
                        x_val: x_val.clone(),
                        w_val: w_val.clone(),
                        eps: *eps,
                    },
                    TapeOp::Cumsum {
                        input,
                        dim,
                        input_shape,
                    } => BackwardOp::Cumsum {
                        input: *input,
                        dim: *dim,
                        input_shape: input_shape.clone(),
                    },
                    TapeOp::MeanAll {
                        input,
                        numel,
                        input_shape,
                    } => BackwardOp::MeanAll {
                        input: *input,
                        numel: *numel,
                        input_shape: input_shape.clone(),
                    },
                    TapeOp::SumAll { input, input_shape } => BackwardOp::SumAll {
                        input: *input,
                        input_shape: input_shape.clone(),
                    },
                    // B4.3c — M1: SsdScan is now boxed.
                    TapeOp::SsdScan(d) => BackwardOp::SsdScan {
                        x_in: d.x,
                        a_in: d.a,
                        b_in: d.b,
                        c_in: d.c,
                        x_val: d.x_val.clone(),
                        a_val: d.a_val.clone(),
                        b_val: d.b_val.clone(),
                        c_val: d.c_val.clone(),
                        block_len: d.block_len,
                    },
                }
            };

            // H2: delegate to shared apply_vjp — deduplicates logic with sub-walk.
            apply_vjp(self, op_snapshot, grad_out, &mut grads, &mut param_grads)?;
        }

        // Clear the tape so subsequent forward passes start fresh.
        // Leaves saved tensors inside TapeOp variants free to be dropped.
        self.tape
            .lock()
            .map_err(|_| CudaError::Internal("tape lock poisoned on clear".into()))?
            .clear();

        // B4.4d — OOM fix: cuMemFreeAsync calls from the Arc<CudaSlice<f32>>
        // drops above are *stream-ordered*, meaning the CUDA driver queues
        // them on the stream but does not retire them synchronously before
        // this function returns.  Without an explicit sync the very next
        // forward pass issues cuMemAllocAsync while those frees are still
        // pending, pushing apparent VRAM usage over the physical 12 GB limit
        // and triggering CUDA_ERROR_OUT_OF_MEMORY at step 1.
        //
        // Cost: ~0.5–2 ms per training step on RTX 5070 — negligible versus
        // the ~100-200 ms/step compute time of the 102 M PhotonMamba model.
        self.stream.synchronize().map_err(CudaError::Driver)?;

        Ok(CudaGradStore { grads: param_grads })
    }

    fn gradient(
        &self,
        store: &CudaGradStore,
        param: &CudaParam,
    ) -> Result<Option<CudaTensor>, CudaError> {
        let _t = crate::profiler::timer("gradient", &self.stream);
        Ok(store.grads.get(&param.inner.id).cloned())
    }

    fn set_gradient(
        &self,
        store: &mut CudaGradStore,
        param: &CudaParam,
        grad: CudaTensor,
    ) -> Result<(), CudaError> {
        let _t = crate::profiler::timer("set_gradient", &self.stream);
        store.grads.insert(param.inner.id, grad);
        Ok(())
    }

    /// Wrap an existing tensor as a fresh trainable parameter.
    ///
    /// The tensor's storage is shared (Arc clone — no data copy).
    /// A new `Leaf` tape entry is registered so `backward()` can
    /// route gradients to this parameter.
    fn param_from_tensor(&self, t: &CudaTensor) -> Result<CudaParam, CudaError> {
        let _t = crate::profiler::timer("param_from_tensor", &self.stream);
        let id = self.alloc_param_id();
        // Clone shares the storage Arc; only node_id changes.
        let cloned = t.clone();
        let tensor = if grad_enabled() {
            register_leaf(&self.tape, cloned, id)
        } else {
            cloned
        };
        Ok(CudaParam::new(tensor, id))
    }

    /// Storage-sharing clone with `node_id` stripped — no `Leaf` tape
    /// entry, no `ParamId`, unlike `param_from_tensor` above. This is
    /// the cheapest possible "stop autograd here": every `Ops` method in
    /// this file gates tape recording on `grad_enabled() &&
    /// operand.node_id().is_some()`, so composing further ops on a
    /// `detach`ed tensor pushes **nothing** onto the shared tape — not
    /// even a throwaway `Leaf` entry (which `param_from_tensor` would
    /// still leave tracked, defeating the point for
    /// `Ops::fused_cross_entropy`'s tiling loop; see `Ops::detach`'s doc
    /// comment).
    fn detach(&self, t: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("detach", &self.stream);
        let mut out = t.clone();
        out.node_id = None;
        // Must also drop `origin` — `CudaTensor::node_id()` checks `origin`
        // *before* the plain `node_id` field, so leaving it set would make
        // a detached clone of a param/checkpoint-boundary tensor silently
        // resurrect its `Leaf` entry (via `LeafOrigin::resolve`) the next
        // time something calls `.node_id()` on it, defeating detach.
        out.origin = None;
        Ok(out)
    }

    /// Scoped tape-off switch: same thread-local guard `backward()`
    /// itself uses for VJP arithmetic (`tape::no_grad`), exposed through
    /// the `Ops` surface so eval loops can run many forwards without
    /// the tape (and the activations its entries keep alive) growing
    /// unboundedly. See `Ops::no_grad_scope`'s doc for the OOM this
    /// prevents.
    fn no_grad_scope<R>(&self, f: impl FnOnce() -> R) -> R {
        no_grad(f)
    }

    /// Merge `src` gradient entries into `dst`.
    ///
    /// Existing entries in `dst` are summed with the incoming value.
    /// New entries are inserted directly.  All arithmetic is inside
    /// `no_grad` to avoid tape side-effects.
    fn merge_grad_stores(
        &self,
        dst: &mut CudaGradStore,
        src: CudaGradStore,
    ) -> Result<(), CudaError> {
        let _t = crate::profiler::timer("merge_grad_stores", &self.stream);
        for (pid, src_grad) in src.grads {
            match dst.grads.remove(&pid) {
                None => {
                    dst.grads.insert(pid, src_grad);
                }
                Some(existing) => {
                    let sum = no_grad(|| self.add_inner(&existing, &src_grad))?;
                    dst.grads.insert(pid, sum);
                }
            }
        }
        Ok(())
    }

    /// In-place SGD update: `param ← param − lr · grad`.
    ///
    /// Allocates a new device buffer for the updated value and replaces
    /// the parameter's internal tensor via `unsafe { param.set_tensor(…) }`.
    ///
    /// A fresh `TapeOp::Leaf` is pushed for the updated parameter so that
    /// the next `backward` call can route gradients correctly.  The old
    /// `NodeId` is **not** reused — reusing it after a `backward` (which
    /// clears the tape) would produce an out-of-bounds `NodeId` on the
    /// next iteration.
    ///
    /// # Safety of the unsafe block
    /// See `CudaParam::set_tensor` — single-threaded training loop
    /// invariant ensures no concurrent readers during the update.
    fn sgd_step(&self, param: &CudaParam, grad: &CudaTensor, lr: f32) -> Result<(), CudaError> {
        let _t = crate::profiler::timer("sgd_step", &self.stream);
        let current = param.as_tensor();
        if current.shape() != grad.shape() {
            return Err(CudaError::Shape(format!(
                "sgd_step: param shape {:?} != grad shape {:?}",
                current.shape(),
                grad.shape()
            )));
        }
        // new_val = current - lr * grad
        let new_val = no_grad(|| {
            let n = current.numel();
            let mut lr_grad = self.stream.alloc_zeros::<f32>(n)?;
            super::kernels::mul_scalar_f32(
                &self.ctx,
                &self.stream,
                grad.storage().as_ref(),
                lr,
                &mut lr_grad,
            )?;
            let lr_grad_t = CudaTensor::new(Arc::new(lr_grad), grad.shape().to_vec());
            self.sub_inner(current, &lr_grad_t)
        })?;

        // Push a new leaf for the updated value so the next forward/backward
        // step can track this parameter.  The tape was cleared by `backward`,
        // so re-using the old NodeId would be out-of-bounds.
        let updated = if grad_enabled() {
            register_leaf(&self.tape, new_val, param.param_id())
        } else {
            new_val
        };

        // SAFETY: single-threaded training loop; no concurrent reads of
        // `param.as_tensor()` while this write is in progress.
        unsafe { param.set_tensor(updated) };
        Ok(())
    }

    /// Overwrite parameter storage with `value` (same shape required).
    ///
    /// Used by optimisers (AdamW) that compute a fresh tensor off-line
    /// and need to replace the parameter atomically.
    ///
    /// A fresh `TapeOp::Leaf` is pushed for the assigned parameter so that
    /// the next `backward` call can route gradients correctly.  The old
    /// `NodeId` is **not** reused for the same reason as in `sgd_step`.
    fn assign(&self, param: &CudaParam, value: &CudaTensor) -> Result<(), CudaError> {
        let _t = crate::profiler::timer("assign", &self.stream);
        let current = param.as_tensor();
        if current.shape() != value.shape() {
            return Err(CudaError::Shape(format!(
                "assign: param shape {:?} != value shape {:?}",
                current.shape(),
                value.shape()
            )));
        }
        // Push a new leaf so the next backward can reach this parameter.
        let updated = if grad_enabled() {
            register_leaf(&self.tape, value.clone(), param.param_id())
        } else {
            value.clone()
        };
        // SAFETY: single-threaded training loop; no concurrent reads during write.
        unsafe { param.set_tensor(updated) };
        Ok(())
    }

    // ---- Loss / reduction helpers -------------------------------------------

    fn log_softmax(&self, x: &CudaTensor, dim: usize) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("log_softmax", &self.stream);
        let shape = x.shape().to_vec();
        let rank = shape.len();
        if rank == 0 {
            return Err(CudaError::Shape("log_softmax: empty shape".to_string()));
        }
        let last_dim = rank - 1;
        let out_t = if dim == last_dim {
            let n = x.numel();
            let mut out = self.stream.alloc_zeros::<f32>(n)?;
            super::kernels::log_softmax_lastdim_f32(
                &self.ctx,
                &self.stream,
                x.storage().as_ref(),
                &mut out,
                &shape,
            )?;
            CudaTensor::new(Arc::new(out), shape.clone())
        } else {
            // Non-last dim: transpose so target dim becomes last, run kernel,
            // then transpose back.
            let x_t = <Self as pm_core::Ops>::transpose(self, x, dim, last_dim)?;
            let x_t_shape = x_t.shape().to_vec();
            let n = x_t.numel();
            let mut out_t = self.stream.alloc_zeros::<f32>(n)?;
            super::kernels::log_softmax_lastdim_f32(
                &self.ctx,
                &self.stream,
                x_t.storage().as_ref(),
                &mut out_t,
                &x_t_shape,
            )?;
            let y_t = CudaTensor::new(Arc::new(out_t), x_t_shape);
            <Self as pm_core::Ops>::transpose(self, &y_t, dim, last_dim)?
        };
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::LogSoftmax {
                    input: x.node_id(),
                    dim,
                    out_val: out_t.clone(),
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    fn gather(
        &self,
        x: &CudaTensor,
        indices: &CudaTensor,
        dim: usize,
    ) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("gather", &self.stream);
        let src_shape = x.shape().to_vec();
        let idx_shape = indices.shape().to_vec();
        let rank = src_shape.len();
        if rank == 0 {
            return Err(CudaError::Shape("gather: empty source shape".to_string()));
        }
        if dim >= rank {
            return Err(CudaError::Shape(format!(
                "gather: dim {dim} out of range for rank {rank}"
            )));
        }
        if idx_shape.len() != rank {
            return Err(CudaError::Shape(format!(
                "gather: indices rank {} != src rank {rank}",
                idx_shape.len()
            )));
        }
        // Output shape matches indices shape.
        let out_shape = idx_shape.clone();
        let n_out: usize = out_shape.iter().product();
        let last_dim = rank - 1;

        let gather_out = if dim == last_dim {
            // Fast path: gather along last dim directly.
            let mut out = self.stream.alloc_zeros::<f32>(n_out)?;
            super::kernels::gather_lastdim_f32(
                &self.ctx,
                &self.stream,
                x.storage().as_ref(),
                indices.storage_i64().as_ref(),
                &mut out,
                src_shape[last_dim],
                idx_shape[last_dim],
            )?;
            CudaTensor::new(Arc::new(out), out_shape)
        } else {
            // Non-last dim: transpose the f32 source so the target dim becomes
            // last, then transpose the i64 indices on host (D2H permute then H2D),
            // run the lastdim kernel, then transpose the f32 output back.
            let x_t = <Self as pm_core::Ops>::transpose(self, x, dim, last_dim)?;

            // Host-side transpose of i64 indices.
            let idx_host = self.stream.clone_dtoh(indices.storage_i64().as_ref())?;
            let mut idx_t_shape = idx_shape.clone();
            idx_t_shape.swap(dim, last_dim);
            let n_idx = idx_host.len();
            let mut idx_t_host = vec![0i64; n_idx];

            // Compute src and dst strides for the transpose.
            let mut src_strides_idx = vec![1usize; rank];
            for i in (0..rank - 1).rev() {
                src_strides_idx[i] = src_strides_idx[i + 1] * idx_shape[i + 1];
            }
            let mut dst_strides_idx = vec![1usize; rank];
            for i in (0..rank - 1).rev() {
                dst_strides_idx[i] = dst_strides_idx[i + 1] * idx_t_shape[i + 1];
            }
            // Permutation: swap dim and last_dim.
            let mut perm: Vec<usize> = (0..rank).collect();
            perm.swap(dim, last_dim);

            for (dst_flat, slot) in idx_t_host.iter_mut().enumerate() {
                // Decode dst multi-index using dst_strides_idx.
                let mut remaining = dst_flat;
                let mut src_flat = 0usize;
                for d in 0..rank {
                    let coord = remaining / dst_strides_idx[d];
                    remaining %= dst_strides_idx[d];
                    // dst axis d comes from src axis perm[d].
                    src_flat += coord * src_strides_idx[perm[d]];
                }
                *slot = idx_host[src_flat];
            }

            let idx_t_dev = self.stream.clone_htod(&idx_t_host)?;
            let idx_t = super::CudaTensor::new_i64(Arc::new(idx_t_dev), idx_t_shape.clone());

            let x_t_shape = x_t.shape().to_vec();
            let n_t: usize = idx_t_shape.iter().product();
            let mut out_t = self.stream.alloc_zeros::<f32>(n_t)?;
            super::kernels::gather_lastdim_f32(
                &self.ctx,
                &self.stream,
                x_t.storage().as_ref(),
                idx_t.storage_i64().as_ref(),
                &mut out_t,
                x_t_shape[last_dim],
                idx_t_shape[last_dim],
            )?;
            let y_t = CudaTensor::new(Arc::new(out_t), idx_t_shape);
            // Transpose back so that the gather dim is restored.
            <Self as pm_core::Ops>::transpose(self, &y_t, dim, last_dim)?
        };
        // Record tape entry if x is tracked (indices are i64, no grad).
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::Gather {
                    input: x.node_id(),
                    indices_val: indices.clone(),
                    dim,
                    input_shape: src_shape,
                },
            );
            Ok(gather_out.with_node_id(nid))
        } else {
            Ok(gather_out)
        }
    }

    fn mean_all(&self, x: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("mean_all", &self.stream);
        let n = x.numel();
        if n == 0 {
            return Err(CudaError::Shape("mean_all: empty tensor".to_string()));
        }
        // Device-side deterministic two-pass reduce (B'.2f) — replaces a
        // D2H → host `iter().sum()` → H2D round trip. See kernels::sum_all_f32.
        let sum_dev = super::kernels::sum_all_f32(&self.ctx, &self.stream, x.storage().as_ref())?;
        let mut out = self.stream.alloc_zeros::<f32>(1)?;
        super::kernels::mul_scalar_f32(
            &self.ctx,
            &self.stream,
            &sum_dev,
            1.0 / n as f32,
            &mut out,
        )?;
        let out_t = CudaTensor::new(Arc::new(out), vec![1]);
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::MeanAll {
                    input: x.node_id(),
                    numel: n,
                    input_shape: x.shape().to_vec(),
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    fn sum_all(&self, x: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("sum_all", &self.stream);
        if x.numel() == 0 {
            return Err(CudaError::Shape("sum_all: empty tensor".to_string()));
        }
        // Device-side deterministic two-pass reduce (B'.2f) — replaces a
        // D2H → host `iter().sum()` → H2D round trip. See kernels::sum_all_f32.
        let sum_dev = super::kernels::sum_all_f32(&self.ctx, &self.stream, x.storage().as_ref())?;
        let out_t = CudaTensor::new(Arc::new(sum_dev), vec![1]);
        // sum_all is typically the final reduction before backward; record it
        // if x is tracked so backward can propagate through it.
        // VJP of sum_all: each input element gets grad_out broadcast.
        // We record this as a Reshape (conceptually flatten → the grad flows back as
        // a reshape to original shape with the scalar broadcast handled by the
        // broadcast-accumulate in the backward). However sum_all's broadcast VJP
        // B4.3b: record sum_all so backward can broadcast grad back.
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::SumAll {
                    input: x.node_id(),
                    input_shape: x.shape().to_vec(),
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    fn mul_scalar(&self, x: &CudaTensor, scale: f32) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("mul_scalar", &self.stream);
        let n = x.numel();
        let mut out = self.stream.alloc_zeros::<f32>(n)?;
        super::kernels::mul_scalar_f32(
            &self.ctx,
            &self.stream,
            x.storage().as_ref(),
            scale,
            &mut out,
        )?;
        let out_t = CudaTensor::new(Arc::new(out), x.shape().to_vec());
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::MulScalar {
                    input: x.node_id(),
                    scale,
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    fn sqrt(&self, x: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("sqrt", &self.stream);
        let n = x.numel();
        let mut out = self.stream.alloc_zeros::<f32>(n)?;
        super::kernels::sqrt_f32(&self.ctx, &self.stream, x.storage().as_ref(), &mut out)?;
        let out_t = CudaTensor::new(Arc::new(out), x.shape().to_vec());
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(
                &self.tape,
                TapeOp::Sqrt {
                    input: x.node_id(),
                    out_val: out_t.clone(),
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    fn div(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("div", &self.stream);
        if a.shape() != b.shape() {
            return Err(CudaError::Shape(format!(
                "div: shape mismatch {:?} vs {:?}",
                a.shape(),
                b.shape()
            )));
        }
        let n = a.numel();
        let mut out = self.stream.alloc_zeros::<f32>(n)?;
        super::kernels::div_f32(
            &self.ctx,
            &self.stream,
            a.storage().as_ref(),
            b.storage().as_ref(),
            &mut out,
        )?;
        let out_t = CudaTensor::new(Arc::new(out), a.shape().to_vec());
        if grad_enabled() && (a.node_id().is_some() || b.node_id().is_some()) {
            let nid = tape_push(
                &self.tape,
                TapeOp::Div {
                    lhs: a.node_id(),
                    rhs: b.node_id(),
                    rhs_val: b.clone(),
                    out_val: out_t.clone(),
                },
            );
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    fn add_scalar(&self, x: &CudaTensor, scalar: f32) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("add_scalar", &self.stream);
        let n = x.numel();
        let mut out = self.stream.alloc_zeros::<f32>(n)?;
        super::kernels::add_scalar_f32(
            &self.ctx,
            &self.stream,
            x.storage().as_ref(),
            scalar,
            &mut out,
        )?;
        let out_t = CudaTensor::new(Arc::new(out), x.shape().to_vec());
        if grad_enabled() && x.node_id().is_some() {
            let nid = tape_push(&self.tape, TapeOp::AddScalar { input: x.node_id() });
            Ok(out_t.with_node_id(nid))
        } else {
            Ok(out_t)
        }
    }

    // ---- Mamba2 SSD scan ---------------------------------------------------

    fn ssd_scan(
        &self,
        x: &CudaTensor,
        a: &CudaTensor,
        b: &CudaTensor,
        c: &CudaTensor,
        block_len: usize,
    ) -> Result<CudaTensor, CudaError> {
        let _t = crate::profiler::timer("ssd_scan", &self.stream);
        // Expected shapes: x (B,T,H,P), a (B,T,H), b (B,T,H,N), c (B,T,H,N).
        let xs = x.shape();
        let as_ = a.shape();
        let bs = b.shape();
        let cs = c.shape();
        if xs.len() != 4 {
            return Err(CudaError::Shape(
                "ssd_scan: x must be rank-4 (B,T,H,P)".to_string(),
            ));
        }
        let (batch, t_len, n_heads, p_dim) = (xs[0], xs[1], xs[2], xs[3]);
        if as_.len() != 3 || as_[0] != batch || as_[1] != t_len || as_[2] != n_heads {
            return Err(CudaError::Shape(format!(
                "ssd_scan: a shape {as_:?} must be (B={batch}, T={t_len}, H={n_heads})"
            )));
        }
        if bs.len() != 4 || bs[0] != batch || bs[1] != t_len || bs[2] != n_heads {
            return Err(CudaError::Shape(format!(
                "ssd_scan: b shape {bs:?} must be (B={batch}, T={t_len}, H={n_heads}, N)"
            )));
        }
        let n_dim = bs[3];
        if cs.len() != 4 || cs[0] != batch || cs[1] != t_len || cs[2] != n_heads || cs[3] != n_dim {
            return Err(CudaError::Shape(format!(
                "ssd_scan: c shape {cs:?} must be (B={batch}, T={t_len}, H={n_heads}, N={n_dim})"
            )));
        }
        // When t_len is not a positive multiple of block_len (e.g. t_len < block_len
        // or not divisible), the PTX kernel cannot run.  Fall back to the pure-Ops
        // scalar path from pm-core.  This mirrors what CandleBackend does via
        // pm_core::mamba2::ssd_scan_ops_default.
        if t_len == 0 || block_len == 0 || t_len % block_len != 0 {
            return pm_core::mamba2::ssd_scan_ops_default(self, x, a, b, c, block_len)
                .map_err(|e| CudaError::Internal(format!("ssd_scan fallback (ops_default): {e}")));
        }
        if n_dim > crate::ssd::MAX_N_DIM {
            return Err(CudaError::Shape(format!(
                "ssd_scan: n_dim={n_dim} exceeds MAX_N_DIM={}",
                crate::ssd::MAX_N_DIM
            )));
        }
        if block_len > crate::ssd::MAX_BLOCK_LEN {
            return Err(CudaError::Shape(format!(
                "ssd_scan: block_len={block_len} exceeds MAX_BLOCK_LEN={}",
                crate::ssd::MAX_BLOCK_LEN
            )));
        }

        let out_shape = vec![batch, t_len, n_heads, p_dim];
        let n_out = batch * t_len * n_heads * p_dim;
        let mut y_storage = self.stream.alloc_zeros::<f32>(n_out)?;

        crate::ssd::ssd_scan_chunked_with_context(
            &self.ctx,
            &self.stream,
            x.storage().as_ref(),
            a.storage().as_ref(),
            b.storage().as_ref(),
            c.storage().as_ref(),
            &mut y_storage,
            batch,
            t_len,
            n_heads,
            p_dim,
            n_dim,
            block_len,
        )?;

        let y = CudaTensor::new(Arc::new(y_storage), out_shape);

        // B4.3c: record SsdScan tape entry if grad tracking is enabled and at
        // least one input is tracked (has a NodeId).
        if grad_enabled()
            && (x.node_id().is_some()
                || a.node_id().is_some()
                || b.node_id().is_some()
                || c.node_id().is_some())
        {
            let nid = tape_push(
                &self.tape,
                TapeOp::SsdScan(Box::new(SsdScanData {
                    x: x.node_id(),
                    a: a.node_id(),
                    b: b.node_id(),
                    c: c.node_id(),
                    x_val: x.clone(),
                    a_val: a.clone(),
                    b_val: b.clone(),
                    c_val: c.clone(),
                    block_len,
                })),
            );
            Ok(y.with_node_id(nid))
        } else {
            Ok(y)
        }
    }

    // ---- Fused cross-entropy over a tied embedding table -------------------

    /// Pure-Ops tiled fused cross-entropy from
    /// `pm_core::loss::fused_cross_entropy_tiled`. Every op the tiling
    /// loop performs starts from an `Ops::detach`ed operand, so — per
    /// this file's `grad_enabled() && operand.node_id().is_some()` gate
    /// on every op above — **zero** entries are pushed onto `self.tape`
    /// for the whole loop; nothing is left for a later `Ops::backward`
    /// to walk or hold alive, matching pm-cuda's requirement that the
    /// shared tape only ever grow for tensors actually meant to be
    /// backpropagated through (`docs/perf-log.md` 2026-07-03's
    /// `predict_logits`/`commit` tape-leak finding is the same failure
    /// mode this sidesteps by construction).
    ///
    /// The tiling loop allocates and frees many `(tile_rows, V)`-sized
    /// buffers in quick succession (several per tile, tiles run back to
    /// back) with no intervening sync point — cudarc's allocator is
    /// stream-ordered, so `cuMemFreeAsync`s from earlier tiles queue on
    /// the stream rather than retiring immediately. This is the same
    /// mechanism `backward()`'s trailing `stream.synchronize()` above
    /// exists for (its own doc comment: "the very next forward pass
    /// issues `cuMemAllocAsync` while those frees are still pending,
    /// pushing apparent VRAM usage over the physical 12 GB limit"), just
    /// triggered here by the tile loop's own allocation churn instead of
    /// the tape clear. Sync once at the end so the *next* op (the
    /// caller's `sum_all(hidden ⊙ grad_hidden)` phantom-loss construction
    /// in `pm-train::fused_cross_entropy_injected`, or the next training
    /// step's forward) starts from a fully-reclaimed allocator state.
    /// Cost: same order as `backward`'s (~1-2 ms), negligible next to a
    /// multi-second training step.
    fn fused_cross_entropy(
        &self,
        hidden: &CudaTensor,
        table: &CudaTensor,
        targets: &CudaTensor,
        tile_rows: usize,
    ) -> Result<(CudaTensor, CudaTensor, CudaTensor), CudaError> {
        let _t = crate::profiler::timer("fused_cross_entropy", &self.stream);
        let result =
            pm_core::loss::fused_cross_entropy_tiled(self, hidden, table, targets, tile_rows);
        self.stream.synchronize().map_err(CudaError::Driver)?;
        result
    }
}

// ---- B4.3b backward helpers -----------------------------------------------

/// Transpose two dimensions on a `CudaTensor`, bypassing tape recording.
///
/// Used inside `no_grad` closures in backward; avoids recursive tape pushes.
fn transpose_inner_fn(
    bk: &CudaBackend,
    x: &CudaTensor,
    dim_a: usize,
    dim_b: usize,
) -> Result<CudaTensor, CudaError> {
    let src_shape = x.shape().to_vec();
    let rank = src_shape.len();
    if dim_a >= rank || dim_b >= rank {
        return Err(CudaError::Shape(format!(
            "transpose_inner: dim_a={dim_a} or dim_b={dim_b} out of range for rank {rank}"
        )));
    }
    if dim_a == dim_b {
        let data = bk.stream.clone_dtoh(x.storage().as_ref())?;
        let out = bk.stream.clone_htod(&data)?;
        return Ok(CudaTensor::new(Arc::new(out), src_shape));
    }
    let mut dst_shape = src_shape.clone();
    dst_shape.swap(dim_a, dim_b);
    let n: usize = src_shape.iter().product();
    let mut out = bk.stream.alloc_zeros::<f32>(n)?;
    if rank == 2 {
        super::kernels::transpose_2d_f32(
            &bk.ctx,
            &bk.stream,
            x.storage().as_ref(),
            &mut out,
            src_shape[0],
            src_shape[1],
        )?;
    } else {
        let mut src_strides = vec![1u32; rank];
        for i in (0..rank - 1).rev() {
            src_strides[i] = src_strides[i + 1] * src_shape[i + 1] as u32;
        }
        let mut out_strides = vec![1u32; rank];
        for i in (0..rank - 1).rev() {
            out_strides[i] = out_strides[i + 1] * dst_shape[i + 1] as u32;
        }
        let mut in_strides_out_order = src_strides.clone();
        in_strides_out_order.swap(dim_a, dim_b);
        super::kernels::transpose_nd_f32(
            &bk.ctx,
            &bk.stream,
            x.storage().as_ref(),
            &mut out,
            &in_strides_out_order,
            &out_strides,
        )?;
    }
    Ok(CudaTensor::new(Arc::new(out), dst_shape))
}

/// Narrow helper that bypasses tape recording. Used inside backward.
fn narrow_inner_fn(
    bk: &CudaBackend,
    x: &CudaTensor,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<CudaTensor, CudaError> {
    let src_shape = x.shape().to_vec();
    let rank = src_shape.len();
    if dim >= rank {
        return Err(CudaError::Shape(format!(
            "narrow_inner: dim {dim} out of range for rank {rank}"
        )));
    }
    let mut dst_shape = src_shape.clone();
    dst_shape[dim] = len;
    let n_out: usize = dst_shape.iter().product();
    let mut out = bk.stream.alloc_zeros::<f32>(n_out)?;
    super::kernels::narrow_copy_f32(
        &bk.ctx,
        &bk.stream,
        x.storage().as_ref(),
        &mut out,
        &src_shape,
        dim,
        start,
        len,
    )?;
    Ok(CudaTensor::new(Arc::new(out), dst_shape))
}

/// Matmul helper that bypasses tape recording. Used inside backward.
/// `PM_CUDA_MATMUL_TRACE=1` — B'.2a per-call matmul sub-op trace (read once).
fn mm_trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PM_CUDA_MATMUL_TRACE").ok().as_deref() == Some("1"))
}

fn matmul_inner_fn(
    bk: &CudaBackend,
    a: &CudaTensor,
    b: &CudaTensor,
) -> Result<CudaTensor, CudaError> {
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();
    let rank = a_shape.len();
    let m = a_shape[rank - 2];
    let k_a = a_shape[rank - 1];
    let k_b = b_shape[rank - 2];
    let n = b_shape[rank - 1];
    if k_a != k_b {
        return Err(CudaError::Shape(format!(
            "matmul_inner: inner dims mismatch {k_a} vs {k_b}"
        )));
    }
    let batch_a: usize = a_shape[..rank - 2].iter().product();
    let batch_b: usize = b_shape[..rank - 2].iter().product();
    let batch = batch_a.max(batch_b);
    let out_shape: Vec<usize> = {
        let mut s: Vec<usize> = a_shape[..rank - 2]
            .iter()
            .zip(b_shape[..rank - 2].iter())
            .map(|(&da, &db)| da.max(db))
            .collect();
        s.push(m);
        s.push(n);
        s
    };
    // Handle batch broadcast if needed. 1-vs-N (the broadcast side has all
    // leading dims == 1) is expressible as a stride-0 batched gemm and goes
    // straight to `kernels::matmul_f32` (B'.2b — this is the VJP-side twin
    // of the `Ops::matmul` fix; the host round-trip below used to dominate
    // backward wall time for PHOTON level-1 shapes).
    let needs_broadcast_a = rank > 2 && batch_a < batch;
    let needs_broadcast_b = rank > 2 && batch_b < batch;
    let stride0_broadcast_ok =
        (!needs_broadcast_a || batch_a == 1) && (!needs_broadcast_b || batch_b == 1);
    if (needs_broadcast_a || needs_broadcast_b) && !stride0_broadcast_ok {
        let a_host = bk.stream.clone_dtoh(a.storage().as_ref())?;
        let b_host = bk.stream.clone_dtoh(b.storage().as_ref())?;
        let mut a_exp = vec![0.0f32; batch * m * k_a];
        broadcast_expand_batch(
            &a_host,
            &a_shape[..rank - 2],
            &out_shape[..rank - 2],
            m * k_a,
            &mut a_exp,
        )?;
        let mut b_exp = vec![0.0f32; batch * k_a * n];
        broadcast_expand_batch(
            &b_host,
            &b_shape[..rank - 2],
            &out_shape[..rank - 2],
            k_a * n,
            &mut b_exp,
        )?;
        let a_flat = [batch, m, k_a];
        let b_flat = [batch, k_a, n];
        let a_dev = bk.stream.clone_htod(&a_exp)?;
        let b_dev = bk.stream.clone_htod(&b_exp)?;
        let out_n = batch * m * n;
        let mut out_s = bk.stream.alloc_zeros::<f32>(out_n)?;
        super::kernels::matmul_f32(&bk.cublas, &a_dev, &a_flat, &b_dev, &b_flat, &mut out_s)?;
        return Ok(CudaTensor::new(Arc::new(out_s), out_shape));
    }
    let out_n = out_shape.iter().product::<usize>();
    let mut out_s = bk.stream.alloc_zeros::<f32>(out_n)?;
    super::kernels::matmul_f32(
        &bk.cublas,
        a.storage().as_ref(),
        &a_shape,
        b.storage().as_ref(),
        &b_shape,
        &mut out_s,
    )?;
    Ok(CudaTensor::new(Arc::new(out_s), out_shape))
}

/// broadcast_as helper bypassing tape recording.
fn broadcast_as_inner_fn(
    bk: &CudaBackend,
    x: &CudaTensor,
    shape: &[usize],
) -> Result<CudaTensor, CudaError> {
    let src_shape = x.shape().to_vec();
    let rank = shape.len();
    if src_shape.len() != rank {
        return Err(CudaError::Shape(format!(
            "broadcast_as_inner: rank mismatch src={} dst={rank}",
            src_shape.len()
        )));
    }
    let n_out: usize = shape.iter().product();
    let mut out = bk.stream.alloc_zeros::<f32>(n_out)?;
    super::kernels::broadcast_copy_f32(
        &bk.ctx,
        &bk.stream,
        x.storage().as_ref(),
        &mut out,
        &src_shape,
        shape,
    )?;
    Ok(CudaTensor::new(Arc::new(out), shape.to_vec()))
}

/// `sum_to_shape`: reduce `grad` so that it matches `target_shape`.
///
/// For each dimension where `target_shape[i] < grad.shape()[i]` (i.e. it was
/// broadcast in the forward pass, meaning target had size 1), we
/// reduce-sum along that axis keeping the dim = 1.  After all reductions the
/// shape should match `target_shape` exactly (including any size-1 dims).
fn sum_to_shape(
    bk: &CudaBackend,
    grad: &CudaTensor,
    target_shape: &[usize],
) -> Result<CudaTensor, CudaError> {
    if grad.shape() == target_shape {
        return Ok(grad.clone());
    }
    let rank = target_shape.len();
    if grad.shape().len() != rank {
        return Err(CudaError::Shape(format!(
            "sum_to_shape: rank mismatch grad={} target={rank}",
            grad.shape().len()
        )));
    }
    let mut cur = grad.clone();
    // Process each dim where target is 1 but grad is > 1.
    for dim in 0..rank {
        if target_shape[dim] == 1 && cur.shape()[dim] > 1 {
            let outer: usize = cur.shape()[..dim].iter().product();
            let axis_len = cur.shape()[dim];
            let inner: usize = cur.shape()[dim + 1..].iter().product();
            let n_out = outer * inner;
            let mut out_s = bk.stream.alloc_zeros::<f32>(n_out)?;
            super::kernels::reduce_sum_dim_keepdim_f32(
                &bk.ctx,
                &bk.stream,
                cur.storage().as_ref(),
                &mut out_s,
                outer,
                axis_len,
                inner,
            )?;
            // Output shape: same as cur.shape() but with cur.shape()[dim] = 1.
            let mut new_shape = cur.shape().to_vec();
            new_shape[dim] = 1;
            cur = CudaTensor::new(Arc::new(out_s), new_shape);
        }
    }
    // At this point cur.shape() should equal target_shape.
    if cur.shape() != target_shape {
        return Err(CudaError::Shape(format!(
            "sum_to_shape: residual shape mismatch after reductions: {:?} vs target {:?}",
            cur.shape(),
            target_shape
        )));
    }
    Ok(cur)
}

/// Like `sum_to_shape` but also handles the case where `target_shape` has fewer
/// dimensions than `grad`.  Leading batch dims that were added by the B4.4a
/// rank-padding in `matmul` are summed and squeezed away.
///
/// If ranks already match, this delegates to `sum_to_shape`.
fn sum_to_shape_rank_reduce(
    bk: &CudaBackend,
    grad: &CudaTensor,
    target_shape: &[usize],
) -> Result<CudaTensor, CudaError> {
    let grad_rank = grad.shape().len();
    let target_rank = target_shape.len();
    if grad_rank == target_rank {
        return sum_to_shape(bk, grad, target_shape);
    }
    if grad_rank < target_rank {
        return Err(CudaError::Shape(format!(
            "sum_to_shape_rank_reduce: grad rank {grad_rank} < target rank {target_rank}"
        )));
    }
    // Sum over the n_extra leading dims.
    let n_extra = grad_rank - target_rank;
    // Build padded target with leading 1s so sum_to_shape can reduce them.
    let mut padded_target = vec![1usize; n_extra];
    padded_target.extend_from_slice(target_shape);
    let reduced = sum_to_shape(bk, grad, &padded_target)?;
    // Reshape from [1, ..., 1, d0, d1, ...] → [d0, d1, ...].
    // Since the leading dims are all 1, reshape is just a metadata change —
    // the underlying data buffer is unchanged and contiguous.
    let data = reduced.storage().clone();
    Ok(CudaTensor::new(data, target_shape.to_vec()))
}

/// Reduce sum along a single dimension, keepdim=true, without tape recording.
///
/// Returns a tensor with shape `[..., 1, ...]` where axis `dim` has been reduced.
fn reduce_sum_keepdim(
    bk: &CudaBackend,
    x: &CudaTensor,
    dim: usize,
    shape: &[usize],
) -> Result<CudaTensor, CudaError> {
    let outer: usize = shape[..dim].iter().product();
    let axis_len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let n_out = outer * inner;
    let mut out = bk.stream.alloc_zeros::<f32>(n_out)?;
    super::kernels::reduce_sum_dim_keepdim_f32(
        &bk.ctx,
        &bk.stream,
        x.storage().as_ref(),
        &mut out,
        outer,
        axis_len,
        inner,
    )?;
    let mut out_shape = shape.to_vec();
    out_shape[dim] = 1;
    // keepdim=true — output has the same rank as input, with shape[dim] == 1.
    Ok(CudaTensor::new(Arc::new(out), out_shape))
}

/// `PM_CUDA_NARROW_TRACE=1` — B'.3 diagnostic (read once, cached): per-call
/// shape/geometry + timing for [`scatter_to_narrow`]. Originally added to
/// measure the host round trip this function used to perform; kept (zero
/// cost when unset) as an ongoing diagnostic now that the same shapes route
/// through a single device kernel launch.
fn narrow_trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PM_CUDA_NARROW_TRACE").ok().as_deref() == Some("1"))
}

/// Scatter grad_out into a zeros tensor at the narrow window.
///
/// Creates a fresh `orig_shape` tensor, zero outside `[start, start+len)`
/// along `dim`, gathered from `grad_out` inside that window — one
/// `kernels::narrow_backward_f32` launch, no host round trip.
///
/// B'.3 (`docs/perf-log.md`): the previous implementation D2H'd `grad_out`,
/// zero-filled and wrote into a host `Vec`, then H2D'd the result. Every
/// `narrow` in `Mamba2Block::forward` slices the trailing (last) axis, which
/// made `inner = orig_shape[dim+1..].product() == 1` — degenerating the
/// host copy into `outer * len` individual 4-byte `copy_from_slice` calls
/// (measured ~1.3 ms/call average, up to ~2 ms for the largest slices).
fn scatter_to_narrow(
    bk: &CudaBackend,
    grad_out: &CudaTensor,
    dim: usize,
    start: usize,
    orig_shape: &[usize],
) -> Result<CudaTensor, CudaError> {
    let trace = narrow_trace_enabled();
    let t0 = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let n_orig: usize = orig_shape.iter().product();
    let len = grad_out.shape()[dim];
    let mut out = bk.stream.alloc_zeros::<f32>(n_orig)?;
    super::kernels::narrow_backward_f32(
        &bk.ctx,
        &bk.stream,
        grad_out.storage().as_ref(),
        &mut out,
        orig_shape,
        dim,
        start,
        len,
    )?;
    if let Some(t0v) = t0 {
        // Kernel launch is async; sync before reading elapsed() so the
        // trace reflects real device time, not just host enqueue cost.
        // Matches broadcast_binary_op's trace (backend_impl.rs).
        bk.stream.synchronize()?;
        let outer: usize = orig_shape[..dim].iter().product();
        let inner: usize = orig_shape[dim + 1..].iter().product();
        let total = t0v.elapsed();
        eprintln!(
            "[narrow-trace] DEVICE orig_shape={orig_shape:?} dim={dim} start={start} len={len} \
             outer={outer} inner={inner} n_orig={n_orig} total_us={:.0}",
            total.as_secs_f64() * 1e6,
        );
    }
    Ok(CudaTensor::new(Arc::new(out), orig_shape.to_vec()))
}

/// Backward pass for conv1d.
///
/// Computes grad_x (via col2im), grad_w (via GEMM), grad_b (via sum).
#[allow(clippy::too_many_arguments)]
fn conv1d_backward(
    bk: &CudaBackend,
    grads: &mut HashMap<NodeId, CudaTensor>,
    grad_out: &CudaTensor,
    x_nid: &Option<NodeId>,
    w_nid: &Option<NodeId>,
    bias_nid: &Option<Option<NodeId>>,
    x_val: &CudaTensor,
    w_val: &CudaTensor,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Result<(), CudaError> {
    let x_shape = x_val.shape().to_vec();
    let w_shape = w_val.shape().to_vec();
    let batch = x_shape[0];
    let c_in = x_shape[1];
    let t_in = x_shape[2];
    let c_out = w_shape[0];
    let c_in_per_group = w_shape[1];
    let k_size = w_shape[2];
    let t_out = grad_out.shape()[2];
    let c_out_per_group = c_out / groups;

    // ---- B'.2e fast path: depthwise (groups == c_in, 1 filter/channel) ----
    //
    // Mirrors the forward dispatch guard in `kernels::conv1d_f32` exactly, so
    // a given conv1d call always takes the GPU path on both forward and
    // backward together. Replaces the O(groups) host GEMM loop below with
    // the two GPU kernels landed (but never wired) in commit 57d3719 —
    // see `conv1d_backward_depthwise_gpu` for why that wiring was deferred.
    if groups == c_in && c_in_per_group == 1 && k_size <= 128 {
        return conv1d_backward_depthwise_gpu(
            bk, grads, grad_out, x_nid, w_nid, bias_nid, x_val, w_val, &w_shape, batch, c_in, t_in,
            t_out, k_size, stride, padding,
        );
    }

    // grad_bias: sum over batch and T.
    if let Some(Some(b_nid)) = bias_nid {
        no_grad(|| -> Result<(), CudaError> {
            // grad_out shape: (B, C_out, T_out)
            // grad_b[co] = sum_{b,t} grad_out[b, co, t]
            let go_host = bk.stream.clone_dtoh(grad_out.storage().as_ref())?;
            let mut gb_host = vec![0.0f32; c_out];
            for b_idx in 0..batch {
                for co in 0..c_out {
                    for t in 0..t_out {
                        gb_host[co] += go_host[b_idx * c_out * t_out + co * t_out + t];
                    }
                }
            }
            let gb_dev = bk.stream.clone_htod(&gb_host)?;
            let gb_t = CudaTensor::new(Arc::new(gb_dev), vec![c_out]);
            accumulate(bk, grads, *b_nid, gb_t)
        })?;
    }

    // Materialise im2col buffer for x: (B * T_out, C_in * K)
    let im2col_rows = batch * t_out;
    let im2col_cols = c_in * k_size;
    let col_n = im2col_rows * im2col_cols;
    let mut col_buf = bk.stream.alloc_zeros::<f32>(col_n)?;
    super::kernels::im2col_f32(
        &bk.ctx,
        &bk.stream,
        x_val.storage().as_ref(),
        &mut col_buf,
        batch,
        c_in,
        t_in,
        t_out,
        k_size,
        stride,
        padding,
    )?;

    let col_host = bk.stream.clone_dtoh(&col_buf)?;
    let w_host = bk.stream.clone_dtoh(w_val.storage().as_ref())?;
    let go_host = bk.stream.clone_dtoh(grad_out.storage().as_ref())?;

    // Per group: process grad_w and grad_x via host GEMM + col2im.
    let mut gx_host = vec![0.0f32; batch * c_in * t_in];
    let mut gw_host = vec![0.0f32; c_out * c_in_per_group * k_size];

    for g_idx in 0..groups {
        let c_in_g_start = g_idx * c_in_per_group;
        let c_in_g_k = c_in_per_group * k_size;
        let c_out_g_start = g_idx * c_out_per_group;

        // Extract col_g: (B * T_out, C_in_per_group * K)
        let col_offset = c_in_g_start * k_size;
        let mut col_g = vec![0.0f32; im2col_rows * c_in_g_k];
        for r in 0..im2col_rows {
            let src_base = r * im2col_cols + col_offset;
            let dst_base = r * c_in_g_k;
            col_g[dst_base..dst_base + c_in_g_k]
                .copy_from_slice(&col_host[src_base..src_base + c_in_g_k]);
        }

        // Extract grad_out_g: (C_out_per_group, B * T_out) from (B, C_out, T_out)
        // → reshape to (C_out_per_group, B * T_out).
        let mut go_g = vec![0.0f32; c_out_per_group * im2col_rows];
        for co_g in 0..c_out_per_group {
            let co = c_out_g_start + co_g;
            for bt in 0..im2col_rows {
                let b_idx = bt / t_out;
                let t = bt % t_out;
                go_g[co_g * im2col_rows + bt] = go_host[b_idx * c_out * t_out + co * t_out + t];
            }
        }

        if w_nid.is_some() {
            // grad_w_g = grad_out_g @ col_g  (C_out/g × B*T_out) × (B*T_out × C_in/g*K)
            //          = (C_out/g × C_in/g*K)
            let gw_offset = c_out_g_start * c_in_g_k;
            for co_g in 0..c_out_per_group {
                for ci_k in 0..c_in_g_k {
                    let mut acc = 0.0f32;
                    for bt in 0..im2col_rows {
                        acc += go_g[co_g * im2col_rows + bt] * col_g[bt * c_in_g_k + ci_k];
                    }
                    gw_host[gw_offset + co_g * c_in_g_k + ci_k] += acc;
                }
            }
        }

        if x_nid.is_some() {
            // grad_col_g = w_g^T @ grad_out_g  (C_in/g*K × C_out/g) × (C_out/g × B*T_out)
            //            = (C_in/g*K × B*T_out) → reshape to (B*T_out, C_in/g*K)
            let w_offset = c_out_g_start * c_in_g_k;
            let w_g = &w_host[w_offset..w_offset + c_out_per_group * c_in_g_k];
            let mut grad_col_g = vec![0.0f32; c_in_g_k * im2col_rows];
            // w_g is (C_out/g, C_in/g*K) → w_g^T is (C_in/g*K, C_out/g)
            for ci_k in 0..c_in_g_k {
                for bt in 0..im2col_rows {
                    let mut acc = 0.0f32;
                    for co_g in 0..c_out_per_group {
                        acc += w_g[co_g * c_in_g_k + ci_k] * go_g[co_g * im2col_rows + bt];
                    }
                    grad_col_g[ci_k * im2col_rows + bt] = acc;
                }
            }

            // col2im: scatter grad_col_g back to (B, C_in_per_group, T_in) portion.
            // grad_col_g layout: (C_in/g*K, B*T_out) → need (B*T_out, C_in/g*K)
            let mut grad_col_g_t = vec![0.0f32; im2col_rows * c_in_g_k];
            for ci_k in 0..c_in_g_k {
                for bt in 0..im2col_rows {
                    grad_col_g_t[bt * c_in_g_k + ci_k] = grad_col_g[ci_k * im2col_rows + bt];
                }
            }
            // Upload and use col2im kernel to scatter into full gx.
            // But col2im operates on (B, C_in, T_in) for all channels.
            // We use the host scatter directly here for correctness.
            for bt in 0..im2col_rows {
                let b_idx = bt / t_out;
                let t_o = bt % t_out;
                for c_g in 0..c_in_per_group {
                    for k in 0..k_size {
                        let t_in_signed = (t_o * stride) as i32 + k as i32 - padding as i32;
                        if t_in_signed < 0 || (t_in_signed as usize) >= t_in {
                            continue;
                        }
                        let t_i = t_in_signed as usize;
                        let c_abs = c_in_g_start + c_g;
                        let grad_col_idx = bt * c_in_g_k + c_g * k_size + k;
                        let gx_idx = b_idx * c_in * t_in + c_abs * t_in + t_i;
                        gx_host[gx_idx] += grad_col_g_t[grad_col_idx];
                    }
                }
            }
        }
    }

    if let Some(w_n) = w_nid {
        no_grad(|| -> Result<(), CudaError> {
            let gw_dev = bk.stream.clone_htod(&gw_host)?;
            let gw_t = CudaTensor::new(Arc::new(gw_dev), w_shape.clone());
            accumulate(bk, grads, *w_n, gw_t)
        })?;
    }
    if let Some(x_n) = x_nid {
        no_grad(|| -> Result<(), CudaError> {
            let gx_dev = bk.stream.clone_htod(&gx_host)?;
            let gx_t = CudaTensor::new(Arc::new(gx_dev), x_shape.clone());
            accumulate(bk, grads, *x_n, gx_t)
        })?;
    }
    Ok(())
}

/// Depthwise conv1d backward — fully device-side (B'.2e).
///
/// `channels` = `C_in` = `C_out` = `groups` (depthwise: 1 filter/channel,
/// checked by the caller's guard). Dispatches straight to the
/// `depthwise_conv1d_bwd_{x,w}_f32` PTX kernels landed alongside the
/// forward kernel in commit 57d3719 but never wired in: that attempt's
/// 52-line dispatcher hung indefinitely at production shape (8 h of 100%
/// GPU util before kill). The kernels themselves were already reviewed as
/// structurally sound (bounded loops, `syncthreads()` outside any
/// per-thread-divergent branch — verified again here); this dispatcher is
/// a from-scratch rewrite with **no host-side loops** (just two bounded
/// kernel launches + a bias reduction reusing `reduce_sum_keepdim`), which
/// removes the most likely source of an indefinite hang. Validated
/// incrementally (unit scale, then production scale) before landing — see
/// `conv1d_depthwise_grad` (`tests/grad_algebra.rs`) and
/// `conv1d_depthwise_bwd_prod_shape` (`tests/conv1d_depthwise_bwd_gpu.rs`).
#[allow(clippy::too_many_arguments)]
fn conv1d_backward_depthwise_gpu(
    bk: &CudaBackend,
    grads: &mut HashMap<NodeId, CudaTensor>,
    grad_out: &CudaTensor,
    x_nid: &Option<NodeId>,
    w_nid: &Option<NodeId>,
    bias_nid: &Option<Option<NodeId>>,
    x_val: &CudaTensor,
    w_val: &CudaTensor,
    w_shape: &[usize],
    batch: usize,
    channels: usize,
    t_in: usize,
    t_out: usize,
    k_size: usize,
    stride: usize,
    padding: usize,
) -> Result<(), CudaError> {
    // grad_bias: sum over (batch, T_out), keeping channels. No single
    // kernel reduces two axes at once, so this composes the existing
    // single-axis `reduce_sum_keepdim` twice (over T_out, then batch) —
    // both passes are device-side, no D2H/H2D round trip.
    if let Some(Some(b_nid)) = bias_nid {
        no_grad(|| -> Result<(), CudaError> {
            let over_t = reduce_sum_keepdim(bk, grad_out, 2, grad_out.shape())?; // (B, C, 1)
            let over_bt = reduce_sum_keepdim(bk, &over_t, 0, over_t.shape())?; // (1, C, 1)
            let gb_t = CudaTensor::new(over_bt.storage().clone(), vec![channels]);
            accumulate(bk, grads, *b_nid, gb_t)
        })?;
    }

    if let Some(x_n) = x_nid {
        no_grad(|| -> Result<(), CudaError> {
            let mut gx_dev = bk.stream.alloc_zeros::<f32>(batch * channels * t_in)?;
            super::kernels::conv1d_depthwise_bwd_x_gpu(
                &bk.ctx,
                &bk.stream,
                grad_out.storage().as_ref(),
                w_val.storage().as_ref(),
                &mut gx_dev,
                batch,
                channels,
                t_in,
                t_out,
                k_size,
                stride,
                padding,
            )?;
            let gx_t = CudaTensor::new(Arc::new(gx_dev), vec![batch, channels, t_in]);
            accumulate(bk, grads, *x_n, gx_t)
        })?;
    }

    if let Some(w_n) = w_nid {
        no_grad(|| -> Result<(), CudaError> {
            let mut gw_dev = bk.stream.alloc_zeros::<f32>(channels * k_size)?;
            super::kernels::conv1d_depthwise_bwd_w_gpu(
                &bk.ctx,
                &bk.stream,
                grad_out.storage().as_ref(),
                x_val.storage().as_ref(),
                &mut gw_dev,
                batch,
                channels,
                t_in,
                t_out,
                k_size,
                stride,
                padding,
            )?;
            let gw_t = CudaTensor::new(Arc::new(gw_dev), w_shape.to_vec());
            accumulate(bk, grads, *w_n, gw_t)
        })?;
    }
    Ok(())
}

impl CudaBackend {
    /// Transpose helper used internally in backward (no tape recording).
    pub(crate) fn transpose_inner(
        &self,
        x: &CudaTensor,
        dim_a: usize,
        dim_b: usize,
    ) -> Result<CudaTensor, CudaError> {
        transpose_inner_fn(self, x, dim_a, dim_b)
    }

    /// Narrow helper bypassing tape.
    pub(crate) fn narrow_inner(
        &self,
        x: &CudaTensor,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<CudaTensor, CudaError> {
        narrow_inner_fn(self, x, dim, start, len)
    }

    /// Matmul helper bypassing tape.
    pub(crate) fn matmul_inner(
        &self,
        a: &CudaTensor,
        b: &CudaTensor,
    ) -> Result<CudaTensor, CudaError> {
        matmul_inner_fn(self, a, b)
    }

    /// broadcast_as helper bypassing tape.
    pub(crate) fn broadcast_as_inner(
        &self,
        x: &CudaTensor,
        shape: &[usize],
    ) -> Result<CudaTensor, CudaError> {
        broadcast_as_inner_fn(self, x, shape)
    }
}

// ---- B4.3c: SsdScan VJP via pure-Ops recompute ----------------------------

// B4.4d — RAII guard for sub-tape cleanup. Placed here so it is local to
// the SsdScan VJP section of this file.

/// Truncates the shared autograd tape back to `start_len` on drop.
///
/// Installed at the top of `ssd_scan_backward_vjp` so that any error
/// path through `apply_vjp` also truncates, preventing GPU tensor leaks
/// from sub-tape entries into the parent graph.
struct SubTapeGuard<'a> {
    tape: &'a std::sync::Mutex<super::tape::Tape>,
    start_len: usize,
}

impl<'a> Drop for SubTapeGuard<'a> {
    fn drop(&mut self) {
        if let Ok(mut t) = self.tape.lock() {
            t.entries.truncate(self.start_len);
        }
        // If the lock is poisoned the process is already going down;
        // truncation is best-effort.
    }
}

/// VJP for a single `SsdScan` forward op.
///
/// This function is called from the main backward loop when a `SsdScan`
/// entry is encountered on the tape.  The strategy (B4.3c Option Y):
///
/// 1. Record the current tape length as `sub_start`.
/// 2. Register `x_val`, `a_val`, `b_val`, `c_val` as fresh leaf tensors
///    (each gets a new `ParamId` and a new `NodeId` via a `Leaf` tape entry).
/// 3. Call `ssd_scan_ops_default` with grad enabled — this appends many
///    primitive-op tape entries starting at `sub_start + 4` (after the 4
///    leaf entries).
/// 4. Compute `phantom = sum_all(y_recomp * grad_out)` (two more tape entries).
/// 5. Run a sub-walk from `phantom`'s `NodeId` back to (but not including)
///    `sub_start`, collecting `NodeId → gradient` and depositing leaf-node
///    gradients into a `ParamId → gradient` map.
/// 6. Extract `grad_x`, `grad_a`, `grad_b`, `grad_c` from the sub-map and
///    accumulate into the parent walk's `grads` map using the *original*
///    NodeIds (`x_in`, `a_in`, …).
/// 7. Truncate the tape back to `sub_start` (cleanup).
///
/// # Note on `grad_out` tracking
/// `grad_out` must NOT push tape entries when used in step 4 (it is a
/// gradient value, not a forward tensor). We detach it by stripping its
/// `NodeId` before passing it to `mul`.
#[allow(clippy::too_many_arguments)]
fn ssd_scan_backward_vjp(
    bk: &CudaBackend,
    parent_grads: &mut HashMap<NodeId, CudaTensor>,
    grad_out: CudaTensor,
    x_in: Option<NodeId>,
    a_in: Option<NodeId>,
    b_in: Option<NodeId>,
    c_in: Option<NodeId>,
    x_val: &CudaTensor,
    a_val: &CudaTensor,
    b_val: &CudaTensor,
    c_val: &CudaTensor,
    block_len: usize,
) -> Result<(), CudaError> {
    use super::param::ParamId;

    // ---- Step 1: record sub-tape start index ----
    let sub_start = bk
        .tape
        .lock()
        .map_err(|_| CudaError::Internal("tape lock poisoned (ssd_scan_backward start)".into()))?
        .entries
        .len();

    // B4.4d — install RAII guard so any early error return through
    // apply_vjp also truncates the sub-tape entries, preventing GPU
    // tensor leaks into the parent graph.
    let sub_tape_guard = SubTapeGuard {
        tape: &bk.tape,
        start_len: sub_start,
    };

    // ---- Step 2: register fresh leaf tensors ----
    // Each gets a new ParamId and a Leaf tape entry at sub_start + 0..3.
    let pid_x = bk.alloc_param_id();
    let pid_a = bk.alloc_param_id();
    let pid_b = bk.alloc_param_id();
    let pid_c = bk.alloc_param_id();

    // Strip node_id from the saved values (they are forward tensors, not
    // leaves in the sub-graph; we register fresh leaves instead).
    // L2: inline the dead _detached aliases directly into register_leaf.
    let x_leaf = register_leaf(&bk.tape, x_val.clone(), pid_x);
    let a_leaf = register_leaf(&bk.tape, a_val.clone(), pid_a);
    let b_leaf = register_leaf(&bk.tape, b_val.clone(), pid_b);
    let c_leaf = register_leaf(&bk.tape, c_val.clone(), pid_c);

    // ---- Step 3: pure-Ops recompute (grad tracking ON) ----
    // `ssd_scan_ops_default` calls cumsum, transpose, reshape, sub,
    // broadcast_as, add, exp, matmul, mul — all tape-recorded.
    let y_recomp =
        pm_core::mamba2::ssd_scan_ops_default(bk, &x_leaf, &a_leaf, &b_leaf, &c_leaf, block_len)?;

    // ---- Step 4: phantom loss = sum_all(y_recomp * grad_out_detached) ----
    // Detach grad_out so the mul/sum don't try to back-prop through the
    // upstream grad computation.
    let grad_detached = {
        let mut g = grad_out.clone();
        g.node_id = None; // detach: not part of the sub-graph
        g.origin = None; // defensive: grad values are never Leaf-origin, but keep node_id/origin in lockstep
        g
    };
    let weighted = pm_core::Ops::mul(bk, &y_recomp, &grad_detached)?;
    let phantom = pm_core::Ops::sum_all(bk, &weighted)?;

    let phantom_nid = match phantom.node_id() {
        Some(nid) => nid,
        None => {
            // Nothing tracked — SubTapeGuard will truncate sub-entries on drop.
            return Ok(());
        }
    };

    // ---- Step 5: sub-walk ----
    // Same generation as `sub_start`/`phantom_nid` — this sub-walk happens
    // entirely within one outer `backward()` call, so nothing bumps the
    // tape's generation between here and there (`SubTapeGuard` only
    // truncates `entries`, it never calls `Tape::clear`).
    let (n_sub_entries, sub_generation) = {
        let t = bk.tape.lock().map_err(|_| {
            CudaError::Internal("tape lock poisoned (ssd_scan_backward walk)".into())
        })?;
        (t.entries.len(), t.generation)
    };

    // We walk from phantom_nid back to (but not including) sub_start.
    // Gradients are accumulated in a local map keyed by NodeId.
    // Leaf nodes deposit into a ParamId map.
    let mut sub_grads: HashMap<NodeId, CudaTensor> = HashMap::new();
    let mut sub_param_grads: HashMap<ParamId, CudaTensor> = HashMap::new();

    // Seed: d(phantom)/d(phantom) = 1.0.
    let seed = no_grad(|| {
        let host = vec![1.0f32];
        bk.stream
            .clone_htod(&host)
            .map(|s| CudaTensor::new(Arc::new(s), phantom.shape().to_vec()))
    })?;
    sub_grads.insert(phantom_nid, seed);

    // Walk the sub-entries in reverse.
    for sub_idx in (sub_start..n_sub_entries).rev() {
        let sub_nid = NodeId {
            generation: sub_generation,
            index: sub_idx as u32,
        };
        let sub_grad_out = match sub_grads.remove(&sub_nid) {
            Some(g) => g,
            None => continue,
        };

        // Snapshot the sub-entry under lock.
        let sub_op = {
            let tape = bk
                .tape
                .lock()
                .map_err(|_| CudaError::Internal("tape lock poisoned (sub-walk)".into()))?;
            match tape.get(sub_nid) {
                TapeOp::Leaf { param_id } => BackwardOp::Leaf {
                    param_id: *param_id,
                },
                TapeOp::Add {
                    lhs,
                    rhs,
                    lhs_shape,
                    rhs_shape,
                } => BackwardOp::Add {
                    lhs: *lhs,
                    rhs: *rhs,
                    lhs_shape: lhs_shape.clone(),
                    rhs_shape: rhs_shape.clone(),
                },
                TapeOp::Sub {
                    lhs,
                    rhs,
                    lhs_shape,
                    rhs_shape,
                } => BackwardOp::Sub {
                    lhs: *lhs,
                    rhs: *rhs,
                    lhs_shape: lhs_shape.clone(),
                    rhs_shape: rhs_shape.clone(),
                },
                TapeOp::Mul {
                    lhs,
                    rhs,
                    lhs_val,
                    rhs_val,
                } => BackwardOp::Mul {
                    lhs: *lhs,
                    rhs: *rhs,
                    lhs_val: lhs_val.clone(),
                    rhs_val: rhs_val.clone(),
                },
                TapeOp::Neg { input } => BackwardOp::Neg { input: *input },
                TapeOp::Reshape { input, orig_shape } => BackwardOp::Reshape {
                    input: *input,
                    orig_shape: orig_shape.clone(),
                },
                TapeOp::MulScalar { input, scale } => BackwardOp::MulScalar {
                    input: *input,
                    scale: *scale,
                },
                TapeOp::AddScalar { input } => BackwardOp::AddScalar { input: *input },
                TapeOp::Sqrt { input, out_val } => BackwardOp::Sqrt {
                    input: *input,
                    out_val: out_val.clone(),
                },
                TapeOp::Div {
                    lhs,
                    rhs,
                    rhs_val,
                    out_val,
                } => BackwardOp::Div {
                    lhs: *lhs,
                    rhs: *rhs,
                    rhs_val: rhs_val.clone(),
                    out_val: out_val.clone(),
                },
                TapeOp::Exp { input, out_val } => BackwardOp::Exp {
                    input: *input,
                    out_val: out_val.clone(),
                },
                TapeOp::Silu { input, input_val } => BackwardOp::Silu {
                    input: *input,
                    input_val: input_val.clone(),
                },
                TapeOp::Sigmoid { input, out_val } => BackwardOp::Sigmoid {
                    input: *input,
                    out_val: out_val.clone(),
                },
                TapeOp::Softplus { input, input_val } => BackwardOp::Softplus {
                    input: *input,
                    input_val: input_val.clone(),
                },
                TapeOp::MatMul { a, b, a_val, b_val } => BackwardOp::MatMul {
                    a: *a,
                    b: *b,
                    a_val: a_val.clone(),
                    b_val: b_val.clone(),
                },
                TapeOp::Transpose {
                    input,
                    dim_a,
                    dim_b,
                } => BackwardOp::Transpose {
                    input: *input,
                    dim_a: *dim_a,
                    dim_b: *dim_b,
                },
                TapeOp::Narrow {
                    input,
                    dim,
                    start,
                    len: _,
                    orig_shape,
                } => BackwardOp::Narrow {
                    input: *input,
                    dim: *dim,
                    start: *start,
                    orig_shape: orig_shape.clone(),
                },
                TapeOp::BroadcastAs {
                    input,
                    orig_shape,
                    target_shape: _,
                } => BackwardOp::BroadcastAs {
                    input: *input,
                    orig_shape: orig_shape.clone(),
                },
                TapeOp::Concat {
                    inputs,
                    dim,
                    input_shapes,
                } => BackwardOp::Concat {
                    inputs: inputs.clone(),
                    dim: *dim,
                    input_shapes: input_shapes.clone(),
                },
                TapeOp::Embedding { .. }
                | TapeOp::Gather { .. }
                | TapeOp::Conv1d { .. }
                | TapeOp::LogSoftmax { .. }
                | TapeOp::RmsNorm { .. } => {
                    // These ops shouldn't appear in a ssd_scan_ops_default sub-graph.
                    // Skip; their inputs won't get gradients.
                    continue;
                }
                TapeOp::MeanAll {
                    input,
                    numel,
                    input_shape,
                } => BackwardOp::MeanAll {
                    input: *input,
                    numel: *numel,
                    input_shape: input_shape.clone(),
                },
                TapeOp::SumAll { input, input_shape } => BackwardOp::SumAll {
                    input: *input,
                    input_shape: input_shape.clone(),
                },
                TapeOp::Cumsum {
                    input,
                    dim,
                    input_shape,
                } => BackwardOp::Cumsum {
                    input: *input,
                    dim: *dim,
                    input_shape: input_shape.clone(),
                },
                TapeOp::SsdScan(_) => {
                    // ssd_scan_ops_default never calls ssd_scan; if we see one
                    // here it would be a bug.  Skip gracefully.
                    continue;
                }
            }
        };

        // H2: apply_vjp is shared with the main backward loop.
        apply_vjp(
            bk,
            sub_op,
            sub_grad_out,
            &mut sub_grads,
            &mut sub_param_grads,
        )?;
    }

    // ---- Step 6: accumulate x/a/b/c grads into the parent graph ----
    macro_rules! maybe_accumulate {
        ($pid:expr, $in_nid:expr) => {
            if let Some(orig_nid) = $in_nid {
                if let Some(g) = sub_param_grads.remove(&$pid) {
                    accumulate(bk, parent_grads, orig_nid, g)?;
                }
            }
        };
    }
    maybe_accumulate!(pid_x, x_in);
    maybe_accumulate!(pid_a, a_in);
    maybe_accumulate!(pid_b, b_in);
    maybe_accumulate!(pid_c, c_in);

    // ---- Step 7: truncate tape back to sub_start ----
    // B4.4d: The SubTapeGuard installed at the top of this function handles
    // truncation on both the happy path (drop at end of scope) and any error
    // path (drop during stack unwind from an earlier `?`).  We drop it
    // explicitly here so the intent is clear and the lock is released before
    // returning.
    drop(sub_tape_guard);

    Ok(())
}

/// Apply a single VJP rule for one backward op, writing gradients into
/// `grads` (node → tensor) and `param_grads` (param id → tensor).
///
/// H2 (B4.3c review fix): this function is called both from the main
/// `backward()` loop and from the sub-walk inside `ssd_scan_backward_vjp`,
/// replacing the former `apply_vjp_sub` duplication (~385 lines removed).
///
/// All VJP arithmetic uses the PTX-kernel paths (same quality as the
/// original main backward).  Ops that cannot appear in an
/// `ssd_scan_ops_default` sub-graph (`Conv1d`, `Embedding`, `Gather`,
/// `LogSoftmax`, `RmsNorm`) are handled correctly by the main backward but
/// silently skipped in the sub-walk — the sub-walk never encounters those
/// variants because `ssd_scan_ops_default` only calls primitive arithmetic ops.
fn apply_vjp(
    bk: &CudaBackend,
    op: BackwardOp,
    grad_out: CudaTensor,
    grads: &mut HashMap<NodeId, CudaTensor>,
    param_grads: &mut HashMap<super::param::ParamId, CudaTensor>,
) -> Result<(), CudaError> {
    // Phase B'.1b: per-VJP breakdown. Shares the ordinary nesting-depth
    // counter with every `Ops` method's own timer (unlike `backward_total`,
    // whose dedicated guard deliberately does not) — so e.g. `vjp:SsdScan`'s
    // call into `ssd_scan_backward_vjp`, which recomputes via
    // `ssd_scan_ops_default`'s primitive `Ops` calls and recurses into
    // `apply_vjp` for its own sub-walk, is folded into this one bucket
    // rather than double-counted across nesting levels.
    let _t = crate::profiler::timer(backward_op_name(&op), &bk.stream);
    match op {
        // Leaf: accumulate into param_grads.
        BackwardOp::Leaf { param_id } => match param_grads.remove(&param_id) {
            None => {
                param_grads.insert(param_id, grad_out);
            }
            Some(existing) => {
                let sum = no_grad(|| bk.add_inner(&existing, &grad_out))?;
                param_grads.insert(param_id, sum);
            }
        },

        // Add: d/dlhs = sum_to(g, lhs_shape), d/drhs = sum_to(g, rhs_shape).
        BackwardOp::Add {
            lhs,
            rhs,
            lhs_shape,
            rhs_shape,
        } => {
            // Use rank-reduce variant to handle bias-style adds where
            // one operand has fewer dims (e.g. bias shape [D] added to [B,T,D]).
            if let Some(l) = lhs {
                let g_l = sum_to_shape_rank_reduce(bk, &grad_out, &lhs_shape)?;
                accumulate(bk, grads, l, g_l)?;
            }
            if let Some(r) = rhs {
                let g_r = sum_to_shape_rank_reduce(bk, &grad_out, &rhs_shape)?;
                accumulate(bk, grads, r, g_r)?;
            }
        }

        // Sub: d/dlhs = sum_to(g, lhs_shape), d/drhs = -sum_to(g, rhs_shape).
        BackwardOp::Sub {
            lhs,
            rhs,
            lhs_shape,
            rhs_shape,
        } => {
            // Use rank-reduce variant for the same reason as Add above.
            if let Some(l) = lhs {
                let g_l = sum_to_shape_rank_reduce(bk, &grad_out, &lhs_shape)?;
                accumulate(bk, grads, l, g_l)?;
            }
            if let Some(r) = rhs {
                let g_r = sum_to_shape_rank_reduce(bk, &grad_out, &rhs_shape)?;
                let neg_g = no_grad(|| bk.neg_inner(&g_r))?;
                accumulate(bk, grads, r, neg_g)?;
            }
        }

        // Mul: d/dlhs = g * rhs_val, d/drhs = g * lhs_val (broadcast-aware).
        BackwardOp::Mul {
            lhs,
            rhs,
            lhs_val,
            rhs_val,
        } => {
            if let Some(l) = lhs {
                let lhs_orig_shape = lhs_val.shape().to_vec();
                let g_l = no_grad(|| {
                    if grad_out.shape() == rhs_val.shape() {
                        bk.mul_inner(&grad_out, &rhs_val)
                    } else {
                        // B'.3: device broadcast kernel, not the host round trip.
                        bk.broadcast_mul_dev(&grad_out, &rhs_val)
                    }
                })?;
                // Use rank-reduce variant for broadcast ops where one operand
                // has fewer dims (e.g. bias [D] added/multiplied to [B,T,D]).
                let g_l = sum_to_shape_rank_reduce(bk, &g_l, &lhs_orig_shape)?;
                accumulate(bk, grads, l, g_l)?;
            }
            if let Some(r) = rhs {
                let rhs_orig_shape = rhs_val.shape().to_vec();
                let g_r = no_grad(|| {
                    if grad_out.shape() == lhs_val.shape() {
                        bk.mul_inner(&grad_out, &lhs_val)
                    } else {
                        // B'.3: device broadcast kernel, not the host round trip.
                        bk.broadcast_mul_dev(&grad_out, &lhs_val)
                    }
                })?;
                let g_r = sum_to_shape_rank_reduce(bk, &g_r, &rhs_orig_shape)?;
                accumulate(bk, grads, r, g_r)?;
            }
        }

        // Neg: d/dinput = -g
        BackwardOp::Neg { input } => {
            if let Some(i) = input {
                let neg_g = no_grad(|| bk.neg_inner(&grad_out))?;
                accumulate(bk, grads, i, neg_g)?;
            }
        }

        // Reshape: d/dinput = reshape(g, orig_shape)
        BackwardOp::Reshape { input, orig_shape } => {
            if let Some(i) = input {
                let g_reshaped = no_grad(|| grad_out.with_shape(orig_shape.clone()));
                accumulate(bk, grads, i, g_reshaped)?;
            }
        }

        // MulScalar: d/dinput = g * scale  (PTX path)
        BackwardOp::MulScalar { input, scale } => {
            if let Some(i) = input {
                let g_scaled = no_grad(|| -> Result<CudaTensor, CudaError> {
                    let n = grad_out.numel();
                    let mut out = bk.stream.alloc_zeros::<f32>(n)?;
                    super::kernels::mul_scalar_f32(
                        &bk.ctx,
                        &bk.stream,
                        grad_out.storage().as_ref(),
                        scale,
                        &mut out,
                    )?;
                    Ok(CudaTensor::new(Arc::new(out), grad_out.shape().to_vec()))
                })?;
                accumulate(bk, grads, i, g_scaled)?;
            }
        }

        // AddScalar: d/dinput = g
        BackwardOp::AddScalar { input } => {
            if let Some(i) = input {
                accumulate(bk, grads, i, grad_out)?;
            }
        }

        // Sqrt: d/dx = g / (2 * out)  (PTX path)
        BackwardOp::Sqrt { input, out_val } => {
            if let Some(i) = input {
                let g_i = no_grad(|| {
                    let n = grad_out.numel();
                    let mut inv2out = bk.stream.alloc_zeros::<f32>(n)?;
                    super::kernels::mul_scalar_f32(
                        &bk.ctx,
                        &bk.stream,
                        out_val.storage().as_ref(),
                        2.0,
                        &mut inv2out,
                    )?;
                    let inv2out_t = CudaTensor::new(Arc::new(inv2out), out_val.shape().to_vec());
                    let ones_host = vec![1.0f32; n];
                    let ones_dev = bk.stream.clone_htod(&ones_host)?;
                    let ones_t = CudaTensor::new(Arc::new(ones_dev), out_val.shape().to_vec());
                    let mut denom = bk.stream.alloc_zeros::<f32>(n)?;
                    super::kernels::div_f32(
                        &bk.ctx,
                        &bk.stream,
                        ones_t.storage().as_ref(),
                        inv2out_t.storage().as_ref(),
                        &mut denom,
                    )?;
                    let denom_t = CudaTensor::new(Arc::new(denom), out_val.shape().to_vec());
                    bk.mul_inner(&grad_out, &denom_t)
                })?;
                accumulate(bk, grads, i, g_i)?;
            }
        }

        // Div: d/dlhs = g / rhs, d/drhs = -g * out / rhs  (PTX path)
        BackwardOp::Div {
            lhs,
            rhs,
            rhs_val,
            out_val,
        } => {
            if let Some(l) = lhs {
                let g_l = no_grad(|| -> Result<CudaTensor, CudaError> {
                    let n = grad_out.numel();
                    let mut out = bk.stream.alloc_zeros::<f32>(n)?;
                    super::kernels::div_f32(
                        &bk.ctx,
                        &bk.stream,
                        grad_out.storage().as_ref(),
                        rhs_val.storage().as_ref(),
                        &mut out,
                    )?;
                    Ok(CudaTensor::new(Arc::new(out), grad_out.shape().to_vec()))
                })?;
                accumulate(bk, grads, l, g_l)?;
            }
            if let Some(r) = rhs {
                let g_r = no_grad(|| -> Result<CudaTensor, CudaError> {
                    let neg_g = bk.neg_inner(&grad_out)?;
                    let neg_g_out = bk.mul_inner(&neg_g, &out_val)?;
                    let n = neg_g_out.numel();
                    let mut out = bk.stream.alloc_zeros::<f32>(n)?;
                    super::kernels::div_f32(
                        &bk.ctx,
                        &bk.stream,
                        neg_g_out.storage().as_ref(),
                        rhs_val.storage().as_ref(),
                        &mut out,
                    )?;
                    Ok(CudaTensor::new(Arc::new(out), neg_g_out.shape().to_vec()))
                })?;
                accumulate(bk, grads, r, g_r)?;
            }
        }

        // Exp: d/dx = g * out  (PTX path)
        BackwardOp::Exp { input, out_val } => {
            if let Some(i) = input {
                let g_i = no_grad(|| bk.mul_inner(&grad_out, &out_val))?;
                accumulate(bk, grads, i, g_i)?;
            }
        }

        // Silu: d/dx = g * (sigma + x * sigma * (1 - sigma))  (PTX path)
        BackwardOp::Silu { input, input_val } => {
            if let Some(i) = input {
                let g_i = no_grad(|| {
                    let n = input_val.numel();
                    let mut sigma_buf = bk.stream.alloc_zeros::<f32>(n)?;
                    super::kernels::sigmoid_f32(
                        &bk.ctx,
                        &bk.stream,
                        input_val.storage().as_ref(),
                        &mut sigma_buf,
                    )?;
                    let sigma = CudaTensor::new(Arc::new(sigma_buf), input_val.shape().to_vec());
                    let ones_host = vec![1.0f32; n];
                    let ones_dev = bk.stream.clone_htod(&ones_host)?;
                    let ones = CudaTensor::new(Arc::new(ones_dev), sigma.shape().to_vec());
                    let one_minus_sigma = bk.sub_inner(&ones, &sigma)?;
                    let x_sigma = bk.mul_inner(&input_val, &sigma)?;
                    let x_sigma_oms = bk.mul_inner(&x_sigma, &one_minus_sigma)?;
                    let dsilu_dx = bk.add_inner(&sigma, &x_sigma_oms)?;
                    bk.mul_inner(&grad_out, &dsilu_dx)
                })?;
                accumulate(bk, grads, i, g_i)?;
            }
        }

        // Sigmoid: d/dx = g * out * (1 - out)  (PTX path)
        BackwardOp::Sigmoid { input, out_val } => {
            if let Some(i) = input {
                let g_i = no_grad(|| {
                    let n = out_val.numel();
                    let ones_host = vec![1.0f32; n];
                    let ones_dev = bk.stream.clone_htod(&ones_host)?;
                    let ones = CudaTensor::new(Arc::new(ones_dev), out_val.shape().to_vec());
                    let one_minus_out = bk.sub_inner(&ones, &out_val)?;
                    let out_omo = bk.mul_inner(&out_val, &one_minus_out)?;
                    bk.mul_inner(&grad_out, &out_omo)
                })?;
                accumulate(bk, grads, i, g_i)?;
            }
        }

        // Softplus: d/dx = g * sigmoid(x)  (PTX path)
        BackwardOp::Softplus { input, input_val } => {
            if let Some(i) = input {
                let g_i = no_grad(|| {
                    let n = input_val.numel();
                    let mut sigma_buf = bk.stream.alloc_zeros::<f32>(n)?;
                    super::kernels::sigmoid_f32(
                        &bk.ctx,
                        &bk.stream,
                        input_val.storage().as_ref(),
                        &mut sigma_buf,
                    )?;
                    let sigma = CudaTensor::new(Arc::new(sigma_buf), input_val.shape().to_vec());
                    bk.mul_inner(&grad_out, &sigma)
                })?;
                accumulate(bk, grads, i, g_i)?;
            }
        }

        // MatMul: grad_A = g @ B^T, grad_B = A^T @ g  (PTX path)
        //
        // B4.4a: handle the case where b_val has fewer dims than a_val
        // (e.g. a=[batch,seq,d_in], b=[d_in,d_out]).  In that case b was
        // zero-padded to rank(a) in the forward pass; the backward must
        // use a_val's rank for the matrix ops, then reduce gb from the
        // expanded rank down to b_val's original rank.
        BackwardOp::MatMul { a, b, a_val, b_val } => {
            // Determine the common rank used during forward.
            let a_rank = a_val.shape().len();
            let b_rank_orig = b_val.shape().len();
            // Pad b_val's shape with leading 1s if needed (mirrors forward padding).
            let b_val_padded_shape: Vec<usize> = if b_rank_orig < a_rank {
                let n_pad = a_rank - b_rank_orig;
                let mut s = vec![1usize; n_pad];
                s.extend_from_slice(b_val.shape());
                s
            } else {
                b_val.shape().to_vec()
            };
            let b_val_for_bwd = CudaTensor::new(b_val.storage().clone(), b_val_padded_shape);

            no_grad(|| -> Result<(), CudaError> {
                if let Some(a_nid) = a {
                    let rank = b_val_for_bwd.shape().len();
                    let last = rank - 1;
                    let bt = bk.transpose_inner(&b_val_for_bwd, last - 1, last)?;
                    let ga = bk.matmul_inner(&grad_out, &bt)?;
                    let ga_summed = sum_to_shape(bk, &ga, a_val.shape())?;
                    accumulate(bk, grads, a_nid, ga_summed)
                } else {
                    Ok(())
                }
            })?;
            no_grad(|| -> Result<(), CudaError> {
                if let Some(b_nid) = b {
                    let rank = a_val.shape().len();
                    let last = rank - 1;
                    let at = bk.transpose_inner(&a_val, last - 1, last)?;
                    let gb = bk.matmul_inner(&at, &grad_out)?;
                    // gb has rank = a_rank; reduce to b_val's original rank by
                    // summing over the leading batch dims that were padded.
                    let gb_summed = sum_to_shape_rank_reduce(bk, &gb, b_val.shape())?;
                    accumulate(bk, grads, b_nid, gb_summed)
                } else {
                    Ok(())
                }
            })?;
        }

        // Transpose: grad_input = transpose(g, dim_a, dim_b)
        BackwardOp::Transpose {
            input,
            dim_a,
            dim_b,
        } => {
            if let Some(i) = input {
                let g_i = no_grad(|| bk.transpose_inner(&grad_out, dim_a, dim_b))?;
                accumulate(bk, grads, i, g_i)?;
            }
        }

        // Narrow: grad_input = zeros(orig_shape) with grad_out at [start..start+len]
        BackwardOp::Narrow {
            input,
            dim,
            start,
            orig_shape,
        } => {
            if let Some(i) = input {
                let g_i = no_grad(|| scatter_to_narrow(bk, &grad_out, dim, start, &orig_shape))?;
                accumulate(bk, grads, i, g_i)?;
            }
        }

        // BroadcastAs: grad_input = sum_to(g, orig_shape)
        BackwardOp::BroadcastAs { input, orig_shape } => {
            if let Some(i) = input {
                let g_i = sum_to_shape(bk, &grad_out, &orig_shape)?;
                accumulate(bk, grads, i, g_i)?;
            }
        }

        // Concat: grad_inputs[k] = narrow(g, dim, offset_k, len_k)
        BackwardOp::Concat {
            inputs,
            dim,
            input_shapes,
        } => {
            no_grad(|| -> Result<(), CudaError> {
                let mut offset = 0usize;
                for (k, inp_nid) in inputs.iter().enumerate() {
                    let inp_len = input_shapes[k][dim];
                    if let Some(nid) = inp_nid {
                        let g_k = bk.narrow_inner(&grad_out, dim, offset, inp_len)?;
                        accumulate(bk, grads, *nid, g_k)?;
                    }
                    offset += inp_len;
                }
                Ok(())
            })?;
        }

        // Embedding: grad_table = scatter_add(indices, grad_out, table_shape)
        BackwardOp::Embedding {
            table,
            indices_val,
            table_shape,
        } => {
            if let Some(t_nid) = table {
                let g_table = no_grad(|| -> Result<CudaTensor, CudaError> {
                    let vocab_size = table_shape[0];
                    let embed_dim = table_shape[1];
                    let n_table = vocab_size * embed_dim;
                    let mut tg = bk.stream.alloc_zeros::<f32>(n_table)?;
                    let indices_len: usize = indices_val.shape().iter().product();
                    super::kernels::scatter_add_embedding_f32(
                        &bk.ctx,
                        &bk.stream,
                        &mut tg,
                        indices_val.storage_i64().as_ref(),
                        grad_out.storage().as_ref(),
                        indices_len,
                        embed_dim,
                        vocab_size,
                    )?;
                    Ok(CudaTensor::new(Arc::new(tg), table_shape.clone()))
                })?;
                accumulate(bk, grads, t_nid, g_table)?;
            }
        }

        // Gather: grad_x = scatter_add(indices, grad_out, along dim)
        BackwardOp::Gather {
            input,
            indices_val,
            dim,
            input_shape,
        } => {
            if let Some(x_nid) = input {
                let g_x = no_grad(|| -> Result<CudaTensor, CudaError> {
                    let n_x: usize = input_shape.iter().product();
                    let rank = input_shape.len();
                    let last_dim = rank - 1;
                    if dim == last_dim {
                        let mut xg = bk.stream.alloc_zeros::<f32>(n_x)?;
                        let last_dim_src = input_shape[last_dim];
                        let idx_shape = indices_val.shape().to_vec();
                        let last_dim_idx = idx_shape[last_dim];
                        super::kernels::scatter_add_gather_lastdim_f32(
                            &bk.ctx,
                            &bk.stream,
                            &mut xg,
                            indices_val.storage_i64().as_ref(),
                            grad_out.storage().as_ref(),
                            last_dim_src,
                            last_dim_idx,
                        )?;
                        Ok(CudaTensor::new(Arc::new(xg), input_shape.clone()))
                    } else {
                        let grad_t = bk.transpose_inner(&grad_out, dim, last_dim)?;
                        let mut transposed_input_shape = input_shape.clone();
                        transposed_input_shape.swap(dim, last_dim);
                        let n_xt: usize = transposed_input_shape.iter().product();
                        let mut xg_t = bk.stream.alloc_zeros::<f32>(n_xt)?;
                        let last_src_t = transposed_input_shape[last_dim];
                        let idx_host = bk.stream.clone_dtoh(indices_val.storage_i64().as_ref())?;
                        let idx_shape = indices_val.shape().to_vec();
                        let mut idx_t_shape = idx_shape.clone();
                        idx_t_shape.swap(dim, last_dim);
                        let n_idx = idx_host.len();
                        let mut idx_t_host = vec![0i64; n_idx];
                        let mut src_s = vec![1usize; rank];
                        for di in (0..rank - 1).rev() {
                            src_s[di] = src_s[di + 1] * idx_shape[di + 1];
                        }
                        let mut dst_s = vec![1usize; rank];
                        for di in (0..rank - 1).rev() {
                            dst_s[di] = dst_s[di + 1] * idx_t_shape[di + 1];
                        }
                        let mut perm: Vec<usize> = (0..rank).collect();
                        perm.swap(dim, last_dim);
                        for (dst_flat, slot) in idx_t_host.iter_mut().enumerate() {
                            let mut remaining = dst_flat;
                            let mut src_flat = 0usize;
                            for d in 0..rank {
                                let coord = remaining / dst_s[d];
                                remaining %= dst_s[d];
                                src_flat += coord * src_s[perm[d]];
                            }
                            *slot = idx_host[src_flat];
                        }
                        let idx_t_dev = bk.stream.clone_htod(&idx_t_host)?;
                        let idx_t =
                            super::CudaTensor::new_i64(Arc::new(idx_t_dev), idx_t_shape.clone());
                        let last_dim_idx_t = idx_t_shape[last_dim];
                        super::kernels::scatter_add_gather_lastdim_f32(
                            &bk.ctx,
                            &bk.stream,
                            &mut xg_t,
                            idx_t.storage_i64().as_ref(),
                            grad_t.storage().as_ref(),
                            last_src_t,
                            last_dim_idx_t,
                        )?;
                        let xg_t_tensor = CudaTensor::new(Arc::new(xg_t), transposed_input_shape);
                        bk.transpose_inner(&xg_t_tensor, dim, last_dim)
                    }
                })?;
                accumulate(bk, grads, x_nid, g_x)?;
            }
        }

        // Conv1d: grad_x via col2im, grad_w via GEMM, grad_b via sum.
        BackwardOp::Conv1d {
            x,
            w,
            bias_node,
            x_val,
            w_val,
            stride,
            padding,
            groups,
        } => {
            conv1d_backward(
                bk, grads, &grad_out, &x, &w, &bias_node, &x_val, &w_val, stride, padding, groups,
            )?;
        }

        // LogSoftmax: grad_x = g - softmax * sum(g, dim, keepdim)
        BackwardOp::LogSoftmax {
            input,
            dim,
            out_val,
        } => {
            if let Some(i) = input {
                let g_i = no_grad(|| -> Result<CudaTensor, CudaError> {
                    let shape = out_val.shape().to_vec();
                    let rank = shape.len();
                    let last_dim = rank - 1;
                    let sm_n = out_val.numel();
                    let mut sm_buf = bk.stream.alloc_zeros::<f32>(sm_n)?;
                    super::kernels::exp_f32(
                        &bk.ctx,
                        &bk.stream,
                        out_val.storage().as_ref(),
                        &mut sm_buf,
                    )?;
                    let softmax = CudaTensor::new(Arc::new(sm_buf), shape.clone());
                    let sum_g = reduce_sum_keepdim(bk, &grad_out, dim, &shape)?;
                    let sum_g_bc = if sum_g.shape() == shape.as_slice() {
                        sum_g
                    } else {
                        bk.broadcast_as_inner(&sum_g, &shape)?
                    };
                    let sm_sg = bk.mul_inner(&softmax, &sum_g_bc)?;
                    let _ = last_dim;
                    bk.sub_inner(&grad_out, &sm_sg)
                })?;
                accumulate(bk, grads, i, g_i)?;
            }
        }

        // RmsNorm: grad_x and grad_w via dedicated PTX kernels.
        BackwardOp::RmsNorm {
            x,
            w,
            x_val,
            w_val,
            eps,
        } => {
            no_grad(|| -> Result<(), CudaError> {
                if let Some(x_nid) = x {
                    let x_shape = x_val.shape().to_vec();
                    let nx = x_val.numel();
                    let mut gx = bk.stream.alloc_zeros::<f32>(nx)?;
                    super::kernels::rmsnorm_backward_x_f32(
                        &bk.ctx,
                        &bk.stream,
                        &mut gx,
                        grad_out.storage().as_ref(),
                        x_val.storage().as_ref(),
                        w_val.storage().as_ref(),
                        &x_shape,
                        eps,
                    )?;
                    let gx_t = CudaTensor::new(Arc::new(gx), x_shape);
                    accumulate(bk, grads, x_nid, gx_t)
                } else {
                    Ok(())
                }
            })?;
            no_grad(|| -> Result<(), CudaError> {
                if let Some(w_nid) = w {
                    let x_shape = x_val.shape().to_vec();
                    let d_model = *x_shape
                        .last()
                        .ok_or_else(|| CudaError::Shape("rmsnorm_bwd: empty x shape".into()))?;
                    let mut gw = bk.stream.alloc_zeros::<f32>(d_model)?;
                    super::kernels::rmsnorm_backward_w_f32(
                        &bk.ctx,
                        &bk.stream,
                        &mut gw,
                        grad_out.storage().as_ref(),
                        x_val.storage().as_ref(),
                        &x_shape,
                        eps,
                    )?;
                    let gw_t = CudaTensor::new(Arc::new(gw), w_val.shape().to_vec());
                    accumulate(bk, grads, w_nid, gw_t)
                } else {
                    Ok(())
                }
            })?;
        }

        // Cumsum: grad_x = reverse_cumsum(g, dim)  (PTX path)
        BackwardOp::Cumsum {
            input,
            dim,
            input_shape,
        } => {
            if let Some(i) = input {
                let g_i = no_grad(|| -> Result<CudaTensor, CudaError> {
                    let rank = input_shape.len();
                    let last_dim = rank - 1;
                    if dim == last_dim {
                        let n = grad_out.numel();
                        let mut out = bk.stream.alloc_zeros::<f32>(n)?;
                        super::kernels::reverse_cumsum_lastdim_f32(
                            &bk.ctx,
                            &bk.stream,
                            grad_out.storage().as_ref(),
                            &mut out,
                            &input_shape,
                        )?;
                        Ok(CudaTensor::new(Arc::new(out), input_shape.clone()))
                    } else {
                        let g_t = bk.transpose_inner(&grad_out, dim, last_dim)?;
                        let g_t_shape = g_t.shape().to_vec();
                        let n = g_t.numel();
                        let mut out_t = bk.stream.alloc_zeros::<f32>(n)?;
                        super::kernels::reverse_cumsum_lastdim_f32(
                            &bk.ctx,
                            &bk.stream,
                            g_t.storage().as_ref(),
                            &mut out_t,
                            &g_t_shape,
                        )?;
                        let rev_t = CudaTensor::new(Arc::new(out_t), g_t_shape);
                        bk.transpose_inner(&rev_t, dim, last_dim)
                    }
                })?;
                accumulate(bk, grads, i, g_i)?;
            }
        }

        // MeanAll: grad_x = ones(x.shape) * (g / numel)
        BackwardOp::MeanAll {
            input,
            numel,
            input_shape,
        } => {
            if let Some(i) = input {
                let g_i = no_grad(|| -> Result<CudaTensor, CudaError> {
                    // Fully device-side (B'.2f): scale the 1-element scalar
                    // by 1/numel, reshape it to `[1; rank]` (zero-copy —
                    // `with_shape` just bumps the storage Arc), then
                    // broadcast up to `input_shape` on-device. Replaces a
                    // D2H → build a full-size host `Vec<f32>` → H2D round
                    // trip that dominated `vjp:SumAll`/grad-norm wall time.
                    let scaled = bk.mul_scalar(&grad_out, 1.0 / numel as f32)?;
                    let scalar_bcast = scaled.with_shape(vec![1; input_shape.len()]);
                    bk.broadcast_as_inner(&scalar_bcast, &input_shape)
                })?;
                accumulate(bk, grads, i, g_i)?;
            }
        }

        // SumAll: grad_x = ones(x.shape) * g
        BackwardOp::SumAll { input, input_shape } => {
            if let Some(i) = input {
                let g_i = no_grad(|| -> Result<CudaTensor, CudaError> {
                    // Fully device-side (B'.2f) — see BackwardOp::MeanAll above.
                    let scalar_bcast = grad_out.with_shape(vec![1; input_shape.len()]);
                    bk.broadcast_as_inner(&scalar_bcast, &input_shape)
                })?;
                accumulate(bk, grads, i, g_i)?;
            }
        }

        // SsdScan: pure-Ops recompute backward (only appears in main backward).
        // In the sub-walk this variant is filtered out before calling apply_vjp.
        BackwardOp::SsdScan {
            x_in,
            a_in,
            b_in,
            c_in,
            x_val,
            a_val,
            b_val,
            c_val,
            block_len,
        } => {
            ssd_scan_backward_vjp(
                bk, grads, grad_out, x_in, a_in, b_in, c_in, &x_val, &a_val, &b_val, &c_val,
                block_len,
            )?;
        }
    }
    Ok(())
}

// ---- BackwardOp: owned snapshot for backward pass -------------------------

/// Owned mirror of `TapeOp` used during the backward pass.
///
/// We take a snapshot of each entry under the tape lock (cloning only
/// the minimal data needed) so that we don't hold the Mutex across
/// potentially expensive GPU kernel launches.
enum BackwardOp {
    Leaf {
        param_id: super::param::ParamId,
    },
    Add {
        lhs: Option<NodeId>,
        rhs: Option<NodeId>,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
    },
    Sub {
        lhs: Option<NodeId>,
        rhs: Option<NodeId>,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
    },
    Mul {
        lhs: Option<NodeId>,
        rhs: Option<NodeId>,
        lhs_val: CudaTensor,
        rhs_val: CudaTensor,
    },
    Neg {
        input: Option<NodeId>,
    },
    Reshape {
        input: Option<NodeId>,
        orig_shape: Vec<usize>,
    },
    MulScalar {
        input: Option<NodeId>,
        scale: f32,
    },
    AddScalar {
        input: Option<NodeId>,
    },
    Sqrt {
        input: Option<NodeId>,
        out_val: CudaTensor,
    },
    Div {
        lhs: Option<NodeId>,
        rhs: Option<NodeId>,
        rhs_val: CudaTensor,
        out_val: CudaTensor,
    },
    Exp {
        input: Option<NodeId>,
        out_val: CudaTensor,
    },
    Silu {
        input: Option<NodeId>,
        input_val: CudaTensor,
    },
    Sigmoid {
        input: Option<NodeId>,
        out_val: CudaTensor,
    },
    Softplus {
        input: Option<NodeId>,
        input_val: CudaTensor,
    },
    // ---- B4.3b ----------------------------------------------------------------
    MatMul {
        a: Option<NodeId>,
        b: Option<NodeId>,
        a_val: CudaTensor,
        b_val: CudaTensor,
    },
    Transpose {
        input: Option<NodeId>,
        dim_a: usize,
        dim_b: usize,
    },
    Narrow {
        input: Option<NodeId>,
        dim: usize,
        start: usize,
        orig_shape: Vec<usize>,
    },
    BroadcastAs {
        input: Option<NodeId>,
        orig_shape: Vec<usize>,
    },
    Concat {
        inputs: Vec<Option<NodeId>>,
        dim: usize,
        input_shapes: Vec<Vec<usize>>,
    },
    Embedding {
        table: Option<NodeId>,
        indices_val: CudaTensor,
        table_shape: Vec<usize>,
    },
    Gather {
        input: Option<NodeId>,
        indices_val: CudaTensor,
        dim: usize,
        input_shape: Vec<usize>,
    },
    Conv1d {
        x: Option<NodeId>,
        w: Option<NodeId>,
        bias_node: Option<Option<NodeId>>,
        x_val: CudaTensor,
        w_val: CudaTensor,
        stride: usize,
        padding: usize,
        groups: usize,
    },
    LogSoftmax {
        input: Option<NodeId>,
        dim: usize,
        out_val: CudaTensor,
    },
    RmsNorm {
        x: Option<NodeId>,
        w: Option<NodeId>,
        x_val: CudaTensor,
        w_val: CudaTensor,
        eps: f32,
    },
    Cumsum {
        input: Option<NodeId>,
        dim: usize,
        input_shape: Vec<usize>,
    },
    MeanAll {
        input: Option<NodeId>,
        numel: usize,
        input_shape: Vec<usize>,
    },
    SumAll {
        input: Option<NodeId>,
        input_shape: Vec<usize>,
    },
    // ---- B4.3c ----------------------------------------------------------------
    SsdScan {
        x_in: Option<NodeId>,
        a_in: Option<NodeId>,
        b_in: Option<NodeId>,
        c_in: Option<NodeId>,
        x_val: CudaTensor,
        a_val: CudaTensor,
        b_val: CudaTensor,
        c_val: CudaTensor,
        block_len: usize,
    },
}

// ---- Phase B'.1b: per-VJP profiler op-name mapping ------------------------

/// Maps a `BackwardOp` variant to a stable `&'static str` op name for the
/// profiler's per-VJP breakdown (`crate::profiler`, Phase B'.1b).
///
/// `vjp:SsdScan` is deliberately the *only* line item for the whole
/// `ssd_scan_backward_vjp` call (recompute + sub-walk): that function's
/// internal `Ops` calls and its own recursive `apply_vjp` invocations all
/// run at nesting depth > 0 relative to this variant's own timer, so the
/// shared depth counter folds their cost into this one bucket instead of
/// double-counting it — see `apply_vjp`'s doc comment.
fn backward_op_name(op: &BackwardOp) -> &'static str {
    match op {
        BackwardOp::Leaf { .. } => "vjp:Leaf",
        BackwardOp::Add { .. } => "vjp:Add",
        BackwardOp::Sub { .. } => "vjp:Sub",
        BackwardOp::Mul { .. } => "vjp:Mul",
        BackwardOp::Neg { .. } => "vjp:Neg",
        BackwardOp::Reshape { .. } => "vjp:Reshape",
        BackwardOp::MulScalar { .. } => "vjp:MulScalar",
        BackwardOp::AddScalar { .. } => "vjp:AddScalar",
        BackwardOp::Sqrt { .. } => "vjp:Sqrt",
        BackwardOp::Div { .. } => "vjp:Div",
        BackwardOp::Exp { .. } => "vjp:Exp",
        BackwardOp::Silu { .. } => "vjp:Silu",
        BackwardOp::Sigmoid { .. } => "vjp:Sigmoid",
        BackwardOp::Softplus { .. } => "vjp:Softplus",
        BackwardOp::MatMul { .. } => "vjp:MatMul",
        BackwardOp::Transpose { .. } => "vjp:Transpose",
        BackwardOp::Narrow { .. } => "vjp:Narrow",
        BackwardOp::BroadcastAs { .. } => "vjp:BroadcastAs",
        BackwardOp::Concat { .. } => "vjp:Concat",
        BackwardOp::Embedding { .. } => "vjp:Embedding",
        BackwardOp::Gather { .. } => "vjp:Gather",
        BackwardOp::Conv1d { .. } => "vjp:Conv1d",
        BackwardOp::LogSoftmax { .. } => "vjp:LogSoftmax",
        BackwardOp::RmsNorm { .. } => "vjp:RmsNorm",
        BackwardOp::Cumsum { .. } => "vjp:Cumsum",
        BackwardOp::MeanAll { .. } => "vjp:MeanAll",
        BackwardOp::SumAll { .. } => "vjp:SumAll",
        BackwardOp::SsdScan { .. } => "vjp:SsdScan",
    }
}
