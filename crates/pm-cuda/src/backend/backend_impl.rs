//! `CudaBackend` — B4 native CUDA backend replacing Candle.
//!
//! B4.1: minimal scaffolding with elementwise ops (add/sub/mul/neg).
//! B4.2a: adds cuBLAS handle for matmul; `CudaTensor` is now storage-variant.
//! B4.3a: adds the autograd tape (`Arc<Mutex<Tape>>`).
//! Full `pm_core::Ops` impl lands in `ops_impl.rs`.

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use pm_core::Tensor as _;

use super::kernels;
use super::tape::Tape;
use super::CudaTensor;
use crate::CudaError;

/// `PM_CUDA_MUL_TRACE=1` — B'.3 diagnostic (read once, cached): per-call
/// shape + timing breakdown for [`CudaBackend::broadcast_binary_op`]'s host
/// round trip. See that method's doc comment for context. Zero cost (one
/// relaxed `OnceLock` read) when unset.
///
/// NOTE (post-B'.3): broadcast `mul` moved to the device path
/// (`broadcast_mul_dev`), so `broadcast_binary_op` — and hence this
/// trace — now only ever sees `add`/`sub`. The env var name is kept for
/// continuity; it does *not* trace `mul` anymore.
fn broadcast_trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("PM_CUDA_MUL_TRACE").ok().as_deref() == Some("1"))
}

/// NumPy-style broadcast shape resolution, shared by the host-path
/// [`CudaBackend::broadcast_binary_op`] (still used for `add`/`sub`) and the
/// device-path [`CudaBackend::broadcast_mul_dev`] (B'.3).
///
/// Pads the shorter shape with leading 1s, then computes the broadcast
/// output shape. Returns `(a_padded, b_padded, out_shape)`, all rank-equal
/// to `out_shape.len()`.
fn broadcast_out_shape(
    a_shape: &[usize],
    b_shape: &[usize],
) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>), CudaError> {
    let rank = a_shape.len().max(b_shape.len());
    let pad = |s: &[usize]| -> Vec<usize> {
        let mut v = vec![1usize; rank - s.len()];
        v.extend_from_slice(s);
        v
    };
    let a_padded = pad(a_shape);
    let b_padded = pad(b_shape);

    let mut out_shape = vec![0usize; rank];
    for i in 0..rank {
        let da = a_padded[i];
        let db = b_padded[i];
        if da == db {
            out_shape[i] = da;
        } else if da == 1 {
            out_shape[i] = db;
        } else if db == 1 {
            out_shape[i] = da;
        } else {
            return Err(CudaError::Shape(format!(
                "broadcast: shapes {a_shape:?} and {b_shape:?} are not broadcastable (dim {i}: {da} vs {db})"
            )));
        }
    }
    Ok((a_padded, b_padded, out_shape))
}

/// Native CUDA backend for `pm-core::Ops`. B4 — `docs/b4-design.md`.
///
/// Cheap to clone: all fields are `Arc`-wrapped handles.  Clones share
/// the same tape, parameter-id allocator, and device handles.
#[derive(Clone)]
pub struct CudaBackend {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) cublas: Arc<CudaBlas>,
    pub(crate) next_param_id: Arc<AtomicU64>,
    /// Reverse-mode autograd tape shared across all clones of this backend.
    pub(crate) tape: Arc<Mutex<Tape>>,
}

impl CudaBackend {
    /// Initialise on CUDA device `ordinal`. Uses the device's default stream.
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let cublas = CudaBlas::new(stream.clone())?;
        Ok(Self {
            ctx,
            stream,
            cublas: Arc::new(cublas),
            next_param_id: Arc::new(AtomicU64::new(1)),
            tape: Arc::new(Mutex::new(Tape::new())),
        })
    }

    pub fn device(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Allocate the next unique `ParamId`.
    pub(crate) fn alloc_param_id(&self) -> super::param::ParamId {
        let id = self
            .next_param_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        super::param::ParamId(id)
    }

    // ---- Construction (internal helpers) ------------------------------------

    pub(crate) fn zeros_f32(&self, shape: &[usize]) -> Result<CudaTensor, CudaError> {
        let n = shape.iter().product::<usize>();
        let storage = self.stream.alloc_zeros::<f32>(n)?;
        Ok(CudaTensor::new(Arc::new(storage), shape.to_vec()))
    }

    pub(crate) fn from_slice_f32_inner(
        &self,
        data: &[f32],
        shape: &[usize],
    ) -> Result<CudaTensor, CudaError> {
        let n = shape.iter().product::<usize>();
        assert_eq!(
            data.len(),
            n,
            "from_slice_f32: data len {} != shape product {n}",
            data.len()
        );
        let storage = self.stream.clone_htod(data)?;
        Ok(CudaTensor::new(Arc::new(storage), shape.to_vec()))
    }

    pub(crate) fn to_vec_f32_inner(&self, x: &CudaTensor) -> Result<Vec<f32>, CudaError> {
        Ok(self.stream.clone_dtoh(x.storage().as_ref())?)
    }

    // ---- Element-wise (direct, used by Ops impls) ---------------------------

    fn binary_op<F>(&self, a: &CudaTensor, b: &CudaTensor, op: F) -> Result<CudaTensor, CudaError>
    where
        F: FnOnce(
            &Arc<CudaContext>,
            &Arc<CudaStream>,
            &CudaSlice<f32>,
            &CudaSlice<f32>,
            &mut CudaSlice<f32>,
        ) -> Result<(), CudaError>,
    {
        assert_eq!(
            a.shape(),
            b.shape(),
            "elementwise binary op: shape mismatch"
        );
        let n = a.numel();
        let mut out = self.stream.alloc_zeros::<f32>(n)?;
        op(
            &self.ctx,
            &self.stream,
            a.storage().as_ref(),
            b.storage().as_ref(),
            &mut out,
        )?;
        Ok(CudaTensor::new(Arc::new(out), a.shape().to_vec()))
    }

    pub(crate) fn add_inner(
        &self,
        a: &CudaTensor,
        b: &CudaTensor,
    ) -> Result<CudaTensor, CudaError> {
        self.binary_op(a, b, kernels::add_f32)
    }

    pub(crate) fn sub_inner(
        &self,
        a: &CudaTensor,
        b: &CudaTensor,
    ) -> Result<CudaTensor, CudaError> {
        self.binary_op(a, b, kernels::sub_f32)
    }

    pub(crate) fn mul_inner(
        &self,
        a: &CudaTensor,
        b: &CudaTensor,
    ) -> Result<CudaTensor, CudaError> {
        self.binary_op(a, b, kernels::mul_f32)
    }

    pub(crate) fn neg_inner(&self, a: &CudaTensor) -> Result<CudaTensor, CudaError> {
        let n = a.numel();
        let mut out = self.stream.alloc_zeros::<f32>(n)?;
        kernels::neg_f32(&self.ctx, &self.stream, a.storage().as_ref(), &mut out)?;
        Ok(CudaTensor::new(Arc::new(out), a.shape().to_vec()))
    }

    /// Elementwise binary op with NumPy-style broadcasting.
    ///
    /// Requires that `a` and `b` have **different** shapes (i.e. actual
    /// broadcast is needed).  For same-shape inputs the public `Ops::add/sub/mul`
    /// implementations call `add_inner/sub_inner/mul_inner` directly (PTX path).
    ///
    /// Algorithm:
    ///  1. Compute the output shape by broadcast rules (size-1 dims expand).
    ///  2. Download both tensors.
    ///  3. Iterate over every element of the output; for each element map its
    ///     multi-index to each input's index (clamp size-1 dims to 0).
    ///  4. Apply `elem_op` per element; upload the result.
    ///
    /// This is used by the public `Ops::add/sub` implementations (broadcasting
    /// bias-style adds — small operand counts in practice, e.g. `dt_bias`
    /// `(1,1,H)` against `(B,T,H)`) so that `ssd_scan_ops_default` works on the
    /// CUDA backend. `mul`'s broadcast path used to call this too, but its
    /// operands are full `(B,T,H,P)`-sized activations (`x_dt`, `d_term` in
    /// `Mamba2Block::forward`) — see [`CudaBackend::broadcast_mul_dev`] for the
    /// device-side replacement (B'.3, `docs/perf-log.md`). The internal
    /// `add_inner/sub_inner/mul_inner` helpers still require exact shape match
    /// and are used only inside backward VJP code.
    pub(crate) fn broadcast_binary_op<F>(
        &self,
        a: &CudaTensor,
        b: &CudaTensor,
        elem_op: F,
    ) -> Result<CudaTensor, crate::CudaError>
    where
        F: Fn(f32, f32) -> f32,
    {
        // B'.3 diagnostic (`PM_CUDA_MUL_TRACE=1`): shape + sub-op timing
        // breakdown for the host round-trip this function performs. Added to
        // identify which broadcast call sites were paying the ~1.3 ms/call
        // `mul`/`vjp:Mul` cost (docs/perf-log.md B'.3) before they were moved
        // to a device kernel. Zero cost when the env var is unset.
        let trace = broadcast_trace_enabled();
        let trace_t0 = if trace {
            let _ = self.stream.synchronize();
            Some(std::time::Instant::now())
        } else {
            None
        };
        let a_shape = a.shape();
        let b_shape = b.shape();
        let (a_padded, b_padded, out_shape) = broadcast_out_shape(a_shape, b_shape)?;
        let rank = out_shape.len();

        let n_out: usize = out_shape.iter().product();
        let a_host = self.stream.clone_dtoh(a.storage().as_ref())?;
        let b_host = self.stream.clone_dtoh(b.storage().as_ref())?;
        let dtoh_elapsed = trace_t0.map(|t| t.elapsed());

        // Compute strides for each input (0 for broadcast dims).
        let strides = |padded: &[usize]| -> Vec<usize> {
            let mut s = vec![0usize; rank];
            let mut stride = 1usize;
            for i in (0..rank).rev() {
                if padded[i] == 1 {
                    s[i] = 0; // broadcast: always maps to index 0 in that dim
                } else {
                    s[i] = stride;
                    stride *= padded[i];
                }
            }
            s
        };
        let a_strides = strides(&a_padded);
        let b_strides = strides(&b_padded);

        let mut result = vec![0f32; n_out];
        // Compute output strides (contiguous row-major).
        let mut out_strides = vec![1usize; rank];
        for i in (0..rank - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }

        for out_idx in 0..n_out {
            let mut a_idx = 0usize;
            let mut b_idx = 0usize;
            let mut rem = out_idx;
            for d in 0..rank {
                let coord = rem / out_strides[d];
                rem %= out_strides[d];
                a_idx += coord * a_strides[d];
                b_idx += coord * b_strides[d];
            }
            result[out_idx] = elem_op(a_host[a_idx], b_host[b_idx]);
        }
        let compute_elapsed = trace_t0.map(|t| t.elapsed());

        let out_dev = self.stream.clone_htod(&result)?;
        if let (Some(t0), Some(dtoh), Some(compute)) = (trace_t0, dtoh_elapsed, compute_elapsed) {
            let total = t0.elapsed();
            eprintln!(
                "[mul-trace] HOST-BROADCAST a={a_shape:?} b={b_shape:?} out={out_shape:?} \
                 n_out={n_out} dtoh_us={:.0} compute_us={:.0} htod_us={:.0} total_us={:.0}",
                dtoh.as_secs_f64() * 1e6,
                (compute - dtoh).as_secs_f64() * 1e6,
                (total - compute).as_secs_f64() * 1e6,
                total.as_secs_f64() * 1e6,
            );
        }
        Ok(CudaTensor::new(Arc::new(out_dev), out_shape))
    }

    /// Device-side NumPy-style broadcast multiply: `mul`'s broadcast path.
    ///
    /// Same shape resolution as [`broadcast_binary_op`](Self::broadcast_binary_op)
    /// (shared via [`broadcast_out_shape`]) but the elementwise work runs in
    /// `kernels::broadcast_mul_f32` on-device — no `clone_dtoh`/`clone_htod`,
    /// no host divmod loop. See that kernel's doc comment for the B'.3
    /// measurement this replaces.
    pub(crate) fn broadcast_mul_dev(
        &self,
        a: &CudaTensor,
        b: &CudaTensor,
    ) -> Result<CudaTensor, crate::CudaError> {
        let (a_padded, b_padded, out_shape) = broadcast_out_shape(a.shape(), b.shape())?;
        let n_out: usize = out_shape.iter().product();
        let mut out = self.stream.alloc_zeros::<f32>(n_out)?;
        kernels::broadcast_mul_f32(
            &self.ctx,
            &self.stream,
            a.storage().as_ref(),
            b.storage().as_ref(),
            &mut out,
            &a_padded,
            &b_padded,
            &out_shape,
        )?;
        Ok(CudaTensor::new(Arc::new(out), out_shape))
    }

    // ---- Public API kept for B4.1 smoke-test backward compat ----------------

    /// Kept for B4.1 smoke test. Delegates to the inner helper.
    pub fn zeros(&self, shape: &[usize]) -> Result<CudaTensor, CudaError> {
        self.zeros_f32(shape)
    }

    pub fn from_slice_f32(&self, data: &[f32], shape: &[usize]) -> Result<CudaTensor, CudaError> {
        self.from_slice_f32_inner(data, shape)
    }

    pub fn to_vec_f32(&self, x: &CudaTensor) -> Result<Vec<f32>, CudaError> {
        self.to_vec_f32_inner(x)
    }

    /// Copy an i64 tensor back to host. B4.2b — used by tests.
    pub fn to_vec_i64(&self, x: &CudaTensor) -> Result<Vec<i64>, CudaError> {
        Ok(self.stream.clone_dtoh(x.storage_i64().as_ref())?)
    }

    /// Return the number of ops currently recorded on the autograd tape.
    ///
    /// Primarily used in tests to verify that `backward()` clears the tape.
    pub fn tape_len(&self) -> usize {
        // SAFETY: poisoning is only possible if a prior backward panicked,
        // in which case aborting is the right behaviour.
        self.tape
            .lock()
            .expect("tape lock poisoned — prior panic in backward?")
            .len()
    }

    /// Discard all ops currently on the autograd tape.
    ///
    /// Use this in Ctrl-C handlers or test `setUp`/teardown to abandon a
    /// partially-built computation graph without running `backward`.
    /// After this call the tape is empty and the next forward pass starts
    /// from a clean state.
    pub fn reset_tape(&self) -> Result<(), crate::CudaError> {
        self.tape
            .lock()
            .map_err(|_| crate::CudaError::Internal("tape lock poisoned".into()))?
            .clear();
        Ok(())
    }

    /// Elementwise add. Delegates to `Ops::add` (tape-aware).
    ///
    /// Kept for B4.1 smoke-test backward compatibility. New code should
    /// call `<CudaBackend as Ops>::add` or use the `Ops` trait directly.
    pub fn add(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, CudaError> {
        pm_core::Ops::add(self, a, b)
    }

    pub fn sub(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, CudaError> {
        pm_core::Ops::sub(self, a, b)
    }

    pub fn mul(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor, CudaError> {
        pm_core::Ops::mul(self, a, b)
    }

    pub fn neg(&self, a: &CudaTensor) -> Result<CudaTensor, CudaError> {
        pm_core::Ops::neg(self, a)
    }
}

impl pm_backend::Backend for CudaBackend {
    fn device_kind(&self) -> pm_backend::DeviceKind {
        // CudaContext::ordinal() returns the device index this context was
        // created for.  See cudarc 0.19.8 src/driver/safe/core.rs.
        pm_backend::DeviceKind::Cuda {
            ordinal: self.ctx.ordinal(),
        }
    }
}
