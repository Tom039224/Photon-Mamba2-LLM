//! Backend-agnostic operations trait.
//!
//! All Mamba2 / PHOTON model code in `pm-core` is generic over `O: Ops`.
//! Backends implement `Ops` (and re-implement hot paths like `ssd_scan`
//! with fused kernels as performance work proceeds).
//!
//! `Ops` is intentionally **not** object-safe: hot paths such as the SSD
//! scan should be monomorphised so they can be inlined into the backend.

use crate::{Dtype, Param, Tensor};

/// Tensor operations exposed by a backend.
///
/// This trait grows over time. Phase 1 only needs the ops listed below;
/// when a new op is required, add it here with a clear doc-comment and
/// implement it in every backend.
pub trait Ops {
    type Tensor: Tensor;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Trainable parameter type for this backend.
    /// Holds whatever optimizer state requires (e.g. `candle_core::Var`).
    type Param: Param<Tensor = Self::Tensor>;

    /// Opaque backend-specific gradient store produced by [`backward`](Ops::backward).
    type GradStore;

    // ---- Construction ----------------------------------------------------

    fn zeros(&self, shape: &[usize], dtype: Dtype) -> Result<Self::Tensor, Self::Error>;
    fn ones(&self, shape: &[usize], dtype: Dtype) -> Result<Self::Tensor, Self::Error>;

    /// Build a tensor from an `f32` host slice. Used by tests and reference
    /// loaders. Production paths load weights via `safetensors` instead.
    ///
    /// Takes `&self` by design (needs the backend's device/context) — not
    /// the `wrong_self_convention` constructor case.
    #[allow(clippy::wrong_self_convention)]
    fn from_slice_f32(&self, data: &[f32], shape: &[usize]) -> Result<Self::Tensor, Self::Error>;

    /// Build an integer tensor from an `i64` host slice. Used by tokenisers
    /// and tests that need to drive `embedding` lookups.
    ///
    /// Takes `&self` by design (needs the backend's device/context) — not
    /// the `wrong_self_convention` constructor case.
    #[allow(clippy::wrong_self_convention)]
    fn from_slice_i64(&self, data: &[i64], shape: &[usize]) -> Result<Self::Tensor, Self::Error>;

    /// Copy a tensor's data back to host as `f32`. For tests / debugging.
    fn to_vec_f32(&self, x: &Self::Tensor) -> Result<Vec<f32>, Self::Error>;

    // ---- Dtype conversion --------------------------------------------------

    /// Cast `t` to `dtype`, preserving shape.
    ///
    /// When `t.dtype() == dtype` this must be a no-op: return the input
    /// unchanged (no new compute, no new autograd node), so callers can
    /// call it unconditionally at a numeric "island" boundary without
    /// any cost on the all-fp32 path.
    ///
    /// Backing the bf16 mixed-precision compute path (memory-efficiency
    /// plan Phase A2, `docs/perf-log.md` 2026-07-03): `pm-core` model
    /// code upcasts numerically-sensitive sub-computations (the SSD
    /// scan's cumsum/exp, softplus, rmsnorm, cross-entropy's
    /// log-softmax) to `F32` and downcasts the result back to the
    /// ambient compute dtype, and casts fp32 parameter tensors to the
    /// ambient dtype inline before a matmul/elementwise op with a
    /// lower-precision activation. Backends whose autograd tracks the
    /// cast (Candle's `to_dtype` records `Op::ToDType`, whose backward
    /// re-casts the incoming gradient to the source dtype) automatically
    /// keep parameter gradients in the parameter's native (fp32) storage
    /// dtype — no separate master-weight copy is needed.
    fn to_dtype(&self, t: &Self::Tensor, dtype: Dtype) -> Result<Self::Tensor, Self::Error>;

    // ---- Element-wise ----------------------------------------------------

    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
    fn sub(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
    fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
    fn neg(&self, a: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;

    // ---- Linear algebra --------------------------------------------------

    /// Batched matrix multiply. Broadcasting follows numpy/torch rules.
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;

    // ---- Activations / Normalisation -------------------------------------

    /// Root-mean-square layer norm with a per-feature `weight`.
    /// `x: (..., D)`, `weight: (D,)`. `eps` is added to the variance.
    fn rmsnorm(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<Self::Tensor, Self::Error>;

    fn silu(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
    fn softplus(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;
    fn sigmoid(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;

    // ---- Convolution -----------------------------------------------------

    /// Grouped 1D convolution.
    /// - `x`: `(B, C_in, T)`
    /// - `weight`: `(C_out, C_in / groups, K)`
    /// - `bias`: `(C_out,)` or `None`
    fn conv1d(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Result<Self::Tensor, Self::Error>;

    // ---- Indexing / reduction --------------------------------------------

    fn cumsum(&self, x: &Self::Tensor, dim: usize) -> Result<Self::Tensor, Self::Error>;

    /// Slice `x` along `dim` from `start` for `len` elements.
    fn narrow(
        &self,
        x: &Self::Tensor,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<Self::Tensor, Self::Error>;

    // ---- Shape manipulation ----------------------------------------------

    fn reshape(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor, Self::Error>;

    /// Swap two dimensions of `x`.
    fn transpose(
        &self,
        x: &Self::Tensor,
        dim_a: usize,
        dim_b: usize,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Expand `x` so it has `shape` via broadcasting size-1 dimensions.
    fn broadcast_as(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor, Self::Error>;

    /// Concatenate tensors along `dim`. All inputs must share the same
    /// shape except along `dim`. At least one tensor is required.
    fn concat(&self, tensors: &[&Self::Tensor], dim: usize) -> Result<Self::Tensor, Self::Error>;

    // ---- Embedding -------------------------------------------------------

    /// Token embedding lookup.
    ///
    /// - `table`: `(V, D)` float tensor.
    /// - `indices`: integer tensor of any shape `S`. Each element must be
    ///   a valid row index `[0, V)`.
    /// - Returns: `(S..., D)` float tensor.
    fn embedding(
        &self,
        table: &Self::Tensor,
        indices: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error>;

    // ---- Element-wise transcendental -------------------------------------

    fn exp(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;

    // ---- Trainable parameters --------------------------------------------

    /// Allocate a trainable parameter initialised from a host `f32` slice.
    fn param_from_slice_f32(
        &self,
        data: &[f32],
        shape: &[usize],
    ) -> Result<Self::Param, Self::Error>;

    /// Allocate a trainable parameter filled with zeros.
    fn param_zeros(&self, shape: &[usize], dtype: Dtype) -> Result<Self::Param, Self::Error>;

    /// Allocate a trainable parameter filled with ones.
    fn param_ones(&self, shape: &[usize], dtype: Dtype) -> Result<Self::Param, Self::Error>;

    // ---- Autograd --------------------------------------------------------

    /// Run reverse-mode autodiff on `loss` (must be a scalar tensor or
    /// reducible to one). The returned store maps every trainable
    /// parameter touched in the forward to its accumulated gradient.
    fn backward(&self, loss: &Self::Tensor) -> Result<Self::GradStore, Self::Error>;

    /// Look up the gradient of `param` in `store`. Returns `None` when
    /// the parameter was not on any backward path from `loss`.
    fn gradient(
        &self,
        store: &Self::GradStore,
        param: &Self::Param,
    ) -> Result<Option<Self::Tensor>, Self::Error>;

    /// Overwrite the gradient of `param` in `store` with `grad`. Used by
    /// gradient-clipping helpers that scale every grad in place before
    /// the optimiser runs.
    fn set_gradient(
        &self,
        store: &mut Self::GradStore,
        param: &Self::Param,
        grad: Self::Tensor,
    ) -> Result<(), Self::Error>;

    /// Run `f` with autograd recording disabled, restoring the previous
    /// recording state afterwards.
    ///
    /// Inference / eval loops that call forward passes many times
    /// without ever calling [`backward`](Self::backward) MUST wrap the
    /// calls in this scope on tape-based backends — otherwise every
    /// forward keeps appending saved activations to the autograd tape
    /// and device memory grows without bound (observed: `pm eval
    /// hellaswag` CUDA OOM after tens of items, 2026-07-05; the
    /// training loop never hits this because `backward` consumes the
    /// tape every step).
    ///
    /// The default implementation just runs `f`: backends whose graph
    /// dies with the tensors that reference it (e.g. Candle) do not
    /// accumulate state across iterations and need no special handling.
    fn no_grad_scope<R>(&self, f: impl FnOnce() -> R) -> R {
        f()
    }

    /// Wrap an existing tensor's value as a fresh trainable [`Param`].
    /// Used by activation checkpointing to introduce a stable id at
    /// segment boundaries so the gradient flowing in can be looked up
    /// after the main backward.
    ///
    /// In Candle this allocates a new `Var`, which copies the underlying
    /// storage — that's the price of getting a backward-addressable
    /// boundary without re-engineering the autograd tape.
    fn param_from_tensor(&self, t: &Self::Tensor) -> Result<Self::Param, Self::Error>;

    /// Merge `src`'s gradient entries into `dst`. For entries already in
    /// `dst`, the values are added; new entries are inserted as-is.
    /// Used by activation checkpointing to fold the recomputed-segment
    /// gradients back into the main backward's grad store.
    fn merge_grad_stores(
        &self,
        dst: &mut Self::GradStore,
        src: Self::GradStore,
    ) -> Result<(), Self::Error>;

    /// Return a value-identical copy of `t` with **no** autograd ancestry
    /// at all: it is not a fresh [`Param`]/leaf (unlike
    /// [`Ops::param_from_tensor`]), it simply carries no backward
    /// information whatsoever, so composing it with further `Ops` calls
    /// costs nothing on a backend's autograd bookkeeping — no new tape
    /// entry, no new `Param`/`ParamId`.
    ///
    /// This is the building block [`Ops::fused_cross_entropy`]'s tiling
    /// loop uses to keep row-tile intermediates from ever joining a
    /// backend's autograd graph: `param_from_tensor` is not a substitute
    /// here, because on backends with a shared/global tape (`pm-cuda`)
    /// its fresh `Leaf` entry is *still* a tracked node — every op built
    /// from it keeps pushing tape entries (and, on `pm-cuda`, those
    /// entries hold `Arc` clones of their `(rows, V)`-sized operands
    /// alive until the *next* `Ops::backward` call clears the whole
    /// tape, i.e. exactly the "every tile stays resident" failure mode
    /// fused cross-entropy exists to avoid). `detach` guarantees the
    /// chain terminates immediately: downstream ops see "no autograd
    /// ancestor" and skip tape recording entirely.
    fn detach(&self, t: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;

    /// Copy an integer tensor's data back to host as `i64`, row-major
    /// flattened. Mirrors [`Ops::to_vec_f32`]. Used by
    /// [`Ops::fused_cross_entropy`]'s default tiling loop to slice
    /// `targets` and build each tile's one-hot matrix on host — targets
    /// tensors are tiny (`B·T` elements) so this is a one-shot,
    /// per-training-step cost, not a per-tile one.
    fn to_vec_i64(&self, x: &Self::Tensor) -> Result<Vec<i64>, Self::Error>;

    /// In-place SGD update: `param ← param - lr · grad`.
    fn sgd_step(
        &self,
        param: &Self::Param,
        grad: &Self::Tensor,
        lr: f32,
    ) -> Result<(), Self::Error>;

    /// Overwrite a parameter's storage with the given tensor (same
    /// shape required). Used by optimizers (AdamW) that compute a
    /// fresh value off-line and need to replace the parameter atomically.
    fn assign(&self, param: &Self::Param, value: &Self::Tensor) -> Result<(), Self::Error>;

    // ---- Loss / reduction helpers ---------------------------------------

    /// `log(softmax(x, dim))`, numerically stable.
    fn log_softmax(&self, x: &Self::Tensor, dim: usize) -> Result<Self::Tensor, Self::Error>;

    /// Gather along `dim`: `out[i0,..,iD-1] = x[i0,.., indices[i0,..], ..,iD-1]`.
    /// `indices` must have the same shape as `x` except at axis `dim`.
    fn gather(
        &self,
        x: &Self::Tensor,
        indices: &Self::Tensor,
        dim: usize,
    ) -> Result<Self::Tensor, Self::Error>;

    /// Mean over all elements, returns a scalar (shape `[]`).
    fn mean_all(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;

    /// Sum over all elements, returns a scalar (shape `[]`).
    fn sum_all(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;

    /// `x` multiplied by `scale` (broadcast scalar).
    fn mul_scalar(&self, x: &Self::Tensor, scale: f32) -> Result<Self::Tensor, Self::Error>;

    /// Element-wise square root.
    fn sqrt(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;

    /// Element-wise division (with broadcasting).
    fn div(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error>;

    /// `x + scalar`, broadcast.
    fn add_scalar(&self, x: &Self::Tensor, scalar: f32) -> Result<Self::Tensor, Self::Error>;

    // ---- Mamba2 SSD scan -------------------------------------------------

    /// Chunked SSD scan from the Mamba2 paper (Listing 1, §5).
    ///
    /// Shapes (B = batch, T = sequence, H = heads, P = head dim, N = state dim):
    /// - `x`: `(B, T, H, P)`
    /// - `a`: `(B, T, H)` — scalar-per-head SSM (§6.2)
    /// - `b`: `(B, T, H, N)`
    /// - `c`: `(B, T, H, N)`
    /// - `block_len`: chunk size Q (paper default: 64)
    ///
    /// Returns `y: (B, T, H, P)`.
    ///
    /// Backends should override with a fused kernel; until then the
    /// reference implementation in `pm-core::mamba2::ssd_scan_naive` is
    /// used by tests for numerical parity.
    ///
    /// **Dtype**: implementations may assume `x`/`a`/`b`/`c` are `F32`.
    /// The internal cumsum + exp of `A`-cumulative differences
    /// under/overflows at `bf16` precision, so callers running a bf16
    /// compute path (`PhotonMamba::compute_dtype`, memory-efficiency
    /// plan Phase A2) must upcast to `F32` before calling this and
    /// downcast the result back afterward — `Mamba2Block::forward` does
    /// so at the call site; see `mamba2::ssd_ops` module docs.
    fn ssd_scan(
        &self,
        x: &Self::Tensor,
        a: &Self::Tensor,
        b: &Self::Tensor,
        c: &Self::Tensor,
        block_len: usize,
    ) -> Result<Self::Tensor, Self::Error>;

    // ---- Fused cross-entropy over a tied embedding table -----------------

    /// Memory-bounded next-token cross-entropy against a tied embedding
    /// table, computing the same value as `pm-train::loss::
    /// cross_entropy_loss(lm_head_logits(hidden, table), targets)`
    /// together with its analytic gradient, **without** ever
    /// materialising the full `(rows, V)` logits (let alone `(B, T, V)`)
    /// — only one row-tile of width `tile_rows` is resident at a time
    /// (memory-efficiency plan: fused/tiled cross-entropy).
    ///
    /// - `hidden`: `(..., D)` — any rank ≥ 2 (typically `(B, T, D)`), the
    ///   decoder's pre-lm_head output. Must carry its real autograd
    ///   ancestry (the rest of the model) — this call does **not**
    ///   detach it internally, so the caller can inject `grad_hidden`
    ///   back through it (see below).
    /// - `table`: `(V, D)` — the tied embedding weight, used for both
    ///   the input embedding lookup and the lm_head (`TokenEmbedding`).
    ///   Same ancestry requirement as `hidden`.
    /// - `targets`: integer tensor with the same element count as
    ///   `hidden`'s leading dims (`B·T`), one class index per row.
    /// - `tile_rows`: row-tile width. Bounds peak transient memory to
    ///   `O(tile_rows · V)`; must be `> 0`.
    ///
    /// Returns `(loss, grad_hidden, grad_table)`:
    /// - `loss`: scalar `F32`, mean NLL over all `B·T` positions — same
    ///   convention as `cross_entropy_loss`.
    /// - `grad_hidden`: same shape/dtype as `hidden`, **fully detached**
    ///   (see [`Ops::detach`]) — `∂loss/∂hidden` treating `table` as
    ///   fixed.
    /// - `grad_table`: `(V, D)` `F32`, **fully detached** —
    ///   `∂loss/∂table` treating `hidden` as fixed.
    ///
    /// Callers reconnect these to the real graph the same way
    /// `pm-core::checkpoint` reconnects a recomputed segment: build a
    /// scalar `phantom = sum_all(hidden ⊙ grad_hidden) + sum_all(table ⊙
    /// grad_table)` and run `Ops::backward` on it once. Because `table`
    /// (the tied embedding) is *also* reachable from `hidden` (input
    /// embedding lookup), that one `backward` call correctly sums both
    /// of `table`'s contributions via ordinary multivariate chain rule —
    /// see `pm-train::loss::fused_cross_entropy_injected`.
    ///
    /// Default implementations should delegate to
    /// `pm_core::loss::fused_cross_entropy_tiled`, the backend-agnostic
    /// reference tiling built only from other `Ops` primitives (plus
    /// [`Ops::detach`]/[`Ops::to_vec_i64`]); a backend may instead
    /// override this with a true fused kernel (PLAN.md Phase 2.5 / M.3)
    /// without touching `pm-train`/`pm-cli`.
    // `(loss, grad_hidden, grad_table)` — see the doc comment above for
    // what each element is; a type alias would obscure more than the
    // 3-tuple does, so this just opts out of the lint.
    #[allow(clippy::type_complexity)]
    fn fused_cross_entropy(
        &self,
        hidden: &Self::Tensor,
        table: &Self::Tensor,
        targets: &Self::Tensor,
        tile_rows: usize,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error>;
}
