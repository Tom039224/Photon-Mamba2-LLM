//! Reverse-mode autograd tape for `CudaBackend`.
//!
//! B4.3a — implements the tape infrastructure used by `backward()`.
//!
//! Design overview
//! ---------------
//! Each forward op that is tape-aware records a `TapeOp` entry and
//! stamps the output `CudaTensor` with the resulting `NodeId`.  When
//! `backward()` is called it replays the tape in reverse order,
//! accumulating VJP (vector-Jacobian product) contributions into a
//! `HashMap<NodeId, CudaTensor>` and depositing the final gradient for
//! each `Leaf` node into a `HashMap<ParamId, CudaTensor>`.
//!
//! No-grad guard
//! -------------
//! All VJP arithmetic in `backward()` is wrapped in `no_grad(|| { … })`
//! so that the backward arithmetic does not itself register tape entries
//! (which would cause infinite recursion / unbounded tape growth).
//!
//! Broadcast / SSD
//! ---------------
//! Broadcast and SSD scan backward are implemented (B4.3b complete).
//! Binary kernel ops (`add_f32`/`sub_f32`/`mul_f32`) currently use
//! `assert_eq!` on shape mismatch rather than returning `Err` — converting
//! these to `Result` is deferred to B4.4 / B4.5 (lower priority since
//! broadcast paths use `BroadcastAs` entries which carry their own shape
//! contract, and `MatMul` broadcast paths save the original unexpanded
//! tensors so VJP correctly reduces via `sum_to_shape`).

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::param::ParamId;
use super::CudaTensor;

// ---- NodeId ----------------------------------------------------------------

/// Unique id stamped on every tape-tracked `CudaTensor`.
///
/// `generation` is bumped by every [`Tape::clear`] call (i.e. every
/// `CudaBackend::backward` call, since `backward` clears the tape as its
/// last step so device memory held by dead intermediates can be freed —
/// see the module doc's "No-grad guard" section and `CudaBackend::backward`'s
/// doc comment). `index` is the position within `Tape::entries` *for that
/// generation* — `entries` is reset to empty on every `clear()`, so `index`
/// alone is only unique within one generation.
///
/// Packing `generation` into the id (rather than reusing bare `index`
/// values across generations) is what makes activation checkpointing
/// correct on this backend: `pm_core::checkpoint_backward` calls
/// `Ops::backward` *multiple times* per logical training step (once for
/// the main loss, once per checkpoint segment), so a `NodeId` captured
/// before an earlier `backward()` call (e.g. cached on a `CudaParam`, or
/// on a checkpoint boundary `Tensor` held across those calls) would, with
/// a bare generation-less `index`, silently alias an unrelated entry
/// pushed at the same position in a *later* generation — corrupting (not
/// just losing) gradients for whatever real op happened to land there.
/// With `generation` included, a stale id simply never equals any id
/// constructed in the current generation, so `HashMap`-keyed lookups
/// (`grads: HashMap<NodeId, CudaTensor>` in `backward()`) correctly treat
/// it as unreachable instead of aliasing. See [`LeafOrigin`] for how
/// `CudaParam`s (and checkpoint boundaries, which are just `CudaParam`s
/// via `Ops::param_from_tensor`) recover a *valid* id in the new
/// generation instead of merely going undetected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId {
    pub generation: u32,
    pub index: u32,
}

// ---- no-grad guard ---------------------------------------------------------

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Returns `true` when grad tracking is active for the current thread.
pub(crate) fn grad_enabled() -> bool {
    GRAD_ENABLED.with(|c| c.get())
}

/// Execute `f` with gradient tracking disabled, then restore the
/// previous state.  Used by `backward()` so VJP arithmetic does not
/// recursively push entries onto the tape.
pub(crate) fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let prev = GRAD_ENABLED.with(|c| c.replace(false));
    let r = f();
    GRAD_ENABLED.with(|c| c.set(prev));
    r
}

// ---- TapeOp ----------------------------------------------------------------

/// A single recorded operation on the autograd tape.
///
/// Each variant stores the minimal information needed to compute the
/// VJP: the `NodeId`s of the inputs and any *saved tensors* (forward
/// values needed at backward time, e.g. the output of `exp` or the
/// input of `silu`).
///
/// Inputs that may be "detached" constants (no prior tape node) are
/// stored as `Option<NodeId>`; the backward pass skips gradient
/// accumulation for `None` inputs.
pub enum TapeOp {
    /// Leaf — a trainable `CudaParam`. Gradients land in the `GradStore`.
    Leaf { param_id: ParamId },

    /// Element-wise add: `out = lhs + rhs` (broadcast-aware).
    ///
    /// `lhs_shape` / `rhs_shape` are the shapes *before* broadcast so the
    /// backward can reduce `grad_out` back via `sum_to_shape` when needed.
    Add {
        lhs: Option<NodeId>,
        rhs: Option<NodeId>,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
    },

    /// Element-wise sub: `out = lhs - rhs` (broadcast-aware).
    Sub {
        lhs: Option<NodeId>,
        rhs: Option<NodeId>,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
    },

    /// Element-wise mul: `out = lhs * rhs` (broadcast-aware).
    ///
    /// Saved tensors: both inputs (needed for cross-multiplied VJP).
    /// `lhs_shape` / `rhs_shape` are the pre-broadcast shapes.
    Mul {
        lhs: Option<NodeId>,
        rhs: Option<NodeId>,
        lhs_val: CudaTensor,
        rhs_val: CudaTensor,
    },

    /// Element-wise neg: `out = -input`.
    Neg { input: Option<NodeId> },

    /// Reshape: `out = reshape(input, new_shape)`.
    ///
    /// `orig_shape` is the shape to restore in the backward pass.
    Reshape {
        input: Option<NodeId>,
        orig_shape: Vec<usize>,
    },

    /// Scalar multiply: `out = input * scale`.
    MulScalar { input: Option<NodeId>, scale: f32 },

    /// Scalar add: `out = input + scalar` (grad passes straight through).
    AddScalar { input: Option<NodeId> },

    /// Element-wise sqrt: `out = sqrt(input)`.
    ///
    /// Saved: output value (`out_val`). VJP: `d/dx = 1 / (2 * sqrt(x)) = 1 / (2 * out)`.
    Sqrt {
        input: Option<NodeId>,
        out_val: CudaTensor,
    },

    /// Element-wise div: `out = lhs / rhs`.
    ///
    /// VJP lhs: `g / rhs`, VJP rhs: `-g * out / rhs`.
    Div {
        lhs: Option<NodeId>,
        rhs: Option<NodeId>,
        rhs_val: CudaTensor,
        out_val: CudaTensor,
    },

    /// Element-wise exp: `out = exp(input)`.
    ///
    /// VJP: `g * out` (exp is its own derivative).
    Exp {
        input: Option<NodeId>,
        out_val: CudaTensor,
    },

    /// SiLU activation: `out = x * sigmoid(x)`.
    ///
    /// VJP: `g * (sigma + x * sigma * (1 - sigma))` where `sigma = sigmoid(x)`.
    Silu {
        input: Option<NodeId>,
        input_val: CudaTensor,
    },

    /// Sigmoid: `out = sigma(x)`.
    ///
    /// VJP: `g * out * (1 - out)`.
    Sigmoid {
        input: Option<NodeId>,
        out_val: CudaTensor,
    },

    /// Softplus: `out = log(1 + exp(x))`.
    ///
    /// VJP: `g * sigmoid(x)`.
    Softplus {
        input: Option<NodeId>,
        input_val: CudaTensor,
    },

    // ---- B4.3b: algebra / shape / reduce backward --------------------------
    /// Matrix multiply: `out = A @ B`.
    ///
    /// VJP A: `grad_out @ B^T`; VJP B: `A^T @ grad_out`.
    MatMul {
        a: Option<NodeId>,
        b: Option<NodeId>,
        /// Saved forward value of A (needed for grad_B).
        a_val: CudaTensor,
        /// Saved forward value of B (needed for grad_A).
        b_val: CudaTensor,
    },

    /// Transpose: `out = transpose(x, dim_a, dim_b)`.
    ///
    /// VJP: `transpose(grad_out, dim_a, dim_b)` (self-inverse).
    Transpose {
        input: Option<NodeId>,
        dim_a: usize,
        dim_b: usize,
    },

    /// Narrow (slice along a dim): `out = x[.., start..start+len, ..]`.
    ///
    /// VJP: zeros of `orig_shape` with `grad_out` scattered into the narrow window.
    Narrow {
        input: Option<NodeId>,
        dim: usize,
        start: usize,
        len: usize,
        orig_shape: Vec<usize>,
    },

    /// Broadcast-expand: `out = broadcast_as(x, target_shape)`.
    ///
    /// VJP: `sum_to(grad_out, orig_shape)`.
    BroadcastAs {
        input: Option<NodeId>,
        orig_shape: Vec<usize>,
        /// Full target (output) shape — kept for documentation/sanity checks.
        target_shape: Vec<usize>,
    },

    /// Concatenation along `dim`.
    ///
    /// VJP: `narrow(grad_out, dim, offset_i, len_i)` for each input.
    Concat {
        inputs: Vec<Option<NodeId>>,
        dim: usize,
        input_shapes: Vec<Vec<usize>>,
    },

    /// Embedding lookup: `out = table[indices]`.
    ///
    /// VJP table: `scatter_add(indices, grad_out, table_shape)`.
    /// VJP indices: None (i64, no grad).
    Embedding {
        table: Option<NodeId>,
        indices_val: CudaTensor,
        table_shape: Vec<usize>,
    },

    /// Gather along `dim`.
    ///
    /// VJP x: `scatter_add(indices, grad_out, along dim)`.
    /// VJP indices: None (i64, no grad).
    Gather {
        input: Option<NodeId>,
        indices_val: CudaTensor,
        dim: usize,
        input_shape: Vec<usize>,
    },

    /// 1D convolution: `out = conv1d(x, w, bias, stride, padding, groups)`.
    ///
    /// VJP x: col2im path.
    /// VJP w: GEMM path.
    /// VJP bias: sum over batch and time.
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

    /// log_softmax along `dim`.
    ///
    /// VJP: `grad_out - softmax * sum(grad_out, dim, keepdim)`.
    LogSoftmax {
        input: Option<NodeId>,
        dim: usize,
        /// Saved output (= log_softmax result); softmax = exp(out_val).
        out_val: CudaTensor,
    },

    /// RMSNorm: `out = x * weight * rsqrt(mean(x^2)+eps)`.
    ///
    /// VJP x and weight.
    RmsNorm {
        x: Option<NodeId>,
        w: Option<NodeId>,
        x_val: CudaTensor,
        w_val: CudaTensor,
        eps: f32,
    },

    /// Inclusive cumsum along `dim`.
    ///
    /// VJP: reverse_cumsum(grad_out, dim).
    Cumsum {
        input: Option<NodeId>,
        dim: usize,
        input_shape: Vec<usize>,
    },

    /// Mean over all elements.
    ///
    /// VJP: broadcast scalar grad_out / numel.
    MeanAll {
        input: Option<NodeId>,
        numel: usize,
        input_shape: Vec<usize>,
    },

    /// Sum over all elements.
    ///
    /// VJP: broadcast scalar grad_out.
    SumAll {
        input: Option<NodeId>,
        input_shape: Vec<usize>,
    },

    // ---- B4.3c ----------------------------------------------------------------
    /// SSD scan (fused PTX forward + pure-Ops recompute backward).
    ///
    /// Saved tensors: clones of x/a/b/c at forward time (Arc bump, no copy).
    /// The backward arm runs `ssd_scan_ops_default` with autograd enabled on a
    /// fresh sub-tape to obtain VJPs for each input, then accumulates into the
    /// parent graph.
    ///
    /// M1 (B4.3c review fix): boxed to avoid clippy::large_enum_variant — this
    /// variant is ~216 bytes (4 × NodeId + 4 × CudaTensor + usize) which would
    /// inflate every `TapeOp` on the stack.
    SsdScan(Box<SsdScanData>),
}

/// Payload for `TapeOp::SsdScan`.  Boxed to keep `TapeOp` small.
pub struct SsdScanData {
    pub x: Option<NodeId>,
    pub a: Option<NodeId>,
    pub b: Option<NodeId>,
    pub c: Option<NodeId>,
    pub x_val: CudaTensor,
    pub a_val: CudaTensor,
    pub b_val: CudaTensor,
    pub c_val: CudaTensor,
    pub block_len: usize,
}

// ---- Tape ------------------------------------------------------------------

/// Autograd tape — a flat list of recorded operations, scoped to one
/// "generation" (see [`NodeId`]).
///
/// `NodeId.index` values are assigned sequentially from 0 within a
/// generation; each `push` returns the `NodeId` of the operation's
/// *output* node.  `backward()` uses the assignment
/// `entries[nid.index as usize] ↔ TapeOp` (for `nid.generation ==
/// self.generation`) to look up ops.
pub struct Tape {
    pub(crate) entries: Vec<TapeOp>,
    /// Bumped by every [`clear`](Tape::clear) call. See [`NodeId`]'s doc.
    pub(crate) generation: u32,
    /// Per-generation cache of the most recent `Leaf` entry pushed for
    /// each `ParamId`, consulted (and populated) by [`LeafOrigin::resolve`].
    /// Cleared alongside `entries` on every [`clear`](Tape::clear) call —
    /// that is precisely what forces a stale `CudaParam`/checkpoint
    /// boundary reference to push (and cache) a *fresh* `Leaf` entry the
    /// next time it's used, instead of reusing an index from a
    /// since-cleared generation.
    pub(crate) leaf_by_param: HashMap<ParamId, NodeId>,
}

impl Tape {
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
            generation: 0,
            leaf_by_param: HashMap::new(),
        }
    }

    /// Append `op` to the tape and return the `NodeId` assigned to the
    /// operation's output.
    pub(crate) fn push(&mut self, op: TapeOp) -> NodeId {
        let id = NodeId {
            generation: self.generation,
            index: self.entries.len() as u32,
        };
        self.entries.push(op);
        id
    }

    /// Return the `TapeOp` for `id`.
    ///
    /// # Panics
    /// Panics if `id` is not from the tape's current generation, or its
    /// `index` is out-of-bounds within that generation — both indicate a
    /// bug in the tape recording/lookup logic (a `NodeId` from an
    /// already-cleared generation reached here instead of being treated
    /// as unreachable by the caller, e.g. via a `grads.remove(&nid)` miss
    /// in `backward()`'s reverse walk).
    pub(crate) fn get(&self, id: NodeId) -> &TapeOp {
        debug_assert_eq!(
            id.generation, self.generation,
            "Tape::get: NodeId from a stale generation ({} vs current {}) — caller should have \
             treated this as unreachable rather than indexing into the current tape",
            id.generation, self.generation
        );
        &self.entries[id.index as usize]
    }

    /// Return the number of recorded ops on the tape.
    ///
    /// Exposed for testing (e.g. verifying that `backward` clears the tape).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Clear all recorded ops and start a new generation.
    ///
    /// Called automatically at the end of `backward()`.  Can also be
    /// invoked via `CudaBackend::reset_tape` in Ctrl-C handlers or test
    /// setup to discard a partially-built graph.
    ///
    /// Bumping `generation` (not just emptying `entries`) is what makes
    /// stale `NodeId`s captured before this call detectably invalid
    /// afterwards — see [`NodeId`]'s doc comment.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.leaf_by_param.clear();
        self.generation = self.generation.wrapping_add(1);
    }
}

// ---- LeafOrigin -------------------------------------------------------------

/// Back-reference recorded on a `CudaTensor` that originated from
/// `CudaBackend::register_leaf` — i.e. any `CudaParam`'s current value, or
/// an activation-checkpoint boundary created via `Ops::param_from_tensor`
/// (`pm_core::checkpoint::forward_checkpointed` wraps a block's output in
/// exactly such a boundary, then clones its tensor as the *next* segment's
/// recompute input — see `checkpoint.rs`'s module doc, F.6).
///
/// `CudaTensor::node_id()` resolves through this (see that method) instead
/// of trusting a plain cached `NodeId`, specifically so that a value held
/// across *multiple* `CudaBackend::backward()` calls — which is exactly
/// what happens once per `pm_core::checkpoint_backward` segment within a
/// single checkpointed training step — keeps routing gradient correctly
/// even though every intervening `backward()` call clears the tape (bumps
/// its generation) and would otherwise make the cached id dangling.
///
/// This mirrors what `CudaBackend::sgd_step`/`assign` already do eagerly
/// right after every `backward()` (push a fresh `Leaf` so the *next* step
/// can route gradients) — `resolve` just does the same thing lazily, on
/// first use in a new generation, which is the only way to cover
/// `checkpoint_backward`'s multiple-`backward()`-calls-per-step pattern
/// without `pm-core` (which is backend-agnostic and correct — the same
/// mechanism works untouched on `pm-candle`) needing to know about
/// pm-cuda's tape generations at all.
///
/// Re-registering the same `param_id` multiple times within one
/// generation (e.g. because several distinct clones of the same
/// checkpoint boundary are each queried once) is safe: `BackwardOp::Leaf`
/// *accumulates* into `param_grads` (`+=`, never overwrites — see
/// `apply_vjp` in `ops_impl.rs`), and the `leaf_by_param` cache below
/// ensures only the *first* query per generation actually pushes a new
/// entry — later queries for the same `param_id` reuse it.
pub(crate) struct LeafOrigin {
    pub(crate) tape: Arc<Mutex<Tape>>,
    pub(crate) param_id: ParamId,
}

impl LeafOrigin {
    /// Return a `NodeId` guaranteed to resolve to a
    /// `TapeOp::Leaf { param_id: self.param_id }` entry in the tape's
    /// *current* generation, pushing (and caching) one if none exists yet.
    pub(crate) fn resolve(&self) -> NodeId {
        let mut tape = self
            .tape
            .lock()
            .expect("autograd tape lock poisoned — prior panic in backward?");
        if let Some(&nid) = tape.leaf_by_param.get(&self.param_id) {
            return nid;
        }
        let nid = tape.push(TapeOp::Leaf {
            param_id: self.param_id,
        });
        tape.leaf_by_param.insert(self.param_id, nid);
        nid
    }
}
