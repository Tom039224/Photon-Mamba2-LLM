//! Device-side tensor used by the pm-cuda native backend.
//!
//! B4.1 introduced `CudaTensor` with a single f32 storage.
//! B4.2a extends it with `CudaStorage` to support i64 tensors
//! (required by `from_slice_i64` / `embedding` / `gather`).
//! B4.3a adds `node_id: Option<NodeId>` for reverse-mode autograd.
//!
//! Layout: always contiguous row-major. Stride support is a follow-up.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use pm_core::{Dtype, Tensor};

use super::tape::{LeafOrigin, NodeId};

/// Device-resident storage variant. Cheap to clone (Arc).
#[derive(Clone)]
pub enum CudaStorage {
    F32(Arc<CudaSlice<f32>>),
    I64(Arc<CudaSlice<i64>>),
}

/// Contiguous row-major tensor living on a single CUDA device.
///
/// Cheap to clone: the underlying `CudaSlice` is reference-counted.
/// `node_id` is set by the autograd tape when the tensor is the output
/// of a recorded operation; `None` means the tensor is detached.
///
/// `origin` is `Some(..)` only for the direct output of
/// `CudaBackend::register_leaf` (a `CudaParam`'s current value, or a
/// `Ops::param_from_tensor` checkpoint boundary) and its plain `.clone()`s
/// — never for the output of any *other* op (every other constructor in
/// this file sets it to `None`). See [`LeafOrigin`] and [`node_id`](Self::node_id)
/// for why this exists: it lets those two specific kinds of tensor keep
/// routing gradient correctly even when read after one or more
/// intervening `CudaBackend::backward()` calls (which each clear the
/// shared tape) — the pattern `pm_core::checkpoint_backward` uses.
#[derive(Clone)]
pub struct CudaTensor {
    pub(crate) storage: CudaStorage,
    pub(crate) shape: Vec<usize>,
    /// Autograd tape node, if this tensor is part of a tracked graph.
    /// Ignored (bypassed) when `origin.is_some()` — see `node_id()`.
    pub(crate) node_id: Option<NodeId>,
    pub(crate) origin: Option<Arc<LeafOrigin>>,
}

impl CudaTensor {
    /// Construct from an f32 storage slice. `node_id` is `None` (detached).
    pub(crate) fn new(storage: Arc<CudaSlice<f32>>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            storage.len(),
            shape.iter().product::<usize>(),
            "CudaTensor::new: storage len {} != shape product {:?}",
            storage.len(),
            shape,
        );
        Self {
            storage: CudaStorage::F32(storage),
            shape,
            node_id: None,
            origin: None,
        }
    }

    /// Construct from an i64 storage slice. `node_id` is `None` (detached).
    pub(crate) fn new_i64(storage: Arc<CudaSlice<i64>>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            storage.len(),
            shape.iter().product::<usize>(),
            "CudaTensor::new_i64: storage len {} != shape product {:?}",
            storage.len(),
            shape,
        );
        Self {
            storage: CudaStorage::I64(storage),
            shape,
            node_id: None,
            origin: None,
        }
    }

    /// Return the autograd tape `NodeId` for this tensor, or `None` if detached.
    ///
    /// When `origin.is_some()` (this tensor is a `CudaParam`'s value or a
    /// checkpoint boundary, or a plain clone of one), the cached
    /// `node_id` field is bypassed entirely in favour of
    /// [`LeafOrigin::resolve`], which always returns an id valid in the
    /// tape's *current* generation — self-healing across the multiple
    /// `CudaBackend::backward()` calls `pm_core::checkpoint_backward`
    /// makes per training step. See `tape.rs`'s `NodeId`/`LeafOrigin` docs
    /// for the full rationale (this is the pm-cuda-specific fix for the
    /// 2026-07 "activation checkpointing produces wrong gradients" bug).
    pub(crate) fn node_id(&self) -> Option<NodeId> {
        match &self.origin {
            Some(origin) => Some(origin.resolve()),
            None => self.node_id,
        }
    }

    /// Consume `self` and return a new `CudaTensor` with the given `NodeId`
    /// stamped on it.  The storage and shape are unchanged.
    pub(crate) fn with_node_id(mut self, nid: NodeId) -> Self {
        self.node_id = Some(nid);
        self
    }

    /// Consume `self` and return a new `CudaTensor` stamped with `origin`
    /// — see the type's doc comment and [`LeafOrigin`]. Used only by
    /// `CudaBackend::register_leaf`.
    pub(crate) fn with_origin(mut self, origin: Arc<LeafOrigin>) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Access the f32 storage.
    ///
    /// # Panics
    /// Panics if the storage is not F32. This is an internal invariant
    /// violation — callers must only call this on f32 tensors.
    pub(crate) fn storage(&self) -> &Arc<CudaSlice<f32>> {
        match &self.storage {
            CudaStorage::F32(s) => s,
            CudaStorage::I64(_) => {
                panic!("CudaTensor::storage() called on I64 tensor — internal invariant violation");
            }
        }
    }

    /// Access the i64 storage.
    ///
    /// # Panics
    /// Panics if the storage is not I64.
    pub(crate) fn storage_i64(&self) -> &Arc<CudaSlice<i64>> {
        match &self.storage {
            CudaStorage::I64(s) => s,
            CudaStorage::F32(_) => {
                panic!(
                    "CudaTensor::storage_i64() called on F32 tensor — internal invariant violation"
                );
            }
        }
    }
}

impl CudaTensor {
    /// Return a new `CudaTensor` that shares the same storage but has a
    /// different shape. The caller must ensure `shape` has the same `numel`
    /// as the current tensor (enforced by a `debug_assert`).
    ///
    /// This is a zero-copy metadata-only operation — the `Arc` refcount
    /// is bumped, no data is moved. Used by `reshape`.
    ///
    /// **Note**: `node_id` is intentionally *not* carried over.  The
    /// `reshape` forward op creates a fresh tape entry and stamps the
    /// resulting tensor with a new `NodeId`.  Carrying the old id here
    /// would double-register the node and corrupt the backward pass.
    /// `origin` is likewise dropped: a reshaped view is one autograd step
    /// removed from whatever leaf it aliases data with (its VJP must run
    /// through `Reshape`'s own backward rule to restore shape), so it must
    /// never resolve straight back to `Leaf(param_id)` — see
    /// `CudaTensor::node_id`'s doc comment.
    pub(crate) fn with_shape(&self, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let storage_len = match &self.storage {
            CudaStorage::F32(s) => s.len(),
            CudaStorage::I64(s) => s.len(),
        };
        debug_assert_eq!(
            storage_len, numel,
            "CudaTensor::with_shape: numel mismatch: storage={storage_len} vs new shape={shape:?}"
        );
        Self {
            storage: self.storage.clone(),
            shape,
            node_id: None,
            origin: None,
        }
    }
}

impl Tensor for CudaTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> Dtype {
        match &self.storage {
            CudaStorage::F32(_) => Dtype::F32,
            CudaStorage::I64(_) => Dtype::I64,
        }
    }
}
