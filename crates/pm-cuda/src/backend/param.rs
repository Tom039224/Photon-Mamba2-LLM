//! Trainable parameter type for `CudaBackend`.
//!
//! B4.3a: integrates with the autograd tape and supports in-place SGD updates.
//!
//! Design decision — `Arc<ParamInner>` with `UnsafeCell<CudaTensor>`
//! ----------------------------------------------------------------
//! `pm_core::Param::as_tensor(&self) -> &Self::Tensor` requires returning a
//! `&CudaTensor` with the *lifetime of `self`*.  This rules out `Mutex<T>`
//! (the guard's lifetime is shorter than self).  We instead use `UnsafeCell`
//! wrapped in an `Arc` so that:
//!
//! 1. `CudaParam` is `Clone` (Arc bump, no data copy).
//! 2. `as_tensor()` can return `&CudaTensor` via `unsafe { &*cell.get() }`.
//! 3. `sgd_step` / `assign` can mutate the tensor storage through the cell.
//!
//! Safety invariant (SAFETY tag must appear at every unsafe site):
//! The training loop is *single-threaded* and *sequential*: `forward` →
//! `backward` → `sgd_step` never overlap.  All shared clones of a
//! `CudaParam` within one step observe the same value because there is
//! no concurrent mutation.  Cross-thread use is intentionally prevented at
//! compile time: `CudaParam: !Send + !Sync` (`Arc<T>: Send` requires
//! `T: Send + Sync`, and we do not implement `Sync` for `ParamInner`).
//! Until B4.4 introduces multi-thread training (none today), this catches
//! accidental misuse at the compiler rather than via runtime UB.

use std::cell::UnsafeCell;
use std::sync::Arc;

use pm_core::Param;

use super::CudaTensor;

/// Unique identifier for a trainable parameter in a `CudaBackend`
/// training run.  Used as the key in `CudaGradStore`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamId(pub u64);

/// Inner storage for a `CudaParam`, shared across clones via `Arc`.
pub(crate) struct ParamInner {
    pub(crate) tensor: UnsafeCell<CudaTensor>,
    pub(crate) id: ParamId,
}

// SAFETY: `ParamInner` is only mutated in `sgd_step`/`assign`, which
// are called sequentially from a single-threaded training loop.
// `CudaSlice<f32>` (inside `CudaTensor`) is `Send` (device memory is
// not CPU-thread-local), and our sequential-access invariant means we
// never have concurrent readers while a write is in progress.
unsafe impl Send for ParamInner {}

// Deliberately NOT `Sync`: `UnsafeCell` is `!Sync` by default and we keep it
// that way.  Adding `unsafe impl Sync` would let callers do
//     std::thread::scope(|s| { s.spawn(|| p.as_tensor()); s.spawn(|| sgd(...)); })
// without writing any `unsafe` — that is UB the compiler should reject for us.
// Because `Arc<T>: Send` requires `T: Send + Sync`, this also makes
// `CudaParam: !Send`.  Training is single-threaded (CLAUDE.md), so this is
// acceptable until B4.4 introduces a multi-thread driver — at which point
// `Mutex<CudaTensor>` (slower) or a callback-based `Param` API would be the
// sound replacement.

/// A trainable parameter for `CudaBackend`.
///
/// Cheap to clone (Arc bump only — the tensor value is shared).
/// In-place updates via `sgd_step`/`assign` are visible through all
/// clones immediately because they share the same `Arc<ParamInner>`.
#[derive(Clone)]
pub struct CudaParam {
    pub(crate) inner: Arc<ParamInner>,
}

impl CudaParam {
    pub(crate) fn new(tensor: CudaTensor, id: ParamId) -> Self {
        // `ParamInner` is deliberately `!Sync` (see module docs above) so
        // that `Arc<ParamInner>` is intentionally `!Send`/`!Sync` too.
        #[allow(clippy::arc_with_non_send_sync)]
        Self {
            inner: Arc::new(ParamInner {
                tensor: UnsafeCell::new(tensor),
                id,
            }),
        }
    }

    /// Return the unique `ParamId` for this parameter.
    pub fn param_id(&self) -> ParamId {
        self.inner.id
    }

    /// Replace the stored tensor value.
    ///
    /// # Safety
    /// Must only be called from a single-threaded context where no
    /// concurrent reader of `as_tensor()` exists (i.e. after `backward`
    /// completes and before the next `forward`).
    pub(crate) unsafe fn set_tensor(&self, t: CudaTensor) {
        // SAFETY: caller guarantees sequential, non-concurrent access.
        unsafe { *self.inner.tensor.get() = t };
    }
}

impl Param for CudaParam {
    type Tensor = CudaTensor;

    fn as_tensor(&self) -> &CudaTensor {
        // SAFETY: sequential single-threaded training loop; no concurrent
        // mutation while this reference is live.
        unsafe { &*self.inner.tensor.get() }
    }
}
