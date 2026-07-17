//! `Ops::no_grad_scope` must suppress autograd-tape recording on the
//! CUDA backend (regression test for the D.2b HellaSwag eval OOM,
//! 2026-07-05: eval loops never call `backward`, so without the scope
//! every forward's saved activations pile up on the shared tape until
//! CUDA_ERROR_OUT_OF_MEMORY — tens of 102M-model items sufficed).
//!
//! Recording is gated on `grad_enabled() && operand.node_id().is_some()`
//! (see `Ops::detach`'s doc in `ops_impl.rs`), so the tracked operand
//! here must be a `Param` — plain `from_slice` tensors never record.

#![cfg(feature = "cuda")]

use pm_core::{Ops, Param};
use pm_cuda::CudaBackend;

#[test]
fn no_grad_scope_suppresses_tape_recording() {
    let bk = CudaBackend::new(0).expect("CUDA init");
    let p = bk
        .param_from_slice_f32(&[1.0, 2.0], &[2])
        .expect("param alloc");
    let x = bk.from_slice_f32(&[3.0, 4.0], &[2]).expect("tensor alloc");

    // Baseline: an op on a tracked param outside the scope must record.
    let before = bk.tape_len();
    let _recorded = bk.mul(p.as_tensor(), &x).expect("recorded mul");
    let after_recorded = bk.tape_len();
    assert!(
        after_recorded > before,
        "mul on a tracked param outside no_grad_scope must push tape entries \
         (before={before}, after={after_recorded})"
    );

    // Inside the scope: same op, tape must not grow, value still correct.
    let y = bk
        .no_grad_scope(|| bk.mul(p.as_tensor(), &x))
        .expect("no-grad mul");
    assert_eq!(
        bk.tape_len(),
        after_recorded,
        "mul inside no_grad_scope must not record"
    );
    assert_eq!(
        bk.to_vec_f32(&y).expect("to_vec"),
        vec![3.0, 8.0],
        "no_grad_scope must not change forward values"
    );

    // The previous recording state is restored on exit.
    let _recorded_again = bk.mul(p.as_tensor(), &x).expect("recorded mul 2");
    assert!(
        bk.tape_len() > after_recorded,
        "recording must resume after the scope exits"
    );
}
