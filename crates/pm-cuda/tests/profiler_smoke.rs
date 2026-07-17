//! GPU smoke test for the Phase B'.1b per-op profiler (`pm_cuda::profiler`).
//!
//! Runs a couple of real `Ops` calls through an actual `CudaBackend` and
//! checks the resulting report mentions them with `calls > 0`. Everything
//! else about the profiler's bookkeeping (registry accumulation, report
//! formatting, the phase state machine, TSV output) is covered GPU-free
//! by the unit tests inside `crates/pm-cuda/src/profiler.rs`.
//!
//! Calls go through fully-qualified `Ops::method(&bk, ...)` syntax rather
//! than `bk.method(...)` dot-call syntax: `CudaBackend` keeps a few
//! inherent methods (`zeros`/`from_slice_f32`/`to_vec_f32`/`to_vec_i64`)
//! for old B4.1-era smoke tests that call straight through to the
//! private `*_inner` helper, bypassing `impl Ops for CudaBackend` (and
//! so the profiler instrumentation, which lives only in that `impl`
//! block) entirely — inherent methods always win dot-call resolution
//! over trait methods, `Ops` import or not. Production training code
//! never hits this: `pm-train`/`pm-cli::model_build` is generic over
//! `O: Ops`, so it only ever sees the trait methods below. Using the
//! same fully-qualified form here keeps this test representative of
//! that real call path instead of the back-compat shortcut.
//!
//! A single `#[test]` (rather than several) keeps this file's use of the
//! profiler's process-global registry / `ENABLED_STATE` deterministic —
//! see `pm_cuda::profiler::force_enable_for_tests`'s doc comment.

#![cfg(feature = "cuda")]

use pm_core::Ops;
use pm_cuda::CudaBackend;

#[test]
fn report_shows_recorded_ops_after_forcing_enable() {
    pm_cuda::profiler::force_enable_for_tests();
    pm_cuda::profiler::reset();

    let bk = CudaBackend::new(0).expect("CUDA init");
    let a = Ops::from_slice_f32(&bk, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("from_slice_f32 a");
    let b = Ops::from_slice_f32(&bk, &[5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("from_slice_f32 b");

    Ops::add(&bk, &a, &b).expect("add");
    Ops::add(&bk, &a, &b).expect("add again");
    Ops::mul(&bk, &a, &b).expect("mul");

    let report = pm_cuda::profiler::report().expect("profiler was force-enabled -> Some");

    assert!(
        report.contains("add"),
        "report should mention `add`:\n{report}"
    );
    assert!(
        report.contains("mul"),
        "report should mention `mul`:\n{report}"
    );
    assert!(
        report.contains("from_slice_f32"),
        "report should mention `from_slice_f32`:\n{report}"
    );
    assert!(
        report.contains("grand total"),
        "report should include a grand-total line:\n{report}"
    );

    pm_cuda::profiler::reset();
}
