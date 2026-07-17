//! Env-gated per-op wall-time profiler for `CudaBackend` (Phase B'.1b,
//! `PLAN.md` "Phase B'"/`HANDOFF.md`).
//!
//! `pm-cuda` runs ~13-19x slower than Candle per training step; matmul is
//! already cuBLAS (B4.2a), so the offending op(s) must be elsewhere
//! (elementwise / reduce / gather / SSD kernel / launch overhead / ckpt
//! recompute — all unprofiled prior to this module, per PLAN.md B'.1).
//! This module answers "where" by wrapping every `Ops` method call in a
//! GPU-synchronized wall-clock timer and aggregating
//! `(phase, op) -> (calls, total_ns)` in a process-global registry,
//! rendered via [`report`] at the end of a training run
//! (`pm-cli::train_cmd`).
//!
//! # Activation
//!
//! Fully opt-in via `PM_CUDA_PROFILE=1` ([`is_enabled`], checked once and
//! cached in an `AtomicU8`): every call in a normal, non-profiled run —
//! i.e. every `Ops` method on `CudaBackend` — costs one relaxed atomic
//! load and a branch; no sync, no allocation, no behavioural change.
//! `PM_CUDA_PROFILE_SKIP=N` (default 0) discards the first `N` training
//! steps (step 0 is dominated by one-time PTX JIT compilation, not
//! steady-state cost) — no synchronization happens during skipped steps
//! either, so skipping is itself close to free. `PM_CUDA_PROFILE_OUT=
//! /path.tsv` additionally dumps the raw `(phase, op, calls, total_ns)`
//! rows as machine-readable TSV.
//!
//! # Phase state machine
//!
//! A training step is `fwd` (model forward) -> `bwd` (`Ops::backward`)
//! -> `post` (grad-clip + optimizer — implemented in `pm-train` purely in
//! terms of generic `Ops` calls, so there is no dedicated op name to key
//! off; everything between `backward()`'s exit and the next step's first
//! `embedding` call is bucketed as `post`). The state machine transitions
//! `fwd -> bwd` when [`enter_backward`] is called, `bwd -> post` when
//! that guard drops, and `post -> fwd` (counted as one completed step)
//! the next time `embedding` — always the first op of a forward pass —
//! is timed; see `maybe_advance_step`.
//!
//! # Nesting
//!
//! Several `Ops` methods call other `Ops` methods internally (e.g.
//! `cumsum`/`log_softmax`/`gather` call `transpose` for non-last-dim
//! inputs; `ssd_scan`'s small-`T` fallback and the SSD backward's
//! pure-Ops recompute each call a dozen primitive ops). Timing both the
//! outer and inner call would double-count wall time when summed. A
//! thread-local depth counter ([`DEPTH`]) ensures only the outermost call
//! in any nested chain is actually timed (see `enter_scoped`); inner
//! calls still get a (cheap, no-sync) guard so the depth counter stays
//! balanced — they just don't record.
//!
//! `Ops::backward`'s own timer (op name `backward_total`, via
//! [`enter_backward`]) is the one deliberate exception: it must coexist
//! with the per-VJP (`vjp:<Variant>`) breakdown recorded from inside the
//! reverse tape walk (`ops_impl::apply_vjp`), so it does **not**
//! participate in the shared depth counter — it has its own dedicated
//! code path, sound because `Ops::backward` is only ever called once per
//! step, at true top level, never reentrantly.

use std::cell::Cell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use cudarc::driver::CudaStream;

// ---- Enable/disable gate ----------------------------------------------------

const STATE_UNINIT: u8 = 0;
const STATE_ON: u8 = 1;
const STATE_OFF: u8 = 2;

static ENABLED_STATE: AtomicU8 = AtomicU8::new(STATE_UNINIT);

/// Whether the profiler is active for this process.
///
/// Reads `PM_CUDA_PROFILE` from the environment at most once per process
/// (races between concurrent first-callers just repeat the same read and
/// store the same value — benign) and caches the result in
/// `ENABLED_STATE`, so every subsequent call is a single relaxed atomic
/// load. This is the only cost [`timer`] imposes on the hot path when
/// profiling is off.
fn is_enabled() -> bool {
    match ENABLED_STATE.load(Ordering::Relaxed) {
        STATE_ON => true,
        STATE_OFF => false,
        _ => {
            let on = std::env::var("PM_CUDA_PROFILE").ok().as_deref() == Some("1");
            ENABLED_STATE.store(if on { STATE_ON } else { STATE_OFF }, Ordering::Relaxed);
            on
        }
    }
}

/// Force the profiler on without touching the environment.
///
/// `std::env::set_var` mutates process-global state, which races against
/// `cargo test`'s default parallel test harness; tests call this instead
/// so enabling the profiler for one test can't leak into (or be
/// clobbered by) unrelated tests sharing the process.
#[doc(hidden)]
pub fn force_enable_for_tests() {
    ENABLED_STATE.store(STATE_ON, Ordering::Relaxed);
}

static SKIP_PARSE_WARNED: AtomicBool = AtomicBool::new(false);

/// Number of leading training steps to discard (see the module docs'
/// "Activation" section). Read once and cached; default `0`.
///
/// If `PM_CUDA_PROFILE_SKIP` is set but cannot be parsed as `u64`, a one-
/// time warning is printed to stderr and the value defaults to `0` — the
/// same warn-once convention used for registry-mutex poison and TSV write
/// failures elsewhere in this module.
fn skip_steps() -> u64 {
    static SKIP: OnceLock<u64> = OnceLock::new();
    *SKIP.get_or_init(|| match std::env::var("PM_CUDA_PROFILE_SKIP") {
        Err(_) => 0, // var not set: normal case
        Ok(v) => match v.parse::<u64>() {
            Ok(n) => n,
            Err(_) => {
                if !SKIP_PARSE_WARNED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "pm-cuda profiler: PM_CUDA_PROFILE_SKIP={v:?} is not a \
                         valid u64; defaulting to 0 (no warm-up skip)"
                    );
                }
                0
            }
        },
    })
}

/// Optional TSV dump path (`PM_CUDA_PROFILE_OUT`). Read once and cached.
fn out_path() -> Option<PathBuf> {
    static OUT: OnceLock<Option<PathBuf>> = OnceLock::new();
    OUT.get_or_init(|| std::env::var_os("PM_CUDA_PROFILE_OUT").map(PathBuf::from))
        .clone()
}

// ---- Phase state machine ----------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fwd,
    Bwd,
    Post,
}

impl Phase {
    fn as_str(self) -> &'static str {
        match self {
            Phase::Fwd => "fwd",
            Phase::Bwd => "bwd",
            Phase::Post => "post",
        }
    }
}

thread_local! {
    /// Nesting depth for ordinary (non-`backward_total`) timed scopes.
    /// Only the outermost (`depth_before == 0`) scope actually times —
    /// see the module docs' "Nesting" section.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    /// Current step phase; see the module docs' "Phase state machine".
    static PHASE: Cell<Phase> = const { Cell::new(Phase::Fwd) };
    /// Completed-step counter, advanced by `maybe_advance_step`.
    static STEP: Cell<u64> = const { Cell::new(0) };
}

/// `embedding` is always the first op of a forward pass (the model has
/// no other entry point into `Ops`), so seeing it while the state
/// machine is still `Post` (left over from the previous step's
/// `backward()` exit) means a new step has begun.
fn maybe_advance_step(op: &'static str) {
    if op != "embedding" {
        return;
    }
    PHASE.with(|p| {
        if p.get() == Phase::Post {
            p.set(Phase::Fwd);
            STEP.with(|s| s.set(s.get() + 1));
        }
    });
}

// ---- Registry ----------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct OpStat {
    calls: u64,
    total_ns: u64,
}

/// `(phase, op)`.
type RegistryKey = (&'static str, &'static str);

fn registry() -> &'static Mutex<HashMap<RegistryKey, OpStat>> {
    static REGISTRY: OnceLock<Mutex<HashMap<RegistryKey, OpStat>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

static POISON_WARNED: AtomicBool = AtomicBool::new(false);

/// Print a one-time warning to stderr the first time the registry mutex
/// is found poisoned (i.e. a prior panic happened while it was locked).
/// Never panics itself — recording is simply skipped from then on.
fn warn_poisoned_once() {
    if !POISON_WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "pm-cuda profiler: registry mutex poisoned by a prior panic; \
             further recordings are skipped for the rest of the process"
        );
    }
}

static SYNC_WARNED: AtomicBool = AtomicBool::new(false);

/// Print a one-time warning to stderr the first time `stream.synchronize()`
/// fails inside a `TimerGuard::drop`. Panicking in `Drop` is forbidden
/// (a double-panic aborts the process without unwinding), so we suppress
/// the error and warn once that elapsed times may be inaccurate.
fn warn_sync_failed_once() {
    if !SYNC_WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "pm-cuda profiler: stream.synchronize() failed during TimerGuard::drop; \
             elapsed times may be inaccurate (further sync failures suppressed)"
        );
    }
}

fn record(phase: &'static str, op: &'static str, ns: u64) {
    let mut guard = match registry().lock() {
        Ok(g) => g,
        Err(_) => {
            warn_poisoned_once();
            return;
        }
    };
    let entry = guard.entry((phase, op)).or_default();
    entry.calls += 1;
    entry.total_ns += ns;
}

fn snapshot_registry() -> Vec<(RegistryKey, OpStat)> {
    match registry().lock() {
        Ok(guard) => guard.iter().map(|(k, v)| (*k, *v)).collect(),
        Err(_) => {
            warn_poisoned_once();
            Vec::new()
        }
    }
}

/// Clear all recorded stats and reset the phase state machine.
///
/// The registry is process-global (shared by every `CudaBackend` clone
/// and every thread), so this is mainly useful for tests that need a
/// clean slate; production code has no reason to call it mid-run.
#[doc(hidden)]
pub fn reset() {
    match registry().lock() {
        Ok(mut guard) => guard.clear(),
        Err(_) => warn_poisoned_once(),
    }
    DEPTH.with(|d| d.set(0));
    PHASE.with(|p| p.set(Phase::Fwd));
    STEP.with(|s| s.set(0));
}

// ---- Timing guard ------------------------------------------------------------

enum GuardKind {
    /// Nested inside another timed scope: no sync, no recording — only
    /// balances `DEPTH` on drop.
    Nested,
    /// Outermost, past warm-up: records elapsed wall time on drop.
    Recording {
        op: &'static str,
        phase: &'static str,
        stream: Arc<CudaStream>,
        start: Instant,
    },
    /// [`enter_backward`]'s dedicated guard. Always flips the phase
    /// `Bwd -> Post` on drop (state-machine bookkeeping runs regardless
    /// of warm-up); only times (`timing: Some`) once past warm-up.
    BackwardTotal {
        timing: Option<(Arc<CudaStream>, Instant)>,
    },
}

/// RAII guard returned by [`timer`] / [`enter_backward`]. Dropping it
/// records the elapsed wall time (if this scope was actually being
/// timed) and performs whatever bookkeeping (depth / phase) the scope
/// owns. Opaque by design — callers only ever bind it to a throwaway
/// local (`let _t = ...;`) and let scope-exit `Drop` do the work.
#[must_use]
pub struct TimerGuard(GuardKind);

impl Drop for TimerGuard {
    fn drop(&mut self) {
        match &self.0 {
            GuardKind::Nested => {
                DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
            }
            GuardKind::Recording {
                op,
                phase,
                stream,
                start,
            } => {
                if stream.synchronize().is_err() {
                    warn_sync_failed_once();
                }
                // `phase`/`op` are `&&'static str` here (match ergonomics,
                // matching `Recording { .. }` through the outer `&self.0`);
                // passing them as-is relies on `&T: Deref<Target = T>`
                // auto-deref coercion at this call site (clippy: explicit
                // `*phase`/`*op` would be redundant with that coercion).
                record(phase, op, start.elapsed().as_nanos() as u64);
                DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
            }
            GuardKind::BackwardTotal { timing } => {
                if let Some((stream, start)) = timing {
                    if stream.synchronize().is_err() {
                        warn_sync_failed_once();
                    }
                    record("bwd", "backward_total", start.elapsed().as_nanos() as u64);
                }
                PHASE.with(|p| p.set(Phase::Post));
            }
        }
    }
}

/// Start (or no-op) a timed scope for `op`. Called once at the top of
/// every `pm_core::Ops` method implementation on `CudaBackend`, and once
/// per reverse-tape-walk VJP application (`ops_impl::apply_vjp`, op name
/// `vjp:<Variant>`).
///
/// Returns `None` when the profiler is disabled (the common case — see
/// the module docs) or while a training step's warm-up steps are being
/// skipped (`PM_CUDA_PROFILE_SKIP`); both are effectively free (one
/// atomic load, no sync, no allocation).
pub fn timer(op: &'static str, stream: &Arc<CudaStream>) -> Option<TimerGuard> {
    if !is_enabled() {
        return None;
    }
    maybe_advance_step(op);
    if STEP.with(Cell::get) < skip_steps() {
        return None;
    }
    Some(enter_scoped(op, stream))
}

fn enter_scoped(op: &'static str, stream: &Arc<CudaStream>) -> TimerGuard {
    let depth_before = DEPTH.with(|d| {
        let v = d.get();
        d.set(v + 1);
        v
    });
    if depth_before > 0 {
        return TimerGuard(GuardKind::Nested);
    }
    let _ = stream.synchronize();
    let phase = PHASE.with(Cell::get).as_str();
    TimerGuard(GuardKind::Recording {
        op,
        phase,
        stream: stream.clone(),
        start: Instant::now(),
    })
}

/// Start the dedicated `backward_total` scope. Called once at the top of
/// `CudaBackend::backward`.
///
/// Unlike [`timer`], this never returns `None` while the profiler is
/// enabled — even during a skipped warm-up step it must still flip the
/// phase state machine `Bwd -> Post` on drop, it just skips the sync +
/// timing (see the module docs' "Phase state machine" / "Nesting").
pub fn enter_backward(stream: &Arc<CudaStream>) -> Option<TimerGuard> {
    if !is_enabled() {
        return None;
    }
    PHASE.with(|p| p.set(Phase::Bwd));
    if STEP.with(Cell::get) < skip_steps() {
        return Some(TimerGuard(GuardKind::BackwardTotal { timing: None }));
    }
    let _ = stream.synchronize();
    Some(TimerGuard(GuardKind::BackwardTotal {
        timing: Some((stream.clone(), Instant::now())),
    }))
}

// ---- Report -------------------------------------------------------------

/// Render the current registry as a human-readable table, and — if
/// `PM_CUDA_PROFILE_OUT` names a path — also dump it as TSV.
///
/// `None` iff the profiler is disabled; `Some` (possibly reporting zero
/// recorded ops, e.g. if the whole run fell inside the warm-up window)
/// whenever `PM_CUDA_PROFILE=1`.
pub fn report() -> Option<String> {
    if !is_enabled() {
        return None;
    }
    let mut rows = snapshot_registry();
    rows.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));

    if let Some(path) = out_path() {
        write_tsv(&path, &rows);
    }

    Some(format_report(&rows))
}

fn pct_of(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64 * 100.0
    }
}

/// `rows` must already be sorted the way the caller wants them printed
/// (`report()` sorts by `total_ns` descending — the single biggest wall-
/// time offender first, matching this module's purpose).
///
/// `backward_total` is an *umbrella row*: it covers roughly the same wall
/// time as Σvjp:* (plus tape-walk + synchronize overhead).  Including it
/// in grand-total / per-phase subtotals would inflate `bwd` by ≈2× and
/// push every `%` value artificially low (H-1). It is therefore excluded
/// from all arithmetic and rendered in a dedicated section at the bottom
/// of the report where `backward_total − Σvjp:*` can be read as the
/// tape-walk overhead sanity-check value.
fn format_report(rows: &[(RegistryKey, OpStat)]) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    if rows.is_empty() {
        let _ = writeln!(out, "PM_CUDA_PROFILE: no operations recorded");
        return out;
    }

    // Partition: umbrella row vs. ordinary rows. Grand-total / subtotal
    // arithmetic applies only to the ordinary set (H-1).
    let regular: Vec<_> = rows
        .iter()
        .filter(|((_, op), _)| *op != "backward_total")
        .collect();
    let bwd_total = rows.iter().find(|((_, op), _)| *op == "backward_total");

    let grand_total_ns: u64 = regular.iter().map(|(_, s)| s.total_ns).sum();

    let _ = writeln!(
        out,
        "{:<6}  {:<24}  {:>8}  {:>10}  {:>10}  {:>6}",
        "phase", "op", "calls", "total_ms", "avg_us", "%"
    );
    for ((phase, op), stat) in &regular {
        let total_ms = stat.total_ns as f64 / 1_000_000.0;
        let avg_us = stat.total_ns as f64 / (stat.calls.max(1) as f64) / 1_000.0;
        let pct = pct_of(stat.total_ns, grand_total_ns);
        let _ = writeln!(
            out,
            "{phase:<6}  {op:<24}  {:>8}  {total_ms:>10.3}  {avg_us:>10.3}  {pct:>5.1}%",
            stat.calls
        );
    }

    let _ = writeln!(out);
    for phase in ["fwd", "bwd", "post"] {
        let subtotal_ns: u64 = regular
            .iter()
            .filter(|((p, _), _)| *p == phase)
            .map(|(_, s)| s.total_ns)
            .sum();
        if subtotal_ns == 0 {
            continue;
        }
        let _ = writeln!(
            out,
            "  subtotal[{phase:<4}] = {:>10.3} ms ({:>5.1}%)",
            subtotal_ns as f64 / 1_000_000.0,
            pct_of(subtotal_ns, grand_total_ns)
        );
    }
    let _ = writeln!(
        out,
        "  grand total  = {:>10.3} ms",
        grand_total_ns as f64 / 1_000_000.0
    );

    // Umbrella section: backward_total is excluded from the numbers above.
    // `backward_total* − Σvjp:*` gives the tape-walk + synchronize
    // overhead — useful for checking that the per-VJP breakdown is
    // complete and that no large cost is hidden in tape management.
    if let Some(((_, _), stat)) = bwd_total {
        let vjp_sum_ns: u64 = regular
            .iter()
            .filter(|((p, op), _)| *p == "bwd" && op.starts_with("vjp:"))
            .map(|(_, s)| s.total_ns)
            .sum();
        let overhead_ns = stat.total_ns.saturating_sub(vjp_sum_ns);
        let _ = writeln!(out);
        let _ = writeln!(out, "--- backward_total* (umbrella, excluded above) ---");
        let _ = writeln!(
            out,
            "  backward_total*      = {:>10.3} ms  ({} calls)",
            stat.total_ns as f64 / 1_000_000.0,
            stat.calls,
        );
        let _ = writeln!(
            out,
            "  backward_total*−Σvjp = {:>10.3} ms  (tape-walk + sync overhead)",
            overhead_ns as f64 / 1_000_000.0,
        );
        let _ = writeln!(
            out,
            "  * umbrella row = Σvjp:* + tape-walk overhead; excluded from subtotals/%"
        );
    }

    out
}

fn write_tsv(path: &Path, rows: &[(RegistryKey, OpStat)]) {
    use std::fmt::Write as _;
    let mut buf = String::from("phase\top\tcalls\ttotal_ns\n");
    for ((phase, op), stat) in rows {
        // Use "bwd*" for the backward_total umbrella row so downstream
        // parsers can filter it out of aggregates — mirrors H-1's
        // exclusion in `format_report`. The `*` suffix is deliberately
        // chosen not to clash with the plain "bwd"/"fwd"/"post" values.
        let phase_col = if *op == "backward_total" {
            "bwd*"
        } else {
            phase
        };
        let _ = writeln!(buf, "{phase_col}\t{op}\t{}\t{}", stat.calls, stat.total_ns);
    }
    if let Err(e) = std::fs::write(path, buf) {
        eprintln!("pm-cuda profiler: failed to write PM_CUDA_PROFILE_OUT={path:?}: {e}");
    }
}

// ---- Tests (GPU-free: exercise only the pure bookkeeping) --------------------
//
// `timer`/`enter_backward` need a real `Arc<CudaStream>` (i.e. a real
// CUDA device), so end-to-end coverage of those lives in the GPU smoke
// test `crates/pm-cuda/tests/profiler_smoke.rs` instead. Everything here
// exercises `record`/`snapshot_registry`/`format_report`/`reset`/
// `maybe_advance_step`/`write_tsv` directly and needs no GPU at all —
// consolidated into one `#[test]` so the process-global registry and
// `ENABLED_STATE` can't race against another test function in this same
// binary (see `force_enable_for_tests`'s doc comment).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiler_bookkeeping() {
        // ---- disabled: report() is None before anything enables it ----
        // Deterministic: `PM_CUDA_PROFILE` is unset for `cargo test`, and
        // this is the only test touching `ENABLED_STATE` in this binary.
        assert!(
            report().is_none(),
            "report() must return None while the profiler is disabled"
        );

        // ---- registry accumulation ----
        force_enable_for_tests();
        reset();

        record("fwd", "unit_test_op", 1_000);
        record("fwd", "unit_test_op", 2_000); // calls=2, total_ns=3_000
        record("bwd", "vjp:UnitTest", 500); //   calls=1, total_ns=500
        record("post", "mid_op", 1_500); //      calls=1, total_ns=1_500
                                         // H-1: umbrella row — must NOT inflate bwd subtotal or grand total
        record("bwd", "backward_total", 10_000);

        let rows = snapshot_registry();
        assert_eq!(rows.len(), 4);
        let stat_of = |phase: &str, op: &str| -> OpStat {
            rows.iter()
                .find(|((p, o), _)| *p == phase && *o == op)
                .map(|(_, s)| *s)
                .unwrap_or_else(|| panic!("{phase}/{op} not recorded"))
        };
        assert_eq!(stat_of("fwd", "unit_test_op").calls, 2);
        assert_eq!(stat_of("fwd", "unit_test_op").total_ns, 3_000);
        assert_eq!(stat_of("bwd", "vjp:UnitTest").calls, 1);
        assert_eq!(stat_of("bwd", "backward_total").total_ns, 10_000);
        assert_eq!(stat_of("post", "mid_op").calls, 1);

        // ---- report(): sorted desc by total_ns, per-phase subtotals, grand total ----
        // H-1: backward_total must appear in the umbrella section (after grand
        // total) and must NOT be counted in bwd subtotal or grand total.
        let text = report().expect("profiler enabled -> Some");
        let pos_unit = text.find("unit_test_op").expect("unit_test_op in report");
        let pos_mid = text.find("mid_op").expect("mid_op in report");
        let pos_vjp = text.find("vjp:UnitTest").expect("vjp:UnitTest in report");
        // regular rows are still sorted 3000 > 1500 > 500:
        assert!(
            pos_unit < pos_mid && pos_mid < pos_vjp,
            "rows must be sorted by total_ns descending (3000 > 1500 > 500):\n{text}"
        );
        assert!(text.contains("subtotal[fwd"));
        assert!(text.contains("subtotal[bwd"));
        assert!(text.contains("subtotal[post"));
        assert!(text.contains("grand total"));
        // H-1: backward_total* umbrella section appears after grand total line
        let pos_grand = text.find("grand total").expect("grand total in report");
        let pos_umbrella = text
            .find("backward_total*")
            .expect("backward_total* must appear in umbrella section");
        assert!(
            pos_grand < pos_umbrella,
            "backward_total* section must follow grand total (excluded from it):\n{text}"
        );
        // H-1: footnote confirms exclusion
        assert!(
            text.contains("excluded from subtotals/%"),
            "umbrella footnote must be present:\n{text}"
        );

        // ---- reset(): clears the registry ----
        reset();
        assert!(snapshot_registry().is_empty());

        // ---- format_report(): empty-registry message, no panic ----
        assert!(report()
            .expect("still enabled -> Some")
            .contains("no operations recorded"));

        // ---- TSV side-output: exercised directly, no env var needed ----
        // (`out_path()` caches `PM_CUDA_PROFILE_OUT` on first read, which
        // would already have resolved to `None` from the `report()` calls
        // above — calling `write_tsv` directly sidesteps that entirely.)
        let tsv_path =
            std::env::temp_dir().join(format!("pm_cuda_profiler_test_{}.tsv", std::process::id()));
        let tsv_rows = [
            (
                ("fwd", "tsv_test_op"),
                OpStat {
                    calls: 3,
                    total_ns: 9_000,
                },
            ),
            // H-1: umbrella row must be emitted with "bwd*" phase in TSV
            (
                ("bwd", "backward_total"),
                OpStat {
                    calls: 1,
                    total_ns: 50_000,
                },
            ),
        ];
        write_tsv(&tsv_path, &tsv_rows);
        let tsv_content = std::fs::read_to_string(&tsv_path).expect("tsv file written");
        assert!(tsv_content.contains("phase\top\tcalls\ttotal_ns"));
        assert!(tsv_content.contains("fwd\ttsv_test_op\t3\t9000"));
        // H-1: downstream parsers identify umbrella row via "bwd*" phase column
        assert!(
            tsv_content.contains("bwd*\tbackward_total\t1\t50000"),
            "backward_total must use 'bwd*' phase in TSV:\n{tsv_content}"
        );
        let _ = std::fs::remove_file(&tsv_path);

        // ---- phase state machine: post -> fwd only on `embedding` ----
        reset();
        maybe_advance_step("embedding"); // starts at Fwd: no `Post` to leave, no-op
        assert_eq!(STEP.with(Cell::get), 0);
        assert!(matches!(PHASE.with(Cell::get), Phase::Fwd));

        PHASE.with(|p| p.set(Phase::Post));
        maybe_advance_step("matmul"); // not `embedding` -> stays Post
        assert!(matches!(PHASE.with(Cell::get), Phase::Post));

        maybe_advance_step("embedding"); // Post -> Fwd, step advances
        assert_eq!(STEP.with(Cell::get), 1);
        assert!(matches!(PHASE.with(Cell::get), Phase::Fwd));

        reset();
    }
}
