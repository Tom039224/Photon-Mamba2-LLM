#!/usr/bin/env python3
"""Stream FineWeb-Edu documents into a FIFO (or stdout) for Phase D.4.

Phase D.4 trains PHOTON α=0 and flat Mamba2 (both ~102M params) on 60M
tokens of general-domain text, streamed live from HuggingFace's
`HuggingFaceFW/fineweb-edu` (config `sample-10BT`) — per user requirement
the corpus must never be persisted to disk as a corpus file ("垂れ流し").

The trick: this script writes one cleaned document per line to a POSIX
FIFO (named pipe) created by `scripts/d4_chain.sh`. `pm-data`'s
`TextFileSource::open` (crates/pm-data/src/text_source.rs) already reads
its `text_data` path lazily, line-by-line, via `BufReader::read_line`, and
`PackedBatcher::next_batch` (crates/pm-data/src/packing.rs) drains exactly
`batch_size * (seq_len + 1)` tokens per training step. Neither Rust file
needs to change — a FIFO satisfies `std::fs::File::open` and blocking
reads exactly like a regular file, so the existing code path streams
FineWeb-Edu with zero Rust changes and nothing ever touches disk as a
corpus artifact.

Cleaning: each document's `text` field has literal `\\r` and `\\n`
replaced with a single space each, then is `.strip()`-ped; empty results
are skipped. One document -> one output line -> `TextFileSource` inserts
exactly one `doc_sep_id` (GPT-2 EOS, 50256) token per document. This is a
deliberate deviation — see docs/deviations.md entry P.8.

Determinism: `load_dataset(..., streaming=True)` is never `.shuffle()`d,
so iteration order is a deterministic function of FineWeb-Edu's shard
layout. Two independent invocations of this script must therefore emit
an identical prefix (verified by the D.4 smoke tests via sha256).

Robustness for a 23-37h unattended run: a background reader thread pulls
raw examples from the HF iterable dataset into a bounded in-memory queue
(capped by total buffered characters, not item count, since document
sizes vary by orders of magnitude) that the main thread drains into the
output stream. The bounded queue is the backpressure mechanism — if the
Rust trainer (or a `head`/`sha256sum` pipe in the smoke test) is slower
than HF can serve documents, the queue fills and the reader thread blocks
in `put()` until space frees up. On any *transient* exception from the HF
iterator (network blip, hub hiccup, etc.) the reader reconnects by
recreating the streaming dataset and fast-forwarding past the exact
number of raw examples already consumed in this process's lifetime — this
guarantees no document is skipped or repeated across a reconnect.

Give-up watchdog (D.4 review FIX 1 — prevents a multi-day silent hang):
if the producer *permanently* fails (bad dataset name, `datasets`
ImportError, or HF unreachable for a long time) the reader would
otherwise retry forever, never emit a doc, and the Rust trainer would
block in `read_line` on the FIFO indefinitely — its
`max_wall_time_seconds` is only checked *between completed steps*, so it
never fires. To bound this, a dedicated watchdog thread tracks time since
the last producer *progress* and, after `--giveup-seconds` (default 900)
with no progress — OR after too many consecutive reconnect failures, OR
immediately on an `ImportError` (datasets not installed) — logs a FATAL
line and `close()`s the queue. The main thread's `queue.get()` then
returns None, the FIFO is closed, and the Rust trainer sees EOF
(`read_line` -> 0) -> `next_batch` returns None -> training stops cleanly
("data exhausted") instead of hanging.

  Backpressure vs. stall: "progress" is refreshed both on a successful
  fetch/emit AND *while the reader is blocked in `put()` on a full queue*
  (healthy backpressure — the producer is alive and ready, the consumer
  is just slow). The watchdog therefore only fires when the reader is
  genuinely stuck *fetching from HF* (retry loop, or a hung `next()`),
  never during a legitimately slow but working trainer. This is a
  deliberate refinement of "time since last successful put" that avoids
  falsely aborting a healthy multi-day run.

This script does NOT tokenize; Rust does that. The token budget is
enforced entirely by the trainer's `n_steps` (see configs/d4_*.toml).

Usage:
    .venv-d4/bin/python scripts/stream_fineweb.py --fifo runtime/d4_corpus.fifo
    .venv-d4/bin/python scripts/stream_fineweb.py --stdout --max-docs 200
"""

from __future__ import annotations

import argparse
import collections
import errno
import os
import sys
import threading
import time
from typing import Callable, Optional

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_SPLIT = "train"

# Bounded in-memory queue: caps buffered-but-unwritten document text by
# total character count (~5M chars), not item count. Document sizes swing
# from a few hundred to tens of thousands of characters, so an item-count
# bound (plain `queue.Queue(maxsize=N)`) doesn't give a predictable memory
# ceiling; a character-count bound does.
QUEUE_CHAR_CAP = 5_000_000

# Reconnect backoff schedule (seconds) when the HF stream raises.
RETRY_BACKOFF_SECONDS = (1, 2, 5, 10, 30, 60)

# Secondary give-up trigger: bail after this many *consecutive* reconnect
# failures even if the time-based watchdog hasn't fired yet. With the
# backoff schedule above, 8 consecutive failures span ~3 min of dead
# time — a strong signal HF is not coming back on its own.
MAX_CONSECUTIVE_RECONNECTS = 8


class _SimulatedReconnect(Exception):
    """Internal signal used only by --simulate-reconnect-after (FIX 3)."""


class _BoundedCharQueue:
    """Thread-safe queue bounded by total buffered character count.

    Provides the backpressure between the HF reader thread (producer) and
    the FIFO/stdout writer (consumer, the main thread) that a plain
    `queue.Queue` can't give when item sizes vary this much.
    """

    def __init__(self, max_chars: int) -> None:
        self._max_chars = max_chars
        self._items: "collections.deque[str]" = collections.deque()
        self._chars = 0
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)
        self._closed = False

    def put(self, text: str, heartbeat: Optional[Callable[[], None]] = None) -> None:
        """Enqueue `text`, blocking (with backpressure) while full.

        `heartbeat`, if given, is called on every wait iteration while
        blocked on a full queue so the give-up watchdog can distinguish
        healthy backpressure (consumer slow, producer alive) from a real
        producer stall (stuck fetching from HF).
        """
        with self._not_full:
            while self._chars >= self._max_chars and not self._closed:
                self._not_full.wait(timeout=1.0)
                if heartbeat is not None:
                    heartbeat()
            if self._closed:
                return
            self._items.append(text)
            self._chars += len(text)
            self._not_empty.notify()

    def get(self) -> Optional[str]:
        with self._not_empty:
            while not self._items and not self._closed:
                self._not_empty.wait(timeout=1.0)
            if not self._items:
                return None
            text = self._items.popleft()
            self._chars -= len(text)
            self._not_full.notify()
            return text

    def buffered_chars(self) -> int:
        with self._lock:
            return self._chars

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_full.notify_all()
            self._not_empty.notify_all()


def _clean_doc(text: str) -> str:
    """Flatten one FineWeb-Edu document to a single line.

    Replaces literal CR/LF with a single space each and strips leading /
    trailing whitespace. Internal whitespace runs are left untouched.
    """
    return text.replace("\r", " ").replace("\n", " ").strip()


def _open_stream(resume_skip: int, force_fatal: bool = False):
    """(Re)create the FineWeb-Edu streaming iterator and fast-forward it.

    `resume_skip` is the number of *raw* examples already pulled from the
    dataset in this process's lifetime, including ones that turned out
    empty after `_clean_doc` and were never emitted. Skipping exactly that
    many on reconnect keeps a network blip from re-emitting or dropping
    documents, since FineWeb-Edu's shard order is stable and no shuffle is
    ever applied.

    `force_fatal` is a test-only hook (--force-fatal): it raises a
    non-ImportError exception on every attempt, simulating a permanently
    unreachable HF so the give-up watchdog / reconnect-failure path can be
    exercised without touching the network.
    """
    if force_fatal:
        raise RuntimeError(
            "--force-fatal: simulated permanent HF failure (D.4 test hook)"
        )
    from datasets import load_dataset  # lazy import: keep --help fast/offline-safe

    ds = load_dataset(
        DATASET_NAME, name=DATASET_CONFIG, split=DATASET_SPLIT, streaming=True
    )
    it = iter(ds)
    for _ in range(resume_skip):
        next(it)
    return it


def _give_up(
    queue: "_BoundedCharQueue",
    stop_event: threading.Event,
    state: dict,
    reason: str,
) -> None:
    """Fatal give-up: log, flag, unblock the main thread, so the trainer EOFs."""
    print(
        f"[stream_fineweb] FATAL: {reason}; giving up. Closing the stream so "
        f"the trainer sees EOF and stops cleanly instead of hanging.",
        file=sys.stderr,
        flush=True,
    )
    state["gave_up"] = True
    stop_event.set()
    queue.close()


def _watchdog_thread(
    queue: "_BoundedCharQueue",
    stop_event: threading.Event,
    state: dict,
    giveup_seconds: float,
) -> None:
    """Give-up watchdog: fire if the producer makes no progress for too long.

    `state['last_activity']` (monotonic seconds) is refreshed by the reader
    on every fetch/emit and while blocked on backpressure. If it goes stale
    for longer than `giveup_seconds`, the producer is genuinely stuck
    fetching from HF (never connected, or stalled mid-run) and we give up.
    """
    while not stop_event.wait(1.0):
        idle = time.monotonic() - state["last_activity"]
        if idle > giveup_seconds:
            _give_up(
                queue,
                stop_event,
                state,
                f"no producer progress for {idle:.0f}s "
                f"(>{giveup_seconds:.0f}s --giveup-seconds); "
                f"emitted={state['emitted']} raw_seen={state['raw_seen']}",
            )
            return


def _backoff(state: dict, fail_count: int, stop_event: threading.Event, what: str) -> None:
    """Interruptible backoff sleep after a transient failure."""
    wait_s = RETRY_BACKOFF_SECONDS[min(fail_count - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
    print(
        f"[stream_fineweb] {what}; retry "
        f"{fail_count}/{MAX_CONSECUTIVE_RECONNECTS} in {wait_s}s "
        f"(resuming after {state['raw_seen']} raw docs)",
        file=sys.stderr,
        flush=True,
    )
    stop_event.wait(wait_s)  # interruptible: watchdog / shutdown can wake us


def _reader_thread(
    queue: "_BoundedCharQueue",
    stop_event: threading.Event,
    max_docs: Optional[int],
    log_every: int,
    state: dict,
    force_fatal: bool,
    simulate_reconnect_after: Optional[int],
) -> None:
    """Background thread: pull docs from HF, clean, push to the bounded queue.

    Reconnects on any *transient* exception by recreating the streaming
    dataset and resuming from `state['raw_seen']` (see `_open_stream`).
    Gives up (clean shutdown, not a hang) on: `ImportError` (datasets not
    installed) immediately; or `MAX_CONSECUTIVE_RECONNECTS` consecutive
    failures. The time-based give-up is handled by `_watchdog_thread`.
    """
    fail_count = 0

    def heartbeat() -> None:
        state["last_activity"] = time.monotonic()

    while not stop_event.is_set():
        # --- (re)connect --------------------------------------------------
        try:
            it = _open_stream(state["raw_seen"], force_fatal)
        except ImportError as exc:
            _give_up(
                queue, stop_event, state,
                f"`datasets` import failed ({exc!r}) — cannot stream",
            )
            return
        except Exception as exc:  # noqa: BLE001 - any connect failure is transient
            if stop_event.is_set():
                break
            fail_count += 1
            if fail_count > MAX_CONSECUTIVE_RECONNECTS:
                _give_up(
                    queue, stop_event, state,
                    f"{fail_count - 1} consecutive reconnect failures "
                    f"(last: {exc!r})",
                )
                return
            _backoff(state, fail_count, stop_event, f"connect error: {exc!r}")
            continue

        fail_count = 0  # successful (re)connect

        # --- stream -------------------------------------------------------
        try:
            for example in it:
                if stop_event.is_set():
                    break
                state["raw_seen"] += 1
                text = example.get("text", "") if isinstance(example, dict) else ""
                cleaned = _clean_doc(text)
                state["last_activity"] = time.monotonic()
                if cleaned:
                    queue.put(cleaned, heartbeat=heartbeat)
                    state["emitted"] += 1
                    state["last_activity"] = time.monotonic()
                    if log_every and state["emitted"] % log_every == 0:
                        print(
                            f"[stream_fineweb] emitted={state['emitted']} "
                            f"raw_seen={state['raw_seen']} "
                            f"queue_buffered={queue.buffered_chars()}chars",
                            file=sys.stderr,
                            flush=True,
                        )
                    if max_docs and state["emitted"] >= max_docs:
                        stop_event.set()
                        break
                # FIX 3: force exactly one reconnect for the resume-path test.
                if (
                    simulate_reconnect_after
                    and not state["reconnect_done"]
                    and state["raw_seen"] >= simulate_reconnect_after
                ):
                    state["reconnect_done"] = True
                    raise _SimulatedReconnect()
            else:
                # Dataset genuinely exhausted. sample-10BT dwarfs any D.4
                # budget, but handle it rather than spinning.
                print("[stream_fineweb] dataset exhausted", file=sys.stderr, flush=True)
                stop_event.set()
        except _SimulatedReconnect:
            print(
                f"[stream_fineweb] simulate-reconnect: forcing one reconnect "
                f"after {state['raw_seen']} raw docs (resume-path test)",
                file=sys.stderr,
                flush=True,
            )
            continue  # no backoff, no fail_count: this is a deliberate test drop
        except Exception as exc:  # noqa: BLE001 - mid-stream failure is transient
            if stop_event.is_set():
                break
            fail_count += 1
            if fail_count > MAX_CONSECUTIVE_RECONNECTS:
                _give_up(
                    queue, stop_event, state,
                    f"{fail_count - 1} consecutive stream failures "
                    f"(last: {exc!r})",
                )
                return
            _backoff(state, fail_count, stop_event, f"stream error: {exc!r}")
            continue

    queue.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream FineWeb-Edu (sample-10BT), one cleaned document per "
            "line, into a FIFO for Phase D.4."
        ),
    )
    out_group = parser.add_mutually_exclusive_group(required=True)
    out_group.add_argument(
        "--fifo", type=str, help="Path to a FIFO (named pipe) to write into."
    )
    out_group.add_argument(
        "--stdout",
        action="store_true",
        help="Write to stdout instead of a FIFO (smoke test / determinism check).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Stop after emitting this many documents (default: unlimited).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Progress line to stderr every N emitted docs (0 disables).",
    )
    parser.add_argument(
        "--giveup-seconds",
        type=float,
        default=900.0,
        help=(
            "Give up (clean shutdown -> trainer EOF) if the producer makes "
            "no progress for this many seconds (default: 900). Prevents a "
            "permanent HF failure from hanging the trainer forever."
        ),
    )
    parser.add_argument(
        "--simulate-reconnect-after",
        type=int,
        default=None,
        help=(
            "TEST: force exactly one reconnect (drop + reopen + skip) after "
            "N raw docs, to verify the resume path is byte-identical."
        ),
    )
    # Hidden test hook: simulate a permanently-unreachable HF.
    parser.add_argument("--force-fatal", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    queue = _BoundedCharQueue(QUEUE_CHAR_CAP)
    stop_event = threading.Event()
    state = {
        "raw_seen": 0,
        "emitted": 0,
        "chars_written": 0,
        "last_activity": time.monotonic(),
        "gave_up": False,
        "reconnect_done": False,
    }

    reader = threading.Thread(
        target=_reader_thread,
        args=(
            queue,
            stop_event,
            args.max_docs,
            args.log_every,
            state,
            args.force_fatal,
            args.simulate_reconnect_after,
        ),
        daemon=True,
        name="fineweb-reader",
    )
    watchdog = threading.Thread(
        target=_watchdog_thread,
        args=(queue, stop_event, state, args.giveup_seconds),
        daemon=True,
        name="fineweb-watchdog",
    )
    reader.start()
    watchdog.start()

    if args.stdout:
        out = sys.stdout
    else:
        # Opening for write blocks until a reader (the Rust trainer, via
        # TextFileSource::open) opens the other end — this is expected.
        print(
            f"[stream_fineweb] opening FIFO {args.fifo} for writing "
            "(blocks until a reader opens it)...",
            file=sys.stderr,
            flush=True,
        )
        out = open(args.fifo, "w", encoding="utf-8")  # noqa: SIM115 - lifetime = whole script

    try:
        while True:
            doc = queue.get()
            if doc is None:
                break
            try:
                out.write(doc)
                out.write("\n")
                state["chars_written"] += len(doc) + 1
            except BrokenPipeError:
                stop_event.set()
                if out is sys.stdout:
                    # Standard idiom: redirect stdout's fd to /dev/null so
                    # Python's atexit flush of sys.stdout doesn't print a
                    # noisy "Exception ignored" traceback on shutdown.
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, sys.stdout.fileno())
                print(
                    "[stream_fineweb] reader closed the pipe; exiting cleanly",
                    file=sys.stderr,
                    flush=True,
                )
                break
            except OSError as exc:
                if exc.errno == errno.EPIPE:
                    stop_event.set()
                    print(
                        "[stream_fineweb] EPIPE; exiting cleanly",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                raise
    finally:
        stop_event.set()
        queue.close()
        try:
            out.flush()
        except OSError:
            pass
        if out is not sys.stdout:
            try:
                out.close()
            except OSError:
                pass

    print(
        f"[stream_fineweb] done: emitted={state['emitted']} docs, "
        f"raw_seen={state['raw_seen']}, "
        f"~{state['chars_written'] / 1_000_000:.1f}MB written"
        + (" [GAVE UP]" if state["gave_up"] else ""),
        file=sys.stderr,
        flush=True,
    )
    # Non-zero exit signals the operator that this run ended on a give-up
    # rather than a clean max-docs / exhaustion / consumer-close.
    return 1 if state["gave_up"] else 0


if __name__ == "__main__":
    sys.exit(main())
