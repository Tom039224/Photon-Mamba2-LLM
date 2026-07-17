#!/usr/bin/env bash
# Phase D.4 comparison chain: 2 training runs (PHOTON alpha=0, flat) on
# 60.0M FineWeb-Edu tokens each (token-matched), + HellaSwag evals.
#
# Data is streamed live from HuggingFace (scripts/stream_fineweb.py) into
# a FIFO that pm-data's TextFileSource reads exactly like a regular file
# — the corpus is never written to disk (see scripts/stream_fineweb.py's
# module docstring and docs/deviations.md's D.4 entry).
#
# Expected wall time: ~2.5-3 days total on RTX 5070 at B=4 no-ckpt
# (~675-725 tok/s per docs/perf-log.md 2026-07-07 B'.3 wave-2; flat's
# actual per-step tok/s at this envelope is confirmed at run time, not
# assumed to match PHOTON's).
#
# Logs: logs/d4_*.log   Status: logs/d4_status.txt
# Checkpoints: checkpoints/d4_*.safetensors
#
# IMPORTANT (fingerprint pitfall, docs/perf-log.md 2026-07-04): `pm train`
# defaults to the Candle backend if `--backend cuda` is omitted — it does
# NOT error out, it silently trains on the wrong backend. Every train/eval
# invocation below passes `--backend cuda` explicitly.
#
# This script runs the two (name, config) pairs sequentially — a single
# RTX 5070 can only host one training run at a time.
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

BIN=./target/release/pm
PYTHON=.venv-d4/bin/python
PRODUCER=scripts/stream_fineweb.py
FIFO=runtime/d4_corpus.fifo
STATUS=logs/d4_status.txt

mkdir -p checkpoints logs runtime

# Which runs to execute. Defaults to both; pass names (e.g. `photon_a0`) to
# run a subset. Used for the PHOTON-only re-run after the 2026-07-07 OOM:
# `d4_chain.sh photon_a0` retrains only PHOTON and must NOT touch the
# already-COMPLETE flat checkpoint/logs. The producer is deterministic
# (no-shuffle), so a fresh PHOTON run sees the identical 60M-token prefix
# flat consumed — token-match is preserved without rerunning flat.
MODELS=("$@")
if [ "${#MODELS[@]}" -eq 0 ]; then MODELS=(photon_a0 flat); fi

PRODUCER_PID=""

# Kill the background producer (if still running) and remove the FIFO so a
# re-run starts clean. Registered with `trap` so it fires on normal exit,
# on error (set -e), and on Ctrl+C — this must never leave an orphaned
# python process holding the FIFO open.
cleanup() {
  if [ -n "$PRODUCER_PID" ] && kill -0 "$PRODUCER_PID" 2>/dev/null; then
    echo "d4_chain: stopping producer (pid $PRODUCER_PID)"
    kill "$PRODUCER_PID" 2>/dev/null || true
    wait "$PRODUCER_PID" 2>/dev/null || true
  fi
  rm -f "$FIFO"
}
trap cleanup EXIT INT TERM

echo "chain (re)started $(date -Is) [models: ${MODELS[*]}]" >> "$STATUS"

for name in "${MODELS[@]}"; do
  echo "=== D.4: ${name} ==="

  echo "d4_chain: creating FIFO $FIFO"
  rm -f "$FIFO"
  mkfifo "$FIFO"

  echo "d4_chain: launching FineWeb-Edu producer for ${name} -> $FIFO"
  # --giveup-seconds 900: if HF is unreachable / datasets missing / the
  # stream stalls for 15 min, the producer closes the FIFO so the trainer
  # sees EOF and stops cleanly instead of hanging for days (FIX 1).
  "$PYTHON" "$PRODUCER" --fifo "$FIFO" --log-every 5000 --giveup-seconds 900 \
    > "logs/d4_${name}_producer.log" 2>&1 &
  PRODUCER_PID=$!

  # Expected step count (token-match authority) read straight from the
  # config so this stays DRY: n_steps * batch_size * seq_len = 60.0M tok.
  EXPECT_STEPS=$(grep -E '^[[:space:]]*n_steps' "configs/d4_${name}.toml" \
    | grep -oE '[0-9]+' | head -1)
  # Hard wall-clock backstop ABOVE the in-trainer max_wall_time_seconds cap
  # (40h/60h): if the trainer itself wedges, `timeout` (exit 124) kills it.
  case "$name" in
    photon_a0) TMO=42h ;;
    flat)      TMO=62h ;;
    *)         TMO=62h ;;
  esac

  echo "train ${name} started (timeout $TMO, expect ${EXPECT_STEPS} steps) $(date -Is)" >> "$STATUS"
  set +e
  timeout "$TMO" $BIN train --backend cuda --config "configs/d4_${name}.toml" \
    > "logs/d4_${name}.log" 2>&1
  rc=$?
  set -e

  # FIX 2: a nonzero exit OR a clean-but-early stop (wall cap / data
  # exhausted before n_steps) BOTH mean the two runs are NOT token-matched.
  # Verify the trainer's own completion line (train_cmd.rs:466 format:
  # "pm train: done — <N> step(s) completed in <s>s (<reason>)"), not just
  # the exit code, before trusting the checkpoint for comparison.
  if [ "$rc" -eq 124 ]; then
    echo "train ${name} FAILED (timeout after $TMO) $(date -Is)" >> "$STATUS"
  elif [ "$rc" -ne 0 ]; then
    echo "train ${name} FAILED (exit $rc) $(date -Is)" >> "$STATUS"
  else
    done_line=$(grep 'pm train: done' "logs/d4_${name}.log" 2>/dev/null | tail -1 || true)
    steps=$(printf '%s' "$done_line" | grep -oE '[0-9]+ step' | grep -oE '[0-9]+' | head -1 || true)
    reason=$(printf '%s' "$done_line" | sed -E 's/.*\(([^)]*)\)[[:space:]]*$/\1/' || true)
    if [ "$reason" = "completed" ] && [ -n "$EXPECT_STEPS" ] && [ "$steps" = "$EXPECT_STEPS" ]; then
      echo "train ${name} COMPLETE (${steps} steps) $(date -Is)" >> "$STATUS"
    else
      echo "train ${name} INCOMPLETE (${steps:-?} steps, reason=${reason:-unknown}) — NOT token-matched, checkpoint invalid for comparison $(date -Is)" >> "$STATUS"
    fi
  fi

  echo "d4_chain: stopping producer (pid $PRODUCER_PID) after ${name} training"
  kill "$PRODUCER_PID" 2>/dev/null || true
  wait "$PRODUCER_PID" 2>/dev/null || true
  PRODUCER_PID=""
  rm -f "$FIFO"
done

for name in "${MODELS[@]}"; do
  ckpt="checkpoints/d4_${name}.safetensors"
  if [ ! -f "$ckpt" ]; then
    echo "eval ${name} SKIPPED (no checkpoint) $(date -Is)" >> "$STATUS"
    continue
  fi
  # Only a COMPLETE (token-matched) checkpoint is valid for the comparison.
  if ! grep -q "train ${name} COMPLETE" "$STATUS"; then
    echo "eval ${name} SKIPPED (training not COMPLETE — checkpoint not token-matched) $(date -Is)" >> "$STATUS"
    continue
  fi
  echo "=== D.4 eval: ${name} ==="
  # --limit 0 = full 10,042-item HellaSwag set (eval_cmd.rs treats 0 as
  # "all"). Eval is minutes vs a multi-day train — worth the lower noise.
  if $BIN eval hellaswag --backend cuda --config "configs/d4_${name}.toml" \
      --model "$ckpt" --data data/hellaswag_val_gpt2.jsonl --limit 0 \
      > "logs/d4_${name}_hellaswag.log" 2>&1; then
    echo "eval ${name} OK: $(tail -1 "logs/d4_${name}_hellaswag.log") $(date -Is)" >> "$STATUS"
  else
    echo "eval ${name} FAILED (exit $?) $(date -Is)" >> "$STATUS"
  fi
done

echo "chain finished $(date -Is)" >> "$STATUS"
echo "d4_chain: done — see $STATUS"
