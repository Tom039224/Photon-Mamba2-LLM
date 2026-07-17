#!/bin/bash
# Phase D.2b comparison chain: 3 training runs (1 epoch each) + HellaSwag evals.
# Launched 2026-07-04. Total expected wall time ~21.5h on RTX 5070.
# Logs: logs/d2b_*.log   Status: logs/d2b_status.txt   Checkpoints: checkpoints/d2b_*.safetensors
cd "$(dirname "$0")/.." || exit 1
BIN=./target/release/pm
STATUS=logs/d2b_status.txt
mkdir -p checkpoints logs
echo "chain started $(date -Is)" > "$STATUS"

for name in photon_a0 photon_a03 flat; do
  echo "train ${name} started $(date -Is)" >> "$STATUS"
  if $BIN train --backend cuda --config "configs/d2b_${name}.toml" > "logs/d2b_${name}.log" 2>&1; then
    echo "train ${name} OK $(date -Is)" >> "$STATUS"
  else
    echo "train ${name} FAILED (exit $?) $(date -Is)" >> "$STATUS"
  fi
done

for name in photon_a0 photon_a03 flat; do
  ckpt="checkpoints/d2b_${name}.safetensors"
  if [ ! -f "$ckpt" ]; then
    echo "eval ${name} SKIPPED (no checkpoint) $(date -Is)" >> "$STATUS"
    continue
  fi
  if $BIN eval hellaswag --backend cuda --config "configs/d2b_${name}.toml" \
      --model "$ckpt" --data data/hellaswag_val_gpt2.jsonl --limit 1000 \
      > "logs/d2b_${name}_hellaswag.log" 2>&1; then
    echo "eval ${name} OK: $(tail -1 "logs/d2b_${name}_hellaswag.log") $(date -Is)" >> "$STATUS"
  else
    echo "eval ${name} FAILED (exit $?) $(date -Is)" >> "$STATUS"
  fi
done
echo "chain finished $(date -Is)" >> "$STATUS"
