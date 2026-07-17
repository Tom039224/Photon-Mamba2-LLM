# Photon × Mamba2 LLM

日本語版: **[README_JA.md](./README_JA.md)**

An experimental LLM engine in Rust. It keeps the hierarchical autoregressive
skeleton of **PHOTON** (Fujitsu × RIKEN AIP) and replaces the local
autoregressive module at each level with a **Mamba2** SSD (State Space Dual)
block — built on a **backend-agnostic core with swappable compute backends**,
then evaluated in controlled experiments.

> **Status (2026-07, wrap-up point).** The engineering goal — memory
> efficiency — was met: a 102M-parameter model trains on a 12 GB consumer GPU,
> and a hand-written CUDA backend works. The *research* hypothesis — that the
> PHOTON hierarchy yields a better language model than a flat Mamba2 stack —
> held at small scale but **reversed at scale**. The full write-up (with
> evidence) is in **[SUMMARY.md](./SUMMARY.md)** (Japanese).

> 🤖 **Built with AI.** This project was developed with heavy assistance from
> AI coding tools (Anthropic's Claude / Claude Code) — architecture, kernel and
> autograd implementation, experiment design, and the honest analysis below
> were produced in a human + AI loop.

## Headline result

Controlled comparison — identical parameter count (±0.74%), data, steps, seed,
lr and clipping; the *only* difference is architecture. Both compared on pure
per-token cross-entropy (PHOTON runs at α=0, so its auxiliary loss has
coefficient 0 → CE only).

| experiment | data / scale | PHOTON CE | flat CE | Δ = flat − PHOTON |
| --- | --- | ---: | ---: | ---: |
| **D.2b** | TinyStories / 5.5M tok | **1.698** | 2.554 | **+0.86 nat (PHOTON wins)** |
| **D.4** | FineWeb-Edu / 60M tok | 7.088 | **5.797** | **−1.29 nat (flat wins)** |

Scaling ~10.8× and moving to a diverse corpus **flipped the ordering**. In D.4
PHOTON's loss stalled by step ~800 while flat kept descending, and PHOTON's
gradient norm ran ~60× higher (mean 147 vs 2.37), hard-clipped almost every
step — an optimization instability. This does not prove PHOTON is fundamentally
worse (an lr sweep / warmup was not attempted), but it does show that the
small-scale advantage did **not** transfer, and that this
architecture + training setup is unstable at scale. See `SUMMARY.md` for the
full discussion of confounds and open questions.

## Design pillars

- **Backend abstraction first.** The core model definition (`pm-core`) depends
  on neither Candle nor cudarc nor any vendor SDK — everything goes through
  `Tensor` / `Ops` / `Backend` traits. Adding a backend is a trait
  implementation only; numerical equivalence (fp32 within 1e-4) is a hard rule.
- **Staged optimization.** Phase 1: a Candle reference backend. Phase 1.5/2: a
  hand-written CUDA backend (cudarc + custom PTX kernels + a reverse-mode
  autograd tape). Phase 3: remap onto Tenstorrent hardware.
- **Training included.** Designed around activation checkpointing from the
  start, not inference-only.

## Crates

`pm-core` (traits / model definition, backend-agnostic) · `pm-candle`
(reference backend, the numerical ground truth) · `pm-cuda` (hand-written CUDA
backend) · `pm-data` · `pm-tokenizer` · `pm-train` · `pm-infer` · `pm-cli`.

## Build / run

```bash
# The native CUDA backend needs --features cuda (and nvcc on PATH).
PATH=/opt/cuda/bin:$PATH CUDA_HOME=/opt/cuda \
  cargo build -p pm-cli --features cuda --release

# Verify pm-core stays backend-agnostic (also enforced in CI).
cargo tree -p pm-core --edges normal | grep -E '(candle|cudarc)' && exit 1 || echo OK

./target/release/pm train    --backend cuda --config configs/<...>.toml
./target/release/pm generate --backend cuda --model checkpoints/<...>.safetensors --prompt "..."
./target/release/pm eval hellaswag --backend cuda --config <...> --model <...> --data <...>
```

Environment: Linux (CachyOS) · NVIDIA RTX 5070 12 GB, sm_120 (Blackwell) ·
CUDA 13.3 · Rust stable (nightly alongside for `nvptx64-nvidia-cuda` in Phase 2).

## License

**BSD 3-Clause License** ([LICENSE](./LICENSE)). © 2026 Tom039224.

The referenced papers (PHOTON, Mamba-2) and evaluation data (TinyStories,
FineWeb-Edu, the GPT-2 tokenizer, HellaSwag) are third-party works and are
**not** included in this repository; see [SUMMARY.md](./SUMMARY.md) §9 for
sources.
