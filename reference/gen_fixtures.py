"""Generate small SSD fixtures used by Rust tests.

Writes per-array .npy files into reference/fixtures/ssd_q64/. Keeping
arrays as individual .npy files avoids pulling a zip/npz reader into
the Rust test pipeline (we use the `npyz` crate).

Run:
    python reference/gen_fixtures.py
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from mamba2_ref import ssd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="reference/fixtures/ssd_q64",
        help="Output directory (one .npy file per tensor).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    # Small shape: batch=1, T=128 (=2 chunks of 64), heads=2, p=8, n=16.
    B, T, H, P, N = 1, 128, 2, 8, 16
    Q = 64

    X = torch.randn(B, T, H, P, dtype=torch.float32)
    A = -torch.rand(B, T, H, dtype=torch.float32)  # negative, like -exp(A_log)
    Bp = torch.randn(B, T, H, N, dtype=torch.float32) * 0.5
    Cp = torch.randn(B, T, H, N, dtype=torch.float32) * 0.5

    Y, final_state = ssd(X, A, Bp, Cp, Q)

    np.save(out / "X.npy", X.numpy())
    np.save(out / "A.npy", A.numpy())
    np.save(out / "B.npy", Bp.numpy())
    np.save(out / "C.npy", Cp.numpy())
    np.save(out / "Y.npy", Y.numpy())
    np.save(out / "final_state.npy", final_state.numpy())
    (out / "meta.txt").write_text(
        f"shape: B={B} T={T} H={H} P={P} N={N}\nblock_len={Q}\nseed={args.seed}\n"
    )
    print(f"wrote fixtures to {out}/ (X{tuple(X.shape)} -> Y{tuple(Y.shape)})")


if __name__ == "__main__":
    main()
