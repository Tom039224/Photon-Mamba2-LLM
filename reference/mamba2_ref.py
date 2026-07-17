"""Mamba2 SSD chunked-scan reference (PyTorch).

Adapted from Listing 1 of:
    Dao, T., & Gu, A. (2024). Transformers are SSMs: Generalized Models and
    Efficient Algorithms Through Structured State Space Duality.

Used purely for cross-checking the Rust ssd_scan implementation in
pm-core / pm-candle. Run `gen_fixtures.py` to write small numpy
fixtures that Rust tests load.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable lower-triangular segment sum.

    For input ``x`` of shape ``(..., T)`` returns ``(..., T, T)`` where
    ``out[..., i, j] = sum_{k=j+1}^{i} x[..., k]`` for ``i >= j``, and
    ``-inf`` above the diagonal so ``exp(out)`` yields the SSM decay mask.
    """
    T = x.size(-1)
    expanded = x[..., None].repeat(*([1] * x.ndim), T)
    lower = torch.tril(
        torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1
    )
    expanded = expanded.masked_fill(~lower, 0)
    cs = torch.cumsum(expanded, dim=-2)
    diag_keep = torch.tril(
        torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0
    )
    return cs.masked_fill(~diag_keep, float("-inf"))


def ssd(
    X: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    block_len: int,
    initial_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SSD chunked scan.

    Shapes:
      X : (b, t, h, p)
      A : (b, t, h)        (scalar-per-head SSM, negative reals)
      B : (b, t, h, n)
      C : (b, t, h, n)

    Returns:
      Y           : (b, t, h, p)
      final_state : (b, h, p, n)
    """
    assert X.size(1) % block_len == 0, "T must be divisible by block_len"

    X = X.unflatten(1, (-1, block_len))  # (b, c, l, h, p)
    A = A.unflatten(1, (-1, block_len))
    B = B.unflatten(1, (-1, block_len))
    C = C.unflatten(1, (-1, block_len))

    A = A.permute(0, 3, 1, 2)  # (b, h, c, l)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Diagonal (intra-chunk) blocks.
    L = torch.exp(segsum(A))  # (b, h, c, l, l)
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Right factor: states accumulated within each chunk.
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)  # (b, h, c, l)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Inter-chunk scan.
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[..., -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Left factor: states -> output via C.
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    Y = (Y_diag + Y_off).flatten(1, 2)  # (b, t, h, p)
    return Y, final_state
