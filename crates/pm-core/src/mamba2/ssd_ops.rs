//! Pure-`Ops` SSD scan (default implementation).
//!
//! Computes the same quantity as
//! [`super::ssd_scan_naive_scalar`] but entirely via the `Ops` trait,
//! so autograd flows through it on any backend whose `Ops` impl tracks
//! gradients (Candle's does, via `candle_core::Var`).
//!
//! ```text
//!   y_{b,t,h,p} = Σ_{s ≤ t} exp(A_cum_{b,t,h} - A_cum_{b,s,h})
//!                          · (C_{b,t,h} · B_{b,s,h})
//!                          · x_{b,s,h,p}
//! ```
//!
//! ### Memory layout
//!
//! The whole pipeline stays in **`(B, H, T_t, T_s)`** layout so the
//! large `(T, T)` decay / bc / combined intermediates only ever exist
//! in one orientation. The earlier version round-tripped through
//! `(B, T, T, H)` for the multiplication, which `contiguous()`-ed
//! several extra `(B, H, T, T)` copies into the autograd tape; with
//! `B=2, T=512, H=12, fp32` each of those was 24 MB so dropping them
//! is the difference between fitting and OOMing on 12 GB VRAM.
//!
//! ### Stages
//! 1. `A_cum = cumsum(a, dim=1)`, permute to `(B, H, T)`.
//! 2. `A_diff[b, h, t, s] = A_cum[b, h, t] - A_cum[b, h, s]` via
//!    broadcast subtract. Zero the upper triangle *before* exp to
//!    avoid overflow, then mask again post-exp.
//! 3. `bc[b, h, t, s] = Σ_n C[b,t,h,n] * B[b,s,h,n]` via one batched
//!    `matmul` on `(B, H, T, N) @ (B, H, N, T)`.
//! 4. `combined = decay * bc`, `(B, H, T, T)`.
//! 5. `y[b, h, t, p] = Σ_s combined · x_perm`, one batched `matmul`
//!    on `(B, H, T, T) @ (B, H, T, P)`. Final transpose → `(B, T, H, P)`.
//!
//! `block_len` is accepted for API symmetry but ignored — the
//! formulation above is `O(T²)` memory. A fused on-device chunked
//! kernel is a Phase-2 concern (pm-cuda).
//!
//! ### Dtype invariant (memory-efficiency plan Phase A2)
//!
//! This scan always computes in `F32`, regardless of the caller's
//! ambient compute dtype: `A_diff` feeds into `exp` after accumulating
//! up to `T` terms via `cumsum`, and a `bf16` accumulator both loses
//! the exponent range needed for that `exp` and under/overflows.
//! `Mamba2Block::forward` enforces this by upcasting `x`/`a`/`b`/`c` to
//! `F32` immediately before calling `Ops::ssd_scan` and downcasting `y`
//! back to the ambient dtype immediately after — so this module never
//! has to branch on dtype itself. The invariant is also self-enforcing
//! here: [`ssd_scan_dense`]'s triangular mask is always built via
//! `from_slice_f32` (always `F32`), so a caller that passed non-`F32`
//! `a`/`b`/`c` would hit a hard dtype-mismatch error at the `add`/`mul`
//! against that mask rather than silently computing in low precision.

use crate::{Dtype, Ops, Tensor};

/// Default `ssd_scan` implementation expressed via the `Ops` trait.
///
/// Dispatches to the chunked variant ([`ssd_scan_chunked`]) when
/// `block_len < t`; otherwise falls through to the dense kernel
/// ([`ssd_scan_dense`]).
pub fn ssd_scan_ops_default<O: Ops>(
    ops: &O,
    x: &O::Tensor,
    a: &O::Tensor,
    b: &O::Tensor,
    c: &O::Tensor,
    block_len: usize,
) -> Result<O::Tensor, O::Error> {
    let t = x.shape()[1];
    if block_len == 0 || block_len >= t {
        return ssd_scan_dense(ops, x, a, b, c);
    }
    if !t.is_multiple_of(block_len) {
        // Dense fallback if the sequence isn't an integer number of chunks.
        // For Phase-1 PHOTON `T = seq_len` is always chunk-aligned, so this
        // branch is exercised only by edge-case tests.
        return ssd_scan_dense(ops, x, a, b, c);
    }
    ssd_scan_chunked(ops, x, a, b, c, block_len)
}

/// Dense `O(T²)` SSD scan in `(B, H, T, T)` layout.
pub fn ssd_scan_dense<O: Ops>(
    ops: &O,
    x: &O::Tensor,
    a: &O::Tensor,
    b: &O::Tensor,
    c: &O::Tensor,
) -> Result<O::Tensor, O::Error> {
    let x_shape = x.shape();
    assert_eq!(x_shape.len(), 4, "ssd_scan: x must be (B,T,H,P)");
    let (batch, t, n_heads, _p_dim) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    let _n_dim = b.shape()[3];
    debug_assert_eq!(a.shape(), &[batch, t, n_heads]);
    debug_assert_eq!(b.shape(), &[batch, t, n_heads, _n_dim]);
    debug_assert_eq!(c.shape(), &[batch, t, n_heads, _n_dim]);

    // 1. A_cum: (B, T, H) → permute to (B, H, T).
    let a_cum = ops.cumsum(a, 1)?;
    let a_cum_bht = ops.transpose(&a_cum, 1, 2)?; // (B, H, T)

    // 2. A_diff: (B, H, T_t, T_s) via broadcast subtract.
    let a_cum_t = ops.reshape(&a_cum_bht, &[batch, n_heads, t, 1])?;
    let a_cum_s = ops.reshape(&a_cum_bht, &[batch, n_heads, 1, t])?;
    let a_diff = ops.sub(&a_cum_t, &a_cum_s)?; // (B, H, T, T)

    // Additive log-space mask: 0 on the lower triangle (including
    // diagonal), a large negative on the upper triangle. After exp
    // the upper triangle collapses to (≈) 0 and we don't need a
    // separate post-exp mask multiplication. That saves two
    // `(B,H,T,T)` intermediates from the autograd tape per block —
    // measurable at T=512, H=12 (each tensor ≈ 24 MB at B=2).
    let mut log_mask_data = vec![0f32; t * t];
    for ti in 0..t {
        for si in (ti + 1)..t {
            log_mask_data[ti * t + si] = -1.0e9;
        }
    }
    let log_mask_2d = ops.from_slice_f32(&log_mask_data, &[1, 1, t, t])?;
    let log_mask = ops.broadcast_as(&log_mask_2d, &[batch, n_heads, t, t])?;

    let a_diff_masked = ops.add(&a_diff, &log_mask)?;
    let decay = ops.exp(&a_diff_masked)?; // (B, H, T, T), upper triangle ≈ 0

    // 3. bc = C_perm @ B_perm.T where C, B both permute to (B, H, T, N).
    let c_perm = ops.transpose(c, 1, 2)?; // (B, H, T, N)
    let b_perm = ops.transpose(b, 1, 2)?; // (B, H, T, N)
    let b_perm_t = ops.transpose(&b_perm, 2, 3)?; // (B, H, N, T)
    let bc = ops.matmul(&c_perm, &b_perm_t)?; // (B, H, T_t, T_s)

    // 4. combined = decay * bc, both already (B, H, T, T).
    let combined = ops.mul(&decay, &bc)?;

    // 5. y[b, h, t, p] = combined @ x_perm.
    let x_perm = ops.transpose(x, 1, 2)?; // (B, H, T, P)
    let y_bhtp = ops.matmul(&combined, &x_perm)?; // (B, H, T_t, P)
    ops.transpose(&y_bhtp, 1, 2) // (B, T, H, P)
}

/// Chunked SSD scan (Mamba2 paper Listing 1). Splits `T` into
/// `n_chunks = T / Q` chunks of size `Q = block_len`. The per-chunk
/// intermediates are only `(Q, Q)` instead of `(T, T)`, and the
/// inter-chunk hidden state is `(B, H, N, P)`. Total peak memory
/// drops from `O(B·H·T²)` to `O(B·H·n_chunks·Q²) + O(B·H·N·P)`.
///
/// For the production target `T=2048, Q=64, B=4, H=12, fp32`:
/// - dense: `(B,H,T,T)` = 805 MB per intermediate × ~6 alive = ~4.8 GB
/// - chunked: `(B,H,n_chunks,Q,Q)` = 25 MB per intermediate × ~6 = 150 MB
///
/// The math splits each output position `y[c, q]` (chunk c, intra-chunk
/// position q) into two parts:
///
/// ```text
///   y[c, q]       = y_intra[c, q] + y_inter[c, q]
///   y_intra[c, q] = Σ_{q'≤q} exp(A_cum[c,q] − A_cum[c,q']) · (C[c,q]·B[c,q']) · x[c,q']
///   y_inter[c, q] = exp(A_cum[c, q]) · C[c, q]ᵀ · h_end[c−1]
///   h_end[c]      = exp(A_cum_end[c]) · h_end[c−1]
///                 + Σ_{q'} exp(A_cum_end[c] − A_cum[c,q']) · B[c,q'] x[c,q']ᵀ
///   h_end[−1]     = 0
/// ```
///
/// `y_intra` is exactly the dense scan applied to each chunk
/// independently (the (B, n_chunks) axes are flattened into the batch
/// dim and we reuse [`ssd_scan_dense`]). `y_inter` requires a
/// sequential scan over chunks, but each step is cheap — only
/// `(B, H, N, P)` worth of arithmetic.
pub fn ssd_scan_chunked<O: Ops>(
    ops: &O,
    x: &O::Tensor,
    a: &O::Tensor,
    b: &O::Tensor,
    c: &O::Tensor,
    block_len: usize,
) -> Result<O::Tensor, O::Error> {
    let x_shape = x.shape();
    assert_eq!(x_shape.len(), 4, "ssd_scan_chunked: x must be (B,T,H,P)");
    let (batch, t, n_heads, p_dim) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    let n_dim = b.shape()[3];
    let q = block_len;
    assert!(t.is_multiple_of(q), "T={t} must be divisible by Q={q}");
    let n_chunks = t / q;
    debug_assert_eq!(a.shape(), &[batch, t, n_heads]);
    debug_assert_eq!(b.shape(), &[batch, t, n_heads, n_dim]);
    debug_assert_eq!(c.shape(), &[batch, t, n_heads, n_dim]);

    // ---- 1. Reshape into chunks: (B, T, ...) → (B*n_chunks, Q, ...) ----
    let x_flat_c = ops.reshape(x, &[batch * n_chunks, q, n_heads, p_dim])?;
    let a_flat_c = ops.reshape(a, &[batch * n_chunks, q, n_heads])?;
    let b_flat_c = ops.reshape(b, &[batch * n_chunks, q, n_heads, n_dim])?;
    let c_flat_c = ops.reshape(c, &[batch * n_chunks, q, n_heads, n_dim])?;

    // ---- 2. Intra-chunk scan via the dense kernel ----
    //   (B*n_chunks, Q, H, P) → (B*n_chunks, Q, H, P)
    // Each chunk gets its own dense scan; flattening (B, n_chunks) lets
    // them all run in one batched matmul.
    let y_intra_flat = ssd_scan_dense(ops, &x_flat_c, &a_flat_c, &b_flat_c, &c_flat_c)?;
    let y_intra = ops.reshape(&y_intra_flat, &[batch, n_chunks, q, n_heads, p_dim])?;

    // ---- 3. End-of-chunk states ----
    // a_cum[b, c, q, h] = Σ_{k=0..q} a[b, c, k, h]  (inclusive cumsum within chunk)
    let a_per_chunk = ops.reshape(a, &[batch, n_chunks, q, n_heads])?;
    let a_cum = ops.cumsum(&a_per_chunk, 2)?; // (B, n_chunks, Q, H)

    // A_cum at the last position per chunk: (B, n_chunks, 1, H).
    let a_cum_end_keep = ops.narrow(&a_cum, 2, q - 1, 1)?;
    // decay_to_end[b, c, q, h] = exp(A_cum_end[b, c, h] - A_cum[b, c, q, h])
    let a_cum_end_b = ops.broadcast_as(&a_cum_end_keep, &[batch, n_chunks, q, n_heads])?;
    let decay_to_end = ops.exp(&ops.sub(&a_cum_end_b, &a_cum)?)?; // (B, n_chunks, Q, H)

    // states[b, c, h, n, p] = Σ_q decay_to_end[b,c,q,h] · B[b,c,q,h,n] · x[b,c,q,h,p]
    //
    // Compute via a batched 3-D matmul (`(B*n_chunks*H, N, Q) @
    // (B*n_chunks*H, Q, P)`). Candle's `matmul` doesn't accept 5-D
    // tensors with non-trailing batch dims, so we flatten the leading
    // (B, n_chunks, H) into a single batch axis for the matmul and
    // unflatten the result back.
    let x_per_chunk = ops.reshape(x, &[batch, n_chunks, q, n_heads, p_dim])?;
    let b_per_chunk = ops.reshape(b, &[batch, n_chunks, q, n_heads, n_dim])?;
    let decay_5d = ops.reshape(&decay_to_end, &[batch, n_chunks, q, n_heads, 1])?;
    let decay_5d_b = ops.broadcast_as(&decay_5d, &[batch, n_chunks, q, n_heads, p_dim])?;
    let weighted_x = ops.mul(&decay_5d_b, &x_per_chunk)?; // (B, n_chunks, Q, H, P)

    // b: (B, n_chunks, Q, H, N) → (B, n_chunks, H, Q, N) → (B, n_chunks, H, N, Q)
    //    → flatten → (B*n_chunks*H, N, Q)
    let b_chq = ops.transpose(&b_per_chunk, 2, 3)?; // (B, n_chunks, H, Q, N)
    let b_chnq = ops.transpose(&b_chq, 3, 4)?; // (B, n_chunks, H, N, Q)
    let b_flat = ops.reshape(&b_chnq, &[batch * n_chunks * n_heads, n_dim, q])?;

    // weighted_x: (B, n_chunks, Q, H, P) → (B, n_chunks, H, Q, P) → flatten
    let wx_chqp = ops.transpose(&weighted_x, 2, 3)?; // (B, n_chunks, H, Q, P)
    let wx_flat = ops.reshape(&wx_chqp, &[batch * n_chunks * n_heads, q, p_dim])?;

    let states_flat = ops.matmul(&b_flat, &wx_flat)?; // (B*n_chunks*H, N, P)
    let states = ops.reshape(&states_flat, &[batch, n_chunks, n_heads, n_dim, p_dim])?;

    // ---- 4. Inter-chunk sequential scan ----
    // decay_full[c] = exp(A_cum_end[c]) — the multiplicative decay over an entire chunk.
    let decay_full_keep = ops.exp(&a_cum_end_keep)?; // (B, n_chunks, 1, H)
                                                     // decay_from_start[c, q] = exp(A_cum[c, q]) — propagator from chunk start to position q.
    let decay_from_start = ops.exp(&a_cum)?; // (B, n_chunks, Q, H)

    let c_per_chunk = ops.reshape(c, &[batch, n_chunks, q, n_heads, n_dim])?;

    // h is the running state of shape (B, H, N, P). Start at zero.
    let mut h = ops.zeros(&[batch, n_heads, n_dim, p_dim], Dtype::F32)?;
    let mut y_inter_pieces: Vec<O::Tensor> = Vec::with_capacity(n_chunks);

    for chunk_idx in 0..n_chunks {
        // Slice chunk-specific tensors. `narrow` returns the same dim
        // count; reshape away the chunk dim.
        let c_this = ops.narrow(&c_per_chunk, 1, chunk_idx, 1)?;
        let c_this = ops.reshape(&c_this, &[batch, q, n_heads, n_dim])?;
        let dfs_this = ops.narrow(&decay_from_start, 1, chunk_idx, 1)?;
        let dfs_this = ops.reshape(&dfs_this, &[batch, q, n_heads])?;
        let df_this = ops.narrow(&decay_full_keep, 1, chunk_idx, 1)?;
        let df_this = ops.reshape(&df_this, &[batch, n_heads])?;
        let s_this = ops.narrow(&states, 1, chunk_idx, 1)?;
        let s_this = ops.reshape(&s_this, &[batch, n_heads, n_dim, p_dim])?;

        // y_inter[c, q, h, p] = decay_from_start[c, q, h] · C[c, q, h, :] · h[:, :]
        // matmul: (B, H, Q, N) @ (B, H, N, P) → (B, H, Q, P)
        let c_bhqn = ops.transpose(&c_this, 1, 2)?; // (B, H, Q, N)
        let prod_bhqp = ops.matmul(&c_bhqn, &h)?; // (B, H, Q, P)
        let prod_bqhp = ops.transpose(&prod_bhqp, 1, 2)?; // (B, Q, H, P)
                                                          // multiply by decay_from_start broadcast on P
        let dfs_4d = ops.reshape(&dfs_this, &[batch, q, n_heads, 1])?;
        let dfs_b = ops.broadcast_as(&dfs_4d, &[batch, q, n_heads, p_dim])?;
        let y_inter_this = ops.mul(&prod_bqhp, &dfs_b)?; // (B, Q, H, P)

        y_inter_pieces.push(ops.reshape(&y_inter_this, &[batch, 1, q, n_heads, p_dim])?);

        // Update h ← decay_full[c] · h + states[c]
        let df_4d = ops.reshape(&df_this, &[batch, n_heads, 1, 1])?;
        let df_b = ops.broadcast_as(&df_4d, &[batch, n_heads, n_dim, p_dim])?;
        let h_decayed = ops.mul(&h, &df_b)?;
        h = ops.add(&h_decayed, &s_this)?;
    }
    let pieces_refs: Vec<&O::Tensor> = y_inter_pieces.iter().collect();
    let y_inter = ops.concat(&pieces_refs, 1)?; // (B, n_chunks, Q, H, P)

    // ---- 5. Combine intra + inter, then unchunk ----
    let y_total = ops.add(&y_intra, &y_inter)?;
    ops.reshape(&y_total, &[batch, t, n_heads, p_dim])
}
