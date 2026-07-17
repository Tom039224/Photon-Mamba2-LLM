//! Mamba2 SSD chunked scan — pure-scalar reference.
//!
//! Implements the SSM recurrence
//!
//! ```text
//!   y_t = Σ_{s=0..t} exp(Σ_{k=s+1..t} A_k) · (C_t · B_s) · x_s
//! ```
//!
//! per `(batch, head)` pair. Equivalent to Mamba2 Listing 1 (Dao & Gu 2024)
//! but written as O(B · H · T² · (P + N)) loops for clarity. Backend
//! implementations (`pm-candle::Ops::ssd_scan`, future `pm-cuda`) are
//! expected to compute the same result via a fused chunked algorithm; this
//! reference is the ground truth they are validated against.
//!
//! The `block_len` parameter is accepted for API symmetry with
//! `Ops::ssd_scan` but ignored here: the scalar formulation does not
//! need chunking.

/// Pure-Rust scalar SSD scan over flat row-major buffers.
///
/// Buffer layouts (row-major, last axis fastest):
/// - `x`: `(batch, t_len, n_heads, p_dim)`
/// - `a`: `(batch, t_len, n_heads)` — scalar-per-head SSM, typically negative
/// - `b`: `(batch, t_len, n_heads, n_dim)`
/// - `c`: `(batch, t_len, n_heads, n_dim)`
///
/// Returns `y` with layout `(batch, t_len, n_heads, p_dim)`.
///
/// # Panics
/// Panics if any buffer length disagrees with the declared shape.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn ssd_scan_naive_scalar(
    x: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    batch: usize,
    t_len: usize,
    n_heads: usize,
    p_dim: usize,
    n_dim: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), batch * t_len * n_heads * p_dim, "x shape mismatch");
    assert_eq!(a.len(), batch * t_len * n_heads, "a shape mismatch");
    assert_eq!(b.len(), batch * t_len * n_heads * n_dim, "b shape mismatch");
    assert_eq!(c.len(), batch * t_len * n_heads * n_dim, "c shape mismatch");

    let mut y = vec![0f32; batch * t_len * n_heads * p_dim];
    let mut a_cum = vec![0f32; batch * t_len * n_heads];

    // Inclusive prefix sum of A along T per (batch, head).
    for bi in 0..batch {
        for hi in 0..n_heads {
            let mut acc = 0f32;
            for ti in 0..t_len {
                acc += a[(bi * t_len + ti) * n_heads + hi];
                a_cum[(bi * t_len + ti) * n_heads + hi] = acc;
            }
        }
    }

    for bi in 0..batch {
        for hi in 0..n_heads {
            for ti in 0..t_len {
                let a_cum_t = a_cum[(bi * t_len + ti) * n_heads + hi];
                let c_off = ((bi * t_len + ti) * n_heads + hi) * n_dim;

                // Precompute (C_t · B_s) over s for this (b, h, t).
                // bc_dot[s] = Σ_n C[b,t,h,n] * B[b,s,h,n]
                let mut bc_dot = vec![0f32; ti + 1];
                for (si, slot) in bc_dot.iter_mut().enumerate() {
                    let b_off = ((bi * t_len + si) * n_heads + hi) * n_dim;
                    let mut acc = 0f32;
                    for ni in 0..n_dim {
                        acc += b[b_off + ni] * c[c_off + ni];
                    }
                    *slot = acc;
                }

                for pi in 0..p_dim {
                    let mut acc_y = 0f32;
                    for si in 0..=ti {
                        let a_cum_s = a_cum[(bi * t_len + si) * n_heads + hi];
                        // exp(Σ_{k=s+1..t} A_k) = exp(a_cum[t] - a_cum[s]).
                        let decay = (a_cum_t - a_cum_s).exp();
                        let x_val = x[((bi * t_len + si) * n_heads + hi) * p_dim + pi];
                        acc_y += decay * bc_dot[si] * x_val;
                    }
                    y[((bi * t_len + ti) * n_heads + hi) * p_dim + pi] = acc_y;
                }
            }
        }
    }

    y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check: with A = 0 the decay matrix is all-ones, so
    /// y_t = Σ_{s<=t} (C_t · B_s) * x_s.
    #[test]
    fn zero_a_collapses_to_simple_sum() {
        let (batch, t_len, h, p, n) = (1, 4, 1, 2, 2);
        let x = vec![1.0f32; batch * t_len * h * p];
        let a = vec![0.0f32; batch * t_len * h];
        let b: Vec<f32> = (0..batch * t_len * h * n).map(|i| i as f32 * 0.1).collect();
        let c: Vec<f32> = (0..batch * t_len * h * n).map(|i| i as f32 * 0.1).collect();
        let y = ssd_scan_naive_scalar(&x, &a, &b, &c, batch, t_len, h, p, n);
        assert_eq!(y.len(), batch * t_len * h * p);
        // y_0 = (C_0 · B_0) * x_0; with x=1, p=2: same value in both p dims.
        assert!((y[0] - y[1]).abs() < 1e-6);
        // y monotonically grows with t when all summands positive.
        for ti in 1..t_len {
            assert!(y[ti * p] > y[(ti - 1) * p]);
        }
    }
}
