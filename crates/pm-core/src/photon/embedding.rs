//! Token and positional embeddings for PHOTON.
//!
//! `TokenEmbedding` wraps a tied embedding table: the same trainable
//! parameter serves both input lookup and the language-modelling head.
//! Matches the GPT-2 / Llama convention and halves the parameter count
//! of those two layers.
//!
//! `RotaryEmbedding` implements RoPE (Su et al., 2021) using the
//! "rotate halves" formulation popularised by GPT-NeoX and Llama.
//! Cos / sin tables are precomputed once on host and held as
//! non-trainable [`Tensor`]s.

use crate::{Module, Ops, Param, Parameterized, Tensor};

/// Tied token embedding + LM head. The `weight` is a trainable param.
pub struct TokenEmbedding<O: Ops> {
    pub weight: O::Param, // (vocab_size, d_model)
    pub vocab_size: usize,
    pub d_model: usize,
}

impl<O: Ops> TokenEmbedding<O> {
    pub fn from_param(vocab_size: usize, d_model: usize, weight: O::Param) -> Self {
        Self {
            weight,
            vocab_size,
            d_model,
        }
    }

    /// Constant-fill constructor for tests.
    pub fn from_constants(
        ops: &O,
        vocab_size: usize,
        d_model: usize,
        weight_scale: f32,
    ) -> Result<Self, O::Error> {
        let data = vec![weight_scale; vocab_size * d_model];
        Ok(Self {
            weight: ops.param_from_slice_f32(&data, &[vocab_size, d_model])?,
            vocab_size,
            d_model,
        })
    }

    /// Tied LM head: `logits = hidden @ weight^T`.
    /// `hidden`: `(..., d_model)` → `(..., vocab_size)`.
    ///
    /// The fp32-native embedding table is cast to `hidden`'s (ambient)
    /// dtype inline before the matmul — a no-op when `hidden` is fp32
    /// (memory-efficiency plan Phase A2). `logits` therefore comes out
    /// in the ambient compute dtype too; `cross_entropy_loss`
    /// (`pm-train::loss`) upcasts it back to fp32 before `log_softmax`.
    pub fn lm_head_logits(&self, ops: &O, hidden: &O::Tensor) -> Result<O::Tensor, O::Error> {
        let cdt = hidden.dtype();
        let w = ops.to_dtype(self.weight.as_tensor(), cdt)?;
        let rank = w.rank();
        let wt = ops.transpose(&w, rank - 2, rank - 1)?;
        ops.matmul(hidden, &wt)
    }
}

impl<O: Ops> Module<O> for TokenEmbedding<O> {
    fn forward(&self, ops: &O, ids: &O::Tensor) -> Result<O::Tensor, O::Error> {
        ops.embedding(self.weight.as_tensor(), ids)
    }
}

impl<O: Ops> Parameterized<O> for TokenEmbedding<O> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>) {
        out.push(&self.weight);
    }
}

/// Default RoPE base used by Llama and GPT-NeoX.
pub const ROPE_DEFAULT_BASE: f32 = 10_000.0;

/// Rotary Position Embedding (GPT-NeoX / Llama "rotate halves" variant).
///
/// Precomputes `cos`/`sin` tables of shape `(max_seq_len, d_head/2)` and
/// applies them to a `(B, T, H, d_head)` tensor:
///
/// ```text
/// x1, x2 = x[..., :d/2], x[..., d/2:]
/// out1 = x1 * cos - x2 * sin
/// out2 = x1 * sin + x2 * cos
/// out  = concat([out1, out2], dim=-1)
/// ```
///
/// `cos` and `sin` are constant (deterministic functions of position) so
/// they are stored as `O::Tensor`, not trainable params.
pub struct RotaryEmbedding<O: Ops> {
    pub cos: O::Tensor,
    pub sin: O::Tensor,
    pub d_head: usize,
    pub max_seq_len: usize,
    pub base: f32,
}

impl<O: Ops> RotaryEmbedding<O> {
    pub fn new(ops: &O, d_head: usize, max_seq_len: usize, base: f32) -> Result<Self, O::Error> {
        assert!(
            d_head.is_multiple_of(2),
            "RoPE requires even d_head, got {d_head}"
        );
        assert!(max_seq_len > 0, "RoPE max_seq_len must be > 0");
        let half = d_head / 2;
        let mut cos = vec![0f32; max_seq_len * half];
        let mut sin = vec![0f32; max_seq_len * half];
        let d_head_f = d_head as f32;
        for t in 0..max_seq_len {
            let t_f = t as f32;
            for i in 0..half {
                let freq = (-(2.0 * i as f32 / d_head_f) * base.ln()).exp();
                let angle = t_f * freq;
                cos[t * half + i] = angle.cos();
                sin[t * half + i] = angle.sin();
            }
        }
        Ok(Self {
            cos: ops.from_slice_f32(&cos, &[max_seq_len, half])?,
            sin: ops.from_slice_f32(&sin, &[max_seq_len, half])?,
            d_head,
            max_seq_len,
            base,
        })
    }

    /// Apply RoPE to `x: (B, T, H, d_head)`. `T` must be ≤ `max_seq_len`.
    pub fn apply(&self, ops: &O, x: &O::Tensor) -> Result<O::Tensor, O::Error> {
        let shape = x.shape();
        assert_eq!(shape.len(), 4, "RoPE expects (B,T,H,D)");
        let (b, t, h, d) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(
            d, self.d_head,
            "d_head mismatch: got {d}, expected {}",
            self.d_head
        );
        assert!(
            t <= self.max_seq_len,
            "T={t} exceeds RoPE max_seq_len={}",
            self.max_seq_len
        );
        let half = d / 2;

        let x1 = ops.narrow(x, 3, 0, half)?;
        let x2 = ops.narrow(x, 3, half, half)?;

        let cos_t = ops.narrow(&self.cos, 0, 0, t)?;
        let sin_t = ops.narrow(&self.sin, 0, 0, t)?;
        let cos_r = ops.reshape(&cos_t, &[1, t, 1, half])?;
        let sin_r = ops.reshape(&sin_t, &[1, t, 1, half])?;
        let cos_b = ops.broadcast_as(&cos_r, &[b, t, h, half])?;
        let sin_b = ops.broadcast_as(&sin_r, &[b, t, h, half])?;

        let r1 = ops.sub(&ops.mul(&x1, &cos_b)?, &ops.mul(&x2, &sin_b)?)?;
        let r2 = ops.add(&ops.mul(&x1, &sin_b)?, &ops.mul(&x2, &cos_b)?)?;
        ops.concat(&[&r1, &r2], 3)
    }
}
