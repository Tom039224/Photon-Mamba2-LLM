//! Padded re-forward generator (G.2 + G.3).
//!
//! For each generation step we:
//! 1. Pad the current id buffer to the next multiple of
//!    `chunk_size^(L-1)` with a configurable pad token (BOS default = 0).
//! 2. Call `PhotonMamba::forward`.
//! 3. Take the logits at the position corresponding to the last *real*
//!    (non-pad) token and sample the next id.
//! 4. Append the sampled id, repeat.
//!
//! This is O(T²) per step due to the ssd_scan, but it works without
//! any new state machinery — and it's *deterministic*: the same prompt
//! plus sampler RNG seed always produces the same continuation.

use pm_core::{Ops, PhotonMamba, Tensor};

use crate::sampling::{Rng, Sampler};

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    /// Multiple of `chunk_size ^ (L-1)` that `forward` requires.
    pub chunk_product: usize,
    pub vocab_size: usize,
    /// Token id used to pad up to the next chunk boundary.
    pub pad_token_id: i64,
    /// Sampler RNG seed.
    pub seed: u64,
}

pub struct Generator<'a, O: Ops> {
    model: &'a PhotonMamba<O>,
    cfg: GenerateConfig,
    sampler: Sampler,
}

impl<'a, O: Ops> Generator<'a, O> {
    #[must_use]
    pub fn new(model: &'a PhotonMamba<O>, cfg: GenerateConfig, sampler: Sampler) -> Self {
        Self {
            model,
            cfg,
            sampler,
        }
    }

    /// Generate `cfg.max_new_tokens` ids appended after `prompt`.
    /// Returns the entire id sequence (prompt + completion).
    pub fn generate(&self, ops: &O, prompt: &[i64]) -> Result<Vec<i64>, O::Error> {
        assert!(!prompt.is_empty(), "prompt must contain at least 1 token");
        let mut rng = Rng::new(self.cfg.seed);
        let mut ids: Vec<i64> = prompt.to_vec();

        for _ in 0..self.cfg.max_new_tokens {
            let next = self.step(ops, &ids, &mut rng)?;
            ids.push(next);
        }
        Ok(ids)
    }

    fn step(&self, ops: &O, ids: &[i64], rng: &mut Rng) -> Result<i64, O::Error> {
        let real_len = ids.len();
        let padded_len = pad_up(real_len, self.cfg.chunk_product);
        let mut buf = vec![self.cfg.pad_token_id; padded_len];
        buf[..real_len].copy_from_slice(ids);
        let ids_tensor = ops.from_slice_i64(&buf, &[1, padded_len])?;
        let out = self.model.forward(ops, &ids_tensor)?;
        // logits: (1, padded_len, vocab). The next-token prediction
        // sits at index `real_len - 1` (logits[t] predicts ids[t+1]).
        let logits_shape = out.logits.shape().to_vec();
        debug_assert_eq!(logits_shape[0], 1, "batch must be 1 for generation");
        let v = logits_shape[2];
        // Slice the row we care about.
        let row = ops.narrow(&out.logits, 1, real_len - 1, 1)?; // (1, 1, V)
        let host: Vec<f32> = ops.to_vec_f32(&row)?;
        assert_eq!(host.len(), v);
        let id = self.sampler.sample(&host, rng);
        Ok(id as i64)
    }
}

fn pad_up(n: usize, multiple: usize) -> usize {
    let r = n.is_multiple_of(multiple);
    if r {
        n.max(multiple)
    } else {
        ((n / multiple) + 1) * multiple
    }
}

/// Stateful recurrent decode (memory-efficiency plan, Phase C:
/// O(1)-memory generation).
///
/// Maintains a fixed-size `PhotonDecodeState` (Mamba2 SSM + conv state
/// per layer, chunk-bounded embedding rings) across generated tokens
/// instead of re-forwarding the whole padded sequence every step
/// (`Generator`, above) — see `pm_core::PhotonMamba::{prefill, commit,
/// predict_logits}` for the per-call math. HierGen-first (Phase C
/// spec): matches `Generator`'s greedy output token-for-token on *any*
/// weights, by construction — `predict_logits` reproduces the same
/// pad-completion `Generator::step`'s whole-sequence padding does,
/// just incrementally and non-destructively.
pub struct StatefulGenerator<'a, O: Ops> {
    model: &'a PhotonMamba<O>,
    /// Reuses `GenerateConfig`; `chunk_product` is unused here — there
    /// is no padding on this path, which is the whole point.
    cfg: GenerateConfig,
    sampler: Sampler,
}

impl<'a, O: Ops> StatefulGenerator<'a, O>
where
    O::Tensor: Clone,
{
    #[must_use]
    pub fn new(model: &'a PhotonMamba<O>, cfg: GenerateConfig, sampler: Sampler) -> Self {
        Self {
            model,
            cfg,
            sampler,
        }
    }

    /// Generate `cfg.max_new_tokens` ids appended after `prompt`.
    /// Returns the entire id sequence (prompt + completion). Peak
    /// memory is flat in `cfg.max_new_tokens` (Phase C's O(1)-memory
    /// goal) — contrast with `Generator::generate`, whose per-step cost
    /// (and, on backends whose autograd retains intermediates, peak
    /// memory) grows with the current sequence length.
    pub fn generate(&self, ops: &O, prompt: &[i64]) -> Result<Vec<i64>, O::Error> {
        self.generate_with_hook(ops, prompt, |_| {})
    }

    /// Same as [`generate`](Self::generate), but calls `on_step(ops)`
    /// after every generated token's `commit` (not during `prefill`).
    ///
    /// This exists so a *caller* can attach best-effort, backend-
    /// specific per-step housekeeping without `pm-infer` (or `pm-core`)
    /// knowing anything about that backend — e.g. `pm-cuda::CudaBackend`
    /// records every op on an autograd tape that only clears on
    /// `Ops::backward`/`CudaBackend::reset_tape`, never on plain
    /// inference-only forward calls (`pm-cli/tests/decode_vram_flat.rs`
    /// measures this directly). Without a hook, `pm-core`'s O(1)-memory
    /// state design is necessary but not sufficient for *that specific
    /// backend*: the state itself stays fixed-size, but the backend's
    /// own bookkeeping can still grow unboundedly underneath it. `O`
    /// stays fully generic here — `on_step` is a closure the caller
    /// supplies, so adding this hook does not require `Ops` growth or
    /// any backend-crate change (CLAUDE.md invariant #2); see
    /// `pm-cli::generate_cmd::end_of_decode_step` for the one
    /// `CudaBackend`-aware hook this workspace actually uses.
    pub fn generate_with_hook(
        &self,
        ops: &O,
        prompt: &[i64],
        mut on_step: impl FnMut(&O),
    ) -> Result<Vec<i64>, O::Error> {
        assert!(!prompt.is_empty(), "prompt must contain at least 1 token");
        let mut rng = Rng::new(self.cfg.seed);
        let mut state = self.model.prefill(ops, prompt, self.cfg.pad_token_id)?;
        let mut out: Vec<i64> = prompt.to_vec();
        for _ in 0..self.cfg.max_new_tokens {
            let logits = self.model.predict_logits(ops, &state)?;
            let host: Vec<f32> = ops.to_vec_f32(&logits)?;
            debug_assert_eq!(host.len(), self.cfg.vocab_size);
            let id = self.sampler.sample(&host, &mut rng) as i64;
            out.push(id);
            self.model.commit(ops, &mut state, id)?;
            on_step(ops);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_up_examples() {
        assert_eq!(pad_up(1, 4), 4);
        assert_eq!(pad_up(3, 4), 4);
        assert_eq!(pad_up(4, 4), 4);
        assert_eq!(pad_up(5, 4), 8);
        assert_eq!(pad_up(16, 4), 16);
    }
}
