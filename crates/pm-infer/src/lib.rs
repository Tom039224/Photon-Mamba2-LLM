//! Photon × Mamba2 inference (generation + evaluation).
//!
//! Two generation strategies:
//!
//! - [`Generator`] — the original *padded re-forward* strategy: every
//!   generation step re-runs `PhotonMamba::forward` on the current id
//!   buffer (padded to a multiple of `chunk_size^(L-1)`), samples the
//!   next token from the trailing logit position, and appends. Cost
//!   (and, on backends whose autograd retains intermediates, peak
//!   memory) grows with the current sequence length — fine for short
//!   generations, and still used by `eval::hellaswag` (G.5 / H.2, which
//!   needs logits at many positions at once) and as a reference/fallback.
//! - [`StatefulGenerator`] — the memory-efficiency plan's Phase C
//!   O(1)-memory decode: carries a fixed-size `PhotonDecodeState`
//!   (`pm_core::PhotonMamba::{prefill, commit, predict_logits}`) across
//!   generated tokens instead of re-forwarding. Matches `Generator`'s
//!   greedy output token-for-token by construction. Default decode mode
//!   for `pm generate`.

#![forbid(unsafe_code)]

pub mod eval;
pub mod generator;
pub mod sampling;

pub use eval::{score_continuation, HellaSwagItem, HellaSwagResult};
pub use generator::{GenerateConfig, Generator, StatefulGenerator};
pub use sampling::Sampler;
