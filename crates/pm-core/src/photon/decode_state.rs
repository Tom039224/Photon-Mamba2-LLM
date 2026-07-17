//! PHOTON decode state (memory-efficiency plan, Phase C: O(1)-memory
//! recurrent decode).
//!
//! Pure data: construction and mutation live on
//! [`PhotonMamba`](crate::PhotonMamba) (`prefill`/`commit`/
//! `predict_logits`, in `model.rs`), which is the one place that
//! already knows the encoder/decoder/embedding wiring these methods
//! need.
//!
//! Paper reference: `Papers/Photon/main.tex` §2.1–2.2 (hierarchical
//! encoder / chunk-local decoder); this struct is the "what has been
//! seen so far" state a PHOTON generation loop must carry to reproduce
//! `PhotonMamba::forward`'s teacher-forced computation one token at a
//! time.

use super::encoder::ContextEncoderState;
use crate::Ops;

/// Persistent state a `PhotonMamba` generation loop carries across
/// `predict_logits`/`commit` calls. Sized `O(1)` in the number of
/// generated tokens: `enc_l0`/`enc_l1` are fixed-size per-layer SSM +
/// conv states ([`ContextEncoderState`]), and the three `Vec<O::Tensor>`
/// buffers below are bounded by the chunk size `C` (PHOTON deviations
/// P.2: `C = 4`), never by sequence length.
///
/// No chunk-local-decoder state is kept: `ChunkLocalDecoder::forward`
/// is cheap to rerun over its bounded `R + C` window every
/// `predict_logits` call (PHOTON §2.2), so there is nothing to persist
/// there. The converter/chunker are stateless `Linear` projections,
/// reused as-is (`DecoderLevel::converter`, `HierarchicalLevel::chunker`).
pub struct PhotonDecodeState<O: Ops> {
    /// Level-0 (token-level) `ContextEncoder` state, advanced by every
    /// committed token `0..committed_len`.
    pub enc_l0: ContextEncoderState<O>,
    /// Level-1 `ContextEncoder` state, advanced once per *completed*
    /// chunk (`committed_len / C` times).
    pub enc_l1: ContextEncoderState<O>,
    /// `e0` (level-0 encoder output) of the current, not-yet-complete
    /// chunk. `len() == committed_len % C`; each entry is `(B, 1, D)`.
    pub e0_ring: Vec<O::Tensor>,
    /// Number of tokens committed so far (`m` in the Phase C spec).
    pub committed_len: usize,
    /// Raw level-0 embeddings `x0` of the last *completed* chunk — the
    /// self-shift teacher-forcing "local" context (deviations P.3) for
    /// whichever chunk is currently in progress. Empty until the first
    /// chunk completes.
    pub prev_chunk_embeds: Vec<O::Tensor>,
    /// Raw level-0 embeddings `x0` of the current, not-yet-complete
    /// chunk (becomes `prev_chunk_embeds` once it completes).
    pub cur_chunk_embeds: Vec<O::Tensor>,
    /// Token id used to pad-complete the in-progress chunk when
    /// computing provisional logits (`PhotonMamba::predict_logits`).
    pub pad_token_id: i64,
    /// Batch size. Fixed at 1 in v1 (single-sequence decode).
    pub batch: usize,
    /// Set exactly when the most recently committed token was the
    /// *last* position of its chunk (`committed_len` is a positive
    /// multiple of `C`): that chunk's real, non-provisional level-1
    /// encoder output, `(B, 1, D)`.
    ///
    /// Exists because `commit` folds a completed chunk's contribution
    /// into `enc_l1` (and caches its `e1` here) *before*
    /// `predict_logits` is asked for that same chunk's own last
    /// position (`predicted[0][committed_len-1]`) — at that exact
    /// instant the chunk is already fully real, so `predict_logits`
    /// must use it directly instead of re-deriving a provisional one
    /// via pad-completion (which would silently answer for the *next*,
    /// not-yet-started chunk instead). Consumed (cleared) at the start
    /// of the *next* `commit`, which is also when `prev_chunk_embeds`
    /// is finally updated to this chunk's embeddings — deferred for
    /// the same reason: `prev_chunk_embeds` must still read as the
    /// chunk *before* this one for as long as `last_chunk_e1` is live.
    pub last_chunk_e1: Option<O::Tensor>,
}

impl<O: Ops> PhotonDecodeState<O> {
    /// All-zero / empty state for a fresh generation.
    pub fn empty(
        enc_l0: ContextEncoderState<O>,
        enc_l1: ContextEncoderState<O>,
        pad_token_id: i64,
        batch: usize,
    ) -> Self {
        Self {
            enc_l0,
            enc_l1,
            e0_ring: Vec::new(),
            committed_len: 0,
            prev_chunk_embeds: Vec::new(),
            cur_chunk_embeds: Vec::new(),
            pad_token_id,
            batch,
            last_chunk_e1: None,
        }
    }
}
