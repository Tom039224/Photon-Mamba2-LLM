//! Top-level PHOTON × Mamba2 model assembly (D.8).

use crate::checkpoint::CheckpointState;
use crate::photon::{
    ContextEncoderState, EncodedHierarchy, HierarchicalDecoder, HierarchicalEncoder,
    PhotonDecodeState, TokenEmbedding,
};
use crate::{Dtype, Module, Ops, Param, Parameterized, Tensor};

pub struct PhotonMamba<O: Ops> {
    pub embed: TokenEmbedding<O>,
    pub encoder: HierarchicalEncoder<O>,
    pub decoder: HierarchicalDecoder<O>,
    /// Ambient dtype the forward pass computes big `(B,T,·)` activations
    /// in. `Dtype::F32` (the default set by [`PhotonMamba::new`])
    /// reproduces the original all-fp32 forward bit-for-bit.
    /// `Dtype::BF16` halves activation-tape memory (memory-efficiency
    /// plan Phase A2, `docs/perf-log.md` 2026-07-03); parameters and the
    /// optimizer stay fp32 regardless — see `Mamba2Block::forward`'s
    /// "fp32 islands" for the numerically-sensitive sub-computations
    /// that always run in fp32. Set via [`PhotonMamba::with_compute_dtype`].
    pub compute_dtype: Dtype,
}

/// Everything a forward pass produces. Kept around so the training loop
/// can compute both the next-token cross-entropy and the per-level
/// recursive consistency loss (F.2) from a single forward.
pub struct PhotonForwardOutput<O: Ops> {
    /// `(B, T, vocab)` — lm_head logits from the bottom-level decoder
    /// prediction, fed through the tied embedding.
    pub logits: O::Tensor,
    /// Encoder intermediate streams. `encoded.encoded[l]` is the post-
    /// encoder representation at level l (recursive consistency target).
    pub encoded: EncodedHierarchy<O>,
    /// `predicted[l]` is the decoder's level-l reconstruction. Same
    /// indexing as `encoded.encoded[..L-1]`.
    pub predicted: Vec<O::Tensor>,
}

/// Same as [`PhotonForwardOutput`] but **without** the lm_head logits.
///
/// `logits = lm_head_logits(predicted[0])` is a `(B, T, vocab)` tensor —
/// for training, materialising it (plus `cross_entropy_loss`'s
/// `log_softmax` copy) is the dominant activation-memory cost at long
/// `T` (memory-efficiency plan: fused/tiled cross-entropy,
/// `docs/perf-log.md` 2026-07-03). [`PhotonMamba::forward_no_lm_head`] /
/// [`PhotonMamba::forward_checkpointed_no_lm_head`] stop short of that
/// last matmul so the training loop can instead call
/// `Ops::fused_cross_entropy` directly on `predicted[0]`, tiling the
/// `(rows, V)` logits instead of holding the full `(B, T, V)` tensor.
/// `PhotonMamba::forward`/`forward_checkpointed` (eval/inference path,
/// unchanged) are thin wrappers that add the lm_head step back on top.
pub struct PhotonHiddenOutput<O: Ops> {
    pub encoded: EncodedHierarchy<O>,
    pub predicted: Vec<O::Tensor>,
}

impl<O: Ops> PhotonMamba<O> {
    pub fn new(
        embed: TokenEmbedding<O>,
        encoder: HierarchicalEncoder<O>,
        decoder: HierarchicalDecoder<O>,
    ) -> Self {
        let n_enc = encoder.n_levels();
        let n_dec = decoder.n_levels();
        assert_eq!(
            n_dec + 1,
            n_enc,
            "decoder must have exactly one fewer level than encoder \
             (got {n_dec} decoder vs {n_enc} encoder)"
        );
        Self {
            embed,
            encoder,
            decoder,
            compute_dtype: Dtype::F32,
        }
    }

    /// Opt into a non-default forward-pass compute dtype. See the
    /// [`compute_dtype`](PhotonMamba::compute_dtype) field docs.
    #[must_use]
    pub fn with_compute_dtype(mut self, dtype: Dtype) -> Self {
        self.compute_dtype = dtype;
        self
    }

    /// End-to-end forward, including the `(B, T, vocab)` lm_head logits.
    ///
    /// Eval / inference / generation call sites (`pm-infer`,
    /// `pm eval hellaswag`, tests) use this unchanged. The training loop
    /// (`pm-cli::train_cmd`) uses [`forward_no_lm_head`](Self::forward_no_lm_head)
    /// instead — see [`PhotonHiddenOutput`]'s docs for why.
    pub fn forward(&self, ops: &O, ids: &O::Tensor) -> Result<PhotonForwardOutput<O>, O::Error> {
        let hidden = self.forward_no_lm_head(ops, ids)?;
        let logits = self.embed.lm_head_logits(ops, &hidden.predicted[0])?;
        Ok(PhotonForwardOutput {
            logits,
            encoded: hidden.encoded,
            predicted: hidden.predicted,
        })
    }

    /// Same as [`forward`](Self::forward) but stops at the decoder's
    /// bottom-level prediction — no lm_head matmul, so no `(B, T,
    /// vocab)` tensor is ever materialised. See [`PhotonHiddenOutput`].
    pub fn forward_no_lm_head(
        &self,
        ops: &O,
        ids: &O::Tensor,
    ) -> Result<PhotonHiddenOutput<O>, O::Error> {
        // The one entry point for `compute_dtype`: everything
        // downstream (encoder/decoder Mamba2Blocks, chunker/converter
        // Linear projections, lm_head) inherits `x0`'s dtype and casts
        // fp32 params to it inline. No-op when `compute_dtype` is the
        // embedding table's native `F32` (memory-efficiency plan Phase A2).
        let x0_native = self.embed.forward(ops, ids)?;
        let x0 = ops.to_dtype(&x0_native, self.compute_dtype)?;
        let encoded = self.encoder.encode(ops, &x0)?;

        let mut level_inputs: Vec<O::Tensor> = Vec::with_capacity(self.encoder.n_levels());
        level_inputs.push(x0);
        for l in 0..self.encoder.n_levels() - 1 {
            // Cheap shape-preserving "clone" via reshape no-op so we
            // don't depend on Tensor: Clone.
            let shape = encoded.chunked[l].shape().to_vec();
            let cloned = ops.reshape(&encoded.chunked[l], &shape)?;
            level_inputs.push(cloned);
        }

        let predicted = self.decoder.decode(ops, &encoded, &level_inputs)?;
        Ok(PhotonHiddenOutput { encoded, predicted })
    }
}

impl<O: Ops> Parameterized<O> for PhotonMamba<O> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>) {
        self.embed.append_params(out);
        self.encoder.append_params(out);
        self.decoder.append_params(out);
    }
}

// -------- Phase C: O(1)-memory recurrent decode --------
//
// `forward` re-scans the whole sequence every call — right for teacher-
// forced training, but reusing it naively for autoregressive decoding
// (`pm-infer::Generator`, the padded re-forward baseline) means every
// generated token re-pays the cost of every token before it. The
// methods below instead carry a fixed-size `PhotonDecodeState` (SSM `h`
// + conv window per Mamba2 layer, chunk-size-bounded embedding rings —
// see its own docs) across calls, giving O(1) memory in the generation
// length. `Mamba2Block::step` has the per-layer math and its own
// derivation that this is exact, not approximate; `predict_logits`
// (below) has the PHOTON-hierarchy side: normally a chunk's prediction
// needs that *whole* chunk's higher-level context, which doesn't exist
// yet mid-chunk, so `predict_logits` provisionally (non-destructively)
// pad-completes the in-progress chunk one token early — the same
// pad-then-predict trick `pm-infer::Generator::step` already does by
// padding the *whole* sequence, just incremental and O(1) here.

impl<O: Ops> PhotonMamba<O>
where
    O::Tensor: Clone,
{
    /// Guard the structural assumptions the decode path relies on.
    /// These are Phase 1 / Phase C simplifications (matching
    /// `configs/photon_mamba_100m.toml` and `Mamba2Block::forward`'s
    /// own `n_groups` assert), not fundamental limits of the design.
    fn assert_decode_supported(&self) {
        assert_eq!(
            self.encoder.n_levels(),
            2,
            "PhotonMamba decode: only L=2 is supported (got {})",
            self.encoder.n_levels()
        );
        assert_eq!(
            self.decoder.n_levels(),
            1,
            "PhotonMamba decode: only 1 decoder level is supported (got {})",
            self.decoder.n_levels()
        );
        for level in &self.encoder.levels {
            for layer in &level.encoder.layers {
                let (g, h) = (layer.config.n_groups, layer.config.n_heads);
                assert!(
                    g == 1 || g == h,
                    "PhotonMamba decode: n_groups must be 1 or n_heads (Phase 1), \
                     got n_groups={g}, n_heads={h}"
                );
            }
        }
    }

    /// Build a fresh [`PhotonDecodeState`] and commit every prompt
    /// token into it (Prefill-A in the Phase C spec: parity with
    /// `forward` by construction, and O(1) memory since each `commit`
    /// only ever touches the fixed-size state). `prompt` must be
    /// non-empty; batch is fixed at 1 (single-sequence decode, v1).
    pub fn prefill(
        &self,
        ops: &O,
        prompt: &[i64],
        pad_token_id: i64,
    ) -> Result<PhotonDecodeState<O>, O::Error> {
        assert!(
            !prompt.is_empty(),
            "PhotonMamba::prefill: prompt must contain at least 1 token"
        );
        self.assert_decode_supported();
        let batch = 1;
        let enc_l0 = self.encoder.levels[0].encoder.zero_state(ops, batch)?;
        let enc_l1 = self.encoder.levels[1].encoder.zero_state(ops, batch)?;
        let mut state = PhotonDecodeState::empty(enc_l0, enc_l1, pad_token_id, batch);
        for &id in prompt {
            self.commit(ops, &mut state, id)?;
        }
        Ok(state)
    }

    /// Commit one real token into `state`: advance the level-0 encoder
    /// by one step, and — every `C`-th token — advance the level-1
    /// encoder by one step over the just-completed chunk
    /// (`HierarchicalEncoder::encode`'s bottom-up recursion, PHOTON
    /// §2.1, specialised to one step at a time).
    pub fn commit(
        &self,
        ops: &O,
        state: &mut PhotonDecodeState<O>,
        id: i64,
    ) -> Result<(), O::Error> {
        // Finalize any chunk completion deferred by the *previous*
        // commit (see `PhotonDecodeState::last_chunk_e1` docs): this
        // token starts a new chunk, so it's now safe to fold the last
        // chunk's embeddings into `prev_chunk_embeds` — up to this
        // point it had to keep reading as the chunk *before* that one,
        // in case `predict_logits` was called for that chunk's own
        // last position in between.
        if state.last_chunk_e1.take().is_some() {
            state.prev_chunk_embeds = std::mem::take(&mut state.cur_chunk_embeds);
        }

        let ids = ops.from_slice_i64(&vec![id; state.batch], &[state.batch, 1])?;
        let x0_native = self.embed.forward(ops, &ids)?;
        let x0 = ops.to_dtype(&x0_native, self.compute_dtype)?; // (B,1,D)

        let (e0, enc_l0_next) = self.encoder.levels[0]
            .encoder
            .step(ops, &x0, &state.enc_l0)?;
        state.enc_l0 = enc_l0_next;
        state.cur_chunk_embeds.push(x0);
        state.e0_ring.push(e0);
        state.committed_len += 1;

        // INVARIANT: `assert_decode_supported` (called from `prefill`,
        // the only entry point that constructs `state`) guarantees
        // n_levels()==2, so level 0 is never top; `HierarchicalEncoder
        // ::from_levels` guarantees a non-top level always carries a
        // chunker.
        let chunker = self.encoder.levels[0]
            .chunker
            .as_ref()
            .expect("level 0 of an L=2 HierarchicalEncoder always has a chunker");
        debug_assert!(state.e0_ring.len() <= chunker.chunk_size);
        if state.e0_ring.len() == chunker.chunk_size {
            let ring_refs: Vec<&O::Tensor> = state.e0_ring.iter().collect();
            let ring_cat = ops.concat(&ring_refs, 1)?; // (B, C, D)
            let ch = chunker.forward(ops, &ring_cat)?; // (B, 1, D)
            let (e1, enc_l1_next) = self.encoder.levels[1]
                .encoder
                .step(ops, &ch, &state.enc_l1)?;
            state.enc_l1 = enc_l1_next;
            // `prev_chunk_embeds` / `cur_chunk_embeds` deliberately
            // untouched here — see the doc comment on `last_chunk_e1`
            // and the top of this function.
            state.last_chunk_e1 = Some(e1);
            state.e0_ring.clear();
        }
        Ok(())
    }

    /// Predict the logits for the *next* token, i.e. `predicted[0][m-1]`
    /// in `forward`'s terms (`m = state.committed_len`), without
    /// mutating `state`.
    ///
    /// Must assemble the chunk-local decoder's input *exactly* as
    /// `HierarchicalDecoder::decode` / `PhotonMamba::forward_checkpointed`
    /// do (self-shift teacher forcing, deviations P.3): `R` converter
    /// latents from the current chunk's high-level context, followed by
    /// the `C` "local" slots — the *previous* chunk's committed
    /// embeddings, or the converter's `starting_latent` seed for chunk
    /// 0. Two cases for the `R`-latent context, both exact (never an
    /// approximation of `forward`):
    /// - `m-1` is *not* a chunk's last position: the chunk is still in
    ///   progress, so provisionally (non-destructively) pad-complete it
    ///   with `pad_token_id` — the same trick `Generator::step` already
    ///   does by padding the *whole* sequence, just incremental here.
    /// - `m-1` *is* a chunk's last position (`state.e0_ring.is_empty()`):
    ///   that chunk is already fully real — `commit` computed and
    ///   cached its exact `e1` in `state.last_chunk_e1` — so no
    ///   pad-completion is needed or correct (padding here would
    ///   silently answer for the *next*, not-yet-started chunk instead).
    ///
    /// Any deviation here would make greedy decode silently diverge
    /// from `forward` — guarded by
    /// `pm-candle/tests/decode_state.rs::prefill_predict_matches_forward_pad_completion`
    /// and `pm-infer/tests/stateful_generator_parity.rs`.
    pub fn predict_logits(
        &self,
        ops: &O,
        state: &PhotonDecodeState<O>,
    ) -> Result<O::Tensor, O::Error> {
        assert!(
            state.committed_len > 0,
            "predict_logits: commit at least one token first (call prefill)"
        );
        let m = state.committed_len;
        // INVARIANT: see `commit`.
        let chunker = self.encoder.levels[0]
            .chunker
            .as_ref()
            .expect("level 0 of an L=2 HierarchicalEncoder always has a chunker");
        let c = chunker.chunk_size;
        let k = (m - 1) / c;
        let j = (m - 1) % c;

        let e1_k: O::Tensor = if let Some(cached) = &state.last_chunk_e1 {
            // `m-1` is chunk k's own last position: chunk k is already
            // fully committed (see the doc comment above).
            cached.clone()
        } else {
            // 1. Provisional pad-completion of the in-progress chunk.
            // Never mutates `state`: `ContextEncoder::step` is
            // functional, so the provisional tail states are simply
            // never written back.
            let enc_l0 = &self.encoder.levels[0].encoder;
            let mut ring: Vec<O::Tensor> = state.e0_ring.clone();
            let mut provisional_l0: Option<ContextEncoderState<O>> = None;
            for _ in 0..(c - ring.len()) {
                let pad_ids =
                    ops.from_slice_i64(&vec![state.pad_token_id; state.batch], &[state.batch, 1])?;
                let xp_native = self.embed.forward(ops, &pad_ids)?;
                let xp = ops.to_dtype(&xp_native, self.compute_dtype)?;
                let cur: &ContextEncoderState<O> = provisional_l0.as_ref().unwrap_or(&state.enc_l0);
                let (e0p, next) = enc_l0.step(ops, &xp, cur)?;
                provisional_l0 = Some(next);
                ring.push(e0p);
            }

            // 2. Chunk the completed ring; 3. provisional level-1 step
            // (discarded — never written back to `state`).
            let ring_refs: Vec<&O::Tensor> = ring.iter().collect();
            let ring_cat = ops.concat(&ring_refs, 1)?; // (B, C, D)
            let ch = chunker.forward(ops, &ring_cat)?; // (B, 1, D)
            let (e1_k, _discarded) =
                self.encoder.levels[1]
                    .encoder
                    .step(ops, &ch, &state.enc_l1)?;
            e1_k
        };

        // 4. Expand to R converter latents.
        let dl = &self.decoder.levels[0];
        let conv_out = dl.converter.forward(ops, &e1_k)?; // (B, R, D)
        let r = dl.decoder.r_l;
        let d_model = dl.converter.d_out;

        // 5. Self-shift "local" C slots (deviations P.3).
        let local = if k == 0 {
            let seed = dl.converter.starting_latent.as_tensor();
            let seed_cast = ops.to_dtype(seed, self.compute_dtype)?;
            let seed_r = ops.reshape(&seed_cast, &[1, 1, d_model])?;
            ops.broadcast_as(&seed_r, &[state.batch, c, d_model])?
        } else {
            assert_eq!(
                state.prev_chunk_embeds.len(),
                c,
                "predict_logits: prev_chunk_embeds must hold exactly {c} entries \
                 once k>=1 (got {})",
                state.prev_chunk_embeds.len()
            );
            let refs: Vec<&O::Tensor> = state.prev_chunk_embeds.iter().collect();
            ops.concat(&refs, 1)? // (B, C, D)
        };

        let conv_4d = ops.reshape(&conv_out, &[state.batch, 1, r, d_model])?;
        let local_4d = ops.reshape(&local, &[state.batch, 1, c, d_model])?;
        let chunk = ops.concat(&[&conv_4d, &local_4d], 2)?; // (B, 1, R+C, D)

        // 6. Chunk-local decode; 7. slice out this token's prediction.
        let dec = dl.decoder.forward(ops, &chunk)?; // (B, 1, R+C, D)
        let pred_slice = ops.narrow(&dec, 2, r + j, 1)?; // (B,1,1,D)
        let pred = ops.reshape(&pred_slice, &[state.batch, 1, d_model])?; // (B,1,D)

        // 8. lm_head.
        self.embed.lm_head_logits(ops, &pred)
    }
}

// -------- Activation checkpointing entry points --------
//
// The default `forward` keeps the whole autograd tape alive, which OOMs
// at B > 1 because the SSD scan inside every `Mamba2Block` retains an
// `(B, H, T, T)` decay matrix (~24 MB at B=2, T=512, H=12). 30 blocks
// share that footprint simultaneously.
//
// `forward_checkpointed` instead inserts a fresh boundary `Param` after
// every `Mamba2Block` and drops the block's intermediates immediately.
// `recompute_block` walks the same `block_id` numbering used during
// forward so backward can re-run individual blocks. See
// `pm-core::checkpoint` for the segment data flow.

impl<O: Ops> PhotonMamba<O>
where
    O::Tensor: Clone,
    O::Param: Clone,
{
    /// Same as [`forward`](Self::forward) but each `Mamba2Block` is
    /// wrapped in a checkpoint segment. Returns the usual output **plus**
    /// the `CheckpointState` the caller must hand to
    /// `pm_core::checkpoint::checkpoint_backward` after the main
    /// backward pass.
    ///
    /// Eval / inference call sites that want checkpointing (none today)
    /// would use this. The training loop
    /// (`pm-cli::train_cmd`, ckpt branch) uses
    /// [`forward_checkpointed_no_lm_head`](Self::forward_checkpointed_no_lm_head)
    /// instead — see [`PhotonHiddenOutput`]'s docs for why.
    pub fn forward_checkpointed(
        &self,
        ops: &O,
        ids: &O::Tensor,
    ) -> Result<(PhotonForwardOutput<O>, CheckpointState<O>), O::Error> {
        let (hidden, cp) = self.forward_checkpointed_no_lm_head(ops, ids)?;
        let logits = self.embed.lm_head_logits(ops, &hidden.predicted[0])?;
        Ok((
            PhotonForwardOutput {
                logits,
                encoded: hidden.encoded,
                predicted: hidden.predicted,
            },
            cp,
        ))
    }

    /// Same as [`forward_checkpointed`](Self::forward_checkpointed) but
    /// stops at the decoder's bottom-level prediction — no lm_head
    /// matmul, so no `(B, T, vocab)` tensor is ever materialised. See
    /// [`PhotonHiddenOutput`].
    pub fn forward_checkpointed_no_lm_head(
        &self,
        ops: &O,
        ids: &O::Tensor,
    ) -> Result<(PhotonHiddenOutput<O>, CheckpointState<O>), O::Error> {
        let mut cp = CheckpointState::new();

        // 1. Embed (cheap, no checkpoint needed). Same `compute_dtype`
        // entry point as `forward` — see its comment.
        let x0_native = self.embed.forward(ops, ids)?;
        let x0 = ops.to_dtype(&x0_native, self.compute_dtype)?;

        // 2. Encoder, checkpointed per Mamba2Block.
        let n_enc = self.encoder.n_levels();
        let mut encoded: Vec<O::Tensor> = Vec::with_capacity(n_enc);
        let mut chunked: Vec<O::Tensor> = Vec::with_capacity(n_enc.saturating_sub(1));
        let mut block_id = 0usize;

        let e0 = self.encoder.levels[0]
            .encoder
            .forward_checkpointed(ops, &x0, &mut cp, block_id)?;
        block_id += self.encoder.levels[0].encoder.n_layers();
        encoded.push(e0);

        for l in 0..n_enc {
            if let Some(ch) = &self.encoder.levels[l].chunker {
                let next_stream = ch.forward(ops, &encoded[l])?;
                chunked.push(next_stream);
                let next_l = l + 1;
                if next_l < n_enc {
                    let e_next = self.encoder.levels[next_l].encoder.forward_checkpointed(
                        ops,
                        &chunked[l],
                        &mut cp,
                        block_id,
                    )?;
                    block_id += self.encoder.levels[next_l].encoder.n_layers();
                    encoded.push(e_next);
                }
            }
        }

        // 3. Build level_inputs (same as PhotonMamba::forward).
        let mut level_inputs: Vec<O::Tensor> = Vec::with_capacity(n_enc);
        level_inputs.push(x0);
        for l in 0..n_enc - 1 {
            let shape = chunked[l].shape().to_vec();
            level_inputs.push(ops.reshape(&chunked[l], &shape)?);
        }

        // 4. Decoder, checkpointed per Mamba2Block. Iterate top-down to
        //    match HierarchicalDecoder::decode.
        let encoded_hier = EncodedHierarchy { encoded, chunked };
        let n_dec = self.decoder.n_levels();
        let mut predicted_top_down: Vec<O::Tensor> = Vec::with_capacity(n_dec);
        for l in (0..n_dec).rev() {
            let dl = &self.decoder.levels[l];
            let x_high = &encoded_hier.encoded[l + 1];
            let x_self = &level_inputs[l];

            let conv_out = dl.converter.forward(ops, x_high)?;
            let shape_self = x_self.shape();
            let (b, t_l, d_l) = (shape_self[0], shape_self[1], shape_self[2]);
            let r = dl.decoder.r_l;
            let c = dl.decoder.c_l;
            let s = t_l / c;

            // Inlined `shift_right_with_seed` (same math as the
            // private helper in hierarchical_decoder, including the
            // fp32-native `seed` -> `x_self`-dtype cast for Phase A2).
            let seed = dl.converter.starting_latent.as_tensor();
            let seed_cast = ops.to_dtype(seed, x_self.dtype())?;
            let seed_r = ops.reshape(&seed_cast, &[1, 1, d_l])?;
            let seed_b = ops.broadcast_as(&seed_r, &[b, c, d_l])?;
            let kept = ops.narrow(x_self, 1, 0, t_l - c)?;
            let shifted = ops.concat(&[&seed_b, &kept], 1)?;

            let conv_4d = ops.reshape(&conv_out, &[b, s, r, d_l])?;
            let shifted_4d = ops.reshape(&shifted, &[b, s, c, d_l])?;
            let chunks = ops.concat(&[&conv_4d, &shifted_4d], 2)?;

            let decoded = dl
                .decoder
                .forward_checkpointed(ops, &chunks, &mut cp, block_id)?;
            block_id += dl.decoder.n_layers();

            let trailing = ops.narrow(&decoded, 2, r, c)?;
            let pred = ops.reshape(&trailing, &[b, t_l, d_l])?;
            predicted_top_down.push(pred);
        }
        predicted_top_down.reverse();

        Ok((
            PhotonHiddenOutput {
                encoded: encoded_hier,
                predicted: predicted_top_down,
            },
            cp,
        ))
    }

    /// Lookup and run the `block_id`-th `Mamba2Block` forward. The
    /// numbering matches the one [`forward_checkpointed`] uses to
    /// allocate ids, so `checkpoint_backward` can recompute exactly
    /// the segment it saved.
    pub fn recompute_block(
        &self,
        ops: &O,
        block_id: usize,
        input: &O::Tensor,
    ) -> Result<O::Tensor, O::Error> {
        let mut counter = 0usize;
        // Encoder (level 0, 1, ..., L-1).
        for level in &self.encoder.levels {
            let n = level.encoder.n_layers();
            if block_id < counter + n {
                return level.encoder.layers[block_id - counter].forward(ops, input);
            }
            counter += n;
        }
        // Decoder (top-down: level L-2, L-3, ..., 0).
        let n_dec = self.decoder.n_levels();
        for l in (0..n_dec).rev() {
            let n = self.decoder.levels[l].decoder.n_layers();
            if block_id < counter + n {
                return self.decoder.levels[l].decoder.layers[block_id - counter]
                    .forward(ops, input);
            }
            counter += n;
        }
        panic!("recompute_block: block_id {block_id} out of range (max {counter})")
    }
}
