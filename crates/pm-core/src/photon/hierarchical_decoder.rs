//! PHOTON hierarchical decoder (D.7).
//!
//! Top-down: at every non-top level `l` we expand the higher level's
//! stream into level-`l` latents (`ContextConverter`), splice those into
//! teacher-forced level-`l` chunks, and run the [`ChunkLocalDecoder`].
//!
//! ### Chunk assembly (self-shift teacher forcing, deviations.md P.3)
//!
//! For each chunk `k` at level `l`:
//!
//! ```text
//!   chunk_k input (R + C, d_l):
//!     [0      .. R)   ← converter output for chunk k (high-level context)
//!     [R      .. R+C) ← level-l tokens at positions [C*k, C*(k+1))
//!                       taken from the right-shifted level-l stream
//!                       (converter.starting_latent fills positions 0..C
//!                        of the shifted stream).
//! ```
//!
//! After the chunk-local decoder, the trailing `C` positions of each
//! chunk's output are kept as the predicted level-`l` stream.
//!
//! Paper reference: `Papers/Photon/main.tex` §2.2.

use crate::photon::{ChunkLocalDecoder, ContextConverter, EncodedHierarchy};
use crate::{Module, Ops, Param, Parameterized, Tensor};

pub struct DecoderLevel<O: Ops> {
    pub converter: ContextConverter<O>,
    pub decoder: ChunkLocalDecoder<O>,
}

impl<O: Ops> DecoderLevel<O> {
    pub fn new(converter: ContextConverter<O>, decoder: ChunkLocalDecoder<O>) -> Self {
        assert_eq!(
            converter.expansion, decoder.r_l,
            "converter.expansion ({}) must equal decoder.r_l ({})",
            converter.expansion, decoder.r_l
        );
        assert_eq!(
            converter.d_out, decoder.layers[0].config.d_model,
            "converter d_out must equal decoder Mamba2 d_model"
        );
        Self { converter, decoder }
    }
}

pub struct HierarchicalDecoder<O: Ops> {
    pub levels: Vec<DecoderLevel<O>>,
}

impl<O: Ops> HierarchicalDecoder<O> {
    pub fn from_levels(levels: Vec<DecoderLevel<O>>) -> Self {
        assert!(
            !levels.is_empty(),
            "HierarchicalDecoder needs at least 1 decoding level"
        );
        Self { levels }
    }

    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Top-down decoding.
    ///
    /// `encoded.encoded.len()` must equal `self.levels.len() + 1`.
    /// `level_inputs.len()` must equal `self.levels.len() + 1`.
    /// `level_inputs[l]` is the tensor that was fed into encoder `l`.
    ///
    /// Returns `predicted` of length `L = self.levels.len()` indexed by
    /// level; `predicted[l]` has the same shape as `level_inputs[l]`.
    pub fn decode(
        &self,
        ops: &O,
        encoded: &EncodedHierarchy<O>,
        level_inputs: &[O::Tensor],
    ) -> Result<Vec<O::Tensor>, O::Error> {
        let n_dec = self.levels.len();
        assert_eq!(
            encoded.encoded.len(),
            n_dec + 1,
            "encoded.encoded length mismatch: got {}, expected {}",
            encoded.encoded.len(),
            n_dec + 1
        );
        assert_eq!(
            level_inputs.len(),
            n_dec + 1,
            "level_inputs length mismatch: got {}, expected {}",
            level_inputs.len(),
            n_dec + 1
        );

        let mut predicted_top_down = Vec::with_capacity(n_dec);
        for l in (0..n_dec).rev() {
            let dl = &self.levels[l];
            let x_high = &encoded.encoded[l + 1];
            let x_self = &level_inputs[l];

            let conv_out = dl.converter.forward(ops, x_high)?;

            let shape_self = x_self.shape();
            assert_eq!(
                shape_self.len(),
                3,
                "level_inputs[{l}] expected (B, T, d), got rank {}",
                shape_self.len()
            );
            let (b, t_l, d_l) = (shape_self[0], shape_self[1], shape_self[2]);
            let r = dl.decoder.r_l;
            let c = dl.decoder.c_l;
            let s = t_l / c;
            assert_eq!(t_l, s * c, "T_l={t_l} must be a multiple of c={c}");

            let shifted = shift_right_with_seed(
                ops,
                x_self,
                dl.converter.starting_latent.as_tensor(),
                c,
                b,
                t_l,
                d_l,
            )?;

            let conv_4d = ops.reshape(&conv_out, &[b, s, r, d_l])?;
            let shifted_4d = ops.reshape(&shifted, &[b, s, c, d_l])?;
            let chunks = ops.concat(&[&conv_4d, &shifted_4d], 2)?;

            let decoded = dl.decoder.forward(ops, &chunks)?;
            let trailing = ops.narrow(&decoded, 2, r, c)?;
            let pred = ops.reshape(&trailing, &[b, t_l, d_l])?;
            predicted_top_down.push(pred);
        }

        predicted_top_down.reverse();
        Ok(predicted_top_down)
    }
}

impl<O: Ops> Parameterized<O> for DecoderLevel<O> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>) {
        self.converter.append_params(out);
        self.decoder.append_params(out);
    }
}

impl<O: Ops> Parameterized<O> for HierarchicalDecoder<O> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>) {
        for lvl in &self.levels {
            lvl.append_params(out);
        }
    }
}

/// Build `[seed seed ... seed | x[..T-C]]` along the time axis.
fn shift_right_with_seed<O: Ops>(
    ops: &O,
    x: &O::Tensor,
    seed: &O::Tensor,
    c: usize,
    b: usize,
    t: usize,
    d: usize,
) -> Result<O::Tensor, O::Error> {
    assert_eq!(seed.shape(), &[d], "seed must be (d,)");
    // `seed` (`ContextConverter::starting_latent`) is a fp32-native
    // Param; cast to `x`'s (ambient) dtype before concatenating them —
    // a no-op when `x` is fp32 (memory-efficiency plan Phase A2).
    let seed_cast = ops.to_dtype(seed, x.dtype())?;
    let seed_r = ops.reshape(&seed_cast, &[1, 1, d])?;
    let seed_b = ops.broadcast_as(&seed_r, &[b, c, d])?;
    let kept = ops.narrow(x, 1, 0, t - c)?;
    ops.concat(&[&seed_b, &kept], 1)
}
