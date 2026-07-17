//! Embedding (token-lookup) layer.

use crate::{Dtype, Module, Ops, Param, Parameterized};

/// Looks up `weight[index]` for each integer entry in the input tensor.
///
/// `weight` has shape `(num_embeddings, embedding_dim)`; input has any
/// integer shape `S`; output has shape `S ++ [embedding_dim]`. `weight`
/// is a trainable [`Param`].
pub struct Embedding<O: Ops> {
    pub weight: O::Param,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
}

impl<O: Ops> Embedding<O> {
    pub fn zeros(
        ops: &O,
        num_embeddings: usize,
        embedding_dim: usize,
        dtype: Dtype,
    ) -> Result<Self, O::Error> {
        Ok(Self {
            weight: ops.param_zeros(&[num_embeddings, embedding_dim], dtype)?,
            num_embeddings,
            embedding_dim,
        })
    }

    /// Constant-fill for tests. Row `r` is filled with `r as f32 * row_step`
    /// so that lookup correctness is easy to check.
    pub fn arange_rows(
        ops: &O,
        num_embeddings: usize,
        embedding_dim: usize,
        row_step: f32,
    ) -> Result<Self, O::Error> {
        let mut data = vec![0f32; num_embeddings * embedding_dim];
        for r in 0..num_embeddings {
            for c in 0..embedding_dim {
                data[r * embedding_dim + c] = r as f32 * row_step;
            }
        }
        Ok(Self {
            weight: ops.param_from_slice_f32(&data, &[num_embeddings, embedding_dim])?,
            num_embeddings,
            embedding_dim,
        })
    }

    pub fn from_param(num_embeddings: usize, embedding_dim: usize, weight: O::Param) -> Self {
        Self {
            weight,
            num_embeddings,
            embedding_dim,
        }
    }
}

impl<O: Ops> Module<O> for Embedding<O> {
    fn forward(&self, ops: &O, ids: &O::Tensor) -> Result<O::Tensor, O::Error> {
        ops.embedding(self.weight.as_tensor(), ids)
    }
}

impl<O: Ops> Parameterized<O> for Embedding<O> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>) {
        out.push(&self.weight);
    }
}
