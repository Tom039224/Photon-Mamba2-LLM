//! Linear (fully-connected) layer.

use crate::{Dtype, Module, Ops, Param, Parameterized};

/// `y = x @ W [+ b]`.
///
/// `weight` is `(in_features, out_features)`; `bias`, when present,
/// broadcasts as `(out_features,)`. Both are trainable [`Param`]s.
pub struct Linear<O: Ops> {
    pub weight: O::Param,
    pub bias: Option<O::Param>,
    pub in_features: usize,
    pub out_features: usize,
}

impl<O: Ops> Linear<O> {
    /// Allocate zero-initialised trainable weights (and optional zero bias).
    /// Use only when the caller will overwrite the tensors afterwards.
    pub fn zeros(
        ops: &O,
        in_features: usize,
        out_features: usize,
        bias: bool,
        dtype: Dtype,
    ) -> Result<Self, O::Error> {
        Ok(Self {
            weight: ops.param_zeros(&[in_features, out_features], dtype)?,
            bias: if bias {
                Some(ops.param_zeros(&[out_features], dtype)?)
            } else {
                None
            },
            in_features,
            out_features,
        })
    }

    /// Constant-fill constructor for tests. `weight` is filled with
    /// `weight_scale`; `bias`, when requested, is zero.
    pub fn from_constants(
        ops: &O,
        in_features: usize,
        out_features: usize,
        bias: bool,
        weight_scale: f32,
    ) -> Result<Self, O::Error> {
        let w_data = vec![weight_scale; in_features * out_features];
        let weight = ops.param_from_slice_f32(&w_data, &[in_features, out_features])?;
        let bias = if bias {
            Some(ops.param_from_slice_f32(&vec![0.0; out_features], &[out_features])?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Construct from pre-existing trainable params (e.g. loaded from a
    /// checkpoint). Shape correctness is the caller's responsibility.
    pub fn from_params(
        in_features: usize,
        out_features: usize,
        weight: O::Param,
        bias: Option<O::Param>,
    ) -> Self {
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }
}

impl<O: Ops> Module<O> for Linear<O> {
    fn forward(&self, ops: &O, x: &O::Tensor) -> Result<O::Tensor, O::Error> {
        use crate::Tensor;
        assert_eq!(
            x.shape().last().copied(),
            Some(self.in_features),
            "Linear: trailing dim of input must equal in_features"
        );
        // Cast the fp32-native weight/bias to `x`'s (ambient) dtype
        // inline — a no-op when `x` is fp32 (memory-efficiency plan
        // Phase A2). This lets `Linear` participate in a bf16 forward
        // (used by `ContextChunker`/`ContextConverter`) without storing
        // anything but fp32 params.
        let cdt = x.dtype();
        let w = ops.to_dtype(self.weight.as_tensor(), cdt)?;
        let y = ops.matmul(x, &w)?;
        match &self.bias {
            Some(b) => {
                let bias = ops.to_dtype(b.as_tensor(), cdt)?;
                ops.add(&y, &bias)
            }
            None => Ok(y),
        }
    }
}

impl<O: Ops> Parameterized<O> for Linear<O> {
    fn append_params<'a>(&'a self, out: &mut Vec<&'a O::Param>) {
        out.push(&self.weight);
        if let Some(b) = &self.bias {
            out.push(b);
        }
    }
}
