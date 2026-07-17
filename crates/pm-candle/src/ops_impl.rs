//! `pm_core::Ops` implementation backed by `candle-core`.
//!
//! Phase 1 surface area (PLAN groups B.2–B.5, C.3, D.1):
//! - element-wise add/sub/mul/neg/exp
//! - matmul, conv1d (grouped/depthwise), cumsum
//! - rmsnorm, silu, softplus, sigmoid
//! - reshape, transpose, narrow, broadcast_as, concat
//! - embedding (index_select-based)
//! - ssd_scan (delegated to `pm-core::mamba2::ssd_scan_ops_default`)
//!
//! Everything stays on-device so Candle's autograd graph reaches all
//! inputs — see `ssd_scan_backward_reaches_all_inputs` for the guard.

use crate::dtype::to_candle;
use crate::{CandleBackend, CandleParam, CandleTensor, Error};
use pm_core::{Dtype, Ops, Param, Tensor};

impl Ops for CandleBackend {
    type Tensor = CandleTensor;
    type Error = Error;
    type Param = CandleParam;
    type GradStore = candle_core::backprop::GradStore;

    // ---- Construction ----------------------------------------------------

    fn zeros(&self, shape: &[usize], dtype: Dtype) -> Result<Self::Tensor, Self::Error> {
        let t = candle_core::Tensor::zeros(shape, to_candle(dtype)?, self.device())?;
        CandleTensor::new(t)
    }

    fn ones(&self, shape: &[usize], dtype: Dtype) -> Result<Self::Tensor, Self::Error> {
        let t = candle_core::Tensor::ones(shape, to_candle(dtype)?, self.device())?;
        CandleTensor::new(t)
    }

    fn from_slice_f32(&self, data: &[f32], shape: &[usize]) -> Result<Self::Tensor, Self::Error> {
        let t = candle_core::Tensor::from_slice(data, shape, self.device())?;
        CandleTensor::new(t)
    }

    fn from_slice_i64(&self, data: &[i64], shape: &[usize]) -> Result<Self::Tensor, Self::Error> {
        let t = candle_core::Tensor::from_slice(data, shape, self.device())?;
        CandleTensor::new(t)
    }

    fn to_vec_f32(&self, x: &Self::Tensor) -> Result<Vec<f32>, Self::Error> {
        let t = x.inner().to_dtype(candle_core::DType::F32)?.flatten_all()?;
        Ok(t.to_vec1::<f32>()?)
    }

    fn to_vec_i64(&self, x: &Self::Tensor) -> Result<Vec<i64>, Self::Error> {
        let t = x.inner().to_dtype(candle_core::DType::I64)?.flatten_all()?;
        Ok(t.to_vec1::<i64>()?)
    }

    // ---- Dtype conversion --------------------------------------------------

    /// `candle_core::Tensor::to_dtype` already short-circuits to a plain
    /// `clone()` (same storage, no new `Op::ToDType` node) when the
    /// source and target dtypes match, so this is a genuine no-op on
    /// the all-fp32 path. See the trait docstring for why callers may
    /// call this unconditionally.
    fn to_dtype(&self, x: &Self::Tensor, dtype: Dtype) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(x.inner().to_dtype(to_candle(dtype)?)?)
    }

    // ---- Element-wise ----------------------------------------------------

    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(a.inner().broadcast_add(b.inner())?)
    }

    fn sub(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(a.inner().broadcast_sub(b.inner())?)
    }

    fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(a.inner().broadcast_mul(b.inner())?)
    }

    fn neg(&self, a: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(a.inner().neg()?)
    }

    // ---- Linear algebra --------------------------------------------------

    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(a.inner().broadcast_matmul(b.inner())?)
    }

    // ---- Activations / Normalisation -------------------------------------

    /// RMSNorm: `x * weight / sqrt(mean(x^2, -1) + eps)`.
    ///
    /// Uses `rms_norm_slow`, **not** `rms_norm`. The latter is built on
    /// `Tensor::apply_op2_no_bwd` (candle-nn 0.11.0 `ops.rs:684`), i.e.
    /// `BackpropOp::none()` — it has **no backward**, so gradients never
    /// flow past it. With the fused `rms_norm`, backprop through a
    /// `Mamba2Block` reached only `out_proj_weight` (the post-norm
    /// matmul's rhs leaf) and the tied embedding; every pre-norm
    /// parameter (`in_proj_weight`, `conv1d_*`, `a_log`, `d_skip`,
    /// `dt_bias`, `norm_weight`) silently received `None` and was
    /// skipped by the optimiser. `rms_norm_slow` is composed of ordinary
    /// differentiable ops (`sqr`/`sum`/`sqrt`/`div`/`broadcast_mul`) and
    /// upcasts F16/BF16 to F32 internally, so it is autograd-correct and
    /// bf16-safe. Numerically identical forward (same formula).
    fn rmsnorm(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        eps: f32,
    ) -> Result<Self::Tensor, Self::Error> {
        let y = candle_nn::ops::rms_norm_slow(x.inner(), weight.inner(), eps)?;
        CandleTensor::new(y)
    }

    fn silu(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        // SiLU(x) = x * sigmoid(x); candle exposes this on Tensor.
        let y = candle_nn::ops::silu(x.inner())?;
        CandleTensor::new(y)
    }

    fn softplus(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        // softplus(x) = log(1 + exp(x)), via numerically stable identity
        // log(1 + exp(x)) = max(x, 0) + log1p(exp(-|x|)).
        let inner = x.inner();
        let abs_x = inner.abs()?;
        let neg_abs = abs_x.neg()?;
        let log1p_exp = (neg_abs.exp()? + 1.0)?.log()?;
        let relu_x = inner.relu()?;
        let y = (relu_x + log1p_exp)?;
        CandleTensor::new(y)
    }

    fn sigmoid(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        let y = candle_nn::ops::sigmoid(x.inner())?;
        CandleTensor::new(y)
    }

    // ---- Convolution -----------------------------------------------------

    fn conv1d(
        &self,
        x: &Self::Tensor,
        weight: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        // candle's conv1d signature: (weight, padding, stride, dilation, groups).
        let dilation = 1;
        let mut y = x
            .inner()
            .conv1d(weight.inner(), padding, stride, dilation, groups)?;
        if let Some(b) = bias {
            // bias shape (C_out,) → broadcast over (B, C_out, T).
            let b = b.inner().reshape(((), 1))?;
            y = y.broadcast_add(&b)?;
        }
        CandleTensor::new(y)
    }

    // ---- Indexing / reduction --------------------------------------------

    fn cumsum(&self, x: &Self::Tensor, dim: usize) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(x.inner().cumsum(dim)?)
    }

    fn narrow(
        &self,
        x: &Self::Tensor,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(x.inner().narrow(dim, start, len)?)
    }

    // ---- Shape manipulation ----------------------------------------------

    fn reshape(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(x.inner().reshape(shape)?)
    }

    fn transpose(
        &self,
        x: &Self::Tensor,
        dim_a: usize,
        dim_b: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        // candle's transpose returns a non-contiguous view; subsequent
        // reshape calls require contiguity, so materialise here.
        CandleTensor::new(x.inner().transpose(dim_a, dim_b)?.contiguous()?)
    }

    fn broadcast_as(&self, x: &Self::Tensor, shape: &[usize]) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(x.inner().broadcast_as(shape)?.contiguous()?)
    }

    fn concat(&self, tensors: &[&Self::Tensor], dim: usize) -> Result<Self::Tensor, Self::Error> {
        if tensors.is_empty() {
            return Err(Error::NotImplemented("concat: empty tensor list"));
        }
        let inners: Vec<&candle_core::Tensor> = tensors.iter().map(|t| t.inner()).collect();
        let y = candle_core::Tensor::cat(&inners, dim)?;
        CandleTensor::new(y)
    }

    // ---- Embedding -------------------------------------------------------

    /// Embedding lookup via Candle's `IndexSelect`.
    ///
    /// Candle requires the index argument to be 1D, so we flatten
    /// `indices` and reshape the result to `indices.shape() ++ [D]`.
    fn embedding(
        &self,
        table: &Self::Tensor,
        indices: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        let table_shape = table.shape();
        if table_shape.len() != 2 {
            return Err(Error::NotImplemented(
                "embedding: table must be rank-2 (V,D)",
            ));
        }
        let d = table_shape[1];

        let idx_inner = indices.inner();
        let flat = idx_inner.flatten_all()?;
        let gathered = table.inner().index_select(&flat, 0)?;

        let mut out_shape = indices.shape().to_vec();
        out_shape.push(d);
        CandleTensor::new(gathered.reshape(out_shape.as_slice())?)
    }

    // ---- Element-wise transcendental -------------------------------------

    fn exp(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(x.inner().exp()?)
    }

    // ---- Trainable parameters --------------------------------------------

    fn param_from_slice_f32(
        &self,
        data: &[f32],
        shape: &[usize],
    ) -> Result<Self::Param, Self::Error> {
        let t = candle_core::Tensor::from_slice(data, shape, self.device())?;
        CandleParam::from_tensor(t)
    }

    fn param_zeros(&self, shape: &[usize], dtype: Dtype) -> Result<Self::Param, Self::Error> {
        let t = candle_core::Tensor::zeros(shape, to_candle(dtype)?, self.device())?;
        CandleParam::from_tensor(t)
    }

    fn param_ones(&self, shape: &[usize], dtype: Dtype) -> Result<Self::Param, Self::Error> {
        let t = candle_core::Tensor::ones(shape, to_candle(dtype)?, self.device())?;
        CandleParam::from_tensor(t)
    }

    // ---- Autograd --------------------------------------------------------

    fn backward(&self, loss: &Self::Tensor) -> Result<Self::GradStore, Self::Error> {
        Ok(loss.inner().backward()?)
    }

    fn gradient(
        &self,
        store: &Self::GradStore,
        param: &Self::Param,
    ) -> Result<Option<Self::Tensor>, Self::Error> {
        match store.get(&param.var) {
            Some(t) => Ok(Some(CandleTensor::new(t.clone())?)),
            None => Ok(None),
        }
    }

    fn set_gradient(
        &self,
        store: &mut Self::GradStore,
        param: &Self::Param,
        grad: Self::Tensor,
    ) -> Result<(), Self::Error> {
        store.insert(&param.var, grad.into_inner());
        Ok(())
    }

    fn param_from_tensor(&self, t: &Self::Tensor) -> Result<Self::Param, Self::Error> {
        // Explicitly `.detach()` first so the resulting Var carries no
        // BackpropOp chain. This is the critical bit for activation
        // checkpointing: without it the boundary tensor would still
        // reference the block's forward intermediates via its op chain
        // and dropping `raw_output` on the caller side would free
        // *nothing*. With detach, the new tensor has BackpropOp::none()
        // and only shares storage (which we then need decoupled too
        // — see below).
        let detached = t.inner().detach();
        CandleParam::from_tensor(detached)
    }

    /// Plain `candle_core::Tensor::detach()` — shares storage (no copy)
    /// but drops the `BackpropOp` chain, same mechanism `param_from_tensor`
    /// uses above minus the `Var` allocation. Unlike `param_from_tensor`,
    /// this does not need a fresh `Var`/`ParamId`: nothing here is ever
    /// looked up by parameter identity, it exists purely to stop further
    /// `Ops` calls from retaining an `Arc` chain back to `t`'s ancestors
    /// (see `Ops::detach`'s doc comment for why that distinction matters
    /// on backends with a shared/global tape).
    fn detach(&self, t: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(t.inner().detach())
    }

    fn merge_grad_stores(
        &self,
        dst: &mut Self::GradStore,
        src: Self::GradStore,
    ) -> Result<(), Self::Error> {
        // candle_core::backprop::GradStore exposes `get_ids() -> Iterator<TensorId>`.
        // For each id in `src`, take the tensor and either add it to `dst[id]` or
        // insert it. We can't borrow-mutate `dst` while iterating `src`, so first
        // collect ids, then move tensors over.
        let ids: Vec<candle_core::TensorId> = src.get_ids().copied().collect();
        for id in ids {
            // `.detach()`, NOT `.clone()`. A gradient returned by Candle's
            // backward still carries its full op-chain back into the forward
            // tape (e.g. `grad(out_proj_w) = normedᵀ · grad_out` holds
            // `normed`, whose chain is the entire block — rmsnorm → gate →
            // the pure-Ops ssd_scan intermediates). If we store that chain,
            // `checkpoint_backward` (which merges one recomputed block's
            // grads per iteration) keeps ALL 30 blocks' forward tapes
            // co-resident, silently defeating activation checkpointing and
            // OOMing at B=1 T=512. Detaching keeps the gradient *value*
            // (shared storage, bit-identical) and drops only the op-chain,
            // so each block's tape is freed before the next is recomputed.
            // Safe because training is first-order: AdamW consumes gradients
            // as values and no second backward is taken through the store.
            let new_t = src.get_id(id).expect("just iterated, must exist").detach();
            match dst.get_id(id) {
                Some(old_t) => {
                    let summed = (old_t + &new_t)?.detach();
                    dst.insert_id(id, summed);
                }
                None => {
                    dst.insert_id(id, new_t);
                }
            }
        }
        Ok(())
    }

    fn sgd_step(
        &self,
        param: &Self::Param,
        grad: &Self::Tensor,
        lr: f32,
    ) -> Result<(), Self::Error> {
        // new = current - lr * grad
        let cur = param.as_tensor().inner();
        let updated = (cur - (grad.inner() * f64::from(lr))?)?;
        param.assign(&updated)?;
        Ok(())
    }

    fn assign(&self, param: &Self::Param, value: &Self::Tensor) -> Result<(), Self::Error> {
        param.assign(value.inner())
    }

    // ---- Loss / reduction helpers ---------------------------------------

    fn log_softmax(&self, x: &Self::Tensor, dim: usize) -> Result<Self::Tensor, Self::Error> {
        let y = candle_nn::ops::log_softmax(x.inner(), dim)?;
        CandleTensor::new(y)
    }

    fn gather(
        &self,
        x: &Self::Tensor,
        indices: &Self::Tensor,
        dim: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(x.inner().gather(indices.inner(), dim)?)
    }

    fn mean_all(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(x.inner().mean_all()?)
    }

    fn sum_all(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(x.inner().sum_all()?)
    }

    fn mul_scalar(&self, x: &Self::Tensor, scale: f32) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new((x.inner() * f64::from(scale))?)
    }

    fn sqrt(&self, x: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(x.inner().sqrt()?)
    }

    fn div(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new(a.inner().broadcast_div(b.inner())?)
    }

    fn add_scalar(&self, x: &Self::Tensor, scalar: f32) -> Result<Self::Tensor, Self::Error> {
        CandleTensor::new((x.inner() + f64::from(scalar))?)
    }

    // ---- Mamba2 SSD scan -------------------------------------------------

    /// Pure-Ops SSD scan from `pm_core::mamba2::ssd_scan_ops_default`.
    /// Stays on-device (no host roundtrip) so autograd flows through.
    ///
    /// `block_len` is currently ignored — the default impl is
    /// `O(B·H·T²)` but autograd-clean. A fused chunked kernel that
    /// honours `block_len` is tracked in PLAN.md Phase 2.
    fn ssd_scan(
        &self,
        x: &Self::Tensor,
        a: &Self::Tensor,
        b: &Self::Tensor,
        c: &Self::Tensor,
        block_len: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        if x.shape().len() != 4 {
            return Err(Error::NotImplemented(
                "ssd_scan: X must be rank-4 (B,T,H,P)",
            ));
        }
        pm_core::mamba2::ssd_scan_ops_default(self, x, a, b, c, block_len)
    }

    // ---- Fused cross-entropy over a tied embedding table ------------------

    /// Pure-Ops tiled fused cross-entropy from
    /// `pm_core::loss::fused_cross_entropy_tiled`. Candle's per-tensor
    /// `Op` DAG has no shared/global tape to pollute, but the reference
    /// tiling's `Ops::detach` calls are still required — without them the
    /// returned `loss`/`grad_table` would transitively `Arc`-hold every
    /// tile's `(rows, V)` intermediates (see that module's doc comment).
    fn fused_cross_entropy(
        &self,
        hidden: &Self::Tensor,
        table: &Self::Tensor,
        targets: &Self::Tensor,
        tile_rows: usize,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        pm_core::loss::fused_cross_entropy_tiled(self, hidden, table, targets, tile_rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pm_core::{Ops, Tensor};

    fn cpu() -> CandleBackend {
        CandleBackend::new_cpu()
    }

    #[test]
    fn add_broadcast() {
        let b = cpu();
        let a = b.from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let c = b.from_slice_f32(&[10.0, 20.0], &[2]).unwrap();
        let y = b.add(&a, &c).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        let v = b.to_vec_f32(&y).unwrap();
        assert_eq!(v, vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn matmul_basic() {
        let b = cpu();
        let a = b
            .from_slice_f32(&[1., 2., 3., 4., 5., 6.], &[2, 3])
            .unwrap();
        let c = b
            .from_slice_f32(&[1., 0., 0., 1., 1., 1.], &[3, 2])
            .unwrap();
        let y = b.matmul(&a, &c).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        // [[1+0+3, 0+2+3], [4+0+6, 0+5+6]] = [[4,5],[10,11]]
        assert_eq!(b.to_vec_f32(&y).unwrap(), vec![4., 5., 10., 11.]);
    }

    #[test]
    fn silu_matches_reference() {
        let b = cpu();
        let x = b.from_slice_f32(&[0.0, 1.0, -1.0, 2.0], &[4]).unwrap();
        let y = b.silu(&x).unwrap();
        let v = b.to_vec_f32(&y).unwrap();
        // silu(0)=0, silu(1)=1*sigmoid(1)=0.7310586, silu(-1)=-0.2689414, silu(2)=1.7615942
        let expected = [0.0_f32, 0.7310586, -0.2689414, 1.7615942];
        for (a, e) in v.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "got {a}, expected {e}");
        }
    }

    #[test]
    fn sigmoid_matches_reference() {
        let b = cpu();
        let x = b.from_slice_f32(&[0.0, 1.0, -1.0], &[3]).unwrap();
        let y = b.sigmoid(&x).unwrap();
        let v = b.to_vec_f32(&y).unwrap();
        let expected = [0.5_f32, 0.7310586, 0.2689414];
        for (a, e) in v.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5);
        }
    }

    #[test]
    fn softplus_matches_reference() {
        let b = cpu();
        let x = b.from_slice_f32(&[0.0, 1.0, -1.0, 10.0], &[4]).unwrap();
        let y = b.softplus(&x).unwrap();
        let v = b.to_vec_f32(&y).unwrap();
        // softplus(0)=ln(2), softplus(1)=1.31326, softplus(-1)=0.31326,
        // softplus(10) ≈ 10.000_046
        let expected = [
            core::f32::consts::LN_2,
            1.313_261_7,
            0.313_261_7,
            10.000_046,
        ];
        for (a, e) in v.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-4, "got {a}, expected {e}");
        }
    }

    #[test]
    fn rmsnorm_known_value() {
        let b = cpu();
        let x = b.from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let w = b.from_slice_f32(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let y = b.rmsnorm(&x, &w, 1e-6).unwrap();
        let v = b.to_vec_f32(&y).unwrap();
        // mean(x^2) = (1+4+9+16)/4 = 7.5, rms = sqrt(7.5) ≈ 2.7386
        let rms = 7.5f32.sqrt();
        let expected = [1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms];
        for (a, e) in v.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "got {a}, expected {e}");
        }
    }

    #[test]
    fn cumsum_axis0() {
        let b = cpu();
        let x = b.from_slice_f32(&[1., 2., 3., 4.], &[2, 2]).unwrap();
        let y = b.cumsum(&x, 0).unwrap();
        assert_eq!(b.to_vec_f32(&y).unwrap(), vec![1., 2., 4., 6.]);
    }

    #[test]
    fn conv1d_depthwise() {
        let b = cpu();
        // x: (B=1, C=2, T=4); identity kernel size 1 → output == input.
        let x = b
            .from_slice_f32(&[1., 2., 3., 4., 5., 6., 7., 8.], &[1, 2, 4])
            .unwrap();
        // weight: (C_out=2, C_in/groups=1, K=1) all ones (depthwise identity).
        let w = b.from_slice_f32(&[1., 1.], &[2, 1, 1]).unwrap();
        let y = b.conv1d(&x, &w, None, 1, 0, 2).unwrap();
        assert_eq!(y.shape(), &[1, 2, 4]);
        assert_eq!(
            b.to_vec_f32(&y).unwrap(),
            vec![1., 2., 3., 4., 5., 6., 7., 8.]
        );
    }

    #[test]
    fn ssd_scan_zero_a_returns_running_sum() {
        // Same sanity check as pm-core scalar test, but exercised through
        // the Ops trait so we catch host/device round-trip bugs.
        let bk = cpu();
        let (batch, t, h, p, n) = (1, 4, 1, 2, 2);
        let x = bk.ones(&[batch, t, h, p], Dtype::F32).unwrap();
        let a = bk.zeros(&[batch, t, h], Dtype::F32).unwrap();
        let bb_data: Vec<f32> = (0..batch * t * h * n).map(|i| i as f32 * 0.1).collect();
        let cc_data: Vec<f32> = (0..batch * t * h * n).map(|i| i as f32 * 0.1).collect();
        let bb = bk.from_slice_f32(&bb_data, &[batch, t, h, n]).unwrap();
        let cc = bk.from_slice_f32(&cc_data, &[batch, t, h, n]).unwrap();
        let y = bk.ssd_scan(&x, &a, &bb, &cc, 4).unwrap();
        assert_eq!(y.shape(), &[batch, t, h, p]);
        let v = bk.to_vec_f32(&y).unwrap();
        // With x=1, a=0, p=2: both p slots get the same value.
        assert!((v[0] - v[1]).abs() < 1e-6);
        // y at t=3 must be greater than y at t=0 (strictly increasing).
        assert!(v[6] > v[0]);
    }

    /// Autograd verification: gradients of `sum(ssd_scan(x,a,b,c))`
    /// must reach every input. Until pure-Ops `ssd_scan` landed this
    /// failed because the old impl did a host roundtrip and detached
    /// from the autograd graph. Keeping this test guards against
    /// regressions when backend ops are rewritten.
    #[test]
    fn ssd_scan_backward_reaches_all_inputs() {
        let bk = cpu();
        let device = bk.device();
        let (batch, t, n_heads, p_dim, n_dim) = (1, 4, 2, 2, 2);

        // Build inputs as Vars so backward() can see them.
        let x_data: Vec<f32> = (0..batch * t * n_heads * p_dim)
            .map(|i| i as f32 * 0.1 + 1.0)
            .collect();
        let a_data: Vec<f32> = (0..batch * t * n_heads)
            .map(|i| -0.1 - 0.01 * i as f32)
            .collect();
        let b_data: Vec<f32> = (0..batch * t * n_heads * n_dim)
            .map(|i| 0.05 + 0.01 * i as f32)
            .collect();
        let c_data: Vec<f32> = (0..batch * t * n_heads * n_dim)
            .map(|i| 0.04 + 0.01 * i as f32)
            .collect();

        let x_t =
            candle_core::Tensor::from_slice(&x_data, &[batch, t, n_heads, p_dim], device).unwrap();
        let a_t = candle_core::Tensor::from_slice(&a_data, &[batch, t, n_heads], device).unwrap();
        let b_t =
            candle_core::Tensor::from_slice(&b_data, &[batch, t, n_heads, n_dim], device).unwrap();
        let c_t =
            candle_core::Tensor::from_slice(&c_data, &[batch, t, n_heads, n_dim], device).unwrap();

        let x_var = candle_core::Var::from_tensor(&x_t).unwrap();
        let a_var = candle_core::Var::from_tensor(&a_t).unwrap();
        let b_var = candle_core::Var::from_tensor(&b_t).unwrap();
        let c_var = candle_core::Var::from_tensor(&c_t).unwrap();

        let x = CandleTensor::new(x_var.as_tensor().clone()).unwrap();
        let a = CandleTensor::new(a_var.as_tensor().clone()).unwrap();
        let b = CandleTensor::new(b_var.as_tensor().clone()).unwrap();
        let c = CandleTensor::new(c_var.as_tensor().clone()).unwrap();

        let y = bk.ssd_scan(&x, &a, &b, &c, 4).unwrap();
        // Reduce to a scalar so .backward() has a definite root.
        let loss = y.inner().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        for (name, var) in [("x", &x_var), ("a", &a_var), ("b", &b_var), ("c", &c_var)] {
            let g = grads.get(var).unwrap_or_else(|| {
                panic!("ssd_scan backward did not propagate gradient to `{name}`")
            });
            // Numerically verify the gradient is non-trivial.
            let g_host: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
            let max_abs = g_host.iter().map(|v| v.abs()).fold(0f32, f32::max);
            assert!(
                max_abs > 1e-6,
                "gradient w.r.t. `{name}` is essentially zero (max_abs={max_abs}); \
                 ssd_scan probably detached the graph again"
            );
            assert!(
                g_host.iter().all(|v| v.is_finite()),
                "gradient w.r.t. `{name}` has non-finite entries"
            );
        }
    }
}
