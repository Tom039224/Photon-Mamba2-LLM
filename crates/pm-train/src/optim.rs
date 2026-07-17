//! Optimisers (F.1).
//!
//! Two implementations:
//! - [`Sgd`] — plain gradient descent. Used in tests to keep the F.4 toy
//!   training fast and easy to inspect.
//! - [`AdamW`] — Loshchilov & Hutter's decoupled-weight-decay Adam.
//!
//! Both are generic over the backend (`O: Ops`) and operate on a flat
//! slice of `O::Param`s. The Trainer pulls those via
//! `pm_core::Parameterized::collect_params`.

use pm_core::{Dtype, Ops, Param, Tensor};

/// Optimiser interface for an already-computed `GradStore`.
pub trait Optimizer<O: Ops> {
    /// Apply one update for the parameters that have a gradient in `grads`.
    fn step(&mut self, ops: &O, params: &[&O::Param], grads: &O::GradStore)
        -> Result<(), O::Error>;
}

// ---------- Plain SGD ----------

pub struct Sgd {
    pub lr: f32,
}

impl Sgd {
    #[must_use]
    pub const fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl<O: Ops> Optimizer<O> for Sgd {
    fn step(
        &mut self,
        ops: &O,
        params: &[&O::Param],
        grads: &O::GradStore,
    ) -> Result<(), O::Error> {
        for p in params {
            if let Some(grad) = ops.gradient(grads, p)? {
                ops.sgd_step(p, &grad, self.lr)?;
            }
        }
        Ok(())
    }
}

// ---------- AdamW (decoupled weight decay) ----------

#[derive(Debug, Clone, Copy)]
pub struct AdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

pub struct AdamW<O: Ops> {
    pub cfg: AdamWConfig,
    step: u64,
    /// First-moment estimates (`m`), one per parameter, same shape.
    m: Vec<O::Tensor>,
    /// Second-moment estimates (`v`).
    v: Vec<O::Tensor>,
}

impl<O: Ops> AdamW<O> {
    /// Allocate moment buffers (initialised to zero) for every parameter.
    /// Call once at training-init with the result of
    /// `Parameterized::collect_params`.
    pub fn new(ops: &O, params: &[&O::Param], cfg: AdamWConfig) -> Result<Self, O::Error> {
        let mut m = Vec::with_capacity(params.len());
        let mut v = Vec::with_capacity(params.len());
        for p in params {
            let shape = p.as_tensor().shape().to_vec();
            m.push(ops.zeros(&shape, Dtype::F32)?);
            v.push(ops.zeros(&shape, Dtype::F32)?);
        }
        Ok(Self { cfg, step: 0, m, v })
    }

    /// Overwrite the LR (cosine schedule helper).
    pub fn set_lr(&mut self, lr: f32) {
        self.cfg.lr = lr;
    }
}

impl<O: Ops> Optimizer<O> for AdamW<O> {
    fn step(
        &mut self,
        ops: &O,
        params: &[&O::Param],
        grads: &O::GradStore,
    ) -> Result<(), O::Error> {
        self.step += 1;
        let t = self.step as i32;
        let bias_correction1 = 1.0 - self.cfg.beta1.powi(t);
        let bias_correction2 = 1.0 - self.cfg.beta2.powi(t);

        for (i, p) in params.iter().enumerate() {
            let Some(raw_grad) = ops.gradient(grads, p)? else {
                continue;
            };

            // AdamW: decoupled weight decay applied directly to the
            // parameter (NOT folded into the gradient).
            // param ← param * (1 - lr * wd)
            if self.cfg.weight_decay > 0.0 {
                let scaled =
                    ops.mul_scalar(p.as_tensor(), 1.0 - self.cfg.lr * self.cfg.weight_decay)?;
                ops.assign(p, &scaled)?;
            }

            // m ← β1·m + (1-β1)·g
            let new_m = {
                let m_term1 = ops.mul_scalar(&self.m[i], self.cfg.beta1)?;
                let m_term2 = ops.mul_scalar(&raw_grad, 1.0 - self.cfg.beta1)?;
                ops.add(&m_term1, &m_term2)?
            };
            // v ← β2·v + (1-β2)·g²
            let new_v = {
                let g_sq = ops.mul(&raw_grad, &raw_grad)?;
                let v_term1 = ops.mul_scalar(&self.v[i], self.cfg.beta2)?;
                let v_term2 = ops.mul_scalar(&g_sq, 1.0 - self.cfg.beta2)?;
                ops.add(&v_term1, &v_term2)?
            };
            self.m[i] = new_m;
            self.v[i] = new_v;

            // m̂ = m / bc1; v̂ = v / bc2
            // update = m̂ / (sqrt(v̂) + ε); param ← param - lr · update
            let m_hat = ops.mul_scalar(&self.m[i], 1.0 / bias_correction1)?;
            let v_hat = ops.mul_scalar(&self.v[i], 1.0 / bias_correction2)?;
            let sqrt_v_hat = ops.sqrt(&v_hat)?;
            let denom = ops.add_scalar(&sqrt_v_hat, self.cfg.eps)?;
            let update = ops.div(&m_hat, &denom)?;
            let scaled_update = ops.mul_scalar(&update, self.cfg.lr)?;
            let new_param = ops.sub(p.as_tensor(), &scaled_update)?;
            ops.assign(p, &new_param)?;
        }
        Ok(())
    }
}
