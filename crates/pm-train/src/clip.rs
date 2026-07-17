//! Global L2-norm gradient clipping (F.5).
//!
//! Computes the global gradient norm `‖g‖₂ = sqrt(Σᵢ ‖gᵢ‖₂²)` across
//! every parameter that has a gradient, then if `‖g‖₂ > max_norm`,
//! scales each gradient by `max_norm / ‖g‖₂` in place inside the
//! `GradStore`. Subsequent optimiser steps therefore see the clipped
//! values transparently.
//!
//! Returns the unclipped global norm — useful for logging.

use pm_core::Ops;

pub fn clip_grad_norm<O: Ops>(
    ops: &O,
    params: &[&O::Param],
    grads: &mut O::GradStore,
    max_norm: f32,
) -> Result<f32, O::Error> {
    // Pass 1: accumulate per-param sum(g²) **on the device** into a
    // single scalar tensor, then sync exactly once. The earlier version
    // synced 246× per step (one host transfer per param) which left the
    // GPU idle waiting for each copy and was a major contributor to the
    // bursty utilisation pattern.
    let mut device_acc: Option<O::Tensor> = None;
    for p in params {
        if let Some(g) = ops.gradient(grads, p)? {
            let sq = ops.mul(&g, &g)?;
            let s = ops.sum_all(&sq)?; // scalar tensor, still on device
            device_acc = Some(match device_acc {
                Some(a) => ops.add(&a, &s)?,
                None => s,
            });
        }
    }
    let total_sq = match device_acc {
        Some(acc) => f64::from(scalar_value(ops, &acc)?),
        None => 0.0,
    };
    let global_norm = total_sq.sqrt() as f32;

    if global_norm <= max_norm || global_norm == 0.0 {
        return Ok(global_norm);
    }
    let scale = max_norm / global_norm;

    // Pass 2: rewrite each gradient in place.
    for p in params {
        if let Some(g) = ops.gradient(grads, p)? {
            let scaled = ops.mul_scalar(&g, scale)?;
            ops.set_gradient(grads, p, scaled)?;
        }
    }
    Ok(global_norm)
}

fn scalar_value<O: Ops>(ops: &O, t: &O::Tensor) -> Result<f32, O::Error> {
    let v = ops.to_vec_f32(t)?;
    assert!(!v.is_empty(), "scalar_value: empty tensor");
    Ok(v[0])
}
