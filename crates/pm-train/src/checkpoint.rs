//! Safetensors-based checkpoint save/load (F.8).
//!
//! Parameters are serialised by position into the safetensors file:
//! `"p_0000"`, `"p_0001"`, ... The load path expects the same number of
//! params in the same order, with matching shapes. This is intentional:
//! the model definition is the source of truth for the parameter
//! ordering (the same `collect_params` walk runs on both ends), so the
//! file format stays simple and we don't need to thread human-readable
//! names through every module.
//!
//! Format note: every tensor is read back through `Ops::to_vec_f32`
//! (f32 host buffer) and re-uploaded via `Ops::from_slice_f32` +
//! `Ops::assign`. Phase-2 work could keep tensors on-device using a
//! backend-specific path; for Phase 1 the host roundtrip is fine.

use std::collections::HashMap;
use std::path::Path;

use pm_core::{Ops, Param, Tensor};
use safetensors::{tensor::TensorView, Dtype, SafeTensors};

use crate::TrainError;

/// Save every parameter to a `.safetensors` file at `path`.
pub fn save<O: Ops, P: AsRef<Path>>(
    ops: &O,
    params: &[&O::Param],
    path: P,
) -> Result<(), TrainError<O::Error>> {
    // Hold owned bytes so `TensorView::new` can borrow them.
    let mut buffers: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(params.len());
    for (i, p) in params.iter().enumerate() {
        let name = format!("p_{i:04}");
        let shape = p.as_tensor().shape().to_vec();
        let data_f32 = ops.to_vec_f32(p.as_tensor()).map_err(TrainError::Backend)?;
        let bytes: Vec<u8> = bytemuck::cast_slice(&data_f32).to_vec();
        buffers.push((name, shape, bytes));
    }

    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::with_capacity(buffers.len());
    for (name, shape, bytes) in &buffers {
        let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
            .map_err(|e| TrainError::Safetensors(format!("create view {name}: {e}")))?;
        tensors.insert(name.clone(), view);
    }

    safetensors::serialize_to_file(&tensors, &None, path.as_ref())
        .map_err(|e| TrainError::Safetensors(format!("serialize: {e}")))?;
    Ok(())
}

/// Load every parameter from a `.safetensors` file. `params.len()` and
/// the per-param shapes must match what was saved.
pub fn load<O: Ops, P: AsRef<Path>>(
    ops: &O,
    params: &[&O::Param],
    path: P,
) -> Result<(), TrainError<O::Error>> {
    let buffer = std::fs::read(path.as_ref())
        .map_err(|e| TrainError::Safetensors(format!("read {}: {e}", path.as_ref().display())))?;
    let st = SafeTensors::deserialize(&buffer)
        .map_err(|e| TrainError::Safetensors(format!("deserialize: {e}")))?;

    let n_loaded = st.tensors().len();
    if n_loaded != params.len() {
        return Err(TrainError::Safetensors(format!(
            "checkpoint has {n_loaded} tensors, model has {} params",
            params.len()
        )));
    }

    for (i, p) in params.iter().enumerate() {
        let name = format!("p_{i:04}");
        let view = st
            .tensor(&name)
            .map_err(|e| TrainError::Safetensors(format!("tensor {name}: {e}")))?;
        if view.dtype() != Dtype::F32 {
            return Err(TrainError::Safetensors(format!(
                "param {name}: expected F32, got {:?}",
                view.dtype()
            )));
        }
        let expected_shape: Vec<usize> = p.as_tensor().shape().to_vec();
        let got_shape: Vec<usize> = view.shape().to_vec();
        if expected_shape != got_shape {
            return Err(TrainError::Safetensors(format!(
                "param {name}: shape mismatch — file {got_shape:?}, model {expected_shape:?}"
            )));
        }
        let data_f32: &[f32] = bytemuck::cast_slice(view.data());
        let new_t = ops
            .from_slice_f32(data_f32, &got_shape)
            .map_err(TrainError::Backend)?;
        ops.assign(p, &new_t).map_err(TrainError::Backend)?;
    }
    Ok(())
}
