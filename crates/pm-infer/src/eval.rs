//! Continuation-scoring helpers used by both `generate` and the
//! HellaSwag evaluator (Group H).
//!
//! HellaSwag scores 4 endings per question by computing the
//! length-normalised log-probability of each ending conditioned on the
//! context, then picks argmax. We expose:
//!
//! - [`score_continuation`] — given a model + (ctx_ids, cont_ids), return
//!   the average log-probability of the continuation tokens.
//! - [`HellaSwagItem`] / [`HellaSwagResult`] — minimal data types so the
//!   CLI can compute accuracy without re-deriving them.

use pm_core::{Ops, PhotonMamba, Tensor};

#[derive(Debug, Clone)]
pub struct HellaSwagItem {
    pub context: Vec<i64>,
    /// Four candidate continuations, each pre-tokenised.
    pub endings: [Vec<i64>; 4],
    pub label: usize, // 0..3
}

#[derive(Debug, Clone, Copy)]
pub struct HellaSwagResult {
    /// Index of the highest-scoring ending (0..3).
    pub predicted: usize,
    /// Per-ending mean log-probability.
    pub scores: [f32; 4],
}

/// Length-normalised log-probability of `continuation` given `context`.
///
/// Returns `(sum_logp, mean_logp)`. We expose both so callers can mix
/// length-normalised and raw scoring (HellaSwag uses mean; other
/// benchmarks use sum).
pub fn score_continuation<O: Ops>(
    ops: &O,
    model: &PhotonMamba<O>,
    chunk_product: usize,
    pad_token_id: i64,
    context: &[i64],
    continuation: &[i64],
) -> Result<(f32, f32), O::Error> {
    assert!(!continuation.is_empty(), "continuation must not be empty");
    let real_len = context.len() + continuation.len();
    let padded_len = pad_up(real_len, chunk_product);

    let mut buf = vec![pad_token_id; padded_len];
    buf[..context.len()].copy_from_slice(context);
    buf[context.len()..real_len].copy_from_slice(continuation);

    let ids = ops.from_slice_i64(&buf, &[1, padded_len])?;
    let out = model.forward(ops, &ids)?;
    let v = out.logits.shape()[2];

    // logits[t] predicts ids[t+1]. To score continuation token at
    // position `context.len() + k`, we use logits at position
    // `context.len() + k - 1`.
    let start = context.len() - 1; // position predicting first continuation token
    let mut sum = 0.0f32;
    for (k, &tok) in continuation.iter().enumerate() {
        let pos = start + k;
        let row = ops.narrow(&out.logits, 1, pos, 1)?; // (1, 1, V)
        let logp = ops.log_softmax(&row, 2)?;
        let host: Vec<f32> = ops.to_vec_f32(&logp)?;
        assert_eq!(host.len(), v);
        let tgt = tok as usize;
        sum += host[tgt];
    }
    let mean = sum / continuation.len() as f32;
    Ok((sum, mean))
}

/// Score every ending of one HellaSwag item; pick the one with the
/// highest length-normalised log-probability.
pub fn score_hellaswag<O: Ops>(
    ops: &O,
    model: &PhotonMamba<O>,
    chunk_product: usize,
    pad_token_id: i64,
    item: &HellaSwagItem,
) -> Result<HellaSwagResult, O::Error> {
    let mut scores = [0.0f32; 4];
    for (i, ending) in item.endings.iter().enumerate() {
        let (_, mean) = score_continuation(
            ops,
            model,
            chunk_product,
            pad_token_id,
            &item.context,
            ending,
        )?;
        scores[i] = mean;
    }
    let mut predicted = 0;
    let mut best = scores[0];
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > best {
            best = s;
            predicted = i;
        }
    }
    Ok(HellaSwagResult { predicted, scores })
}

fn pad_up(n: usize, multiple: usize) -> usize {
    if n.is_multiple_of(multiple) {
        n.max(multiple)
    } else {
        ((n / multiple) + 1) * multiple
    }
}
