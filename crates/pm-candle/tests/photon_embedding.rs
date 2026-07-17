//! Tests for `pm-core::photon::{TokenEmbedding, RotaryEmbedding}`.

use pm_candle::CandleBackend;
use pm_core::photon::{RotaryEmbedding, TokenEmbedding};
use pm_core::{Module, Ops, Tensor};

#[test]
fn token_embedding_forward_then_lm_head_roundtrip() {
    let bk = CandleBackend::new_cpu();
    let vocab = 8;
    let d_model = 4;

    // Build a tied embedding with a non-trivial pattern: weight[r,c] = r + 0.1*c.
    let mut data = vec![0f32; vocab * d_model];
    for r in 0..vocab {
        for c in 0..d_model {
            data[r * d_model + c] = r as f32 + 0.1 * c as f32;
        }
    }
    let weight = bk.param_from_slice_f32(&data, &[vocab, d_model]).unwrap();
    let emb = TokenEmbedding::from_param(vocab, d_model, weight);

    // Lookup ids (1, 3) = [3, 5, 0]
    let ids = bk.from_slice_i64(&[3, 5, 0], &[1, 3]).unwrap();
    let hidden = emb.forward(&bk, &ids).unwrap();
    assert_eq!(hidden.shape(), &[1, 3, d_model]);
    let h = bk.to_vec_f32(&hidden).unwrap();
    // Row 3: [3.0, 3.1, 3.2, 3.3]
    assert!((h[0] - 3.0).abs() < 1e-6);
    assert!((h[1] - 3.1).abs() < 1e-6);
    // Row 5: [5.0, 5.1, 5.2, 5.3]
    assert!((h[4] - 5.0).abs() < 1e-6);
    // Row 0: zeros
    assert!(h[8].abs() < 1e-6);

    // LM head gives shape (1, 3, vocab).
    let logits = emb.lm_head_logits(&bk, &hidden).unwrap();
    assert_eq!(logits.shape(), &[1, 3, vocab]);

    // logits[0, 0, k] = <weight[3], weight[k]>; pick a known value.
    // weight[3] = [3, 3.1, 3.2, 3.3]; weight[3] dot weight[3]
    // = 3^2 + 3.1^2 + 3.2^2 + 3.3^2 = 9 + 9.61 + 10.24 + 10.89 = 39.74
    let l = bk.to_vec_f32(&logits).unwrap();
    let dot = 9.0 + 9.61 + 10.24 + 10.89;
    // logits[batch=0, pos=0, vocab=3]
    let got = l[3];
    assert!((got - dot).abs() < 1e-3, "got {got}, expected {dot}");
}

#[test]
fn rope_preserves_per_pair_norm() {
    // Rotation in the (x[i], x[i+d/2]) plane is an isometry: applying RoPE
    // must keep `x[i]^2 + x[i+d/2]^2` invariant per pair, per position.
    let bk = CandleBackend::new_cpu();
    let d_head = 8;
    let (b, t, h) = (2, 5, 3);
    let rope = RotaryEmbedding::new(&bk, d_head, 16, pm_core::photon::ROPE_DEFAULT_BASE).unwrap();

    let x_data: Vec<f32> = (0..b * t * h * d_head)
        .map(|i| (i as f32 * 0.31).sin())
        .collect();
    let x = bk.from_slice_f32(&x_data, &[b, t, h, d_head]).unwrap();
    let y = rope.apply(&bk, &x).unwrap();
    assert_eq!(y.shape(), &[b, t, h, d_head]);

    let half = d_head / 2;
    let yv = bk.to_vec_f32(&y).unwrap();
    let xv = bk.to_vec_f32(&x).unwrap();
    // Per (b,t,h), per i ∈ [0, half): x1[i]^2 + x2[i]^2 == y1[i]^2 + y2[i]^2.
    for bi in 0..b {
        for ti in 0..t {
            for hi in 0..h {
                for i in 0..half {
                    let base = ((bi * t + ti) * h + hi) * d_head;
                    let x1 = xv[base + i];
                    let x2 = xv[base + half + i];
                    let y1 = yv[base + i];
                    let y2 = yv[base + half + i];
                    let nx = x1 * x1 + x2 * x2;
                    let ny = y1 * y1 + y2 * y2;
                    assert!(
                        (nx - ny).abs() < 1e-5,
                        "norm mismatch at b={bi} t={ti} h={hi} i={i}: {nx} vs {ny}"
                    );
                }
            }
        }
    }
}

#[test]
fn rope_at_t0_is_identity() {
    // cos(0) = 1, sin(0) = 0, so position 0 should leave x unchanged.
    let bk = CandleBackend::new_cpu();
    let d_head = 8;
    let rope = RotaryEmbedding::new(&bk, d_head, 4, pm_core::photon::ROPE_DEFAULT_BASE).unwrap();
    let x_data: Vec<f32> = (0..d_head).map(|i| (i as f32 + 1.0) * 0.7).collect();
    // Shape (1, 1, 1, d_head): only position 0.
    let x = bk.from_slice_f32(&x_data, &[1, 1, 1, d_head]).unwrap();
    let y = rope.apply(&bk, &x).unwrap();
    let yv = bk.to_vec_f32(&y).unwrap();
    for (got, want) in yv.iter().zip(x_data.iter()) {
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }
}

#[test]
fn rope_pair_rotation_matches_explicit_formula() {
    // Numerical check at a single (t, i): the standard RoPE formula
    // out1 = x1*cos - x2*sin
    // out2 = x1*sin + x2*cos
    let bk = CandleBackend::new_cpu();
    let d_head = 4;
    let max_t = 8;
    let base = 10_000.0_f32;
    let rope = RotaryEmbedding::new(&bk, d_head, max_t, base).unwrap();

    // x[0,3,0,:] = [1.0, 2.0, 3.0, 4.0]; x1 = [1,2], x2 = [3,4].
    let mut x_data = vec![0f32; max_t * d_head];
    let t_query = 3usize;
    let base_idx = t_query * d_head;
    x_data[base_idx] = 1.0;
    x_data[base_idx + 1] = 2.0;
    x_data[base_idx + 2] = 3.0;
    x_data[base_idx + 3] = 4.0;
    let x = bk.from_slice_f32(&x_data, &[1, max_t, 1, d_head]).unwrap();
    let y = rope.apply(&bk, &x).unwrap();
    let yv = bk.to_vec_f32(&y).unwrap();

    let half = d_head / 2;
    for i in 0..half {
        let freq = (-(2.0 * i as f32 / d_head as f32) * base.ln()).exp();
        let angle = t_query as f32 * freq;
        let cos = angle.cos();
        let sin = angle.sin();
        let x1 = (i + 1) as f32; // 1, 2
        let x2 = (half + i + 1) as f32; // 3, 4
        let expect1 = x1 * cos - x2 * sin;
        let expect2 = x1 * sin + x2 * cos;
        let got1 = yv[t_query * d_head + i];
        let got2 = yv[t_query * d_head + half + i];
        assert!(
            (got1 - expect1).abs() < 1e-5,
            "pair {i} half1: got {got1}, expected {expect1}"
        );
        assert!(
            (got2 - expect2).abs() < 1e-5,
            "pair {i} half2: got {got2}, expected {expect2}"
        );
    }
}
