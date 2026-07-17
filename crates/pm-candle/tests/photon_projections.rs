//! Tests for ContextChunker (D.2) and ContextConverter (D.4).

use pm_candle::CandleBackend;
use pm_core::nn::Linear;
use pm_core::photon::{ContextChunker, ContextConverter};
use pm_core::{Module, Ops, Tensor};

#[test]
fn chunker_output_shape_matches_paper() {
    let bk = CandleBackend::new_cpu();
    let (b, t, d_in) = (2, 16, 8);
    let d_out = 12;
    let chunk_size = 4;
    let chunker: ContextChunker<_> =
        ContextChunker::from_constants(&bk, d_in, d_out, chunk_size, 0.1).unwrap();
    let x_data: Vec<f32> = (0..b * t * d_in).map(|i| i as f32 * 0.01).collect();
    let x = bk.from_slice_f32(&x_data, &[b, t, d_in]).unwrap();
    let y = chunker.forward(&bk, &x).unwrap();
    assert_eq!(y.shape(), &[b, t / chunk_size, d_out]);
    let v = bk.to_vec_f32(&y).unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn chunker_chunk1_is_pure_linear() {
    // chunk_size=1 reduces to a plain Linear projection per position.
    let bk = CandleBackend::new_cpu();
    let (b, t, d_in, d_out) = (1, 3, 2, 4);
    let chunker = ContextChunker::from_constants(&bk, d_in, d_out, 1, 1.0).unwrap();
    let x = bk
        .from_slice_f32(&[1., 2., 3., 4., 5., 6.], &[b, t, d_in])
        .unwrap();
    let y = chunker.forward(&bk, &x).unwrap();
    assert_eq!(y.shape(), &[b, t, d_out]);

    // weight is all 1.0, bias 0.0; so y[..., j] = sum(x[..., :]) per position.
    // pos 0: x=[1,2], sum=3; pos 1: x=[3,4], sum=7; pos 2: x=[5,6], sum=11.
    let v = bk.to_vec_f32(&y).unwrap();
    for j in 0..d_out {
        assert!((v[j] - 3.0).abs() < 1e-6);
        assert!((v[d_out + j] - 7.0).abs() < 1e-6);
        assert!((v[2 * d_out + j] - 11.0).abs() < 1e-6);
    }
}

#[test]
fn chunker_concatenates_chunk_in_order() {
    // chunk_size=2, d_in=1, d_out=2; weight is identity-on-first-then-zero
    // so we can directly observe the concatenation order.
    let bk = CandleBackend::new_cpu();
    let (b, t, d_in, d_out, c) = (1, 4, 1, 2, 2);
    // weight (c*d_in=2, d_out=2):
    // [[1, 0],   <- picks first-in-chunk into y[0]
    //  [0, 1]]   <- picks second-in-chunk into y[1]
    let weight = bk
        .param_from_slice_f32(&[1., 0., 0., 1.], &[c * d_in, d_out])
        .unwrap();
    let bias = bk.param_from_slice_f32(&[0., 0.], &[d_out]).unwrap();
    let proj = Linear::from_params(c * d_in, d_out, weight, Some(bias));
    let chunker = ContextChunker::from_linear(d_in, d_out, c, proj);

    // x = [10, 20, 30, 40] reshaped as (1, 4, 1)
    let x = bk
        .from_slice_f32(&[10., 20., 30., 40.], &[b, t, d_in])
        .unwrap();
    let y = chunker.forward(&bk, &x).unwrap();
    assert_eq!(y.shape(), &[b, t / c, d_out]);
    let v = bk.to_vec_f32(&y).unwrap();
    // Chunk 0 = [10, 20] → y[0] = [10, 20]; chunk 1 = [30, 40] → y[1] = [30, 40].
    assert_eq!(v, vec![10., 20., 30., 40.]);
}

#[test]
fn converter_output_shape_matches_paper() {
    let bk = CandleBackend::new_cpu();
    let (b, s, d_in) = (2, 4, 12);
    let d_out = 8;
    let r = 4;
    let conv: ContextConverter<_> =
        ContextConverter::from_constants(&bk, d_in, d_out, r, 0.1).unwrap();
    let x_data: Vec<f32> = (0..b * s * d_in).map(|i| i as f32 * 0.01).collect();
    let x = bk.from_slice_f32(&x_data, &[b, s, d_in]).unwrap();
    let y = conv.forward(&bk, &x).unwrap();
    assert_eq!(y.shape(), &[b, s * r, d_out]);
    use pm_core::Param;
    assert_eq!(conv.starting_latent.as_tensor().shape(), &[d_out]);
    let v = bk.to_vec_f32(&y).unwrap();
    assert!(v.iter().all(|x| x.is_finite()));
}

#[test]
fn converter_r1_is_pure_linear_identity_when_d_out_equals_d_in() {
    // R=1 with identity weight should reproduce the input.
    let bk = CandleBackend::new_cpu();
    let (b, s, d) = (1, 3, 2);
    let mut w = vec![0f32; d * d];
    for i in 0..d {
        w[i * d + i] = 1.0;
    }
    let weight = bk.param_from_slice_f32(&w, &[d, d]).unwrap();
    let bias = bk.param_from_slice_f32(&vec![0.0; d], &[d]).unwrap();
    let proj = Linear::from_params(d, d, weight, Some(bias));
    let starting = bk.param_from_slice_f32(&vec![0.0; d], &[d]).unwrap();
    let conv = ContextConverter::from_parts(d, d, 1, proj, starting);

    let x = bk
        .from_slice_f32(&[1., 2., 3., 4., 5., 6.], &[b, s, d])
        .unwrap();
    let y = conv.forward(&bk, &x).unwrap();
    assert_eq!(y.shape(), &[b, s, d]);
    assert_eq!(bk.to_vec_f32(&y).unwrap(), vec![1., 2., 3., 4., 5., 6.]);
}

#[test]
fn chunker_then_converter_recovers_shape_for_r_equals_c() {
    // With R = C, chunker → converter is shape-invariant. (Values won't
    // match because the projections aren't pseudo-inverses, but the round
    // trip is the dataflow PHOTON relies on for the recursive consistency
    // loss; verifying shape here keeps us honest about the contract.)
    let bk = CandleBackend::new_cpu();
    let (b, t, d) = (1, 16, 8);
    let c = 4;
    let chunker = ContextChunker::from_constants(&bk, d, d, c, 0.1).unwrap();
    let converter = ContextConverter::from_constants(&bk, d, d, c, 0.1).unwrap();
    let x = bk
        .from_slice_f32(&vec![0.5f32; b * t * d], &[b, t, d])
        .unwrap();
    let mid = chunker.forward(&bk, &x).unwrap();
    assert_eq!(mid.shape(), &[b, t / c, d]);
    let out = converter.forward(&bk, &mid).unwrap();
    assert_eq!(out.shape(), x.shape());
}
