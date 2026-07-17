//! Sanity tests for `pm-core::nn::{Linear, Embedding}` driven through
//! the Candle backend.

use pm_candle::CandleBackend;
use pm_core::nn::{Embedding, Linear};
use pm_core::{Module, Ops, Tensor};

#[test]
fn linear_forward_with_bias_matches_handcalc() {
    let bk = CandleBackend::new_cpu();
    let in_f = 3;
    let out_f = 2;

    // weight (3,2) = [[1,2],[3,4],[5,6]]
    let weight = bk
        .param_from_slice_f32(&[1., 2., 3., 4., 5., 6.], &[in_f, out_f])
        .unwrap();
    // bias (2,) = [10, 20]
    let bias = bk.param_from_slice_f32(&[10., 20.], &[out_f]).unwrap();
    let layer: Linear<CandleBackend> = Linear::from_params(in_f, out_f, weight, Some(bias));

    // x (1,3) = [[1,1,1]]; expected y = [1+3+5+10, 2+4+6+20] = [19, 32]
    let x = bk.from_slice_f32(&[1., 1., 1.], &[1, in_f]).unwrap();
    let y = layer.forward(&bk, &x).unwrap();
    assert_eq!(y.shape(), &[1, out_f]);
    assert_eq!(bk.to_vec_f32(&y).unwrap(), vec![19., 32.]);
}

#[test]
fn linear_without_bias_is_pure_matmul() {
    let bk = CandleBackend::new_cpu();
    let layer = Linear::from_constants(&bk, 4, 2, false, 0.5).unwrap();
    let x = bk.from_slice_f32(&[1., 1., 1., 1.], &[1, 4]).unwrap();
    let y = layer.forward(&bk, &x).unwrap();
    // y = sum(x) * 0.5 broadcast over 2 outputs = [2.0, 2.0]
    assert_eq!(bk.to_vec_f32(&y).unwrap(), vec![2.0, 2.0]);
}

#[test]
fn embedding_lookup_returns_correct_rows() {
    let bk = CandleBackend::new_cpu();
    let vocab = 5;
    let dim = 3;
    // Row r filled with r (so row 2 = [2,2,2], row 4 = [4,4,4]).
    let emb = Embedding::arange_rows(&bk, vocab, dim, 1.0).unwrap();

    // ids (1,4): [0, 2, 4, 1]
    let ids = bk.from_slice_i64(&[0, 2, 4, 1], &[1, 4]).unwrap();
    let y = emb.forward(&bk, &ids).unwrap();
    assert_eq!(y.shape(), &[1, 4, dim]);
    let v = bk.to_vec_f32(&y).unwrap();
    let expected = vec![
        0.0, 0.0, 0.0, // ids[0]=0
        2.0, 2.0, 2.0, // ids[1]=2
        4.0, 4.0, 4.0, // ids[2]=4
        1.0, 1.0, 1.0, // ids[3]=1
    ];
    assert_eq!(v, expected);
}

#[test]
fn concat_along_last_dim_stitches() {
    let bk = CandleBackend::new_cpu();
    let a = bk.from_slice_f32(&[1., 2.], &[1, 2]).unwrap();
    let b = bk.from_slice_f32(&[3., 4., 5.], &[1, 3]).unwrap();
    let y = bk.concat(&[&a, &b], 1).unwrap();
    assert_eq!(y.shape(), &[1, 5]);
    assert_eq!(bk.to_vec_f32(&y).unwrap(), vec![1., 2., 3., 4., 5.]);
}
