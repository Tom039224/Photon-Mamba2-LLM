//! Unit tests for `PackedBatcher`. Doesn't depend on a real tokenizer
//! — we'd need a `.tokenizer.json` file for that — so instead we
//! exercise the validation path and the (ids/targets) shape math.

use pm_data::{DataError, PackedBatcher};

#[test]
fn rejects_zero_batch_size() {
    let err = PackedBatcher::new(0, 16, 4, 0).unwrap_err();
    assert!(matches!(err, DataError::Config(_)));
}

#[test]
fn rejects_zero_seq_len() {
    let err = PackedBatcher::new(2, 0, 4, 0).unwrap_err();
    assert!(matches!(err, DataError::Config(_)));
}

#[test]
fn rejects_seq_len_not_multiple_of_chunk_product() {
    let err = PackedBatcher::new(2, 10, 4, 0).unwrap_err();
    assert!(matches!(err, DataError::Config(_)));
}

#[test]
fn accepts_valid_config() {
    let p = PackedBatcher::new(2, 16, 4, 0).unwrap();
    assert_eq!(p.batch_size, 2);
    assert_eq!(p.seq_len, 16);
    assert_eq!(p.pad_token_id, 0);
}
