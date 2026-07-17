//! Cross-check `pm-core::mamba2::ssd_scan_naive_scalar` against the
//! PyTorch reference fixture in `reference/fixtures/ssd_q64/`.
//!
//! Regenerate the fixture with `python reference/gen_fixtures.py`.

use std::path::{Path, PathBuf};

use pm_core::mamba2::ssd_scan_naive_scalar;

fn fixtures_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/pm-core; fixtures live at workspace root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("reference")
        .join("fixtures")
        .join("ssd_q64")
}

fn load_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let reader = npyz::NpyFile::new(bytes.as_slice())
        .unwrap_or_else(|e| panic!("parse npy {}: {e}", path.display()));
    reader
        .into_vec::<f32>()
        .unwrap_or_else(|e| panic!("decode f32 {}: {e}", path.display()))
}

#[test]
fn ssd_scan_naive_scalar_matches_pytorch_fixture() {
    let dir = fixtures_dir();
    if !dir.exists() {
        // Fixture missing in clean checkout. Surface a helpful skip rather
        // than a confusing FileNotFound.
        eprintln!(
            "skipping: fixture dir not present at {}. \
             Generate it with `python reference/gen_fixtures.py`.",
            dir.display()
        );
        return;
    }

    // Shape matches reference/gen_fixtures.py.
    let (batch, t_len, n_heads, p_dim, n_dim) = (1, 128, 2, 8, 16);

    let x = load_f32(&dir.join("X.npy"));
    let a = load_f32(&dir.join("A.npy"));
    let b = load_f32(&dir.join("B.npy"));
    let c = load_f32(&dir.join("C.npy"));
    let y_ref = load_f32(&dir.join("Y.npy"));

    let y = ssd_scan_naive_scalar(&x, &a, &b, &c, batch, t_len, n_heads, p_dim, n_dim);

    assert_eq!(y.len(), y_ref.len());
    let mut max_abs_err = 0f32;
    let mut max_rel_err = 0f32;
    for (yi, yri) in y.iter().zip(y_ref.iter()) {
        let abs_err = (yi - yri).abs();
        max_abs_err = max_abs_err.max(abs_err);
        let denom = yri.abs().max(1e-3);
        max_rel_err = max_rel_err.max(abs_err / denom);
    }
    // The scalar Rust impl sums in a different order than PyTorch's
    // chunked SSD (different einsum reductions). For T=128, accumulated
    // fp32 rounding can reach ~1e-3 in the worst element. This is below
    // the fp32 sensitivity threshold for downstream training (loss-level
    // noise is dominated by the input randomness); we widen accordingly.
    assert!(
        max_abs_err < 5e-3,
        "max abs error {max_abs_err} exceeds 5e-3"
    );
    assert!(
        max_rel_err < 5e-3,
        "max rel error {max_rel_err} exceeds 5e-3"
    );
    eprintln!("ssd_fixture: max_abs_err = {max_abs_err:.2e}, max_rel_err = {max_rel_err:.2e}");
}
