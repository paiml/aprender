// =========================================================================
// FALSIFY-ICA: ica-v1.yaml contract (aprender ICA)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-ICA-* tests
//   Why 2: ICA tests only in tests/contracts/, not near implementation
//   Why 3: no mapping from ica-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: FastICA was "obviously correct" (textbook negentropy maximization)
//
// References:
//   - provable-contracts/contracts/ica-v1.yaml
//   - Hyvarinen & Oja (2000) "Independent Component Analysis"
// =========================================================================

use super::*;
use crate::primitives::Matrix;

/// FALSIFY-ICA-001: Output dimensionality matches n_components
#[test]
fn falsify_ica_001_output_shape() {
    let x = Matrix::from_vec(10, 3, vec![
        1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0,
        -1.0, 0.0, -0.5, 0.0, -1.0, -0.5, -1.0, -1.0, -1.0,
        0.5, -0.5, 0.0, -0.5, 0.5, 0.0, 0.5, 0.5, 0.5,
        0.3, 0.7, 0.5,
    ]).expect("valid");

    let mut ica = ICA::new(2);
    ica.fit(&x).expect("fit");

    let transformed = ica.transform(&x).expect("transform");
    let (rows, cols) = transformed.shape();
    assert_eq!(rows, 10, "FALSIFIED ICA-001: output rows={rows}, expected 10");
    assert_eq!(cols, 2, "FALSIFIED ICA-001: output cols={cols}, expected 2");
}

/// FALSIFY-ICA-002: Transformed values are finite
#[test]
fn falsify_ica_002_finite_output() {
    let x = Matrix::from_vec(8, 2, vec![
        1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0,
        0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5,
    ]).expect("valid");

    let mut ica = ICA::new(2);
    ica.fit(&x).expect("fit");

    let transformed = ica.transform(&x).expect("transform");
    let (rows, cols) = transformed.shape();
    for i in 0..rows {
        for j in 0..cols {
            let v = transformed.get(i, j);
            assert!(
                v.is_finite(),
                "FALSIFIED ICA-002: output[{i},{j}] = {v} is not finite"
            );
        }
    }
}

/// FALSIFY-ICA-003: Deterministic with same data
#[test]
fn falsify_ica_003_deterministic() {
    let x = Matrix::from_vec(6, 2, vec![
        1.0, 0.0, 0.0, 1.0, -1.0, 0.0,
        0.0, -1.0, 0.5, 0.5, -0.5, -0.5,
    ]).expect("valid");

    let mut ica1 = ICA::new(2);
    ica1.fit(&x).expect("fit 1");
    let t1 = ica1.transform(&x).expect("transform 1");

    let mut ica2 = ICA::new(2);
    ica2.fit(&x).expect("fit 2");
    let t2 = ica2.transform(&x).expect("transform 2");

    let (rows, cols) = t1.shape();
    for i in 0..rows {
        for j in 0..cols {
            assert!(
                (t1.get(i, j) - t2.get(i, j)).abs() < 1e-4,
                "FALSIFIED ICA-003: output differs at [{i},{j}]"
            );
        }
    }
}
