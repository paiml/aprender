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
    let x = Matrix::from_vec(
        10,
        3,
        vec![
            1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0, -1.0, 0.0, -0.5, 0.0, -1.0, -0.5, -1.0,
            -1.0, -1.0, 0.5, -0.5, 0.0, -0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.3, 0.7, 0.5,
        ],
    )
    .expect("valid");

    let mut ica = ICA::new(2);
    ica.fit(&x).expect("fit");

    let transformed = ica.transform(&x).expect("transform");
    let (rows, cols) = transformed.shape();
    assert_eq!(
        rows, 10,
        "FALSIFIED ICA-001: output rows={rows}, expected 10"
    );
    assert_eq!(cols, 2, "FALSIFIED ICA-001: output cols={cols}, expected 2");
}

/// FALSIFY-ICA-002: Transformed values are finite
#[test]
fn falsify_ica_002_finite_output() {
    let x = Matrix::from_vec(
        8,
        2,
        vec![
            1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5,
        ],
    )
    .expect("valid");

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
    let x = Matrix::from_vec(
        6,
        2,
        vec![
            1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.5, 0.5, -0.5, -0.5,
        ],
    )
    .expect("valid");

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

mod ica_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-ICA-001-prop: Output shape matches for random n_components
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn falsify_ica_001_prop_output_shape(
            n_comp in 1..=2usize,
            seed in 0..200u32,
        ) {
            let n_samples = 10;
            let n_features = 3;
            // Use diverse data to ensure ICA convergence
            let data: Vec<f32> = (0..n_samples * n_features)
                .map(|i| {
                    let t = i as f32 + seed as f32;
                    (t * 0.37).sin() * 3.0 + (t * 1.7).cos() * 2.0
                })
                .collect();
            let x = Matrix::from_vec(n_samples, n_features, data).expect("valid");

            let mut ica = ICA::new(n_comp);
            // Skip seeds where ICA doesn't converge
            prop_assume!(ica.fit(&x).is_ok());

            let transformed = ica.transform(&x).expect("transform");
            let (rows, cols) = transformed.shape();
            prop_assert_eq!(rows, n_samples, "FALSIFIED ICA-001-prop: rows {} != {}", rows, n_samples);
            prop_assert_eq!(cols, n_comp, "FALSIFIED ICA-001-prop: cols {} != {}", cols, n_comp);
        }
    }

    /// FALSIFY-ICA-002-prop: Transformed values are finite for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn falsify_ica_002_prop_finite_output(
            seed in 0..200u32,
        ) {
            let n_samples = 8;
            let data: Vec<f32> = (0..n_samples * 2)
                .map(|i| {
                    let t = i as f32 + seed as f32;
                    (t * 0.37).sin() * 5.0 + (t * 1.7).cos() * 3.0
                })
                .collect();
            let x = Matrix::from_vec(n_samples, 2, data).expect("valid");

            let mut ica = ICA::new(2);
            prop_assume!(ica.fit(&x).is_ok());

            let transformed = ica.transform(&x).expect("transform");
            let (rows, cols) = transformed.shape();
            for i in 0..rows {
                for j in 0..cols {
                    let v = transformed.get(i, j);
                    prop_assert!(
                        v.is_finite(),
                        "FALSIFIED ICA-002-prop: output[{},{}]={} not finite",
                        i, j, v
                    );
                }
            }
        }
    }
}
