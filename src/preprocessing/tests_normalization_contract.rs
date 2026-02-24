// =========================================================================
// FALSIFY-PN: preprocessing-normalization-v1.yaml contract
//            (aprender StandardScaler, MinMaxScaler)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-PN-* tests for scalers
//   Why 2: preprocessing tests only in tests/contracts/, not near implementation
//   Why 3: no mapping from preprocessing-normalization-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: StandardScaler was "obviously correct" (z-score normalization)
//
// References:
//   - provable-contracts/contracts/preprocessing-normalization-v1.yaml
// =========================================================================

use super::*;
use crate::primitives::Matrix;
use crate::traits::Transformer;

/// FALSIFY-PN-001: StandardScaler output has zero mean (within tolerance)
#[test]
fn falsify_pn_001_standard_scaler_zero_mean() {
    let x = Matrix::from_vec(
        5,
        2,
        vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0],
    )
    .expect("valid");

    let mut scaler = StandardScaler::new();
    scaler.fit(&x).expect("fit");
    let transformed = scaler.transform(&x).expect("transform");

    let (n, p) = transformed.shape();
    for j in 0..p {
        let mean: f32 = (0..n).map(|i| transformed.get(i, j)).sum::<f32>() / n as f32;
        assert!(
            mean.abs() < 1e-5,
            "FALSIFIED PN-001: column {j} mean={mean}, expected ≈ 0"
        );
    }
}

/// FALSIFY-PN-002: MinMaxScaler output in [0, 1] range
#[test]
fn falsify_pn_002_minmax_scaler_bounded() {
    let x = Matrix::from_vec(
        5,
        2,
        vec![
            -10.0, 100.0, 0.0, 200.0, 10.0, 300.0, 20.0, 400.0, 30.0, 500.0,
        ],
    )
    .expect("valid");

    let mut scaler = MinMaxScaler::new();
    scaler.fit(&x).expect("fit");
    let transformed = scaler.transform(&x).expect("transform");

    let (n, p) = transformed.shape();
    for i in 0..n {
        for j in 0..p {
            let v = transformed.get(i, j);
            assert!(
                (-1e-6..=1.0 + 1e-6).contains(&v),
                "FALSIFIED PN-002: value[{i},{j}]={v} outside [0, 1]"
            );
        }
    }
}

/// FALSIFY-PN-003: Output shape preserved
#[test]
fn falsify_pn_003_shape_preserved() {
    let x = Matrix::from_vec(
        4,
        3,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .expect("valid");

    let mut std_scaler = StandardScaler::new();
    std_scaler.fit(&x).expect("fit");
    let t1 = std_scaler.transform(&x).expect("transform");
    assert_eq!(
        t1.shape(),
        (4, 3),
        "FALSIFIED PN-003: StandardScaler changed shape"
    );

    let mut mm_scaler = MinMaxScaler::new();
    mm_scaler.fit(&x).expect("fit");
    let t2 = mm_scaler.transform(&x).expect("transform");
    assert_eq!(
        t2.shape(),
        (4, 3),
        "FALSIFIED PN-003: MinMaxScaler changed shape"
    );
}

/// FALSIFY-PN-004: MinMaxScaler inverse_transform round-trip
#[test]
fn falsify_pn_004_minmax_inverse_roundtrip() {
    let x =
        Matrix::from_vec(4, 2, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]).expect("valid");

    let mut scaler = MinMaxScaler::new();
    scaler.fit(&x).expect("fit");
    let transformed = scaler.transform(&x).expect("transform");
    let recovered = scaler
        .inverse_transform(&transformed)
        .expect("inverse_transform");

    let (n, p) = x.shape();
    for i in 0..n {
        for j in 0..p {
            assert!(
                (x.get(i, j) - recovered.get(i, j)).abs() < 1e-4,
                "FALSIFIED PN-004: round-trip error at [{i},{j}]: original={}, recovered={}",
                x.get(i, j),
                recovered.get(i, j)
            );
        }
    }
}

mod pn_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-PN-002-prop: MinMaxScaler output in [0, 1] for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_pn_002_prop_minmax_bounded(
            n in 4..=12usize,
            seed in 0..200u32,
        ) {
            let data: Vec<f32> = (0..n * 2)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 100.0)
                .collect();
            let x = Matrix::from_vec(n, 2, data).expect("valid");

            let mut scaler = MinMaxScaler::new();
            scaler.fit(&x).expect("fit");
            let transformed = scaler.transform(&x).expect("transform");

            let (rows, cols) = transformed.shape();
            for i in 0..rows {
                for j in 0..cols {
                    let v = transformed.get(i, j);
                    prop_assert!(
                        (-1e-5..=1.0 + 1e-5).contains(&v),
                        "FALSIFIED PN-002-prop: value[{},{}]={} outside [0,1]",
                        i, j, v
                    );
                }
            }
        }
    }

    /// FALSIFY-PN-003-prop: Output shape preserved for random sizes
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_pn_003_prop_shape_preserved(
            n in 3..=10usize,
            p in 1..=4usize,
            seed in 0..200u32,
        ) {
            let data: Vec<f32> = (0..n * p)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 50.0)
                .collect();
            let x = Matrix::from_vec(n, p, data).expect("valid");

            let mut scaler = StandardScaler::new();
            scaler.fit(&x).expect("fit");
            let transformed = scaler.transform(&x).expect("transform");
            prop_assert_eq!(
                transformed.shape(),
                (n, p),
                "FALSIFIED PN-003-prop: shape changed"
            );
        }
    }
}
