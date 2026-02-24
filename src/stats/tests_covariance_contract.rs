// =========================================================================
// FALSIFY-CV: covariance/correlation contract (aprender stats)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-CV-* tests for covariance
//   Why 2: covariance tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for covariance yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Cov(X,Y) was "obviously correct" (basic statistics)
//
// References:
//   - Pearson (1895) "Note on regression and inheritance"
// =========================================================================

use super::*;
use crate::primitives::Vector;

/// FALSIFY-CV-001: Correlation of identical vectors is 1.0
#[test]
fn falsify_cv_001_self_correlation_is_one() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let r = corr(&x, &x).expect("valid input");

    assert!(
        (r - 1.0).abs() < 1e-5,
        "FALSIFIED CV-001: corr(x,x)={r}, expected 1.0"
    );
}

/// FALSIFY-CV-002: Correlation is bounded in [-1, 1]
#[test]
fn falsify_cv_002_correlation_bounded() {
    let x = Vector::from_slice(&[1.0, 3.0, 2.0, 5.0, 4.0]);
    let y = Vector::from_slice(&[2.0, 1.0, 4.0, 3.0, 5.0]);
    let r = corr(&x, &y).expect("valid input");

    assert!(
        (-1.0..=1.0).contains(&r),
        "FALSIFIED CV-002: corr={r} outside [-1, 1]"
    );
}

/// FALSIFY-CV-003: Perfect negative correlation is -1.0
#[test]
fn falsify_cv_003_perfect_negative_correlation() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = Vector::from_slice(&[5.0, 4.0, 3.0, 2.0, 1.0]);
    let r = corr(&x, &y).expect("valid input");

    assert!(
        (r - (-1.0)).abs() < 1e-5,
        "FALSIFIED CV-003: corr(x, -x)={r}, expected -1.0"
    );
}

/// FALSIFY-CV-004: Covariance is symmetric: Cov(X,Y) = Cov(Y,X)
#[test]
fn falsify_cv_004_covariance_symmetric() {
    let x = Vector::from_slice(&[1.0, 3.0, 5.0, 7.0]);
    let y = Vector::from_slice(&[2.0, 4.0, 1.0, 3.0]);

    let cov_xy = cov(&x, &y).expect("valid input");
    let cov_yx = cov(&y, &x).expect("valid input");

    assert!(
        (cov_xy - cov_yx).abs() < 1e-6,
        "FALSIFIED CV-004: Cov(X,Y)={cov_xy} != Cov(Y,X)={cov_yx}"
    );
}

mod cv_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-CV-002-prop: Correlation bounded in [-1, 1] for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_cv_002_prop_correlation_bounded(
            n in 3..=20usize,
            seed in 0..500u32,
        ) {
            let x_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let y_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32 + 1.0) * 0.53).sin() * 10.0)
                .collect();
            let x = Vector::from_vec(x_data);
            let y = Vector::from_vec(y_data);
            let r = corr(&x, &y).expect("valid");
            prop_assert!(
                (-1.0 - 1e-5..=1.0 + 1e-5).contains(&r),
                "FALSIFIED CV-002-prop: corr={} outside [-1,1]",
                r
            );
        }
    }

    /// FALSIFY-CV-004-prop: Covariance symmetric for random data
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_cv_004_prop_covariance_symmetric(
            n in 3..=20usize,
            seed in 0..500u32,
        ) {
            let x_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let y_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32 + 1.0) * 0.53).sin() * 10.0)
                .collect();
            let x = Vector::from_vec(x_data);
            let y = Vector::from_vec(y_data);
            let cxy = cov(&x, &y).expect("valid");
            let cyx = cov(&y, &x).expect("valid");
            prop_assert!(
                (cxy - cyx).abs() < 1e-4,
                "FALSIFIED CV-004-prop: Cov(X,Y)={} != Cov(Y,X)={}",
                cxy, cyx
            );
        }
    }
}
