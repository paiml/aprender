// =========================================================================
// FALSIFY-DR: Drift detection contract (aprender metrics)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-DR-* tests for drift detection
//   Why 2: drift tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for drift detection yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Drift detection was "obviously correct" (statistical test)
//
// References:
//   - Gama et al. (2004) "Learning with Drift Detection"
// =========================================================================

use super::*;
use crate::primitives::Vector;

/// FALSIFY-DR-001: Identical distributions yield NoDrift
#[test]
fn falsify_dr_001_no_drift_on_identical() {
    let reference = Vector::from_slice(&(0..50).map(|i| i as f32).collect::<Vec<_>>());
    let current = Vector::from_slice(&(0..50).map(|i| i as f32).collect::<Vec<_>>());

    let detector = DriftDetector::new(DriftConfig::default().with_min_samples(10));
    let status = detector.detect_univariate(&reference, &current);

    assert!(
        matches!(status, DriftStatus::NoDrift),
        "FALSIFIED DR-001: identical data triggered {:?}",
        status
    );
}

/// FALSIFY-DR-002: Very different distributions yield Drift
#[test]
fn falsify_dr_002_drift_on_shifted_data() {
    let reference = Vector::from_slice(&(0..50).map(|i| i as f32).collect::<Vec<_>>());
    let current = Vector::from_slice(&(0..50).map(|i| (i as f32) + 1000.0).collect::<Vec<_>>());

    let detector = DriftDetector::new(DriftConfig::default().with_min_samples(10));
    let status = detector.detect_univariate(&reference, &current);

    assert!(
        status.needs_retraining(),
        "FALSIFIED DR-002: heavily shifted data did not trigger drift: {:?}",
        status
    );
}

/// FALSIFY-DR-003: Drift score is non-negative when present
#[test]
fn falsify_dr_003_score_nonneg() {
    let reference = Vector::from_slice(&(0..50).map(|i| i as f32).collect::<Vec<_>>());
    let current = Vector::from_slice(&(0..50).map(|i| (i as f32) * 2.0).collect::<Vec<_>>());

    let detector = DriftDetector::new(DriftConfig::default().with_min_samples(10));
    let status = detector.detect_univariate(&reference, &current);

    if let Some(score) = status.score() {
        assert!(
            score >= 0.0,
            "FALSIFIED DR-003: drift score={score}, expected >= 0.0"
        );
    }
}

mod dr_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-DR-001-prop: Identical distributions yield NoDrift
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_dr_001_prop_no_drift_identical(
            n in 20..=50usize,
            seed in 0..200u32,
        ) {
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let reference = Vector::from_vec(data.clone());
            let current = Vector::from_vec(data);

            let detector = DriftDetector::new(DriftConfig::default().with_min_samples(10));
            let status = detector.detect_univariate(&reference, &current);
            prop_assert!(
                matches!(status, DriftStatus::NoDrift),
                "FALSIFIED DR-001-prop: identical data triggered {:?}",
                status
            );
        }
    }

    /// FALSIFY-DR-003-prop: Drift score is non-negative when present
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_dr_003_prop_score_nonneg(
            n in 20..=50usize,
            seed in 0..200u32,
        ) {
            let ref_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            let cur_data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0 + 50.0)
                .collect();
            let reference = Vector::from_vec(ref_data);
            let current = Vector::from_vec(cur_data);

            let detector = DriftDetector::new(DriftConfig::default().with_min_samples(10));
            let status = detector.detect_univariate(&reference, &current);
            if let Some(score) = status.score() {
                prop_assert!(
                    score >= 0.0,
                    "FALSIFIED DR-003-prop: drift score={} < 0",
                    score
                );
            }
        }
    }
}
