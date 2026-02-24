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
