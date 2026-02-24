// =========================================================================
// FALSIFY-MR: metrics-regression-v1.yaml contract (aprender regression metrics)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-MR-* tests for regression metrics
//   Why 2: metric tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from metrics-regression-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: R² was "obviously correct" (1 - SS_res/SS_tot)
//
// References:
//   - provable-contracts/contracts/metrics-regression-v1.yaml
// =========================================================================

use super::*;
use crate::primitives::Vector;

/// FALSIFY-MR-001: R² = 1.0 for perfect predictions
#[test]
fn falsify_mr_001_r2_perfect() {
    let y_true = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    let r2 = r_squared(&y_pred, &y_true);
    assert!(
        (r2 - 1.0).abs() < 1e-6,
        "FALSIFIED MR-001: R²={r2} for perfect predictions, expected 1.0"
    );
}

/// FALSIFY-MR-002: R² ≤ 1.0 always
#[test]
fn falsify_mr_002_r2_upper_bound() {
    let y_true = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_pred = Vector::from_slice(&[1.1, 2.2, 2.8, 4.1, 4.9]);

    let r2 = r_squared(&y_pred, &y_true);
    assert!(r2 <= 1.0 + 1e-6, "FALSIFIED MR-002: R²={r2} > 1.0");
}

/// FALSIFY-MR-003: R² < 0 when predictions are worse than mean
#[test]
fn falsify_mr_003_r2_negative_for_bad_predictions() {
    let y_true = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_pred = Vector::from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0]);

    let r2 = r_squared(&y_pred, &y_true);
    assert!(
        r2 < 0.0,
        "FALSIFIED MR-003: R²={r2} >= 0 for terrible predictions (expected negative)"
    );
}

/// FALSIFY-MR-004: R² is symmetric when pred == target (reflexive)
#[test]
fn falsify_mr_004_r2_deterministic() {
    let y = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    let r2_1 = r_squared(&y, &y);
    let r2_2 = r_squared(&y, &y);
    assert_eq!(r2_1, r2_2, "FALSIFIED MR-004: R² differs on same input");
}
