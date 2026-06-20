pub(crate) use super::*;

#[test]
fn test_temperature_scaling_new() {
    let ts = TemperatureScaling::new();
    assert_eq!(ts.temperature(), 1.0);
}

#[test]
fn test_temperature_scaling_calibrate() {
    let mut ts = TemperatureScaling::new();
    ts.temperature = 2.0;

    let logits = Vector::from_slice(&[2.0, 4.0, 6.0]);
    let calibrated = ts.calibrate(&logits);

    assert_eq!(calibrated.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_temperature_scaling_predict_proba() {
    let ts = TemperatureScaling::new();
    let logits = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let probs = ts.predict_proba(&logits);

    let sum: f32 = probs.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_temperature_scaling_fit() {
    let mut ts = TemperatureScaling::new();

    let logits = vec![
        Vector::from_slice(&[2.0, 1.0]),
        Vector::from_slice(&[1.0, 2.0]),
        Vector::from_slice(&[3.0, 0.0]),
    ];
    let labels = vec![0, 1, 0];

    ts.fit(&logits, &labels);
    assert!(ts.temperature() > 0.0);
}

#[test]
fn test_platt_scaling_new() {
    let ps = PlattScaling::new();
    assert_eq!(ps.params(), (1.0, 0.0));
}

#[test]
fn test_platt_scaling_predict_proba() {
    let ps = PlattScaling::new();
    let prob = ps.predict_proba(0.0);
    assert!((prob - 0.5).abs() < 1e-5);
}

#[test]
fn test_platt_scaling_fit() {
    let mut ps = PlattScaling::new();
    let logits = vec![2.0, 1.0, -1.0, -2.0, 0.5, -0.5];
    let labels = vec![true, true, false, false, true, false];

    ps.fit(&logits, &labels);
    // After fitting, higher logits should give higher probability
    assert!(ps.predict_proba(2.0) > ps.predict_proba(-2.0));
}

#[test]
fn test_ece_perfect_calibration() {
    let predictions = vec![0.9, 0.9, 0.1, 0.1];
    let labels = vec![true, true, false, false];

    let ece = expected_calibration_error(&predictions, &labels, 10);
    assert!(ece < 0.2);
}

#[test]
fn test_ece_poor_calibration() {
    let predictions = vec![0.9, 0.9, 0.9, 0.9];
    let labels = vec![true, false, false, false];

    let ece = expected_calibration_error(&predictions, &labels, 10);
    assert!(ece > 0.5);
}

#[test]
fn test_mce() {
    let predictions = vec![0.9, 0.9, 0.1, 0.1];
    let labels = vec![true, true, false, false];

    let mce = maximum_calibration_error(&predictions, &labels, 10);
    assert!(mce < 0.2);
}

#[test]
fn test_softmax() {
    let logits = vec![1.0, 2.0, 3.0];
    let probs = softmax(&logits);

    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(probs[2] > probs[1]);
    assert!(probs[1] > probs[0]);
}

#[test]
fn test_sigmoid() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-5);
    assert!(sigmoid(10.0) > 0.99);
    assert!(sigmoid(-10.0) < 0.01);
}

#[test]
fn test_isotonic_new() {
    let iso = IsotonicRegression::new();
    assert!(iso.thresholds.is_empty());
    assert!(iso.values.is_empty());
}

#[test]
fn test_isotonic_fit() {
    let mut iso = IsotonicRegression::new();
    let predictions = vec![0.1, 0.4, 0.6, 0.9];
    let labels = vec![false, false, true, true];

    iso.fit(&predictions, &labels);

    assert!(!iso.thresholds.is_empty());
    assert!(!iso.values.is_empty());
}

#[test]
fn test_isotonic_predict() {
    let mut iso = IsotonicRegression::new();
    let predictions = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    let labels = vec![false, false, true, true, true];

    iso.fit(&predictions, &labels);

    // Test predictions at various points
    let p1 = iso.predict(0.2);
    let p2 = iso.predict(0.8);

    // Low prediction should give low calibrated value
    // High prediction should give high calibrated value
    assert!(
        p2 >= p1,
        "Higher predictions should give higher calibrated values"
    );
    assert!((0.0..=1.0).contains(&p1));
    assert!((0.0..=1.0).contains(&p2));
}

#[test]
fn test_isotonic_monotonic() {
    let mut iso = IsotonicRegression::new();
    // Non-monotonic accuracy pattern
    let predictions = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let labels = vec![false, true, false, true, true, false, true, true, true];

    iso.fit(&predictions, &labels);

    // Calibrated values should be monotonically non-decreasing
    let mut prev = iso.predict(0.0);
    for i in 1..=10 {
        let x = i as f32 / 10.0;
        let curr = iso.predict(x);
        assert!(
            curr >= prev - 1e-6,
            "Isotonic should be monotonic: {curr} < {prev}"
        );
        prev = curr;
    }
}

// PMAT-870: PAV pooled-block flatness (sklearn IsotonicRegression parity).
//
// After Pool-Adjacent-Violators pooling, the fitted function must be FLAT at the
// pooled value across the WHOLE x-range of a pooled block. The bug recorded only
// the block's min-x as a knot, discarding the max-x, so a query strictly inside a
// pooled block was linearly interpolated toward the NEXT block instead of returning
// the constant pooled value.
//
// Reference: sklearn.isotonic.IsotonicRegression
//   X=[0.0,0.1,0.2,0.3], y=[0,1,0,1] → PAV pools x=0.1,0.2 to value 0.5
//   X_thresholds_=[0.0,0.1,0.2,0.3], y_thresholds_=[0.0,0.5,0.5,1.0]
//   predict(0.2)=0.5, predict(0.15)=0.5, predict(0.25)=0.75
#[test]
fn test_isotonic_pav_block_is_flat_pmat_870() {
    let mut iso = IsotonicRegression::new();
    let predictions = vec![0.0, 0.1, 0.2, 0.3];
    let labels = vec![false, true, false, true]; // y = [0, 1, 0, 1]

    iso.fit(&predictions, &labels);

    // Inside the pooled [0.1, 0.2] block → flat at the pooled value 0.5.
    // RED (bug): predict(0.2)=0.75, predict(0.15)=0.625
    // GREEN (fix): both = 0.5
    let p_max_edge = iso.predict(0.2);
    let p_mid = iso.predict(0.15);
    assert!(
        (p_max_edge - 0.5).abs() < 1e-5,
        "PMAT-870 FALSIFIED: predict(0.2)={p_max_edge}, expected 0.5 (flat pooled block)"
    );
    assert!(
        (p_mid - 0.5).abs() < 1e-5,
        "PMAT-870 FALSIFIED: predict(0.15)={p_mid}, expected 0.5 (flat pooled block)"
    );

    // Between the pooled block (val 0.5 at x=0.2) and the final block (val 1.0 at
    // x=0.3): linear interpolation. predict(0.25) = 0.5 + 0.5*(1.0-0.5) = 0.75.
    // RED (bug): 0.875
    let p_interp = iso.predict(0.25);
    assert!(
        (p_interp - 0.75).abs() < 1e-5,
        "PMAT-870 FALSIFIED: predict(0.25)={p_interp}, expected 0.75 (interp between blocks)"
    );
}

// PMAT-870: NO-POOL guard — when no adjacent violators exist, behavior must be
// unchanged. Each x is its own block, so interpolation between distinct single-point
// blocks still applies. X=[0,0.2,0.4,0.6,0.8,1.0], y=[0,0,0,1,1,1].
// PAV pools the two halves: block A (x 0..0.4, val 0) and block B (x 0.6..1.0, val 1).
// predict(0.5) interpolates between A's max-edge (x=0.4, val 0) and B's min-edge
// (x=0.6, val 1): 0.0 + 0.5*(1.0-0.0) = 0.5.
#[test]
fn test_isotonic_no_pool_guard_pmat_870() {
    let mut iso = IsotonicRegression::new();
    let predictions = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let labels = vec![false, false, false, true, true, true];

    iso.fit(&predictions, &labels);

    let p = iso.predict(0.5);
    assert!(
        (p - 0.5).abs() < 1e-5,
        "PMAT-870 FALSIFIED (no-pool guard): predict(0.5)={p}, expected 0.5"
    );
    // Endpoints stay clamped at the block values.
    assert!((iso.predict(0.1) - 0.0).abs() < 1e-5);
    assert!((iso.predict(0.9) - 1.0).abs() < 1e-5);
}

#[test]
fn test_reliability_diagram() {
    let predictions = vec![0.1, 0.2, 0.8, 0.9];
    let labels = vec![false, false, true, true];

    let diagram = reliability_diagram(&predictions, &labels, 5);

    assert_eq!(diagram.len(), 5);
    for bin in &diagram {
        assert!(bin.0 >= 0.0 && bin.0 <= 1.0);
        assert!(bin.1 >= 0.0 && bin.1 <= 1.0);
    }
}

#[test]
fn test_brier_score() {
    // Perfect predictions
    let predictions = vec![1.0, 0.0, 1.0, 0.0];
    let labels = vec![true, false, true, false];
    let brier = brier_score(&predictions, &labels);
    assert!((brier - 0.0).abs() < 1e-6);

    // Worst predictions
    let predictions = vec![0.0, 1.0, 0.0, 1.0];
    let labels = vec![true, false, true, false];
    let brier = brier_score(&predictions, &labels);
    assert!((brier - 1.0).abs() < 1e-6);
}
