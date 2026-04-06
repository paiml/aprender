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
