//! performance_prediction_f1251 - Part 2

use cbtop::{FittedModel, ModelType, PerformancePredictor, Prediction, MIN_SAMPLES_FOR_FIT};

// =============================================================================
// F1259: Prediction Tests
// =============================================================================

/// F1259.1: Interpolation prediction
#[test]
fn f1259_interpolation() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    let pred = predictor.predict_at_size(5000).unwrap();

    assert!(!pred.is_extrapolation);
    assert!(pred.is_reasonable());
    assert!(pred.predicted > 0.0);
}

/// F1259.2: Extrapolation prediction
#[test]
fn f1259_extrapolation() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    // Beyond training range
    let pred = predictor.predict_at_size(20000).unwrap();

    assert!(pred.is_extrapolation);
    // Extrapolation should have wider bounds
    assert!(pred.range_width() > 0.0);
}

/// F1259.3: Prediction has confidence bounds
#[test]
fn f1259_confidence_bounds() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    let pred = predictor.predict_at_size(5000).unwrap();

    assert!(pred.lower_bound <= pred.predicted);
    assert!(pred.upper_bound >= pred.predicted);
    assert!(pred.confidence_level > 0.0);
}

// =============================================================================
// F1260: Utility Tests
// =============================================================================

/// F1260.1: Size range calculation
#[test]
fn f1260_size_range() {
    let mut predictor = PerformancePredictor::new();

    predictor.add(1000, 10.0, 1.0);
    predictor.add(5000, 50.0, 5.0);
    predictor.add(10000, 100.0, 10.0);

    let (min, max) = predictor.size_range().unwrap();
    assert_eq!(min, 1000);
    assert_eq!(max, 10000);
}

/// F1260.2: Empty predictor size range
#[test]
fn f1260_size_range_empty() {
    let predictor = PerformancePredictor::new();
    assert!(predictor.size_range().is_none());
}

/// F1260.3: Clear predictor
#[test]
fn f1260_clear() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    assert_eq!(predictor.point_count(), 10);

    predictor.clear();
    assert_eq!(predictor.point_count(), 0);
}

/// F1260.4: Export model
#[test]
fn f1260_export() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=5 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    predictor.fit_linear();

    let export = predictor.export_model(ModelType::Linear).unwrap();
    assert!(export.contains("linear"));
    assert!(export.contains("r_squared"));
    assert!(export.contains("coefficients"));
}

/// F1260.5: Get specific model
#[test]
fn f1260_get_model() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=5 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    predictor.fit_linear();

    let model = predictor.get_model(ModelType::Linear);
    assert!(model.is_some());

    let missing = predictor.get_model(ModelType::Roofline);
    assert!(missing.is_none());
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test fitted model good fit threshold
#[test]
fn test_good_fit() {
    let good = FittedModel {
        model_type: ModelType::Linear,
        coefficients: vec![1.0, 0.0],
        r_squared: 0.95,
        rss: 0.1,
        sample_count: 10,
    };
    assert!(good.is_good_fit());

    let bad = FittedModel {
        model_type: ModelType::Linear,
        coefficients: vec![1.0, 0.0],
        r_squared: 0.5,
        rss: 100.0,
        sample_count: 10,
    };
    assert!(!bad.is_good_fit());
}

/// Test prediction uncertainty percent
#[test]
fn test_uncertainty_percent() {
    let pred = Prediction {
        size: 1000,
        predicted: 100.0,
        lower_bound: 80.0,
        upper_bound: 120.0,
        confidence_level: 0.95,
        model_type: ModelType::Linear,
        is_extrapolation: false,
    };

    // range_width = 40, uncertainty = 40/100*100/2 = 20%
    assert!((pred.uncertainty_percent() - 20.0).abs() < 0.001);
}

/// Test min samples constant
#[test]
fn test_min_samples_constant() {
    assert_eq!(MIN_SAMPLES_FOR_FIT, 5);
}

/// Test roofline model prediction
#[test]
fn test_roofline_predict() {
    let model = FittedModel {
        model_type: ModelType::Roofline,
        coefficients: vec![100.0, 1000.0, 0.1], // peak=100, knee=1000, slope=0.1
        r_squared: 0.95,
        rss: 10.0,
        sample_count: 10,
    };

    // Small size: linear region
    let small = model.predict(500);
    assert!((small - 50.0).abs() < 0.001); // 0.1 * 500 = 50

    // Large size: capped at peak
    let large = model.predict(2000);
    assert!((large - 100.0).abs() < 0.001); // capped at peak=100
}

/// Test exponential decay model prediction
#[test]
fn test_exponential_decay_predict() {
    let model = FittedModel {
        model_type: ModelType::ExponentialDecay,
        coefficients: vec![10.0, 0.001, 5.0], // a=10, b=0.001, c=5
        r_squared: 0.95,
        rss: 10.0,
        sample_count: 10,
    };

    let prediction = model.predict(0);
    // 10 * exp(0) + 5 = 10 + 5 = 15
    assert!((prediction - 15.0).abs() < 0.001);
}

/// Test prediction not reasonable with zero
#[test]
fn test_prediction_not_reasonable() {
    let bad_pred = Prediction {
        size: 1000,
        predicted: 0.0,
        lower_bound: 0.0,
        upper_bound: 0.0,
        confidence_level: 0.95,
        model_type: ModelType::Linear,
        is_extrapolation: false,
    };

    assert!(!bad_pred.is_reasonable());
}
