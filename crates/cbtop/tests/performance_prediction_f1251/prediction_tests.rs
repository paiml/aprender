//! performance_prediction_f1251 - Part 1

use cbtop::{DataPoint, FittedModel, ModelType, PerformancePredictor, MIN_SAMPLES_FOR_FIT};

// =============================================================================
// F1251: Data Point Tests
// =============================================================================

/// F1251.1: Data point stores all fields correctly
#[test]
fn f1251_data_point_creation() {
    let point = DataPoint::new(1024, 100.0, 10.5);

    assert_eq!(point.size, 1024);
    assert_eq!(point.performance, 100.0);
    assert_eq!(point.latency_us, 10.5);
}

/// F1251.2: Data point with edge values
#[test]
fn f1251_data_point_edge_values() {
    let point = DataPoint::new(0, 0.0, 0.0);
    assert_eq!(point.size, 0);

    let large = DataPoint::new(usize::MAX, f64::MAX, f64::MAX);
    assert_eq!(large.size, usize::MAX);
}

// =============================================================================
// F1252: Model Type Tests
// =============================================================================

/// F1252.1: Model type names are correct
#[test]
fn f1252_model_type_names() {
    assert_eq!(ModelType::Linear.name(), "linear");
    assert_eq!(ModelType::Polynomial.name(), "polynomial");
    assert_eq!(ModelType::ExponentialDecay.name(), "exponential_decay");
    assert_eq!(ModelType::Logarithmic.name(), "logarithmic");
    assert_eq!(ModelType::Roofline.name(), "roofline");
}

/// F1252.2: Model types are comparable
#[test]
fn f1252_model_type_equality() {
    assert_eq!(ModelType::Linear, ModelType::Linear);
    assert_ne!(ModelType::Linear, ModelType::Polynomial);
}

// =============================================================================
// F1253: Predictor Creation Tests
// =============================================================================

/// F1253.1: New predictor is empty
#[test]
fn f1253_predictor_empty() {
    let predictor = PerformancePredictor::new();
    assert_eq!(predictor.point_count(), 0);
    assert!(!predictor.has_sufficient_data());
}

/// F1253.2: Predictor with custom confidence level
#[test]
fn f1253_predictor_confidence() {
    let predictor = PerformancePredictor::new().with_confidence(0.90);
    assert_eq!(predictor.point_count(), 0);
}

// =============================================================================
// F1254: Data Addition Tests
// =============================================================================

/// F1254.1: Add points tracks count
#[test]
fn f1254_add_points() {
    let mut predictor = PerformancePredictor::new();

    predictor.add(1000, 10.0, 1.0);
    assert_eq!(predictor.point_count(), 1);

    predictor.add(2000, 20.0, 2.0);
    assert_eq!(predictor.point_count(), 2);
}

/// F1254.2: Add point via struct
#[test]
fn f1254_add_point_struct() {
    let mut predictor = PerformancePredictor::new();

    let point = DataPoint::new(1024, 50.0, 5.0);
    predictor.add_point(point);

    assert_eq!(predictor.point_count(), 1);
}

/// F1254.3: Sufficient data threshold
#[test]
fn f1254_sufficient_data() {
    let mut predictor = PerformancePredictor::new();

    for i in 0..MIN_SAMPLES_FOR_FIT - 1 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }
    assert!(!predictor.has_sufficient_data());

    predictor.add(10000, 100.0, 10.0);
    assert!(predictor.has_sufficient_data());
}

// =============================================================================
// F1255: Linear Fit Tests
// =============================================================================

/// F1255.1: Linear fit on linear data has high R²
#[test]
fn f1255_linear_fit_perfect() {
    let mut predictor = PerformancePredictor::new();

    // Perfect linear: y = 0.01*x
    for i in 1..=10 {
        let size = i * 1000;
        let perf = i as f64 * 10.0;
        predictor.add(size, perf, 1.0);
    }

    let model = predictor.fit_linear().unwrap();
    assert!(model.r_squared > 0.99, "Expected R² > 0.99, got {}", model.r_squared);
    assert_eq!(model.model_type, ModelType::Linear);
}

/// F1255.2: Linear fit requires minimum samples
#[test]
fn f1255_linear_fit_insufficient() {
    let mut predictor = PerformancePredictor::new();

    predictor.add(1000, 10.0, 1.0);
    predictor.add(2000, 20.0, 2.0);

    let result = predictor.fit_linear();
    assert!(result.is_none());
}

/// F1255.3: Linear model predicts correctly
#[test]
fn f1255_linear_predict() {
    let model = FittedModel {
        model_type: ModelType::Linear,
        coefficients: vec![0.01, 0.0], // y = 0.01*x
        r_squared: 1.0,
        rss: 0.0,
        sample_count: 10,
    };

    assert!((model.predict(1000) - 10.0).abs() < 0.001);
    assert!((model.predict(5000) - 50.0).abs() < 0.001);
}

// =============================================================================
// F1256: Polynomial Fit Tests
// =============================================================================

/// F1256.1: Polynomial fit works
#[test]
fn f1256_polynomial_fit() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        let size = i * 1000;
        let perf = i as f64 * 10.0; // Linear data
        predictor.add(size, perf, 1.0);
    }

    let model = predictor.fit_polynomial().unwrap();
    assert_eq!(model.model_type, ModelType::Polynomial);
    assert!(model.r_squared > 0.9);
}

/// F1256.2: Polynomial model has correct coefficients structure
#[test]
fn f1256_polynomial_coefficients() {
    let model = FittedModel {
        model_type: ModelType::Polynomial,
        coefficients: vec![0.001, 0.1, 0.0], // y = 0.001*x² + 0.1*x
        r_squared: 0.95,
        rss: 10.0,
        sample_count: 10,
    };

    let prediction = model.predict(100);
    // 0.001*10000 + 0.1*100 = 10 + 10 = 20
    assert!((prediction - 20.0).abs() < 0.001);
}

// =============================================================================
// F1257: Logarithmic Fit Tests
// =============================================================================

/// F1257.1: Logarithmic fit works
#[test]
fn f1257_logarithmic_fit() {
    let mut predictor = PerformancePredictor::new();

    // Logarithmic-ish data
    for i in 1..=10 {
        let size = i * 1000;
        let perf = (size as f64).ln() * 10.0;
        predictor.add(size, perf, 1.0);
    }

    let model = predictor.fit_logarithmic().unwrap();
    assert_eq!(model.model_type, ModelType::Logarithmic);
    assert!(model.r_squared > 0.9);
}

/// F1257.2: Logarithmic model handles positive values
#[test]
fn f1257_logarithmic_predict() {
    let model = FittedModel {
        model_type: ModelType::Logarithmic,
        coefficients: vec![10.0, 0.0], // y = 10*ln(x)
        r_squared: 1.0,
        rss: 0.0,
        sample_count: 10,
    };

    // ln(1000) ≈ 6.907, so y ≈ 69.07
    let prediction = model.predict(1000);
    assert!((prediction - 69.07).abs() < 0.1);
}

// =============================================================================
// F1258: Fit All and Model Selection Tests
// =============================================================================

/// F1258.1: Fit all returns best model type
#[test]
fn f1258_fit_all() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    let best = predictor.fit_all();
    assert!(best.is_some());
}

/// F1258.2: Best model is accessible
#[test]
fn f1258_best_model() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    predictor.fit_all();
    let model = predictor.best_model();
    assert!(model.is_some());
}

/// F1258.3: Compare models returns ranked list
#[test]
fn f1258_compare_models() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    predictor.fit_all();
    let comparisons = predictor.compare_models();

    assert!(!comparisons.is_empty());
    // First should have highest R²
    if comparisons.len() >= 2 {
        assert!(comparisons[0].1 >= comparisons[1].1);
    }
}
