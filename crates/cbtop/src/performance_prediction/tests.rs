use super::*;

#[test]
fn test_data_point() {
    let p = DataPoint::new(1024, 100.0, 10.0);
    assert_eq!(p.size, 1024);
    assert_eq!(p.performance, 100.0);
}

#[test]
fn test_model_type_names() {
    assert_eq!(ModelType::Linear.name(), "linear");
    assert_eq!(ModelType::Polynomial.name(), "polynomial");
    assert_eq!(ModelType::Logarithmic.name(), "logarithmic");
}

#[test]
fn test_predictor_add_points() {
    let mut predictor = PerformancePredictor::new();

    predictor.add(1024, 100.0, 10.0);
    predictor.add(2048, 150.0, 15.0);

    assert_eq!(predictor.point_count(), 2);
    assert!(!predictor.has_sufficient_data());

    predictor.add(4096, 200.0, 20.0);
    predictor.add(8192, 250.0, 25.0);
    predictor.add(16384, 300.0, 30.0);

    assert!(predictor.has_sufficient_data());
}

#[test]
fn test_linear_fit() {
    let mut predictor = PerformancePredictor::new();

    // Perfect linear data
    for i in 1..=10 {
        let size = i * 1000;
        let perf = i as f64 * 10.0;
        predictor.add(size, perf, 1.0);
    }

    let model = predictor.fit_linear().unwrap();
    assert!(model.r_squared > 0.99);
}

#[test]
fn test_prediction() {
    let mut predictor = PerformancePredictor::new();

    // Linear data
    for i in 1..=10 {
        let size = i * 1000;
        let perf = i as f64 * 10.0;
        predictor.add(size, perf, 1.0);
    }

    predictor.fit_all();

    // Interpolation
    let pred = predictor.predict_at_size(5000).unwrap();
    assert!(!pred.is_extrapolation);
    assert!(pred.is_reasonable());

    // Extrapolation
    let pred_ext = predictor.predict_at_size(20000).unwrap();
    assert!(pred_ext.is_extrapolation);
}

#[test]
fn test_size_range() {
    let mut predictor = PerformancePredictor::new();

    predictor.add(1000, 10.0, 1.0);
    predictor.add(5000, 50.0, 5.0);
    predictor.add(10000, 100.0, 10.0);

    let (min, max) = predictor.size_range().unwrap();
    assert_eq!(min, 1000);
    assert_eq!(max, 10000);
}

#[test]
fn test_compare_models() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=10 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    predictor.fit_all();

    let comparisons = predictor.compare_models();
    assert!(!comparisons.is_empty());
}

#[test]
fn test_prediction_reasonable() {
    let pred = Prediction {
        size: 1000,
        predicted: 100.0,
        lower_bound: 80.0,
        upper_bound: 120.0,
        confidence_level: 0.95,
        model_type: ModelType::Linear,
        is_extrapolation: false,
    };

    assert!(pred.is_reasonable());
    assert_eq!(pred.range_width(), 40.0);
}

#[test]
fn test_export_model() {
    let mut predictor = PerformancePredictor::new();

    for i in 1..=5 {
        predictor.add(i * 1000, i as f64 * 10.0, 1.0);
    }

    predictor.fit_linear();

    let export = predictor.export_model(ModelType::Linear).unwrap();
    assert!(export.contains("linear"));
    assert!(export.contains("r_squared"));
}
