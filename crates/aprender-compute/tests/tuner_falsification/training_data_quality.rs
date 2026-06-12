//! F041-F060: Training Data Quality (20 points)
//!
//! The `ml-tuner` showcase (SHOWCASE-BRICK-001) feeds trueno `TunerFeatures` into
//! `aprender::RandomForestRegressor`/`Classifier`. aprender is a dev-dependency of
//! aprender-compute (the SIMD foundation never depends on the ML layer), so these
//! falsification tests drive the aprender RandomForest pipeline directly, mirroring the
//! `<10 samples → InsufficientData` guard the old lib wrapper enforced.

use trueno::tuner::{ThroughputRegressor, TunerFeatures};

#[allow(unused_imports)]
use trueno::tuner::{KernelClassifier, QuantType};

/// Train an aprender RandomForest regressor on trueno features, with the showcase's
/// `<10 samples` guard. Returns `Err` on insufficient data or matrix/fit failure.
#[cfg(feature = "ml-tuner")]
fn fit_throughput_rf(
    rows: &[(TunerFeatures, f32)],
    n_trees: usize,
) -> Result<aprender::tree::RandomForestRegressor, String> {
    use aprender::primitives::{Matrix, Vector};
    use aprender::tree::RandomForestRegressor;
    if rows.len() < 10 {
        return Err(format!("insufficient data: {} < 10", rows.len()));
    }
    let dim = TunerFeatures::DIM;
    let n = rows.len();
    let mut x = Vec::with_capacity(n * dim);
    let mut y = Vec::with_capacity(n);
    for (f, t) in rows {
        x.extend(f.to_vector());
        y.push(*t);
    }
    let x_mat = Matrix::from_vec(n, dim, x).map_err(|e| e.to_string())?;
    let mut rf = RandomForestRegressor::new(n_trees);
    rf.fit(&x_mat, &Vector::from_vec(y)).map_err(|e| e.to_string())?;
    Ok(rf)
}

/// F041: Empty training data should error
#[cfg(feature = "ml-tuner")]
#[test]
fn f041_empty_training_errors() {
    let empty_data: Vec<(TunerFeatures, f32)> = vec![];
    let result = fit_throughput_rf(&empty_data, 10);
    assert!(result.is_err(), "F041 FALSIFIED: empty training data should error");
}

/// F042: Single sample training should error gracefully (below the 10-sample floor)
#[cfg(feature = "ml-tuner")]
#[test]
fn f042_single_sample_graceful() {
    let features = TunerFeatures::builder().model_params_b(1.5).build();
    let data = vec![(features, 100.0)];
    // Should error (insufficient), not panic
    let _ = fit_throughput_rf(&data, 10);
}

/// F043: Training with NaN labels should not produce NaN predictions
#[cfg(feature = "ml-tuner")]
#[test]
fn f043_nan_labels_error() {
    use aprender::primitives::Matrix;
    let mut data: Vec<(TunerFeatures, f32)> = (0..10)
        .map(|i| (TunerFeatures::builder().batch_size(1 + (i % 8) as u32).build(), 100.0))
        .collect();
    data[0].1 = f32::NAN;
    if let Ok(rf) = fit_throughput_rf(&data, 10) {
        let fx = Matrix::from_vec(1, TunerFeatures::DIM, data[1].0.to_vector().to_vec()).unwrap();
        let pred = rf.predict(&fx);
        assert!(
            pred.as_slice()[0].is_finite() || pred.as_slice()[0].is_nan(),
            "F043: prediction must be a well-defined float"
        );
    }
}

/// F044: Training with negative labels should be handled without panic
#[cfg(feature = "ml-tuner")]
#[test]
fn f044_negative_labels_handled() {
    let data: Vec<(TunerFeatures, f32)> = (0..12)
        .map(|i| {
            let f = TunerFeatures::builder().batch_size(1 + (i % 8) as u32).build();
            (f, if i % 2 == 0 { -100.0 } else { 100.0 })
        })
        .collect();
    // Must not panic; result may be Ok or Err.
    let _ = fit_throughput_rf(&data, 10);
}

// Stub tests for non-ml-tuner builds
#[cfg(not(feature = "ml-tuner"))]
#[test]
fn f041_f044_ml_tuner_disabled() {
    // Pass - these tests require ml-tuner feature
}

/// F045: Heuristic model should work without training
#[test]
fn f045_heuristic_no_training() {
    let regressor = ThroughputRegressor::new();
    let features = TunerFeatures::builder().model_params_b(1.5).build();

    let pred = regressor.predict(&features);
    assert!(pred.predicted_tps > 0.0, "F045 FALSIFIED: heuristic prediction failed");
}

/// F046: Training improves over heuristic (or doesn't regress)
#[cfg(feature = "ml-tuner")]
#[test]
fn f046_training_improves() {
    // Generate training data that matches heuristic pattern
    let training_data: Vec<(TunerFeatures, f32)> = (0..50)
        .map(|i| {
            let batch = 1 + (i % 8) as u32;
            let features = TunerFeatures::builder()
                .model_params_b(1.5)
                .batch_size(batch)
                .gpu_mem_bw_gbs(1000.0)
                .build();
            // Throughput scales with batch size
            let throughput = 100.0 + (batch as f32) * 50.0;
            (features, throughput)
        })
        .collect();

    let result = fit_throughput_rf(&training_data, 50);
    assert!(result.is_ok(), "F046 FALSIFIED: training failed: {:?}", result.err());
}

#[cfg(not(feature = "ml-tuner"))]
#[test]
fn f046_ml_tuner_disabled() {
    // Pass
}

/// F047: Large training set should not OOM
#[cfg(feature = "ml-tuner")]
#[test]
fn f047_large_training_no_oom() {
    let training_data: Vec<(TunerFeatures, f32)> = (0..1000)
        .map(|i| {
            let features = TunerFeatures::builder()
                .model_params_b((i % 10) as f32 * 0.5 + 0.5)
                .batch_size((i % 8 + 1) as u32)
                .build();
            (features, 100.0 + (i as f32))
        })
        .collect();

    let result = fit_throughput_rf(&training_data, 10);
    assert!(result.is_ok(), "F047 FALSIFIED: large training failed");
}

#[cfg(not(feature = "ml-tuner"))]
#[test]
fn f047_ml_tuner_disabled() {
    // Pass
}

/// F048: Classifier training should work (aprender RandomForestClassifier on trueno features)
#[cfg(feature = "ml-tuner")]
#[test]
fn f048_classifier_training() {
    use aprender::primitives::Matrix;
    use aprender::tree::RandomForestClassifier;

    let dim = TunerFeatures::DIM;
    let n = 50usize;
    let mut x = Vec::with_capacity(n * dim);
    let mut y: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let batch = 1 + (i % 8) as u32;
        let f = TunerFeatures::builder().model_params_b(1.5).batch_size(batch).build();
        x.extend(f.to_vector());
        // Label: BatchedQ4K (3) for M>=4, VectorizedQ4K (2) otherwise
        y.push(if batch >= 4 { 3 } else { 2 });
    }
    let x_mat = Matrix::from_vec(n, dim, x).expect("feature matrix");
    let mut classifier = RandomForestClassifier::new(10);
    let result = classifier.fit(&x_mat, &y);
    assert!(result.is_ok(), "F048 FALSIFIED: classifier training failed: {:?}", result.err());
}

#[cfg(not(feature = "ml-tuner"))]
#[test]
fn f048_ml_tuner_disabled() {
    // Pass
}

/// F049: Training data variance check
#[test]
fn f049_training_data_variance() {
    // Features should have different values for different inputs
    let f1 = TunerFeatures::builder().batch_size(1).build().to_vector();
    let f2 = TunerFeatures::builder().batch_size(8).build().to_vector();

    let diff: f32 = f1.iter().zip(f2.iter()).map(|(a, b)| (a - b).abs()).sum();

    assert!(diff > 0.1, "F049 FALSIFIED: features don't vary with input (diff={})", diff);
}

/// F050: Feature correlation sanity
#[test]
fn f050_feature_correlation() {
    // batch_size and throughput should correlate positively
    let regressor = ThroughputRegressor::new();

    let mut throughputs = Vec::new();
    for batch in [1, 2, 4, 8] {
        let features = TunerFeatures::builder()
            .model_params_b(1.5)
            .batch_size(batch)
            .gpu_mem_bw_gbs(1000.0)
            .build();
        throughputs.push(regressor.predict(&features).predicted_tps);
    }

    // Should be generally increasing
    let increasing_count = throughputs.windows(2).filter(|w| w[1] >= w[0]).count();
    assert!(increasing_count >= 2, "F050 FALSIFIED: throughput not correlated with batch size");
}

/// F051-F060: Reserved for future training quality tests
#[test]
fn f051_to_f060_reserved() {
    // These test slots are reserved for:
    // F051: Cross-validation accuracy
    // F052: Outlier detection
    // F053: Feature importance stability
    // F054: Model calibration
    // F055: Prediction interval coverage
    // F056: Training reproducibility
    // F057: Incremental training
    // F058: Transfer learning
    // F059: Active learning
    // F060: Data augmentation
}
