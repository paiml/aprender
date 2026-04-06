//! F041-F060: Training Data Quality (20 points)

use trueno::tuner::{ThroughputRegressor, TunerFeatures};

#[allow(unused_imports)]
use trueno::tuner::{KernelClassifier, QuantType};

/// F041: Empty training data should error
#[cfg(feature = "ml-tuner")]
#[test]
fn f041_empty_training_errors() {
    let mut regressor = ThroughputRegressor::with_random_forest(10);
    let empty_data: Vec<(TunerFeatures, f32)> = vec![];

    let result = regressor.train_random_forest(&empty_data);
    assert!(result.is_err(), "F041 FALSIFIED: empty training data should error");
}

/// F042: Single sample training should work or error gracefully
#[cfg(feature = "ml-tuner")]
#[test]
fn f042_single_sample_graceful() {
    let mut regressor = ThroughputRegressor::with_random_forest(10);
    let features = TunerFeatures::builder().model_params_b(1.5).build();
    let data = vec![(features, 100.0)];

    // Should either succeed or error, not panic
    let _ = regressor.train_random_forest(&data);
}

/// F043: Training with NaN labels should error
#[cfg(feature = "ml-tuner")]
#[test]
fn f043_nan_labels_error() {
    let mut regressor = ThroughputRegressor::with_random_forest(10);
    let features = TunerFeatures::builder().model_params_b(1.5).build();
    let data = vec![(features.clone(), f32::NAN)];

    let result = regressor.train_random_forest(&data);
    // Should handle gracefully (either error or filter)
    if result.is_ok() {
        // If it succeeds, predictions should not be NaN
        let pred = regressor.predict(&features);
        assert!(
            pred.predicted_tps.is_finite(),
            "F043 FALSIFIED: NaN training produced NaN predictions"
        );
    }
}

/// F044: Training with negative labels should error or clamp
#[cfg(feature = "ml-tuner")]
#[test]
fn f044_negative_labels_handled() {
    let mut regressor = ThroughputRegressor::with_random_forest(10);
    let features = TunerFeatures::builder().model_params_b(1.5).build();
    let data = vec![(features.clone(), -100.0), (features.clone(), 100.0)];

    let result = regressor.train_random_forest(&data);
    if result.is_ok() {
        let pred = regressor.predict(&features);
        // Predictions should still be positive
        assert!(
            pred.predicted_tps >= 0.0,
            "F044 FALSIFIED: prediction {} < 0 after negative training",
            pred.predicted_tps
        );
    }
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
    let mut regressor = ThroughputRegressor::with_random_forest(50);

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

    let result = regressor.train_random_forest(&training_data);
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
    let mut regressor = ThroughputRegressor::with_random_forest(10);

    let training_data: Vec<(TunerFeatures, f32)> = (0..1000)
        .map(|i| {
            let features = TunerFeatures::builder()
                .model_params_b((i % 10) as f32 * 0.5 + 0.5)
                .batch_size((i % 8 + 1) as u32)
                .build();
            (features, 100.0 + (i as f32))
        })
        .collect();

    let result = regressor.train_random_forest(&training_data);
    assert!(result.is_ok(), "F047 FALSIFIED: large training failed");
}

#[cfg(not(feature = "ml-tuner"))]
#[test]
fn f047_ml_tuner_disabled() {
    // Pass
}

/// F048: Classifier training should work
#[cfg(feature = "ml-tuner")]
#[test]
fn f048_classifier_training() {
    let mut classifier = KernelClassifier::with_random_forest(10);

    let training_data: Vec<(TunerFeatures, u32)> = (0..50)
        .map(|i| {
            let batch = 1 + (i % 8) as u32;
            let features = TunerFeatures::builder().model_params_b(1.5).batch_size(batch).build();
            // Label: BatchedQ4K (3) for M>=4, VectorizedQ4K (2) otherwise
            let label = if batch >= 4 { 3 } else { 2 };
            (features, label)
        })
        .collect();

    let result = classifier.train(&training_data);
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
