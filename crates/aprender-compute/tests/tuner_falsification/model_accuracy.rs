//! F001-F020: Model Accuracy (20 points)

use trueno::tuner::{
    BottleneckClass, BrickTuner, KernelClassifier, KernelType, QuantType, ThroughputRegressor,
    TunerFeatures,
};

/// F001: Throughput predictions must be positive
#[test]
fn f001_throughput_positive() {
    let regressor = ThroughputRegressor::new();
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .batch_size(4)
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(1000.0)
        .build();

    let pred = regressor.predict(&features);
    assert!(
        pred.predicted_tps > 0.0,
        "F001 FALSIFIED: throughput must be positive, got {}",
        pred.predicted_tps
    );
}

/// F002: Throughput predictions must have valid confidence
#[test]
fn f002_throughput_confidence_valid() {
    let regressor = ThroughputRegressor::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let pred = regressor.predict(&features);
    assert!(
        (0.0..=1.0).contains(&pred.confidence),
        "F002 FALSIFIED: confidence must be in [0,1], got {}",
        pred.confidence
    );
}

/// F003: Roofline bound must be respected
#[test]
fn f003_roofline_bound_respected() {
    let regressor = ThroughputRegressor::new();

    // 7B model on 1000 GB/s GPU with Q4K (0.5625 bytes/param)
    // Roofline: 1000 GB/s / (7B * 0.5625) = 254 tok/s theoretical max
    let features = TunerFeatures::builder()
        .model_params_b(7.0)
        .batch_size(1)
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(1000.0)
        .build();

    let pred = regressor.predict(&features);
    let roofline_max = 1000.0 / (7.0 * 0.5625);

    assert!(
        pred.predicted_tps <= roofline_max * 1.1, // 10% tolerance
        "F003 FALSIFIED: prediction {} exceeds roofline {} (with 10% tolerance)",
        pred.predicted_tps,
        roofline_max
    );
}

/// F004: Larger batch size should increase throughput
#[test]
fn f004_batch_size_monotonic() {
    let regressor = ThroughputRegressor::new();

    let small_batch =
        TunerFeatures::builder().model_params_b(1.5).batch_size(1).gpu_mem_bw_gbs(1000.0).build();

    let large_batch =
        TunerFeatures::builder().model_params_b(1.5).batch_size(8).gpu_mem_bw_gbs(1000.0).build();

    let small_pred = regressor.predict(&small_batch);
    let large_pred = regressor.predict(&large_batch);

    assert!(
        large_pred.predicted_tps >= small_pred.predicted_tps,
        "F004 FALSIFIED: batch=8 ({}) should be >= batch=1 ({})",
        large_pred.predicted_tps,
        small_pred.predicted_tps
    );
}

/// F005: Kernel classifier must return valid kernel
#[test]
fn f005_kernel_selection_valid() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .batch_size(4)
        .quant_type(QuantType::Q4K)
        .build();

    let rec = classifier.predict(&features);
    assert!(
        rec.confidence >= 0.0 && rec.confidence <= 1.0,
        "F005 FALSIFIED: kernel confidence {} out of range",
        rec.confidence
    );
}

/// F006: Kernel selection should prefer BatchedQ4K for large batches
#[test]
fn f006_kernel_batch_preference() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .batch_size(8)
        .quant_type(QuantType::Q4K)
        .build();

    let rec = classifier.predict(&features);
    assert!(
        rec.top_kernel == KernelType::BatchedQ4K || rec.top_kernel == KernelType::VectorizedQ4K,
        "F006 FALSIFIED: expected BatchedQ4K or VectorizedQ4K for batch=8, got {:?}",
        rec.top_kernel
    );
}

/// F007: Kernel selection should prefer single-sequence kernels for M=1
#[test]
fn f007_kernel_single_preference() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .batch_size(1)
        .quant_type(QuantType::Q4K)
        .build();

    let rec = classifier.predict(&features);
    // For M=1, we expect non-batched kernels
    assert!(
        rec.top_kernel == KernelType::VectorizedQ4K
            || rec.top_kernel == KernelType::TiledQ4K
            || rec.top_kernel == KernelType::CoalescedQ4K,
        "F007 FALSIFIED: expected single-sequence kernel for batch=1, got {:?}",
        rec.top_kernel
    );
}

/// F008: Predictions must be deterministic
#[test]
fn f008_prediction_deterministic() {
    let regressor = ThroughputRegressor::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let pred1 = regressor.predict(&features);
    let pred2 = regressor.predict(&features);

    assert!(
        (pred1.predicted_tps - pred2.predicted_tps).abs() < 0.001,
        "F008 FALSIFIED: predictions not deterministic: {} vs {}",
        pred1.predicted_tps,
        pred2.predicted_tps
    );
}

/// F009: Classifier predictions must be deterministic
#[test]
fn f009_classifier_deterministic() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let rec1 = classifier.predict(&features);
    let rec2 = classifier.predict(&features);

    assert_eq!(
        rec1.top_kernel, rec2.top_kernel,
        "F009 FALSIFIED: classifier not deterministic: {:?} vs {:?}",
        rec1.top_kernel, rec2.top_kernel
    );
}

/// F010: Prediction latency must be < 1ms
#[test]
fn f010_prediction_latency() {
    let regressor = ThroughputRegressor::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = regressor.predict(&features);
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() / 100;

    assert!(avg_us < 1000, "F010 FALSIFIED: prediction latency {} us >= 1ms", avg_us);
}

/// F011: Top features must be non-empty
#[test]
fn f011_top_features_present() {
    let regressor = ThroughputRegressor::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let pred = regressor.predict(&features);
    assert!(!pred.top_features.is_empty(), "F011 FALSIFIED: top_features must not be empty");
}

/// F012: Top features importances must sum to <= 1.0
#[test]
fn f012_feature_importance_sum() {
    let regressor = ThroughputRegressor::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let pred = regressor.predict(&features);
    let sum: f32 = pred.top_features.iter().map(|(_, v)| v).sum();

    assert!(sum <= 1.0 + 0.001, "F012 FALSIFIED: feature importance sum {} > 1.0", sum);
}

/// F013: Alternatives must have decreasing confidence
#[test]
fn f013_alternatives_ordered() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let rec = classifier.predict(&features);
    for i in 1..rec.alternatives.len() {
        assert!(
            rec.alternatives[i].1 <= rec.alternatives[i - 1].1,
            "F013 FALSIFIED: alternatives not sorted by confidence at index {}",
            i
        );
    }
}

/// F014: No catastrophic failures (prediction > 2x expected)
#[test]
fn f014_no_catastrophic_overpredict() {
    let regressor = ThroughputRegressor::new();

    // 32B model should not predict > 1000 tok/s on any reasonable GPU
    let features = TunerFeatures::builder()
        .model_params_b(32.0)
        .batch_size(1)
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(1000.0)
        .build();

    let pred = regressor.predict(&features);
    let sane_max = 500.0; // 32B at ~60 tok/s baseline, 500 is generous

    assert!(
        pred.predicted_tps <= sane_max,
        "F014 FALSIFIED: 32B prediction {} > {} (catastrophic)",
        pred.predicted_tps,
        sane_max
    );
}

/// F015: Smaller models should predict higher throughput
#[test]
fn f015_model_size_inverse() {
    let regressor = ThroughputRegressor::new();

    let small_model =
        TunerFeatures::builder().model_params_b(0.5).batch_size(4).gpu_mem_bw_gbs(1000.0).build();

    let large_model =
        TunerFeatures::builder().model_params_b(7.0).batch_size(4).gpu_mem_bw_gbs(1000.0).build();

    let small_pred = regressor.predict(&small_model);
    let large_pred = regressor.predict(&large_model);

    assert!(
        small_pred.predicted_tps >= large_pred.predicted_tps,
        "F015 FALSIFIED: 0.5B ({}) should be >= 7B ({})",
        small_pred.predicted_tps,
        large_pred.predicted_tps
    );
}

/// F016: Higher memory bandwidth should increase throughput
#[test]
fn f016_bandwidth_monotonic() {
    let regressor = ThroughputRegressor::new();

    let slow_gpu =
        TunerFeatures::builder().model_params_b(1.5).batch_size(4).gpu_mem_bw_gbs(500.0).build();

    let fast_gpu =
        TunerFeatures::builder().model_params_b(1.5).batch_size(4).gpu_mem_bw_gbs(1000.0).build();

    let slow_pred = regressor.predict(&slow_gpu);
    let fast_pred = regressor.predict(&fast_gpu);

    assert!(
        fast_pred.predicted_tps >= slow_pred.predicted_tps,
        "F016 FALSIFIED: 1000 GB/s ({}) should be >= 500 GB/s ({})",
        fast_pred.predicted_tps,
        slow_pred.predicted_tps
    );
}

/// F017: Full tuner recommendation must be consistent
#[test]
fn f017_tuner_consistency() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let rec = tuner.recommend(&features);

    assert!(rec.throughput.predicted_tps > 0.0);
    assert!(rec.confidence_overall >= 0.0 && rec.confidence_overall <= 1.0);
}

/// F018: Suggested experiments must not be empty
#[test]
fn f018_experiments_present() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let rec = tuner.recommend(&features);
    assert!(
        !rec.suggested_experiments.is_empty(),
        "F018 FALSIFIED: suggested_experiments must not be empty"
    );
}

/// F019: Model version must be valid semver
#[test]
fn f019_model_version_valid() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().model_params_b(1.5).build();

    let rec = tuner.recommend(&features);
    assert!(
        rec.model_version.contains('.'),
        "F019 FALSIFIED: model_version '{}' is not semver",
        rec.model_version
    );
}

/// F020: Bottleneck classification must be valid
#[test]
fn f020_bottleneck_valid() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let rec = tuner.recommend(&features);
    // Bottleneck should be one of the valid types
    let valid = matches!(
        rec.bottleneck.class,
        BottleneckClass::Unknown
            | BottleneckClass::MemoryBound
            | BottleneckClass::ComputeBound
            | BottleneckClass::LaunchBound
            | BottleneckClass::AttentionBound
    );
    assert!(valid, "F020 FALSIFIED: invalid bottleneck {:?}", rec.bottleneck.class);
}
