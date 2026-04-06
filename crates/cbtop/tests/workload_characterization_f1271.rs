#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Falsification Tests for PMAT-035: Workload Characterization System
//!
//! F1271-F1280: Workload characterization falsification tests

use cbtop::{
    ClassificationResult, RecommendedBackend, WorkloadCategory, WorkloadCharacterizer,
    WorkloadFeatures,
};

// =============================================================================
// F1271: Feature Extraction Tests
// =============================================================================

/// F1271.1: Feature extraction works (valid fingerprint)
#[test]
fn f1271_feature_extraction() {
    let characterizer = WorkloadCharacterizer::new();

    let features = characterizer.extract_features(
        1_000_000.0, // FLOPs
        100_000.0,   // Bytes accessed
        1_000_000,   // Memory footprint
        100_000,     // Working set
    );

    assert_eq!(features.arithmetic_intensity, 10.0); // 1M / 100K = 10
    assert_eq!(features.memory_footprint, 1_000_000);
}

/// F1271.2: Feature vector conversion
#[test]
fn f1271_feature_to_vec() {
    let features = WorkloadFeatures::new().with_intensity(5.0).with_compute_density(4.0);

    let vec = features.to_vec();
    assert_eq!(vec[0], 5.0); // arithmetic_intensity
    assert_eq!(vec[4], 4.0); // compute_density
}

// =============================================================================
// F1272: GEMM Classification Tests
// =============================================================================

/// F1272.1: GEMM classified correctly (compute-bound)
#[test]
fn f1272_gemm_classification() {
    let characterizer = WorkloadCharacterizer::new();

    // High intensity, high compute density, high reuse = GEMM
    let features = WorkloadFeatures::new()
        .with_intensity(45.0)
        .with_compute_density(7.0)
        .with_data_reuse(30.0)
        .with_branch_rate(0.01);

    let result = characterizer.classify(&features);
    assert_eq!(result.category, WorkloadCategory::Gemm);
}

/// F1272.2: GEMM is compute-bound
#[test]
fn f1272_gemm_is_compute_bound() {
    assert!(WorkloadCategory::Gemm.is_compute_bound());
    assert!(!WorkloadCategory::Gemm.is_memory_bound());
}

// =============================================================================
// F1273: Bandwidth Classification Tests
// =============================================================================

/// F1273.1: Bandwidth classified (memory-bound detected)
#[test]
fn f1273_bandwidth_classification() {
    let characterizer = WorkloadCharacterizer::new();

    // Low intensity, sequential access = Bandwidth
    let features = WorkloadFeatures::new()
        .with_intensity(0.2)
        .with_compute_density(0.5)
        .with_access_pattern(1.0)
        .with_data_reuse(1.0);

    let result = characterizer.classify(&features);
    assert_eq!(result.category, WorkloadCategory::Bandwidth);
}

/// F1273.2: Bandwidth is memory-bound
#[test]
fn f1273_bandwidth_is_memory_bound() {
    assert!(WorkloadCategory::Bandwidth.is_memory_bound());
    assert!(!WorkloadCategory::Bandwidth.is_compute_bound());
}

// =============================================================================
// F1274: Attention Classification Tests
// =============================================================================

/// F1274.1: Attention classified (mixed workload)
#[test]
fn f1274_attention_classification() {
    let characterizer = WorkloadCharacterizer::new();

    // Medium intensity, mixed access = Attention
    let features = WorkloadFeatures::new()
        .with_intensity(4.5)
        .with_compute_density(3.8)
        .with_access_pattern(0.6)
        .with_data_reuse(4.0);

    let result = characterizer.classify(&features);
    assert_eq!(result.category, WorkloadCategory::Attention);
}

/// F1274.2: Attention typical intensity range
#[test]
fn f1274_attention_intensity_range() {
    let (low, high) = WorkloadCategory::Attention.typical_intensity_range();
    assert!(low < high);
    assert!(low >= 1.0);
    assert!(high <= 20.0);
}

// =============================================================================
// F1275: Similarity Metric Tests
// =============================================================================

/// F1275.1: Similarity metric valid (0-1 range)
#[test]
fn f1275_similarity_range() {
    let characterizer = WorkloadCharacterizer::new();

    let a = WorkloadFeatures::new().with_intensity(10.0);
    let b = WorkloadFeatures::new().with_intensity(20.0);

    let sim = characterizer.workload_similarity(&a, &b);
    assert!((0.0..=1.0).contains(&sim));
}

/// F1275.2: Identical features have similarity ~1
#[test]
fn f1275_identical_similarity() {
    let characterizer = WorkloadCharacterizer::new();

    let a = WorkloadFeatures::new().with_intensity(10.0);
    let b = WorkloadFeatures::new().with_intensity(10.0);

    let sim = characterizer.workload_similarity(&a, &b);
    assert!(sim > 0.99);
}

/// F1275.3: Cosine similarity in valid range
#[test]
fn f1275_cosine_similarity() {
    let a = WorkloadFeatures::new().with_intensity(10.0).with_compute_density(5.0);
    let b = WorkloadFeatures::new().with_intensity(20.0).with_compute_density(10.0);

    let sim = a.cosine_similarity(&b);
    assert!((-1.0..=1.0).contains(&sim));
}

// =============================================================================
// F1276: Unknown Workload Tests
// =============================================================================

/// F1276.1: Unknown workload handled (nearest match)
#[test]
fn f1276_unknown_workload() {
    let characterizer = WorkloadCharacterizer::new();

    // Features that don't match any prototype well
    let features = WorkloadFeatures::new()
        .with_intensity(1000.0) // Very unusual
        .with_compute_density(100.0);

    let result = characterizer.classify(&features);
    // Should still return a classification
    assert!(result.category != WorkloadCategory::Unknown || result.confidence < 0.5);
}

/// F1276.2: Classification confidence provided
#[test]
fn f1276_classification_confidence() {
    let characterizer = WorkloadCharacterizer::new();

    let features = WorkloadFeatures::new().with_intensity(50.0);
    let result = characterizer.classify(&features);

    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

// =============================================================================
// F1277: Backend Recommendation Tests
// =============================================================================

/// F1277.1: Valid backend returned (small size)
#[test]
fn f1277_backend_small_size() {
    let characterizer = WorkloadCharacterizer::new();

    let backend = characterizer.recommend_backend(WorkloadCategory::Gemm, 1000);
    assert_eq!(backend, RecommendedBackend::CpuSimd);
}

/// F1277.2: Valid backend returned (large size)
#[test]
fn f1277_backend_large_size() {
    let characterizer = WorkloadCharacterizer::new();

    let backend = characterizer.recommend_backend(WorkloadCategory::Gemm, 1_000_000);
    assert_eq!(backend, RecommendedBackend::Gpu);
}

/// F1277.3: Backend names correct
#[test]
fn f1277_backend_names() {
    assert_eq!(RecommendedBackend::CpuSimd.name(), "cpu_simd");
    assert_eq!(RecommendedBackend::Gpu.name(), "gpu");
    assert_eq!(RecommendedBackend::Either.name(), "either");
}

// =============================================================================
// F1278: Size Threshold Tests
// =============================================================================

/// F1278.1: Crossover point found
#[test]
fn f1278_crossover_point() {
    let characterizer = WorkloadCharacterizer::new();

    let crossover = characterizer.predict_crossover(WorkloadCategory::Gemm);
    assert!(crossover.is_some());
    assert!(crossover.unwrap() > 0);
}

/// F1278.2: GPU crossover size in result
#[test]
fn f1278_gpu_crossover_in_result() {
    let characterizer = WorkloadCharacterizer::new();
    let features = WorkloadFeatures::new().with_intensity(50.0);

    let result = characterizer.classify(&features);
    assert!(result.gpu_crossover_size.is_some());
}

// =============================================================================
// F1279: Feature Normalization Tests
// =============================================================================

/// F1279.1: Z-score normalized
#[test]
fn f1279_feature_normalization() {
    let features = WorkloadFeatures::new().with_intensity(10.0).with_compute_density(5.0);

    let means = vec![10.0, 0.0, 0.0, 0.5, 5.0, 0.0, 1.0];
    let stds = vec![2.0, 1.0, 1.0, 0.2, 1.0, 0.1, 0.5];

    let normalized = features.normalize(&means, &stds);

    // (10 - 10) / 2 = 0 for intensity
    assert!((normalized[0] - 0.0).abs() < 0.001);
    // (5 - 5) / 1 = 0 for compute_density
    assert!((normalized[4] - 0.0).abs() < 0.001);
}

// =============================================================================
// F1280: Classification Confidence Tests
// =============================================================================

/// F1280.1: 0-1 probability
#[test]
fn f1280_confidence_probability() {
    let characterizer = WorkloadCharacterizer::new();

    let features = WorkloadFeatures::new().with_intensity(50.0);
    let result = characterizer.classify(&features);

    assert!(result.confidence >= 0.0);
    assert!(result.confidence <= 1.0);
}

/// F1280.2: is_confident helper
#[test]
fn f1280_is_confident() {
    let result = ClassificationResult {
        category: WorkloadCategory::Gemm,
        confidence: 0.8,
        distance: 5.0,
        recommended_backend: RecommendedBackend::Gpu,
        gpu_crossover_size: Some(10000),
    };

    assert!(result.is_confident());

    let low_conf = ClassificationResult { confidence: 0.5, ..result };
    assert!(!low_conf.is_confident());
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test workload category names
#[test]
fn test_category_names() {
    assert_eq!(WorkloadCategory::Gemm.name(), "gemm");
    assert_eq!(WorkloadCategory::Bandwidth.name(), "bandwidth");
    assert_eq!(WorkloadCategory::Attention.name(), "attention");
    assert_eq!(WorkloadCategory::Conv2d.name(), "conv2d");
    assert_eq!(WorkloadCategory::Elementwise.name(), "elementwise");
    assert_eq!(WorkloadCategory::Reduction.name(), "reduction");
    assert_eq!(WorkloadCategory::Unknown.name(), "unknown");
}

/// Test feature distance
#[test]
fn test_feature_distance() {
    let a = WorkloadFeatures::new().with_intensity(10.0);
    let b = WorkloadFeatures::new().with_intensity(20.0);

    let dist = a.distance(&b);
    assert!(dist > 0.0);
}

/// Test add custom prototype
#[test]
fn test_add_prototype() {
    let mut characterizer = WorkloadCharacterizer::new();
    let initial_count = characterizer.get_prototypes().len();

    characterizer
        .add_prototype(WorkloadCategory::Unknown, WorkloadFeatures::new().with_intensity(100.0));

    assert_eq!(characterizer.get_prototypes().len(), initial_count + 1);
}

/// Test conv2d classification
#[test]
fn test_conv2d_classification() {
    let characterizer = WorkloadCharacterizer::new();

    let features = WorkloadFeatures::new()
        .with_intensity(18.0)
        .with_compute_density(5.5)
        .with_access_pattern(0.7)
        .with_data_reuse(8.0);

    let result = characterizer.classify(&features);
    assert_eq!(result.category, WorkloadCategory::Conv2d);
}

/// Test elementwise classification
#[test]
fn test_elementwise_classification() {
    let characterizer = WorkloadCharacterizer::new();

    let features = WorkloadFeatures::new()
        .with_intensity(0.12)
        .with_compute_density(1.0)
        .with_access_pattern(1.0)
        .with_data_reuse(1.0);

    let result = characterizer.classify(&features);
    assert_eq!(result.category, WorkloadCategory::Elementwise);
}
