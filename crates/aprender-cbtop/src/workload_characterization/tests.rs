use super::*;

#[test]
fn test_workload_category_names() {
    assert_eq!(WorkloadCategory::Gemm.name(), "gemm");
    assert_eq!(WorkloadCategory::Bandwidth.name(), "bandwidth");
    assert_eq!(WorkloadCategory::Attention.name(), "attention");
}

#[test]
fn test_workload_category_bound() {
    assert!(WorkloadCategory::Gemm.is_compute_bound());
    assert!(WorkloadCategory::Bandwidth.is_memory_bound());
    assert!(!WorkloadCategory::Attention.is_compute_bound());
}

#[test]
fn test_feature_creation() {
    let features = WorkloadFeatures::new()
        .with_intensity(10.0)
        .with_memory(1024, 512)
        .with_access_pattern(0.8);

    assert_eq!(features.arithmetic_intensity, 10.0);
    assert_eq!(features.memory_footprint, 1024);
    assert_eq!(features.access_pattern, 0.8);
}

#[test]
fn test_feature_to_vec() {
    let features = WorkloadFeatures::new().with_intensity(5.0);
    let vec = features.to_vec();
    assert_eq!(vec[0], 5.0);
}

#[test]
fn test_cosine_similarity() {
    let a = WorkloadFeatures::new()
        .with_intensity(10.0)
        .with_compute_density(5.0);
    let b = WorkloadFeatures::new()
        .with_intensity(20.0)
        .with_compute_density(10.0);

    let sim = a.cosine_similarity(&b);
    assert!(sim > 0.9); // Same direction, similar features
}

#[test]
fn test_characterizer_creation() {
    let characterizer = WorkloadCharacterizer::new();
    assert!(!characterizer.prototypes.is_empty());
}

#[test]
fn test_gemm_classification() {
    let characterizer = WorkloadCharacterizer::new();
    let features = WorkloadFeatures::new()
        .with_intensity(40.0)
        .with_compute_density(7.0)
        .with_data_reuse(30.0);

    let result = characterizer.classify(&features);
    assert_eq!(result.category, WorkloadCategory::Gemm);
}

#[test]
fn test_bandwidth_classification() {
    let characterizer = WorkloadCharacterizer::new();
    let features = WorkloadFeatures::new()
        .with_intensity(0.2)
        .with_compute_density(0.5)
        .with_access_pattern(1.0);

    let result = characterizer.classify(&features);
    assert_eq!(result.category, WorkloadCategory::Bandwidth);
}

#[test]
fn test_backend_recommendation() {
    let characterizer = WorkloadCharacterizer::new();

    // Small GEMM: CPU
    let backend = characterizer.recommend_backend(WorkloadCategory::Gemm, 1000);
    assert_eq!(backend, RecommendedBackend::CpuSimd);

    // Large GEMM: GPU
    let backend = characterizer.recommend_backend(WorkloadCategory::Gemm, 1_000_000);
    assert_eq!(backend, RecommendedBackend::Gpu);
}

#[test]
fn test_workload_similarity() {
    let characterizer = WorkloadCharacterizer::new();
    let a = WorkloadFeatures::new().with_intensity(10.0);
    let b = WorkloadFeatures::new().with_intensity(10.0);

    let sim = characterizer.workload_similarity(&a, &b);
    assert!(sim > 0.99); // Identical features
}
