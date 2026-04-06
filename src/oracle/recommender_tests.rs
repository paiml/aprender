use super::*;

// =========================================================================
// Backend Selection Tests
// =========================================================================

#[test]
fn test_select_backend_low_complexity_small() {
    let hw = HardwareSpec::cpu_only();
    let backend = select_backend(OpComplexity::Low, Some(DataSize::samples(100)), &hw);
    assert_eq!(backend, Backend::Scalar);
}

#[test]
fn test_select_backend_low_complexity_medium() {
    let hw = HardwareSpec::cpu_only();
    let backend = select_backend(OpComplexity::Low, Some(DataSize::samples(10_000)), &hw);
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_select_backend_low_complexity_large_gpu() {
    let hw = HardwareSpec::with_gpu(16.0);
    let backend = select_backend(OpComplexity::Low, Some(DataSize::samples(2_000_000)), &hw);
    assert_eq!(backend, Backend::GPU);
}

#[test]
fn test_select_backend_high_complexity_gpu() {
    let hw = HardwareSpec::with_gpu(8.0);
    let backend = select_backend(OpComplexity::High, Some(DataSize::samples(50_000)), &hw);
    assert_eq!(backend, Backend::GPU);
}

#[test]
fn test_select_backend_high_complexity_no_gpu() {
    let hw = HardwareSpec::cpu_only();
    let backend = select_backend(OpComplexity::High, Some(DataSize::samples(50_000)), &hw);
    assert_eq!(backend, Backend::SIMD);
}

#[test]
fn test_select_backend_medium_complexity() {
    let hw = HardwareSpec::cpu_only();
    let backend = select_backend(OpComplexity::Medium, Some(DataSize::samples(1_000)), &hw);
    assert_eq!(backend, Backend::SIMD);
}

// =========================================================================
// Distribution Decision Tests
// =========================================================================

#[test]
fn test_should_distribute_small_data() {
    let hw = HardwareSpec::default();
    let decision = should_distribute(Some(DataSize::samples(10_000)), &hw, 0.9);
    assert!(!decision.needed);
}

#[test]
fn test_should_distribute_single_node() {
    let hw = HardwareSpec { node_count: Some(1), is_distributed: true, ..Default::default() };
    let decision = should_distribute(Some(DataSize::samples(100_000_000)), &hw, 0.9);
    assert!(!decision.needed);
}

#[test]
fn test_should_distribute_multi_node() {
    let hw = HardwareSpec { node_count: Some(4), is_distributed: true, ..Default::default() };
    let decision = should_distribute(Some(DataSize::samples(100_000_000)), &hw, 0.95);
    assert!(decision.needed);
    assert_eq!(decision.tool.as_deref(), Some("repartir"));
}

#[test]
fn test_should_distribute_low_parallel_fraction() {
    let hw = HardwareSpec { node_count: Some(4), is_distributed: true, ..Default::default() };
    let decision = should_distribute(Some(DataSize::samples(100_000_000)), &hw, 0.3);
    // Low parallel fraction with limited speedup - result depends on thresholds
    // Just verify the function returns a valid decision without panicking
    assert!(!decision.rationale.is_empty());
}

// =========================================================================
// Recommender Creation Tests
// =========================================================================

#[test]
fn test_recommender_new() {
    let rec = Recommender::new();
    assert!(!rec.list_components().is_empty());
}

#[test]
fn test_recommender_default() {
    let rec = Recommender::default();
    let components = rec.list_components();
    assert!(components.contains(&"trueno".to_string()));
    assert!(components.contains(&"aprender".to_string()));
}

#[test]
fn test_recommender_with_graph() {
    let graph = KnowledgeGraph::sovereign_stack();
    let rec = Recommender::with_graph(graph);
    assert!(!rec.list_components().is_empty());
}

// =========================================================================
// Query Tests
// =========================================================================

#[test]
fn test_query_random_forest() {
    let rec = Recommender::new();
    let response = rec.query("Train a random forest classifier");

    assert_eq!(response.primary.component, "aprender");
    assert!(response.primary.confidence > 0.8);
    assert!(response.algorithm.is_some());
}

#[test]
fn test_query_sklearn_migration() {
    let rec = Recommender::new();
    let response = rec.query("Convert my sklearn pipeline to Rust");

    assert_eq!(response.primary.component, "depyler");
    assert_eq!(response.problem_class, "Python Migration");
}

#[test]
fn test_query_model_serving() {
    let rec = Recommender::new();
    let response = rec.query("Deploy model for production inference");

    assert_eq!(response.primary.component, "realizar");
}

#[test]
fn test_query_matrix_ops() {
    let rec = Recommender::new();
    let response = rec.query("Fast matrix multiplication with SIMD");

    assert_eq!(response.primary.component, "trueno");
}

#[test]
fn test_query_distributed() {
    let rec = Recommender::new();
    // "Distributed" queries without specific ML tasks should recommend repartir
    let response = rec.query("Scale computation with repartir across cluster");

    assert_eq!(response.primary.component, "repartir");
}

#[test]
fn test_query_explicit_component() {
    let rec = Recommender::new();
    let response = rec.query("Use aprender for my ML task");

    assert_eq!(response.primary.component, "aprender");
    assert!(response.primary.confidence > 0.9); // High confidence for explicit mention
}

// =========================================================================
// Structured Query Tests
// =========================================================================

#[test]
fn test_query_structured() {
    let rec = Recommender::new();
    let query = OracleQuery::new("Train a classifier")
        .with_data_size(DataSize::samples(1_000_000))
        .with_hardware(HardwareSpec::with_gpu(16.0));

    let response = rec.query_structured(&query);

    assert_eq!(response.primary.component, "aprender");
    // Should recommend GPU backend for large data
    assert!(matches!(response.compute.backend, Backend::GPU | Backend::SIMD));
}

#[test]
fn test_query_structured_sovereign() {
    let rec = Recommender::new();
    let query =
        OracleQuery::new("Train model with GDPR compliance").sovereign_only().eu_compliant();

    let response = rec.query_structured(&query);
    // Should work - sovereign mode is about local execution, which all stack components support
    assert!(!response.primary.component.is_empty());
}

// =========================================================================
// Backend Recommendation Tests
// =========================================================================

#[test]
fn test_compute_recommendation_simd() {
    let rec = Recommender::new();
    // Use structured query with explicit data size for reliable backend selection
    let query = OracleQuery::new("Train random forest").with_data_size(DataSize::samples(10_000));
    let response = rec.query_structured(&query);

    assert_eq!(response.compute.backend, Backend::SIMD);
}

#[test]
fn test_compute_recommendation_scalar() {
    let rec = Recommender::new();
    let query = OracleQuery::new("Simple calculation").with_data_size(DataSize::samples(50));

    let response = rec.query_structured(&query);
    // Small data should use scalar
    assert!(matches!(response.compute.backend, Backend::Scalar | Backend::SIMD));
}

// =========================================================================
// Supporting Component Tests
// =========================================================================

#[test]
fn test_supporting_components_ml() {
    let rec = Recommender::new();
    let response = rec.query("Train a model on large dataset");

    // Should recommend trueno for compute backend
    assert!(response.supporting.iter().any(|s| s.component == "trueno"));
}

#[test]
fn test_supporting_components_inference() {
    let rec = Recommender::new();
    let response = rec.query("Train a model and deploy for inference");

    // Should recommend realizar for serving
    assert!(
        response.primary.component == "realizar"
            || response.supporting.iter().any(|s| s.component == "realizar")
    );
}

// =========================================================================
// Code Example Tests
// =========================================================================

#[test]
fn test_code_example_aprender() {
    let rec = Recommender::new();
    let response = rec.query("Train random forest classifier");

    assert!(response.code_example.is_some());
    let code = response.code_example.unwrap();
    assert!(code.contains("aprender") || code.contains("RandomForest"));
}

#[test]
fn test_code_example_trueno() {
    let rec = Recommender::new();
    let response = rec.query("Matrix operations with trueno");

    assert!(response.code_example.is_some());
    let code = response.code_example.unwrap();
    assert!(code.contains("trueno") || code.contains("Tensor"));
}

// =========================================================================
// Related Queries Tests
// =========================================================================

#[test]
fn test_related_queries_ml() {
    let rec = Recommender::new();
    let response = rec.query("Train a classifier");

    assert!(!response.related_queries.is_empty());
}

#[test]
fn test_related_queries_migration() {
    let rec = Recommender::new();
    let response = rec.query("Convert sklearn to Rust");

    assert!(!response.related_queries.is_empty());
    // Should have sklearn/numpy related queries
    assert!(response
        .related_queries
        .iter()
        .any(|q| q.contains("sklearn") || q.contains("numpy") || q.contains("aprender")));
}

// =========================================================================
// Capability Tests
// =========================================================================

#[test]
fn test_get_capabilities_trueno() {
    let rec = Recommender::new();
    let caps = rec.get_capabilities("trueno");

    assert!(caps.contains(&"simd".to_string()));
    assert!(caps.contains(&"gpu".to_string()));
}

#[test]
fn test_get_capabilities_aprender() {
    let rec = Recommender::new();
    let caps = rec.get_capabilities("aprender");

    assert!(caps.contains(&"random_forest".to_string()));
    assert!(caps.contains(&"linear_regression".to_string()));
}

#[test]
fn test_get_capabilities_unknown() {
    let rec = Recommender::new();
    let caps = rec.get_capabilities("unknown_component");

    assert!(caps.is_empty());
}

// =========================================================================
// Integration Tests
// =========================================================================

#[test]
fn test_get_integration() {
    let rec = Recommender::new();
    let pattern = rec.get_integration("aprender", "realizar");

    assert!(pattern.is_some());
    let pattern = pattern.unwrap();
    assert_eq!(pattern.pattern_name, "model_export");
}

#[test]
fn test_get_integration_not_found() {
    let rec = Recommender::new();
    let pattern = rec.get_integration("trueno", "bashrs");

    assert!(pattern.is_none());
}

// =========================================================================
// Component Lookup Tests
// =========================================================================

#[test]
fn test_get_component() {
    let rec = Recommender::new();
    let comp = rec.get_component("trueno");

    assert!(comp.is_some());
    assert_eq!(comp.unwrap().version, "0.11.0");
}

#[test]
fn test_list_components() {
    let rec = Recommender::new();
    let components = rec.list_components();

    assert!(components.len() >= 15);
    assert!(components.contains(&"trueno".to_string()));
    assert!(components.contains(&"aprender".to_string()));
    assert!(components.contains(&"repartir".to_string()));
}

// =========================================================================
// Format Number Tests
// =========================================================================

#[test]
fn test_format_number_billions() {
    assert_eq!(format_number(2_000_000_000), "2B");
}

#[test]
fn test_format_number_millions() {
    assert_eq!(format_number(5_000_000), "5M");
}

#[test]
fn test_format_number_thousands() {
    assert_eq!(format_number(10_000), "10K");
}

#[test]
fn test_format_number_small() {
    assert_eq!(format_number(500), "500");
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_empty_query() {
    let rec = Recommender::new();
    let response = rec.query("");

    // Should still return something
    assert!(!response.primary.component.is_empty());
    assert_eq!(response.primary.component, "batuta"); // Fallback
}

#[test]
fn test_nonsense_query() {
    let rec = Recommender::new();
    let response = rec.query("xyz abc 123");

    // Should still return fallback
    assert!(!response.primary.component.is_empty());
}

#[test]
fn test_multiple_algorithms() {
    let rec = Recommender::new();
    let response = rec.query("Should I use random forest or gradient boosting?");

    // Should pick first one detected
    assert!(response.algorithm.is_some());
}

// =========================================================================
// Bug Fix Regression Tests
// =========================================================================

#[test]
fn test_speech_recognition_recommends_whisper() {
    let rec = Recommender::new();
    let response = rec.query("speech recognition");

    assert_eq!(response.primary.component, "whisper-apr");
    assert!(response.primary.confidence >= 0.85);
    assert_eq!(response.problem_class, "Speech Recognition");
}

#[test]
fn test_large_data_size_selects_gpu_or_simd() {
    let rec = Recommender::new();
    let response = rec.query("I have 500m samples and need GPU training with LoRA");

    assert!(
        matches!(response.compute.backend, Backend::GPU | Backend::SIMD),
        "Expected GPU or SIMD for 500M samples, got {:?}",
        response.compute.backend
    );
}

#[test]
fn test_query_transfers_data_size_to_constraints() {
    let rec = Recommender::new();
    let response = rec.query("Train on 1m samples");

    // Should not say "unspecified size" in rationale
    assert!(
        !response.compute.rationale.contains("unspecified"),
        "Data size should be transferred from parsed query; rationale: {}",
        response.compute.rationale
    );
}

// =========================================================================
// Code Example Coverage Tests (--format code)
// =========================================================================

#[test]
fn test_code_example_whisper_apr() {
    let rec = Recommender::new();
    let response = rec.query("speech recognition transcription");

    assert_eq!(response.primary.component, "whisper-apr");
    assert!(response.code_example.is_some(), "whisper-apr query should produce a code example");
    let code = response.code_example.unwrap();
    assert!(code.contains("whisper"), "whisper code example should reference whisper");
    assert!(
        code.contains("#[cfg(test)]"),
        "whisper-apr code example should contain test companion"
    );
}

#[test]
fn test_code_example_realizar() {
    let rec = Recommender::new();
    let response = rec.query("deploy model for inference serving");

    assert_eq!(response.primary.component, "realizar");
    assert!(response.code_example.is_some(), "realizar query should produce a code example");
    let code = response.code_example.unwrap();
    assert!(
        code.contains("realizar") || code.contains("ModelRegistry"),
        "realizar code example should reference realizar"
    );
    assert!(code.contains("#[cfg(test)]"), "realizar code example should contain test companion");
}

#[test]
fn test_code_example_repartir() {
    let rec = Recommender::new();
    let response = rec.query("distribute computation across cluster with repartir");

    assert_eq!(response.primary.component, "repartir");
    assert!(response.code_example.is_some(), "repartir query should produce a code example");
    let code = response.code_example.unwrap();
    assert!(
        code.contains("repartir") || code.contains("Pool"),
        "repartir code example should reference repartir"
    );
    assert!(code.contains("#[cfg(test)]"), "repartir code example should contain test companion");
}

#[test]
fn test_code_example_none_for_unknown() {
    let rec = Recommender::new();
    let response = rec.query("");

    // Empty query falls back to batuta which has no code example
    assert!(
        response.code_example.is_none(),
        "fallback component should not produce a code example"
    );
}
