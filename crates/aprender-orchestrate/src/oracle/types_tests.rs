use super::*;

// =========================================================================
// StackLayer Tests
// =========================================================================

#[test]
fn test_stack_layer_index() {
    assert_eq!(StackLayer::Primitives.index(), 0);
    assert_eq!(StackLayer::MlAlgorithms.index(), 1);
    assert_eq!(StackLayer::MlPipeline.index(), 2);
    assert_eq!(StackLayer::Transpilers.index(), 3);
    assert_eq!(StackLayer::Orchestration.index(), 4);
    assert_eq!(StackLayer::Quality.index(), 5);
    assert_eq!(StackLayer::Data.index(), 6);
}

#[test]
fn test_stack_layer_all() {
    let layers = StackLayer::all();
    assert_eq!(layers.len(), 8);
    assert_eq!(layers[0], StackLayer::Primitives);
    assert_eq!(layers[6], StackLayer::Data);
    assert_eq!(layers[7], StackLayer::Media);
}

#[test]
fn test_stack_layer_display() {
    assert_eq!(StackLayer::Primitives.to_string(), "Compute Primitives");
    assert_eq!(StackLayer::MlAlgorithms.to_string(), "ML Algorithms");
    assert_eq!(StackLayer::Transpilers.to_string(), "Transpilers");
}

#[test]
fn test_stack_layer_serialization() {
    let layer = StackLayer::MlAlgorithms;
    let json = serde_json::to_string(&layer).unwrap();
    let parsed: StackLayer = serde_json::from_str(&json).unwrap();
    assert_eq!(layer, parsed);
}

// =========================================================================
// Capability Tests
// =========================================================================

#[test]
fn test_capability_new() {
    let cap = Capability::new("simd", CapabilityCategory::Compute);
    assert_eq!(cap.name, "simd");
    assert_eq!(cap.category, CapabilityCategory::Compute);
    assert!(cap.description.is_none());
}

#[test]
fn test_capability_with_description() {
    let cap = Capability::new("vector_ops", CapabilityCategory::Compute)
        .with_description("SIMD-accelerated vector operations");
    assert_eq!(cap.description.as_deref(), Some("SIMD-accelerated vector operations"));
}

#[test]
fn test_capability_equality() {
    let cap1 = Capability::new("simd", CapabilityCategory::Compute);
    let cap2 = Capability::new("simd", CapabilityCategory::Compute);
    let cap3 = Capability::new("gpu", CapabilityCategory::Compute);
    assert_eq!(cap1, cap2);
    assert_ne!(cap1, cap3);
}

// =========================================================================
// StackComponent Tests
// =========================================================================

#[test]
fn test_stack_component_new() {
    let comp = StackComponent::new(
        "trueno",
        "0.7.3",
        StackLayer::Primitives,
        "SIMD-accelerated tensor operations",
    );
    assert_eq!(comp.name, "trueno");
    assert_eq!(comp.version, "0.7.3");
    assert_eq!(comp.layer, StackLayer::Primitives);
    assert!(comp.capabilities.is_empty());
}

#[test]
fn test_stack_component_with_capability() {
    let comp = StackComponent::new("trueno", "0.7.3", StackLayer::Primitives, "SIMD tensors")
        .with_capability(Capability::new("simd", CapabilityCategory::Compute))
        .with_capability(Capability::new("gpu", CapabilityCategory::Compute));
    assert_eq!(comp.capabilities.len(), 2);
    assert!(comp.has_capability("simd"));
    assert!(comp.has_capability("gpu"));
    assert!(!comp.has_capability("tpu"));
}

#[test]
fn test_stack_component_with_capabilities() {
    let caps = vec![
        Capability::new("vector_ops", CapabilityCategory::Compute),
        Capability::new("matrix_ops", CapabilityCategory::Compute),
    ];
    let comp = StackComponent::new("trueno", "0.7.3", StackLayer::Primitives, "SIMD tensors")
        .with_capabilities(caps);
    assert_eq!(comp.capabilities.len(), 2);
}

#[test]
fn test_stack_component_serialization() {
    let comp = StackComponent::new("aprender", "0.12.0", StackLayer::MlAlgorithms, "ML library")
        .with_capability(Capability::new("random_forest", CapabilityCategory::MachineLearning));
    let json = serde_json::to_string(&comp).unwrap();
    let parsed: StackComponent = serde_json::from_str(&json).unwrap();
    assert_eq!(comp.name, parsed.name);
    assert_eq!(comp.version, parsed.version);
    assert_eq!(comp.capabilities.len(), parsed.capabilities.len());
}

// =========================================================================
// HardwareSpec Tests
// =========================================================================

#[test]
fn test_hardware_spec_default() {
    let hw = HardwareSpec::default();
    assert!(!hw.has_gpu);
    assert!(hw.gpu_memory_gb.is_none());
    assert!(!hw.is_distributed);
}

#[test]
fn test_hardware_spec_cpu_only() {
    let hw = HardwareSpec::cpu_only();
    assert!(!hw.has_gpu());
}

#[test]
fn test_hardware_spec_with_gpu() {
    let hw = HardwareSpec::with_gpu(16.0);
    assert!(hw.has_gpu());
    assert_eq!(hw.gpu_memory_gb, Some(16.0));
}

// =========================================================================
// DataSize Tests
// =========================================================================

#[test]
fn test_data_size_samples() {
    let size = DataSize::samples(1_000_000);
    assert_eq!(size.as_samples(), Some(1_000_000));
    assert!(size.is_large());
}

#[test]
fn test_data_size_bytes() {
    let size = DataSize::bytes(2_000_000_000);
    assert!(size.is_large());
    assert!(size.as_samples().is_none());
}

#[test]
fn test_data_size_small() {
    let size = DataSize::samples(1000);
    assert!(!size.is_large());
}

#[test]
fn test_data_size_unknown() {
    let size = DataSize::Unknown;
    assert!(!size.is_large());
    assert!(size.as_samples().is_none());
}

// =========================================================================
// OracleQuery Tests
// =========================================================================

#[test]
fn test_oracle_query_new() {
    let query = OracleQuery::new("Train a random forest on 1M samples");
    assert_eq!(query.description, "Train a random forest on 1M samples");
    assert!(!query.constraints.sovereign_only);
    assert!(!query.constraints.eu_compliant);
}

#[test]
fn test_oracle_query_with_data_size() {
    let query =
        OracleQuery::new("classification task").with_data_size(DataSize::samples(1_000_000));
    assert_eq!(query.constraints.data_size, Some(DataSize::Samples(1_000_000)));
}

#[test]
fn test_oracle_query_sovereign_only() {
    let query = OracleQuery::new("GDPR compliant training").sovereign_only();
    assert!(query.constraints.sovereign_only);
}

#[test]
fn test_oracle_query_eu_compliant() {
    let query = OracleQuery::new("EU AI Act compliant").eu_compliant();
    assert!(query.constraints.eu_compliant);
}

#[test]
fn test_oracle_query_with_hardware() {
    let query = OracleQuery::new("GPU training").with_hardware(HardwareSpec::with_gpu(24.0));
    assert!(query.constraints.hardware.has_gpu());
}

#[test]
fn test_oracle_query_serialization() {
    let query =
        OracleQuery::new("Test query").with_data_size(DataSize::samples(10000)).sovereign_only();
    let json = serde_json::to_string(&query).unwrap();
    let parsed: OracleQuery = serde_json::from_str(&json).unwrap();
    assert_eq!(query.description, parsed.description);
    assert_eq!(query.constraints.sovereign_only, parsed.constraints.sovereign_only);
}

// =========================================================================
// Backend Tests
// =========================================================================

#[test]
fn test_backend_display() {
    assert_eq!(Backend::Scalar.to_string(), "Scalar");
    assert_eq!(Backend::SIMD.to_string(), "SIMD");
    assert_eq!(Backend::GPU.to_string(), "GPU");
    assert_eq!(Backend::Distributed.to_string(), "Distributed");
}

#[test]
fn test_backend_serialization() {
    let backend = Backend::GPU;
    let json = serde_json::to_string(&backend).unwrap();
    let parsed: Backend = serde_json::from_str(&json).unwrap();
    assert_eq!(backend, parsed);
}

// =========================================================================
// OracleResponse Tests
// =========================================================================

#[test]
fn test_oracle_response_new() {
    let rec = ComponentRecommendation {
        component: "aprender".into(),
        path: Some("aprender::tree::RandomForest".into()),
        confidence: 0.95,
        rationale: "Random forest is ideal for tabular data".into(),
    };
    let response = OracleResponse::new("supervised_learning", rec);
    assert_eq!(response.problem_class, "supervised_learning");
    assert_eq!(response.primary.component, "aprender");
    assert!(!response.distribution.needed);
}

#[test]
fn test_oracle_response_with_algorithm() {
    let rec = ComponentRecommendation {
        component: "aprender".into(),
        path: None,
        confidence: 0.9,
        rationale: "Test".into(),
    };
    let response = OracleResponse::new("ml", rec).with_algorithm("random_forest");
    assert_eq!(response.algorithm.as_deref(), Some("random_forest"));
}

#[test]
fn test_oracle_response_with_supporting() {
    let primary = ComponentRecommendation {
        component: "aprender".into(),
        path: None,
        confidence: 0.9,
        rationale: "Primary".into(),
    };
    let supporting = ComponentRecommendation {
        component: "trueno".into(),
        path: None,
        confidence: 0.8,
        rationale: "Backend compute".into(),
    };
    let response = OracleResponse::new("ml", primary).with_supporting(supporting);
    assert_eq!(response.supporting.len(), 1);
    assert_eq!(response.supporting[0].component, "trueno");
}

#[test]
fn test_oracle_response_with_code_example() {
    let rec = ComponentRecommendation {
        component: "aprender".into(),
        path: None,
        confidence: 0.9,
        rationale: "Test".into(),
    };
    let code = r#"use aprender::tree::RandomForest;
let model = RandomForest::new();"#;
    let response = OracleResponse::new("ml", rec).with_code_example(code);
    assert!(response.code_example.is_some());
    assert!(response.code_example.unwrap().contains("RandomForest"));
}

// =========================================================================
// ProblemDomain Tests
// =========================================================================

#[test]
fn test_problem_domain_display() {
    assert_eq!(ProblemDomain::SupervisedLearning.to_string(), "Supervised Learning");
    assert_eq!(ProblemDomain::PythonMigration.to_string(), "Python Migration");
    assert_eq!(ProblemDomain::GraphAnalytics.to_string(), "Graph Analytics");
}

#[test]
fn test_problem_domain_serialization() {
    let domain = ProblemDomain::DeepLearning;
    let json = serde_json::to_string(&domain).unwrap();
    let parsed: ProblemDomain = serde_json::from_str(&json).unwrap();
    assert_eq!(domain, parsed);
}

// =========================================================================
// OptimizationTarget Tests
// =========================================================================

#[test]
fn test_optimization_target_default() {
    let target = OptimizationTarget::default();
    assert_eq!(target, OptimizationTarget::Speed);
}

#[test]
fn test_optimization_target_serialization() {
    let target = OptimizationTarget::Memory;
    let json = serde_json::to_string(&target).unwrap();
    let parsed: OptimizationTarget = serde_json::from_str(&json).unwrap();
    assert_eq!(target, parsed);
}

// =========================================================================
// QueryConstraints Tests
// =========================================================================

#[test]
fn test_query_constraints_default() {
    let constraints = QueryConstraints::default();
    assert!(constraints.max_latency_ms.is_none());
    assert!(constraints.data_size.is_none());
    assert!(!constraints.sovereign_only);
    assert!(!constraints.eu_compliant);
}

// =========================================================================
// QueryPreferences Tests
// =========================================================================

#[test]
fn test_query_preferences_default() {
    let prefs = QueryPreferences::default();
    assert_eq!(prefs.optimize_for, OptimizationTarget::Speed);
    assert_eq!(prefs.simplicity_weight, 0.0);
    assert!(prefs.existing_components.is_empty());
}

// =========================================================================
// IntegrationPattern Tests
// =========================================================================

#[test]
fn test_integration_pattern() {
    let pattern = IntegrationPattern {
        from: "aprender".into(),
        to: "realizar".into(),
        pattern_name: "model_export".into(),
        description: "Export trained model for serving".into(),
        code_template: Some("model.export_apr(\"model.apr\")".into()),
    };
    assert_eq!(pattern.from, "aprender");
    assert_eq!(pattern.to, "realizar");
    assert!(pattern.code_template.is_some());
}

#[test]
fn test_integration_pattern_serialization() {
    let pattern = IntegrationPattern {
        from: "depyler".into(),
        to: "aprender".into(),
        pattern_name: "sklearn_convert".into(),
        description: "Convert sklearn to aprender".into(),
        code_template: None,
    };
    let json = serde_json::to_string(&pattern).unwrap();
    let parsed: IntegrationPattern = serde_json::from_str(&json).unwrap();
    assert_eq!(pattern.from, parsed.from);
    assert_eq!(pattern.to, parsed.to);
}

#[test]
fn test_stack_component_has_capability() {
    let component =
        StackComponent::new("test", "1.0.0", StackLayer::MlAlgorithms, "Test component")
            .with_capability(Capability::new("test_cap", CapabilityCategory::Compute));

    assert!(component.has_capability("test_cap"));
    assert!(!component.has_capability("nonexistent"));
}

#[test]
fn test_data_size_is_large() {
    let small = DataSize::Samples(999);
    assert!(!small.is_large());

    let large = DataSize::Samples(1_000_001);
    assert!(large.is_large());

    let small_bytes = DataSize::Bytes(999);
    assert!(!small_bytes.is_large());

    let large_bytes = DataSize::Bytes(1_000_000_001);
    assert!(large_bytes.is_large());
}

#[test]
fn test_oracle_query_with_constraints() {
    let constraints = QueryConstraints { max_latency_ms: Some(100), ..Default::default() };
    let query = OracleQuery::new("test").with_constraints(constraints);
    assert!(query.constraints.max_latency_ms.is_some());
}

#[test]
fn test_oracle_query_with_preferences() {
    let prefs = QueryPreferences { optimize_for: OptimizationTarget::Memory, ..Default::default() };
    let query = OracleQuery::new("test").with_preferences(prefs);
    assert_eq!(query.preferences.optimize_for, OptimizationTarget::Memory);
}

#[test]
fn test_component_recommendation_new_with_path() {
    let rec = ComponentRecommendation::with_path(
        "aprender",
        0.9,
        "Use via depyler",
        "aprender::tree::RandomForest".to_string(),
    );
    assert_eq!(rec.component, "aprender");
    assert!(rec.path.is_some());
}

#[test]
fn test_oracle_response_with_compute() {
    let primary = ComponentRecommendation::new("aprender", 0.9, "test");
    let compute = ComputeRecommendation {
        backend: Backend::SIMD,
        rationale: "SIMD for performance".to_string(),
    };
    let response = OracleResponse::new("ml", primary).with_compute(compute);
    assert_eq!(response.compute.backend, Backend::SIMD);
}

#[test]
fn test_oracle_response_with_distribution() {
    let primary = ComponentRecommendation::new("aprender", 0.9, "test");
    let dist = DistributionRecommendation {
        tool: Some("repartir".to_string()),
        needed: true,
        rationale: "Large data".to_string(),
        node_count: Some(4),
    };
    let response = OracleResponse::new("ml", primary).with_distribution(dist);
    assert!(response.distribution.needed);
}

#[test]
fn test_distribution_recommendation_not_needed() {
    let dist = DistributionRecommendation::not_needed("Data fits in memory");
    assert!(!dist.needed);
    assert!(dist.tool.is_none());
}

#[test]
fn test_data_size_as_samples_bytes() {
    let bytes = DataSize::Bytes(1000);
    assert!(bytes.as_samples().is_none());
}

#[test]
fn test_hardware_spec_has_gpu() {
    let cpu_only = HardwareSpec::cpu_only();
    assert!(!cpu_only.has_gpu());

    let with_gpu = HardwareSpec::with_gpu(8.0);
    assert!(with_gpu.has_gpu());
}

// =========================================================================
// Coverage gap tests: StackLayer Display + ProblemDomain Display
// =========================================================================

#[test]
fn test_stack_layer_display_all_variants() {
    // Cover MlPipeline, Orchestration, Quality, Data display arms
    assert_eq!(StackLayer::MlPipeline.to_string(), "Training & Inference");
    assert_eq!(StackLayer::Orchestration.to_string(), "Orchestration");
    assert_eq!(StackLayer::Quality.to_string(), "Quality & Profiling");
    assert_eq!(StackLayer::Data.to_string(), "Data Loading");
}

#[test]
fn test_problem_domain_display_all_variants() {
    // Cover all uncovered ProblemDomain Display arms
    assert_eq!(ProblemDomain::DeepLearning.to_string(), "Deep Learning");
    assert_eq!(ProblemDomain::VectorSearch.to_string(), "Vector Search");
    assert_eq!(ProblemDomain::CMigration.to_string(), "C/C++ Migration");
    assert_eq!(ProblemDomain::ShellMigration.to_string(), "Shell Migration");
    assert_eq!(ProblemDomain::DistributedCompute.to_string(), "Distributed Computing");
    assert_eq!(ProblemDomain::DataPipeline.to_string(), "Data Pipeline");
    assert_eq!(ProblemDomain::ModelServing.to_string(), "Model Serving");
    assert_eq!(ProblemDomain::Testing.to_string(), "Testing");
    assert_eq!(ProblemDomain::Profiling.to_string(), "Profiling");
    assert_eq!(ProblemDomain::Validation.to_string(), "Validation");
}

#[test]
fn test_problem_domain_display_remaining() {
    // Cover Inference and SpeechRecognition
    assert_eq!(ProblemDomain::Inference.to_string(), "Model Inference");
    assert_eq!(ProblemDomain::SpeechRecognition.to_string(), "Speech Recognition");
    assert_eq!(ProblemDomain::LinearAlgebra.to_string(), "Linear Algebra");
}
