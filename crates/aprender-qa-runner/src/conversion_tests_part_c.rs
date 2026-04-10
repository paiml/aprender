#[test]
fn test_bug_type_equality() {
    assert_eq!(
        ConversionBugType::EmbeddingTransposition,
        ConversionBugType::EmbeddingTransposition
    );
    assert_ne!(
        ConversionBugType::EmbeddingTransposition,
        ConversionBugType::TokenizerMissing
    );
}

#[test]
fn test_conversion_evidence_source_format() {
    let evidence = ConversionEvidence {
        source_hash: "abc123".to_string(),
        converted_hash: "def456".to_string(),
        max_diff: 0.1,
        diff_indices: vec![0, 5, 10],
        source_format: Format::SafeTensors,
        target_format: Format::Apr,
        backend: Backend::Gpu,
        failure_type: None,
        quant_type: None,
    };
    assert_eq!(evidence.source_format, Format::SafeTensors);
    assert_eq!(evidence.target_format, Format::Apr);
    assert_eq!(evidence.backend, Backend::Gpu);
}

#[test]
fn test_conversion_test_model_id() {
    let model_id = ModelId::new("my-org", "my-model");
    let test = ConversionTest::new(Format::Gguf, Format::Apr, Backend::Cpu, model_id.clone());
    assert_eq!(test.model_id.org, "my-org");
    assert_eq!(test.model_id.name, "my-model");
}

#[test]
fn test_semantic_conversion_test_backend() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Gpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(test.backend, Backend::Gpu);
}

#[test]
fn test_round_trip_test_model_id() {
    let model_id = ModelId::new("org", "name");
    let rt = RoundTripTest::new(
        vec![Format::Gguf, Format::Apr],
        Backend::Cpu,
        model_id.clone(),
    );
    assert_eq!(rt.model_id.org, "org");
    assert_eq!(rt.model_id.name, "name");
}

#[test]
fn test_conversion_config_backends() {
    let config = ConversionConfig::default();
    assert_eq!(config.backends.len(), 2);
    assert!(config.backends.contains(&Backend::Cpu));
    assert!(config.backends.contains(&Backend::Gpu));
}

#[test]
fn test_conversion_config_custom() {
    let config = ConversionConfig {
        test_all_pairs: false,
        test_round_trips: false,
        backends: vec![Backend::Cpu],
        no_gpu: true,
        ..Default::default()
    };
    assert!(!config.test_all_pairs);
    assert!(!config.test_round_trips);
    assert_eq!(config.backends.len(), 1);
}

#[test]
fn test_conversion_executor_config_access() {
    let config = ConversionConfig::cpu_only();
    let executor = ConversionExecutor::new(config);
    assert!(executor.config.no_gpu);
    assert!(executor.config.test_all_pairs);
}

#[test]
fn test_all_conversion_pairs_bidirectional() {
    let pairs = all_conversion_pairs();
    // Should have GGUF -> APR and APR -> GGUF
    let has_gguf_to_apr = pairs.contains(&(Format::Gguf, Format::Apr));
    let has_apr_to_gguf = pairs.contains(&(Format::Apr, Format::Gguf));
    assert!(has_gguf_to_apr);
    assert!(has_apr_to_gguf);
}

#[test]
fn test_epsilon_value() {
    assert!((EPSILON - 1e-6).abs() < 1e-10);
}

#[test]
fn test_conversion_result_corroborated_max_diff() {
    let result = ConversionResult::Corroborated {
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        max_diff: 1e-8,
    };
    match result {
        ConversionResult::Corroborated { max_diff, .. } => {
            assert!(max_diff < EPSILON);
        }
        ConversionResult::Falsified { .. } => panic!("Expected Corroborated"),
    }
}

#[test]
fn test_conversion_result_falsified_gate_id() {
    let result = ConversionResult::Falsified {
        gate_id: "F-CONV-G-A".to_string(),
        reason: "Outputs differ".to_string(),
        evidence: ConversionEvidence {
            source_hash: "a".to_string(),
            converted_hash: "b".to_string(),
            max_diff: 0.5,
            diff_indices: vec![],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: None,
            quant_type: None,
        },
    };
    match result {
        ConversionResult::Falsified { gate_id, .. } => {
            assert_eq!(gate_id, "F-CONV-G-A");
        }
        ConversionResult::Corroborated { .. } => panic!("Expected Falsified"),
    }
}

#[test]
fn test_semantic_test_result_corroborated_outputs() {
    let result = SemanticTestResult::Corroborated {
        source_output: "answer is 4".to_string(),
        target_output: "answer is 4".to_string(),
    };
    match result {
        SemanticTestResult::Corroborated {
            source_output,
            target_output,
        } => {
            assert_eq!(source_output, target_output);
        }
        SemanticTestResult::Falsified { .. } => panic!("Expected Corroborated"),
    }
}

#[test]
fn test_semantic_test_result_falsified_stderr() {
    let result = SemanticTestResult::Falsified {
        bug_type: ConversionBugType::TokenizerMissing,
        source_output: "4".to_string(),
        target_output: "garbage".to_string(),
        stderr: "PMAT-172: tokenizer missing".to_string(),
    };
    match result {
        SemanticTestResult::Falsified { stderr, .. } => {
            assert!(stderr.contains("PMAT-172"));
        }
        SemanticTestResult::Corroborated { .. } => panic!("Expected Falsified"),
    }
}

#[test]
fn test_all_bug_types_have_gate_ids() {
    let bug_types = [
        ConversionBugType::EmbeddingTransposition,
        ConversionBugType::TokenizerMissing,
        ConversionBugType::WeightCorruption,
        ConversionBugType::ShapeMismatch,
        ConversionBugType::SemanticDrift,
        ConversionBugType::Unknown,
    ];
    for bug_type in bug_types {
        let gate_id = bug_type.gate_id();
        assert!(!gate_id.is_empty());
        assert!(gate_id.starts_with("F-CONV-"));
    }
}

#[test]
fn test_all_bug_types_have_descriptions() {
    let bug_types = [
        ConversionBugType::EmbeddingTransposition,
        ConversionBugType::TokenizerMissing,
        ConversionBugType::WeightCorruption,
        ConversionBugType::ShapeMismatch,
        ConversionBugType::SemanticDrift,
        ConversionBugType::Unknown,
    ];
    for bug_type in bug_types {
        let desc = bug_type.description();
        assert!(!desc.is_empty());
    }
}

#[test]
fn test_conversion_evidence_diff_indices() {
    let evidence = ConversionEvidence {
        source_hash: "a".to_string(),
        converted_hash: "b".to_string(),
        max_diff: 0.1,
        diff_indices: vec![0, 1, 2, 3, 4],
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        failure_type: None,
        quant_type: None,
    };
    assert_eq!(evidence.diff_indices.len(), 5);
}

#[test]
fn test_round_trip_test_full_cycle() {
    let rt = RoundTripTest::new(
        vec![Format::Gguf, Format::Apr, Format::SafeTensors, Format::Gguf],
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(rt.formats.len(), 4);
    assert_eq!(rt.formats[0], Format::Gguf);
    assert_eq!(rt.formats[3], Format::Gguf);
}

#[test]
fn test_conversion_config_clone_equality() {
    let config1 = ConversionConfig::default();
    let config2 = config1.clone();
    assert_eq!(config1.test_all_pairs, config2.test_all_pairs);
    assert_eq!(config1.test_round_trips, config2.test_round_trips);
    assert_eq!(config1.no_gpu, config2.no_gpu);
    assert_eq!(config1.backends.len(), config2.backends.len());
}

#[test]
fn test_generate_conversion_tests_contains_all_backends() {
    let model_id = ModelId::new("test", "model");
    let tests = generate_conversion_tests(&model_id);

    let cpu_backend_present = tests.iter().any(|t| t.backend == Backend::Cpu);
    let gpu_backend_present = tests.iter().any(|t| t.backend == Backend::Gpu);

    assert!(cpu_backend_present);
    assert!(gpu_backend_present);
}

#[test]
fn test_garbage_patterns_constant() {
    assert!(!GARBAGE_PATTERNS.is_empty());
    assert!(GARBAGE_PATTERNS.contains(&"PAD"));
    assert!(GARBAGE_PATTERNS.contains(&"<pad>"));
}
