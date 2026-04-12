/// Verify ARITHMETIC_EXPECTED constant is populated with expected values
#[test]
fn test_arithmetic_expected_constant() {
    assert!(!ARITHMETIC_EXPECTED.is_empty());
    assert!(ARITHMETIC_EXPECTED.contains(&"4"));
    assert!(ARITHMETIC_EXPECTED.contains(&"four"));
}

// Additional tests for coverage

/// Verify all ConversionBugType variants survive serde round-trip
#[test]
fn test_conversion_bug_type_serialization() {
    let bug_types = [
        ConversionBugType::EmbeddingTransposition,
        ConversionBugType::TokenizerMissing,
        ConversionBugType::WeightCorruption,
        ConversionBugType::ShapeMismatch,
        ConversionBugType::SemanticDrift,
        ConversionBugType::Unknown,
    ];
    for bug_type in bug_types {
        let json = serde_json::to_string(&bug_type).unwrap();
        let parsed: ConversionBugType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, bug_type);
    }
}

/// Verify ConversionTest serialization preserves format fields
#[test]
fn test_conversion_test_serialization() {
    let test = ConversionTest {
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        model_id: ModelId::new("org", "name"),
        epsilon: 1e-7,
        binary: default_binary(),
        quant_type: None,
        output_dir: None,
    };
    let json = serde_json::to_string(&test).unwrap();
    let parsed: ConversionTest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.source_format, Format::Gguf);
    assert_eq!(parsed.target_format, Format::Apr);
}

/// Verify Corroborated result serialization preserves max_diff
#[test]
fn test_conversion_result_serialization_corroborated() {
    let result = ConversionResult::Corroborated {
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Gpu,
        max_diff: 1e-9,
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: ConversionResult = serde_json::from_str(&json).unwrap();
    match parsed {
        ConversionResult::Corroborated { max_diff, .. } => {
            assert!(max_diff < EPSILON);
        }
        ConversionResult::Falsified { .. } => panic!("Expected Corroborated"),
    }
}

/// Verify Falsified result serialization preserves gate_id and evidence
#[test]
fn test_conversion_result_serialization_falsified() {
    let result = ConversionResult::Falsified {
        gate_id: "F-CONV-G-A".to_string(),
        reason: "Test failure".to_string(),
        evidence: ConversionEvidence {
            source_hash: "abc".to_string(),
            converted_hash: "def".to_string(),
            max_diff: 0.5,
            diff_indices: vec![0, 1, 2],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: None,
            quant_type: None,
        },
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: ConversionResult = serde_json::from_str(&json).unwrap();
    match parsed {
        ConversionResult::Falsified { gate_id, .. } => {
            assert_eq!(gate_id, "F-CONV-G-A");
        }
        ConversionResult::Corroborated { .. } => panic!("Expected Falsified"),
    }
}

/// Verify ConversionEvidence serialization preserves hashes and indices
#[test]
fn test_conversion_evidence_serialization() {
    let evidence = ConversionEvidence {
        source_hash: "hash1".to_string(),
        converted_hash: "hash2".to_string(),
        max_diff: 0.05,
        diff_indices: vec![1, 3, 5],
        source_format: Format::SafeTensors,
        target_format: Format::Gguf,
        backend: Backend::Gpu,
        failure_type: None,
        quant_type: None,
    };
    let json = serde_json::to_string(&evidence).unwrap();
    let parsed: ConversionEvidence = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.source_hash, "hash1");
    assert_eq!(parsed.diff_indices.len(), 3);
}

/// Verify SemanticTestResult::Falsified clone preserves bug_type and stderr
#[test]
fn test_semantic_test_result_clone() {
    let result = SemanticTestResult::Falsified {
        bug_type: ConversionBugType::TokenizerMissing,
        source_output: "source".to_string(),
        target_output: "target".to_string(),
        stderr: "error".to_string(),
    };
    let cloned = result.clone();
    match cloned {
        SemanticTestResult::Falsified {
            bug_type, stderr, ..
        } => {
            assert_eq!(bug_type, ConversionBugType::TokenizerMissing);
            assert_eq!(stderr, "error");
        }
        SemanticTestResult::Corroborated { .. } => panic!("Expected Falsified"),
    }
}

/// Verify classify_bug returns Unknown when source is empty but target has content
#[test]
fn test_classify_bug_source_empty_target_has_content() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Source empty, target has content - unusual case, returns Unknown
    let bug = test.classify_bug("", "Some output", false);
    assert_eq!(bug, Some(ConversionBugType::Unknown));
}

/// Verify classify_bug returns None when both outputs are empty
#[test]
fn test_classify_bug_both_empty() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Both empty - no bug
    let bug = test.classify_bug("", "", false);
    assert!(bug.is_none());
}

/// Verify classify_bug returns Unknown when source lacks expected but target has it
#[test]
fn test_classify_bug_source_no_expected_target_has_expected() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Source doesn't have expected, target does - weird but not a bug in our heuristic
    let bug = test.classify_bug("random text", "The answer is 4", false);
    // Outputs differ but no clear pattern
    assert_eq!(bug, Some(ConversionBugType::Unknown));
}

/// Verify compute_diff handles unicode characters correctly
#[test]
fn test_compute_diff_unicode() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let diff = test.compute_diff("hello 你好", "hello 世界");
    assert!(diff > 0.0);
    assert!(diff < 1.0);
}

/// Verify find_diff_indices detects unicode character differences
#[test]
fn test_find_diff_indices_unicode() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let indices = test.find_diff_indices("ab你好", "abXX");
    // Comparing "你" vs "X" and "好" vs "X"
    assert!(indices.len() >= 2);
}

/// Verify hash_output produces deterministic 16-char hex for unicode
#[test]
fn test_hash_output_unicode() {
    let hash1 = ConversionTest::hash_output("hello 你好 世界");
    let hash2 = ConversionTest::hash_output("hello 你好 世界");
    assert_eq!(hash1, hash2);
    assert_eq!(hash1.len(), 16); // 16 hex chars
}

/// Verify pass_rate calculates correct percentage for partial passes
#[test]
fn test_conversion_execution_result_pass_rate_partial() {
    let result = ConversionExecutionResult {
        passed: 7,
        failed: 3,
        total: 10,
        evidence: vec![],
        results: vec![],
        duration_ms: 1000,
    };
    let rate = result.pass_rate();
    assert!((rate - 70.0).abs() < f64::EPSILON);
}

/// Verify ConversionConfig accepts specific backend lists
#[test]
fn test_conversion_config_with_specific_backends() {
    let config = ConversionConfig {
        test_all_pairs: true,
        test_round_trips: false,
        backends: vec![Backend::Gpu],
        no_gpu: false,
        ..Default::default()
    };
    assert_eq!(config.backends.len(), 1);
    assert_eq!(config.backends[0], Backend::Gpu);
    assert!(!config.test_round_trips);
}

/// Verify SemanticConversionTest preserves constructor fields
#[test]
fn test_semantic_conversion_test_fields() {
    let test = SemanticConversionTest::new(
        Format::SafeTensors,
        Format::Apr,
        Backend::Gpu,
        ModelId::new("org", "model"),
    );
    assert_eq!(test.source_format, Format::SafeTensors);
    assert_eq!(test.target_format, Format::Apr);
    assert_eq!(test.backend, Backend::Gpu);
    assert_eq!(test.model_id.org, "org");
}

/// Verify RoundTripTest stores format chain and backend
#[test]
fn test_round_trip_test_with_two_formats() {
    let rt = RoundTripTest::new(
        vec![Format::Apr, Format::Gguf],
        Backend::Gpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(rt.formats.len(), 2);
    assert_eq!(rt.backend, Backend::Gpu);
}

/// Verify ConversionEvidence with empty diff_indices and zero max_diff
#[test]
fn test_conversion_evidence_with_empty_diff_indices() {
    let evidence = ConversionEvidence {
        source_hash: "same".to_string(),
        converted_hash: "same".to_string(),
        max_diff: 0.0,
        diff_indices: vec![],
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        failure_type: None,
        quant_type: None,
    };
    assert!(evidence.diff_indices.is_empty());
    assert!((evidence.max_diff - 0.0).abs() < f64::EPSILON);
}

/// Verify all_conversion_pairs returns six bidirectional format pairs
#[test]
fn test_all_conversion_pairs_complete() {
    let pairs = all_conversion_pairs();
    // Should have bidirectional pairs for all format combinations
    // 3 formats = 6 pairs (A->B, B->A for each pair)
    assert_eq!(pairs.len(), 6);

    // Check specific pairs exist
    assert!(pairs.contains(&(Format::Gguf, Format::Apr)));
    assert!(pairs.contains(&(Format::Apr, Format::Gguf)));
    assert!(pairs.contains(&(Format::Gguf, Format::SafeTensors)));
    assert!(pairs.contains(&(Format::SafeTensors, Format::Gguf)));
    assert!(pairs.contains(&(Format::Apr, Format::SafeTensors)));
    assert!(pairs.contains(&(Format::SafeTensors, Format::Apr)));
}

/// Verify generate_conversion_tests preserves model_id across all tests
#[test]
fn test_generate_conversion_tests_model_id_preserved() {
    let model_id = ModelId::new("my-org", "my-model-v1");
    let tests = generate_conversion_tests(&model_id);

    for test in &tests {
        assert_eq!(test.model_id.org, "my-org");
        assert_eq!(test.model_id.name, "my-model-v1");
    }
}

/// Verify ConversionTest debug format includes type name and formats
#[test]
fn test_conversion_test_debug_format() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let debug = format!("{test:?}");
    assert!(debug.contains("ConversionTest"));
    assert!(debug.contains("Gguf"));
    assert!(debug.contains("Apr"));
}

/// Verify classify_bug detects EmbeddingTransposition for multiple garbage patterns
#[test]
fn test_classify_bug_with_multiple_garbage_patterns() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Target has multiple garbage patterns
    let bug = test.classify_bug("The answer is 4", "PAD <pad> <|endoftext|> 151935", false);
    assert_eq!(bug, Some(ConversionBugType::EmbeddingTransposition));
}

/// Verify classify_bug detects WeightCorruption for whitespace-only target
#[test]
fn test_classify_bug_target_only_whitespace() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Source has content but no expected arithmetic, target is whitespace
    let bug = test.classify_bug("Some random output", "   \t\n  ", false);
    assert_eq!(bug, Some(ConversionBugType::WeightCorruption));
}

/// Verify ConversionExecutor preserves custom config settings
#[test]
fn test_conversion_executor_custom_config() {
    let config = ConversionConfig {
        test_all_pairs: false,
        test_round_trips: true,
        backends: vec![Backend::Cpu],
        no_gpu: true,
        ..Default::default()
    };
    let executor = ConversionExecutor::new(config);
    assert!(!executor.config.test_all_pairs);
    assert!(executor.config.test_round_trips);
    assert!(executor.config.no_gpu);
}

/// Verify Corroborated semantic result reports as pass
#[test]
fn test_semantic_test_result_is_pass_corroborated() {
    let result = SemanticTestResult::Corroborated {
        source_output: "test".to_string(),
        target_output: "test".to_string(),
    };
    assert!(result.is_pass());
}

/// Verify Falsified semantic result reports as not pass
#[test]
fn test_semantic_test_result_is_pass_falsified() {
    let result = SemanticTestResult::Falsified {
        bug_type: ConversionBugType::Unknown,
        source_output: "a".to_string(),
        target_output: "b".to_string(),
        stderr: String::new(),
    };
    assert!(!result.is_pass());
}

/// Verify Corroborated semantic result has no bug_type
#[test]
fn test_semantic_test_result_bug_type_corroborated() {
    let result = SemanticTestResult::Corroborated {
        source_output: "test".to_string(),
        target_output: "test".to_string(),
    };
    assert!(result.bug_type().is_none());
}

/// Verify Falsified semantic result returns correct bug_type
#[test]
fn test_semantic_test_result_bug_type_falsified() {
    let result = SemanticTestResult::Falsified {
        bug_type: ConversionBugType::SemanticDrift,
        source_output: "a".to_string(),
        target_output: "b".to_string(),
        stderr: "warning".to_string(),
    };
    assert_eq!(result.bug_type(), Some(ConversionBugType::SemanticDrift));
}

/// Verify Corroborated result round-trips through JSON with max_diff intact
#[test]
fn test_conversion_result_corroborated_serialization() {
    let result = ConversionResult::Corroborated {
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        max_diff: 0.001,
    };
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("Corroborated"));
    let deserialized: ConversionResult = serde_json::from_str(&json).unwrap();
    if let ConversionResult::Corroborated { max_diff, .. } = deserialized {
        assert!((max_diff - 0.001).abs() < f64::EPSILON);
    } else {
        panic!("Expected Corroborated");
    }
}

/// Verify Falsified result JSON contains variant name and gate_id
#[test]
fn test_conversion_result_falsified_serialization() {
    let result = ConversionResult::Falsified {
        gate_id: "F-TEST-001".to_string(),
        reason: "Test failure".to_string(),
        evidence: ConversionEvidence {
            source_hash: "abc".to_string(),
            converted_hash: "def".to_string(),
            max_diff: 0.5,
            diff_indices: vec![1, 2, 3],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: None,
            quant_type: None,
        },
    };
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("Falsified"));
    assert!(json.contains("F-TEST-001"));
}

/// Verify ConversionTest accepts custom epsilon values
#[test]
fn test_conversion_test_new_with_epsilon() {
    let test = ConversionTest {
        source_format: Format::Apr,
        target_format: Format::Gguf,
        backend: Backend::Gpu,
        model_id: ModelId::new("org", "model"),
        epsilon: 1e-10,
        binary: default_binary(),
        quant_type: None,
        output_dir: None,
    };
    assert!((test.epsilon - 1e-10).abs() < 1e-15);
}
