#[test]
fn test_all_conversion_pairs() {
    let pairs = all_conversion_pairs();
    assert_eq!(pairs.len(), 6);
}

#[test]
fn test_all_backends() {
    let backends = all_backends();
    assert_eq!(backends.len(), 2);
}

#[test]
fn test_generate_conversion_tests() {
    let model_id = ModelId::new("test", "model");
    let tests = generate_conversion_tests(&model_id);
    // 6 pairs × 2 backends = 12 tests
    assert_eq!(tests.len(), 12);
}

#[test]
fn test_gate_id() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(test.gate_id(), "F-CONV-G-A");
}

#[test]
fn test_compute_diff_identical() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let diff = test.compute_diff("hello", "hello");
    assert!((diff - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_compute_diff_different() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let diff = test.compute_diff("hello", "world");
    assert!(diff > 0.0);
}

#[test]
fn test_hash_output() {
    let hash1 = ConversionTest::hash_output("test");
    let hash2 = ConversionTest::hash_output("test");
    assert_eq!(hash1, hash2);

    let hash3 = ConversionTest::hash_output("different");
    assert_ne!(hash1, hash3);
}

#[test]
fn test_find_diff_indices() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let indices = test.find_diff_indices("hello", "hallo");
    assert_eq!(indices, vec![1]);
}

#[test]
fn test_conversion_result_to_evidence_corroborated() {
    let result = ConversionResult::Corroborated {
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        max_diff: 0.0,
    };
    let evidence: Evidence = result.into();
    assert!(evidence.outcome.is_pass());
}

#[test]
fn test_conversion_result_to_evidence_falsified() {
    let result = ConversionResult::Falsified {
        gate_id: "F-CONV-G-A".to_string(),
        reason: "Test failure".to_string(),
        evidence: ConversionEvidence {
            source_hash: "abc".to_string(),
            converted_hash: "def".to_string(),
            max_diff: 0.5,
            diff_indices: vec![0, 1],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: None,
            quant_type: None,
        },
    };
    let evidence: Evidence = result.into();
    assert!(!evidence.outcome.is_pass());
}

#[test]
fn test_round_trip_test_new() {
    let rt = RoundTripTest::new(
        vec![Format::Gguf, Format::Apr, Format::SafeTensors],
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(rt.formats.len(), 3);
}

#[test]
fn test_default_epsilon() {
    assert!((default_epsilon() - 1e-6).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_test_epsilon() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    assert!((test.epsilon - EPSILON).abs() < f64::EPSILON);
}

#[test]
fn test_compute_diff_empty_strings() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let diff = test.compute_diff("", "");
    assert!((diff - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_compute_diff_one_empty() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let diff = test.compute_diff("hello", "");
    assert!(diff > 0.0);
}

#[test]
fn test_find_diff_indices_empty() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let indices = test.find_diff_indices("", "");
    assert!(indices.is_empty());
}

#[test]
fn test_find_diff_indices_all_different() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let indices = test.find_diff_indices("abc", "xyz");
    assert_eq!(indices.len(), 3);
}

#[test]
fn test_gate_id_safetensors() {
    let test = ConversionTest::new(
        Format::SafeTensors,
        Format::Gguf,
        Backend::Gpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(test.gate_id(), "F-CONV-S-G");
}

#[test]
fn test_gate_id_apr() {
    let test = ConversionTest::new(
        Format::Apr,
        Format::SafeTensors,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(test.gate_id(), "F-CONV-A-S");
}

#[test]
fn test_all_conversion_pairs_unique() {
    let pairs = all_conversion_pairs();
    for (i, p1) in pairs.iter().enumerate() {
        for (j, p2) in pairs.iter().enumerate() {
            if i != j {
                assert!(p1 != p2, "Duplicate pair found");
            }
        }
    }
}

#[test]
fn test_conversion_evidence_clone() {
    let evidence = ConversionEvidence {
        source_hash: "abc".to_string(),
        converted_hash: "def".to_string(),
        max_diff: 0.5,
        diff_indices: vec![0, 1],
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        failure_type: None,
        quant_type: None,
    };
    let cloned = evidence.clone();
    assert_eq!(evidence.source_hash, cloned.source_hash);
    assert!((evidence.max_diff - cloned.max_diff).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_result_clone() {
    let result = ConversionResult::Corroborated {
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        max_diff: 0.0,
    };
    let cloned = result.clone();
    match cloned {
        ConversionResult::Corroborated { max_diff, .. } => {
            assert!((max_diff - 0.0).abs() < f64::EPSILON);
        }
        ConversionResult::Falsified { .. } => panic!("Expected Corroborated"),
    }
}

#[test]
fn test_conversion_test_clone() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let cloned = test.clone();
    assert_eq!(test.source_format, cloned.source_format);
    assert_eq!(test.target_format, cloned.target_format);
}

#[test]
fn test_round_trip_test_formats() {
    let rt = RoundTripTest::new(
        vec![Format::Gguf, Format::Apr],
        Backend::Gpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(rt.formats.len(), 2);
    assert_eq!(rt.backend, Backend::Gpu);
}

#[test]
fn test_conversion_test_debug() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let debug_str = format!("{test:?}");
    assert!(debug_str.contains("ConversionTest"));
}

#[test]
fn test_conversion_evidence_debug() {
    let evidence = ConversionEvidence {
        source_hash: "abc".to_string(),
        converted_hash: "def".to_string(),
        max_diff: 0.5,
        diff_indices: vec![0, 1],
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        failure_type: None,
        quant_type: None,
    };
    let debug_str = format!("{evidence:?}");
    assert!(debug_str.contains("ConversionEvidence"));
}

#[test]
fn test_conversion_result_debug() {
    let result = ConversionResult::Corroborated {
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        max_diff: 0.0,
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("Corroborated"));
}

#[test]
fn test_epsilon_constant() {
    assert!(EPSILON > 0.0);
    assert!(EPSILON < 1.0);
}

#[test]
fn test_generate_conversion_tests_all_formats() {
    let model_id = ModelId::new("org", "model");
    let tests = generate_conversion_tests(&model_id);

    // Verify all format pairs are covered
    let has_gguf_to_apr = tests
        .iter()
        .any(|t| t.source_format == Format::Gguf && t.target_format == Format::Apr);
    let has_apr_to_safetensors = tests
        .iter()
        .any(|t| t.source_format == Format::Apr && t.target_format == Format::SafeTensors);

    assert!(has_gguf_to_apr);
    assert!(has_apr_to_safetensors);
}

#[test]
fn test_conversion_config_default() {
    let config = ConversionConfig::default();
    assert!(config.test_all_pairs);
    assert!(config.test_round_trips);
    assert_eq!(config.backends.len(), 2);
    assert!(!config.no_gpu);
}

#[test]
fn test_conversion_config_cpu_only() {
    let config = ConversionConfig::cpu_only();
    assert!(config.test_all_pairs);
    assert!(config.test_round_trips);
    assert_eq!(config.backends.len(), 1);
    assert_eq!(config.backends[0], Backend::Cpu);
    assert!(config.no_gpu);
}

#[test]
fn test_conversion_executor_new() {
    let config = ConversionConfig::default();
    let executor = ConversionExecutor::new(config);
    assert!(!executor.config.no_gpu);
}

#[test]
fn test_conversion_executor_with_defaults() {
    let executor = ConversionExecutor::with_defaults();
    assert!(executor.config.test_all_pairs);
}

#[test]
fn test_conversion_config_debug() {
    let config = ConversionConfig::default();
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("ConversionConfig"));
}

#[test]
fn test_conversion_config_clone() {
    let config = ConversionConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.test_all_pairs, config.test_all_pairs);
    assert_eq!(cloned.no_gpu, config.no_gpu);
}

#[test]
fn test_conversion_executor_debug() {
    let executor = ConversionExecutor::with_defaults();
    let debug_str = format!("{executor:?}");
    assert!(debug_str.contains("ConversionExecutor"));
}

#[test]
fn test_round_trip_test_debug() {
    let rt = RoundTripTest::new(
        vec![Format::Gguf, Format::Apr],
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let debug_str = format!("{rt:?}");
    assert!(debug_str.contains("RoundTripTest"));
}

#[test]
fn test_round_trip_test_clone() {
    let rt = RoundTripTest::new(
        vec![Format::Gguf, Format::Apr],
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let cloned = rt.clone();
    assert_eq!(cloned.formats.len(), rt.formats.len());
    assert_eq!(cloned.backend, rt.backend);
}

#[test]
fn test_conversion_test_with_epsilon() {
    let test = ConversionTest {
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        model_id: ModelId::new("test", "model"),
        epsilon: 1e-9,
        binary: default_binary(),
        quant_type: None,
        output_dir: None,
    };
    assert!((test.epsilon - 1e-9).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_execution_result() {
    let result = ConversionExecutionResult {
        passed: 10,
        failed: 2,
        total: 12,
        evidence: vec![],
        results: vec![],
        duration_ms: 1000,
    };
    assert_eq!(result.passed, 10);
    assert_eq!(result.failed, 2);
    assert_eq!(result.total, 12);
}
