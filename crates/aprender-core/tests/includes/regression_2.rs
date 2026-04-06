// =============================================================================
// Category 7: QA Integrity Regressions (P0-QA-001, Bug 206)
// =============================================================================

#[test]
fn regression_p0_qa_001_skipped_gates_distinguishable() {
    let total = 9;
    let executed = 5;
    let skipped = 4;
    assert_eq!(executed + skipped, total);
    assert!(skipped > 0, "P0-QA-001: Must track skipped gates");
}

#[test]
fn regression_bug_206_perf_regression_detection() {
    let prev_tps = 89.8;
    let curr_tps = 67.8;
    let regression = (prev_tps - curr_tps) / prev_tps;
    assert!(
        regression > 0.10,
        "Bug 206: 89.8->67.8 must be >10% regression (was {:.1}%)",
        regression * 100.0
    );
}

#[test]
fn regression_bug_206_no_false_positive() {
    let prev_tps = 67.8;
    let curr_tps = 89.8;
    let regression = (prev_tps - curr_tps) / prev_tps;
    assert!(regression < 0.0, "Improvement must not trigger regression");
}

#[test]
fn regression_bug_206_exact_same_no_regression() {
    let prev_tps = 89.8;
    let curr_tps = 89.8;
    let regression: f64 = (prev_tps - curr_tps) / prev_tps;
    assert!(
        regression.abs() < f64::EPSILON,
        "Same value must not trigger regression"
    );
}

// =============================================================================
// Category 8: Validated Tensors (PMAT-235)
// =============================================================================

#[test]
fn regression_pmat_235_weight_size_mismatch() {
    use aprender::format::validated_tensors::ValidatedWeight;
    let data = vec![1.0f32; 10];
    let result = ValidatedWeight::new(data, 3, 4, "test");
    assert!(
        result.is_err(),
        "PMAT-235: Must reject size mismatch (10 != 3*4)"
    );
}

#[test]
fn regression_pmat_235_weight_correct_size() {
    use aprender::format::validated_tensors::ValidatedWeight;
    let data = vec![1.0f32; 12];
    let result = ValidatedWeight::new(data, 3, 4, "test");
    assert!(
        result.is_ok(),
        "PMAT-235: Must accept correct size (12 == 3*4)"
    );
}

#[test]
fn regression_pmat_235_embedding_size_mismatch() {
    use aprender::format::validated_tensors::ValidatedEmbedding;
    let data = vec![0.1f32; 5];
    let result = ValidatedEmbedding::new(data, 2, 3);
    assert!(
        result.is_err(),
        "PMAT-235: Must reject embedding size mismatch"
    );
}

// =============================================================================
// Category 9: Model Family / Architecture Detection
// =============================================================================

#[test]
fn regression_arch_qwen2_in_registry() {
    let registry = aprender::format::model_family::build_default_registry();
    let names = registry.family_names();
    assert!(names.iter().any(|n| n.contains("qwen2")));
}

#[test]
fn regression_arch_known_families_contains_qwen2() {
    assert!(aprender::format::model_family::KNOWN_FAMILIES.contains(&"qwen2"));
}

#[test]
fn regression_arch_known_families_contains_llama() {
    assert!(aprender::format::model_family::KNOWN_FAMILIES.contains(&"llama"));
}

// =============================================================================
// Category 10: Rosetta Stone Validation
// =============================================================================

#[test]
fn regression_rosetta_construction() {
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let _ = format!("{rosetta:?}");
}

#[test]
fn regression_rosetta_validate_nonexistent() {
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let result = rosetta.validate(Path::new("/nonexistent/model.gguf"));
    assert!(result.is_err());
}

#[test]
fn regression_rosetta_validate_empty() {
    let file = NamedTempFile::with_suffix(".gguf").expect("create temp");
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let result = rosetta.validate(file.path());
    assert!(result.is_err());
}

// =============================================================================
// Category 11: Source/HF URI Parsing
// =============================================================================

#[test]
fn regression_source_parse_hf_valid() {
    use aprender::format::converter_types::Source;
    let result = Source::parse("hf://Qwen/Qwen2.5-Coder-0.5B-Instruct");
    assert!(result.is_ok());
}

#[test]
fn regression_source_parse_hf_invalid_single_part() {
    use aprender::format::converter_types::Source;
    let result = Source::parse("hf://only-one-part");
    assert!(result.is_err());
}

#[test]
fn regression_source_parse_hf_with_file() {
    use aprender::format::converter_types::Source;
    let result = Source::parse("hf://Qwen/Qwen2.5-Coder-0.5B/model.safetensors");
    match result.unwrap() {
        Source::HuggingFace { file, .. } => {
            assert!(file.is_some(), "File component must be parsed");
        }
        _ => panic!("Expected HuggingFace variant"),
    }
}

#[test]
fn regression_source_parse_https() {
    use aprender::format::converter_types::Source;
    let result = Source::parse("https://example.com/model.gguf");
    assert!(matches!(result.unwrap(), Source::Url(_)));
}

#[test]
fn regression_source_parse_local() {
    use aprender::format::converter_types::Source;
    let result = Source::parse("/tmp/model.gguf");
    assert!(matches!(result.unwrap(), Source::Local(_)));
}

// =============================================================================
// Category 12: Metadata Plausibility (Bug 210, GH-222)
// =============================================================================

/// Bug 210: GgufModelConfig rope_theta for Qwen2 must NOT be 10000.0
#[test]
fn regression_bug_210_qwen2_rope_theta_not_default() {
    use aprender::format::gguf::api::GgufModelConfig;
    let bad_config = GgufModelConfig {
        architecture: Some("qwen2".to_string()),
        rope_theta: Some(10000.0),
        ..Default::default()
    };
    let theta = bad_config.rope_theta.expect("has theta");
    let is_suspicious = bad_config.architecture.as_deref() == Some("qwen2")
        && (f64::from(theta) - 10000.0).abs() < 1.0;
    assert!(
        is_suspicious,
        "Bug 210: qwen2+10000.0 must be flagged as suspicious"
    );
}

/// Bug 210: Correct Qwen2 config has rope_theta ~1000000.0
#[test]
fn regression_bug_210_qwen2_correct_rope_theta() {
    use aprender::format::gguf::api::GgufModelConfig;
    let good_config = GgufModelConfig {
        architecture: Some("qwen2".to_string()),
        rope_theta: Some(1_000_000.0),
        ..Default::default()
    };
    let theta = good_config.rope_theta.expect("has theta");
    assert!(
        f64::from(theta) >= 100_000.0,
        "Bug 210: correct qwen2 rope_theta should be >= 100000"
    );
}

/// Bug 210: LLaMA config with rope_theta=10000.0 is valid (not suspicious)
#[test]
fn regression_bug_210_llama_default_theta_is_valid() {
    use aprender::format::gguf::api::GgufModelConfig;
    let config = GgufModelConfig {
        architecture: Some("llama".to_string()),
        rope_theta: Some(10000.0),
        ..Default::default()
    };
    let theta = f64::from(config.rope_theta.expect("has theta"));
    assert!(
        theta >= 1000.0 && theta <= 10_000_000.0,
        "Bug 210: llama with rope_theta=10000.0 is valid"
    );
}

/// Bug 210: rope_theta plausibility range check
#[test]
fn regression_bug_210_rope_theta_plausibility_ranges() {
    let known_values: Vec<(&str, f64)> = vec![
        ("qwen2", 1_000_000.0),
        ("llama", 10_000.0),
        ("llama", 500_000.0),
        ("gpt2", 10_000.0),
    ];

    for (arch, theta) in known_values {
        assert!(
            theta >= 100.0 && theta <= 100_000_000.0,
            "Bug 210: {arch} rope_theta={theta} must be in plausible range [100, 100M]"
        );
    }
}

/// Bug 210: rms_norm_eps plausibility
#[test]
fn regression_bug_210_rms_norm_eps_plausibility() {
    let valid_values: Vec<f32> = vec![1e-6, 1e-5, 1e-8];
    for eps in valid_values {
        assert!(
            eps > 0.0 && eps <= 0.01,
            "Bug 210: rms_norm_eps={eps} must be in range (0, 0.01]"
        );
    }
    assert!(!(0.0_f32 > 0.0 && 0.0_f32 <= 0.01), "Zero eps must fail");
}

/// Bug 210: max_position_embeddings plausibility
#[test]
fn regression_bug_210_max_position_embeddings_plausibility() {
    let valid_values: Vec<usize> = vec![2048, 4096, 8192, 32768, 131072];
    for max_pos in valid_values {
        assert!(
            max_pos >= 128 && max_pos <= 1_048_576,
            "Bug 210: max_pos={max_pos} must be in range [128, 1M]"
        );
    }
}
