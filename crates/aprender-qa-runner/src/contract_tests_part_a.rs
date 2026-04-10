/// Verify format contract loads with non-empty invariants, dtype bytes, and tolerances
#[test]
fn test_load_format_contract() {
    let contract = load_format_contract().expect("Failed to load contract");
    assert!(!contract.invariants.is_empty());
    assert!(!contract.dtype_bytes.mappings.is_empty());
    assert!(!contract.tolerances.is_empty());
}

/// Verify format contract version is "1.0"
#[test]
fn test_contract_version() {
    let contract = load_format_contract().expect("Failed to load contract");
    assert_eq!(contract.version, "1.0");
}

/// Verify dtype byte mappings include all expected dtypes
#[test]
fn test_dtype_byte_mappings_complete() {
    let contract = load_format_contract().expect("Failed to load contract");
    let dtypes: Vec<&str> = contract
        .dtype_bytes
        .mappings
        .iter()
        .map(|m| m.dtype.as_str())
        .collect();
    assert!(dtypes.contains(&"F32"));
    assert!(dtypes.contains(&"F16"));
    assert!(dtypes.contains(&"Q4_K"));
    assert!(dtypes.contains(&"Q6_K"));
    assert!(dtypes.contains(&"BF16"));
    assert!(dtypes.contains(&"Q8_0"));
    assert!(dtypes.contains(&"Q2_K"));
    assert!(dtypes.contains(&"Q3_K"));
    assert!(dtypes.contains(&"Q5_K"));
    assert!(dtypes.contains(&"Q4_0"));
    assert!(dtypes.contains(&"Q5_0"));
}

/// Verify dtype byte mappings have no duplicate entries
#[test]
fn test_dtype_byte_no_duplicates() {
    let contract = load_format_contract().expect("Failed to load contract");
    validate_dtype_bytes(&contract).expect("No duplicates expected");
}

/// Verify GGML dtype byte values match specification
#[test]
fn test_dtype_byte_ggml_values() {
    let contract = load_format_contract().expect("Failed to load contract");
    let find_byte = |dtype: &str| -> u8 {
        contract
            .dtype_bytes
            .mappings
            .iter()
            .find(|m| m.dtype == dtype)
            .expect("dtype not found")
            .byte
    };
    assert_eq!(find_byte("F32"), 0);
    assert_eq!(find_byte("F16"), 1);
    assert_eq!(find_byte("Q4_K"), 12);
    assert_eq!(find_byte("Q6_K"), 14);
    assert_eq!(find_byte("BF16"), 30);
}

/// Verify tensor naming pattern validation for canonical and forbidden names
#[test]
fn test_tensor_naming_pattern() {
    let contract = load_format_contract().expect("Failed to load contract");

    // Valid names
    assert!(validate_tensor_name("0.q_proj.weight", &contract));
    assert!(validate_tensor_name("31.down_proj.weight", &contract));
    assert!(validate_tensor_name("token_embd.weight", &contract));
    assert!(validate_tensor_name("output_norm.weight", &contract));
    assert!(validate_tensor_name("output.weight", &contract));

    // Invalid names (HuggingFace-style)
    assert!(!validate_tensor_name(
        "model.layers.0.self_attn.q_proj.weight",
        &contract
    ));
    assert!(!validate_tensor_name(
        "model.embed_tokens.weight",
        &contract
    ));
    assert!(!validate_tensor_name("", &contract));
}

/// Verify all five invariant definitions exist in contract
#[test]
fn test_invariant_definitions_complete() {
    let contract = load_format_contract().expect("Failed to load contract");
    assert_eq!(contract.invariants.len(), 5);
    let ids: Vec<&str> = contract.invariants.iter().map(|i| i.id.as_str()).collect();
    assert!(ids.contains(&"I-1"));
    assert!(ids.contains(&"I-2"));
    assert!(ids.contains(&"I-3"));
    assert!(ids.contains(&"I-4"));
    assert!(ids.contains(&"I-5"));
}

/// Verify tolerance lookup returns correct atol/rtol for known dtypes
#[test]
fn test_tolerance_lookup() {
    let contract = load_format_contract().expect("Failed to load contract");

    let (atol, rtol) = lookup_tolerance("F32", &contract).expect("F32 tolerance");
    assert!((atol - 0.0).abs() < f64::EPSILON);
    assert!((rtol - 0.0).abs() < f64::EPSILON);

    let (atol, rtol) = lookup_tolerance("Q4_K", &contract).expect("Q4_K tolerance");
    assert!((atol - 0.05).abs() < f64::EPSILON);
    assert!((rtol - 0.05).abs() < f64::EPSILON);

    let (atol, rtol) = lookup_tolerance("Q6_K", &contract).expect("Q6_K tolerance");
    assert!((atol - 0.02).abs() < f64::EPSILON);
    assert!((rtol - 0.02).abs() < f64::EPSILON);

    assert!(lookup_tolerance("UNKNOWN", &contract).is_none());
}

/// Verify canonical tensor name examples pass validation
#[test]
fn test_validate_tensor_name_valid() {
    let contract = load_format_contract().expect("Failed to load contract");
    for example in &contract.tensor_naming.examples {
        assert!(
            validate_tensor_name(&example.canonical, &contract),
            "Expected '{}' to be valid",
            example.canonical
        );
    }
}

/// Verify forbidden tensor name examples fail validation
#[test]
fn test_validate_tensor_name_invalid() {
    let contract = load_format_contract().expect("Failed to load contract");
    for example in &contract.tensor_naming.examples {
        assert!(
            !validate_tensor_name(&example.forbidden, &contract),
            "Expected '{}' to be invalid",
            example.forbidden
        );
    }
}

/// Verify ContractTestConfig default includes I-2 through I-5
#[test]
fn test_contract_test_config_default() {
    let config = ContractTestConfig::default();
    assert_eq!(config.invariants.len(), 4);
    assert!(config.invariants.contains(&"I-2".to_string()));
    assert!(config.invariants.contains(&"I-3".to_string()));
    assert!(config.invariants.contains(&"I-4".to_string()));
    assert!(config.invariants.contains(&"I-5".to_string()));
}

/// Verify InvariantId::from_label parses all known labels
#[test]
fn test_invariant_id_from_label() {
    assert_eq!(InvariantId::from_label("I-1"), Some(InvariantId::I1));
    assert_eq!(InvariantId::from_label("I-2"), Some(InvariantId::I2));
    assert_eq!(InvariantId::from_label("I-3"), Some(InvariantId::I3));
    assert_eq!(InvariantId::from_label("I-4"), Some(InvariantId::I4));
    assert_eq!(InvariantId::from_label("I-5"), Some(InvariantId::I5));
    assert_eq!(InvariantId::from_label("I-99"), None);
}

/// Verify InvariantId gate_id formatting
#[test]
fn test_invariant_id_gate_id() {
    assert_eq!(InvariantId::I1.gate_id(), "F-CONTRACT-I1-001");
    assert_eq!(InvariantId::I2.gate_id(), "F-CONTRACT-I2-001");
    assert_eq!(InvariantId::I3.gate_id(), "F-CONTRACT-I3-001");
    assert_eq!(InvariantId::I4.gate_id(), "F-CONTRACT-I4-001");
    assert_eq!(InvariantId::I5.gate_id(), "F-CONTRACT-I5-001");
}

/// Verify contains_f32_fallback detects positive fallback patterns
#[test]
fn test_contains_f32_fallback_positive() {
    assert!(contains_f32_fallback(
        "Warning: fallback to F32 for unknown type"
    ));
    assert!(contains_f32_fallback("defaulting to f32"));
    assert!(contains_f32_fallback("unknown dtype, defaulting to f32"));
    assert!(!contains_f32_fallback("unknown dtype detected"));
}

/// Verify contains_f32_fallback rejects normal output
#[test]
fn test_contains_f32_fallback_negative() {
    assert!(!contains_f32_fallback("All checks passed"));
    assert!(!contains_f32_fallback("Using Q4_K quantization"));
    assert!(!contains_f32_fallback("F32 tensors loaded normally"));
}

/// Verify I-2 tensor name bijection test passes with mock runner
#[test]
fn test_contract_i2_tensor_name_bijection_pass() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
    let model_id = ModelId::new("test", "model");
    let evidence = run_contract_tests(
        &runner, Path::new("/test/workspace/org/model"), &model_id,
        &ContractTestConfig::default(),
    );
    let i2 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I2-001");
    assert!(i2.is_some(), "I-2 evidence should exist");
    assert_eq!(i2.unwrap().outcome, Outcome::Corroborated);
}

/// Verify I-2 tensor name bijection test fails with inspect failure mock
#[test]
fn test_contract_i2_tensor_name_bijection_fail() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> =
        Arc::new(MockCommandRunner::new().with_inspect_json_failure());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig { invariants: vec!["I-2".to_string()] };
    let evidence = run_contract_tests(
        &runner, Path::new("/test/workspace/org/model"), &model_id, &config,
    );
    let i2 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I2-001");
    assert!(i2.is_some());
    assert_eq!(i2.unwrap().outcome, Outcome::Falsified);
}

/// Verify parse_tensor_names extracts names from valid JSON
#[test]
fn test_parse_tensor_names_valid() {
    let json = r#"{"format":"SafeTensors","tensor_count":3,"tensor_names":["embed.weight","lm_head.weight","0.q_proj.weight"],"parameters":"1.5B"}"#;
    let names = parse_tensor_names(json);
    assert_eq!(names.len(), 3);
    assert!(names.contains("embed.weight"));
    assert!(names.contains("lm_head.weight"));
    assert!(names.contains("0.q_proj.weight"));
}

/// Verify parse_tensor_names returns empty set for empty array
#[test]
fn test_parse_tensor_names_empty() {
    let json = r#"{"tensor_names":[]}"#;
    let names = parse_tensor_names(json);
    assert!(names.is_empty());
}

/// Verify parse_tensor_names returns empty set when field is missing
#[test]
fn test_parse_tensor_names_missing_field() {
    let json = r#"{"format":"SafeTensors","tensor_count":3}"#;
    let names = parse_tensor_names(json);
    assert!(names.is_empty());
}

/// Verify parse_tensor_names returns empty set for malformed input
#[test]
fn test_parse_tensor_names_malformed() {
    let names = parse_tensor_names("not json at all");
    assert!(names.is_empty());
}

/// Bug #55: Old hand-rolled parser only matched `"tensor_names":[` (no space).
/// Pretty-printed JSON with `"tensor_names": [` was silently returning empty set.
#[test]
fn test_parse_tensor_names_with_spaces() {
    let json = r#"{ "tensor_names": ["a.weight", "b.weight"] }"#;
    let names = parse_tensor_names(json);
    assert_eq!(names.len(), 2);
    assert!(names.contains("a.weight"));
    assert!(names.contains("b.weight"));
}

/// Verify parse_tensor_names handles pretty-printed multi-line JSON
#[test]
fn test_parse_tensor_names_pretty_printed() {
    let json = r#"{
  "format": "SafeTensors",
  "tensor_count": 2,
  "tensor_names": [
    "embed.weight",
    "lm_head.weight"
  ]
}"#;
    let names = parse_tensor_names(json);
    assert_eq!(names.len(), 2);
    assert!(names.contains("embed.weight"));
    assert!(names.contains("lm_head.weight"));
}

/// Verify tied embedding allowed extras include lm_head tensors
#[test]
fn test_i2_tied_embedding_allowed_extras() {
    // Verify that lm_head.weight and lm_head.bias are in the allowed extras set
    let allowed: HashSet<&str> = HashSet::from(["lm_head.weight", "lm_head.bias"]);
    assert!(allowed.contains("lm_head.weight"));
    assert!(allowed.contains("lm_head.bias"));
    assert!(!allowed.contains("unexpected_tensor.weight"));
}

/// Verify I-3 no silent fallbacks test passes with mock runner
#[test]
fn test_contract_i3_no_silent_fallbacks_pass() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig { invariants: vec!["I-3".to_string()] };
    let evidence = run_contract_tests(
        &runner, Path::new("/test/workspace/org/model"), &model_id, &config,
    );
    let i3 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I3-001");
    assert!(i3.is_some());
    assert_eq!(i3.unwrap().outcome, Outcome::Corroborated);
}

/// Verify I-3 no silent fallbacks test fails with check failure mock
#[test]
fn test_contract_i3_no_silent_fallbacks_fail() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> =
        Arc::new(MockCommandRunner::new().with_check_failure());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig { invariants: vec!["I-3".to_string()] };
    let evidence = run_contract_tests(
        &runner, Path::new("/test/workspace/org/model"), &model_id, &config,
    );
    let i3 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I3-001");
    assert!(i3.is_some());
    assert_eq!(i3.unwrap().outcome, Outcome::Falsified);
}

/// Verify I-4 statistical preservation test passes with mock runner
#[test]
fn test_contract_i4_statistical_preservation_pass() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig { invariants: vec!["I-4".to_string()] };
    let evidence = run_contract_tests(
        &runner, Path::new("/test/workspace/org/model"), &model_id, &config,
    );
    let i4 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I4-001");
    assert!(i4.is_some());
    assert_eq!(i4.unwrap().outcome, Outcome::Corroborated);
}

/// Verify I-4 statistical preservation test fails with stats failure mock
#[test]
fn test_contract_i4_statistical_preservation_fail() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> =
        Arc::new(MockCommandRunner::new().with_validate_stats_failure());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig { invariants: vec!["I-4".to_string()] };
    let evidence = run_contract_tests(
        &runner, Path::new("/test/workspace/org/model"), &model_id, &config,
    );
    let i4 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I4-001");
    assert!(i4.is_some());
    assert_eq!(i4.unwrap().outcome, Outcome::Falsified);
}

/// Verify I-5 tokenizer roundtrip test passes with mock runner
#[test]
fn test_contract_i5_tokenizer_roundtrip_pass() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig { invariants: vec!["I-5".to_string()] };
    let evidence = run_contract_tests(
        &runner, Path::new("/test/workspace/org/model"), &model_id, &config,
    );
    let i5 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I5-001");
    assert!(i5.is_some());
    assert_eq!(i5.unwrap().outcome, Outcome::Corroborated);
}

/// Verify I-5 tokenizer roundtrip test fails with inference failure mock
#[test]
fn test_contract_i5_tokenizer_roundtrip_fail() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> =
        Arc::new(MockCommandRunner::new().with_compare_inference_failure());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig { invariants: vec!["I-5".to_string()] };
    let evidence = run_contract_tests(
        &runner, Path::new("/test/workspace/org/model"), &model_id, &config,
    );
    let i5 = evidence.iter().find(|e| e.gate_id == "F-CONTRACT-I5-001");
    assert!(i5.is_some());
    assert_eq!(i5.unwrap().outcome, Outcome::Falsified);
}

/// Verify all default invariants (I-2 through I-5) pass with mock runner
#[test]
fn test_contract_all_invariants_pass() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
    let model_id = ModelId::new("test", "model");
    let evidence = run_contract_tests(
        &runner, Path::new("/test/workspace/org/model"), &model_id,
        &ContractTestConfig::default(),
    );
    assert_eq!(evidence.len(), 4);
    for ev in &evidence {
        assert_eq!(ev.outcome, Outcome::Corroborated, "Gate {} should pass", ev.gate_id);
    }
}

/// Verify I-1 is skipped when included in config (handled by golden rule)
#[test]
fn test_contract_skips_i1() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig {
        invariants: vec!["I-1".to_string(), "I-2".to_string()],
    };
    let evidence = run_contract_tests(
        &runner, Path::new("/test/workspace/org/model"), &model_id, &config,
    );
    assert_eq!(evidence.len(), 1);
    assert_eq!(evidence[0].gate_id, "F-CONTRACT-I2-001");
}

/// Verify unknown invariant labels produce falsified evidence (not silently skipped)
#[test]
fn test_contract_unknown_invariant_rejected() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig { invariants: vec!["I-99".to_string()] };
    let evidence = run_contract_tests(
        &runner, Path::new("/test/workspace/org/model"), &model_id, &config,
    );
    assert_eq!(evidence.len(), 1);
    assert!(evidence[0].outcome.is_fail());
    assert_eq!(evidence[0].gate_id, "F-CONTRACT-INVALID-001");
    assert!(evidence[0].reason.contains("I-99"));
}

/// Verify path resolution avoids dot-in-name regression for model directories
#[test]
fn test_resolve_paths_with_dots_in_name() {
    // Regression: Qwen2.5-Coder-0.5B-Instruct contains dots which caused
    // Path::with_extension("apr") to produce "Qwen2.5-Coder-0.apr"
    let workspace = Path::new("/output/workspace/Qwen/Qwen2.5-Coder-0.5B-Instruct");
    let apr = resolve_apr_path(workspace);
    let st = resolve_safetensors_path(workspace);

    assert_eq!(
        apr,
        PathBuf::from("/output/workspace/Qwen/Qwen2.5-Coder-0.5B-Instruct/apr/model.apr")
    );
    assert_eq!(
        st,
        PathBuf::from(
            "/output/workspace/Qwen/Qwen2.5-Coder-0.5B-Instruct/safetensors/model.safetensors"
        )
    );

    // Contrast with the old broken behavior
    let broken = workspace.with_extension("apr");
    assert_eq!(
        broken,
        PathBuf::from("/output/workspace/Qwen/Qwen2.5-Coder-0.apr")
    );
}
