#[test]
fn test_validate_1d_layer_tensors_with_layers() {
    let config = LayoutModelConfig {
        num_hidden_layers: Some(2),
        hidden_size: Some(4096),
        ..LayoutModelConfig::default()
    };

    let spec = make_spec("model.layers.{n}.input_layernorm.weight", "[hidden]", false);

    let mut all_tensors = HashMap::new();
    all_tensors.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        vec![4096],
    );
    all_tensors.insert(
        "model.layers.1.input_layernorm.weight".to_string(),
        vec![4096],
    );

    let mut results = Vec::new();
    validate_1d_layer_tensors(
        "model.layers.{n}.input_layernorm.weight",
        &all_tensors,
        &config,
        &spec,
        &mut results,
    );

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.passed));
}

#[test]
fn test_validate_1d_layer_tensors_invalid_shape() {
    let config = LayoutModelConfig {
        num_hidden_layers: Some(1),
        hidden_size: Some(4096),
        ..LayoutModelConfig::default()
    };

    let spec = make_spec("model.layers.{n}.norm.weight", "[hidden]", false);

    let mut all_tensors = HashMap::new();
    all_tensors.insert("model.layers.0.norm.weight".to_string(), vec![9999]);

    let mut results = Vec::new();
    validate_1d_layer_tensors(
        "model.layers.{n}.norm.weight",
        &all_tensors,
        &config,
        &spec,
        &mut results,
    );

    assert_eq!(results.len(), 1);
    assert!(!results[0].passed);
}

// ========================================================================
// 13. get_validation_rules
// ========================================================================

#[test]
fn test_get_validation_rules_returns_rules() {
    let mut contract = make_contract();
    contract.validation_rules = vec![
        ValidationRule {
            id: "F-LAYOUT-CONTRACT-001".to_string(),
            name: "2D transpose".to_string(),
            description: "All 2D weights are transposed".to_string(),
            severity: "P0".to_string(),
            critical: true,
            reference: None,
        },
        ValidationRule {
            id: "F-LAYOUT-CONTRACT-002".to_string(),
            name: "lm_head shape".to_string(),
            description: "lm_head shape matches".to_string(),
            severity: "P0".to_string(),
            critical: true,
            reference: Some("GH-202".to_string()),
        },
    ];

    let rules = get_validation_rules(&contract);
    assert_eq!(rules.len(), 2);
    assert_eq!(rules[0].id, "F-LAYOUT-CONTRACT-001");
    assert_eq!(rules[1].id, "F-LAYOUT-CONTRACT-002");
}

#[test]
fn test_get_validation_rules_empty() {
    let contract = make_contract();
    let rules = get_validation_rules(&contract);
    assert!(rules.is_empty());
}

// ========================================================================
// 14. collect_tensor_metadata with parse error
// ========================================================================

#[test]
fn test_collect_tensor_metadata_with_parse_error() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();

    // Create a corrupt safetensors file
    let bad_file = dir.path().join("corrupt.safetensors");
    let bad_header = b"not valid json at all";
    let header_len = bad_header.len() as u64;
    let mut file = std::fs::File::create(&bad_file).unwrap();
    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(bad_header).unwrap();

    let mut results = Vec::new();
    let tensors = collect_tensor_metadata(dir.path(), &mut results);

    assert!(tensors.is_empty());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].rule_id, "PARSE-ERROR");
    assert!(!results[0].passed);
}

#[test]
fn test_collect_tensor_metadata_valid() {
    let dir = tempfile::tempdir().unwrap();
    create_test_safetensors(
        &dir.path().join("model.safetensors"),
        &[("weight.a", &[10, 20]), ("weight.b", &[30])],
    );

    let mut results = Vec::new();
    let tensors = collect_tensor_metadata(dir.path(), &mut results);

    assert!(results.is_empty());
    assert_eq!(tensors.len(), 2);
    assert_eq!(tensors["weight.a"], vec![10, 20]);
    assert_eq!(tensors["weight.b"], vec![30]);
}

// ========================================================================
// 15. validate_lm_head (orchestrator) with/without lm_head in tensors
// ========================================================================

#[test]
fn test_validate_lm_head_with_tensors() {
    let config = make_config_full();
    let contract = make_contract();

    let mut all_tensors = HashMap::new();
    all_tensors.insert("lm_head.weight".to_string(), vec![32000, 4096]);

    let mut results = Vec::new();
    let mut critical_failures = Vec::new();

    validate_lm_head(
        &all_tensors,
        &config,
        &contract,
        &mut results,
        &mut critical_failures,
    );

    assert_eq!(results.len(), 1);
    assert!(results[0].passed);
    assert!(critical_failures.is_empty());
}

#[test]
fn test_validate_lm_head_with_invalid_tensors() {
    let config = make_config_full();
    let contract = make_contract();

    let mut all_tensors = HashMap::new();
    // Transposed shape => mismatch
    all_tensors.insert("lm_head.weight".to_string(), vec![4096, 32000]);

    let mut results = Vec::new();
    let mut critical_failures = Vec::new();

    validate_lm_head(
        &all_tensors,
        &config,
        &contract,
        &mut results,
        &mut critical_failures,
    );

    assert_eq!(results.len(), 1);
    assert!(!results[0].passed);
    assert_eq!(critical_failures.len(), 1);
}

#[test]
fn test_validate_lm_head_without_lm_head() {
    let config = make_config_full();
    let contract = make_contract();

    let all_tensors = HashMap::new(); // no lm_head.weight

    let mut results = Vec::new();
    let mut critical_failures = Vec::new();

    validate_lm_head(
        &all_tensors,
        &config,
        &contract,
        &mut results,
        &mut critical_failures,
    );

    // Nothing happens when lm_head is absent
    assert!(results.is_empty());
    assert!(critical_failures.is_empty());
}

// ========================================================================
// 16. validate_2d_tensors
// ========================================================================

#[test]
fn test_validate_2d_tensors_skips_non_transpose() {
    let mut contract = make_contract();
    // Insert a non-transpose tensor => should be skipped
    contract.tensors.insert(
        "norm".to_string(),
        make_spec("model.norm.weight", "[hidden]", false),
    );

    let all_tensors = HashMap::new();
    let config = make_config_full();
    let mut results = Vec::new();

    validate_2d_tensors(&contract, &all_tensors, &config, &mut results);
    assert!(results.is_empty());
}

#[test]
fn test_validate_2d_tensors_processes_layer_pattern() {
    let mut contract = make_contract();
    contract.tensors.insert(
        "q_proj".to_string(),
        make_spec(
            "model.layers.{n}.self_attn.q_proj.weight",
            "[vocab, hidden]",
            true,
        ),
    );

    let config = LayoutModelConfig {
        num_hidden_layers: Some(1),
        vocab_size: Some(100),
        hidden_size: Some(200),
        ..LayoutModelConfig::default()
    };

    let mut all_tensors = HashMap::new();
    all_tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        vec![100, 200],
    );

    let mut results = Vec::new();
    validate_2d_tensors(&contract, &all_tensors, &config, &mut results);

    assert_eq!(results.len(), 1);
    assert!(results[0].passed);
}

#[test]
fn test_validate_2d_tensors_processes_single() {
    let mut contract = make_contract();
    contract.tensors.insert(
        "embed".to_string(),
        make_spec("model.embed_tokens.weight", "[vocab, hidden]", true),
    );

    let config = make_config_full();

    let mut all_tensors = HashMap::new();
    all_tensors.insert("model.embed_tokens.weight".to_string(), vec![32000, 4096]);

    let mut results = Vec::new();
    validate_2d_tensors(&contract, &all_tensors, &config, &mut results);

    assert_eq!(results.len(), 1);
    assert!(results[0].passed);
}

#[test]
fn test_validate_2d_tensors_missing_tensor() {
    let mut contract = make_contract();
    contract.tensors.insert(
        "embed".to_string(),
        make_spec("model.embed_tokens.weight", "[vocab, hidden]", true),
    );

    let config = make_config_full();
    let all_tensors = HashMap::new(); // no tensors present

    let mut results = Vec::new();
    validate_2d_tensors(&contract, &all_tensors, &config, &mut results);

    // Tensor not found => not validated (no result added)
    assert!(results.is_empty());
}

// ========================================================================
// 17. validate_1d_tensors
// ========================================================================

#[test]
fn test_validate_1d_tensors_skips_transpose() {
    let mut contract = make_contract();
    // Insert a transpose=true tensor => should be skipped by 1D validation
    contract.tensors.insert(
        "proj".to_string(),
        make_spec("model.proj.weight", "[vocab, hidden]", true),
    );

    let all_tensors = HashMap::new();
    let config = make_config_full();
    let mut results = Vec::new();

    validate_1d_tensors(&contract, &all_tensors, &config, &mut results);
    assert!(results.is_empty());
}

#[test]
fn test_validate_1d_tensors_processes_layer_pattern() {
    let mut contract = make_contract();
    contract.tensors.insert(
        "layernorm".to_string(),
        make_spec("model.layers.{n}.input_layernorm.weight", "[hidden]", false),
    );

    let config = LayoutModelConfig {
        num_hidden_layers: Some(1),
        hidden_size: Some(4096),
        ..LayoutModelConfig::default()
    };

    let mut all_tensors = HashMap::new();
    all_tensors.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        vec![4096],
    );

    let mut results = Vec::new();
    validate_1d_tensors(&contract, &all_tensors, &config, &mut results);

    assert_eq!(results.len(), 1);
    assert!(results[0].passed);
}

#[test]
fn test_validate_1d_tensors_processes_single() {
    let mut contract = make_contract();
    contract.tensors.insert(
        "norm".to_string(),
        make_spec("model.norm.weight", "[hidden]", false),
    );

    let config = make_config_full();

    let mut all_tensors = HashMap::new();
    all_tensors.insert("model.norm.weight".to_string(), vec![4096]);

    let mut results = Vec::new();
    validate_1d_tensors(&contract, &all_tensors, &config, &mut results);

    assert_eq!(results.len(), 1);
    assert!(results[0].passed);
}

#[test]
fn test_validate_1d_tensors_missing_tensor() {
    let mut contract = make_contract();
    contract.tensors.insert(
        "norm".to_string(),
        make_spec("model.norm.weight", "[hidden]", false),
    );

    let config = make_config_full();
    let all_tensors = HashMap::new();

    let mut results = Vec::new();
    validate_1d_tensors(&contract, &all_tensors, &config, &mut results);

    assert!(results.is_empty());
}

/// validate_1d_layer_tensors: num_hidden_layers=None → UNVALIDATED early return (lines 437-450)
#[test]
fn test_validate_1d_layer_tensors_no_num_hidden_layers() {
    let config = LayoutModelConfig::default(); // num_hidden_layers = None
    let spec = make_spec("model.layers.{n}.input_layernorm.weight", "[hidden]", false);
    let all_tensors = HashMap::new();
    let mut results = Vec::new();

    validate_1d_layer_tensors(
        "model.layers.{n}.input_layernorm.weight",
        &all_tensors,
        &config,
        &spec,
        &mut results,
    );

    // Popper: missing num_hidden_layers → UNVALIDATED failure
    assert_eq!(results.len(), 1);
    assert!(!results[0].passed);
    assert!(
        results[0].details.contains("UNVALIDATED"),
        "Expected UNVALIDATED in details: {}",
        results[0].details
    );
    assert!(
        results[0].details.contains("num_hidden_layers"),
        "Expected num_hidden_layers in details: {}",
        results[0].details
    );
    assert_eq!(results[0].rule_id, "F-LAYOUT-CONTRACT-003");
}

// ========================================================================
// 18. run_all_validations end-to-end
// ========================================================================
