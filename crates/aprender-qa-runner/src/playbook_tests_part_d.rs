#[test]
fn test_size_category_auto_alignment_from_family_yaml() {
    use crate::family_contract::FamilyContract;

    // FALSIFY-FAM-001: Size category alignment
    let yaml = r#"
family: qwen2
size_variants:
  7b:
    parameters: "7B"
    hidden_dim: 3584
    num_layers: 28
    num_heads: 28
certification:
  size_categories:
    0.5b: tiny
    1.5b: small
    3b: small
    7b: medium
    14b: large
"#;
    let contract = FamilyContract::from_yaml(yaml).expect("parse");

    // Start with default (Tiny)
    let mut config = ModelConfig {
        hf_repo: "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::Tiny, // default
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    // Populate from contract with 7b size
    let result = config.populate_from_family_contract(&contract, "7b");
    assert!(result);

    // PMAT-270: Verify size_category auto-set to Medium (from 7b -> medium mapping)
    assert_eq!(config.size_category, SizeCategory::Medium);
}

#[test]
fn test_size_category_explicit_not_overridden() {
    use crate::family_contract::FamilyContract;

    let yaml = r#"
family: qwen2
size_variants:
  7b:
    parameters: "7B"
    hidden_dim: 3584
    num_layers: 28
    num_heads: 28
certification:
  size_categories:
    7b: medium
"#;
    let contract = FamilyContract::from_yaml(yaml).expect("parse");

    // Explicitly set to Large (user override)
    let mut config = ModelConfig {
        hf_repo: "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::Large, // explicitly set, not default
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    // Populate from contract
    config.populate_from_family_contract(&contract, "7b");

    // Should NOT override explicit setting - Large remains Large
    assert_eq!(config.size_category, SizeCategory::Large);
}

#[test]
fn test_size_category_from_str_lowercase() {
    assert_eq!(
        SizeCategory::from_str_lowercase("tiny").unwrap(),
        SizeCategory::Tiny
    );
    assert_eq!(
        SizeCategory::from_str_lowercase("small").unwrap(),
        SizeCategory::Small
    );
    assert_eq!(
        SizeCategory::from_str_lowercase("medium").unwrap(),
        SizeCategory::Medium
    );
    assert_eq!(
        SizeCategory::from_str_lowercase("large").unwrap(),
        SizeCategory::Large
    );
    assert_eq!(
        SizeCategory::from_str_lowercase("xlarge").unwrap(),
        SizeCategory::Xlarge
    );
    assert_eq!(
        SizeCategory::from_str_lowercase("huge").unwrap(),
        SizeCategory::Huge
    );

    // Case insensitive
    assert_eq!(
        SizeCategory::from_str_lowercase("TINY").unwrap(),
        SizeCategory::Tiny
    );
    assert_eq!(
        SizeCategory::from_str_lowercase("Medium").unwrap(),
        SizeCategory::Medium
    );

    // Invalid
    let err = SizeCategory::from_str_lowercase("invalid").unwrap_err();
    assert!(err.to_string().contains("Invalid size category"));
}

#[test]
fn test_size_category_no_certification_config() {
    use crate::family_contract::FamilyContract;

    // No certification section at all
    let yaml = r#"
family: qwen2
size_variants:
  0.5b:
    parameters: "0.5B"
    hidden_dim: 896
    num_layers: 24
    num_heads: 14
"#;
    let contract = FamilyContract::from_yaml(yaml).expect("parse");

    let mut config = ModelConfig {
        hf_repo: "test".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::Tiny, // default
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    config.populate_from_family_contract(&contract, "0.5b");

    // Should remain default since no certification config
    assert_eq!(config.size_category, SizeCategory::Tiny);
}

// ── Playbook deserialize_bool_or_string coverage ──────────────────────

/// Verify FormatValidationConfig deserializes `enabled: "true"` string as true
#[test]
fn test_format_validation_config_string_true() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
differential_tests:
  format_validation:
    enabled: "true"
    checks: ["dtype_mapping"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let diff = playbook.differential_tests.expect("should have differential_tests");
    let fv = diff.format_validation.expect("should have format_validation");
    assert!(fv.enabled);
}

/// Verify FormatValidationConfig deserializes `enabled: "yes"` string as true
#[test]
fn test_format_validation_config_string_yes() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
differential_tests:
  format_validation:
    enabled: "yes"
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let diff = playbook.differential_tests.expect("should have differential_tests");
    let fv = diff.format_validation.expect("should have format_validation");
    assert!(fv.enabled);
}

/// Verify FormatValidationConfig deserializes `enabled: "false"` string as false
#[test]
fn test_format_validation_config_string_false() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
differential_tests:
  format_validation:
    enabled: "false"
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let diff = playbook.differential_tests.expect("should have differential_tests");
    let fv = diff.format_validation.expect("should have format_validation");
    assert!(!fv.enabled);
}

/// Verify playbook rejects invalid string for bool fields
#[test]
fn test_format_validation_config_string_invalid() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
differential_tests:
  format_validation:
    enabled: "maybe"
"#;
    let result = Playbook::from_yaml(yaml);
    assert!(result.is_err());
}

// ── GH-6/AC-2: Ollama parity config tests ────────────────────────────

#[test]
fn test_playbook_with_ollama_parity() {
    let yaml = r#"
name: ollama-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
  model_tag: "qwen2.5-coder:7b-instruct-q4_k_m"
  quantizations: ["q4_k_m", "q6_k"]
  prompts: ["What is 2+2?", "def hello():"]
  temperature: 0.0
  min_perf_ratio: 0.9
  gates: ["F-OLLAMA-001", "F-OLLAMA-002"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let ollama = playbook.ollama_parity.expect("Should have ollama parity");

    assert!(ollama.enabled);
    assert_eq!(
        ollama.model_tag,
        Some("qwen2.5-coder:7b-instruct-q4_k_m".to_string())
    );
    assert_eq!(ollama.quantizations.len(), 2);
    assert_eq!(ollama.prompts.len(), 2);
    assert!((ollama.temperature - 0.0).abs() < f64::EPSILON);
    assert!((ollama.min_perf_ratio - 0.9).abs() < f64::EPSILON);
    assert_eq!(ollama.gates.len(), 2);
}

#[test]
fn test_playbook_without_ollama_parity() {
    let yaml = r#"
name: no-ollama
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    assert!(playbook.ollama_parity.is_none());
}

#[test]
fn test_ollama_parity_config_defaults() {
    let yaml = r#"
name: ollama-defaults
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
ollama_parity:
  enabled: true
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let ollama = playbook.ollama_parity.expect("should exist");

    assert!(ollama.enabled);
    assert!(ollama.model_tag.is_none());
    assert_eq!(ollama.quantizations, vec!["q4_k_m"]);
    assert_eq!(ollama.prompts, vec!["What is 2+2?"]);
    assert!((ollama.temperature - 0.0).abs() < f64::EPSILON);
    assert!((ollama.min_perf_ratio - 0.8).abs() < f64::EPSILON);
    assert!(ollama.gates.is_empty());
}

/// Verify from_yaml rejects empty hf_repo
#[test]
fn test_validation_empty_hf_repo() {
    let yaml = r#"
name: bad-playbook
version: "1.0.0"
model:
  hf_repo: ""
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let result = Playbook::from_yaml(yaml);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("hf_repo"));
}

/// Verify from_yaml rejects empty modalities
#[test]
fn test_validation_empty_modalities() {
    let yaml = r#"
name: bad-playbook
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: []
  backends: [cpu]
  scenario_count: 1
"#;
    let result = Playbook::from_yaml(yaml);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("modalities"));
}

/// Verify from_yaml rejects scenario_count of 0
#[test]
fn test_validation_zero_scenario_count() {
    let yaml = r#"
name: bad-playbook
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 0
"#;
    let result = Playbook::from_yaml(yaml);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("scenario_count"));
}
