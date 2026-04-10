#[test]
fn test_effective_max_workers_respects_size() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  size_category: large
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    // Large model caps at 1 worker regardless of request
    assert_eq!(playbook.effective_max_workers(4), 1);
    assert_eq!(playbook.effective_max_workers(8), 1);
    assert_eq!(playbook.effective_max_workers(1), 1);
}

#[test]
fn test_effective_max_workers_small_model() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  size_category: small
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    // Small model allows up to 4 workers
    assert_eq!(playbook.effective_max_workers(4), 4);
    assert_eq!(playbook.effective_max_workers(8), 4); // capped at 4
    assert_eq!(playbook.effective_max_workers(2), 2); // respects lower request
}

#[test]
fn test_effective_max_workers_medium_model() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  size_category: medium
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    // Medium model caps at 2 workers
    assert_eq!(playbook.effective_max_workers(4), 2);
    assert_eq!(playbook.effective_max_workers(1), 1);
}

// ── PMAT-266 Naming convention tests ─────────────────────────────────

#[test]
fn test_validate_playbook_name_basic() {
    let result = validate_playbook_name("qwen2.5-coder-0.5b-mvp.playbook.yaml");
    assert!(result.is_ok());
    let parts = result.unwrap();
    assert_eq!(parts.family, "qwen2.5-coder");
    assert_eq!(parts.size, "0.5b");
    assert_eq!(parts.tier, Some("mvp".to_string()));
}

#[test]
fn test_validate_playbook_name_no_tier() {
    let result = validate_playbook_name("llama3.2-7b.playbook.yaml");
    assert!(result.is_ok());
    let parts = result.unwrap();
    assert_eq!(parts.family, "llama3.2");
    assert_eq!(parts.size, "7b");
    assert_eq!(parts.tier, None);
}

#[test]
fn test_validate_playbook_name_large_model() {
    let result = validate_playbook_name("deepseek-coder-v2-16b-full.playbook.yaml");
    assert!(result.is_ok());
    let parts = result.unwrap();
    assert_eq!(parts.family, "deepseek-coder-v2");
    assert_eq!(parts.size, "16b");
    assert_eq!(parts.tier, Some("full".to_string()));
}

#[test]
fn test_validate_playbook_name_various_tiers() {
    for tier in VALID_TIERS {
        let filename = format!("model-1b-{tier}.playbook.yaml");
        let result = validate_playbook_name(&filename);
        assert!(result.is_ok(), "Failed for tier: {tier}");
        assert_eq!(result.unwrap().tier, Some((*tier).to_string()));
    }
}

#[test]
fn test_validate_playbook_name_various_sizes() {
    let sizes = ["0.5b", "1b", "1.5b", "3b", "7b", "13b", "70b", "405b"];
    for size in sizes {
        let filename = format!("model-{size}.playbook.yaml");
        let result = validate_playbook_name(&filename);
        assert!(result.is_ok(), "Failed for size: {size}");
        assert_eq!(result.unwrap().size, size);
    }
}

#[test]
fn test_validate_playbook_name_invalid_no_size() {
    let result = validate_playbook_name("qwen2.5-coder-mvp.playbook.yaml");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("does not match naming convention"));
}

#[test]
fn test_validate_playbook_name_invalid_wrong_extension() {
    let result = validate_playbook_name("qwen2.5-coder-0.5b-mvp.yaml");
    assert!(result.is_err());
}

#[test]
fn test_validate_playbook_name_invalid_tier() {
    let result = validate_playbook_name("qwen2.5-coder-0.5b-unknown.playbook.yaml");
    assert!(result.is_err());
}

#[test]
fn test_validate_playbook_name_invalid_format() {
    let invalid_names = [
        "model.playbook.yaml",         // no size
        "model-big.playbook.yaml",     // invalid size format
        "model-7gb.playbook.yaml",     // wrong unit (gb instead of b)
        ".playbook.yaml",              // empty name
        "model-7b-test.playbook.yaml", // invalid tier
    ];
    for name in invalid_names {
        let result = validate_playbook_name(name);
        assert!(result.is_err(), "Expected error for: {name}");
    }
}

#[test]
fn test_validate_playbook_path() {
    let path = std::path::Path::new("/some/path/qwen2.5-coder-1.5b-mvp.playbook.yaml");
    let result = validate_playbook_path(path);
    assert!(result.is_ok());
    let parts = result.unwrap();
    assert_eq!(parts.family, "qwen2.5-coder");
    assert_eq!(parts.size, "1.5b");
    assert_eq!(parts.tier, Some("mvp".to_string()));
}

#[test]
fn test_playbook_name_parts_to_filename() {
    let parts = PlaybookNameParts {
        family: "qwen2.5-coder".to_string(),
        size: "0.5b".to_string(),
        tier: Some("mvp".to_string()),
    };
    assert_eq!(parts.to_filename(), "qwen2.5-coder-0.5b-mvp.playbook.yaml");

    let parts_no_tier = PlaybookNameParts {
        family: "llama3.2".to_string(),
        size: "7b".to_string(),
        tier: None,
    };
    assert_eq!(parts_no_tier.to_filename(), "llama3.2-7b.playbook.yaml");
}

#[test]
fn test_playbook_name_parts_eq() {
    let parts1 = PlaybookNameParts {
        family: "model".to_string(),
        size: "1b".to_string(),
        tier: Some("mvp".to_string()),
    };
    let parts2 = PlaybookNameParts {
        family: "model".to_string(),
        size: "1b".to_string(),
        tier: Some("mvp".to_string()),
    };
    assert_eq!(parts1, parts2);
}

#[test]
fn test_valid_tiers_constant() {
    assert_eq!(VALID_TIERS.len(), 8);
    assert!(VALID_TIERS.contains(&"dim-smoke"));
    assert!(VALID_TIERS.contains(&"mvp"));
    assert!(VALID_TIERS.contains(&"smoke"));
    assert!(VALID_TIERS.contains(&"quick"));
    assert!(VALID_TIERS.contains(&"ci"));
    assert!(VALID_TIERS.contains(&"full"));
    assert!(VALID_TIERS.contains(&"nightly"));
    assert!(VALID_TIERS.contains(&"release"));
}

#[test]
fn test_validate_playbook_name_dim_smoke() {
    let result = validate_playbook_name("qwen2.5-coder-0.5b-dim-smoke.playbook.yaml");
    assert!(result.is_ok());
    let parts = result.unwrap();
    assert_eq!(parts.family, "qwen2.5-coder");
    assert_eq!(parts.size, "0.5b");
    assert_eq!(parts.tier, Some("dim-smoke".to_string()));
}

// ── PMAT-269 Test matrix generation tests ────────────────────────────

#[test]
fn test_populate_from_family_contract() {
    use crate::family_contract::FamilyContract;

    // PMAT-270: Include certification.size_categories for auto-alignment test
    let yaml = r#"
family: qwen2
size_variants:
  0.5b:
    parameters: "0.5B"
    hidden_dim: 896
    num_layers: 24
    num_heads: 14
    num_kv_heads: 2
    vocab_size: 151936
    intermediate_dim: 4864
certification:
  size_categories:
    0.5b: tiny
    1.5b: small
    7b: medium
"#;
    let contract = FamilyContract::from_yaml(yaml).expect("parse");

    let mut config = ModelConfig {
        hf_repo: "Qwen/Qwen2.5-Coder-0.5B-Instruct".to_string(),
        local_path: None,
        formats: vec![Format::Gguf],
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

    // Populate from contract
    let result = config.populate_from_family_contract(&contract, "0.5b");
    assert!(result);

    // Verify values populated
    assert_eq!(config.family, Some("qwen2".to_string()));
    assert_eq!(config.size_variant, Some("0.5b".to_string()));
    assert_eq!(config.expected_hidden_dim, Some(896));
    assert_eq!(config.expected_num_layers, Some(24));
    assert_eq!(config.expected_num_heads, Some(14));
    assert_eq!(config.expected_num_kv_heads, Some(2));
    assert_eq!(config.expected_vocab_size, Some(151_936));
    assert_eq!(config.expected_intermediate_dim, Some(4864));
    // PMAT-270: Verify size_category auto-populated
    assert_eq!(config.size_category, SizeCategory::Tiny);
}

#[test]
fn test_populate_from_family_contract_missing_size() {
    use crate::family_contract::FamilyContract;

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
        size_category: SizeCategory::default(),
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    // Try to populate with non-existent size
    let result = config.populate_from_family_contract(&contract, "7b");
    assert!(!result);

    // Values should remain None
    assert!(config.expected_hidden_dim.is_none());
}

#[test]
fn test_has_expected_params() {
    let config_empty = ModelConfig {
        hf_repo: "test".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::default(),
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };
    assert!(!config_empty.has_expected_params());

    let config_with_params = ModelConfig {
        hf_repo: "test".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::default(),
        expected_hidden_dim: Some(896),
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };
    assert!(config_with_params.has_expected_params());
}

#[test]
fn test_validate_architecture_match() {
    let config = ModelConfig {
        hf_repo: "test".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::default(),
        expected_hidden_dim: Some(896),
        expected_num_layers: Some(24),
        expected_num_heads: Some(14),
        expected_num_kv_heads: Some(2),
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    // All match
    let mismatches = config.validate_architecture(896, 24, Some(14), Some(2));
    assert!(mismatches.is_empty());
}

#[test]
fn test_validate_architecture_mismatch() {
    let config = ModelConfig {
        hf_repo: "test".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::default(),
        expected_hidden_dim: Some(896),
        expected_num_layers: Some(24),
        expected_num_heads: Some(14),
        expected_num_kv_heads: Some(2),
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    // All wrong
    let mismatches = config.validate_architecture(1024, 12, Some(16), Some(4));
    assert_eq!(mismatches.len(), 4);
    assert!(mismatches[0].contains("hidden_dim"));
    assert!(mismatches[1].contains("num_layers"));
    assert!(mismatches[2].contains("num_heads"));
    assert!(mismatches[3].contains("num_kv_heads"));
}

#[test]
fn test_validate_architecture_partial_expected() {
    let config = ModelConfig {
        hf_repo: "test".to_string(),
        local_path: None,
        formats: vec![],
        quantizations: vec![],
        size_category: SizeCategory::default(),
        expected_hidden_dim: Some(896),
        expected_num_layers: None, // Not set
        expected_num_heads: None,  // Not set
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    // Only hidden_dim is checked
    let mismatches = config.validate_architecture(896, 999, Some(999), Some(999));
    assert!(mismatches.is_empty()); // hidden_dim matches, others not checked
}

// ── PMAT-270: Size category auto-alignment tests ─────────────────────────
