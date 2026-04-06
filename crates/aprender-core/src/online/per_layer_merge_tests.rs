use super::*;

// ============================================================================
// Layer rule matching tests
// ============================================================================

#[test]
fn test_match_layer_rule_exact() {
    let rules = vec![LayerRule {
        layer_pattern: "layers\\.0\\.".to_string(),
        strategy: "slerp".to_string(),
        weights: Some(vec![0.3]),
        scale: None,
    }];
    let result = match_layer_rule("model.layers.0.self_attn.weight", &rules);
    assert!(result.is_some());
    assert_eq!(result.unwrap().strategy, "slerp");
}

#[test]
fn test_match_layer_rule_attn_pattern() {
    let rules = vec![
        LayerRule {
            layer_pattern: "self_attn".to_string(),
            strategy: "slerp".to_string(),
            weights: None,
            scale: None,
        },
        LayerRule {
            layer_pattern: "mlp".to_string(),
            strategy: "average".to_string(),
            weights: None,
            scale: None,
        },
    ];
    let result = match_layer_rule("model.layers.5.self_attn.q_proj.weight", &rules);
    assert!(result.is_some());
    assert_eq!(result.unwrap().strategy, "slerp");

    let result = match_layer_rule("model.layers.5.mlp.gate_proj.weight", &rules);
    assert!(result.is_some());
    assert_eq!(result.unwrap().strategy, "average");
}

#[test]
fn test_match_layer_rule_no_match() {
    let rules = vec![LayerRule {
        layer_pattern: "nonexistent".to_string(),
        strategy: "slerp".to_string(),
        weights: None,
        scale: None,
    }];
    let result = match_layer_rule("model.layers.0.self_attn.weight", &rules);
    assert!(result.is_none());
}

#[test]
fn test_match_layer_rule_first_wins() {
    let rules = vec![
        LayerRule {
            layer_pattern: "layers".to_string(),
            strategy: "first".to_string(),
            weights: None,
            scale: None,
        },
        LayerRule {
            layer_pattern: "layers".to_string(),
            strategy: "second".to_string(),
            weights: None,
            scale: None,
        },
    ];
    let result = match_layer_rule("model.layers.0.weight", &rules);
    assert_eq!(result.unwrap().strategy, "first");
}

// ============================================================================
// YAML parsing tests
// ============================================================================

#[test]
fn test_parse_merge_yaml_basic() {
    let yaml = r#"
models:
  - path: model_a.safetensors
    weight: 0.7
  - path: model_b.safetensors
    weight: 0.3
output: merged.safetensors
default_strategy: average
"#;
    let config = parse_merge_yaml(yaml).unwrap();
    assert_eq!(config.models.len(), 2);
    assert_eq!(config.models[0].path, "model_a.safetensors");
    assert_eq!(config.models[0].weight, Some(0.7));
    assert_eq!(config.output, "merged.safetensors");
    assert_eq!(config.default_strategy, "average");
}

#[test]
fn test_parse_merge_yaml_no_weights() {
    let yaml = r#"
models:
  - path: a.safetensors
  - path: b.safetensors
output: out.safetensors
default_strategy: slerp
"#;
    let config = parse_merge_yaml(yaml).unwrap();
    assert_eq!(config.models.len(), 2);
    assert_eq!(config.models[0].weight, None);
}

#[test]
fn test_parse_merge_yaml_empty() {
    let result = parse_merge_yaml("");
    assert!(result.is_err());
}

// ============================================================================
// Validation tests
// ============================================================================

#[test]
fn test_validate_merge_config_ok() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: "a.safetensors".to_string(),
                weight: Some(0.5),
            },
            ModelSource {
                path: "b.safetensors".to_string(),
                weight: Some(0.5),
            },
        ],
        output: "out.safetensors".to_string(),
        default_strategy: "average".to_string(),
        layers: None,
    };
    assert!(validate_merge_config(&config).is_ok());
}

#[test]
fn test_validate_merge_config_too_few_models() {
    let config = MergeYamlConfig {
        models: vec![ModelSource {
            path: "a.safetensors".to_string(),
            weight: None,
        }],
        output: "out.safetensors".to_string(),
        default_strategy: "average".to_string(),
        layers: None,
    };
    assert!(validate_merge_config(&config).is_err());
}

#[test]
fn test_validate_merge_config_empty_output() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: "a.safetensors".to_string(),
                weight: None,
            },
            ModelSource {
                path: "b.safetensors".to_string(),
                weight: None,
            },
        ],
        output: String::new(),
        default_strategy: "average".to_string(),
        layers: None,
    };
    assert!(validate_merge_config(&config).is_err());
}

// ============================================================================
// LayerMergeConfig tests
// ============================================================================

#[test]
fn test_layer_merge_config() {
    let cfg = LayerMergeConfig {
        layer_rules: vec![LayerRule {
            layer_pattern: "attn".to_string(),
            strategy: "slerp".to_string(),
            weights: Some(vec![0.4]),
            scale: None,
        }],
        default_strategy: "average".to_string(),
        default_weights: vec![0.5, 0.5],
    };
    assert_eq!(cfg.layer_rules.len(), 1);
    assert_eq!(cfg.default_strategy, "average");
}

// ============================================================================
// Falsification tests
// ============================================================================

/// FALSIFY-PERLAYER-001: match_layer_rule returns None for empty rules.
#[test]
fn falsify_perlayer_001_empty_rules() {
    let result = match_layer_rule("model.layers.0.weight", &[]);
    assert!(result.is_none(), "Empty rules should match nothing");
}

/// FALSIFY-PERLAYER-002: Valid YAML always parses.
#[test]
fn falsify_perlayer_002_valid_yaml_parses() {
    let yamls = vec![
        "models:\n  - path: a.st\n  - path: b.st\noutput: out.st\ndefault_strategy: average\n",
        "models:\n  - path: x.apr\n    weight: 1.0\n  - path: y.apr\n    weight: 0.0\noutput: z.apr\ndefault_strategy: weighted\n",
    ];
    for yaml in yamls {
        let result = parse_merge_yaml(yaml);
        assert!(
            result.is_ok(),
            "Valid YAML should parse: {:?}",
            result.err()
        );
    }
}

/// FALSIFY-PERLAYER-003: Validation rejects invalid configs.
#[test]
fn falsify_perlayer_003_validation_rejects() {
    let invalid_configs = vec![
        // No models
        MergeYamlConfig {
            models: vec![],
            output: "out.st".to_string(),
            default_strategy: "average".to_string(),
            layers: None,
        },
        // Empty output
        MergeYamlConfig {
            models: vec![
                ModelSource {
                    path: "a".to_string(),
                    weight: None,
                },
                ModelSource {
                    path: "b".to_string(),
                    weight: None,
                },
            ],
            output: String::new(),
            default_strategy: "average".to_string(),
            layers: None,
        },
    ];
    for config in &invalid_configs {
        assert!(
            validate_merge_config(config).is_err(),
            "Should reject invalid config"
        );
    }
}
