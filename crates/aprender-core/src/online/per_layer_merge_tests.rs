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

// ============================================================================
// validate_merge_config: exhaustive branch coverage (47 uncovered lines)
// ============================================================================

/// Validate: unknown default strategy rejected
#[test]
fn test_validate_merge_config_unknown_default_strategy() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: "a.st".to_string(),
                weight: None,
            },
            ModelSource {
                path: "b.st".to_string(),
                weight: None,
            },
        ],
        output: "out.st".to_string(),
        default_strategy: "quantum_merge".to_string(),
        layers: None,
    };
    let err = validate_merge_config(&config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("unknown default strategy"));
}

/// Validate: empty model path rejected
#[test]
fn test_validate_merge_config_empty_model_path() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: String::new(),
                weight: None,
            },
            ModelSource {
                path: "b.st".to_string(),
                weight: None,
            },
        ],
        output: "out.st".to_string(),
        default_strategy: "average".to_string(),
        layers: None,
    };
    let err = validate_merge_config(&config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("empty path"));
}

/// Validate: negative model weight rejected
#[test]
fn test_validate_merge_config_negative_weight() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: "a.st".to_string(),
                weight: Some(-0.5),
            },
            ModelSource {
                path: "b.st".to_string(),
                weight: Some(0.5),
            },
        ],
        output: "out.st".to_string(),
        default_strategy: "average".to_string(),
        layers: None,
    };
    let err = validate_merge_config(&config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("non-negative"));
}

/// Validate: NaN model weight rejected
#[test]
fn test_validate_merge_config_nan_weight() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: "a.st".to_string(),
                weight: Some(f64::NAN),
            },
            ModelSource {
                path: "b.st".to_string(),
                weight: Some(0.5),
            },
        ],
        output: "out.st".to_string(),
        default_strategy: "average".to_string(),
        layers: None,
    };
    assert!(validate_merge_config(&config).is_err());
}

/// Validate: Inf model weight rejected
#[test]
fn test_validate_merge_config_inf_weight() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: "a.st".to_string(),
                weight: Some(f64::INFINITY),
            },
            ModelSource {
                path: "b.st".to_string(),
                weight: Some(0.5),
            },
        ],
        output: "out.st".to_string(),
        default_strategy: "average".to_string(),
        layers: None,
    };
    assert!(validate_merge_config(&config).is_err());
}

/// Validate: layer rule with empty pattern rejected
#[test]
fn test_validate_merge_config_empty_layer_pattern() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: "a.st".to_string(),
                weight: None,
            },
            ModelSource {
                path: "b.st".to_string(),
                weight: None,
            },
        ],
        output: "out.st".to_string(),
        default_strategy: "average".to_string(),
        layers: Some(vec![LayerRule {
            layer_pattern: String::new(),
            strategy: "slerp".to_string(),
            weights: None,
            scale: None,
        }]),
    };
    let err = validate_merge_config(&config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("empty pattern"));
}

/// Validate: layer rule with unknown strategy rejected
#[test]
fn test_validate_merge_config_invalid_layer_strategy() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: "a.st".to_string(),
                weight: None,
            },
            ModelSource {
                path: "b.st".to_string(),
                weight: None,
            },
        ],
        output: "out.st".to_string(),
        default_strategy: "average".to_string(),
        layers: Some(vec![LayerRule {
            layer_pattern: "attn".to_string(),
            strategy: "nonexistent_strategy".to_string(),
            weights: None,
            scale: None,
        }]),
    };
    let err = validate_merge_config(&config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("unknown strategy"));
}

/// Validate: layer rule with NaN weight rejected
#[test]
fn test_validate_merge_config_layer_nan_weight() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: "a.st".to_string(),
                weight: None,
            },
            ModelSource {
                path: "b.st".to_string(),
                weight: None,
            },
        ],
        output: "out.st".to_string(),
        default_strategy: "average".to_string(),
        layers: Some(vec![LayerRule {
            layer_pattern: "attn".to_string(),
            strategy: "slerp".to_string(),
            weights: Some(vec![f64::NAN]),
            scale: None,
        }]),
    };
    let err = validate_merge_config(&config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("not finite"));
}

/// Validate: layer rule with Inf weight rejected
#[test]
fn test_validate_merge_config_layer_inf_weight() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: "a.st".to_string(),
                weight: None,
            },
            ModelSource {
                path: "b.st".to_string(),
                weight: None,
            },
        ],
        output: "out.st".to_string(),
        default_strategy: "average".to_string(),
        layers: Some(vec![LayerRule {
            layer_pattern: "attn".to_string(),
            strategy: "slerp".to_string(),
            weights: Some(vec![f64::INFINITY]),
            scale: None,
        }]),
    };
    assert!(validate_merge_config(&config).is_err());
}

/// Validate: layer rule with NaN scale rejected
#[test]
fn test_validate_merge_config_layer_nan_scale() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: "a.st".to_string(),
                weight: None,
            },
            ModelSource {
                path: "b.st".to_string(),
                weight: None,
            },
        ],
        output: "out.st".to_string(),
        default_strategy: "average".to_string(),
        layers: Some(vec![LayerRule {
            layer_pattern: "attn".to_string(),
            strategy: "slerp".to_string(),
            weights: None,
            scale: Some(f64::NAN),
        }]),
    };
    let err = validate_merge_config(&config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(msg.contains("scale is not finite"));
}

/// Validate: layer rule with Inf scale rejected
#[test]
fn test_validate_merge_config_layer_inf_scale() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: "a.st".to_string(),
                weight: None,
            },
            ModelSource {
                path: "b.st".to_string(),
                weight: None,
            },
        ],
        output: "out.st".to_string(),
        default_strategy: "average".to_string(),
        layers: Some(vec![LayerRule {
            layer_pattern: "mlp".to_string(),
            strategy: "average".to_string(),
            weights: None,
            scale: Some(f64::NEG_INFINITY),
        }]),
    };
    assert!(validate_merge_config(&config).is_err());
}

/// Validate: valid config with all layer rule fields passes
#[test]
fn test_validate_merge_config_valid_with_layers() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: "a.st".to_string(),
                weight: Some(0.5),
            },
            ModelSource {
                path: "b.st".to_string(),
                weight: Some(0.5),
            },
        ],
        output: "out.st".to_string(),
        default_strategy: "average".to_string(),
        layers: Some(vec![
            LayerRule {
                layer_pattern: "attn".to_string(),
                strategy: "slerp".to_string(),
                weights: Some(vec![0.7, 0.3]),
                scale: Some(1.0),
            },
            LayerRule {
                layer_pattern: "mlp".to_string(),
                strategy: "ties".to_string(),
                weights: None,
                scale: None,
            },
        ]),
    };
    assert!(validate_merge_config(&config).is_ok());
}

/// Validate: zero weight is allowed (non-negative and finite)
#[test]
fn test_validate_merge_config_zero_weight_ok() {
    let config = MergeYamlConfig {
        models: vec![
            ModelSource {
                path: "a.st".to_string(),
                weight: Some(0.0),
            },
            ModelSource {
                path: "b.st".to_string(),
                weight: Some(1.0),
            },
        ],
        output: "out.st".to_string(),
        default_strategy: "average".to_string(),
        layers: None,
    };
    assert!(validate_merge_config(&config).is_ok());
}

/// Validate: all valid strategies accepted
#[test]
fn test_validate_merge_config_all_valid_strategies() {
    for strategy in &["average", "weighted_average", "slerp", "ties", "dare", "passthrough"] {
        let config = MergeYamlConfig {
            models: vec![
                ModelSource {
                    path: "a.st".to_string(),
                    weight: None,
                },
                ModelSource {
                    path: "b.st".to_string(),
                    weight: None,
                },
            ],
            output: "out.st".to_string(),
            default_strategy: strategy.to_string(),
            layers: None,
        };
        assert!(
            validate_merge_config(&config).is_ok(),
            "Strategy '{}' should be valid",
            strategy
        );
    }
}

// ============================================================================
// parse_merge_yaml additional coverage
// ============================================================================

/// Parse YAML with layer rules
#[test]
fn test_parse_merge_yaml_with_layers() {
    let yaml = r#"
models:
  - path: a.safetensors
    weight: 0.6
  - path: b.safetensors
    weight: 0.4
output: merged.safetensors
default_strategy: weighted_average
layers:
  - layer_pattern: attn
    strategy: slerp
    weights: [0.7, 0.3]
    scale: 1.5
  - layer_pattern: mlp
    strategy: average
"#;
    let config = parse_merge_yaml(yaml).expect("should parse");
    assert_eq!(config.models.len(), 2);
    assert_eq!(config.default_strategy, "weighted_average");
    let layers = config.layers.expect("should have layers");
    assert_eq!(layers.len(), 2);
    assert_eq!(layers[0].layer_pattern, "attn");
    assert_eq!(layers[0].strategy, "slerp");
    assert!(layers[0].weights.is_some());
    assert_eq!(layers[0].scale, Some(1.5));
    assert_eq!(layers[1].layer_pattern, "mlp");
}

/// Parse YAML with quoted paths
#[test]
fn test_parse_merge_yaml_quoted_paths() {
    let yaml = r#"
models:
  - path: "model with spaces.safetensors"
  - path: 'another model.safetensors'
output: "output file.safetensors"
default_strategy: average
"#;
    let config = parse_merge_yaml(yaml).expect("should parse");
    assert_eq!(config.models[0].path, "model with spaces.safetensors");
    assert_eq!(config.models[1].path, "another model.safetensors");
    assert_eq!(config.output, "output file.safetensors");
}

// ============================================================================
// pattern_matches additional coverage
// ============================================================================

/// Wildcard pattern matching
#[test]
fn test_pattern_matches_wildcard() {
    assert!(pattern_matches(
        "model.layers.0.self_attn.q_proj.weight",
        "layers*weight"
    ));
    assert!(!pattern_matches(
        "model.layers.0.self_attn.q_proj.bias",
        "layers*weight"
    ));
}

/// Escaped dot pattern matching
#[test]
fn test_pattern_matches_escaped_dot() {
    assert!(pattern_matches("layers.0.weight", "layers\\.0\\."));
    assert!(!pattern_matches("layers10.weight", "layers\\.0\\."));
}

/// LayerMergeReport methods
#[test]
fn test_layer_merge_report_tracking() {
    let mut report = LayerMergeReport::new();
    report.record_tensor(Some("attn"));
    report.record_tensor(Some("attn"));
    report.record_tensor(Some("mlp"));
    report.record_tensor(None);

    assert_eq!(report.tensors_processed, 4);
    assert_eq!(report.total_matched(), 3);
    assert_eq!(report.total_defaulted(), 1);
    assert_eq!(*report.rules_matched.get("attn").unwrap_or(&0), 2);
}

/// LayerMergeReport default
#[test]
fn test_layer_merge_report_default() {
    let report = LayerMergeReport::default();
    assert_eq!(report.tensors_processed, 0);
    assert_eq!(report.total_matched(), 0);
    assert_eq!(report.total_defaulted(), 0);
}
