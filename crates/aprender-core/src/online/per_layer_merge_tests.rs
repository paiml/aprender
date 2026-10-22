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
// Report accounting and parser helper tests
// ============================================================================

#[test]
fn report_counts_matched_and_defaulted_tensors() {
    let mut report = LayerMergeReport::default();
    assert_eq!(report.tensors_processed, 0);
    assert_eq!(report.total_matched(), 0);
    assert_eq!(report.total_defaulted(), 0);

    report.record_tensor(Some("attn"));
    report.record_tensor(Some("attn"));
    report.record_tensor(Some("mlp"));
    report.record_tensor(None);
    report.record_tensor(None);

    assert_eq!(report.tensors_processed, 5);
    assert_eq!(report.rules_matched["attn"], 2);
    assert_eq!(report.rules_matched["mlp"], 1);
    assert_eq!(report.total_matched(), 3);
    assert_eq!(report.total_defaulted(), 2);
}

#[test]
fn report_defaulted_saturates_instead_of_underflowing() {
    let mut report = LayerMergeReport::new();
    report.rules_matched.insert("x".to_string(), 7);
    report.tensors_processed = 3;
    assert_eq!(report.total_defaulted(), 0);
}

#[test]
fn wildcard_pattern_requires_segments_in_order() {
    assert!(pattern_matches(
        "model.layers.0.q_proj.weight",
        "layers.0.*weight"
    ));
    assert!(!pattern_matches(
        "model.layers.0.q_proj.bias",
        "layers.0.*weight"
    ));
    assert!(!pattern_matches("weight.layers.0.q", "layers.0.*weight"));
    assert!(pattern_matches("anything", "*"));
    assert!(pattern_matches("a.b", "a**b"));
}

#[test]
fn escaped_dot_is_a_literal_dot() {
    assert!(pattern_matches("layers.10.attn", "layers\\.10\\."));
    assert!(!pattern_matches("layers_10_attn", "layers\\.10\\."));
}

#[test]
fn parse_kv_handles_spaced_bare_and_missing_keys() {
    assert_eq!(
        parse_kv("  output: /tmp/m.apr  ", "output"),
        Some("/tmp/m.apr")
    );
    assert_eq!(parse_kv("models:", "models"), None);
    assert_eq!(parse_kv("output:/tmp/m.apr", "output"), Some("/tmp/m.apr"));
    assert_eq!(parse_kv("output:", "output"), None);
    assert_eq!(parse_kv("weight: 0.5", "output"), None);
}

#[test]
fn unquote_strips_matching_quotes_only() {
    assert_eq!(unquote("\"abc\""), "abc");
    assert_eq!(unquote("'abc'"), "abc");
    assert_eq!(unquote("  plain  "), "plain");
    assert_eq!(unquote("\"mixed'"), "\"mixed'");
    assert_eq!(unquote("\""), "\"");
    assert_eq!(unquote("\"\""), "");
}

#[test]
fn parse_float_list_skips_unparseable_items() {
    assert_eq!(parse_float_list("[0.7, 0.3]"), vec![0.7, 0.3]);
    assert_eq!(parse_float_list(" [ ] "), Vec::<f64>::new());
    assert_eq!(parse_float_list("[1, x, 2.5]"), vec![1.0, 2.5]);
    assert_eq!(parse_float_list("0.5"), vec![0.5]);
}

#[test]
fn flush_entries_only_fire_in_their_own_section() {
    let mut models = Vec::new();
    let mut path = "a.apr".to_string();
    let mut weight = Some(0.4);
    flush_model_entry(&ParserSection::Root, &mut models, &mut path, &mut weight);
    assert!(models.is_empty());
    assert_eq!(path, "a.apr");
    flush_model_entry(
        &ParserSection::ModelEntry,
        &mut models,
        &mut path,
        &mut weight,
    );
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].weight, Some(0.4));
    assert!(path.is_empty() && weight.is_none());

    let mut rules = Vec::new();
    let (mut pat, mut strat, mut w, mut scale) =
        ("attn".to_string(), String::new(), None, Some(2.0));
    flush_layer_entry(
        &ParserSection::Layers,
        &mut rules,
        &mut pat,
        &mut strat,
        &mut w,
        &mut scale,
    );
    assert!(rules.is_empty());
    flush_layer_entry(
        &ParserSection::LayerEntry,
        &mut rules,
        &mut pat,
        &mut strat,
        &mut w,
        &mut scale,
    );
    assert_eq!(rules.len(), 1);
    assert_eq!(
        rules[0].strategy, "average",
        "empty strategy defaults to average"
    );
    assert_eq!(rules[0].scale, Some(2.0));
    assert!(pat.is_empty() && scale.is_none());
}

fn valid_config() -> MergeYamlConfig {
    parse_merge_yaml(
        "models:\n  - path: a.apr\n  - path: b.apr\noutput: o.apr\ndefault_strategy: average\n",
    )
    .expect("valid")
}

#[test]
fn validate_rejects_each_bad_field_by_name() {
    let mut c = valid_config();
    c.models[0].weight = Some(f64::NAN);
    assert!(validate_merge_config(&c)
        .unwrap_err()
        .to_string()
        .contains("weight"));

    let mut c = valid_config();
    c.models[1].path.clear();
    assert!(validate_merge_config(&c)
        .unwrap_err()
        .to_string()
        .contains("empty path"));

    let mut c = valid_config();
    c.output.clear();
    assert!(validate_merge_config(&c)
        .unwrap_err()
        .to_string()
        .contains("output path"));

    let rule = |p: &str, s: &str, w: Option<Vec<f64>>, sc: Option<f64>| LayerRule {
        layer_pattern: p.to_string(),
        strategy: s.to_string(),
        weights: w,
        scale: sc,
    };
    for (r, needle) in [
        (rule("", "average", None, None), "empty pattern"),
        (rule("a", "nosuch", None, None), "unknown strategy"),
        (
            rule("a", "average", Some(vec![f64::INFINITY]), None),
            "weight[0]",
        ),
        (rule("a", "average", None, Some(f64::NAN)), "scale"),
    ] {
        let mut c = valid_config();
        c.layers = Some(vec![r]);
        let msg = validate_merge_config(&c).unwrap_err().to_string();
        assert!(msg.contains(needle), "{needle}: {msg}");
    }
    assert!(validate_merge_config(&valid_config()).is_ok());
}
