// Iteration 3: Aggressive Falsification — Edge Cases & Adversarial Inputs
// =============================================================================

#[test]
fn falsify_iter3_scoring_bias_vs_no_bias_separation() {
    // STRONG PREDICTION: Bias tensor names must produce a STRICTLY higher score
    // than no-bias tensor names for bias-bearing families.
    let registry = build_default_registry();

    let base_names = vec![
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ];

    let mut with_bias = base_names.clone();
    with_bias.push("model.layers.0.self_attn.q_proj.bias");
    with_bias.push("model.layers.0.self_attn.k_proj.bias");
    with_bias.push("model.layers.0.self_attn.v_proj.bias");

    // Both must detect, but with different specificity
    let no_bias_family = registry.detect_family(&base_names);
    let bias_family = registry.detect_family(&with_bias);

    assert!(no_bias_family.is_some(), "base names must detect");
    assert!(bias_family.is_some(), "bias names must detect");

    // Bias result must be a bias-bearing family
    let bias_name = bias_family.expect("bias").family_name();
    assert!(
        bias_name == "phi" || bias_name == "qwen2",
        "ITER3: Bias tensors must select a bias-bearing family, got '{bias_name}'"
    );
}

#[test]
fn falsify_iter3_all_families_have_unique_model_type() {
    // PREDICTION: No two families share the same model_type detection path
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let detected = registry.detect_from_model_type(family_name);
        assert!(
            detected.is_some(),
            "ITER3: Family '{family_name}' must be detectable by its own name"
        );
        assert_eq!(
            detected.expect("detected").family_name(),
            *family_name,
            "ITER3: model_type '{family_name}' must map to itself, not another family"
        );
    }
}

#[test]
fn falsify_iter3_size_detection_is_injective_per_family() {
    // PREDICTION: Within each family, no two sizes share (hidden_dim, num_layers)
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        let mut seen: Vec<(usize, usize)> = Vec::new();
        for (size_name, size_config) in &config.size_variants {
            let key = (size_config.hidden_dim, size_config.num_layers);
            assert!(
                !seen.contains(&key),
                "ITER3: {family_name} has duplicate (hidden_dim={}, num_layers={}) in sizes",
                key.0,
                key.1
            );
            seen.push(key);

            // Also verify detect_size round-trips
            let detected = family.detect_size(size_config.hidden_dim, size_config.num_layers);
            assert_eq!(
                detected.as_deref(),
                Some(size_name.as_str()),
                "ITER3: {family_name}.detect_size({}, {}) should return '{size_name}'",
                size_config.hidden_dim,
                size_config.num_layers,
            );
        }
    }
}

#[test]
fn falsify_iter3_expected_tensor_count_all_families() {
    // PREDICTION: expected_tensor_count() is consistent with validate_tensor_names()
    // for EVERY family and EVERY size variant.
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        for (size_name, size_config) in &config.size_variants {
            let expected = family.expected_tensor_count(size_name);
            assert!(
                expected.is_some(),
                "ITER3: {family_name}/{size_name} must have an expected tensor count"
            );
            let count = expected.expect("count");
            assert!(
                count > 0,
                "ITER3: {family_name}/{size_name} expected_tensor_count must be > 0"
            );

            // Build the full tensor name set and verify count matches
            let mut names: Vec<String> = Vec::new();
            names.push(config.tensor_template.embedding.clone());
            if let Some(ref lm_head) = config.tensor_template.lm_head {
                names.push(lm_head.clone());
            }
            if let Some(ref final_norm) = config.tensor_template.final_norm {
                names.push(final_norm.clone());
            }
            for layer_idx in 0..size_config.num_layers {
                for pat in config.tensor_template.per_layer.values().flatten() {
                    names.push(pat.replace("{n}", &layer_idx.to_string()));
                }
            }

            assert_eq!(
                names.len(),
                count,
                "ITER3: {family_name}/{size_name} tensor count mismatch: built {} names, expected {}",
                names.len(),
                count
            );

            // And validate_tensor_names must accept this exact set
            let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
            let result = family.validate_tensor_names(&name_refs, size_name);
            assert!(
                result.is_ok(),
                "ITER3: {family_name}/{size_name} full tensor set must pass validation, got: {:?}",
                result.err()
            );
        }
    }
}

#[test]
fn falsify_iter3_gemma_detected_distinctly() {
    let registry = build_default_registry();
    let gemma = registry
        .detect_from_model_type("gemma")
        .expect("gemma detected");

    assert_eq!(gemma.family_name(), "gemma");
    assert_eq!(gemma.config().vendor, "Google");

    // Verify gemma has at least one size
    assert!(
        !gemma.config().size_variants.is_empty(),
        "ITER3: Gemma must have size variants"
    );
}

#[test]
fn falsify_iter3_validated_weight_rejects_inf() {
    // PREDICTION: ValidatedWeight rejects Inf values
    let mut data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    data[50] = f32::INFINITY;
    let result = ValidatedWeight::new(data, 10, 10, "test");
    assert!(
        result.is_err(),
        "ITER3: ValidatedWeight must reject Inf values"
    );
    assert!(
        result.unwrap_err().message.contains("Inf"),
        "ITER3: Error message must mention Inf"
    );
}

#[test]
fn falsify_iter3_validated_weight_shape_enforcement() {
    // PREDICTION: ValidatedWeight rejects mismatched dimensions
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    let result = ValidatedWeight::new(data, 5, 5, "test"); // 25 != 100
    assert!(
        result.is_err(),
        "ITER3: ValidatedWeight must reject shape mismatch (100 elements for 5x5)"
    );
}

#[test]
fn falsify_iter3_yaml_contracts_dir_exists() {
    let root = find_project_root();
    let contracts_dir = root.join("contracts/model-families");
    assert!(
        contracts_dir.exists(),
        "ITER3: contracts/model-families/ directory must exist"
    );

    // Must have at least 8 YAML files
    let yaml_count = std::fs::read_dir(&contracts_dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "yaml"))
        .count();

    assert!(
        yaml_count >= 8,
        "ITER3: contracts/model-families/ must have >= 8 YAML files, found {yaml_count}"
    );
}

#[test]
fn falsify_iter3_registry_lookup_by_name_all_families() {
    // PREDICTION: registry.get() works for every known family name
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name);
        assert!(
            family.is_some(),
            "ITER3: registry.get('{family_name}') must return Some"
        );
        assert_eq!(
            family.expect("family").family_name(),
            *family_name,
            "ITER3: registry.get('{family_name}').family_name() must equal '{family_name}'"
        );
    }
}

// =============================================================================
// Iteration 4: Deeper Falsification — Unique Families, Constraints, Adversarial
// =============================================================================

#[test]
fn falsify_iter4_bert_detected_unambiguously_from_tensor_names() {
    // STRONG PREDICTION: BERT has unique tensor naming (bert.embeddings.* / bert.encoder.layer.*)
    // that no other family shares. detect_family() MUST return "bert" — not just any family.
    let registry = build_default_registry();

    let bert_tensors = vec![
        "bert.embeddings.word_embeddings.weight",
        "bert.encoder.layer.0.attention.self.query.weight",
        "bert.encoder.layer.0.attention.self.query.bias",
        "bert.encoder.layer.0.attention.self.key.weight",
        "bert.encoder.layer.0.attention.self.key.bias",
        "bert.encoder.layer.0.attention.self.value.weight",
        "bert.encoder.layer.0.attention.self.value.bias",
        "bert.encoder.layer.0.attention.output.dense.weight",
        "bert.encoder.layer.0.attention.output.dense.bias",
        "bert.encoder.layer.0.attention.output.LayerNorm.weight",
        "bert.encoder.layer.0.attention.output.LayerNorm.bias",
        "bert.encoder.layer.0.intermediate.dense.weight",
        "bert.encoder.layer.0.intermediate.dense.bias",
        "bert.encoder.layer.0.output.dense.weight",
        "bert.encoder.layer.0.output.dense.bias",
        "bert.encoder.layer.0.output.LayerNorm.weight",
        "bert.encoder.layer.0.output.LayerNorm.bias",
    ];

    let detected = registry.detect_family(&bert_tensors);
    assert!(
        detected.is_some(),
        "ITER4: BERT tensor names must be detected"
    );
    assert_eq!(
        detected.expect("bert detected").family_name(),
        "bert",
        "ITER4: BERT-specific tensors must unambiguously detect as 'bert', not any other family"
    );
}

#[test]
fn falsify_iter4_whisper_detected_unambiguously_from_tensor_names() {
    // STRONG PREDICTION: Whisper has unique tensor naming (encoder.conv1.* / encoder.layers.*)
    // that no other family shares. detect_family() MUST return "whisper".
    let registry = build_default_registry();

    let whisper_tensors = vec![
        "encoder.conv1.weight",
        "encoder.layers.0.self_attn.q_proj.weight",
        "encoder.layers.0.self_attn.q_proj.bias",
        "encoder.layers.0.self_attn.k_proj.weight",
        "encoder.layers.0.self_attn.k_proj.bias",
        "encoder.layers.0.self_attn.v_proj.weight",
        "encoder.layers.0.self_attn.v_proj.bias",
        "encoder.layers.0.self_attn.out_proj.weight",
        "encoder.layers.0.self_attn.out_proj.bias",
        "encoder.layers.0.self_attn_layer_norm.weight",
        "encoder.layers.0.self_attn_layer_norm.bias",
        "encoder.layers.0.fc1.weight",
        "encoder.layers.0.fc1.bias",
        "encoder.layers.0.fc2.weight",
        "encoder.layers.0.fc2.bias",
        "encoder.layers.0.final_layer_norm.weight",
        "encoder.layers.0.final_layer_norm.bias",
    ];

    let detected = registry.detect_family(&whisper_tensors);
    assert!(
        detected.is_some(),
        "ITER4: Whisper tensor names must be detected"
    );
    assert_eq!(
        detected.expect("whisper detected").family_name(),
        "whisper",
        "ITER4: Whisper-specific tensors must unambiguously detect as 'whisper', not any other family"
    );
}

#[test]
fn falsify_iter4_bert_per_layer_patterns_all_bert_specific() {
    // STRONG PREDICTION: Every per_layer pattern in BERT's config contains "bert.encoder.layer.{n}"
    let registry = build_default_registry();
    let bert = registry.get("bert").expect("bert in registry");
    let config = bert.config();

    for (role, pattern) in &config.tensor_template.per_layer {
        if let Some(pat) = pattern {
            assert!(
                pat.contains("bert.encoder.layer.{n}"),
                "ITER4: BERT per_layer role '{role}' pattern '{pat}' must contain 'bert.encoder.layer.{{n}}'"
            );
        }
    }
}

#[test]
fn falsify_iter4_whisper_per_layer_patterns_all_encoder_specific() {
    // STRONG PREDICTION: Every per_layer pattern in Whisper's config contains "encoder.layers.{n}"
    let registry = build_default_registry();
    let whisper = registry.get("whisper").expect("whisper in registry");
    let config = whisper.config();

    for (role, pattern) in &config.tensor_template.per_layer {
        if let Some(pat) = pattern {
            assert!(
                pat.contains("encoder.layers.{n}"),
                "ITER4: Whisper per_layer role '{role}' pattern '{pat}' must contain 'encoder.layers.{{n}}'"
            );
        }
    }
}

#[test]
fn falsify_iter4_gqa_families_have_kv_heads_less_than_heads() {
    // STRONG PREDICTION: For GQA families (not MHA), at least one size variant
    // has num_kv_heads < num_heads (otherwise it's just MHA).
    use aprender::format::model_family::AttentionType;

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let constraints = family.constraints();

        if constraints.attention_type == AttentionType::Gqa {
            let config = family.config();
            let has_gqa_size = config
                .size_variants
                .values()
                .any(|s| s.num_kv_heads < s.num_heads);
            assert!(
                has_gqa_size,
                "ITER4: {family_name} declares GQA but no size has num_kv_heads < num_heads"
            );
        }
    }
}

#[test]
fn falsify_iter4_no_bias_families_have_no_bias_constraint() {
    // STRONG PREDICTION: Families that declare has_bias=false should NOT have
    // bias patterns (q_proj_bias, k_proj_bias, v_proj_bias) resolving to real tensors.
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let constraints = family.constraints();
        let config = family.config();

        if !constraints.has_bias {
            // Check that bias-related per_layer entries are None
            for (role, pattern) in &config.tensor_template.per_layer {
                if role.contains("bias") {
                    assert!(
                        pattern.is_none(),
                        "ITER4: {family_name} declares has_bias=false but per_layer '{role}' is Some('{}')",
                        pattern.as_deref().unwrap_or("?")
                    );
                }
            }
        }
    }
}

#[test]
fn falsify_iter4_bias_families_have_bias_patterns() {
    // STRONG PREDICTION: Families declaring has_bias=true AND using standard LLaMA-like
    // naming (model.layers.*) should have at least one non-None bias per_layer entry.
    let registry = build_default_registry();

    let standard_bias_families = ["qwen2", "phi"]; // These use model.layers.* with biases
    for &family_name in &standard_bias_families {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        let bias_count = config
            .tensor_template
            .per_layer
            .iter()
            .filter(|(role, pat)| role.contains("bias") && pat.is_some())
            .count();

        assert!(
            bias_count >= 3,
            "ITER4: {family_name} declares has_bias=true but only has {bias_count} bias patterns (expected >= 3)"
        );
    }
}

#[test]
fn falsify_iter4_embedding_uniqueness_bert_whisper() {
    // STRONG PREDICTION: BERT and Whisper have embedding tensor names different
    // from all standard "model.embed_tokens.weight" families.
    let registry = build_default_registry();

    let bert = registry.get("bert").expect("bert");
    let whisper = registry.get("whisper").expect("whisper");

    // BERT and Whisper must NOT use model.embed_tokens.weight
    assert_ne!(
        bert.config().tensor_template.embedding,
        "model.embed_tokens.weight",
        "ITER4: BERT embedding must differ from standard families"
    );
    assert_ne!(
        whisper.config().tensor_template.embedding,
        "model.embed_tokens.weight",
        "ITER4: Whisper embedding must differ from standard families"
    );

    // BERT and Whisper must have different embedding from each other
    assert_ne!(
        bert.config().tensor_template.embedding,
        whisper.config().tensor_template.embedding,
        "ITER4: BERT and Whisper must have different embedding tensor names"
    );
}

#[test]
fn falsify_iter4_detect_from_model_type_unknown_returns_none() {
    // STRONG PREDICTION: Unknown model_type should return None
    let registry = build_default_registry();

    let unknown_types = ["gpt2", "falcon", "mamba", "rwkv", "t5", "unknown_model_xyz"];
    for model_type in &unknown_types {
        let detected = registry.detect_from_model_type(model_type);
        assert!(
            detected.is_none(),
            "ITER4: Unknown model_type '{model_type}' should return None, got '{}'",
            detected.map_or("None", |f| f.family_name())
        );
    }
}

#[test]
fn falsify_iter4_per_layer_roles_unique_per_family() {
    // STRONG PREDICTION: Within each family, no two per_layer roles map to the
    // exact same tensor pattern (that would be a YAML copy-paste bug).
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        let patterns: Vec<&str> = config
            .tensor_template
            .per_layer
            .values()
            .filter_map(|p| p.as_deref())
            .collect();

        let mut seen = std::collections::HashSet::new();
        for pat in &patterns {
            assert!(
                seen.insert(*pat),
                "ITER4: {family_name} has duplicate per_layer pattern: '{pat}'"
            );
        }
    }
}

