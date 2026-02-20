// Iteration 5: Architectural Consistency & Mathematical Invariants
// =============================================================================

#[test]
fn falsify_iter5_intermediate_dim_greater_than_hidden_dim() {
    // STRONG PREDICTION: For ALL families and sizes, intermediate_dim > hidden_dim
    // (the FFN expands the representation before projecting back down).
    // If intermediate_dim <= hidden_dim, the FFN is a bottleneck, which is invalid.
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        for (size_name, size_config) in &config.size_variants {
            if size_config.intermediate_dim > 0 {
                assert!(
                    size_config.intermediate_dim > size_config.hidden_dim,
                    "ITER5: {family_name}/{size_name} intermediate_dim ({}) must be > hidden_dim ({})",
                    size_config.intermediate_dim,
                    size_config.hidden_dim
                );
            }
        }
    }
}

#[test]
fn falsify_iter5_rope_families_have_nonzero_rope_theta() {
    // STRONG PREDICTION: Families using RoPE positional encoding must have
    // rope_theta > 0 for all size variants.
    use aprender::format::model_family::PositionalEncoding;

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let constraints = family.constraints();

        if constraints.positional_encoding == PositionalEncoding::Rope {
            let config = family.config();
            for (size_name, size_config) in &config.size_variants {
                assert!(
                    size_config.rope_theta > 0.0,
                    "ITER5: {family_name}/{size_name} uses RoPE but rope_theta is {} (must be > 0)",
                    size_config.rope_theta
                );
            }
        }
    }
}

#[test]
fn falsify_iter5_non_rope_families_have_zero_or_default_rope_theta() {
    // STRONG PREDICTION: Families NOT using RoPE should have rope_theta == 0.0
    // (default value, not meaningful). If they have a nonzero value, either
    // the positional encoding label is wrong or the theta is misleading.
    use aprender::format::model_family::PositionalEncoding;

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let constraints = family.constraints();

        if constraints.positional_encoding != PositionalEncoding::Rope {
            let config = family.config();
            for (size_name, size_config) in &config.size_variants {
                // rope_theta should be 0.0 (default) or at least the YAML shouldn't set it
                assert!(
                    size_config.rope_theta == 0.0 || size_config.rope_theta == 1e-6,
                    "ITER5: {family_name}/{size_name} does NOT use RoPE but has rope_theta={} (should be 0)",
                    size_config.rope_theta
                );
            }
        }
    }
}

#[test]
fn falsify_iter5_mha_families_have_kv_heads_equal_heads() {
    // STRONG PREDICTION: For MHA families, num_kv_heads == num_heads for ALL sizes.
    // (MHA means every head has its own K/V projection — no sharing.)
    use aprender::format::model_family::AttentionType;

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let constraints = family.constraints();

        if constraints.attention_type == AttentionType::Mha {
            let config = family.config();
            for (size_name, size_config) in &config.size_variants {
                assert_eq!(
                    size_config.num_kv_heads, size_config.num_heads,
                    "ITER5: {family_name}/{size_name} declares MHA but num_kv_heads ({}) != num_heads ({})",
                    size_config.num_kv_heads, size_config.num_heads
                );
            }
        }
    }
}

#[test]
fn falsify_iter5_vocab_size_positive_for_all() {
    // STRONG PREDICTION: Every family/size must have vocab_size > 0.
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        for (size_name, size_config) in &config.size_variants {
            assert!(
                size_config.vocab_size > 0,
                "ITER5: {family_name}/{size_name} vocab_size must be > 0, got {}",
                size_config.vocab_size
            );
        }
    }
}

#[test]
fn falsify_iter5_norm_eps_in_valid_range() {
    // STRONG PREDICTION: norm_eps must be in (0, 0.01) for all families/sizes.
    // Too small (0) causes div-by-zero, too large (>0.01) corrupts normalization.
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        for (size_name, size_config) in &config.size_variants {
            assert!(
                size_config.norm_eps > 0.0 && size_config.norm_eps < 0.01,
                "ITER5: {family_name}/{size_name} norm_eps ({}) must be in (0, 0.01)",
                size_config.norm_eps
            );
        }
    }
}

#[test]
fn falsify_iter5_registry_returns_consistent_data() {
    // STRONG PREDICTION: Multiple calls to build_default_registry() return
    // identical data (the generated code is deterministic).
    let registry1 = build_default_registry();
    let registry2 = build_default_registry();

    assert_eq!(
        registry1.len(),
        registry2.len(),
        "ITER5: Two registry builds must have same length"
    );

    for family_name in KNOWN_FAMILIES {
        let f1 = registry1.get(family_name).expect("f1");
        let f2 = registry2.get(family_name).expect("f2");

        assert_eq!(
            f1.family_name(),
            f2.family_name(),
            "ITER5: {family_name} name mismatch between registries"
        );
        assert_eq!(
            f1.config().vendor,
            f2.config().vendor,
            "ITER5: {family_name} vendor mismatch between registries"
        );
        assert_eq!(
            f1.config().size_variants.len(),
            f2.config().size_variants.len(),
            "ITER5: {family_name} size variant count mismatch"
        );
    }
}

#[test]
fn falsify_iter5_partial_tensor_set_rejected() {
    // STRONG PREDICTION: A tensor set with only embedding + 1 layer (but contract
    // expects 24 layers) should fail validation because layer 1..23 tensors are missing.
    let registry = build_default_registry();
    let qwen2 = registry.get("qwen2").expect("qwen2");
    let config = qwen2.config();

    let mut names: Vec<String> = Vec::new();
    names.push(config.tensor_template.embedding.clone());
    if let Some(ref lm_head) = config.tensor_template.lm_head {
        names.push(lm_head.clone());
    }
    if let Some(ref final_norm) = config.tensor_template.final_norm {
        names.push(final_norm.clone());
    }
    // Only layer 0 — layers 1..23 missing
    for pat in config.tensor_template.per_layer.values().flatten() {
        names.push(pat.replace("{n}", "0"));
    }

    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let result = qwen2.validate_tensor_names(&name_refs, "0.5b");
    assert!(
        result.is_err(),
        "ITER5: Partial tensor set (only layer 0) must be rejected for 0.5b (24 layers expected)"
    );
}

#[test]
fn falsify_iter5_all_families_have_at_least_one_quantization() {
    // STRONG PREDICTION: Every family supports at least one quantization format.
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        assert!(
            !config.quantizations.is_empty(),
            "ITER5: {family_name} must support at least one quantization format"
        );
    }
}

#[test]
fn falsify_iter5_all_standard_families_have_lm_head() {
    // STRONG PREDICTION: All causal LM families (not BERT, not Whisper) must
    // have an lm_head tensor for next-token prediction.
    let registry = build_default_registry();

    let causal_lm_families = ["qwen2", "llama", "mistral", "phi", "deepseek", "gemma"];
    for &family_name in &causal_lm_families {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        assert!(
            config.tensor_template.lm_head.is_some(),
            "ITER5: Causal LM family {family_name} must have lm_head tensor"
        );
    }
}

#[test]
fn falsify_iter5_all_standard_families_have_final_norm() {
    // STRONG PREDICTION: All causal LM families must have a final normalization
    // layer (model.norm.weight or equivalent).
    let registry = build_default_registry();

    let causal_lm_families = ["qwen2", "llama", "mistral", "phi", "deepseek", "gemma"];
    for &family_name in &causal_lm_families {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        assert!(
            config.tensor_template.final_norm.is_some(),
            "ITER5: Causal LM family {family_name} must have final_norm tensor"
        );
    }
}

#[test]
fn falsify_iter5_gqa_kv_heads_divides_heads() {
    // STRONG PREDICTION: For GQA families, num_heads must be divisible by
    // num_kv_heads (each KV group serves the same number of query heads).
    use aprender::format::model_family::AttentionType;

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let constraints = family.constraints();

        if constraints.attention_type == AttentionType::Gqa {
            let config = family.config();
            for (size_name, size_config) in &config.size_variants {
                if size_config.num_kv_heads > 0 {
                    assert_eq!(
                        size_config.num_heads % size_config.num_kv_heads,
                        0,
                        "ITER5: {family_name}/{size_name} num_heads ({}) must be divisible by num_kv_heads ({})",
                        size_config.num_heads,
                        size_config.num_kv_heads
                    );
                }
            }
        }
    }
}

// =============================================================================
