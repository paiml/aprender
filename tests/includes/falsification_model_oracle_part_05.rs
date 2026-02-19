#[test]
fn falsify_iter4_adversarial_trailing_whitespace_not_detected() {
    // STRONG PREDICTION: Tensor names with trailing whitespace should NOT match.
    let registry = build_default_registry();

    let adversarial_names = vec![
        "model.embed_tokens.weight ",               // trailing space
        " model.embed_tokens.weight",               // leading space
        "model.layers.0.self_attn.q_proj.weight\t", // trailing tab
    ];

    let detected = registry.detect_family(&adversarial_names);
    assert!(
        detected.is_none(),
        "ITER4: Adversarial tensor names with whitespace should not match any family"
    );
}

#[test]
fn falsify_iter4_adversarial_case_sensitivity() {
    // STRONG PREDICTION: Tensor names are case-sensitive — "Model.Embed_Tokens.Weight"
    // should NOT match "model.embed_tokens.weight".
    let registry = build_default_registry();

    let wrong_case = vec![
        "Model.Embed_Tokens.Weight",
        "model.layers.0.Self_Attn.Q_Proj.Weight",
    ];

    let detected = registry.detect_family(&wrong_case);
    assert!(
        detected.is_none(),
        "ITER4: Wrong-case tensor names must not match any family"
    );
}

#[test]
fn falsify_iter4_all_families_constraints_consistent() {
    // STRONG PREDICTION: Constraint cross-checks:
    // - SwiGLU MLP implies SiLU activation (SiLU-gated linear unit)
    // - GELU MLP implies GELU activation (standard GELU feedforward)
    // - Gated MLP implies GELU activation (GeGLU = GELU-gated linear unit, e.g. Gemma)
    use aprender::format::model_family::{Activation, MlpType};

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let constraints = family.constraints();

        // SwiGLU always uses SiLU activation
        if constraints.mlp_type == MlpType::SwiGlu {
            assert_eq!(
                constraints.activation,
                Activation::Silu,
                "ITER4: {family_name} uses SwiGLU MLP but activation is {:?}, expected SiLU",
                constraints.activation
            );
        }

        // GELU MLP uses GELU activation
        if constraints.mlp_type == MlpType::GeluMlp {
            assert_eq!(
                constraints.activation,
                Activation::Gelu,
                "ITER4: {family_name} uses GELU MLP but activation is {:?}, expected GELU",
                constraints.activation
            );
        }

        // Gated MLP (GeGLU) uses GELU activation
        if constraints.mlp_type == MlpType::GatedMlp {
            assert_eq!(
                constraints.activation,
                Activation::Gelu,
                "ITER4: {family_name} uses Gated MLP (GeGLU) but activation is {:?}, expected GELU",
                constraints.activation
            );
        }
    }
}

#[test]
fn falsify_iter4_bert_validate_tensor_names_complete() {
    // STRONG PREDICTION: Building BERT tensor names from its config and validating
    // against its own contract passes for BOTH sizes.
    let registry = build_default_registry();
    let bert = registry.get("bert").expect("bert");
    let config = bert.config();

    for (size_name, size_config) in &config.size_variants {
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

        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let result = bert.validate_tensor_names(&name_refs, size_name);
        assert!(
            result.is_ok(),
            "ITER4: BERT/{size_name} full tensor set must pass self-validation, got: {:?}",
            result.err()
        );
    }
}

#[test]
fn falsify_iter4_whisper_validate_tensor_names_complete() {
    // STRONG PREDICTION: Building Whisper tensor names from its config and validating
    // against its own contract passes for ALL sizes.
    let registry = build_default_registry();
    let whisper = registry.get("whisper").expect("whisper");
    let config = whisper.config();

    for (size_name, size_config) in &config.size_variants {
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

        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let result = whisper.validate_tensor_names(&name_refs, size_name);
        assert!(
            result.is_ok(),
            "ITER4: Whisper/{size_name} full tensor set must pass self-validation, got: {:?}",
            result.err()
        );
    }
}

#[test]
fn falsify_iter4_cross_family_validate_tensor_names_rejects() {
    // STRONG PREDICTION: Every family rejects every other family's tensor names.
    let registry = build_default_registry();

    // Build tensor name sets for selected families
    let test_families = ["bert", "whisper", "qwen2"];

    for &source_family_name in &test_families {
        let source = registry.get(source_family_name).expect("source");
        let config = source.config();

        // Get the first size variant name
        let (first_size_name, first_size) = config
            .size_variants
            .iter()
            .next()
            .expect("at least one size");

        // Build tensor names for source family
        let mut names: Vec<String> = Vec::new();
        names.push(config.tensor_template.embedding.clone());
        if let Some(ref lm_head) = config.tensor_template.lm_head {
            names.push(lm_head.clone());
        }
        if let Some(ref final_norm) = config.tensor_template.final_norm {
            names.push(final_norm.clone());
        }
        for layer_idx in 0..first_size.num_layers {
            for pat in config.tensor_template.per_layer.values().flatten() {
                names.push(pat.replace("{n}", &layer_idx.to_string()));
            }
        }
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();

        // Try validating against other families — should fail
        for &target_family_name in &test_families {
            if target_family_name == source_family_name {
                continue;
            }
            let target = registry.get(target_family_name).expect("target");
            let target_config = target.config();
            let (target_size_name, _) = target_config
                .size_variants
                .iter()
                .next()
                .expect("at least one size");

            let result = target.validate_tensor_names(&name_refs, target_size_name);
            assert!(
                result.is_err(),
                "ITER4: {source_family_name}'s tensors (size={first_size_name}) must be REJECTED by {target_family_name}'s contract (size={target_size_name})"
            );
        }
    }
}

#[test]
fn falsify_iter4_head_dim_consistency() {
    // PREDICTION: For most families/sizes, head_dim == hidden_dim / num_heads.
    // Exception: Some models (e.g., Gemma 7B) intentionally override head_dim
    // to a larger value for improved attention quality. For such models,
    // head_dim * num_heads > hidden_dim, meaning attention operates in a
    // higher-dimensional space than the residual stream.
    //
    // The falsifiable prediction is: head_dim is EITHER:
    //   (a) == hidden_dim / num_heads (standard), OR
    //   (b) > hidden_dim / num_heads (intentional override — valid)
    //
    // head_dim < hidden_dim / num_heads would be suspicious and likely a bug.
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        for (size_name, size_config) in &config.size_variants {
            if size_config.num_heads > 0 {
                let standard_head_dim = size_config.hidden_dim / size_config.num_heads;
                assert!(
                    size_config.head_dim >= standard_head_dim,
                    "ITER4: {family_name}/{size_name} head_dim ({}) < hidden_dim/num_heads ({}/{}={}). \
                     head_dim should be >= standard to avoid information loss.",
                    size_config.head_dim, size_config.hidden_dim, size_config.num_heads, standard_head_dim
                );
            }
        }
    }
}

// =============================================================================
