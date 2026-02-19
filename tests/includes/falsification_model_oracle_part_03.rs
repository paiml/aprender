// §7.4 — FALSIFY-BGN-001: Generated Code Matches YAML
// =============================================================================
//
// Prediction: Build-time generated constants match the values in YAML files.

#[test]
fn falsify_bgn_001_generated_constants_match_yaml() {
    // Cross-reference generated constants against known YAML values
    // If build.rs misparses YAML, these will fail

    // Qwen2 YAML: vendor: Alibaba, 0.5b hidden_dim: 896, num_layers: 24
    assert_eq!(
        QWEN2_VENDOR, "Alibaba",
        "FALSIFY-BGN-001: QWEN2_VENDOR must match qwen2.yaml"
    );
    assert_eq!(
        QWEN2_0_5B_HIDDEN_DIM, 896,
        "FALSIFY-BGN-001: QWEN2_0_5B_HIDDEN_DIM must match qwen2.yaml"
    );
    assert_eq!(
        QWEN2_0_5B_NUM_LAYERS, 24,
        "FALSIFY-BGN-001: QWEN2_0_5B_NUM_LAYERS must match qwen2.yaml"
    );

    // LLaMA YAML: vendor: Meta, 8b hidden_dim: 4096, num_layers: 32
    assert_eq!(
        LLAMA_VENDOR, "Meta",
        "FALSIFY-BGN-001: LLAMA_VENDOR must match llama.yaml"
    );
    assert_eq!(
        LLAMA_8B_HIDDEN_DIM, 4096,
        "FALSIFY-BGN-001: LLAMA_8B_HIDDEN_DIM must match llama.yaml"
    );
    assert_eq!(
        LLAMA_8B_NUM_LAYERS, 32,
        "FALSIFY-BGN-001: LLAMA_8B_NUM_LAYERS must match llama.yaml"
    );

    // BERT YAML: vendor: Google, base hidden_dim: 768
    assert_eq!(
        BERT_VENDOR, "Google",
        "FALSIFY-BGN-001: BERT_VENDOR must match bert.yaml"
    );
    assert_eq!(
        BERT_BASE_HIDDEN_DIM, 768,
        "FALSIFY-BGN-001: BERT_BASE_HIDDEN_DIM must match bert.yaml"
    );

    // Whisper YAML: vendor: OpenAI
    assert_eq!(
        WHISPER_VENDOR, "OpenAI",
        "FALSIFY-BGN-001: WHISPER_VENDOR must match whisper.yaml"
    );

    // Phase 5 families
    assert_eq!(
        MISTRAL_VENDOR, "Mistral AI",
        "FALSIFY-BGN-001: MISTRAL_VENDOR must match mistral.yaml"
    );
    assert_eq!(
        PHI_VENDOR, "Microsoft",
        "FALSIFY-BGN-001: PHI_VENDOR must match phi.yaml"
    );
    assert_eq!(
        GEMMA_VENDOR, "Google",
        "FALSIFY-BGN-001: GEMMA_VENDOR must match gemma.yaml"
    );
    assert_eq!(
        DEEPSEEK_VENDOR, "DeepSeek",
        "FALSIFY-BGN-001: DEEPSEEK_VENDOR must match deepseek.yaml"
    );
}

#[test]
fn falsify_bgn_001_known_families_matches_yaml_directory() {
    // KNOWN_FAMILIES should contain exactly the families from YAML files
    assert!(
        KNOWN_FAMILIES.len() >= 8,
        "FALSIFY-BGN-001: KNOWN_FAMILIES should have >= 8 families, got {}",
        KNOWN_FAMILIES.len()
    );

    let expected = [
        "bert", "deepseek", "gemma", "llama", "mistral", "phi", "qwen2", "whisper",
    ];
    for family in &expected {
        assert!(
            KNOWN_FAMILIES.contains(family),
            "FALSIFY-BGN-001: KNOWN_FAMILIES must contain '{family}'"
        );
    }
}

#[test]
fn falsify_bgn_001_registry_from_codegen_matches_yaml() {
    // Build registry from codegen and verify it matches YAML values
    let registry = build_default_registry();

    // Verify count matches
    assert!(
        registry.len() >= 8,
        "FALSIFY-BGN-001: Registry should have >= 8 families from codegen, got {}",
        registry.len()
    );

    // Verify a specific family's config values match YAML
    let qwen2 = registry.get("qwen2").expect("qwen2 in registry");
    let config = qwen2.config();

    // These values come from qwen2.yaml via build.rs codegen
    assert_eq!(
        config.tensor_template.embedding,
        "model.embed_tokens.weight"
    );
    assert_eq!(
        config.tensor_template.lm_head.as_deref(),
        Some("lm_head.weight")
    );
    assert_eq!(
        config.tensor_template.final_norm.as_deref(),
        Some("model.norm.weight")
    );

    // Verify per_layer tensor patterns from YAML
    assert!(config
        .tensor_template
        .per_layer
        .get("q_proj")
        .is_some_and(|v| v.as_deref() == Some("model.layers.{n}.self_attn.q_proj.weight")));
}

// =============================================================================
// §7.4 — FALSIFY-BGN-002: Invalid YAML Causes Build Failure
// =============================================================================
//
// Prediction: A YAML file with missing required fields causes cargo build to fail.
//
// This is a build-time test and cannot be verified at runtime. Instead, we verify
// that the build.rs parser is strict enough by testing its output:
// - All families in the registry have non-empty family names
// - All families have at least one size variant
// - All families have non-empty embedding tensor names

#[test]
fn falsify_bgn_002_all_families_have_required_fields() {
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).unwrap_or_else(|| {
            panic!("FALSIFY-BGN-002: Family '{family_name}' must be in registry")
        });

        let config = family.config();

        // Required: non-empty family name
        assert!(
            !config.family.is_empty(),
            "FALSIFY-BGN-002: Family name must not be empty"
        );

        // Required: non-empty display name
        assert!(
            !config.display_name.is_empty(),
            "FALSIFY-BGN-002: {family_name} display_name must not be empty"
        );

        // Required: non-empty vendor
        assert!(
            !config.vendor.is_empty(),
            "FALSIFY-BGN-002: {family_name} vendor must not be empty"
        );

        // Required: at least one architecture
        assert!(
            !config.architectures.is_empty(),
            "FALSIFY-BGN-002: {family_name} must have at least one architecture"
        );

        // Required: at least one size variant
        assert!(
            !config.size_variants.is_empty(),
            "FALSIFY-BGN-002: {family_name} must have at least one size variant"
        );

        // Required: non-empty embedding tensor
        assert!(
            !config.tensor_template.embedding.is_empty(),
            "FALSIFY-BGN-002: {family_name} must have a non-empty embedding tensor name"
        );

        // Verify each size variant has valid dimensions
        for (size_name, size_config) in &config.size_variants {
            assert!(
                size_config.hidden_dim > 0,
                "FALSIFY-BGN-002: {family_name}/{size_name} hidden_dim must be > 0"
            );
            assert!(
                size_config.num_layers > 0,
                "FALSIFY-BGN-002: {family_name}/{size_name} num_layers must be > 0"
            );
            assert!(
                size_config.num_heads > 0,
                "FALSIFY-BGN-002: {family_name}/{size_name} num_heads must be > 0"
            );
        }
    }
}

#[test]
fn falsify_bgn_002_build_rs_exists_and_references_contracts() {
    // Verify the build.rs file exists and references the contracts directory
    let project_root = find_project_root();
    let build_rs = project_root.join("build.rs");
    assert!(
        build_rs.exists(),
        "FALSIFY-BGN-002: build.rs must exist for YAML-to-Rust codegen"
    );

    let content = std::fs::read_to_string(&build_rs).expect("read build.rs");
    assert!(
        content.contains("contracts/model-families"),
        "FALSIFY-BGN-002: build.rs must reference contracts/model-families"
    );
    assert!(
        content.contains("rerun-if-changed"),
        "FALSIFY-BGN-002: build.rs must use cargo:rerun-if-changed for YAML tracking"
    );
}

// =============================================================================
// Cross-Cutting Falsification: Expected Tensor Count Consistency
// =============================================================================

#[test]
fn falsify_cross_expected_tensor_count_consistent_with_validation() {
    let registry = build_default_registry();
    let qwen2 = registry
        .detect_from_model_type("qwen2")
        .expect("qwen2 detected");

    // Get expected count
    let expected_count = qwen2
        .expected_tensor_count("0.5b")
        .expect("expected count for 0.5b");

    // Build the full tensor set
    let config = qwen2.config();
    let mut names: Vec<String> = Vec::new();
    names.push(config.tensor_template.embedding.clone());
    if let Some(ref lm_head) = config.tensor_template.lm_head {
        names.push(lm_head.clone());
    }
    if let Some(ref final_norm) = config.tensor_template.final_norm {
        names.push(final_norm.clone());
    }
    for layer_idx in 0..24 {
        for pat in config.tensor_template.per_layer.values().flatten() {
            names.push(pat.replace("{n}", &layer_idx.to_string()));
        }
    }

    assert_eq!(
        names.len(),
        expected_count,
        "Expected tensor count ({expected_count}) must match actual generated tensor names ({})",
        names.len()
    );
}

// =============================================================================
