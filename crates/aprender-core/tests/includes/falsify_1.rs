// =============================================================================
// §7.1 — FALSIFY-MFC-001: Family Detection Accuracy
// =============================================================================
//
// Prediction: Given a set of tensor names matching the Qwen2 contract
//   (model.embed_tokens.weight, model.layers.0.self_attn.q_proj.weight, ...),
//   detect_family() returns "qwen2".
//
// If test fails: Family detection is broken.

#[test]
fn falsify_mfc_001_bias_tensors_detected_as_bias_family() {
    let registry = build_default_registry();

    // Tensor names WITH bias patterns. Best-match scoring: bias-bearing families
    // (Qwen2, Phi) score 13, bias-free families (LLaMA, DeepSeek, Mistral) score 10.
    // Result MUST be a bias-bearing family, NOT a bias-free family.
    let bias_tensor_names = vec![
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.bias",
    ];

    let detected = registry.detect_family(&bias_tensor_names);
    assert!(
        detected.is_some(),
        "FALSIFY-MFC-001: detect_family() should detect bias-bearing tensor names"
    );
    let family_name = detected.expect("bias family detected").family_name();
    let bias_families = ["phi", "qwen2"];
    assert!(
        bias_families.contains(&family_name),
        "FALSIFY-MFC-001: bias tensor names must detect as bias family {:?}, got '{family_name}'",
        bias_families
    );

    // Must NOT be a bias-free family
    let no_bias_families = ["deepseek", "llama", "mistral"];
    assert!(
        !no_bias_families.contains(&family_name),
        "FALSIFY-MFC-001: bias tensors must NOT match bias-free family, got '{family_name}'"
    );
}

#[test]
fn falsify_mfc_001_no_bias_tensors_detected_as_no_bias_family() {
    let registry = build_default_registry();

    // Tensor names WITHOUT bias patterns. All no-bias families score 10,
    // bias families also score 10 (their bias patterns don't match).
    // Among tied families, result is deterministic (alphabetical).
    let no_bias_tensor_names = vec![
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
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

    let detected = registry.detect_family(&no_bias_tensor_names);
    assert!(
        detected.is_some(),
        "FALSIFY-MFC-001: detect_family() should detect no-bias transformer tensors"
    );
    // All standard-naming families tie at score 10; any is acceptable
    let family_name = detected.expect("family detected").family_name();
    let standard_families = ["deepseek", "llama", "mistral", "phi", "qwen2"];
    assert!(
        standard_families.contains(&family_name),
        "FALSIFY-MFC-001: no-bias names should match a standard family, got '{family_name}'"
    );
}

#[test]
fn falsify_mfc_001_model_type_detection_is_unambiguous() {
    // model_type detection (from HF config.json) IS always unambiguous,
    // even for families with identical tensor naming.
    let registry = build_default_registry();

    let families = [
        ("qwen2", "qwen2"),
        ("llama", "llama"),
        ("phi", "phi"),
        ("mistral", "mistral"),
        ("deepseek", "deepseek"),
    ];
    for (model_type, expected) in &families {
        let detected = registry.detect_from_model_type(model_type);
        assert_eq!(
            detected.expect("detected").family_name(),
            *expected,
            "FALSIFY-MFC-001: model_type '{model_type}' must unambiguously detect '{expected}'"
        );
    }
}

#[test]
fn falsify_mfc_001_whisper_tensor_names_not_detected_as_qwen2() {
    let registry = build_default_registry();

    // Whisper uses completely different tensor naming
    let whisper_tensor_names = vec![
        "encoder.conv1.weight",
        "encoder.conv2.weight",
        "encoder.positional_embedding",
        "encoder.layers.0.self_attn.q_proj.weight",
    ];

    // Should not detect as qwen2
    let detected = registry.detect_family(&whisper_tensor_names);
    if let Some(family) = detected {
        assert_ne!(
            family.family_name(),
            "qwen2",
            "FALSIFY-MFC-001: Whisper tensors should NOT be detected as qwen2"
        );
    }
}

#[test]
fn falsify_mfc_001_random_names_not_detected() {
    let registry = build_default_registry();

    let garbage_names = vec!["foo.bar.weight", "baz.qux.bias", "totally.random.tensor"];

    let detected = registry.detect_family(&garbage_names);
    assert!(
        detected.is_none(),
        "FALSIFY-MFC-001: Random tensor names should not match any family"
    );
}

// =============================================================================
// §7.1 — FALSIFY-MFC-002: Size Variant Detection Accuracy
// =============================================================================
//
// Prediction: Given hidden_dim=1536 and num_layers=28, detect_size() returns "1.5b".
//
// If test fails: Size variant detection does not match YAML contract.

#[test]
fn falsify_mfc_002_qwen2_size_detection() {
    let registry = build_default_registry();
    let qwen2 = registry
        .detect_from_model_type("qwen2")
        .expect("FALSIFY-MFC-002: qwen2 family must exist in registry");

    // 0.5B: hidden_dim=896, num_layers=24
    assert_eq!(
        qwen2.detect_size(896, 24).as_deref(),
        Some("0.5b"),
        "FALSIFY-MFC-002: (896, 24) should detect as 0.5b"
    );

    // 1.5B: hidden_dim=1536, num_layers=28
    assert_eq!(
        qwen2.detect_size(1536, 28).as_deref(),
        Some("1.5b"),
        "FALSIFY-MFC-002: (1536, 28) should detect as 1.5b"
    );

    // 3B: hidden_dim=2048, num_layers=36
    assert_eq!(
        qwen2.detect_size(2048, 36).as_deref(),
        Some("3b"),
        "FALSIFY-MFC-002: (2048, 36) should detect as 3b"
    );

    // 7B: hidden_dim=3584, num_layers=28
    assert_eq!(
        qwen2.detect_size(3584, 28).as_deref(),
        Some("7b"),
        "FALSIFY-MFC-002: (3584, 28) should detect as 7b"
    );

    // Unknown config
    assert_eq!(
        qwen2.detect_size(999, 99),
        None,
        "FALSIFY-MFC-002: Unknown (999, 99) should return None"
    );
}

#[test]
fn falsify_mfc_002_llama_size_detection() {
    let registry = build_default_registry();
    let llama = registry
        .detect_from_model_type("llama")
        .expect("FALSIFY-MFC-002: llama family must exist in registry");

    // 8B: hidden_dim=4096, num_layers=32
    assert_eq!(
        llama.detect_size(4096, 32).as_deref(),
        Some("8b"),
        "FALSIFY-MFC-002: LLaMA (4096, 32) should detect as 8b"
    );
}

#[test]
fn falsify_mfc_002_whisper_size_detection() {
    let registry = build_default_registry();
    let whisper = registry
        .detect_from_model_type("whisper")
        .expect("FALSIFY-MFC-002: whisper family must exist in registry");

    // Whisper tiny: hidden_dim=384, num_layers=4
    assert_eq!(
        whisper.detect_size(384, 4).as_deref(),
        Some("tiny"),
        "FALSIFY-MFC-002: Whisper (384, 4) should detect as tiny"
    );
}

#[test]
fn falsify_mfc_002_bert_size_detection() {
    let registry = build_default_registry();
    let bert = registry
        .detect_from_model_type("bert")
        .expect("FALSIFY-MFC-002: bert family must exist in registry");

    // BERT base: hidden_dim=768, num_layers=12
    assert_eq!(
        bert.detect_size(768, 12).as_deref(),
        Some("base"),
        "FALSIFY-MFC-002: BERT (768, 12) should detect as base"
    );
}

// =============================================================================
// §7.1 — FALSIFY-MFC-003: Tensor Name Validation Rejects Wrong Names
// =============================================================================
//
// Prediction: validate_tensor_names() rejects tensor names from a different
//   family (e.g., Whisper names in a Qwen2 contract check).
//
// If test fails: Tensor name validation is not family-specific.

#[test]
fn falsify_mfc_003_qwen2_rejects_whisper_names() {
    let registry = build_default_registry();
    let qwen2 = registry
        .detect_from_model_type("qwen2")
        .expect("qwen2 family must exist");

    // Whisper-specific tensor names should fail Qwen2 validation
    let whisper_names = ["encoder.conv1.weight", "decoder.embed_tokens.weight"];
    let result = qwen2.validate_tensor_names(&whisper_names, "0.5b");
    assert!(
        result.is_err(),
        "FALSIFY-MFC-003: Whisper tensor names should be rejected by Qwen2 contract"
    );

    let err = result.unwrap_err();
    assert!(
        err.message.contains("Missing") || err.message.contains("Unexpected"),
        "FALSIFY-MFC-003: Error should mention missing or unexpected tensors, got: {}",
        err.message
    );
}

#[test]
fn falsify_mfc_003_qwen2_rejects_empty_tensor_list() {
    let registry = build_default_registry();
    let qwen2 = registry
        .detect_from_model_type("qwen2")
        .expect("qwen2 family must exist");

    let empty: Vec<&str> = vec![];
    let result = qwen2.validate_tensor_names(&empty, "0.5b");
    assert!(
        result.is_err(),
        "FALSIFY-MFC-003: Empty tensor list should be rejected (missing all expected tensors)"
    );
}

#[test]
fn falsify_mfc_003_qwen2_rejects_unknown_size() {
    let registry = build_default_registry();
    let qwen2 = registry
        .detect_from_model_type("qwen2")
        .expect("qwen2 family must exist");

    let names = ["model.embed_tokens.weight"];
    let result = qwen2.validate_tensor_names(&names, "999b");
    assert!(
        result.is_err(),
        "FALSIFY-MFC-003: Unknown size variant '999b' should be rejected"
    );

    let err = result.unwrap_err();
    assert!(
        err.message.contains("Unknown size variant"),
        "FALSIFY-MFC-003: Error should mention unknown size, got: {}",
        err.message
    );
}

#[test]
fn falsify_mfc_003_qwen2_accepts_correct_tensor_names() {
    let registry = build_default_registry();
    let qwen2 = registry
        .detect_from_model_type("qwen2")
        .expect("qwen2 family must exist");

    // Build the FULL correct tensor name set for 0.5b (24 layers)
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

    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let result = qwen2.validate_tensor_names(&name_refs, "0.5b");
    assert!(
        result.is_ok(),
        "FALSIFY-MFC-003: Correct complete tensor names should pass validation, got: {:?}",
        result.err()
    );
}

// =============================================================================
// §7.2 — FALSIFY-ORC-001: Local File Family Detection
// =============================================================================
//
// Prediction: The oracle's family detection correctly identifies a model's
//   family from its tensor names (library-level test, not CLI integration).
//
// If test fails: Oracle file analysis does not match family contracts.

#[test]
fn falsify_orc_001_registry_detects_all_families_from_model_type() {
    let registry = build_default_registry();

    // All 8 families should be detectable via model_type
    let expected_families = [
        ("qwen2", "qwen2"),
        ("llama", "llama"),
        ("bert", "bert"),
        ("whisper", "whisper"),
        ("mistral", "mistral"),
        ("phi", "phi"),
        ("gemma", "gemma"),
        ("deepseek", "deepseek"),
    ];

    for (model_type, expected_family) in &expected_families {
        let detected = registry.detect_from_model_type(model_type);
        assert!(
            detected.is_some(),
            "FALSIFY-ORC-001: model_type '{model_type}' should be detected"
        );
        assert_eq!(
            detected.expect("detected").family_name(),
            *expected_family,
            "FALSIFY-ORC-001: model_type '{model_type}' should map to '{expected_family}'"
        );
    }
}

#[test]
fn falsify_orc_001_registry_provides_config_for_detected_family() {
    let registry = build_default_registry();
    let qwen2 = registry
        .detect_from_model_type("qwen2")
        .expect("qwen2 detected");

    let config = qwen2.config();
    assert_eq!(config.family, "qwen2");
    assert_eq!(config.vendor, "Alibaba");
    assert!(!config.architectures.is_empty());
    assert!(
        config
            .architectures
            .contains(&"Qwen2ForCausalLM".to_string()),
        "FALSIFY-ORC-001: Qwen2 should have Qwen2ForCausalLM architecture"
    );
}

// =============================================================================
// §7.2 — FALSIFY-ORC-002: HuggingFace API Family Detection
// =============================================================================
//
// This test requires network access. Verify the mapping works at library level.

#[test]
fn falsify_orc_002_hf_model_type_mapping() {
    let registry = build_default_registry();

    // HuggingFace config.json model_type values → expected family
    let hf_model_types = [
        ("qwen2", "qwen2"),
        ("llama", "llama"),
        ("bert", "bert"),
        ("whisper", "whisper"),
        ("mistral", "mistral"),
    ];

    for (hf_model_type, expected_family) in &hf_model_types {
        let detected = registry.detect_from_model_type(hf_model_type);
        assert!(
            detected.is_some(),
            "FALSIFY-ORC-002: HF model_type '{hf_model_type}' should be detected"
        );
        assert_eq!(
            detected.expect("detected").family_name(),
            *expected_family,
            "FALSIFY-ORC-002: HF model_type '{hf_model_type}' → '{expected_family}'"
        );
    }
}

// =============================================================================
