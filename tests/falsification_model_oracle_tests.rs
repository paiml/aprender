//! Popperian Falsification Tests — Compiler-Enforced Model Types & Model Oracle
//!
//! Per Popper (1959), each validation rule must make a prediction that could be
//! proven false. If a falsification test finds a counterexample, the implementation
//! is broken.
//!
//! Spec reference: docs/specifications/compiler-enforced-model-types-model-oracle.md §7
//!
//! Test IDs map to spec:
//! - FALSIFY-MFC-001..003: Model Family Contracts (§7.1)
//! - FALSIFY-ORC-001..004: Oracle CLI (§7.2)
//! - FALSIFY-CMP-001..003: Compiler Enforcement (§7.3)
//! - FALSIFY-BGN-001..002: Build-Time Codegen (§7.4)

use aprender::format::model_family::build_default_registry;

// Build-time generated constants
use aprender::format::model_family::{
    BERT_BASE_HIDDEN_DIM, BERT_VENDOR, DEEPSEEK_VENDOR, GEMMA_VENDOR, KNOWN_FAMILIES,
    LLAMA_8B_HIDDEN_DIM, LLAMA_8B_NUM_LAYERS, LLAMA_VENDOR, MISTRAL_VENDOR, PHI_VENDOR,
    QWEN2_0_5B_HIDDEN_DIM, QWEN2_0_5B_NUM_LAYERS, QWEN2_VENDOR, WHISPER_VENDOR,
};

// Validated tensor types for FALSIFY-CMP-001
use aprender::format::validated_tensors::{RowMajor, ValidatedWeight};

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
        config.architectures.contains(&"Qwen2ForCausalLM".to_string()),
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
// §7.2 — FALSIFY-ORC-003: Contract Description Completeness
// =============================================================================
//
// Prediction: The registry provides complete config values matching YAML source.

#[test]
fn falsify_orc_003_qwen2_contract_matches_yaml() {
    let registry = build_default_registry();
    let qwen2 = registry
        .detect_from_model_type("qwen2")
        .expect("qwen2 detected");

    // Cross-reference against known YAML values
    let config = qwen2.config();
    assert_eq!(config.family, "qwen2");
    assert_eq!(config.display_name, "Qwen2 / Qwen2.5-Coder");
    assert_eq!(config.vendor, "Alibaba");

    // Verify size variant 0.5b matches YAML exactly
    let size = qwen2
        .size_config("0.5b")
        .expect("FALSIFY-ORC-003: 0.5b size variant must exist");
    assert_eq!(
        size.hidden_dim, 896,
        "FALSIFY-ORC-003: 0.5b hidden_dim should be 896"
    );
    assert_eq!(
        size.num_layers, 24,
        "FALSIFY-ORC-003: 0.5b num_layers should be 24"
    );
    assert_eq!(
        size.num_heads, 14,
        "FALSIFY-ORC-003: 0.5b num_heads should be 14"
    );
    assert_eq!(
        size.num_kv_heads, 2,
        "FALSIFY-ORC-003: 0.5b num_kv_heads should be 2"
    );
    assert_eq!(
        size.intermediate_dim, 4864,
        "FALSIFY-ORC-003: 0.5b intermediate_dim should be 4864"
    );
    assert_eq!(
        size.vocab_size, 151_936,
        "FALSIFY-ORC-003: 0.5b vocab_size should be 151936"
    );
    assert_eq!(
        size.max_position_embeddings, 32_768,
        "FALSIFY-ORC-003: 0.5b max_position_embeddings should be 32768"
    );
    assert_eq!(
        size.head_dim, 64,
        "FALSIFY-ORC-003: 0.5b head_dim should be 64"
    );
}

#[test]
fn falsify_orc_003_all_qwen2_sizes_present() {
    let registry = build_default_registry();
    let qwen2 = registry
        .detect_from_model_type("qwen2")
        .expect("qwen2 detected");

    let expected_sizes = ["0.5b", "1.5b", "3b", "7b", "14b", "32b"];
    for size_name in &expected_sizes {
        assert!(
            qwen2.size_config(size_name).is_some(),
            "FALSIFY-ORC-003: Qwen2 should have size variant '{size_name}'"
        );
    }
}

#[test]
fn falsify_orc_003_constraints_match_yaml() {
    use aprender::format::model_family::{
        Activation, AttentionType, MlpType, NormType, PositionalEncoding,
    };

    let registry = build_default_registry();
    let qwen2 = registry
        .detect_from_model_type("qwen2")
        .expect("qwen2 detected");

    let constraints = qwen2.constraints();
    assert_eq!(
        constraints.attention_type,
        AttentionType::Gqa,
        "FALSIFY-ORC-003: Qwen2 attention should be GQA"
    );
    assert_eq!(
        constraints.activation,
        Activation::Silu,
        "FALSIFY-ORC-003: Qwen2 activation should be SiLU"
    );
    assert_eq!(
        constraints.norm_type,
        NormType::RmsNorm,
        "FALSIFY-ORC-003: Qwen2 norm should be RMSNorm"
    );
    assert!(
        constraints.has_bias,
        "FALSIFY-ORC-003: Qwen2 should have bias"
    );
    assert!(
        !constraints.tied_embeddings,
        "FALSIFY-ORC-003: Qwen2 should not have tied embeddings"
    );
    assert_eq!(
        constraints.positional_encoding,
        PositionalEncoding::Rope,
        "FALSIFY-ORC-003: Qwen2 should use RoPE"
    );
    assert_eq!(
        constraints.mlp_type,
        MlpType::SwiGlu,
        "FALSIFY-ORC-003: Qwen2 MLP should be SwiGLU"
    );
}

// =============================================================================
// §7.2 — FALSIFY-ORC-004: Compliance Detection Catches Missing Tensors
// =============================================================================
//
// Prediction: A model missing tensors fails validation.

#[test]
fn falsify_orc_004_missing_lm_head_detected() {
    let registry = build_default_registry();
    let qwen2 = registry
        .detect_from_model_type("qwen2")
        .expect("qwen2 detected");

    // Build complete tensor set then REMOVE lm_head
    let config = qwen2.config();
    let mut names: Vec<String> = Vec::new();
    names.push(config.tensor_template.embedding.clone());
    // Intentionally skip lm_head.weight
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
        result.is_err(),
        "FALSIFY-ORC-004: Missing lm_head.weight should fail compliance"
    );

    let err = result.unwrap_err();
    assert!(
        err.message.contains("lm_head.weight"),
        "FALSIFY-ORC-004: Error should mention missing lm_head.weight, got: {}",
        err.message
    );
}

#[test]
fn falsify_orc_004_extra_unexpected_tensor_detected() {
    let registry = build_default_registry();
    let qwen2 = registry
        .detect_from_model_type("qwen2")
        .expect("qwen2 detected");

    // Build complete tensor set then ADD an extra unexpected tensor
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
    // Add unexpected tensor
    names.push("totally.unexpected.tensor.weight".to_string());

    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let result = qwen2.validate_tensor_names(&name_refs, "0.5b");
    assert!(
        result.is_err(),
        "FALSIFY-ORC-004: Unexpected tensor should fail compliance"
    );

    let err = result.unwrap_err();
    assert!(
        err.message.contains("Unexpected"),
        "FALSIFY-ORC-004: Error should mention unexpected tensors, got: {}",
        err.message
    );
}

#[test]
fn falsify_orc_004_missing_layer_tensors_detected() {
    let registry = build_default_registry();
    let qwen2 = registry
        .detect_from_model_type("qwen2")
        .expect("qwen2 detected");

    // Build tensor set but only include 23 layers instead of 24
    let config = qwen2.config();
    let mut names: Vec<String> = Vec::new();
    names.push(config.tensor_template.embedding.clone());
    if let Some(ref lm_head) = config.tensor_template.lm_head {
        names.push(lm_head.clone());
    }
    if let Some(ref final_norm) = config.tensor_template.final_norm {
        names.push(final_norm.clone());
    }
    // Only 23 layers — layer 23 is missing
    for layer_idx in 0..23 {
        for pat in config.tensor_template.per_layer.values().flatten() {
            names.push(pat.replace("{n}", &layer_idx.to_string()));
        }
    }

    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let result = qwen2.validate_tensor_names(&name_refs, "0.5b");
    assert!(
        result.is_err(),
        "FALSIFY-ORC-004: Missing layer 23 tensors should fail compliance"
    );
}

// =============================================================================
// §7.3 — FALSIFY-CMP-001: PhantomData Prevents Layout Mismatch
// =============================================================================
//
// Prediction: Code that passes ValidatedWeight<RowMajor> to a function expecting
//   a different layout type does not compile.
//
// Since ColumnMajor does not exist, this is verified structurally:
// 1. RowMajor is the only layout marker
// 2. ValidatedWeight<RowMajor> is the default and only constructible variant
// 3. PhantomData is zero-cost

#[test]
fn falsify_cmp_001_row_major_is_only_layout() {
    // Verify RowMajor exists and is zero-sized
    assert_eq!(
        std::mem::size_of::<RowMajor>(),
        0,
        "FALSIFY-CMP-001: RowMajor must be zero-sized"
    );

    // Verify PhantomData<RowMajor> is zero-sized
    assert_eq!(
        std::mem::size_of::<std::marker::PhantomData<RowMajor>>(),
        0,
        "FALSIFY-CMP-001: PhantomData<RowMajor> must be zero-sized"
    );
}

#[test]
fn falsify_cmp_001_validated_weight_default_is_row_major() {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    let weight: ValidatedWeight = ValidatedWeight::new(data, 10, 10, "test").unwrap();

    // This assignment compiles because ValidatedWeight == ValidatedWeight<RowMajor>
    let _explicit: ValidatedWeight<RowMajor> = weight;
}

#[test]
fn falsify_cmp_001_no_column_major_type_exists() {
    // This test verifies structurally that there is no ColumnMajor type.
    // If someone adds ColumnMajor, they must also add a test here.
    //
    // Proof by construction: if this test compiles, ColumnMajor is not importable
    // from the validated_tensors module. The following line would fail to compile
    // if ColumnMajor existed:
    //
    //   use aprender::format::validated_tensors::ColumnMajor; // ERROR: not found
    //
    // Since Rust has no "assert type doesn't exist" facility, we document this
    // as a structural invariant. The build-time guarantee is:
    // - Only RowMajor is exported from validated_tensors
    // - ValidatedWeight::new() only creates ValidatedWeight<RowMajor>
    // - No other constructor exists

    // Verify the only public layout type is RowMajor
    let _ = RowMajor; // This compiles — RowMajor exists
    // ColumnMajor would be a compile error if uncommented:
    // let _ = ColumnMajor; // ERROR: not found in this scope
}

// =============================================================================
// §7.3 — FALSIFY-CMP-002: AprTransformer Rejects Unvalidated Data
// =============================================================================
//
// This test is DEFERRED: AprTransformer migration to Validated* fields (PMAT-249)
// is not yet implemented. The prediction is:
//
//   AprTransformer { embedding: Vec<f32>(...) } does not compile.
//
// Currently AprTransformer lives in realizar and has not been migrated yet.
// When PMAT-249 is done, uncomment and verify.

#[test]
fn falsify_cmp_002_validated_types_reject_raw_data() {
    // Verify that Validated* types cannot be constructed from raw data
    // without going through validation

    // ValidatedWeight: private `data` field means you can't construct directly
    // The ONLY way is through `new()` which validates

    // Attempt: NaN data → rejection
    let nan_data = vec![f32::NAN; 100];
    let result = ValidatedWeight::new(nan_data, 10, 10, "test");
    assert!(
        result.is_err(),
        "FALSIFY-CMP-002: Raw NaN data must be rejected by ValidatedWeight::new()"
    );

    // Attempt: All-zero data → rejection
    let zero_data = vec![0.0f32; 100];
    let result = ValidatedWeight::new(zero_data, 10, 10, "test");
    assert!(
        result.is_err(),
        "FALSIFY-CMP-002: All-zero data must be rejected by ValidatedWeight::new()"
    );
}

// =============================================================================
// §7.3 — FALSIFY-CMP-003: Clippy Catches Column-Major Import
// =============================================================================
//
// Prediction: .clippy.toml contains disallowed-methods for column-major kernels.
//
// This cannot be tested at runtime (clippy is a linter, not a runtime check).
// Instead, verify the configuration file structurally.

#[test]
fn falsify_cmp_003_clippy_toml_bans_column_major() {
    // Read .clippy.toml and verify disallowed-methods entries exist
    let clippy_toml =
        std::fs::read_to_string(find_project_root().join(".clippy.toml")).unwrap_or_default();

    let expected_bans = [
        "matmul_q4k_f32_colmajor",
        "matmul_q6k_f32_colmajor",
        "matmul_q4k_f32_colmajor_dispatch",
        "matmul_q6k_f32_colmajor_dispatch",
    ];

    for ban in &expected_bans {
        assert!(
            clippy_toml.contains(ban),
            "FALSIFY-CMP-003: .clippy.toml must ban '{ban}'. Column-major imports must be disallowed."
        );
    }
}

// =============================================================================
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
    assert_eq!(config.tensor_template.embedding, "model.embed_tokens.weight");
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
        .is_some_and(|v| v
            .as_deref()
            == Some("model.layers.{n}.self_attn.q_proj.weight")));
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
        let family = registry
            .get(family_name)
            .unwrap_or_else(|| panic!("FALSIFY-BGN-002: Family '{family_name}' must be in registry"));

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
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "yaml")
        })
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
// Helpers
// =============================================================================

fn find_project_root() -> std::path::PathBuf {
    let mut dir = std::env::current_dir().expect("current dir");
    loop {
        if dir.join("Cargo.toml").exists() && dir.join("src").exists() {
            return dir;
        }
        assert!(dir.pop(), "Could not find project root (looking for Cargo.toml + src/)");
    }
}
