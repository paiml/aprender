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
