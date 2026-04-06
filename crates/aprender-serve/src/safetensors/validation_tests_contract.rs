
// ============================================================================
// ContractValidationError Display/Error/From impls
// ============================================================================

#[test]
fn test_contract_validation_error_display() {
    let err = ContractValidationError {
        tensor_name: "token_embedding".to_string(),
        rule_id: "F-DATA-QUALITY-001".to_string(),
        message: "94.5% zeros detected".to_string(),
    };
    let display = format!("{}", err);
    assert!(display.contains("F-DATA-QUALITY-001"));
    assert!(display.contains("token_embedding"));
    assert!(display.contains("94.5% zeros"));
}

#[test]
fn test_contract_validation_error_debug() {
    let err = ContractValidationError {
        tensor_name: "test".to_string(),
        rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
        message: "shape mismatch".to_string(),
    };
    let debug = format!("{:?}", err);
    assert!(debug.contains("ContractValidationError"));
}

#[test]
fn test_contract_validation_error_clone() {
    let err = ContractValidationError {
        tensor_name: "test".to_string(),
        rule_id: "F-DATA-QUALITY-002".to_string(),
        message: "NaN found".to_string(),
    };
    let cloned = err.clone();
    assert_eq!(cloned.tensor_name, "test");
    assert_eq!(cloned.rule_id, "F-DATA-QUALITY-002");
}

#[test]
fn test_contract_validation_error_is_std_error() {
    let err = ContractValidationError {
        tensor_name: "test".to_string(),
        rule_id: "TEST".to_string(),
        message: "test error".to_string(),
    };
    // Verify it implements std::error::Error (via trait bound)
    let _: &dyn std::error::Error = &err;
}

#[test]
fn test_contract_validation_error_from_into_realizar_error() {
    let err = ContractValidationError {
        tensor_name: "embed".to_string(),
        rule_id: "F-DATA-QUALITY-001".to_string(),
        message: "too many zeros".to_string(),
    };
    let realizar_err: RealizarError = err.into();
    let msg = format!("{}", realizar_err);
    assert!(msg.contains("F-DATA-QUALITY-001"));
    assert!(msg.contains("embed"));
}

// ============================================================================
// validate_embedding edge cases
// ============================================================================

#[test]
fn test_validate_embedding_with_inf() {
    let vocab = 10;
    let dim = 8;
    let mut data: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.01).collect();
    data[5] = f32::INFINITY;
    let result = validate_embedding("test", &data, vocab, dim);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("Inf")));
}

#[test]
fn test_validate_embedding_zero_l2() {
    let vocab = 10;
    let dim = 8;
    let data = vec![1e-11f32; vocab * dim]; // Very small L2
    let result = validate_embedding("test", &data, vocab, dim);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("L2")));
}

#[test]
fn test_validate_embedding_constant_values() {
    let vocab = 10;
    let dim = 8;
    let data = vec![0.5f32; vocab * dim]; // All identical
    let result = validate_embedding("test", &data, vocab, dim);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("identical")));
}

// ============================================================================
// validate_weight edge cases
// ============================================================================

#[test]
fn test_validate_weight_shape_mismatch() {
    let data = vec![0.1f32; 100];
    let result = validate_weight("test", &data, 5, 5); // expects 25
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("Shape")));
}

#[test]
fn test_validate_weight_nan() {
    let mut data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    data[50] = f32::NAN;
    let result = validate_weight("test", &data, 10, 10);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("NaN")));
}

#[test]
fn test_validate_weight_inf() {
    let mut data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    data[50] = f32::INFINITY;
    let result = validate_weight("test", &data, 10, 10);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("Inf")));
}

#[test]
fn test_validate_weight_zero_l2() {
    let data = vec![1e-11f32; 100];
    let result = validate_weight("test", &data, 10, 10);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("L2")));
}

#[test]
fn test_validate_weight_high_density_zeros() {
    let mut data = vec![0.0f32; 100];
    // Only 10% non-zero -> 90% zeros, exceeds 80% threshold
    for i in 90..100 {
        data[i] = 0.1;
    }
    let result = validate_weight("test", &data, 10, 10);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("DENSITY")));
}

#[test]
fn test_validate_weight_good_data() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let result = validate_weight("test", &data, 10, 10);
    assert!(
        result.passed,
        "Good data should pass: {:?}",
        result.failures
    );
}

// ============================================================================
// validate_vector edge cases
// ============================================================================

#[test]
fn test_validate_vector_length_mismatch() {
    let data = vec![1.0f32; 50];
    let result = validate_vector("test", &data, 100);
    assert!(!result.passed);
    assert!(result.failures.iter().any(|f| f.contains("Length")));
}

#[test]
fn test_validate_vector_nan() {
    let mut data = vec![1.0f32; 64];
    data[32] = f32::NAN;
    let result = validate_vector("test", &data, 64);
    assert!(!result.passed);
}

#[test]
fn test_validate_vector_inf() {
    let mut data = vec![1.0f32; 64];
    data[32] = f32::INFINITY;
    let result = validate_vector("test", &data, 64);
    assert!(!result.passed);
}

#[test]
fn test_validate_vector_good_data() {
    let data = vec![1.0f32; 64];
    let result = validate_vector("test", &data, 64);
    assert!(result.passed);
}

// ============================================================================
// enforce_embedding_validation and enforce_weight_validation
// ============================================================================

#[test]
fn test_enforce_embedding_validation_passes() {
    let vocab = 10;
    let dim = 8;
    let data: Vec<f32> = (0..vocab * dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
        .collect();
    let result = enforce_embedding_validation("test_emb", &data, vocab, dim);
    assert!(result.is_ok());
}

#[test]
fn test_enforce_embedding_validation_fails() {
    let data = vec![0.0f32; 100]; // All zeros
    let result = enforce_embedding_validation("bad_emb", &data, 10, 10);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("bad_emb"));
}

#[test]
fn test_enforce_weight_validation_passes() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01) + 0.01).collect();
    let result = enforce_weight_validation("test_w", &data, 10, 10);
    assert!(result.is_ok());
}

#[test]
fn test_enforce_weight_validation_fails() {
    let data = vec![0.0f32; 100]; // All zeros -> density failure
    let result = enforce_weight_validation("bad_w", &data, 10, 10);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("bad_w"));
}

// ============================================================================
// ValidationResult
// ============================================================================

#[test]
fn test_validation_result_debug() {
    let result = ValidationResult {
        passed: true,
        stats: TensorStats::compute(&[1.0, 2.0]),
        failures: vec![],
    };
    let debug = format!("{:?}", result);
    assert!(debug.contains("ValidationResult"));
}

// ============================================================================
// ValidatedAprTransformer with optional biases
// ============================================================================

#[test]
fn test_validated_transformer_with_output_norm_bias() {
    let mut t = make_valid_transformer(1);
    t.output_norm_bias = Some(vec![1.0; 16]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid output_norm_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_lm_head_bias() {
    let mut t = make_valid_transformer(1);
    t.lm_head_bias = Some(vec![0.1; 32]); // vocab_size = 32
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid lm_head_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_attn_norm_bias() {
    let mut t = make_valid_transformer(1);
    t.layers[0].attn_norm_bias = Some(vec![0.1; 16]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid attn_norm_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_qkv_bias() {
    let mut t = make_valid_transformer(1);
    let hidden_dim = 16;
    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;
    t.layers[0].qkv_bias = Some(vec![0.1; qkv_out_dim]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid qkv_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_attn_output_bias() {
    let mut t = make_valid_transformer(1);
    t.layers[0].attn_output_bias = Some(vec![0.1; 16]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid attn_output_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_ffn_gate_bias() {
    let mut t = make_valid_transformer(1);
    t.layers[0].ffn_gate_bias = Some(vec![0.1; 64]); // intermediate_dim = 64
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid ffn_gate_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_ffn_up_bias() {
    let mut t = make_valid_transformer(1);
    t.layers[0].ffn_up_bias = Some(vec![0.1; 64]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid ffn_up_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_ffn_down_bias() {
    let mut t = make_valid_transformer(1);
    t.layers[0].ffn_down_bias = Some(vec![0.1; 16]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid ffn_down_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_with_ffn_norm_bias() {
    let mut t = make_valid_transformer(1);
    t.layers[0].ffn_norm_bias = Some(vec![0.1; 16]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "Should accept valid ffn_norm_bias: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_rejects_nan_output_norm_bias() {
    let mut t = make_valid_transformer(1);
    t.output_norm_bias = Some(vec![f32::NAN; 16]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.tensor_name.contains("output_norm_bias"));
}

#[test]
fn test_validated_transformer_rejects_nan_lm_head_bias() {
    let mut t = make_valid_transformer(1);
    t.lm_head_bias = Some(vec![f32::NAN; 32]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.tensor_name.contains("lm_head_bias"));
}

#[test]
fn test_validated_transformer_rejects_nan_ffn_norm_bias() {
    let mut t = make_valid_transformer(1);
    t.layers[0].ffn_norm_bias = Some(vec![f32::NAN; 16]);
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
}

#[test]
fn test_validated_transformer_with_all_biases() {
    let mut t = make_valid_transformer(2);
    let hidden_dim = 16;
    let intermediate_dim = 64;
    let vocab_size = 32;
    let qkv_out_dim = hidden_dim + 2 * hidden_dim; // non-GQA

    t.output_norm_bias = Some(vec![0.1; hidden_dim]);
    t.lm_head_bias = Some(vec![0.1; vocab_size]);

    for layer in &mut t.layers {
        layer.attn_norm_bias = Some(vec![0.1; hidden_dim]);
        layer.qkv_bias = Some(vec![0.1; qkv_out_dim]);
        layer.attn_output_bias = Some(vec![0.1; hidden_dim]);
        layer.ffn_gate_bias = Some(vec![0.1; intermediate_dim]);
        layer.ffn_up_bias = Some(vec![0.1; intermediate_dim]);
        layer.ffn_down_bias = Some(vec![0.1; hidden_dim]);
        layer.ffn_norm_bias = Some(vec![0.1; hidden_dim]);
    }

    let result = ValidatedAprTransformer::validate(t);
    assert!(
        result.is_ok(),
        "All-biases transformer should validate: {:?}",
        result.err()
    );
}

#[test]
fn test_validated_transformer_rejects_wrong_output_norm() {
    let mut t = make_valid_transformer(1);
    t.output_norm_weight = vec![f32::NAN; 16];
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
}

#[test]
fn test_validated_transformer_rejects_wrong_attn_output_weight() {
    let mut t = make_valid_transformer(1);
    let len = t.layers[0].attn_output_weight.len();
    t.layers[0].attn_output_weight = vec![0.0; len]; // all zeros
    let result = ValidatedAprTransformer::validate(t);
    assert!(result.is_err());
}
