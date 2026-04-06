
#[test]
fn test_generate_config_custom() {
    let config = GenerateConfig {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.95,
        top_k: 50,
        repetition_penalty: 1.1,
        trace: false,
        stop_tokens: vec![],
    };

    assert_eq!(config.max_tokens, 100);
    assert!((config.temperature - 0.7).abs() < 1e-6);
    assert!((config.top_p - 0.95).abs() < 1e-6);
    assert_eq!(config.top_k, 50);
    assert!((config.repetition_penalty - 1.1).abs() < 1e-6);
}

#[test]
fn test_generate_config_clone() {
    let original = GenerateConfig {
        max_tokens: 50,
        temperature: 0.5,
        top_p: 0.8,
        top_k: 40,
        repetition_penalty: 1.2,
        trace: false,
        stop_tokens: vec![],
    };

    let cloned = original.clone();

    assert_eq!(cloned.max_tokens, original.max_tokens);
    assert!((cloned.temperature - original.temperature).abs() < 1e-6);
}

#[test]
fn test_generate_config_debug() {
    let config = GenerateConfig::default();
    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("max_tokens"));
    assert!(debug_str.contains("temperature"));
}

// ============================================================================
// Part 5: AprTransformerConfig Tests
// ============================================================================

#[test]
fn test_transformer_config_default() {
    let config = AprTransformerConfig::default();

    assert_eq!(config.architecture, "unknown");
    assert_eq!(config.hidden_dim, 512);
    assert_eq!(config.num_layers, 6);
    assert_eq!(config.num_heads, 8);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.intermediate_dim, 2048);
    assert_eq!(config.context_length, 2048);
    assert!((config.rope_theta - 10000.0).abs() < 1e-6);
    assert!((config.eps - 1e-5).abs() < 1e-9);
}

#[test]
fn test_transformer_config_custom() {
    let config = AprTransformerConfig {
        architecture: "llama".to_string(),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 128000,
        intermediate_dim: 14336,
        context_length: 8192,
        rope_theta: 500000.0,
        eps: 1e-6,
            eos_token_id: None,
    ..Default::default()
    };

    assert_eq!(config.architecture, "llama");
    assert_eq!(config.hidden_dim, 4096);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_kv_heads, 8); // GQA
}

#[test]
fn test_transformer_config_serialization() {
    let config = create_test_apr_config();

    // Serialize to JSON
    let json = serde_json::to_string(&config).expect("JSON serialization failed");

    // Deserialize
    let recovered: AprTransformerConfig = serde_json::from_str(&json).expect("JSON deserialization failed");

    assert_eq!(config, recovered);
}

#[test]
fn test_transformer_config_clone() {
    let original = create_test_apr_config();
    let cloned = original.clone();

    assert_eq!(original, cloned);
}

#[test]
fn test_transformer_config_debug() {
    let config = create_test_apr_config();
    let debug_str = format!("{:?}", config);

    assert!(debug_str.contains("hidden_dim"));
    assert!(debug_str.contains("num_layers"));
}

// ============================================================================
// Part 6: AprTransformerLayer Tests
// ============================================================================

#[test]
fn test_layer_empty() {
    let layer = AprTransformerLayer::empty(64, 128);

    assert_eq!(layer.attn_norm_weight.len(), 64);
    assert!(layer
        .attn_norm_weight
        .iter()
        .all(|&x| (x - 1.0).abs() < 1e-6));

    assert_eq!(layer.qkv_weight.len(), 64 * 3 * 64);
    assert!(layer.qkv_weight.iter().all(|&x| x == 0.0));

    assert_eq!(layer.attn_output_weight.len(), 64 * 64);
    assert_eq!(layer.ffn_up_weight.len(), 64 * 128);
    assert_eq!(layer.ffn_down_weight.len(), 128 * 64);

    assert!(layer.attn_norm_bias.is_none());
    assert!(layer.qkv_bias.is_none());
    assert!(layer.ffn_gate_weight.is_none());
}

#[test]
fn test_layer_empty_gqa() {
    let hidden_dim = 64;
    let num_heads = 8;
    let num_kv_heads = 2;
    let intermediate_dim = 128;

    let layer =
        AprTransformerLayer::empty_gqa(hidden_dim, num_heads, num_kv_heads, intermediate_dim);

    // For GQA: QKV = Q + K + V = hidden_dim + kv_dim + kv_dim
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    assert_eq!(layer.qkv_weight.len(), hidden_dim * qkv_out_dim);
    assert_eq!(layer.attn_output_weight.len(), hidden_dim * hidden_dim);
}

#[test]
fn test_layer_num_parameters() {
    let layer = AprTransformerLayer::empty(64, 128);
    let params = layer.num_parameters();

    // attn_norm (64) + qkv (64*192) + attn_out (64*64) + ffn_up (64*128) + ffn_down (128*64)
    let expected = 64 + (64 * 3 * 64) + (64 * 64) + (64 * 128) + (128 * 64);
    assert_eq!(params, expected);
}

#[test]
fn test_layer_num_parameters_with_optionals() {
    let mut layer = AprTransformerLayer::empty(64, 128);

    // Add optional fields
    layer.attn_norm_bias = Some(vec![0.0; 64]);
    layer.qkv_bias = Some(vec![0.0; 64 * 3]);
    layer.ffn_gate_weight = Some(vec![0.0; 64 * 128]);
    layer.ffn_norm_weight = Some(vec![1.0; 64]);

    let params = layer.num_parameters();

    // Base params + optionals
    let base = 64 + (64 * 3 * 64) + (64 * 64) + (64 * 128) + (128 * 64);
    let optionals = 64 + (64 * 3) + (64 * 128) + 64;
    assert_eq!(params, base + optionals);
}

#[test]
fn test_layer_serialization() {
    let layer = AprTransformerLayer::empty(32, 64);

    // Serialize to JSON
    let json = serde_json::to_string(&layer).expect("JSON serialization failed");

    // Deserialize
    let recovered: AprTransformerLayer = serde_json::from_str(&json).expect("JSON deserialization failed");

    assert_eq!(
        layer.attn_norm_weight.len(),
        recovered.attn_norm_weight.len()
    );
    assert_eq!(layer.qkv_weight.len(), recovered.qkv_weight.len());
}

// ============================================================================
// Part 7: Q4KLayerWeights Tests
// ============================================================================

#[test]
fn test_q4k_layer_weights_default() {
    let weights = Q4KLayerWeights::default();

    assert!(weights.qkv_weight.is_none());
    assert!(weights.attn_q_weight.is_none());
    assert!(weights.attn_k_weight.is_none());
    assert!(weights.attn_v_weight.is_none());
    assert!(weights.attn_v_weight_q6k.is_none());
    assert!(weights.attn_output_weight.is_none());
    assert!(weights.ffn_gate_weight.is_none());
    assert!(weights.ffn_up_weight.is_none());
    assert!(weights.ffn_down_weight.is_none());
    assert!(weights.ffn_down_weight_q6k.is_none());
    assert!(weights.ffn_up_weight_q6k.is_none());
}

#[test]
fn test_q4k_layer_weights_with_data() {
    let weights = Q4KLayerWeights {
        qkv_weight: Some(vec![0u8; 144]), // 1 Q4_K block
        attn_q_weight: Some(vec![0u8; 144]),
        attn_k_weight: Some(vec![0u8; 144]),
        attn_v_weight: Some(vec![0u8; 144]),
        attn_v_weight_q6k: Some(vec![0u8; 210]),
        attn_output_weight: Some(vec![0u8; 144]),
        ffn_gate_weight: Some(vec![0u8; 144]),
        ffn_up_weight: Some(vec![0u8; 144]),
        ffn_down_weight: Some(vec![0u8; 144]),
        ffn_down_weight_q6k: Some(vec![0u8; 210]),
        ffn_up_weight_q6k: Some(vec![0u8; 210]),
    };

    assert!(weights.qkv_weight.is_some());
    assert_eq!(weights.qkv_weight.as_ref().expect("test value should be present").len(), 144);
    assert!(weights.attn_v_weight_q6k.is_some());
    assert_eq!(weights.attn_v_weight_q6k.as_ref().expect("test value should be present").len(), 210);
}

#[test]
fn test_q4k_layer_weights_serialization() {
    let weights = Q4KLayerWeights {
        attn_q_weight: Some(vec![1u8, 2, 3]),
        ..Default::default()
    };

    let json = serde_json::to_string(&weights).expect("JSON serialization failed");
    let recovered: Q4KLayerWeights = serde_json::from_str(&json).expect("JSON deserialization failed");

    assert_eq!(weights.attn_q_weight, recovered.attn_q_weight);
}

#[test]
fn test_q4k_layer_weights_clone() {
    let original = Q4KLayerWeights {
        ffn_up_weight: Some(vec![42u8; 100]),
        ..Default::default()
    };

    let cloned = original.clone();
    assert_eq!(original.ffn_up_weight, cloned.ffn_up_weight);
}

// ============================================================================
// Part 8: Dequantization Tests
// ============================================================================

#[test]
fn test_f16_to_f32_zero() {
    use super::super::dequant::f16_to_f32;

    // Positive zero
    assert_eq!(f16_to_f32(0x0000), 0.0);
    // Negative zero
    assert_eq!(f16_to_f32(0x8000), -0.0);
}

#[test]
fn test_f16_to_f32_one() {
    use super::super::dequant::f16_to_f32;

    // 1.0 in f16 = 0x3C00 (sign=0, exp=15, mant=0)
    let result = f16_to_f32(0x3C00);
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_f16_to_f32_negative_one() {
    use super::super::dequant::f16_to_f32;

    // -1.0 in f16 = 0xBC00 (sign=1, exp=15, mant=0)
    let result = f16_to_f32(0xBC00);
    assert!((result - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_f16_to_f32_infinity() {
    use super::super::dequant::f16_to_f32;

    // +Inf in f16 = 0x7C00 (sign=0, exp=31, mant=0)
    let result = f16_to_f32(0x7C00);
    assert!(result.is_infinite() && result.is_sign_positive());

    // -Inf in f16 = 0xFC00 (sign=1, exp=31, mant=0)
    let result_neg = f16_to_f32(0xFC00);
    assert!(result_neg.is_infinite() && result_neg.is_sign_negative());
}

#[test]
fn test_f16_to_f32_nan() {
    use super::super::dequant::f16_to_f32;

    // NaN in f16 = 0x7C01 (sign=0, exp=31, mant!=0)
    let result = f16_to_f32(0x7C01);
    assert!(result.is_nan());
}

#[test]
fn test_f16_to_f32_subnormal() {
    use super::super::dequant::f16_to_f32;

    // Smallest positive subnormal: exp=0, mant=1
    let result = f16_to_f32(0x0001);
    assert!(result > 0.0);
    assert!(result < 1e-4); // Very small number
}

#[test]
fn test_f16_to_f32_normal_range() {
    use super::super::dequant::f16_to_f32;

    // 2.0 in f16 = 0x4000 (sign=0, exp=16, mant=0)
    let result = f16_to_f32(0x4000);
    assert!((result - 2.0).abs() < 1e-4);

    // 0.5 in f16 = 0x3800 (sign=0, exp=14, mant=0)
    let result_half = f16_to_f32(0x3800);
    assert!((result_half - 0.5).abs() < 1e-4);
}

#[test]
fn test_extract_scale_min_first_4_blocks() {
    use super::super::dequant::extract_scale_min_apr;

    // 12-byte scales array
    let scales = [
        0b00111111u8,
        0b00111110,
        0b00111101,
        0b00111100, // first 4 bytes (low 6 bits)
        0b00101010,
        0b00101011,
        0b00101100,
        0b00101101, // second 4 bytes (min values)
        0,
        0,
        0,
        0, // last 4 bytes (high bits)
    ];

    // Block 0: scale = scales[0] & 63 = 63, min = scales[4] & 63 = 42
    let (scale, min) = extract_scale_min_apr(&scales, 0);
    assert_eq!(scale, 63.0);
    assert_eq!(min, 42.0);

    // Block 1
    let (scale, min) = extract_scale_min_apr(&scales, 1);
    assert_eq!(scale, 62.0);
    assert_eq!(min, 43.0);
}

#[test]
fn test_extract_scale_min_last_4_blocks() {
    use super::super::dequant::extract_scale_min_apr;

    // Construct scales for blocks 4-7 (packed layout)
    let scales = [
        0b11_000001u8,
        0b11_000010,
        0b11_000011,
        0b11_000100, // first 4 (high bits used for 4-7)
        0b11_000101,
        0b11_000110,
        0b11_000111,
        0b11_001000, // second 4 (mins for 0-3)
        0x15,
        0x26,
        0x37,
        0x48, // last 4 (scale/min for 4-7 low bits)
    ];

    // Block 4: uses packed layout
    // d = (scales[8] & 0x0F) | ((scales[0] >> 6) << 4) = (0x15 & 0x0F) | (0b11 << 4) = 5 | 48 = 53
    let (scale, _min) = extract_scale_min_apr(&scales, 4);
    assert_eq!(scale, 53.0);
}

#[test]
fn test_dequantize_q4_k_empty() {
    use super::super::dequant::dequantize_q4_k_apr;

    let result = dequantize_q4_k_apr(&[], 0);
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q4_k_insufficient_data() {
    use super::super::dequant::dequantize_q4_k_apr;

    // Request 256 elements but provide only 10 bytes (need 144)
    let data = vec![0u8; 10];
    let result = dequantize_q4_k_apr(&data, 256);

    // Should return zeros
    assert_eq!(result.len(), 256);
    assert!(result.iter().all(|&x| x == 0.0));
}
