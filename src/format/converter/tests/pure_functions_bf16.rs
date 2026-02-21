
// ============================================================================
// dequantize_bf16_to_f32: BF16 byte stream dequantization
// ============================================================================

#[test]
fn test_dequantize_bf16_to_f32_zero() {
    let bytes = [0x00, 0x00];
    let result = dequantize_bf16_to_f32(&bytes, 1);
    assert_eq!(result[0], 0.0);
}

#[test]
fn test_dequantize_bf16_to_f32_one() {
    // 1.0 in BF16 = 0x3F80 (upper 16 bits of f32 1.0 = 0x3F800000)
    // LE: 0x80, 0x3F
    let bytes = [0x80, 0x3F];
    let result = dequantize_bf16_to_f32(&bytes, 1);
    assert_eq!(result[0], 1.0);
}

#[test]
fn test_dequantize_bf16_to_f32_negative_one() {
    // -1.0 in BF16 = 0xBF80, LE: 0x80, 0xBF
    let bytes = [0x80, 0xBF];
    let result = dequantize_bf16_to_f32(&bytes, 1);
    assert_eq!(result[0], -1.0);
}

#[test]
fn test_dequantize_bf16_preserves_sign_and_exponent() {
    // BF16 has same exponent range as f32, just truncated mantissa
    // 2.0 in f32 = 0x40000000, BF16 = 0x4000, LE: 0x00, 0x40
    let bytes = [0x00, 0x40];
    let result = dequantize_bf16_to_f32(&bytes, 1);
    assert_eq!(result[0], 2.0);
}

// ============================================================================
// dequantize_q8_0_to_f32: Q8_0 block dequantization
// ============================================================================

#[test]
fn test_dequantize_q8_0_zero_scale() {
    // f16 zero scale = 0x0000, then 32 quant values should all produce 0
    let mut bytes = vec![0x00, 0x00]; // f16 scale = 0
    bytes.extend_from_slice(&[42u8; 32]); // 32 non-zero quant values
    let result = dequantize_q8_0_to_f32(&bytes, 32);
    assert_eq!(result.len(), 32);
    for &v in &result {
        assert_eq!(v, 0.0, "zero scale should produce all zeros");
    }
}

#[test]
fn test_dequantize_q8_0_nan_scale_clamped() {
    // NaN f16 scale should be clamped to 0 (GH-186 fix)
    let mut bytes = vec![0x00, 0x7E]; // f16 NaN
    bytes.extend_from_slice(&[1u8; 32]);
    let result = dequantize_q8_0_to_f32(&bytes, 32);
    for &v in &result {
        assert_eq!(v, 0.0, "NaN scale should clamp to 0");
    }
}

#[test]
fn test_dequantize_q8_0_inf_scale_clamped() {
    // +inf f16 scale should be clamped to 0 (GH-186 fix)
    let mut bytes = vec![0x00, 0x7C]; // f16 +inf
    bytes.extend_from_slice(&[1u8; 32]);
    let result = dequantize_q8_0_to_f32(&bytes, 32);
    for &v in &result {
        assert_eq!(v, 0.0, "inf scale should clamp to 0");
    }
}

#[test]
fn test_dequantize_q8_0_subnormal_scale_clamped() {
    // Subnormal f16 scale below F16_MIN_NORMAL should be clamped to 0
    let mut bytes = vec![0x01, 0x00]; // f16 smallest subnormal
    bytes.extend_from_slice(&[127u8; 32]);
    let result = dequantize_q8_0_to_f32(&bytes, 32);
    // The subnormal f16 value 0x0001 maps to ~5.96e-8 which is below F16_MIN_NORMAL (~6.1e-5)
    for &v in &result {
        assert_eq!(v, 0.0, "subnormal scale should clamp to 0");
    }
}

#[test]
fn test_dequantize_q8_0_partial_last_block() {
    // Request fewer elements than a full block
    let mut bytes = vec![0x00, 0x3C]; // f16 scale = 1.0
    bytes.extend_from_slice(&[1u8; 32]); // 32 int8 values of 1
    let result = dequantize_q8_0_to_f32(&bytes, 16);
    assert_eq!(result.len(), 16, "should respect num_elements limit");
}

#[test]
fn test_dequantize_q8_0_multiple_blocks() {
    // Two complete blocks
    let mut bytes = Vec::new();
    // Block 0: scale=1.0, all quants = 2
    bytes.extend_from_slice(&[0x00, 0x3C]); // f16 1.0
    bytes.extend_from_slice(&[2u8; 32]);
    // Block 1: scale=2.0, all quants = 3
    bytes.extend_from_slice(&[0x00, 0x40]); // f16 2.0
    bytes.extend_from_slice(&[3u8; 32]);

    let result = dequantize_q8_0_to_f32(&bytes, 64);
    assert_eq!(result.len(), 64);
    // Block 0: 2 * 1.0 = 2.0
    assert!((result[0] - 2.0).abs() < 0.01);
    // Block 1: 3 * 2.0 = 6.0
    assert!((result[32] - 6.0).abs() < 0.01);
}

#[test]
fn test_dequantize_q8_0_signed_quant_values() {
    // Test that quant bytes are treated as signed int8
    let mut bytes = vec![0x00, 0x3C]; // f16 scale = 1.0
                                      // Byte 0xFF as i8 = -1, byte 0x80 as i8 = -128
    let mut quants = vec![0u8; 32];
    quants[0] = 0xFF; // -1 as i8
    quants[1] = 0x80; // -128 as i8
    quants[2] = 0x7F; // 127 as i8
    bytes.extend_from_slice(&quants);

    let result = dequantize_q8_0_to_f32(&bytes, 32);
    // -1 * 1.0 = -1.0
    assert!(
        (result[0] + 1.0).abs() < 0.01,
        "0xFF should be -1: got {}",
        result[0]
    );
    // -128 * 1.0 = -128.0
    assert!(
        (result[1] + 128.0).abs() < 0.01,
        "0x80 should be -128: got {}",
        result[1]
    );
    // 127 * 1.0 = 127.0
    assert!(
        (result[2] - 127.0).abs() < 0.01,
        "0x7F should be 127: got {}",
        result[2]
    );
}

// ============================================================================
// quantize_tensors: Embedding skip logic (BUG-EXPORT-004)
// ============================================================================

#[test]
fn test_quantize_tensors_skips_embeddings() {
    let mut tensors = BTreeMap::new();
    let embed_data = vec![0.1_f32; 256];
    // Use varied data that definitely changes under INT4 quantization
    // INT4 has only 16 levels, so fine-grained values will be rounded
    let weight_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        (embed_data.clone(), vec![16, 16]),
    );
    tensors.insert(
        "layers.0.attn_q.weight".to_string(),
        (weight_data.clone(), vec![16, 16]),
    );

    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int4).expect("quantize");
    let result = result.as_ref();

    // Embedding should be unchanged (skipped)
    let (embed_q, _) = result.get("model.embed_tokens.weight").expect("embed");
    assert_eq!(
        embed_q, &embed_data,
        "embedding should be kept in original F32"
    );

    // Weight should be quantized (different from original due to INT4 rounding)
    let (weight_q, _) = result.get("layers.0.attn_q.weight").expect("weight");
    assert_ne!(
        weight_q, &weight_data,
        "non-embedding weight should be quantized"
    );
}

#[test]
fn test_quantize_tensors_skips_all_embedding_variants() {
    let mut tensors = BTreeMap::new();
    let data = vec![0.05_f32; 64];
    // All four embedding name patterns from the code
    tensors.insert(
        "embed_tokens.weight".to_string(),
        (data.clone(), vec![8, 8]),
    );
    tensors.insert("token_embd.weight".to_string(), (data.clone(), vec![8, 8]));
    tensors.insert("wte.weight".to_string(), (data.clone(), vec![8, 8]));
    tensors.insert(
        "word_embeddings.weight".to_string(),
        (data.clone(), vec![8, 8]),
    );

    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int8).expect("quantize");
    let result = result.as_ref();

    for name in [
        "embed_tokens.weight",
        "token_embd.weight",
        "wte.weight",
        "word_embeddings.weight",
    ] {
        let (q, _) = result.get(name).expect(name);
        assert_eq!(q, &data, "embedding '{}' should be skipped", name);
    }
}

// ============================================================================
// ConvertOptions: Default validation
// ============================================================================

#[test]
fn test_convert_options_default_has_validate_true() {
    let opts = ConvertOptions::default();
    assert!(opts.validate, "default should enable validation");
    assert!(opts.quantize.is_none());
    assert!(opts.compress.is_none());
}

// ============================================================================
// AprConverter builder pattern
// ============================================================================

#[test]
fn test_apr_converter_default() {
    let converter = AprConverter::default();
    let debug = format!("{:?}", converter);
    assert!(
        debug.contains("Auto"),
        "default architecture should be Auto"
    );
}

#[test]
fn test_apr_converter_builder_chain() {
    let converter = AprConverter::new()
        .architecture(Architecture::Qwen2)
        .validate(ValidationConfig::Strict)
        .quantize(QuantizationType::Int8)
        .compress(Compression::Lz4);
    let debug = format!("{:?}", converter);
    assert!(debug.contains("Qwen2"));
    assert!(debug.contains("Int8"));
}

// ============================================================================
// DOUBLE-QUANT-001: Compile-time double quantization prevention
// ============================================================================

#[test]
fn test_double_quant_001_native_tensors_accepted() {
    // NativeF32Tensors should be accepted by quantize_tensors
    let mut map = BTreeMap::new();
    map.insert("w".to_string(), (vec![1.0_f32, 2.0, 3.0], vec![3]));
    let native = NativeF32Tensors::new(map);
    let result = quantize_tensors(&native, &QuantizationType::Int8);
    assert!(result.is_ok(), "native tensors should be quantizable");
}

#[test]
fn test_double_quant_001_dequantized_type_exists() {
    // Verify DequantizedTensors can be constructed (but NOT passed to quantize_tensors)
    let map = BTreeMap::new();
    let deq = DequantizedTensors::new(map, QuantizationType::Q4K);
    assert_eq!(deq.original_quant, QuantizationType::Q4K);
    // The following would NOT compile — compile-time enforcement:
    // quantize_tensors(&deq, &QuantizationType::Int8);
}

#[test]
fn test_double_quant_001_provenance_native_variant() {
    let mut map = BTreeMap::new();
    map.insert("w".to_string(), (vec![1.0_f32], vec![1]));
    let native = NativeF32Tensors::new(map);
    let prov = TensorProvenance::Native(native);
    assert_eq!(prov.as_map().len(), 1);
    let inner = prov.into_map();
    assert!(inner.contains_key("w"));
}

#[test]
fn test_double_quant_001_provenance_dequantized_variant() {
    let mut map = BTreeMap::new();
    map.insert("w".to_string(), (vec![1.0_f32], vec![1]));
    let deq = DequantizedTensors::new(map, QuantizationType::Q4K);
    let prov = TensorProvenance::Dequantized(deq);
    assert_eq!(prov.as_map().len(), 1);
    let inner = prov.into_map();
    assert!(inner.contains_key("w"));
}

#[test]
fn test_double_quant_001_native_into_inner_roundtrip() {
    let mut map = BTreeMap::new();
    map.insert("t".to_string(), (vec![42.0_f32], vec![1]));
    let native = NativeF32Tensors::new(map);
    let inner = native.into_inner();
    assert_eq!(inner["t"].0, vec![42.0_f32]);
}

#[test]
fn test_double_quant_001_dequantized_into_inner_roundtrip() {
    let mut map = BTreeMap::new();
    map.insert("t".to_string(), (vec![42.0_f32], vec![1]));
    let deq = DequantizedTensors::new(map, QuantizationType::Fp16);
    assert_eq!(deq.original_quant, QuantizationType::Fp16);
    let inner = deq.into_inner();
    assert_eq!(inner["t"].0, vec![42.0_f32]);
}

// ============================================================================

#[test]
fn test_apr_converter_convert_without_source_fails() {
    let result = AprConverter::new().convert();
    assert!(result.is_err(), "convert without source should fail");
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("No source"), "should say no source: {}", msg);
}

// ============================================================================
// build_q4k_metadata: Coverage tests (impact 32.9)
// ============================================================================

#[test]
fn test_build_q4k_metadata_with_full_config() {
    let cfg = InferredQ4kConfig {
        hidden_size: Some(896),
        num_layers: Some(24),
        num_kv_heads: Some(2),
        vocab_size: Some(151936),
        intermediate_size: Some(4864),
        num_heads: Some(14),
    };

    let meta = build_q4k_metadata(&cfg, 500_000_000);

    assert_eq!(meta.model_type, "qwen2");
    assert_eq!(meta.name, Some("Quantized Model".to_string()));
    assert_eq!(
        meta.description,
        Some("Q4K quantized from SafeTensors".to_string())
    );
    assert_eq!(meta.original_format, Some("safetensors".to_string()));
    assert_eq!(meta.architecture, Some("qwen2".to_string()));
    assert_eq!(meta.hidden_size, Some(896));
    assert_eq!(meta.num_layers, Some(24));
    assert_eq!(meta.num_heads, Some(14));
    assert_eq!(meta.num_kv_heads, Some(2));
    assert_eq!(meta.vocab_size, Some(151936));
    assert_eq!(meta.intermediate_size, Some(4864));
    assert_eq!(meta.max_position_embeddings, Some(32768));
    assert_eq!(meta.rope_theta, Some(1000000.0));
    assert_eq!(meta.rope_type, Some(2));
    assert_eq!(meta.rms_norm_eps, Some(1e-6));
    assert_eq!(meta.param_count, 500_000_000);

    // Verify quantization metadata
    let quant = meta
        .quantization
        .as_ref()
        .expect("quantization should be set");
    assert_eq!(quant.quant_type, "q4_k");
    assert_eq!(quant.bits, 4);
    assert_eq!(quant.block_size, Some(256));
    assert!(!quant.symmetric);
}

#[test]
fn test_build_q4k_metadata_with_none_fields() {
    let cfg = InferredQ4kConfig {
        hidden_size: None,
        num_layers: None,
        num_kv_heads: None,
        vocab_size: None,
        intermediate_size: None,
        num_heads: None,
    };

    let meta = build_q4k_metadata(&cfg, 0);

    assert_eq!(meta.model_type, "qwen2");
    assert_eq!(meta.hidden_size, None);
    assert_eq!(meta.num_layers, None);
    assert_eq!(meta.num_heads, None);
    assert_eq!(meta.num_kv_heads, None);
    assert_eq!(meta.vocab_size, None);
    assert_eq!(meta.intermediate_size, None);
    // Defaults should still be set
    assert_eq!(meta.max_position_embeddings, Some(32768));
    assert_eq!(meta.rope_theta, Some(1000000.0));
    assert_eq!(meta.rope_type, Some(2));
    assert_eq!(meta.rms_norm_eps, Some(1e-6));
    assert_eq!(meta.param_count, 0);
    assert_eq!(meta.version, Some("1.0.0".to_string()));
}

#[test]
fn test_build_q4k_metadata_serializes_to_valid_json() {
    let cfg = InferredQ4kConfig {
        hidden_size: Some(128),
        num_layers: Some(2),
        num_kv_heads: Some(1),
        vocab_size: Some(256),
        intermediate_size: Some(512),
        num_heads: Some(2),
    };

    let meta = build_q4k_metadata(&cfg, 100_000);
    let json = meta.to_json_pretty().expect("should serialize to JSON");
    assert!(json.contains("qwen2"), "JSON should contain model_type");
    assert!(json.contains("Quantized Model"), "JSON should contain name");
    assert!(json.contains("q4_k"), "JSON should contain quant_type");
}
