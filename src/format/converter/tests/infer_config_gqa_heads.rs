
// ============================================================================
// GQA head dim inference: head_dim=96 and head_dim=80
// ============================================================================

#[test]
fn test_infer_config_gqa_head_dim_64_first() {
    // The function checks head_dims in order [64, 128, 96, 80]
    // For q_dim=768, kv_dim=192: head_dim=64 matches first (768/64=12, 192/64=3)
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![5000, 768]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        dummy_tensor(vec![768, 768]), // q_dim=768
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        dummy_tensor(vec![192, 768]), // kv_dim=192
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    // head_dim=64 first match: 768/64=12 heads, 192/64=3 kv_heads
    assert_eq!(config.num_heads, Some(12));
    assert_eq!(config.num_kv_heads, Some(3));
}

#[test]
fn test_infer_config_no_q_no_k_projections() {
    // No Q/K projections - num_heads and num_kv_heads should be None
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![1000, 128]),
    );
    tensors.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        dummy_tensor(vec![512, 128]),
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    assert!(
        config.num_heads.is_none(),
        "Without Q/K projections, num_heads should be None"
    );
}

#[test]
fn test_infer_config_no_intermediate_size() {
    // No FFN tensors -> intermediate_size should be None
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![1000, 128]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        dummy_tensor(vec![128, 128]),
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    assert!(
        config.intermediate_size.is_none(),
        "Without FFN tensors, intermediate_size should be None"
    );
}

// ============================================================================
// infer_model_config: up_proj naming variant
// ============================================================================

#[test]
fn test_infer_config_up_proj_naming() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![1000, 128]),
    );
    tensors.insert(
        "model.layers.0.mlp.up_proj.weight".to_string(),
        dummy_tensor(vec![512, 128]),
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    assert_eq!(config.intermediate_size, Some(512));
}

// ============================================================================
// infer_model_config: defaults (max_position_embeddings, rope_theta, rms_norm_eps)
// ============================================================================

#[test]
fn test_infer_config_defaults() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        dummy_tensor(vec![1000, 128]),
    );

    let config = infer_model_config_from_tensors(&tensors).expect("config should be Some");
    assert_eq!(config.max_position_embeddings, Some(4096));
    assert_eq!(config.rope_theta, Some(10000.0));
    assert_eq!(config.rms_norm_eps, Some(1e-6));
}

// ============================================================================
// GH-234: lm_head.weight must skip quantization
// ============================================================================

#[test]
fn test_gh234_quantize_tensors_skips_lm_head() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "lm_head.weight".to_string(),
        (vec![0.01, 0.02, 0.03, 0.04], vec![2, 2]),
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
    );

    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int8);
    assert!(result.is_ok());
    let quantized = result.expect("quantize should succeed");
    let quantized = quantized.as_ref();

    // lm_head should be unchanged (F32 preserved) — GH-234
    let lm_head = &quantized["lm_head.weight"].0;
    assert!(
        (lm_head[0] - 0.01).abs() < 1e-6,
        "GH-234: lm_head.weight must NOT be quantized, got {}",
        lm_head[0]
    );
}

#[test]
fn test_gh234_quantize_tensors_skips_output_weight() {
    // GGUF naming: output.weight is the lm_head equivalent
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "output.weight".to_string(),
        (vec![0.01, 0.02, 0.03, 0.04], vec![2, 2]),
    );

    let native = NativeF32Tensors::new(tensors);
    let result = quantize_tensors(&native, &QuantizationType::Int4);
    assert!(result.is_ok());
    let quantized = result.expect("quantize should succeed");
    let lm_head = &quantized.as_ref()["output.weight"].0;
    assert!(
        (lm_head[0] - 0.01).abs() < 1e-6,
        "GH-234: output.weight (GGUF lm_head) must NOT be quantized"
    );
}

// ============================================================================
// GH-235: GPT-2 config.json uses n_embd, not hidden_size
// ============================================================================

#[test]
fn test_gh235_load_config_gpt2_n_embd_field() {
    // Simulate GPT-2 config.json with n_embd instead of hidden_size
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let model_path = temp_dir.path().join("model.safetensors");
    let config_path = temp_dir.path().join("config.json");

    // Write GPT-2 style config.json
    std::fs::write(
        &config_path,
        r#"{
            "model_type": "gpt2",
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "n_positions": 1024,
            "vocab_size": 50257
        }"#,
    )
    .expect("write config");

    let config = import::load_model_config_from_json(&model_path);
    assert!(config.is_some(), "GH-235: Should parse GPT-2 config.json");
    let config = config.expect("config");

    assert_eq!(
        config.hidden_size,
        Some(768),
        "GH-235: n_embd=768 must map to hidden_size=768, not head_dim"
    );
    assert_eq!(config.num_heads, Some(12), "GH-235: n_head must map");
    assert_eq!(config.num_layers, Some(12), "GH-235: n_layer must map");
    assert_eq!(
        config.max_position_embeddings,
        Some(1024),
        "GH-235: n_positions must map"
    );
    assert_eq!(config.vocab_size, Some(50257));
    assert_eq!(
        config.architecture.as_deref(),
        Some("gpt2"),
        "GH-235: model_type should be gpt2"
    );
}

#[test]
fn test_gh235_load_config_gpt2_n_inner_fallback() {
    // GPT-2 without n_inner should default to 4 * n_embd
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let model_path = temp_dir.path().join("model.safetensors");
    let config_path = temp_dir.path().join("config.json");

    std::fs::write(
        &config_path,
        r#"{
            "model_type": "gpt2",
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "vocab_size": 50257
        }"#,
    )
    .expect("write config");

    let config = import::load_model_config_from_json(&model_path);
    assert!(config.is_some());
    let config = config.expect("config");

    assert_eq!(
        config.intermediate_size,
        Some(3072),
        "GH-235: GPT-2 intermediate_size should default to 4 * n_embd = 3072"
    );
}

// ============================================================================
// GH-237: Write-time contract enforcement — Q8/Q4 byte count and density
// ============================================================================

/// Helper: write APR with a single tensor and read it back, returning the tensor entry dtype
/// and raw data length.
fn roundtrip_tensor_dtype(
    add_fn: impl FnOnce(&mut crate::format::v2::AprV2Writer),
    tensor_name: &str,
) -> (crate::format::v2::TensorDType, usize) {
    use crate::format::v2::{AprV2Metadata, AprV2Reader, AprV2Writer};

    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);
    add_fn(&mut writer);

    let bytes = writer.write().expect("write APR");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read APR");
    let entry = reader
        .get_tensor(tensor_name)
        .expect("tensor not found in roundtrip");
    let data = reader
        .get_tensor_data(tensor_name)
        .expect("tensor data not found");
    (entry.dtype, data.len())
}

#[test]
fn test_gh237_q8_tensor_byte_count_contract() {
    use crate::format::v2::TensorDType;

    // 2048 elements: should produce 4 (scale) + 2048 (i8 values) = 2052 bytes
    let data: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) / 100.0).collect();
    let (dtype, data_len) = roundtrip_tensor_dtype(
        |w| w.add_q8_tensor("test.weight", vec![64, 32], &data),
        "test.weight",
    );
    assert_eq!(dtype, TensorDType::Q8);
    assert_eq!(
        data_len,
        4 + 2048,
        "Q8 packed size: 4-byte scale + N i8 values"
    );
}

#[test]
fn test_gh237_q4_tensor_byte_count_contract() {
    use crate::format::v2::TensorDType;

    // 2048 elements: (2048 + 31) / 32 = 64 blocks, 64 * 18 = 1152 bytes
    let data: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) / 100.0).collect();
    let (dtype, data_len) = roundtrip_tensor_dtype(
        |w| w.add_q4_tensor("test.weight", vec![64, 32], &data),
        "test.weight",
    );
    assert_eq!(dtype, TensorDType::Q4);
    let expected_blocks = (2048 + 31) / 32;
    assert_eq!(
        data_len,
        expected_blocks * 18,
        "Q4 packed size: num_blocks * 18 bytes"
    );
}

#[test]
fn test_gh237_q8_empty_tensor_allowed() {
    use crate::format::v2::TensorDType;

    let (dtype, data_len) = roundtrip_tensor_dtype(
        |w| w.add_q8_tensor("empty.weight", vec![0], &[]),
        "empty.weight",
    );
    assert_eq!(dtype, TensorDType::Q8);
    assert_eq!(data_len, 0);
}

#[test]
fn test_gh237_q4_empty_tensor_allowed() {
    use crate::format::v2::TensorDType;

    let (dtype, data_len) = roundtrip_tensor_dtype(
        |w| w.add_q4_tensor("empty.weight", vec![0], &[]),
        "empty.weight",
    );
    assert_eq!(dtype, TensorDType::Q4);
    assert_eq!(data_len, 0);
}

#[test]
#[should_panic(expected = "Q8 DENSITY VIOLATION")]
fn test_gh237_q8_density_violation_detected() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};

    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);

    // Data that's >99% zeros will trigger the density assertion (catches packing bugs).
    // With 2048 elements (≥1024 threshold), we need >2028 zeros.
    // Set only 5 values to non-zero (~99.8% zeros).
    let mut data = vec![0.0f32; 2048];
    for (i, val) in data.iter_mut().enumerate().take(5) {
        *val = 1.0 + i as f32;
    }
    writer.add_q8_tensor("bad.weight", vec![64, 32], &data);
}

#[test]
fn test_gh237_convert_save_uses_real_q8_packing() {
    use crate::format::v2::TensorDType;

    // GH-237: Verify that add_tensor_with_quantization dispatches to Q8 packing
    // instead of always writing F32.
    let data: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) / 100.0).collect();
    let (dtype, data_len) = roundtrip_tensor_dtype(
        |w| {
            super::super::add_tensor_with_quantization(
                w,
                "model.layers.0.self_attn.q_proj.weight",
                &[64, 32],
                &data,
                Some(QuantizationType::Int8),
            );
        },
        "model.layers.0.self_attn.q_proj.weight",
    );

    // Must be Q8, not F32 (the old broken behavior)
    assert_eq!(
        dtype,
        TensorDType::Q8,
        "GH-237: Tensor must be packed as Q8, not stored as F32"
    );
    assert_eq!(
        data_len,
        4 + 2048,
        "GH-237: Q8 data must be 4 (scale) + N (i8 values)"
    );
}

#[test]
fn test_gh237_convert_save_uses_real_q4_packing() {
    use crate::format::v2::TensorDType;

    let data: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) / 100.0).collect();
    let (dtype, _data_len) = roundtrip_tensor_dtype(
        |w| {
            super::super::add_tensor_with_quantization(
                w,
                "model.layers.0.self_attn.q_proj.weight",
                &[64, 32],
                &data,
                Some(QuantizationType::Int4),
            );
        },
        "model.layers.0.self_attn.q_proj.weight",
    );

    assert_eq!(
        dtype,
        TensorDType::Q4,
        "GH-237: Tensor must be packed as Q4, not stored as F32"
    );
}

#[test]
fn test_gh237_convert_save_skips_quant_for_embeddings() {
    use crate::format::v2::TensorDType;

    let data: Vec<f32> = (0..2048).map(|i| (i as f32) / 1000.0).collect();
    let (dtype, _data_len) = roundtrip_tensor_dtype(
        |w| {
            super::super::add_tensor_with_quantization(
                w,
                "model.embed_tokens.weight",
                &[64, 32],
                &data,
                Some(QuantizationType::Int8),
            );
        },
        "model.embed_tokens.weight",
    );

    assert_eq!(
        dtype,
        TensorDType::F32,
        "GH-237: Embedding tensors must remain F32 when quantization is requested"
    );
}

#[test]
fn test_gh237_convert_save_skips_quant_for_lm_head() {
    use crate::format::v2::TensorDType;

    let data: Vec<f32> = (0..2048).map(|i| (i as f32) / 1000.0).collect();
    let (dtype, _data_len) = roundtrip_tensor_dtype(
        |w| {
            super::super::add_tensor_with_quantization(
                w,
                "lm_head.weight",
                &[64, 32],
                &data,
                Some(QuantizationType::Int4),
            );
        },
        "lm_head.weight",
    );

    assert_eq!(
        dtype,
        TensorDType::F32,
        "GH-237/GH-234: lm_head.weight must remain F32 when quantization is requested"
    );
}
