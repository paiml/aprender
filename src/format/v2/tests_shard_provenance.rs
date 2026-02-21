use super::*;

#[test]
fn test_shard_info_creation() {
    let info = ShardInfo {
        filename: "shard.apr".to_string(),
        index: 0,
        size: 1024,
        tensors: vec!["a".to_string(), "b".to_string()],
    };
    assert_eq!(info.filename, "shard.apr");
    assert_eq!(info.tensors.len(), 2);
}

/// DD6: Model provenance must be tracked in APR metadata
/// Falsification: If source/origin is lost after conversion, provenance tracking fails
#[test]
fn test_dd6_provenance_tracked() {
    let mut metadata = AprV2Metadata::new("test_model");

    // Set provenance information
    metadata.source = Some("hf://openai/whisper-tiny".to_string());
    metadata.original_format = Some("safetensors".to_string());

    // Verify provenance is preserved in serialization
    let json = metadata.to_json().expect("serialize");
    let parsed: AprV2Metadata = serde_json::from_slice(&json).expect("deserialize");

    assert_eq!(
        parsed.source,
        Some("hf://openai/whisper-tiny".to_string()),
        "DD6 FALSIFIED: Source provenance lost after serialization"
    );
    assert_eq!(
        parsed.original_format,
        Some("safetensors".to_string()),
        "DD6 FALSIFIED: Original format lost after serialization"
    );
}

/// DD6b: Verify provenance survives full APR write/read cycle
#[test]
fn test_dd6_provenance_roundtrip() {
    let mut metadata = AprV2Metadata::new("whisper");
    metadata.source = Some("local:///models/whisper-tiny.safetensors".to_string());
    metadata.original_format = Some("safetensors".to_string());
    metadata.author = Some("OpenAI".to_string());

    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("test", vec![4], &[1.0, 2.0, 3.0, 4.0]);

    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let read_meta = reader.metadata();
    assert!(
        read_meta.source.is_some(),
        "DD6 FALSIFIED: Source provenance not preserved in APR file"
    );
    assert_eq!(
        read_meta.source.as_deref(),
        Some("local:///models/whisper-tiny.safetensors"),
        "DD6 FALSIFIED: Source URI corrupted"
    );
    assert_eq!(
        read_meta.original_format.as_deref(),
        Some("safetensors"),
        "DD6 FALSIFIED: Original format corrupted"
    );
}

// ========================================================================
// Chat Template Metadata Tests (CTA-01 to CTA-04)
// Per spec: chat-template-improvement-spec.md Part VIII
// ========================================================================

/// CTA-01: chat_template stored in APR v2 metadata section
#[test]
fn test_cta_01_chat_template_in_metadata() {
    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.chat_template =
        Some("{% for message in messages %}<|im_start|>{{ message.role }}".to_string());

    let json = metadata.to_json().expect("serialize");
    let parsed = AprV2Metadata::from_json(&json).expect("deserialize");

    assert!(
        parsed.chat_template.is_some(),
        "CTA-01 FALSIFIED: chat_template not stored in metadata"
    );
    assert!(parsed
        .chat_template
        .as_ref()
        .expect("chat_template")
        .contains("<|im_start|>"));
}

/// CTA-02: Backward compatibility - APR files without chat_template still load
#[test]
fn test_cta_02_backward_compatibility() {
    // JSON without chat_template field (old format)
    let json = r#"{"model_type":"qwen2","name":"test","param_count":500000000}"#;

    let parsed = AprV2Metadata::from_json(json.as_bytes()).expect("deserialize");
    assert_eq!(parsed.model_type, "qwen2");
    assert!(
        parsed.chat_template.is_none(),
        "CTA-02: Missing chat_template should be None, not error"
    );
}

/// CTA-03: chat_format field indicates detected format
#[test]
fn test_cta_03_chat_format_field() {
    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.chat_format = Some("chatml".to_string());

    let json = metadata.to_json().expect("serialize");
    let parsed = AprV2Metadata::from_json(&json).expect("deserialize");

    assert_eq!(
        parsed.chat_format.as_deref(),
        Some("chatml"),
        "CTA-03 FALSIFIED: chat_format not preserved"
    );
}

/// CTA-04: Special tokens stored in special_tokens object
#[test]
fn test_cta_04_special_tokens_in_metadata() {
    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.special_tokens = Some(ChatSpecialTokens {
        bos_token: Some("<|endoftext|>".to_string()),
        eos_token: Some("<|im_end|>".to_string()),
        im_start_token: Some("<|im_start|>".to_string()),
        im_end_token: Some("<|im_end|>".to_string()),
        ..Default::default()
    });

    let json = metadata.to_json().expect("serialize");
    let parsed = AprV2Metadata::from_json(&json).expect("deserialize");

    let tokens = parsed
        .special_tokens
        .expect("CTA-04 FALSIFIED: special_tokens not stored");
    assert_eq!(tokens.im_start_token.as_deref(), Some("<|im_start|>"));
    assert_eq!(tokens.im_end_token.as_deref(), Some("<|im_end|>"));
}

/// CTA-05: Chat template survives full APR write/read cycle
#[test]
fn test_cta_05_chat_template_roundtrip() {
    let template = "{% for m in messages %}{{ m.content }}{% endfor %}";

    let mut metadata = AprV2Metadata::new("tinyllama");
    metadata.chat_template = Some(template.to_string());
    metadata.chat_format = Some("llama2".to_string());
    metadata.special_tokens = Some(ChatSpecialTokens {
        bos_token: Some("<s>".to_string()),
        eos_token: Some("</s>".to_string()),
        ..Default::default()
    });

    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("test", vec![4], &[1.0, 2.0, 3.0, 4.0]);

    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let read_meta = reader.metadata();
    assert_eq!(
        read_meta.chat_template.as_deref(),
        Some(template),
        "CTA-05 FALSIFIED: chat_template not preserved in APR file"
    );
    assert_eq!(
        read_meta.chat_format.as_deref(),
        Some("llama2"),
        "CTA-05 FALSIFIED: chat_format not preserved"
    );
    assert!(
        read_meta.special_tokens.is_some(),
        "CTA-05 FALSIFIED: special_tokens not preserved"
    );
}

/// ChatSpecialTokens default is empty
#[test]
fn test_chat_special_tokens_default() {
    let tokens = ChatSpecialTokens::default();
    assert!(tokens.bos_token.is_none());
    assert!(tokens.eos_token.is_none());
    assert!(tokens.im_start_token.is_none());
    assert!(tokens.im_end_token.is_none());
}

// ========================================================================
// Quantization Tests - True Packed Storage (APR-QUANT-001)
// ========================================================================

/// F16 tensor roundtrip - 2x compression
#[test]
fn test_f16_tensor_roundtrip() {
    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);

    let original = vec![1.0f32, -2.5, 3.14159, 0.0, 65504.0, -65504.0];
    writer.add_f16_tensor("weights", vec![6], &original);

    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let entry = reader.get_tensor("weights").expect("tensor exists");
    assert_eq!(entry.dtype, TensorDType::F16);
    // F16 = 2 bytes per element
    assert_eq!(entry.size, 12);

    let recovered = reader.get_tensor_as_f32("weights").expect("dequantize");
    assert_eq!(recovered.len(), 6);

    // F16 has ~3 decimal digits precision
    for (orig, rec) in original.iter().zip(recovered.iter()) {
        let diff = (orig - rec).abs();
        let rel_err = if *orig != 0.0 {
            diff / orig.abs()
        } else {
            diff
        };
        assert!(
            rel_err < 0.01,
            "F16 precision loss too high: {orig} -> {rec}"
        );
    }
}

/// Q8 tensor roundtrip - 4x compression
#[test]
fn test_q8_tensor_roundtrip() {
    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);

    let original: Vec<f32> = (-64..64).map(|i| i as f32 * 0.1).collect();
    writer.add_q8_tensor("weights", vec![128], &original);

    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let entry = reader.get_tensor("weights").expect("tensor exists");
    assert_eq!(entry.dtype, TensorDType::Q8);
    // Q8 = 4 bytes scale + 1 byte per element = 132 bytes
    assert_eq!(entry.size, 132);

    let recovered = reader.get_tensor_as_f32("weights").expect("dequantize");
    assert_eq!(recovered.len(), 128);

    // Q8 has ~7 bit precision
    for (orig, rec) in original.iter().zip(recovered.iter()) {
        let diff = (orig - rec).abs();
        let max_val = original.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel_err = diff / max_val;
        assert!(
            rel_err < 0.02,
            "Q8 precision loss too high: {orig} -> {rec}"
        );
    }
}

/// Q4 tensor roundtrip - 7x compression
#[test]
fn test_q4_tensor_roundtrip() {
    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);

    // 64 values = 2 blocks of 32
    let original: Vec<f32> = (-32..32).map(|i| i as f32 * 0.25).collect();
    writer.add_q4_tensor("weights", vec![64], &original);

    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let entry = reader.get_tensor("weights").expect("tensor exists");
    assert_eq!(entry.dtype, TensorDType::Q4);
    // Q4 = 2 blocks × 18 bytes/block = 36 bytes
    assert_eq!(entry.size, 36);

    let recovered = reader.get_tensor_as_f32("weights").expect("dequantize");
    assert_eq!(recovered.len(), 64);

    // Q4 has ~4 bit precision (16 levels)
    for (orig, rec) in original.iter().zip(recovered.iter()) {
        let diff = (orig - rec).abs();
        // Q4 can have up to ~15% error per value due to only 16 quantization levels
        assert!(
            diff < 1.5,
            "Q4 error too high: {orig} -> {rec} (diff={diff})"
        );
    }
}

/// Verify F16 produces smaller files than F32
#[test]
fn test_f16_compression_ratio() {
    let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();

    // F32 version
    let mut writer_f32 = AprV2Writer::new(AprV2Metadata::new("f32"));
    writer_f32.add_f32_tensor("w", vec![1000], &data);
    let bytes_f32 = writer_f32.write().expect("write f32");

    // F16 version
    let mut writer_f16 = AprV2Writer::new(AprV2Metadata::new("f16"));
    writer_f16.add_f16_tensor("w", vec![1000], &data);
    let bytes_f16 = writer_f16.write().expect("write f16");

    // F16 should be ~50% the size of F32 for tensor data
    let ratio = bytes_f16.len() as f32 / bytes_f32.len() as f32;
    assert!(
        ratio < 0.6,
        "F16 should be <60% of F32 size, got {ratio:.2}"
    );
}

/// Verify Q4 produces much smaller files than F32
#[test]
fn test_q4_compression_ratio() {
    let data: Vec<f32> = (0..1024).map(|i| (i % 16) as f32 - 8.0).collect();

    // F32 version
    let mut writer_f32 = AprV2Writer::new(AprV2Metadata::new("f32"));
    writer_f32.add_f32_tensor("w", vec![1024], &data);
    let bytes_f32 = writer_f32.write().expect("write f32");

    // Q4 version
    let mut writer_q4 = AprV2Writer::new(AprV2Metadata::new("q4"));
    writer_q4.add_q4_tensor("w", vec![1024], &data);
    let bytes_q4 = writer_q4.write().expect("write q4");

    // Q4 should be ~15-20% the size of F32 (32 blocks × 18 bytes = 576 vs 4096)
    // Actual ratio depends on metadata overhead for small tensors
    let ratio = bytes_q4.len() as f32 / bytes_f32.len() as f32;
    assert!(
        ratio < 0.30,
        "Q4 should be <30% of F32 size, got {ratio:.2}"
    );
}

/// Test f16 conversion edge cases
#[test]
fn test_f16_edge_cases() {
    // Zero
    assert_eq!(f32_to_f16(0.0), 0);
    assert_eq!(f16_to_f32(0), 0.0);

    // Negative zero
    assert_eq!(f32_to_f16(-0.0) & 0x7FFF, 0);

    // One
    let one_f16 = f32_to_f16(1.0);
    assert!((f16_to_f32(one_f16) - 1.0).abs() < 0.001);

    // Max f16 value (~65504)
    let max_f16 = f32_to_f16(65504.0);
    assert!((f16_to_f32(max_f16) - 65504.0).abs() < 100.0);

    // Infinity
    let inf_f16 = f32_to_f16(f32::INFINITY);
    assert_eq!(inf_f16 & 0x7FFF, 0x7C00);

    // NaN
    let nan_f16 = f32_to_f16(f32::NAN);
    assert!(nan_f16 & 0x7FFF > 0x7C00); // NaN has exponent all 1s and non-zero mantissa
}

/// ReaderRef also supports quantized tensors
#[test]
fn test_reader_ref_quantized() {
    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);

    writer.add_f16_tensor("f16", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    writer.add_q8_tensor("q8", vec![4], &[1.0, 2.0, 3.0, 4.0]);

    let bytes = writer.write().expect("write");
    let reader = AprV2ReaderRef::from_bytes(&bytes).expect("read");

    let f16_data = reader.get_tensor_as_f32("f16").expect("f16");
    assert_eq!(f16_data.len(), 4);
    assert!((f16_data[0] - 1.0).abs() < 0.01);

    let q8_data = reader.get_tensor_as_f32("q8").expect("q8");
    assert_eq!(q8_data.len(), 4);
    assert!((q8_data[0] - 1.0).abs() < 0.1);
}

// ========================================================================
// Pygmy-Based Tests (T-COV-95)
// ========================================================================

#[test]
fn test_pygmy_apr_metadata_full() {
    use crate::format::test_factory::{build_pygmy_apr_with_config, PygmyConfig};

    let config = PygmyConfig::llama_style();
    let data = build_pygmy_apr_with_config(config);
    let reader = AprV2Reader::from_bytes(&data).expect("parse");

    // Check metadata fields
    let metadata = reader.metadata();
    assert_eq!(metadata.architecture, Some("llama".to_string()));
    assert!(metadata.hidden_size.is_some());
    assert!(metadata.vocab_size.is_some());
    assert!(metadata.num_layers.is_some());
}

#[test]
fn test_pygmy_apr_tensor_lookup() {
    use crate::format::test_factory::build_pygmy_apr;

    let data = build_pygmy_apr();
    let reader = AprV2Reader::from_bytes(&data).expect("parse");

    // Test tensor lookup
    let embed = reader.get_tensor("model.embed_tokens.weight");
    assert!(embed.is_some());

    let nonexistent = reader.get_tensor("nonexistent.weight");
    assert!(nonexistent.is_none());
}

#[test]
fn test_pygmy_apr_tensor_data() {
    use crate::format::test_factory::build_pygmy_apr;

    let data = build_pygmy_apr();
    let reader = AprV2Reader::from_bytes(&data).expect("parse");

    // Test getting tensor data
    let tensor_data = reader.get_tensor_data("model.embed_tokens.weight");
    assert!(tensor_data.is_some());

    // F32 tensor should have 4 bytes per element
    let entry = reader.get_tensor("model.embed_tokens.weight").unwrap();
    let expected_bytes = entry.element_count() * 4;
    assert_eq!(tensor_data.unwrap().len(), expected_bytes);
}
