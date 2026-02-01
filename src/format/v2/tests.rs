//\! V2 Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

use super::*;

#[test]
fn test_magic_v2() {
    assert_eq!(MAGIC_V2, [0x41, 0x50, 0x52, 0x00]); // "APR\0"
    assert_eq!(&MAGIC_V2, b"APR\0");
}

#[test]
fn test_header_size() {
    assert_eq!(HEADER_SIZE_V2, 64);
    assert!(is_aligned_64(HEADER_SIZE_V2));
}

#[test]
fn test_flags() {
    let flags = AprV2Flags::new()
        .with(AprV2Flags::LZ4_COMPRESSED)
        .with(AprV2Flags::QUANTIZED);

    assert!(flags.is_lz4_compressed());
    assert!(flags.is_quantized());
    assert!(!flags.is_encrypted());
    assert!(!flags.is_sharded());
}

#[test]
fn test_header_new() {
    let header = AprV2Header::new();
    assert_eq!(header.magic, MAGIC_V2);
    assert_eq!(header.version, VERSION_V2);
    assert!(header.is_valid());
}

#[test]
fn test_header_roundtrip() {
    let mut header = AprV2Header::new();
    header.tensor_count = 42;
    header.metadata_size = 1024;
    header.update_checksum();

    let bytes = header.to_bytes();
    assert_eq!(bytes.len(), HEADER_SIZE_V2);

    let parsed = AprV2Header::from_bytes(&bytes).unwrap();
    assert_eq!(parsed.tensor_count, 42);
    assert_eq!(parsed.metadata_size, 1024);
    assert!(parsed.verify_checksum());
}

#[test]
fn test_header_invalid_magic() {
    let bytes = [0xFF; HEADER_SIZE_V2];
    let result = AprV2Header::from_bytes(&bytes);
    assert!(matches!(result, Err(V2FormatError::InvalidMagic(_))));
}

#[test]
fn test_metadata_json_roundtrip() {
    let mut metadata = AprV2Metadata::new("whisper");
    metadata.name = Some("whisper-tiny".to_string());
    metadata.param_count = 39_000_000;

    let json = metadata.to_json().unwrap();
    let parsed = AprV2Metadata::from_json(&json).unwrap();

    assert_eq!(parsed.model_type, "whisper");
    assert_eq!(parsed.name.as_deref(), Some("whisper-tiny"));
    assert_eq!(parsed.param_count, 39_000_000);
}

#[test]
fn test_align_up() {
    assert_eq!(align_up(0, 64), 0);
    assert_eq!(align_up(1, 64), 64);
    assert_eq!(align_up(63, 64), 64);
    assert_eq!(align_up(64, 64), 64);
    assert_eq!(align_up(65, 64), 128);
}

#[test]
fn test_align_64() {
    assert_eq!(align_64(0), 0);
    assert_eq!(align_64(1), 64);
    assert_eq!(align_64(100), 128);
    assert_eq!(align_64(128), 128);
}

#[test]
fn test_is_aligned_64() {
    assert!(is_aligned_64(0));
    assert!(is_aligned_64(64));
    assert!(is_aligned_64(128));
    assert!(!is_aligned_64(1));
    assert!(!is_aligned_64(63));
    assert!(!is_aligned_64(65));
}

#[test]
fn test_tensor_dtype() {
    assert_eq!(TensorDType::F32.bytes_per_element(), 4);
    assert_eq!(TensorDType::F16.bytes_per_element(), 2);
    assert_eq!(TensorDType::F64.bytes_per_element(), 8);
    assert_eq!(TensorDType::I8.bytes_per_element(), 1);
    assert_eq!(TensorDType::Q4.bytes_per_element(), 0);
}

#[test]
fn test_tensor_dtype_name() {
    assert_eq!(TensorDType::F32.name(), "f32");
    assert_eq!(TensorDType::BF16.name(), "bf16");
    assert_eq!(TensorDType::Q8.name(), "q8");
}

#[test]
fn test_tensor_index_entry_roundtrip() {
    let entry = TensorIndexEntry::new(
        "encoder.layer.0.weight",
        TensorDType::F32,
        vec![512, 768],
        0,
        512 * 768 * 4,
    );

    let bytes = entry.to_bytes();
    let (parsed, _) = TensorIndexEntry::from_bytes(&bytes).unwrap();

    assert_eq!(parsed.name, "encoder.layer.0.weight");
    assert_eq!(parsed.dtype, TensorDType::F32);
    assert_eq!(parsed.shape, vec![512, 768]);
    assert_eq!(parsed.element_count(), 512 * 768);
}

#[test]
fn test_writer_reader_roundtrip() {
    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);

    writer.add_f32_tensor("weight", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    writer.add_f32_tensor("bias", vec![3], &[0.1, 0.2, 0.3]);

    let bytes = writer.write().unwrap();

    let reader = AprV2Reader::from_bytes(&bytes).unwrap();
    assert_eq!(reader.metadata().model_type, "test");
    assert_eq!(reader.tensor_names(), vec!["bias", "weight"]); // Sorted

    let weight = reader.get_f32_tensor("weight").unwrap();
    assert_eq!(weight.len(), 6);
    assert!((weight[0] - 1.0).abs() < 1e-6);

    // Verify alignment
    assert!(reader.verify_alignment());
}

#[test]
fn test_writer_alignment() {
    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);

    // Add tensor with non-aligned size
    writer.add_f32_tensor("test", vec![7], &[1.0; 7]); // 28 bytes, not aligned

    let bytes = writer.write().unwrap();
    let reader = AprV2Reader::from_bytes(&bytes).unwrap();

    // Data should still be 64-byte aligned
    assert!(reader.verify_alignment());
}

#[test]
fn test_shard_manifest() {
    let mut manifest = ShardManifest::new(2);

    manifest.add_shard(ShardInfo {
        filename: "model-00000-of-00002.apr".to_string(),
        index: 0,
        size: 1024,
        tensors: vec!["layer1.weight".to_string(), "layer1.bias".to_string()],
    });

    manifest.add_shard(ShardInfo {
        filename: "model-00001-of-00002.apr".to_string(),
        index: 1,
        size: 2048,
        tensors: vec!["layer2.weight".to_string()],
    });

    assert_eq!(manifest.shard_count, 2);
    assert_eq!(manifest.tensor_count, 3);
    assert_eq!(manifest.total_size, 3072);

    assert_eq!(manifest.shard_for_tensor("layer1.weight"), Some(0));
    assert_eq!(manifest.shard_for_tensor("layer2.weight"), Some(1));
    assert_eq!(manifest.shard_for_tensor("nonexistent"), None);

    // JSON roundtrip
    let json = manifest.to_json().unwrap();
    let parsed = ShardManifest::from_json(&json).unwrap();
    assert_eq!(parsed.shard_count, 2);
}

#[test]
fn test_v2_format_error_display() {
    let err = V2FormatError::InvalidMagic([0x00, 0x01, 0x02, 0x03]);
    assert!(err.to_string().contains("00010203"));

    let err = V2FormatError::ChecksumMismatch;
    assert_eq!(err.to_string(), "Checksum mismatch");
}

#[test]
fn test_quantization_metadata() {
    let quant = QuantizationMetadata {
        quant_type: "int8".to_string(),
        bits: 8,
        block_size: Some(32),
        symmetric: true,
    };

    let mut metadata = AprV2Metadata::new("llm");
    metadata.quantization = Some(quant);

    let json = metadata.to_json().unwrap();
    let parsed = AprV2Metadata::from_json(&json).unwrap();

    let quant = parsed.quantization.unwrap();
    assert_eq!(quant.quant_type, "int8");
    assert_eq!(quant.bits, 8);
    assert_eq!(quant.block_size, Some(32));
}

#[test]
fn test_flags_combinations() {
    let flags = AprV2Flags::new()
        .with(AprV2Flags::LZ4_COMPRESSED)
        .with(AprV2Flags::SHARDED)
        .with(AprV2Flags::HAS_VOCAB);

    assert!(flags.is_lz4_compressed());
    assert!(flags.is_sharded());
    assert!(flags.contains(AprV2Flags::HAS_VOCAB));
    assert!(!flags.is_encrypted());

    let without = flags.without(AprV2Flags::SHARDED);
    assert!(!without.is_sharded());
    assert!(without.is_lz4_compressed());
}

#[test]
fn test_metadata_custom_fields() {
    let mut metadata = AprV2Metadata::new("custom");
    metadata.custom.insert(
        "custom_field".to_string(),
        serde_json::json!("custom_value"),
    );
    metadata
        .custom
        .insert("nested".to_string(), serde_json::json!({"key": "value"}));

    let json = metadata.to_json().unwrap();
    let parsed = AprV2Metadata::from_json(&json).unwrap();

    assert_eq!(
        parsed.custom.get("custom_field"),
        Some(&serde_json::json!("custom_value"))
    );
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_tensor_dtype_from_u8() {
    assert_eq!(TensorDType::from_u8(0), Some(TensorDType::F32));
    assert_eq!(TensorDType::from_u8(1), Some(TensorDType::F16));
    assert_eq!(TensorDType::from_u8(2), Some(TensorDType::BF16));
    assert_eq!(TensorDType::from_u8(3), Some(TensorDType::F64));
    assert_eq!(TensorDType::from_u8(4), Some(TensorDType::I32));
    assert_eq!(TensorDType::from_u8(5), Some(TensorDType::I64));
    assert_eq!(TensorDType::from_u8(6), Some(TensorDType::I8));
    assert_eq!(TensorDType::from_u8(7), Some(TensorDType::U8));
    assert_eq!(TensorDType::from_u8(99), None);
}

#[test]
fn test_v2_format_error_variants() {
    let err = V2FormatError::InvalidHeader("bad header".to_string());
    assert!(err.to_string().contains("bad header") || err.to_string().contains("Invalid"));

    let err = V2FormatError::InvalidTensorIndex("corrupt index".to_string());
    assert!(err.to_string().contains("corrupt") || err.to_string().contains("index"));

    let err = V2FormatError::MetadataError("invalid metadata".to_string());
    assert!(err.to_string().contains("metadata") || err.to_string().contains("Metadata"));

    let err = V2FormatError::AlignmentError("alignment off".to_string());
    assert!(err.to_string().contains("alignment") || err.to_string().contains("Alignment"));

    let err = V2FormatError::IoError("read failed".to_string());
    assert!(err.to_string().contains("read failed") || err.to_string().contains("I/O"));

    let err = V2FormatError::CompressionError("decompress failed".to_string());
    assert!(err.to_string().contains("decompress") || err.to_string().contains("Compression"));
}

#[test]
fn test_header_checksum_compute() {
    let mut header = AprV2Header::new();
    header.version = (2, 0);
    let checksum = header.compute_checksum();
    assert!(checksum != 0);
}

#[test]
fn test_header_update_checksum() {
    let mut header = AprV2Header::new();
    header.checksum = 0;
    header.update_checksum();
    assert!(header.checksum != 0);
}

#[test]
fn test_header_verify_checksum() {
    let mut header = AprV2Header::new();
    header.update_checksum();
    assert!(header.verify_checksum());
    header.version = (99, 0);
    assert!(!header.verify_checksum());
}

#[test]
fn test_metadata_to_json_pretty() {
    let metadata = AprV2Metadata::new("llama");
    let json = metadata.to_json_pretty().unwrap();
    assert!(json.contains("llama"));
    assert!(json.contains('\n')); // Pretty format has newlines
}

#[test]
fn test_tensor_index_entry_element_count() {
    let entry = TensorIndexEntry::new(
        "test",
        TensorDType::F32,
        vec![2, 3, 4],
        0,
        96, // 2*3*4*4 bytes
    );
    assert_eq!(entry.element_count(), 24);
}

#[test]
fn test_tensor_index_entry_to_bytes() {
    let entry = TensorIndexEntry::new("t", TensorDType::F32, vec![10], 0, 40);
    let bytes = entry.to_bytes();
    assert!(!bytes.is_empty());
}

#[test]
fn test_writer_with_lz4() {
    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);
    writer.with_lz4_compression();
    // Just verify it doesn't panic
}

#[test]
fn test_writer_with_sharding() {
    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);
    writer.with_sharding(4, 0);
    // Just verify it doesn't panic
}

#[test]
fn test_reader_ref_from_bytes() {
    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("w", vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let bytes = writer.write().unwrap();

    let reader = AprV2ReaderRef::from_bytes(&bytes).unwrap();
    assert_eq!(reader.header().version.0, 2);
    assert_eq!(reader.metadata().model_type, "test");
    assert_eq!(reader.tensor_names().len(), 1);
    assert!(reader.get_tensor("w").is_some());
    assert!(reader.verify_alignment());
}

#[test]
fn test_reader_ref_get_tensor_data() {
    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("w", vec![2], &[1.0, 2.0]);
    let bytes = writer.write().unwrap();

    let reader = AprV2ReaderRef::from_bytes(&bytes).unwrap();
    let data = reader.get_tensor_data("w");
    assert!(data.is_some());
}

#[test]
fn test_reader_ref_get_f32_tensor() {
    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("w", vec![3], &[1.0, 2.0, 3.0]);
    let bytes = writer.write().unwrap();

    let reader = AprV2ReaderRef::from_bytes(&bytes).unwrap();
    let tensor = reader.get_f32_tensor("w").unwrap();
    assert_eq!(tensor.len(), 3);
    assert!((tensor[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_sharding_metadata() {
    let shard = ShardingMetadata {
        shard_count: 4,
        shard_index: 0,
        total_size: 10_000_000,
        pattern: Some("model-{:05d}-of-{:05d}.apr".to_string()),
    };
    assert_eq!(shard.shard_count, 4);
    assert_eq!(shard.total_size, 10_000_000);
    assert!(shard.pattern.is_some());
}

#[test]
fn test_flags_all_bits() {
    let flags = AprV2Flags::new()
        .with(AprV2Flags::LZ4_COMPRESSED)
        .with(AprV2Flags::ENCRYPTED)
        .with(AprV2Flags::SIGNED)
        .with(AprV2Flags::SHARDED)
        .with(AprV2Flags::HAS_VOCAB)
        .with(AprV2Flags::QUANTIZED);

    assert!(flags.is_lz4_compressed());
    assert!(flags.is_encrypted());
    assert!(flags.contains(AprV2Flags::SIGNED));
    assert!(flags.is_sharded());
    assert!(flags.contains(AprV2Flags::HAS_VOCAB));
    assert!(flags.is_quantized());
}

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

#[test]
fn test_pygmy_apr_f32_tensor() {
    use crate::format::test_factory::build_pygmy_apr;

    let data = build_pygmy_apr();
    let reader = AprV2Reader::from_bytes(&data).expect("parse");

    // Test getting tensor as f32
    let f32_data = reader.get_f32_tensor("model.embed_tokens.weight");
    assert!(f32_data.is_some());

    // Check that values are reasonable (not NaN or Inf)
    for val in f32_data.unwrap() {
        assert!(val.is_finite(), "Tensor values should be finite");
    }
}

#[test]
fn test_pygmy_apr_f16_parsing() {
    use crate::format::test_factory::build_pygmy_apr_f16;

    let data = build_pygmy_apr_f16();
    let reader = AprV2Reader::from_bytes(&data).expect("parse");

    // Test that F16 APR parses correctly
    let embed = reader.get_tensor("model.embed_tokens.weight");
    assert!(embed.is_some());
    assert_eq!(embed.unwrap().dtype, TensorDType::F16);
}

#[test]
fn test_pygmy_apr_q8_parsing() {
    use crate::format::test_factory::build_pygmy_apr_q8;

    let data = build_pygmy_apr_q8();
    let reader = AprV2Reader::from_bytes(&data).expect("parse");

    // Test that Q8 APR parses correctly
    let tensor = reader.get_tensor("model.layers.0.self_attn.q_proj.weight");
    assert!(tensor.is_some());
    assert_eq!(tensor.unwrap().dtype, TensorDType::Q8);
}

#[test]
fn test_pygmy_apr_q4_parsing() {
    use crate::format::test_factory::build_pygmy_apr_q4;

    let data = build_pygmy_apr_q4();
    let reader = AprV2Reader::from_bytes(&data).expect("parse");

    // Test that Q4 APR parses correctly
    let tensor = reader.get_tensor("model.layers.0.self_attn.q_proj.weight");
    assert!(tensor.is_some());
    assert_eq!(tensor.unwrap().dtype, TensorDType::Q4);
}

#[test]
fn test_pygmy_apr_reader_ref_vs_reader() {
    use crate::format::test_factory::build_pygmy_apr;

    let data = build_pygmy_apr();

    // Compare AprV2Reader and AprV2ReaderRef
    let reader = AprV2Reader::from_bytes(&data).expect("parse");
    let reader_ref = AprV2ReaderRef::from_bytes(&data).expect("parse ref");

    // Should have same metadata
    assert_eq!(
        reader.metadata().model_type,
        reader_ref.metadata().model_type
    );
    assert_eq!(
        reader.metadata().architecture,
        reader_ref.metadata().architecture
    );

    // Should have same tensors
    assert_eq!(reader.tensor_names().len(), reader_ref.tensor_names().len());
}

#[test]
fn test_pygmy_apr_shard_manifest() {
    let mut manifest = ShardManifest::new(3);
    assert_eq!(manifest.shard_count, 3);
    assert_eq!(manifest.tensor_count, 0);

    manifest.add_shard(ShardInfo {
        filename: "shard-00001.apr".to_string(),
        index: 0,
        size: 1000,
        tensors: vec!["tensor.a".to_string(), "tensor.b".to_string()],
    });

    assert_eq!(manifest.tensor_count, 2);
    assert_eq!(manifest.total_size, 1000);
    assert_eq!(manifest.shard_for_tensor("tensor.a"), Some(0));
    assert_eq!(manifest.shard_for_tensor("nonexistent"), None);
}

#[test]
fn test_pygmy_apr_shard_manifest_json() {
    let mut manifest = ShardManifest::new(2);
    manifest.add_shard(ShardInfo {
        filename: "shard-1.apr".to_string(),
        index: 0,
        size: 500,
        tensors: vec!["a".to_string()],
    });

    // Test JSON roundtrip
    let json = manifest.to_json().expect("serialize");
    let parsed = ShardManifest::from_json(&json).expect("deserialize");

    assert_eq!(parsed.shard_count, 2);
    assert_eq!(parsed.tensor_count, 1);
    assert_eq!(parsed.shard_for_tensor("a"), Some(0));
}

#[test]
fn test_v2_format_error_display_all_variants() {
    let errors = vec![
        V2FormatError::InvalidMagic([0xFF, 0xFF, 0xFF, 0xFF]),
        V2FormatError::InvalidHeader("bad header".to_string()),
        V2FormatError::InvalidTensorIndex("bad index".to_string()),
        V2FormatError::MetadataError("bad metadata".to_string()),
        V2FormatError::ChecksumMismatch,
        V2FormatError::AlignmentError("misaligned".to_string()),
        V2FormatError::IoError("io failed".to_string()),
        V2FormatError::CompressionError("compress failed".to_string()),
    ];

    for error in errors {
        let display = format!("{error}");
        assert!(!display.is_empty());
    }
}

#[test]
fn test_tensor_dtype_coverage() {
    // Test all TensorDType variants
    let dtypes = [
        TensorDType::F32,
        TensorDType::F16,
        TensorDType::BF16,
        TensorDType::F64,
        TensorDType::I32,
        TensorDType::I64,
        TensorDType::I8,
        TensorDType::U8,
        TensorDType::Q4,
        TensorDType::Q8,
        TensorDType::Q4K,
        TensorDType::Q6K,
    ];

    for dtype in dtypes {
        // Test name()
        assert!(!dtype.name().is_empty());

        // Test bytes_per_element()
        let _ = dtype.bytes_per_element();

        // Test from_u8 roundtrip
        let value = dtype as u8;
        assert_eq!(TensorDType::from_u8(value), Some(dtype));
    }

    // Test invalid dtype
    assert_eq!(TensorDType::from_u8(255), None);
}

#[test]
fn test_tensor_index_entry_element_count_3d() {
    let entry = TensorIndexEntry::new(
        "test",
        TensorDType::F32,
        vec![10, 20, 30],
        0,
        10 * 20 * 30 * 4,
    );

    assert_eq!(entry.element_count(), 6000); // 10 * 20 * 30
}
