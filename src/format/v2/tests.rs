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

// ====================================================================
// Coverage: flag methods
// ====================================================================

#[test]
fn test_flags_zstd_compressed() {
    let flags = AprV2Flags::from_bits(AprV2Flags::ZSTD_COMPRESSED);
    assert!(flags.is_zstd_compressed());
    assert!(!flags.is_lz4_compressed());
    assert!(!flags.is_encrypted());
}

#[test]
fn test_flags_encrypted() {
    let flags = AprV2Flags::from_bits(AprV2Flags::ENCRYPTED);
    assert!(flags.is_encrypted());
    assert!(!flags.is_zstd_compressed());
}

#[test]
fn test_flags_sharded() {
    let flags = AprV2Flags::from_bits(AprV2Flags::SHARDED);
    assert!(flags.is_sharded());
    assert!(!flags.is_quantized());
}

#[test]
fn test_flags_quantized() {
    let flags = AprV2Flags::from_bits(AprV2Flags::QUANTIZED);
    assert!(flags.is_quantized());
    assert!(!flags.is_sharded());
}

#[test]
fn test_flags_combined() {
    let flags = AprV2Flags::from_bits(AprV2Flags::LZ4_COMPRESSED | AprV2Flags::QUANTIZED);
    assert!(flags.is_lz4_compressed());
    assert!(flags.is_quantized());
    assert!(!flags.is_encrypted());
    assert!(!flags.is_sharded());
    assert!(!flags.is_zstd_compressed());
}

// ====================================================================
// Coverage: padding_to_align utility
// ====================================================================

#[test]
fn test_padding_to_align() {
    assert_eq!(padding_to_align(0, 64), 0);
    assert_eq!(padding_to_align(1, 64), 63);
    assert_eq!(padding_to_align(63, 64), 1);
    assert_eq!(padding_to_align(64, 64), 0);
    assert_eq!(padding_to_align(65, 64), 63);
    assert_eq!(padding_to_align(128, 64), 0);
}

// ====================================================================
// Coverage: AprV2Header::default
// ====================================================================

#[test]
fn test_header_default() {
    let h = AprV2Header::default();
    assert_eq!(h.magic, MAGIC_V2);
    assert_eq!(h.version, (2, 0));
}

// ====================================================================
// Coverage: write_to / from_reader (io trait paths)
// ====================================================================

#[test]
fn test_writer_write_to() {
    let meta = AprV2Metadata::new("llama");
    let mut writer = AprV2Writer::new(meta);
    writer.add_f32_tensor("test", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut buf = Vec::new();
    writer.write_to(&mut buf).expect("write_to");
    assert!(!buf.is_empty());
    // Should be valid APR
    assert_eq!(&buf[0..4], &MAGIC_V2);
}

#[test]
fn test_reader_from_reader() {
    let meta = AprV2Metadata::new("llama");
    let mut writer = AprV2Writer::new(meta);
    writer.add_f32_tensor("t", vec![2], &[1.0, 2.0]);
    let bytes = writer.write().expect("write");
    let mut cursor = std::io::Cursor::new(bytes);
    let reader = AprV2Reader::from_reader(&mut cursor).expect("from_reader");
    assert_eq!(reader.tensor_names().len(), 1);
}

#[test]
fn test_reader_header_getter() {
    let meta = AprV2Metadata::new("llama");
    let mut writer = AprV2Writer::new(meta);
    writer.add_f32_tensor("t", vec![2], &[1.0, 2.0]);
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("from_bytes");
    let header = reader.header();
    assert_eq!(header.magic, MAGIC_V2);
    assert_eq!(header.version, (2, 0));
}

// ====================================================================
// Coverage: f32_to_f16 overflow/underflow edges
// ====================================================================

#[test]
fn test_f32_to_f16_overflow_to_inf() {
    let val = super::f32_to_f16(65536.0); // Too large for f16 (max ~65504)
    let back = super::f16_to_f32(val);
    assert!(back.is_infinite());
}

#[test]
fn test_f32_to_f16_underflow_to_zero() {
    let val = super::f32_to_f16(1e-10); // Way below f16 min denorm
    let back = super::f16_to_f32(val);
    assert_eq!(back, 0.0);
}

#[test]
fn test_f32_to_f16_roundtrip_normal() {
    // 1.5 should roundtrip cleanly (exact in f16)
    let h = super::f32_to_f16(1.5);
    let back = super::f16_to_f32(h);
    assert!((back - 1.5).abs() < 1e-3);
}

#[test]
fn test_f32_to_f16_negative() {
    let h = super::f32_to_f16(-2.0);
    let back = super::f16_to_f32(h);
    assert!((back - (-2.0)).abs() < 1e-3);
}

// ====================================================================
// GH-200: Q4K / Q6K dequantization in get_tensor_as_f32()
// ====================================================================

/// GH-200: Q4K tensor written via add_q4k_raw_tensor can be read back as f32.
#[test]
fn test_q4k_tensor_roundtrip() {
    // Q4_K super-block: 144 bytes per 256 elements
    let raw_q4k = vec![0u8; 144];
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_q4k_raw_tensor("test_q4k", vec![16, 16], raw_q4k);
    let bytes = writer.write().expect("write APR");

    let reader = AprV2Reader::from_bytes(&bytes).expect("parse APR");
    let f32_data = reader.get_tensor_as_f32("test_q4k");
    assert!(f32_data.is_some(), "Q4K tensor must dequantize to f32");
    let f32_data = f32_data.unwrap();
    assert_eq!(f32_data.len(), 256);
    for &v in &f32_data {
        assert!(v.is_finite(), "Q4K dequant must produce finite values");
    }
}

/// GH-200: Q6K tensor written via add_q6k_raw_tensor can be read back as f32.
#[test]
fn test_q6k_tensor_roundtrip() {
    // Q6_K super-block: 210 bytes per 256 elements
    let raw_q6k = vec![0u8; 210];
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_q6k_raw_tensor("test_q6k", vec![16, 16], raw_q6k);
    let bytes = writer.write().expect("write APR");

    let reader = AprV2Reader::from_bytes(&bytes).expect("parse APR");
    let f32_data = reader.get_tensor_as_f32("test_q6k");
    assert!(f32_data.is_some(), "Q6K tensor must dequantize to f32");
    let f32_data = f32_data.unwrap();
    assert_eq!(f32_data.len(), 256);
    for &v in &f32_data {
        assert!(v.is_finite(), "Q6K dequant must produce finite values");
    }
}

/// GH-200: Q4K dequant with non-zero data produces non-trivial values.
#[test]
fn test_q4k_dequant_nontrivial() {
    // Q4_K layout: d (f16, 2B) + dmin (f16, 2B) + scales (12B) + qs (128B) = 144B
    let mut raw = vec![0u8; 144];
    // Set d = 1.0 in f16 (0x3C00)
    raw[0] = 0x00;
    raw[1] = 0x3C;
    // Set scales bytes (offset 4..16) to non-zero so sub-block scales are non-zero.
    // scales[i] = scales_bytes[i] & 0x3F, so byte value 1 → scale 1.
    for i in 4..16 {
        raw[i] = 0x01;
    }
    // Set quant values (offset 16..144) to non-zero nibbles
    for i in 16..144 {
        raw[i] = 0x55; // nibbles (5, 5)
    }

    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_q4k_raw_tensor("q4k_nonzero", vec![256], raw);
    let bytes = writer.write().expect("write APR");

    let reader = AprV2Reader::from_bytes(&bytes).expect("parse APR");
    let f32_data = reader.get_tensor_as_f32("q4k_nonzero").expect("dequant");
    assert_eq!(f32_data.len(), 256);
    let nonzero_count = f32_data.iter().filter(|&&v| v != 0.0).count();
    assert!(
        nonzero_count > 0,
        "Non-zero Q4K data must produce non-zero f32 values"
    );
}

/// GH-200: Q6K dequant with non-zero data produces non-trivial values.
#[test]
fn test_q6k_dequant_nontrivial() {
    // Q6_K layout: ql (128B) + qh (64B) + scales (16B) + d (f16, 2B) = 210B
    let mut raw = vec![0u8; 210];
    // Set ql values (offset 0..128) to non-zero
    for i in 0..128 {
        raw[i] = 0x33;
    }
    // Set scales (offset 192..208) to non-zero (i8 value 1)
    for i in 192..208 {
        raw[i] = 0x01;
    }
    // Set d = 1.0 in f16 (0x3C00) at offset 208
    raw[208] = 0x00;
    raw[209] = 0x3C;

    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_q6k_raw_tensor("q6k_nonzero", vec![256], raw);
    let bytes = writer.write().expect("write APR");

    let reader = AprV2Reader::from_bytes(&bytes).expect("parse APR");
    let f32_data = reader.get_tensor_as_f32("q6k_nonzero").expect("dequant");
    assert_eq!(f32_data.len(), 256);
    let nonzero_count = f32_data.iter().filter(|&&v| v != 0.0).count();
    assert!(
        nonzero_count > 0,
        "Non-zero Q6K data must produce non-zero f32 values"
    );
}

/// GH-200: AprV2ReaderRef also handles Q4K/Q6K (same code path, different reader).
#[test]
fn test_q4k_via_ref_reader() {
    let raw_q4k = vec![0u8; 144];
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_q4k_raw_tensor("ref_q4k", vec![16, 16], raw_q4k);
    let bytes = writer.write().expect("write APR");

    let reader = AprV2ReaderRef::from_bytes(&bytes).expect("parse APR ref");
    let f32_data = reader.get_tensor_as_f32("ref_q4k");
    assert!(f32_data.is_some(), "Q4K via ref reader must dequantize");
    assert_eq!(f32_data.unwrap().len(), 256);
}

/// GH-200: AprV2ReaderRef also handles Q6K.
#[test]
fn test_q6k_via_ref_reader() {
    let raw_q6k = vec![0u8; 210];
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_q6k_raw_tensor("ref_q6k", vec![16, 16], raw_q6k);
    let bytes = writer.write().expect("write APR");

    let reader = AprV2ReaderRef::from_bytes(&bytes).expect("parse APR ref");
    let f32_data = reader.get_tensor_as_f32("ref_q6k");
    assert!(f32_data.is_some(), "Q6K via ref reader must dequantize");
    assert_eq!(f32_data.unwrap().len(), 256);
}

// ============================================================================
// LAYOUT-002 Jidoka Guard Tests
// ============================================================================

/// LAYOUT-002: New APR files should have LAYOUT_ROW_MAJOR flag set.
#[test]
fn test_layout_002_writer_sets_row_major_flag() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    let bytes = writer.write().expect("write APR");

    let header = AprV2Header::from_bytes(&bytes).expect("parse header");
    assert!(
        header.flags.is_row_major(),
        "LAYOUT-002: New APR files must have LAYOUT_ROW_MAJOR flag set"
    );
    assert!(
        !header.flags.is_column_major(),
        "LAYOUT-002: New APR files must NOT have LAYOUT_COLUMN_MAJOR flag set"
    );
}

/// LAYOUT-002: Reader should accept valid row-major APR files.
#[test]
fn test_layout_002_reader_accepts_row_major() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_f32_tensor("test", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    let bytes = writer.write().expect("write APR");

    // Both readers should accept the file
    assert!(AprV2Reader::from_bytes(&bytes).is_ok());
    assert!(AprV2ReaderRef::from_bytes(&bytes).is_ok());
}

/// LAYOUT-002: Reader should reject "dirty" APR files with LAYOUT_COLUMN_MAJOR flag.
#[test]
fn test_layout_002_jidoka_guard_rejects_column_major() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_f32_tensor("test", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    let mut bytes = writer.write().expect("write APR");

    // Manually set the LAYOUT_COLUMN_MAJOR flag (simulating a dirty APR file)
    // Flag bits are at offset 6-7 (u16 little-endian)
    let current_flags = u16::from_le_bytes([bytes[6], bytes[7]]);
    let dirty_flags = current_flags | AprV2Flags::LAYOUT_COLUMN_MAJOR;
    bytes[6] = (dirty_flags & 0xFF) as u8;
    bytes[7] = ((dirty_flags >> 8) & 0xFF) as u8;

    // Update checksum after modifying flags
    let mut header = AprV2Header::from_bytes(&bytes).expect("parse header");
    header.update_checksum();
    let header_bytes = header.to_bytes();
    bytes[..HEADER_SIZE_V2].copy_from_slice(&header_bytes);

    // Both readers should reject the dirty file
    let result1 = AprV2Reader::from_bytes(&bytes);
    assert!(
        result1.is_err(),
        "LAYOUT-002: Reader must reject column-major APR"
    );
    assert!(
        result1.unwrap_err().to_string().contains("LAYOUT-002"),
        "Error message should mention LAYOUT-002"
    );

    let result2 = AprV2ReaderRef::from_bytes(&bytes);
    assert!(
        result2.is_err(),
        "LAYOUT-002: ReaderRef must reject column-major APR"
    );
}

/// LAYOUT-002: Flags helper functions work correctly.
#[test]
fn test_layout_002_flags_helpers() {
    // Test row-major flag
    let row_major = AprV2Flags::new().with(AprV2Flags::LAYOUT_ROW_MAJOR);
    assert!(row_major.is_row_major());
    assert!(!row_major.is_column_major());
    assert!(row_major.is_layout_valid());

    // Test column-major flag (forbidden)
    let col_major = AprV2Flags::new().with(AprV2Flags::LAYOUT_COLUMN_MAJOR);
    assert!(!col_major.is_row_major());
    assert!(col_major.is_column_major());
    assert!(!col_major.is_layout_valid());

    // Test both flags (invalid combination)
    let both = AprV2Flags::new()
        .with(AprV2Flags::LAYOUT_ROW_MAJOR)
        .with(AprV2Flags::LAYOUT_COLUMN_MAJOR);
    assert!(!both.is_layout_valid(), "Both flags set should be invalid");

    // Test no layout flags (pre-LAYOUT-002 files - assumed valid)
    let no_flags = AprV2Flags::new();
    assert!(
        no_flags.is_layout_valid(),
        "Pre-LAYOUT-002 files should be accepted"
    );
}

// ============================================================================
// Extended Coverage Tests (T-COV-96)
// Target: exercise the 589 missed lines in mod.rs
// ============================================================================

// ---------------------------------------------------------------------------
// crc32 - direct tests
// ---------------------------------------------------------------------------

#[test]
fn test_crc32_empty_input() {
    let result = super::crc32(&[]);
    // CRC32 of empty data is 0x00000000
    assert_eq!(result, 0x0000_0000);
}

#[test]
fn test_crc32_deterministic() {
    let data = b"hello world";
    let a = super::crc32(data);
    let b = super::crc32(data);
    assert_eq!(a, b, "CRC32 must be deterministic");
}

#[test]
fn test_crc32_different_inputs_differ() {
    let a = super::crc32(b"abc");
    let b = super::crc32(b"abd");
    assert_ne!(a, b, "Different inputs should produce different checksums");
}

#[test]
fn test_crc32_known_value() {
    // CRC32 of "123456789" is 0xCBF43926 (standard test vector)
    let data = b"123456789";
    let result = super::crc32(data);
    assert_eq!(result, 0xCBF4_3926);
}

// ---------------------------------------------------------------------------
// f32_to_f16 / f16_to_f32 - comprehensive edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_f32_to_f16_negative_infinity() {
    let h = super::f32_to_f16(f32::NEG_INFINITY);
    let back = super::f16_to_f32(h);
    assert!(back.is_infinite() && back.is_sign_negative());
}

#[test]
fn test_f32_to_f16_denormal_f32_zero() {
    // A denormalized f32 (exp == 0, mantissa != 0) should become zero in f16
    let tiny = f32::from_bits(0x0000_0001); // smallest positive denormal f32
    let h = super::f32_to_f16(tiny);
    let back = super::f16_to_f32(h);
    assert_eq!(back, 0.0, "Denormal f32 should become zero in f16");
}

#[test]
fn test_f32_to_f16_negative_denormal_f32() {
    // Negative denormalized f32 (sign=1, exp=0, mantissa!=0)
    let bits: u32 = 0x8000_0001; // negative smallest denormal
    let tiny = f32::from_bits(bits);
    let h = super::f32_to_f16(tiny);
    let back = super::f16_to_f32(h);
    assert_eq!(back, -0.0, "Negative denormal f32 should become -0 in f16");
}

#[test]
fn test_f16_to_f32_denormal_f16() {
    // Denormal f16: exp=0, mantissa!=0, e.g. 0x0001 = smallest positive denormal
    let back = super::f16_to_f32(0x0001);
    assert!(back > 0.0, "Denormal f16 should produce positive f32");
    assert!(back < 1e-4, "Denormal f16 should be very small");
}

#[test]
fn test_f16_to_f32_denormal_f16_various() {
    // Several denormal f16 values
    for mantissa in [0x0001u16, 0x0010, 0x0100, 0x03FF] {
        let val = super::f16_to_f32(mantissa);
        assert!(
            val.is_finite(),
            "Denormal f16 {mantissa:#06x} should be finite"
        );
        assert!(val > 0.0, "Denormal f16 {mantissa:#06x} should be positive");
    }
}

#[test]
fn test_f16_to_f32_negative_denormal_f16() {
    // Negative denormal f16: sign=1, exp=0, mantissa=0x0001 => 0x8001
    let back = super::f16_to_f32(0x8001);
    assert!(
        back < 0.0,
        "Negative denormal f16 should produce negative f32"
    );
}

#[test]
fn test_f16_to_f32_inf() {
    // Positive infinity: 0x7C00
    let val = super::f16_to_f32(0x7C00);
    assert!(val.is_infinite() && val.is_sign_positive());
}

#[test]
fn test_f16_to_f32_negative_inf() {
    // Negative infinity: 0xFC00
    let val = super::f16_to_f32(0xFC00);
    assert!(val.is_infinite() && val.is_sign_negative());
}

#[test]
fn test_f16_to_f32_nan() {
    // NaN: exp=31, mantissa!=0, e.g. 0x7E00
    let val = super::f16_to_f32(0x7E00);
    assert!(val.is_nan(), "f16 NaN should produce f32 NaN");
}

#[test]
fn test_f32_to_f16_negative_nan() {
    let h = super::f32_to_f16(-f32::NAN);
    // The sign bit should be set
    assert!(h & 0x8000 != 0, "Negative NaN should preserve sign");
    assert!(h & 0x7FFF > 0x7C00, "Negative NaN should have NaN payload");
}

#[test]
fn test_f32_to_f16_negative_inf_and_back() {
    let h = super::f32_to_f16(f32::NEG_INFINITY);
    assert_eq!(h, 0xFC00, "Negative infinity should be 0xFC00 in f16");
}

// ---------------------------------------------------------------------------
// dequantize_q4 - edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_dequantize_q4_empty_data() {
    let result = super::dequantize_q4(&[], 0);
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q4_empty_data_nonzero_count() {
    // Empty data but requested elements: should pad with zeros
    let result = super::dequantize_q4(&[], 10);
    assert_eq!(result.len(), 10);
    for &v in &result {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_dequantize_q4_nan_scale() {
    // NaN scale: f16 NaN = 0x7E00
    let mut data = vec![0u8; 18]; // One block: 2 byte scale + 16 bytes nibbles
    data[0] = 0x00;
    data[1] = 0x7E; // NaN f16
                    // Set some nibbles
    for i in 2..18 {
        data[i] = 0x55;
    }
    let result = super::dequantize_q4(&data, 32);
    assert_eq!(result.len(), 32);
    // With NaN scale clamped to 0, all values should be 0
    for &v in &result {
        assert!(v.is_finite(), "NaN scale should be clamped to 0");
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_dequantize_q4_inf_scale() {
    // Inf scale: f16 Inf = 0x7C00
    let mut data = vec![0u8; 18];
    data[0] = 0x00;
    data[1] = 0x7C; // Inf f16
    for i in 2..18 {
        data[i] = 0x55;
    }
    let result = super::dequantize_q4(&data, 32);
    assert_eq!(result.len(), 32);
    // Inf scale should be clamped to 0
    for &v in &result {
        assert!(v.is_finite(), "Inf scale should be clamped to 0");
    }
}

#[test]
fn test_dequantize_q4_subnormal_scale() {
    // Subnormal scale: a very small f16 value below F16_MIN_NORMAL
    let mut data = vec![0u8; 18];
    // f16 0x0001 is the smallest denormal: ~5.96e-8 which is < F16_MIN_NORMAL
    data[0] = 0x01;
    data[1] = 0x00;
    for i in 2..18 {
        data[i] = 0x88;
    }
    let result = super::dequantize_q4(&data, 32);
    assert_eq!(result.len(), 32);
    // Subnormal scale should be clamped to 0
    for &v in &result {
        assert_eq!(v, 0.0, "Subnormal scale should be clamped to 0");
    }
}

#[test]
fn test_dequantize_q4_partial_block() {
    // Request only 10 elements from a block of 32
    let mut data = vec![0u8; 18];
    // Set scale = 1.0 in f16 (0x3C00)
    data[0] = 0x00;
    data[1] = 0x3C;
    for i in 2..18 {
        data[i] = 0x88; // nibbles (8, 8) -> q=(8-8)=0, q=(8-8)=0 -> 0*scale = 0
    }
    let result = super::dequantize_q4(&data, 10);
    assert_eq!(result.len(), 10);
}

#[test]
fn test_dequantize_q4_truncated_data() {
    // Data too short (only 1 byte for scale)
    let data = vec![0x00u8; 1];
    let result = super::dequantize_q4(&data, 32);
    // Should pad with zeros since it can't read scale
    assert_eq!(result.len(), 32);
    for &v in &result {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_dequantize_q4_multiple_blocks() {
    // Two blocks: 64 elements
    let mut data = vec![0u8; 36]; // 2 * 18 bytes
                                  // Block 1: scale = 1.0
    data[0] = 0x00;
    data[1] = 0x3C;
    // Block 2: scale = 2.0 (f16 0x4000)
    data[18] = 0x00;
    data[19] = 0x40;
    let result = super::dequantize_q4(&data, 64);
    assert_eq!(result.len(), 64);
}

// ---------------------------------------------------------------------------
// AprV2Header::from_bytes - error paths
// ---------------------------------------------------------------------------

#[test]
fn test_header_from_bytes_too_small() {
    let buf = [0u8; 10]; // Too small
    let result = AprV2Header::from_bytes(&buf);
    assert!(matches!(result, Err(V2FormatError::InvalidHeader(_))));
}

#[test]
fn test_header_from_bytes_exactly_64_valid() {
    let mut header = AprV2Header::new();
    header.update_checksum();
    let bytes = header.to_bytes();
    assert_eq!(bytes.len(), 64);
    let parsed = AprV2Header::from_bytes(&bytes).expect("exactly 64 bytes should work");
    assert!(parsed.is_valid());
}

// ---------------------------------------------------------------------------
// TensorIndexEntry::from_bytes - error paths
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_index_from_bytes_too_small() {
    let buf = [0u8; 2]; // Too small (need at least 4)
    let result = TensorIndexEntry::from_bytes(&buf);
    assert!(matches!(result, Err(V2FormatError::InvalidTensorIndex(_))));
}

#[test]
fn test_tensor_index_from_bytes_name_too_large() {
    // name_len = 300, but buffer only has a few bytes after that
    let mut buf = vec![0u8; 6];
    // name_len = 300 (u16 LE)
    buf[0] = 0x2C;
    buf[1] = 0x01;
    let result = TensorIndexEntry::from_bytes(&buf);
    assert!(matches!(result, Err(V2FormatError::InvalidTensorIndex(_))));
}

#[test]
fn test_tensor_index_from_bytes_invalid_dtype() {
    // Valid name_len=1, name='a', then invalid dtype byte
    let entry = TensorIndexEntry::new("a", TensorDType::F32, vec![4], 0, 16);
    let mut bytes = entry.to_bytes();
    // Corrupt the dtype byte (after 2 bytes name_len + 1 byte name = offset 3)
    bytes[3] = 255; // invalid dtype
    let result = TensorIndexEntry::from_bytes(&bytes);
    assert!(matches!(result, Err(V2FormatError::InvalidTensorIndex(_))));
}

#[test]
fn test_tensor_index_from_bytes_truncated_shape() {
    // Create a valid entry, then truncate before shape dimensions
    let entry = TensorIndexEntry::new("ab", TensorDType::F32, vec![10, 20], 0, 800);
    let bytes = entry.to_bytes();
    // Truncate after ndim byte but before shape values
    // 2 (name_len) + 2 (name) + 1 (dtype) + 1 (ndim) = 6 bytes, need +16 for 2 dims
    let truncated = &bytes[..7];
    let result = TensorIndexEntry::from_bytes(truncated);
    assert!(matches!(result, Err(V2FormatError::InvalidTensorIndex(_))));
}

#[test]
fn test_tensor_index_from_bytes_truncated_offset() {
    // Create a valid entry, truncate before offset field
    let entry = TensorIndexEntry::new("a", TensorDType::F32, vec![4], 0, 16);
    let bytes = entry.to_bytes();
    // 2 (name_len) + 1 (name) + 1 (dtype) + 1 (ndim) + 8 (dim) = 13
    // Need 16 more for offset+size, truncate before that
    let truncated = &bytes[..13];
    let result = TensorIndexEntry::from_bytes(truncated);
    assert!(matches!(result, Err(V2FormatError::InvalidTensorIndex(_))));
}

#[test]
fn test_tensor_index_entry_empty_shape() {
    let entry = TensorIndexEntry::new("scalar", TensorDType::F32, vec![], 0, 4);
    assert_eq!(entry.element_count(), 1); // product of empty vec is 1
    let bytes = entry.to_bytes();
    let (parsed, _consumed) = TensorIndexEntry::from_bytes(&bytes).expect("parse");
    assert_eq!(parsed.shape.len(), 0);
}

#[test]
fn test_tensor_index_entry_many_dims() {
    // Shape with 8 dimensions (max before truncation)
    let shape = vec![2, 3, 4, 5, 6, 7, 8, 9];
    let entry = TensorIndexEntry::new("multi", TensorDType::F32, shape.clone(), 0, 100);
    let bytes = entry.to_bytes();
    let (parsed, _consumed) = TensorIndexEntry::from_bytes(&bytes).expect("parse");
    assert_eq!(parsed.shape, shape);
}

// ---------------------------------------------------------------------------
// AprV2Reader - error paths
// ---------------------------------------------------------------------------

#[test]
fn test_reader_from_bytes_too_small() {
    let buf = [0u8; 10];
    let result = AprV2Reader::from_bytes(&buf);
    assert!(matches!(result, Err(V2FormatError::InvalidHeader(_))));
}

#[test]
fn test_reader_from_bytes_checksum_mismatch() {
    let mut header = AprV2Header::new();
    header.update_checksum();
    let mut bytes = header.to_bytes().to_vec();
    // Corrupt a non-checksum byte after updating checksum
    bytes[4] = 99; // Corrupt version byte
                   // Pad to make it large enough
    bytes.resize(256, 0);
    let result = AprV2Reader::from_bytes(&bytes);
    assert!(matches!(result, Err(V2FormatError::ChecksumMismatch)));
}

#[test]
fn test_reader_ref_from_bytes_too_small() {
    let buf = [0u8; 10];
    let result = AprV2ReaderRef::from_bytes(&buf);
    assert!(matches!(result, Err(V2FormatError::InvalidHeader(_))));
}

#[test]
fn test_reader_ref_from_bytes_checksum_mismatch() {
    let mut header = AprV2Header::new();
    header.update_checksum();
    let mut bytes = header.to_bytes().to_vec();
    bytes[4] = 99; // Corrupt version
    bytes.resize(256, 0);
    let result = AprV2ReaderRef::from_bytes(&bytes);
    assert!(matches!(result, Err(V2FormatError::ChecksumMismatch)));
}

#[test]
fn test_reader_file_too_small_for_metadata() {
    // Create a header that claims metadata extends beyond file
    let mut header = AprV2Header::new();
    header.metadata_offset = 64;
    header.metadata_size = 9999; // Way beyond file size
    header.update_checksum();
    let bytes = header.to_bytes().to_vec();
    // File is only 64 bytes, metadata claims to be at 64..10063
    let result = AprV2Reader::from_bytes(&bytes);
    assert!(matches!(result, Err(V2FormatError::InvalidHeader(_))));
}

#[test]
fn test_reader_ref_file_too_small_for_metadata() {
    let mut header = AprV2Header::new();
    header.metadata_offset = 64;
    header.metadata_size = 9999;
    header.update_checksum();
    let bytes = header.to_bytes().to_vec();
    let result = AprV2ReaderRef::from_bytes(&bytes);
    assert!(matches!(result, Err(V2FormatError::InvalidHeader(_))));
}

// ---------------------------------------------------------------------------
// Reader - get_tensor_data / get_f32_tensor edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_reader_get_tensor_nonexistent() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_f32_tensor("exists", vec![2], &[1.0, 2.0]);
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    assert!(reader.get_tensor("nonexistent").is_none());
    assert!(reader.get_tensor_data("nonexistent").is_none());
    assert!(reader.get_f32_tensor("nonexistent").is_none());
    assert!(reader.get_tensor_as_f32("nonexistent").is_none());
}

#[test]
fn test_reader_get_f32_tensor_wrong_dtype() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_f16_tensor("f16_weight", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    // get_f32_tensor should return None for non-F32 tensors
    assert!(reader.get_f32_tensor("f16_weight").is_none());
}

#[test]
fn test_reader_ref_get_f32_tensor_wrong_dtype() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_f16_tensor("f16_weight", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    let bytes = writer.write().expect("write");
    let reader = AprV2ReaderRef::from_bytes(&bytes).expect("read");

    assert!(reader.get_f32_tensor("f16_weight").is_none());
}

#[test]
fn test_reader_get_tensor_as_f32_unsupported_dtype() {
    // BF16 is not supported by get_tensor_as_f32
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_tensor("bf16_w", TensorDType::BF16, vec![4], vec![0u8; 8]);
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let result = reader.get_tensor_as_f32("bf16_w");
    assert!(result.is_none(), "BF16 dequant should return None");
}

#[test]
fn test_reader_ref_get_tensor_as_f32_unsupported_dtype() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_tensor("i64_w", TensorDType::I64, vec![2], vec![0u8; 16]);
    let bytes = writer.write().expect("write");
    let reader = AprV2ReaderRef::from_bytes(&bytes).expect("read");

    assert!(reader.get_tensor_as_f32("i64_w").is_none());
}

#[test]
fn test_reader_get_tensor_as_f32_q8_too_short() {
    // Q8 data needs at least 4 bytes for scale
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_tensor("short_q8", TensorDType::Q8, vec![1], vec![0u8; 2]);
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    assert!(
        reader.get_tensor_as_f32("short_q8").is_none(),
        "Q8 with <4 bytes should return None"
    );
}

#[test]
fn test_reader_ref_get_tensor_as_f32_q8_too_short() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_tensor("short_q8", TensorDType::Q8, vec![1], vec![0u8; 2]);
    let bytes = writer.write().expect("write");
    let reader = AprV2ReaderRef::from_bytes(&bytes).expect("read");

    assert!(reader.get_tensor_as_f32("short_q8").is_none());
}

// ---------------------------------------------------------------------------
// ReaderRef - get_tensor_as_f32 for Q4 path
// ---------------------------------------------------------------------------

#[test]
fn test_reader_ref_get_tensor_as_f32_q4() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    // Use larger values so f16 scale exponent >= 15 (avoids debug underflow in f16_to_f32)
    let data: Vec<f32> = (0..32).map(|i| (i as f32) * 2.0 - 31.0).collect();
    writer.add_q4_tensor("q4_test", vec![32], &data);
    let bytes = writer.write().expect("write");
    let reader = AprV2ReaderRef::from_bytes(&bytes).expect("read");

    let result = reader.get_tensor_as_f32("q4_test");
    assert!(
        result.is_some(),
        "Q4 tensor via ReaderRef should dequantize"
    );
    assert_eq!(result.expect("q4").len(), 32);
}

// ---------------------------------------------------------------------------
// Writer - empty tensor paths
// ---------------------------------------------------------------------------

#[test]
fn test_writer_add_q8_empty_tensor() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_q8_tensor("empty_q8", vec![0], &[]);
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let entry = reader.get_tensor("empty_q8").expect("tensor exists");
    assert_eq!(entry.dtype, TensorDType::Q8);
    assert_eq!(entry.size, 0);
}

#[test]
fn test_writer_add_q4_empty_tensor() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_q4_tensor("empty_q4", vec![0], &[]);
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let entry = reader.get_tensor("empty_q4").expect("tensor exists");
    assert_eq!(entry.dtype, TensorDType::Q4);
    assert_eq!(entry.size, 0);
}

#[test]
fn test_writer_add_q8_all_zeros() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_q8_tensor("zero_q8", vec![4], &[0.0, 0.0, 0.0, 0.0]);
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let data = reader.get_tensor_as_f32("zero_q8").expect("dequant");
    assert_eq!(data.len(), 4);
    for &v in &data {
        assert_eq!(v, 0.0);
    }
}

// ---------------------------------------------------------------------------
// Writer - Q4 partial block (not multiple of 32)
// ---------------------------------------------------------------------------

#[test]
fn test_writer_add_q4_partial_block() {
    // 20 elements = 1 partial block (not a full 32)
    let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_q4_tensor("partial_q4", vec![20], &data);
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let entry = reader.get_tensor("partial_q4").expect("tensor");
    assert_eq!(entry.dtype, TensorDType::Q4);
    let result = reader.get_tensor_as_f32("partial_q4").expect("dequant");
    assert_eq!(result.len(), 20);
}

#[test]
fn test_writer_add_q4_odd_count() {
    // 33 elements = 1 full block + 1 partial block
    let data: Vec<f32> = (0..33).map(|i| i as f32 * 0.5 - 8.0).collect();
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_q4_tensor("odd_q4", vec![33], &data);
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let result = reader.get_tensor_as_f32("odd_q4").expect("dequant");
    assert_eq!(result.len(), 33);
}

// ---------------------------------------------------------------------------
// Writer - sorting verification with multiple tensors
// ---------------------------------------------------------------------------

#[test]
fn test_writer_sorts_tensors_alphabetically() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    // Add in reverse alphabetical order
    writer.add_f32_tensor("z_weight", vec![2], &[1.0, 2.0]);
    writer.add_f32_tensor("a_bias", vec![2], &[3.0, 4.0]);
    writer.add_f32_tensor("m_param", vec![2], &[5.0, 6.0]);

    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let names = reader.tensor_names();
    assert_eq!(names, vec!["a_bias", "m_param", "z_weight"]);
}

// ---------------------------------------------------------------------------
// Metadata - JSON error paths
// ---------------------------------------------------------------------------

#[test]
fn test_metadata_from_json_invalid() {
    let invalid_json = b"not valid json {{{";
    let result = AprV2Metadata::from_json(invalid_json);
    assert!(matches!(result, Err(V2FormatError::MetadataError(_))));
}

#[test]
fn test_shard_manifest_from_json_invalid() {
    let result = ShardManifest::from_json("not json at all");
    assert!(matches!(result, Err(V2FormatError::MetadataError(_))));
}

// ---------------------------------------------------------------------------
// Metadata - transformer config fields roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_metadata_transformer_config_roundtrip() {
    let mut metadata = AprV2Metadata::new("qwen2");
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(4096);
    metadata.num_layers = Some(32);
    metadata.num_heads = Some(32);
    metadata.num_kv_heads = Some(8);
    metadata.vocab_size = Some(152064);
    metadata.intermediate_size = Some(11008);
    metadata.max_position_embeddings = Some(8192);
    metadata.rope_theta = Some(1_000_000.0);
    metadata.rope_type = Some(2);
    metadata.rms_norm_eps = Some(1e-6);

    let json = metadata.to_json().expect("serialize");
    let parsed = AprV2Metadata::from_json(&json).expect("deserialize");

    assert_eq!(parsed.architecture, Some("qwen2".to_string()));
    assert_eq!(parsed.hidden_size, Some(4096));
    assert_eq!(parsed.num_layers, Some(32));
    assert_eq!(parsed.num_heads, Some(32));
    assert_eq!(parsed.num_kv_heads, Some(8));
    assert_eq!(parsed.vocab_size, Some(152064));
    assert_eq!(parsed.intermediate_size, Some(11008));
    assert_eq!(parsed.max_position_embeddings, Some(8192));
    // Compare f32 with tolerance
    assert!((parsed.rope_theta.expect("rope_theta") - 1_000_000.0).abs() < 1.0);
    assert_eq!(parsed.rope_type, Some(2));
    assert!((parsed.rms_norm_eps.expect("rms_norm_eps") - 1e-6).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// V2FormatError - std::error::Error trait
// ---------------------------------------------------------------------------

#[test]
fn test_v2_format_error_is_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(V2FormatError::ChecksumMismatch);
    assert_eq!(err.to_string(), "Checksum mismatch");
    // source() should return None (no underlying cause)
    assert!(err.source().is_none());
}

// ---------------------------------------------------------------------------
// Writer + Reader with F16 and Q8 via get_tensor_as_f32
// ---------------------------------------------------------------------------

#[test]
fn test_reader_get_tensor_as_f32_f32_path() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_f32_tensor("f32_w", vec![3], &[1.5, 2.5, 3.5]);
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let data = reader.get_tensor_as_f32("f32_w").expect("f32 dequant");
    assert_eq!(data.len(), 3);
    assert!((data[0] - 1.5).abs() < 1e-6);
    assert!((data[1] - 2.5).abs() < 1e-6);
    assert!((data[2] - 3.5).abs() < 1e-6);
}

#[test]
fn test_reader_get_tensor_as_f32_f16_path() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_f16_tensor("f16_w", vec![3], &[1.0, 2.0, 3.0]);
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let data = reader.get_tensor_as_f32("f16_w").expect("f16 dequant");
    assert_eq!(data.len(), 3);
    assert!((data[0] - 1.0).abs() < 0.01);
    assert!((data[1] - 2.0).abs() < 0.01);
}

#[test]
fn test_reader_get_tensor_as_f32_q8_path() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_q8_tensor("q8_w", vec![3], &[1.0, -2.0, 3.0]);
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let data = reader.get_tensor_as_f32("q8_w").expect("q8 dequant");
    assert_eq!(data.len(), 3);
    // Q8 should be close to original
    assert!((data[0] - 1.0).abs() < 0.1);
}

#[test]
fn test_reader_get_tensor_as_f32_q4_path() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    let original: Vec<f32> = (0..32).map(|i| (i as f32) - 16.0).collect();
    writer.add_q4_tensor("q4_w", vec![32], &original);
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    let data = reader.get_tensor_as_f32("q4_w").expect("q4 dequant");
    assert_eq!(data.len(), 32);
}

// ---------------------------------------------------------------------------
// ReaderRef - get_tensor_as_f32 for F32/F16/Q8 paths
// ---------------------------------------------------------------------------

#[test]
fn test_reader_ref_get_tensor_as_f32_f32_path() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_f32_tensor("w", vec![2], &[10.0, 20.0]);
    let bytes = writer.write().expect("write");
    let reader = AprV2ReaderRef::from_bytes(&bytes).expect("read");

    let data = reader.get_tensor_as_f32("w").expect("f32");
    assert_eq!(data.len(), 2);
    assert!((data[0] - 10.0).abs() < 1e-6);
}

#[test]
fn test_reader_ref_get_tensor_as_f32_f16_path() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_f16_tensor("w", vec![2], &[5.0, 10.0]);
    let bytes = writer.write().expect("write");
    let reader = AprV2ReaderRef::from_bytes(&bytes).expect("read");

    let data = reader.get_tensor_as_f32("w").expect("f16");
    assert_eq!(data.len(), 2);
    assert!((data[0] - 5.0).abs() < 0.05);
}

#[test]
fn test_reader_ref_get_tensor_as_f32_q8_path() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_q8_tensor("w", vec![4], &[1.0, -1.0, 0.5, -0.5]);
    let bytes = writer.write().expect("write");
    let reader = AprV2ReaderRef::from_bytes(&bytes).expect("read");

    let data = reader.get_tensor_as_f32("w").expect("q8");
    assert_eq!(data.len(), 4);
}

// ---------------------------------------------------------------------------
// ReaderRef - nonexistent tensor and data bounds
// ---------------------------------------------------------------------------

#[test]
fn test_reader_ref_get_tensor_nonexistent() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_f32_tensor("w", vec![2], &[1.0, 2.0]);
    let bytes = writer.write().expect("write");
    let reader = AprV2ReaderRef::from_bytes(&bytes).expect("read");

    assert!(reader.get_tensor("nope").is_none());
    assert!(reader.get_tensor_data("nope").is_none());
    assert!(reader.get_f32_tensor("nope").is_none());
    assert!(reader.get_tensor_as_f32("nope").is_none());
}

// ---------------------------------------------------------------------------
// Writer flags: verify LZ4 and sharding flags are set
// ---------------------------------------------------------------------------

#[test]
fn test_writer_lz4_flag_in_output() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.with_lz4_compression();
    writer.add_f32_tensor("w", vec![2], &[1.0, 2.0]);
    let bytes = writer.write().expect("write");
    let header = AprV2Header::from_bytes(&bytes).expect("parse header");
    assert!(header.flags.is_lz4_compressed());
}

#[test]
fn test_writer_sharding_flag_and_metadata() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.with_sharding(3, 1);
    writer.add_f32_tensor("w", vec![2], &[1.0, 2.0]);
    let bytes = writer.write().expect("write");

    let reader = AprV2Reader::from_bytes(&bytes).expect("read");
    assert!(reader.header().flags.is_sharded());
    let shard = reader
        .metadata()
        .sharding
        .as_ref()
        .expect("sharding metadata");
    assert_eq!(shard.shard_count, 3);
    assert_eq!(shard.shard_index, 1);
}

// ---------------------------------------------------------------------------
// Header - is_valid for corrupted magic
// ---------------------------------------------------------------------------

#[test]
fn test_header_is_valid_false() {
    let mut header = AprV2Header::new();
    header.magic = [0xFF, 0xFF, 0xFF, 0xFF];
    assert!(!header.is_valid());
}

// ---------------------------------------------------------------------------
// Flags - from_bits and bits roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_flags_bits_roundtrip() {
    let original = 0b0000_0101_0011_1111u16;
    let flags = AprV2Flags::from_bits(original);
    assert_eq!(flags.bits(), original);
}

#[test]
fn test_flags_default_is_empty() {
    let flags = AprV2Flags::default();
    assert_eq!(flags.bits(), 0);
    assert!(!flags.is_lz4_compressed());
    assert!(!flags.is_encrypted());
    assert!(!flags.is_sharded());
    assert!(!flags.is_quantized());
    assert!(!flags.is_zstd_compressed());
    assert!(!flags.is_row_major());
    assert!(!flags.is_column_major());
    assert!(flags.is_layout_valid());
}

// ---------------------------------------------------------------------------
// TensorDType - additional coverage for Q4, Q8, Q4K, Q6K
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_dtype_from_u8_q4_q8_q4k_q6k() {
    assert_eq!(TensorDType::from_u8(8), Some(TensorDType::Q4));
    assert_eq!(TensorDType::from_u8(9), Some(TensorDType::Q8));
    assert_eq!(TensorDType::from_u8(12), Some(TensorDType::Q4K));
    assert_eq!(TensorDType::from_u8(14), Some(TensorDType::Q6K));
}

#[test]
fn test_tensor_dtype_names_complete() {
    assert_eq!(TensorDType::F64.name(), "f64");
    assert_eq!(TensorDType::I32.name(), "i32");
    assert_eq!(TensorDType::I64.name(), "i64");
    assert_eq!(TensorDType::I8.name(), "i8");
    assert_eq!(TensorDType::U8.name(), "u8");
    assert_eq!(TensorDType::Q4.name(), "q4");
    assert_eq!(TensorDType::Q4K.name(), "q4_k");
    assert_eq!(TensorDType::Q6K.name(), "q6_k");
}

#[test]
fn test_tensor_dtype_bytes_per_element_complete() {
    assert_eq!(TensorDType::F32.bytes_per_element(), 4);
    assert_eq!(TensorDType::I32.bytes_per_element(), 4);
    assert_eq!(TensorDType::F16.bytes_per_element(), 2);
    assert_eq!(TensorDType::BF16.bytes_per_element(), 2);
    assert_eq!(TensorDType::F64.bytes_per_element(), 8);
    assert_eq!(TensorDType::I64.bytes_per_element(), 8);
    assert_eq!(TensorDType::I8.bytes_per_element(), 1);
    assert_eq!(TensorDType::U8.bytes_per_element(), 1);
    assert_eq!(TensorDType::Q8.bytes_per_element(), 1);
    assert_eq!(TensorDType::Q4.bytes_per_element(), 0);
    assert_eq!(TensorDType::Q4K.bytes_per_element(), 0);
    assert_eq!(TensorDType::Q6K.bytes_per_element(), 0);
}

// ---------------------------------------------------------------------------
// Header serialization field coverage
// ---------------------------------------------------------------------------

#[test]
fn test_header_to_bytes_all_fields() {
    let mut header = AprV2Header::new();
    header.tensor_count = 42;
    header.metadata_offset = 128;
    header.metadata_size = 256;
    header.tensor_index_offset = 512;
    header.data_offset = 1024;
    header.flags = AprV2Flags::from_bits(0xABCD);
    header.reserved = [0xFF; 20];
    header.update_checksum();

    let bytes = header.to_bytes();
    let parsed = AprV2Header::from_bytes(&bytes).expect("parse");

    assert_eq!(parsed.tensor_count, 42);
    assert_eq!(parsed.metadata_offset, 128);
    assert_eq!(parsed.metadata_size, 256);
    assert_eq!(parsed.tensor_index_offset, 512);
    assert_eq!(parsed.data_offset, 1024);
    assert_eq!(parsed.flags.bits(), 0xABCD);
    assert_eq!(parsed.reserved, [0xFF; 20]);
    assert!(parsed.verify_checksum());
}

// ---------------------------------------------------------------------------
// Reader from_reader with io error (empty reader)
// ---------------------------------------------------------------------------

#[test]
fn test_reader_from_reader_empty() {
    let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
    let result = AprV2Reader::from_reader(&mut cursor);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Metadata - ChatSpecialTokens full field coverage
// ---------------------------------------------------------------------------

#[test]
fn test_chat_special_tokens_all_fields() {
    let tokens = ChatSpecialTokens {
        bos_token: Some("<s>".to_string()),
        eos_token: Some("</s>".to_string()),
        unk_token: Some("<unk>".to_string()),
        pad_token: Some("<pad>".to_string()),
        im_start_token: Some("<|im_start|>".to_string()),
        im_end_token: Some("<|im_end|>".to_string()),
    };

    let mut metadata = AprV2Metadata::new("test");
    metadata.special_tokens = Some(tokens);

    let json = metadata.to_json().expect("serialize");
    let parsed = AprV2Metadata::from_json(&json).expect("deserialize");

    let t = parsed.special_tokens.expect("tokens");
    assert_eq!(t.bos_token.as_deref(), Some("<s>"));
    assert_eq!(t.eos_token.as_deref(), Some("</s>"));
    assert_eq!(t.unk_token.as_deref(), Some("<unk>"));
    assert_eq!(t.pad_token.as_deref(), Some("<pad>"));
    assert_eq!(t.im_start_token.as_deref(), Some("<|im_start|>"));
    assert_eq!(t.im_end_token.as_deref(), Some("<|im_end|>"));
}

// ---------------------------------------------------------------------------
// Metadata - all optional fields present
// ---------------------------------------------------------------------------

#[test]
fn test_metadata_all_fields_roundtrip() {
    let mut metadata = AprV2Metadata::new("full_test");
    metadata.name = Some("test-model".to_string());
    metadata.description = Some("A test model".to_string());
    metadata.author = Some("Test Author".to_string());
    metadata.license = Some("MIT".to_string());
    metadata.version = Some("1.0.0".to_string());
    metadata.source = Some("hf://test/model".to_string());
    metadata.original_format = Some("safetensors".to_string());
    metadata.created_at = Some("2025-01-01T00:00:00Z".to_string());
    metadata.total_size = 1_000_000;
    metadata.param_count = 500_000;

    let json = metadata.to_json().expect("serialize");
    let parsed = AprV2Metadata::from_json(&json).expect("deserialize");

    assert_eq!(parsed.model_type, "full_test");
    assert_eq!(parsed.name.as_deref(), Some("test-model"));
    assert_eq!(parsed.description.as_deref(), Some("A test model"));
    assert_eq!(parsed.author.as_deref(), Some("Test Author"));
    assert_eq!(parsed.license.as_deref(), Some("MIT"));
    assert_eq!(parsed.version.as_deref(), Some("1.0.0"));
    assert_eq!(parsed.source.as_deref(), Some("hf://test/model"));
    assert_eq!(parsed.original_format.as_deref(), Some("safetensors"));
    assert_eq!(parsed.created_at.as_deref(), Some("2025-01-01T00:00:00Z"));
    assert_eq!(parsed.total_size, 1_000_000);
    assert_eq!(parsed.param_count, 500_000);
}

// ---------------------------------------------------------------------------
// QuantizationMetadata - default and full
// ---------------------------------------------------------------------------

#[test]
fn test_quantization_metadata_default() {
    let q = QuantizationMetadata::default();
    assert_eq!(q.quant_type, "");
    assert_eq!(q.bits, 0);
    assert!(q.block_size.is_none());
    assert!(!q.symmetric);
}

// ---------------------------------------------------------------------------
// ShardingMetadata - default
// ---------------------------------------------------------------------------

#[test]
fn test_sharding_metadata_default() {
    let s = ShardingMetadata::default();
    assert_eq!(s.shard_count, 0);
    assert_eq!(s.shard_index, 0);
    assert_eq!(s.total_size, 0);
    assert!(s.pattern.is_none());
}

// ---------------------------------------------------------------------------
// Writer write_to with error (using a write-limited writer)
// ---------------------------------------------------------------------------

#[test]
fn test_writer_write_to_io_error() {
    struct FailWriter;
    impl std::io::Write for FailWriter {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "forced error",
            ))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_f32_tensor("w", vec![2], &[1.0, 2.0]);
    let result = writer.write_to(&mut FailWriter);
    assert!(matches!(result, Err(V2FormatError::IoError(_))));
}

// ---------------------------------------------------------------------------
// Verify alignment of data_offset in written files
// ---------------------------------------------------------------------------

#[test]
fn test_written_data_offset_is_aligned() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_f32_tensor("weight_a", vec![100], &vec![1.0f32; 100]);
    writer.add_f32_tensor("weight_b", vec![50], &vec![2.0f32; 50]);
    let bytes = writer.write().expect("write");

    let reader = AprV2Reader::from_bytes(&bytes).expect("read");
    assert!(is_aligned_64(reader.header().data_offset as usize));
    assert!(is_aligned_64(reader.header().metadata_offset as usize));
    assert!(is_aligned_64(reader.header().tensor_index_offset as usize));
    assert!(reader.verify_alignment());
}

// ---------------------------------------------------------------------------
// Header roundtrip preserves all version and flag combos
// ---------------------------------------------------------------------------

#[test]
fn test_header_roundtrip_varied_versions() {
    for major in [0u8, 1, 2, 3, 255] {
        for minor in [0u8, 1, 99, 255] {
            let mut header = AprV2Header::new();
            header.version = (major, minor);
            header.update_checksum();
            let bytes = header.to_bytes();
            let parsed = AprV2Header::from_bytes(&bytes).expect("parse");
            assert_eq!(parsed.version, (major, minor));
            assert!(parsed.verify_checksum());
        }
    }
}

// ---------------------------------------------------------------------------
// Writer with no tensors at all
// ---------------------------------------------------------------------------

#[test]
fn test_writer_no_tensors() {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("empty_model"));
    let bytes = writer.write().expect("write");
    let reader = AprV2Reader::from_bytes(&bytes).expect("read");

    assert_eq!(reader.tensor_names().len(), 0);
    assert_eq!(reader.header().tensor_count, 0);
    assert!(reader.verify_alignment());
}

// ---------------------------------------------------------------------------
// Flags - HAS_FILTERBANK, HAS_MODEL_CARD, STREAMING, SIGNED
// ---------------------------------------------------------------------------

#[test]
fn test_flags_filterbank_and_model_card() {
    let flags = AprV2Flags::new()
        .with(AprV2Flags::HAS_FILTERBANK)
        .with(AprV2Flags::HAS_MODEL_CARD)
        .with(AprV2Flags::STREAMING)
        .with(AprV2Flags::SIGNED);

    assert!(flags.contains(AprV2Flags::HAS_FILTERBANK));
    assert!(flags.contains(AprV2Flags::HAS_MODEL_CARD));
    assert!(flags.contains(AprV2Flags::STREAMING));
    assert!(flags.contains(AprV2Flags::SIGNED));

    let without = flags
        .without(AprV2Flags::HAS_FILTERBANK)
        .without(AprV2Flags::STREAMING);
    assert!(!without.contains(AprV2Flags::HAS_FILTERBANK));
    assert!(!without.contains(AprV2Flags::STREAMING));
    assert!(without.contains(AprV2Flags::HAS_MODEL_CARD));
    assert!(without.contains(AprV2Flags::SIGNED));
}

// ---------------------------------------------------------------------------
// Dequantize Q4 - valid scale with known nibbles for value verification
// ---------------------------------------------------------------------------

#[test]
fn test_dequantize_q4_known_values() {
    // scale = 1.0 (f16 0x3C00), all nibbles = 0x08 (unsigned 8 -> signed 0)
    let mut data = vec![0u8; 18];
    data[0] = 0x00;
    data[1] = 0x3C; // f16 1.0
                    // Nibbles: each byte = 0x88 -> low nibble = 8, high nibble = 8
                    // q = 8 - 8 = 0, so all values should be ~0.0
    for i in 2..18 {
        data[i] = 0x88;
    }
    let result = super::dequantize_q4(&data, 32);
    assert_eq!(result.len(), 32);
    for &v in &result {
        assert!(
            v.abs() < 0.01,
            "Nibble 8 (signed 0) * scale 1.0 should be ~0, got {v}"
        );
    }
}

#[test]
fn test_dequantize_q4_nonzero_nibbles() {
    // scale = 2.0 (f16 0x4000), nibble = 0x0F (unsigned 15 -> signed 7)
    let mut data = vec![0u8; 18];
    data[0] = 0x00;
    data[1] = 0x40; // f16 2.0
                    // Low nibble = 0x0F = 15, high nibble = 0x0F = 15
    for i in 2..18 {
        data[i] = 0xFF;
    }
    let result = super::dequantize_q4(&data, 32);
    assert_eq!(result.len(), 32);
    // q = 15 - 8 = 7, value = 7 * 2.0 = 14.0
    for &v in &result {
        assert!((v - 14.0).abs() < 0.5, "Expected ~14.0, got {v}");
    }
}
