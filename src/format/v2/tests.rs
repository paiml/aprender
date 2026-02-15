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

include!("tests_part_02.rs");
include!("tests_part_03.rs");
include!("tests_part_04.rs");
