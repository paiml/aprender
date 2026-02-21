use super::*;

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
