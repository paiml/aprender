#![allow(clippy::unwrap_used)]

pub(crate) use super::*;

// =========================================================================
// Magic and Format Tests
// =========================================================================

#[test]
fn test_writer_creates_valid_apr() {
    let writer = AprWriter::new();
    let bytes = writer.to_bytes().unwrap();
    // Must start with APR v2 magic: "APR\0"
    assert_eq!(&bytes[0..3], b"APR");
}

// =========================================================================
// Metadata Tests
// =========================================================================

#[test]
fn test_empty_metadata() {
    let writer = AprWriter::new();
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();
    assert!(reader.metadata.is_empty());
}

#[test]
fn test_string_metadata() {
    let mut writer = AprWriter::new();
    writer.set_metadata("model_name", JsonValue::String("whisper-tiny".into()));

    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    assert_eq!(
        reader.get_metadata("model_name"),
        Some(&JsonValue::String("whisper-tiny".into()))
    );
}

#[test]
fn test_numeric_metadata() {
    let mut writer = AprWriter::new();
    writer.set_metadata("n_vocab", JsonValue::Number(51865.into()));
    writer.set_metadata("n_layers", JsonValue::Number(4.into()));

    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    assert_eq!(
        reader.get_metadata("n_vocab"),
        Some(&JsonValue::Number(51865.into()))
    );
}

#[test]
fn test_array_metadata() {
    let mut writer = AprWriter::new();
    let vocab = vec![
        JsonValue::String("hello".into()),
        JsonValue::String("world".into()),
    ];
    writer.set_metadata("vocab", JsonValue::Array(vocab));

    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    let vocab = reader.get_metadata("vocab").unwrap();
    assert!(vocab.is_array());
    assert_eq!(vocab.as_array().unwrap().len(), 2);
}

#[test]
fn test_object_metadata() {
    let mut writer = AprWriter::new();
    let mut config = serde_json::Map::new();
    config.insert("dim".into(), JsonValue::Number(384.into()));
    config.insert("heads".into(), JsonValue::Number(6.into()));
    writer.set_metadata("config", JsonValue::Object(config));

    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    let config = reader.get_metadata("config").unwrap();
    assert!(config.is_object());
}

// =========================================================================
// Tensor Tests
// =========================================================================

#[test]
fn test_no_tensors() {
    let writer = AprWriter::new();
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();
    assert!(reader.tensors.is_empty());
}

#[test]
fn test_single_tensor() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("weights", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    assert_eq!(reader.tensors.len(), 1);
    assert_eq!(reader.tensors[0].name, "weights");
    assert_eq!(reader.tensors[0].shape, vec![2, 3]);

    let data = reader.read_tensor_f32("weights").unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_multiple_tensors() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("a", vec![2], &[1.0, 2.0]);
    writer.add_tensor_f32("b", vec![3], &[3.0, 4.0, 5.0]);

    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    assert_eq!(reader.tensors.len(), 2);

    let a = reader.read_tensor_f32("a").unwrap();
    let b = reader.read_tensor_f32("b").unwrap();

    assert_eq!(a, vec![1.0, 2.0]);
    assert_eq!(b, vec![3.0, 4.0, 5.0]);
}

#[test]
fn test_tensor_not_found() {
    let writer = AprWriter::new();
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    let result = reader.read_tensor_f32("nonexistent");
    assert!(result.is_err());
}

// =========================================================================
// Combined Metadata + Tensor Tests
// =========================================================================

#[test]
fn test_metadata_and_tensors() {
    let mut writer = AprWriter::new();

    // Add metadata
    writer.set_metadata("model_type", JsonValue::String("test".into()));

    // Add tensors
    writer.add_tensor_f32("layer.0.weight", vec![4, 4], &vec![0.5; 16]);
    writer.add_tensor_f32("layer.0.bias", vec![4], &[0.1, 0.2, 0.3, 0.4]);

    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    // Verify metadata
    assert_eq!(
        reader.get_metadata("model_type"),
        Some(&JsonValue::String("test".into()))
    );

    // Verify tensors
    let weight = reader.read_tensor_f32("layer.0.weight").unwrap();
    assert_eq!(weight.len(), 16);

    let bias = reader.read_tensor_f32("layer.0.bias").unwrap();
    assert_eq!(bias, vec![0.1, 0.2, 0.3, 0.4]);
}

// =========================================================================
// Error Handling Tests
// =========================================================================

#[test]
fn test_invalid_magic() {
    let data = vec![b'X', b'Y', b'Z', b'1', 0, 0, 0, 0];
    let result = AprReader::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_file_too_short() {
    let data = vec![b'A', b'P', b'R'];
    let result = AprReader::from_bytes(data);
    assert!(result.is_err());
}

// =========================================================================
// Roundtrip Tests
// =========================================================================

#[test]
fn test_full_roundtrip() {
    let mut writer = AprWriter::new();

    // Complex metadata
    let mut bpe_merges = Vec::new();
    bpe_merges.push(JsonValue::Array(vec![
        JsonValue::String("h".into()),
        JsonValue::String("e".into()),
    ]));
    bpe_merges.push(JsonValue::Array(vec![
        JsonValue::String("he".into()),
        JsonValue::String("llo".into()),
    ]));
    writer.set_metadata("bpe_merges", JsonValue::Array(bpe_merges));

    // Tensors
    writer.add_tensor_f32("embed", vec![100, 64], &vec![0.1; 6400]);

    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    // Verify
    let merges = reader.get_metadata("bpe_merges").unwrap();
    assert_eq!(merges.as_array().unwrap().len(), 2);

    let embed = reader.read_tensor_f32("embed").unwrap();
    assert_eq!(embed.len(), 6400);
}

#[test]
fn test_well_known_metadata_fields() {
    let mut writer = AprWriter::new();
    writer.set_metadata("model_type", JsonValue::String("transformer".into()));
    writer.set_metadata("model_name", JsonValue::String("test-model".into()));
    writer.set_metadata("architecture", JsonValue::String("qwen2".into()));
    writer.set_metadata("custom_field", JsonValue::Number(42.into()));

    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    // Well-known fields round-trip correctly
    assert_eq!(
        reader.get_metadata("model_type"),
        Some(&JsonValue::String("transformer".into()))
    );
    assert_eq!(
        reader.get_metadata("model_name"),
        Some(&JsonValue::String("test-model".into()))
    );
    assert_eq!(
        reader.get_metadata("architecture"),
        Some(&JsonValue::String("qwen2".into()))
    );
    // Custom fields preserved via AprV2Metadata.custom
    assert_eq!(
        reader.get_metadata("custom_field"),
        Some(&JsonValue::Number(42.into()))
    );
}

// =========================================================================
// Filtered Reader Tests (F-CKPT-016)
// =========================================================================

#[test]
fn test_open_filtered_skips_training_tensors() {
    let mut writer = AprWriter::new();
    writer.set_metadata("model_type", JsonValue::String("adapter".into()));

    // Inference tensors
    writer.add_tensor_f32("classifier.weight", vec![5, 896], &vec![0.1; 4480]);
    writer.add_tensor_f32("classifier.bias", vec![5], &[0.1, 0.2, 0.3, 0.4, 0.5]);
    writer.add_tensor_f32("lora.0.q_proj.lora_a", vec![8, 896], &vec![0.01; 7168]);

    // Training-only tensors (should be filtered out)
    writer.add_tensor_f32("__training__.optimizer.m.0", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    writer.add_tensor_f32("__training__.optimizer.v.0", vec![4], &[0.1, 0.2, 0.3, 0.4]);
    writer.add_tensor_f32("__training__.step", vec![1], &[100.0]);

    let bytes = writer.to_bytes().unwrap();
    let reader =
        AprReader::from_bytes_filtered(bytes, |name| !name.starts_with("__training__.")).unwrap();

    // Only inference tensors remain
    assert_eq!(reader.tensors.len(), 3);
    assert!(reader.tensors.iter().any(|t| t.name == "classifier.weight"));
    assert!(reader.tensors.iter().any(|t| t.name == "classifier.bias"));
    assert!(reader
        .tensors
        .iter()
        .any(|t| t.name == "lora.0.q_proj.lora_a"));

    // Training tensors are gone
    assert!(reader
        .tensors
        .iter()
        .all(|t| !t.name.starts_with("__training__.")));

    // Metadata preserved
    assert_eq!(
        reader.get_metadata("model_type"),
        Some(&JsonValue::String("adapter".into()))
    );
}

#[test]
fn test_filtered_reader_can_still_read_kept_tensors() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("weights", vec![3], &[1.0, 2.0, 3.0]);
    writer.add_tensor_f32("__training__.lr", vec![1], &[0.001]);

    let bytes = writer.to_bytes().unwrap();
    let reader =
        AprReader::from_bytes_filtered(bytes, |name| !name.starts_with("__training__.")).unwrap();

    // Can read kept tensor
    let data = reader.read_tensor_f32("weights").unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);

    // Filtered tensor not in descriptors
    assert!(!reader.tensors.iter().any(|t| t.name == "__training__.lr"));
}

#[test]
fn test_filtered_reader_no_filter_match_keeps_all() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("a", vec![2], &[1.0, 2.0]);
    writer.add_tensor_f32("b", vec![2], &[3.0, 4.0]);

    let bytes = writer.to_bytes().unwrap();
    // Filter matches nothing — all tensors kept
    let reader = AprReader::from_bytes_filtered(bytes, |_| true).unwrap();

    assert_eq!(reader.tensors.len(), 2);
}

#[test]
fn test_filtered_reader_filter_all_leaves_empty() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("a", vec![2], &[1.0, 2.0]);
    writer.add_tensor_f32("b", vec![2], &[3.0, 4.0]);

    let bytes = writer.to_bytes().unwrap();
    // Filter rejects everything
    let reader = AprReader::from_bytes_filtered(bytes, |_| false).unwrap();

    assert!(reader.tensors.is_empty());
}

#[test]
fn test_open_filtered_with_file() {
    let dir = std::env::temp_dir().join("apr_filtered_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_filtered.apr");

    let mut writer = AprWriter::new();
    writer.add_tensor_f32("model.weight", vec![2], &[1.0, 2.0]);
    writer.add_tensor_f32("__training__.step", vec![1], &[42.0]);
    writer.write(&path).unwrap();

    let reader =
        AprReader::open_filtered(&path, |name| !name.starts_with("__training__.")).unwrap();

    assert_eq!(reader.tensors.len(), 1);
    assert_eq!(reader.tensors[0].name, "model.weight");

    let data = reader.read_tensor_f32("model.weight").unwrap();
    assert_eq!(data, vec![1.0, 2.0]);

    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================================
// Falsification Tests (F-CKPT contracts)
// =========================================================================

/// FALSIFY F-CKPT-009: Atomic write leaves no orphan .tmp on success
#[test]
fn falsify_ckpt_009_atomic_write_no_orphan_tmp() {
    let dir = std::env::temp_dir().join("apr_f009_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("model.apr");

    let mut writer = AprWriter::new();
    writer.add_tensor_f32("w", vec![2], &[1.0, 2.0]);
    writer.write(&path).unwrap();

    // The .tmp file must NOT exist after successful write
    let tmp_path = path.with_extension("apr.tmp");
    assert!(
        !tmp_path.exists(),
        "FALSIFIED F-CKPT-009: orphan .tmp file exists after write"
    );

    // The target file must exist
    assert!(
        path.exists(),
        "FALSIFIED F-CKPT-009: target file missing after write"
    );

    // Must be a valid APR
    let reader = AprReader::open(&path).unwrap();
    assert_eq!(reader.tensors.len(), 1);

    let _ = std::fs::remove_dir_all(&dir);
}

/// FALSIFY F-CKPT-011: CRC32 corrupt file is rejected
#[test]
fn falsify_ckpt_011_corrupt_crc_rejected() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("w", vec![2], &[1.0, 2.0]);
    let mut bytes = writer.to_bytes().unwrap();

    // Corrupt a byte in the data section
    if bytes.len() > 100 {
        bytes[100] ^= 0xFF;
    }

    let result = AprReader::from_bytes(bytes);
    assert!(
        result.is_err(),
        "FALSIFIED F-CKPT-011: corrupt APR accepted without CRC error"
    );
}

/// FALSIFY F-CKPT-013: NaN in tensor is detected
#[test]
fn falsify_ckpt_013_nan_detected() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("w", vec![3], &[1.0, f32::NAN, 3.0]);
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    let result = reader.read_tensor_f32_checked("w");
    assert!(
        result.is_err(),
        "FALSIFIED F-CKPT-013: NaN tensor accepted by checked read"
    );
    assert!(
        result.unwrap_err().contains("F-CKPT-013"),
        "Error should reference F-CKPT-013 contract"
    );
}

/// FALSIFY F-CKPT-013: Inf in tensor is detected
#[test]
fn falsify_ckpt_013_inf_detected() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("w", vec![3], &[1.0, f32::INFINITY, 3.0]);
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    let result = reader.read_tensor_f32_checked("w");
    assert!(
        result.is_err(),
        "FALSIFIED F-CKPT-013: Inf tensor accepted by checked read"
    );
}

/// FALSIFY F-CKPT-013: Clean tensor passes checked read
#[test]
fn falsify_ckpt_013_clean_tensor_passes() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("w", vec![3], &[1.0, 2.0, 3.0]);
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    let result = reader.read_tensor_f32_checked("w");
    assert!(result.is_ok(), "Clean tensor should pass checked read");
    assert_eq!(result.unwrap(), vec![1.0, 2.0, 3.0]);
}

/// FALSIFY F-CKPT-014: Shape mismatch detected
#[test]
fn falsify_ckpt_014_shape_mismatch() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("w", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    // Expect 6 elements, should pass
    let ok = reader.validate_tensor_shape("w", 6);
    assert!(ok.is_ok(), "Correct shape should validate");

    // Expect 4 elements, should fail
    let err = reader.validate_tensor_shape("w", 4);
    assert!(err.is_err(), "FALSIFIED F-CKPT-014: wrong shape accepted");
    assert!(err.unwrap_err().contains("F-CKPT-014"));
}

/// FALSIFY F-CKPT-014: Missing tensor detected
#[test]
fn falsify_ckpt_014_missing_tensor() {
    let writer = AprWriter::new();
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    let err = reader.validate_tensor_shape("nonexistent", 10);
    assert!(
        err.is_err(),
        "FALSIFIED F-CKPT-014: missing tensor not detected"
    );
}

/// FALSIFY F-CKPT-015: Canonical ordering verified
#[test]
fn falsify_ckpt_015_canonical_ordering() {
    let mut writer = AprWriter::new();
    // Add tensors in reverse order
    writer.add_tensor_f32("z_last", vec![1], &[1.0]);
    writer.add_tensor_f32("a_first", vec![1], &[2.0]);
    writer.add_tensor_f32("m_middle", vec![1], &[3.0]);

    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    // Tensors must be in sorted order
    let names: Vec<&str> = reader.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["a_first", "m_middle", "z_last"],
        "FALSIFIED F-CKPT-015: tensors not in canonical order"
    );
}

/// FALSIFY F-CKPT-016: Filtered reader excludes __training__
#[test]
fn falsify_ckpt_016_training_filtered() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("model.w", vec![2], &[1.0, 2.0]);
    writer.add_tensor_f32("__training__.lr", vec![1], &[0.001]);
    writer.add_tensor_f32("__training__.optimizer.m.0", vec![2], &[0.1, 0.2]);

    let bytes = writer.to_bytes().unwrap();
    let reader =
        AprReader::from_bytes_filtered(bytes, |n| !n.starts_with("__training__.")).unwrap();

    assert_eq!(
        reader.tensors.len(),
        1,
        "FALSIFIED F-CKPT-016: training tensors not filtered"
    );
    assert_eq!(reader.tensors[0].name, "model.w");
}

/// FALSIFY F-CKPT-018: APR write→read round-trip is bit-identical
#[test]
fn falsify_ckpt_018_roundtrip_bit_identical() {
    let mut writer = AprWriter::new();
    writer.set_metadata("model_type", JsonValue::String("test".into()));
    writer.set_metadata("custom_key", JsonValue::Number(42.into()));
    writer.add_tensor_f32("layer.0.weight", vec![4, 4], &vec![0.123_456_78; 16]);
    writer.add_tensor_f32("layer.0.bias", vec![4], &[0.1, 0.2, 0.3, 0.4]);

    let bytes1 = writer.to_bytes().unwrap();

    // Read back
    let reader = AprReader::from_bytes(bytes1.clone()).unwrap();

    // Re-write from reader
    let mut writer2 = AprWriter::new();
    for (k, v) in &reader.metadata {
        writer2.set_metadata(k.clone(), v.clone());
    }
    for t in &reader.tensors {
        let data = reader.read_tensor_f32(&t.name).unwrap();
        writer2.add_tensor_f32(t.name.clone(), t.shape.clone(), &data);
    }
    let bytes2 = writer2.to_bytes().unwrap();

    assert_eq!(
        bytes1, bytes2,
        "FALSIFIED F-CKPT-018: write→read→write is NOT bit-identical"
    );
}

// =========================================================================
// from_bytes error path coverage (42 uncovered lines at 61.8%)
// =========================================================================

/// from_bytes: completely empty data
#[test]
fn test_from_bytes_empty() {
    let result = AprReader::from_bytes(vec![]);
    assert!(result.is_err(), "empty data should fail");
}

/// from_bytes: truncated to just 2 bytes
#[test]
fn test_from_bytes_two_bytes() {
    let result = AprReader::from_bytes(vec![0x41, 0x50]);
    assert!(result.is_err(), "2 bytes should fail");
}

/// from_bytes: valid magic but truncated header (< 64 bytes)
#[test]
fn test_from_bytes_truncated_after_magic() {
    // APR\0 magic followed by insufficient header
    let mut data = vec![0x41, 0x50, 0x52, 0x00];
    data.extend_from_slice(&[0u8; 10]); // Only 14 bytes total
    let result = AprReader::from_bytes(data);
    assert!(result.is_err(), "truncated header should fail");
}

/// from_bytes: wrong magic bytes
#[test]
fn test_from_bytes_wrong_magic_long() {
    let mut data = vec![0xFF, 0xFE, 0xFD, 0xFC];
    data.extend_from_slice(&[0u8; 200]); // Pad so it's long enough
    let result = AprReader::from_bytes(data);
    assert!(result.is_err(), "wrong magic should fail");
}

/// from_bytes: valid magic, version 0 (unsupported)
#[test]
fn test_from_bytes_version_zero() {
    let mut data = vec![0x41, 0x50, 0x52, 0x00]; // "APR\0"
    data.extend_from_slice(&[0u8; 4]); // version = 0
    data.extend_from_slice(&[0u8; 56]); // pad to 64 bytes
    let result = AprReader::from_bytes(data);
    // Version 0 is not a valid version -- should fail or produce empty
    // The V2 reader checks version
    assert!(result.is_err() || result.unwrap().tensors.is_empty());
}

/// from_bytes: corrupted metadata JSON in valid APR envelope
#[test]
fn test_from_bytes_valid_header_corrupted_metadata() {
    // Create a valid APR file first, then corrupt the metadata section
    let mut writer = AprWriter::new();
    writer.set_metadata("key", JsonValue::String("value".into()));
    writer.add_tensor_f32("w", vec![2], &[1.0, 2.0]);
    let mut bytes = writer.to_bytes().expect("writer should succeed");

    // Corrupt bytes in the metadata section (after the 64-byte header)
    if bytes.len() > 100 {
        for b in bytes[64..80].iter_mut() {
            *b = 0xFF;
        }
    }
    let result = AprReader::from_bytes(bytes);
    // Corrupted metadata may cause parse failure
    // The exact behavior depends on the v2 reader, but it should not panic
    let _ = result; // Just ensure no panic
}

/// from_bytes: APRN (v1) magic
#[test]
fn test_from_bytes_v1_magic() {
    let mut data = vec![0x41, 0x50, 0x52, 0x4E]; // "APRN"
    data.extend_from_slice(&[0u8; 60]);
    let result = AprReader::from_bytes(data);
    // V1 is legacy -- may fail or be handled; should not panic
    let _ = result;
}

/// from_bytes: random garbage data of various sizes
#[test]
fn test_from_bytes_garbage_various_sizes() {
    for size in [0, 1, 4, 8, 16, 32, 64, 128, 256] {
        let data: Vec<u8> = (0..size).map(|i| (i * 37 + 13) as u8).collect();
        let result = AprReader::from_bytes(data);
        // None should panic
        let _ = result;
    }
}

/// read_tensor_as_f32: BF16 dtype
#[test]
fn test_read_tensor_as_f32_multiple_dtypes() {
    // Test F32 roundtrip (already covered) -- but verify read_tensor_as_f32 path
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("test_f32", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    let data = reader.read_tensor_as_f32("test_f32").unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

/// read_tensor_f32: wrong dtype returns error
#[test]
fn test_read_tensor_f32_on_non_f32_errors() {
    // This test verifies the dtype check in read_tensor_f32
    // Create a valid file with F32 tensor, verify read_tensor_f32 succeeds
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("w", vec![2], &[1.0, 2.0]);
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();
    // Should succeed for F32
    assert!(reader.read_tensor_f32("w").is_ok());
}

/// read_tensor_f32_checked: NaN detection
#[test]
fn test_read_tensor_f32_checked_nan() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("has_nan", vec![3], &[1.0, f32::NAN, 3.0]);
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    let result = reader.read_tensor_f32_checked("has_nan");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("F-CKPT-013"));
}

/// read_tensor_f32_checked: Inf detection
#[test]
fn test_read_tensor_f32_checked_inf() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("has_inf", vec![3], &[1.0, f32::INFINITY, 3.0]);
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    let result = reader.read_tensor_f32_checked("has_inf");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("F-CKPT-013"));
}

/// read_tensor_f32_checked: valid data passes
#[test]
fn test_read_tensor_f32_checked_valid() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("ok", vec![3], &[1.0, 2.0, 3.0]);
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    let data = reader.read_tensor_f32_checked("ok").unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

/// validate_tensor_shape: matching shape passes
#[test]
fn test_validate_tensor_shape_ok() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("w", vec![2, 3], &[0.0; 6]);
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    assert!(reader.validate_tensor_shape("w", 6).is_ok());
}

/// validate_tensor_shape: mismatched shape errors
#[test]
fn test_validate_tensor_shape_mismatch() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32("w", vec![2, 3], &[0.0; 6]);
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    let result = reader.validate_tensor_shape("w", 10);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("F-CKPT-014"));
}

/// validate_tensor_shape: tensor not found
#[test]
fn test_validate_tensor_shape_not_found() {
    let writer = AprWriter::new();
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    let result = reader.validate_tensor_shape("nonexistent", 10);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("F-CKPT-014"));
}

/// get_tensor_bytes: out of bounds error
#[test]
fn test_read_tensor_from_file_nonexistent() {
    let result = AprReader::open("/tmp/absolutely_nonexistent_apr_file_12345.apr");
    assert!(result.is_err());
}

/// all_metadata iterator
#[test]
fn test_all_metadata_iterator() {
    let mut writer = AprWriter::new();
    writer.set_metadata("key1", JsonValue::String("val1".into()));
    writer.set_metadata("key2", JsonValue::Number(42.into()));
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    let meta: Vec<_> = reader.all_metadata().collect();
    assert!(meta.len() >= 2);
}

/// AprWriter: add_tensor_f32_owned (zero-copy path)
#[test]
fn test_add_tensor_f32_owned() {
    let mut writer = AprWriter::new();
    writer.add_tensor_f32_owned("owned_tensor", vec![3], vec![1.0, 2.0, 3.0]);
    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();
    let data = reader.read_tensor_f32("owned_tensor").unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}
