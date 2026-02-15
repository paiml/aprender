use super::super::*;

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

include!("tests_layout_part_02.rs");
include!("tests_layout_part_03.rs");
