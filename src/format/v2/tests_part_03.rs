use super::*;

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
