
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
