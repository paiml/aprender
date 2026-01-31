use super::*;

// =========================================================================
// Magic and Format Tests
// =========================================================================

#[test]
fn test_apr_magic() {
    assert_eq!(APR_MAGIC, [b'A', b'P', b'R', b'1']);
}

#[test]
fn test_writer_creates_valid_magic() {
    let writer = AprWriter::new();
    let bytes = writer.to_bytes().unwrap();
    assert_eq!(&bytes[0..4], &APR_MAGIC);
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
    writer.set_metadata("model", JsonValue::String("test".into()));
    writer.set_metadata("version", JsonValue::Number(1.into()));

    // Add tensors
    writer.add_tensor_f32("layer.0.weight", vec![4, 4], &vec![0.5; 16]);
    writer.add_tensor_f32("layer.0.bias", vec![4], &[0.1, 0.2, 0.3, 0.4]);

    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    // Verify metadata
    assert_eq!(
        reader.get_metadata("model"),
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
    assert!(result.unwrap_err().contains("Invalid magic"));
}

#[test]
fn test_file_too_short() {
    let data = vec![b'A', b'P', b'R'];
    let result = AprReader::from_bytes(data);
    assert!(result.is_err());
}

// =========================================================================
// CRC32 Tests
// =========================================================================

#[test]
fn test_crc32_empty() {
    assert_eq!(crc32(&[]), 0x0000_0000);
}

#[test]
fn test_crc32_hello() {
    // Known CRC32 for "hello"
    let crc = crc32(b"hello");
    assert_eq!(crc, 0x3610_A686);
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

// =========================================================================
// Compression Tests (GH-146)
// =========================================================================

#[test]
fn test_apr_magic_distinct() {
    // APR1 (uncompressed) vs APR\0 (compressed) - distinct formats
    assert_eq!(APR_MAGIC, [b'A', b'P', b'R', b'1']);
    assert_eq!(APR_MAGIC_COMPRESSED, [b'A', b'P', b'R', 0]);
    assert_ne!(APR_MAGIC, APR_MAGIC_COMPRESSED);
}

#[test]
fn test_compression_byte_roundtrip() {
    assert_eq!(Compression::None.as_byte(), 0);
    assert_eq!(Compression::from_byte(0), Some(Compression::None));

    #[cfg(feature = "format-compression")]
    {
        assert_eq!(Compression::Lz4.as_byte(), 1);
        assert_eq!(Compression::from_byte(1), Some(Compression::Lz4));
        assert_eq!(Compression::Zstd.as_byte(), 2);
        assert_eq!(Compression::from_byte(2), Some(Compression::Zstd));
    }

    assert_eq!(Compression::from_byte(255), None);
}

#[test]
fn test_with_compression_builder() {
    let writer = AprWriter::new().with_compression(Compression::None);
    let bytes = writer.to_bytes().unwrap();
    // Should produce APR1 format (no compression)
    assert_eq!(&bytes[0..4], &APR_MAGIC);
}

#[cfg(feature = "format-compression")]
#[test]
fn test_lz4_compression_roundtrip() {
    let mut writer = AprWriter::new().with_compression(Compression::Lz4);
    writer.set_metadata("model", JsonValue::String("test-lz4".into()));
    writer.add_tensor_f32("weights", vec![100], &vec![0.5; 100]);

    let bytes = writer.to_bytes().unwrap();

    // Should produce APR2 format
    assert_eq!(&bytes[0..4], &APR_MAGIC_COMPRESSED);
    assert_eq!(bytes[4], 1); // LZ4 compression byte

    // Reader should auto-detect and decompress
    let reader = AprReader::from_bytes(bytes).unwrap();
    assert_eq!(
        reader.get_metadata("model"),
        Some(&JsonValue::String("test-lz4".into()))
    );
    let data = reader.read_tensor_f32("weights").unwrap();
    assert_eq!(data, vec![0.5; 100]);
}

#[cfg(feature = "format-compression")]
#[test]
fn test_zstd_compression_roundtrip() {
    let mut writer = AprWriter::new().with_compression(Compression::Zstd);
    writer.set_metadata("model", JsonValue::String("test-zstd".into()));
    writer.add_tensor_f32("bias", vec![50], &vec![0.1; 50]);

    let bytes = writer.to_bytes().unwrap();

    // Should produce APR2 format
    assert_eq!(&bytes[0..4], &APR_MAGIC_COMPRESSED);
    assert_eq!(bytes[4], 2); // ZSTD compression byte

    // Reader should auto-detect and decompress
    let reader = AprReader::from_bytes(bytes).unwrap();
    assert_eq!(
        reader.get_metadata("model"),
        Some(&JsonValue::String("test-zstd".into()))
    );
    let data = reader.read_tensor_f32("bias").unwrap();
    assert_eq!(data, vec![0.1; 50]);
}

#[cfg(feature = "format-compression")]
#[test]
fn test_compression_reduces_size() {
    // Create data with high compressibility (repeated values)
    let mut writer_uncompressed = AprWriter::new();
    writer_uncompressed.add_tensor_f32("data", vec![10000], &vec![0.0; 10000]);
    let uncompressed = writer_uncompressed.to_bytes().unwrap();

    let mut writer_lz4 = AprWriter::new().with_compression(Compression::Lz4);
    writer_lz4.add_tensor_f32("data", vec![10000], &vec![0.0; 10000]);
    let compressed = writer_lz4.to_bytes().unwrap();

    // LZ4 should compress repeated zeros very well
    assert!(
        compressed.len() < uncompressed.len() / 10,
        "LZ4 should compress repeated data significantly: {} vs {}",
        compressed.len(),
        uncompressed.len()
    );
}

#[cfg(feature = "format-compression")]
#[test]
fn test_large_model_compression_roundtrip() {
    // Simulate a small ML model
    let mut writer = AprWriter::new().with_compression(Compression::Lz4);

    writer.set_metadata("model_type", JsonValue::String("whisper-tiny".into()));
    writer.set_metadata("n_vocab", JsonValue::Number(51865.into()));

    // Add multiple tensors like a real model
    writer.add_tensor_f32("encoder.embed", vec![384, 80], &vec![0.01; 384 * 80]);
    writer.add_tensor_f32(
        "encoder.conv1.weight",
        vec![384, 80, 3],
        &vec![0.02; 384 * 80 * 3],
    );
    writer.add_tensor_f32("decoder.embed", vec![51865, 384], &vec![0.001; 51865 * 384]);

    let bytes = writer.to_bytes().unwrap();
    let reader = AprReader::from_bytes(bytes).unwrap();

    // Verify metadata
    assert_eq!(
        reader.get_metadata("model_type"),
        Some(&JsonValue::String("whisper-tiny".into()))
    );

    // Verify tensors
    assert_eq!(reader.tensors.len(), 3);
    let embed = reader.read_tensor_f32("encoder.embed").unwrap();
    assert_eq!(embed.len(), 384 * 80);
}
