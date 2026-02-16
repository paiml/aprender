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
