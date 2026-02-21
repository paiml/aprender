use super::*;

#[test]
fn test_accessor_vocabulary_none_when_missing() {
    let data = create_gguf_header(0, 0);
    let reader = GgufReader::from_bytes(data).expect("parse empty GGUF");
    assert!(reader.vocabulary().is_none());
}

#[test]
fn test_accessor_tokenizer_model_none_when_missing() {
    let data = create_gguf_header(0, 0);
    let reader = GgufReader::from_bytes(data).expect("parse empty GGUF");
    assert!(reader.tokenizer_model().is_none());
}

#[test]
fn test_accessor_bos_token_id_none_when_missing() {
    let data = create_gguf_header(0, 0);
    let reader = GgufReader::from_bytes(data).expect("parse empty GGUF");
    assert!(reader.bos_token_id().is_none());
}

#[test]
fn test_accessor_eos_token_id_none_when_missing() {
    let data = create_gguf_header(0, 0);
    let reader = GgufReader::from_bytes(data).expect("parse empty GGUF");
    assert!(reader.eos_token_id().is_none());
}

#[test]
fn test_accessor_merges_none_when_missing() {
    let data = create_gguf_header(0, 0);
    let reader = GgufReader::from_bytes(data).expect("parse empty GGUF");
    assert!(reader.merges().is_none());
}

#[test]
fn test_accessor_vocabulary_none_when_empty() {
    // Build GGUF with empty token array
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

    // tokenizer.ggml.tokens = empty string array
    let key = "tokenizer.ggml.tokens";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes()); // Array
    data.extend_from_slice(&8u32.to_le_bytes()); // elem_type String
    data.extend_from_slice(&0u64.to_le_bytes()); // count = 0

    let reader = GgufReader::from_bytes(data).expect("parse GGUF with empty vocab");
    assert!(
        reader.vocabulary().is_none(),
        "Empty vocab should return None"
    );
}

#[test]
fn test_accessor_merges_none_when_empty() {
    // Build GGUF with empty merges array
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "tokenizer.ggml.merges";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes()); // Array
    data.extend_from_slice(&8u32.to_le_bytes()); // elem_type String
    data.extend_from_slice(&0u64.to_le_bytes()); // count = 0

    let reader = GgufReader::from_bytes(data).expect("parse GGUF with empty merges");
    assert!(reader.merges().is_none(), "Empty merges should return None");
}

// ========================================================================
// Mixed Metadata Tests (parsed + skipped in same file)
// ========================================================================

#[test]
fn test_from_bytes_mixed_parsed_and_skipped_metadata() {
    // One parsed key (tokenizer.*) and one skipped key (custom.*)
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&2u64.to_le_bytes()); // metadata_count = 2

    fn write_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    // Skipped: custom key with type Float32
    write_str(&mut data, "custom.learning_rate");
    data.extend_from_slice(&6u32.to_le_bytes()); // Float32
    data.extend_from_slice(&0.001f32.to_le_bytes());

    // Parsed: tokenizer key with type Uint32
    write_str(&mut data, "tokenizer.ggml.bos_token_id");
    data.extend_from_slice(&4u32.to_le_bytes()); // Uint32
    data.extend_from_slice(&1u32.to_le_bytes());

    let reader = GgufReader::from_bytes(data).expect("parse mixed metadata");
    assert!(!reader.metadata.contains_key("custom.learning_rate"));
    assert_eq!(reader.bos_token_id(), Some(1));
}

// ========================================================================
// read_u32 / read_u64 / read_string edge cases
// ========================================================================

#[test]
fn test_read_u32_eof() {
    let bytes = [0u8; 3]; // need 4
    let result = read_u32(&bytes, 0);
    assert!(result.is_err());
}

#[test]
fn test_read_u64_eof() {
    let bytes = [0u8; 7]; // need 8
    let result = read_u64(&bytes, 0);
    assert!(result.is_err());
}

#[test]
fn test_read_string_length_exceeds_data() {
    // Claim string is 100 bytes but only provide 5
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&100u64.to_le_bytes()); // length = 100
    bytes.extend_from_slice(b"short"); // only 5 bytes
    let result = read_string(&bytes, 0);
    assert!(result.is_err());
}

#[test]
fn test_read_string_empty() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0u64.to_le_bytes()); // length = 0
    let (s, consumed) = read_string(&bytes, 0).expect("read empty string");
    assert_eq!(s, "");
    assert_eq!(consumed, 8); // just the length prefix
}

// ========================================================================
// Additional read_metadata_value array branch tests
// ========================================================================

#[test]
fn test_read_metadata_value_array_of_strings() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&8u32.to_le_bytes()); // elem_type = String
    bytes.extend_from_slice(&2u64.to_le_bytes()); // count = 2
                                                  // string 1: "abc"
    bytes.extend_from_slice(&3u64.to_le_bytes());
    bytes.extend_from_slice(b"abc");
    // string 2: "de"
    bytes.extend_from_slice(&2u64.to_le_bytes());
    bytes.extend_from_slice(b"de");
    let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array of strings");
    // 12 (header) + 8+3 + 8+2 = 12 + 11 + 10 = 33
    assert_eq!(consumed, 33);
    match result {
        GgufValue::ArrayString(v) => {
            assert_eq!(v, vec!["abc".to_string(), "de".to_string()]);
        }
        other => panic!("Expected ArrayString, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_array_of_uint32() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&4u32.to_le_bytes()); // elem_type = Uint32
    bytes.extend_from_slice(&3u64.to_le_bytes()); // count = 3
    bytes.extend_from_slice(&100u32.to_le_bytes());
    bytes.extend_from_slice(&200u32.to_le_bytes());
    bytes.extend_from_slice(&300u32.to_le_bytes());
    let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array of uint32");
    assert_eq!(consumed, 24); // 12 + 3*4
    match result {
        GgufValue::ArrayUint32(v) => assert_eq!(v, vec![100, 200, 300]),
        other => panic!("Expected ArrayUint32, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_array_of_float32() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&6u32.to_le_bytes()); // elem_type = Float32
    bytes.extend_from_slice(&2u64.to_le_bytes()); // count = 2
    bytes.extend_from_slice(&1.5f32.to_le_bytes());
    bytes.extend_from_slice(&(-2.5f32).to_le_bytes());
    let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array of float32");
    assert_eq!(consumed, 20); // 12 + 2*4
    match result {
        GgufValue::ArrayFloat32(v) => {
            assert!((v[0] - 1.5).abs() < f32::EPSILON);
            assert!((v[1] - (-2.5)).abs() < f32::EPSILON);
        }
        other => panic!("Expected ArrayFloat32, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_array_of_int8() {
    // Array of Int8 (elem_type=1) -> "other" branch, 1-byte elements
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1u32.to_le_bytes()); // elem_type = 1 (Int8)
    bytes.extend_from_slice(&5u64.to_le_bytes()); // count = 5
    bytes.extend_from_slice(&[0u8; 5]);
    let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array int8");
    assert_eq!(consumed, 17); // 12 + 5*1
    match result {
        GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
        other => panic!("Expected empty ArrayUint32, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_array_of_int16() {
    // Array of Int16 (elem_type=3) -> "other" branch, 2-byte elements
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&3u32.to_le_bytes()); // elem_type = 3 (Int16)
    bytes.extend_from_slice(&4u64.to_le_bytes()); // count = 4
    bytes.extend_from_slice(&[0u8; 8]); // 4 * 2 bytes
    let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array int16");
    assert_eq!(consumed, 20); // 12 + 4*2
    match result {
        GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
        other => panic!("Expected empty ArrayUint32, got {other:?}"),
    }
}
