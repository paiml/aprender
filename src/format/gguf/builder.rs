use super::*;

#[test]
fn test_from_bytes_zero_tensors_zero_metadata() {
    let data = create_gguf_header(0, 0);
    let reader = GgufReader::from_bytes(data).expect("valid empty GGUF");
    assert_eq!(reader.tensor_count, 0);
    assert!(reader.tensors.is_empty());
    assert!(reader.metadata.is_empty());
    assert_eq!(reader.version, 3);
}

// ========================================================================
// GgufReader::from_bytes with Tensor Metadata Tests
// ========================================================================

/// Build a complete synthetic GGUF file with one F32 tensor and optional metadata
fn build_synthetic_gguf_with_tensor(
    tensor_name: &str,
    dims: &[u64],
    dtype: u32,
    tensor_data: &[u8],
    metadata: &[(&str, u32, &[u8])], // (key, value_type, value_bytes)
) -> Vec<u8> {
    let mut data = Vec::new();

    // Header
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes()); // version
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
    data.extend_from_slice(&(metadata.len() as u64).to_le_bytes()); // metadata_count

    // Metadata KV pairs
    for (key, value_type, value_bytes) in metadata {
        // Key string (length-prefixed)
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        // Value type
        data.extend_from_slice(&value_type.to_le_bytes());
        // Value bytes
        data.extend_from_slice(value_bytes);
    }

    // Tensor info
    // Name (length-prefixed)
    data.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    data.extend_from_slice(tensor_name.as_bytes());
    // n_dims
    let n_dims = dims.len() as u32;
    data.extend_from_slice(&n_dims.to_le_bytes());
    // dims
    for d in dims {
        data.extend_from_slice(&d.to_le_bytes());
    }
    // dtype
    data.extend_from_slice(&dtype.to_le_bytes());
    // offset within tensor data section
    data.extend_from_slice(&0u64.to_le_bytes());

    // Alignment padding
    let padding = padding_for_alignment(data.len(), GGUF_DEFAULT_ALIGNMENT);
    data.extend(std::iter::repeat(0u8).take(padding));

    // Tensor data
    data.extend_from_slice(tensor_data);

    data
}

#[test]
fn test_from_bytes_with_one_f32_tensor() {
    // 2x2 F32 tensor = 4 elements * 4 bytes = 16 bytes
    let tensor_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let data = build_synthetic_gguf_with_tensor("test.weight", &[2, 2], 0, &tensor_data, &[]);

    let reader = GgufReader::from_bytes(data).expect("parse GGUF with tensor");
    assert_eq!(reader.tensor_count, 1);
    assert_eq!(reader.tensors.len(), 1);
    assert_eq!(reader.tensors[0].name, "test.weight");
    assert_eq!(reader.tensors[0].dims, vec![2, 2]);
    assert_eq!(reader.tensors[0].dtype, 0); // F32

    // Verify we can extract tensor data
    let (extracted, shape) = reader
        .get_tensor_f32("test.weight")
        .expect("extract tensor");
    assert_eq!(shape, vec![2, 2]);
    assert_eq!(extracted.len(), 4);
    assert!((extracted[0] - 1.0).abs() < f32::EPSILON);
    assert!((extracted[3] - 4.0).abs() < f32::EPSILON);
}

#[test]
fn test_from_bytes_tensor_excessive_dims() {
    // n_dims > MAX_DIMS should fail
    let mut data = Vec::new();

    // Header
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

    // Tensor info: name
    let name = "bad.tensor";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    // n_dims = MAX_DIMS + 1 = 17
    data.extend_from_slice(&(MAX_DIMS + 1).to_le_bytes());
    // Provide enough dummy dim data
    for _ in 0..=MAX_DIMS {
        data.extend_from_slice(&1u64.to_le_bytes());
    }
    data.extend_from_slice(&0u32.to_le_bytes()); // dtype
    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    let result = GgufReader::from_bytes(data);
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("dimensions") && err.contains("exceeds"),
        "Error should mention excessive dimensions: {err}"
    );
}

#[test]
fn test_from_bytes_tensor_at_max_dims() {
    // n_dims = MAX_DIMS (16) should be allowed
    let mut data = Vec::new();

    // Header
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // tensor_count = 1
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

    // Tensor info: name
    let name = "ok.tensor";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    // n_dims = MAX_DIMS (16)
    data.extend_from_slice(&MAX_DIMS.to_le_bytes());
    // All dims = 1
    for _ in 0..MAX_DIMS {
        data.extend_from_slice(&1u64.to_le_bytes());
    }
    data.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    // Add alignment padding + tiny tensor data (1 element F32 = 4 bytes)
    let padding = padding_for_alignment(data.len(), GGUF_DEFAULT_ALIGNMENT);
    data.extend(std::iter::repeat(0u8).take(padding));
    data.extend_from_slice(&1.0f32.to_le_bytes());

    let reader = GgufReader::from_bytes(data).expect("MAX_DIMS should be accepted");
    assert_eq!(reader.tensors[0].dims.len(), MAX_DIMS as usize);
}

// ========================================================================
// skip_metadata_value Tests (via from_bytes with non-parsed keys)
// ========================================================================

/// Build a GGUF with metadata that will be skipped (key prefix not in parsed set)
fn build_gguf_with_skipped_metadata(key: &str, value_type: u32, value_bytes: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();

    // Header
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes()); // version
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1

    // Metadata KV: key with prefix that does NOT match tokenizer./general./llama./qwen2./phi./mistral.
    // so it will be skipped via skip_metadata_value
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&value_type.to_le_bytes());
    data.extend_from_slice(value_bytes);

    data
}

#[test]
fn test_skip_metadata_value_uint8() {
    let data = build_gguf_with_skipped_metadata("custom.u8", 0, &[42u8]);
    let reader = GgufReader::from_bytes(data).expect("skip uint8");
    assert!(!reader.metadata.contains_key("custom.u8"));
}

#[test]
fn test_skip_metadata_value_int8() {
    let data = build_gguf_with_skipped_metadata("custom.i8", 1, &[0xFEu8]);
    let reader = GgufReader::from_bytes(data).expect("skip int8");
    assert!(!reader.metadata.contains_key("custom.i8"));
}

#[test]
fn test_skip_metadata_value_uint16() {
    let data = build_gguf_with_skipped_metadata("custom.u16", 2, &1000u16.to_le_bytes());
    let reader = GgufReader::from_bytes(data).expect("skip uint16");
    assert!(!reader.metadata.contains_key("custom.u16"));
}

#[test]
fn test_skip_metadata_value_int16() {
    let data = build_gguf_with_skipped_metadata("custom.i16", 3, &(-500i16).to_le_bytes());
    let reader = GgufReader::from_bytes(data).expect("skip int16");
    assert!(!reader.metadata.contains_key("custom.i16"));
}

#[test]
fn test_skip_metadata_value_bool() {
    let data = build_gguf_with_skipped_metadata("custom.flag", 7, &[1u8]);
    let reader = GgufReader::from_bytes(data).expect("skip bool");
    assert!(!reader.metadata.contains_key("custom.flag"));
}

#[test]
fn test_skip_metadata_value_string() {
    // String: length-prefixed (8 bytes length + content)
    let s = "hello world";
    let mut value_bytes = Vec::new();
    value_bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
    value_bytes.extend_from_slice(s.as_bytes());
    let data = build_gguf_with_skipped_metadata("custom.str", 8, &value_bytes);
    let reader = GgufReader::from_bytes(data).expect("skip string");
    assert!(!reader.metadata.contains_key("custom.str"));
}

#[test]
fn test_skip_metadata_value_uint64() {
    let data = build_gguf_with_skipped_metadata("custom.u64", 10, &999u64.to_le_bytes());
    let reader = GgufReader::from_bytes(data).expect("skip uint64");
    assert!(!reader.metadata.contains_key("custom.u64"));
}

#[test]
fn test_skip_metadata_value_int64() {
    let data = build_gguf_with_skipped_metadata("custom.i64", 11, &(-1i64).to_le_bytes());
    let reader = GgufReader::from_bytes(data).expect("skip int64");
    assert!(!reader.metadata.contains_key("custom.i64"));
}

#[test]
fn test_skip_metadata_value_float64() {
    let data =
        build_gguf_with_skipped_metadata("custom.f64", 12, &std::f64::consts::E.to_le_bytes());
    let reader = GgufReader::from_bytes(data).expect("skip float64");
    assert!(!reader.metadata.contains_key("custom.f64"));
}

#[test]
fn test_skip_metadata_value_unknown_type() {
    // Unknown type (e.g., 99) should skip 4 bytes
    let data = build_gguf_with_skipped_metadata("custom.unk", 99, &[0u8; 4]);
    let reader = GgufReader::from_bytes(data).expect("skip unknown");
    assert!(!reader.metadata.contains_key("custom.unk"));
}

#[test]
fn test_skip_metadata_value_array_of_uint32() {
    // Array type=9, elem_type=4 (uint32), count=2
    let mut value_bytes = Vec::new();
    value_bytes.extend_from_slice(&4u32.to_le_bytes()); // elem_type Uint32
    value_bytes.extend_from_slice(&2u64.to_le_bytes()); // count
    value_bytes.extend_from_slice(&10u32.to_le_bytes());
    value_bytes.extend_from_slice(&20u32.to_le_bytes());
    let data = build_gguf_with_skipped_metadata("custom.arr_u32", 9, &value_bytes);
    let reader = GgufReader::from_bytes(data).expect("skip array uint32");
    assert!(!reader.metadata.contains_key("custom.arr_u32"));
}

#[test]
fn test_skip_metadata_value_array_of_strings() {
    // Array type=9, elem_type=8 (string), count=2
    let mut value_bytes = Vec::new();
    value_bytes.extend_from_slice(&8u32.to_le_bytes()); // elem_type String
    value_bytes.extend_from_slice(&2u64.to_le_bytes()); // count
                                                        // string 1: "hi"
    value_bytes.extend_from_slice(&2u64.to_le_bytes());
    value_bytes.extend_from_slice(b"hi");
    // string 2: "world"
    value_bytes.extend_from_slice(&5u64.to_le_bytes());
    value_bytes.extend_from_slice(b"world");
    let data = build_gguf_with_skipped_metadata("custom.arr_str", 9, &value_bytes);
    let reader = GgufReader::from_bytes(data).expect("skip array of strings");
    assert!(!reader.metadata.contains_key("custom.arr_str"));
}

#[test]
fn test_skip_metadata_value_array_of_uint8() {
    // Array type=9, elem_type=0 (uint8), count=3
    let mut value_bytes = Vec::new();
    value_bytes.extend_from_slice(&0u32.to_le_bytes()); // elem_type Uint8
    value_bytes.extend_from_slice(&3u64.to_le_bytes()); // count
    value_bytes.extend_from_slice(&[1u8, 2u8, 3u8]);
    let data = build_gguf_with_skipped_metadata("custom.arr_u8", 9, &value_bytes);
    let reader = GgufReader::from_bytes(data).expect("skip array of uint8");
    assert!(!reader.metadata.contains_key("custom.arr_u8"));
}

#[test]
fn test_skip_metadata_value_array_of_uint64() {
    // Array type=9, elem_type=10 (uint64), count=1
    let mut value_bytes = Vec::new();
    value_bytes.extend_from_slice(&10u32.to_le_bytes()); // elem_type Uint64
    value_bytes.extend_from_slice(&1u64.to_le_bytes()); // count
    value_bytes.extend_from_slice(&42u64.to_le_bytes());
    let data = build_gguf_with_skipped_metadata("custom.arr_u64", 9, &value_bytes);
    let reader = GgufReader::from_bytes(data).expect("skip array of uint64");
    assert!(!reader.metadata.contains_key("custom.arr_u64"));
}

#[test]
fn test_skip_metadata_value_array_of_int16() {
    // Array type=9, elem_type=3 (int16), count=2
    let mut value_bytes = Vec::new();
    value_bytes.extend_from_slice(&3u32.to_le_bytes()); // elem_type Int16
    value_bytes.extend_from_slice(&2u64.to_le_bytes()); // count
    value_bytes.extend_from_slice(&(-1i16).to_le_bytes());
    value_bytes.extend_from_slice(&100i16.to_le_bytes());
    let data = build_gguf_with_skipped_metadata("custom.arr_i16", 9, &value_bytes);
    let reader = GgufReader::from_bytes(data).expect("skip array of int16");
    assert!(!reader.metadata.contains_key("custom.arr_i16"));
}

// ========================================================================
// GgufReader Accessor Method Tests
// ========================================================================

/// Build a GGUF with tokenizer metadata (parsed keys)
fn build_gguf_with_tokenizer_metadata() -> Vec<u8> {
    let mut data = Vec::new();

    // We'll add 6 metadata entries: tokens, model, bos, eos, merges, architecture
    let metadata_count = 6u64;

    // Header
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&metadata_count.to_le_bytes());

    // Helper: write a length-prefixed string
    fn write_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    // 1. tokenizer.ggml.tokens (ArrayString)
    write_str(&mut data, "tokenizer.ggml.tokens");
    data.extend_from_slice(&9u32.to_le_bytes()); // type = Array
    data.extend_from_slice(&8u32.to_le_bytes()); // elem_type = String
    data.extend_from_slice(&3u64.to_le_bytes()); // count = 3
    write_str(&mut data, "<unk>");
    write_str(&mut data, "hello");
    write_str(&mut data, "world");

    // 2. tokenizer.ggml.model (String)
    write_str(&mut data, "tokenizer.ggml.model");
    data.extend_from_slice(&8u32.to_le_bytes()); // type = String
    write_str(&mut data, "llama");

    // 3. tokenizer.ggml.bos_token_id (Uint32)
    write_str(&mut data, "tokenizer.ggml.bos_token_id");
    data.extend_from_slice(&4u32.to_le_bytes()); // type = Uint32
    data.extend_from_slice(&1u32.to_le_bytes()); // value = 1

    // 4. tokenizer.ggml.eos_token_id (Uint32)
    write_str(&mut data, "tokenizer.ggml.eos_token_id");
    data.extend_from_slice(&4u32.to_le_bytes()); // type = Uint32
    data.extend_from_slice(&2u32.to_le_bytes()); // value = 2

    // 5. tokenizer.ggml.merges (ArrayString)
    write_str(&mut data, "tokenizer.ggml.merges");
    data.extend_from_slice(&9u32.to_le_bytes()); // type = Array
    data.extend_from_slice(&8u32.to_le_bytes()); // elem_type = String
    data.extend_from_slice(&2u64.to_le_bytes()); // count = 2
    write_str(&mut data, "h e");
    write_str(&mut data, "l o");

    // 6. general.architecture (String)
    write_str(&mut data, "general.architecture");
    data.extend_from_slice(&8u32.to_le_bytes()); // type = String
    write_str(&mut data, "llama");

    data
}

#[test]
fn test_accessor_vocabulary() {
    let data = build_gguf_with_tokenizer_metadata();
    let reader = GgufReader::from_bytes(data).expect("parse tokenizer metadata");
    let vocab = reader.vocabulary().expect("vocabulary should exist");
    assert_eq!(vocab.len(), 3);
    assert_eq!(vocab[0], "<unk>");
    assert_eq!(vocab[1], "hello");
    assert_eq!(vocab[2], "world");
}

#[test]
fn test_accessor_tokenizer_model() {
    let data = build_gguf_with_tokenizer_metadata();
    let reader = GgufReader::from_bytes(data).expect("parse tokenizer metadata");
    let model = reader
        .tokenizer_model()
        .expect("tokenizer model should exist");
    assert_eq!(model, "llama");
}

#[test]
fn test_accessor_bos_token_id() {
    let data = build_gguf_with_tokenizer_metadata();
    let reader = GgufReader::from_bytes(data).expect("parse tokenizer metadata");
    let bos = reader.bos_token_id().expect("bos_token_id should exist");
    assert_eq!(bos, 1);
}

#[test]
fn test_accessor_eos_token_id() {
    let data = build_gguf_with_tokenizer_metadata();
    let reader = GgufReader::from_bytes(data).expect("parse tokenizer metadata");
    let eos = reader.eos_token_id().expect("eos_token_id should exist");
    assert_eq!(eos, 2);
}

#[test]
fn test_accessor_merges() {
    let data = build_gguf_with_tokenizer_metadata();
    let reader = GgufReader::from_bytes(data).expect("parse tokenizer metadata");
    let merges = reader.merges().expect("merges should exist");
    assert_eq!(merges.len(), 2);
    assert_eq!(merges[0], "h e");
    assert_eq!(merges[1], "l o");
}

#[test]
fn test_accessor_architecture() {
    let data = build_gguf_with_tokenizer_metadata();
    let reader = GgufReader::from_bytes(data).expect("parse tokenizer metadata");
    let arch = reader.architecture().expect("architecture should exist");
    assert_eq!(arch, "llama");
}
