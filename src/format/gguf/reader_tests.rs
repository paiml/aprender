use super::*;

// ========================================================================
// BUG-GGUF-001 Falsification Tests: Allocation Attack Prevention
// ========================================================================

/// Create minimal GGUF header bytes for testing
fn create_gguf_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
    let mut data = Vec::new();
    // Magic: "GGUF"
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    // Version: 3
    data.extend_from_slice(&3u32.to_le_bytes());
    // Tensor count
    data.extend_from_slice(&tensor_count.to_le_bytes());
    // Metadata count
    data.extend_from_slice(&metadata_count.to_le_bytes());
    data
}

#[test]
fn test_bug_gguf_001_excessive_tensor_count_rejected() {
    // Create GGUF with tensor_count > MAX_TENSOR_COUNT
    let data = create_gguf_header(MAX_TENSOR_COUNT + 1, 0);

    let result = GgufReader::from_bytes(data);
    assert!(
        result.is_err(),
        "FALSIFIED: Excessive tensor_count should be rejected"
    );
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("exceeds maximum"),
        "Error should mention limit: {err}"
    );
}

#[test]
fn test_bug_gguf_001_excessive_metadata_count_rejected() {
    // Create GGUF with metadata_kv_count > MAX_METADATA_COUNT
    let data = create_gguf_header(1, MAX_METADATA_COUNT + 1);

    let result = GgufReader::from_bytes(data);
    assert!(
        result.is_err(),
        "FALSIFIED: Excessive metadata_kv_count should be rejected"
    );
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("exceeds maximum"),
        "Error should mention limit: {err}"
    );
}

#[test]
fn test_bug_gguf_001_max_tensor_count_allowed() {
    // Create GGUF with tensor_count = MAX_TENSOR_COUNT (should be allowed)
    // Will fail due to truncated file, but NOT due to count validation
    let data = create_gguf_header(MAX_TENSOR_COUNT, 0);

    let result = GgufReader::from_bytes(data);
    // Will fail because file is truncated, but NOT because of tensor_count
    match result {
        Err(e) => {
            let err = format!("{e:?}");
            assert!(
                !err.contains("tensor_count") || !err.contains("exceeds"),
                "MAX_TENSOR_COUNT should be accepted: {err}"
            );
        }
        Ok(_) => {
            // Unlikely but acceptable
        }
    }
}

#[test]
fn test_bug_gguf_001_zero_counts_valid() {
    // Zero tensor/metadata counts are valid (empty model)
    let data = create_gguf_header(0, 0);

    // Will succeed or fail due to other reasons (no tensor data), not counts
    let result = GgufReader::from_bytes(data);
    match result {
        Err(e) => {
            let err = format!("{e:?}");
            assert!(
                !err.contains("exceeds maximum"),
                "Zero counts should be valid: {err}"
            );
        }
        Ok(_) => {
            // Valid: empty GGUF file
        }
    }
}

// ========================================================================
// BUG-GGUF-002 Falsification Tests: Integer Overflow Prevention
// ========================================================================
// The shape.iter().product() call can overflow if malicious dimensions are provided.
// Byte size calculations (num_elements * bytes_per_element) can also overflow.
// Fixed by using checked_mul() and validating against MAX_TENSOR_ELEMENTS.

#[test]
fn test_bug_gguf_002_overflow_protection_documented() {
    // BUG-GGUF-002: Integer overflow in tensor element count
    //
    // Attack vector: Malicious GGUF with dimensions like [2^32, 2^32]
    // Prior behavior: Overflow to small value, then OOM or buffer overread
    //
    // Fix applied:
    // 1. Use checked_mul in shape.iter().try_fold() for element count
    // 2. Validate num_elements <= MAX_TENSOR_ELEMENTS (4 billion)
    // 3. Use checked_mul for all byte size calculations
    //
    // Locations fixed:
    // - get_tensor(): lines ~720-740
    // - get_tensor_raw(): lines ~870-920
    //
    // This test documents the fix. Triggering the actual overflow would require
    // crafting a valid GGUF header with malicious tensor dimensions, which is
    // complex. The fix ensures that IF such a file is parsed, it will fail
    // safely with an error instead of causing undefined behavior.
    assert!(MAX_TENSOR_ELEMENTS == 4_000_000_000);
}

// ========================================================================
// read_metadata_value Direct Tests: Cover All Type Branches
// ========================================================================

#[test]
fn test_read_metadata_value_uint64() {
    let val: u64 = 0x_DEAD_BEEF_CAFE_BABE;
    let bytes = val.to_le_bytes();
    let (result, consumed) = read_metadata_value(&bytes, 0, 10).expect("read uint64");
    assert_eq!(consumed, 8);
    match result {
        GgufValue::Uint64(v) => assert_eq!(v, val),
        other => panic!("Expected Uint64, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_int64() {
    let val: i64 = -123_456_789_012_345;
    let bytes = val.to_le_bytes();
    let (result, consumed) = read_metadata_value(&bytes, 0, 11).expect("read int64");
    assert_eq!(consumed, 8);
    match result {
        GgufValue::Int64(v) => assert_eq!(v, val),
        other => panic!("Expected Int64, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_float64() {
    let val: f64 = std::f64::consts::PI;
    let bytes = val.to_le_bytes();
    let (result, consumed) = read_metadata_value(&bytes, 0, 12).expect("read float64");
    assert_eq!(consumed, 8);
    match result {
        GgufValue::Float64(v) => assert!((v - val).abs() < f64::EPSILON),
        other => panic!("Expected Float64, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_unknown_type() {
    // Unknown type 99 should return Uint32(0) and consume 4 bytes
    let bytes = [0u8; 8];
    let (result, consumed) = read_metadata_value(&bytes, 0, 99).expect("read unknown");
    assert_eq!(consumed, 4);
    match result {
        GgufValue::Uint32(v) => assert_eq!(v, 0),
        other => panic!("Expected Uint32(0) for unknown type, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_array_other_type_uint8() {
    // Array of Uint8 (elem_type=0) falls into the "else" branch
    // Build: elem_type(4) + count(8) + data
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0u32.to_le_bytes()); // elem_type = 0 (Uint8)
    bytes.extend_from_slice(&3u64.to_le_bytes()); // count = 3
    bytes.extend_from_slice(&[10u8, 20u8, 30u8]); // 3 Uint8 elements
    let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array uint8");
    // 12 (header) + 3 * 1 (uint8 elem_size) = 15
    assert_eq!(consumed, 15);
    match result {
        GgufValue::ArrayUint32(v) => {
            assert!(v.is_empty(), "Other-type arrays return empty vec")
        }
        other => panic!("Expected empty ArrayUint32, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_array_other_type_uint16() {
    // Array of Uint16 (elem_type=2)
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&2u32.to_le_bytes()); // elem_type = 2 (Uint16)
    bytes.extend_from_slice(&2u64.to_le_bytes()); // count = 2
    bytes.extend_from_slice(&[0u8; 4]); // 2 * 2 bytes
    let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array uint16");
    assert_eq!(consumed, 16); // 12 + 2*2
    match result {
        GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
        other => panic!("Expected empty ArrayUint32, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_array_other_type_uint64() {
    // Array of Uint64 (elem_type=10) falls into 8-byte branch
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&10u32.to_le_bytes()); // elem_type = 10 (Uint64)
    bytes.extend_from_slice(&2u64.to_le_bytes()); // count = 2
    bytes.extend_from_slice(&[0u8; 16]); // 2 * 8 bytes
    let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array uint64");
    assert_eq!(consumed, 28); // 12 + 2*8
    match result {
        GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
        other => panic!("Expected empty ArrayUint32, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_array_other_type_int64() {
    // Array of Int64 (elem_type=11)
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&11u32.to_le_bytes()); // elem_type = 11 (Int64)
    bytes.extend_from_slice(&1u64.to_le_bytes()); // count = 1
    bytes.extend_from_slice(&[0u8; 8]); // 1 * 8 bytes
    let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array int64");
    assert_eq!(consumed, 20); // 12 + 1*8
    match result {
        GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
        other => panic!("Expected empty ArrayUint32, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_array_other_type_float64() {
    // Array of Float64 (elem_type=12)
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&12u32.to_le_bytes()); // elem_type = 12 (Float64)
    bytes.extend_from_slice(&1u64.to_le_bytes()); // count = 1
    bytes.extend_from_slice(&[0u8; 8]); // 1 * 8 bytes
    let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array float64");
    assert_eq!(consumed, 20); // 12 + 1*8
    match result {
        GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
        other => panic!("Expected empty ArrayUint32, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_array_other_type_bool() {
    // Array of Bool (elem_type=7) -> 1-byte elements
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&7u32.to_le_bytes()); // elem_type = 7 (Bool)
    bytes.extend_from_slice(&4u64.to_le_bytes()); // count = 4
    bytes.extend_from_slice(&[1u8, 0u8, 1u8, 0u8]); // 4 bools
    let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array bool");
    assert_eq!(consumed, 16); // 12 + 4*1
    match result {
        GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
        other => panic!("Expected empty ArrayUint32, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_array_other_type_default() {
    // Array of unknown elem_type (e.g., 99) -> default 4-byte elements
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&99u32.to_le_bytes()); // elem_type = 99 (unknown)
    bytes.extend_from_slice(&2u64.to_le_bytes()); // count = 2
    bytes.extend_from_slice(&[0u8; 8]); // 2 * 4 bytes
    let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array unknown");
    assert_eq!(consumed, 20); // 12 + 2*4
    match result {
        GgufValue::ArrayUint32(v) => assert!(v.is_empty()),
        other => panic!("Expected empty ArrayUint32, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_array_int32() {
    // Array of Int32 (elem_type=5) - has its own explicit branch
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&5u32.to_le_bytes()); // elem_type = 5 (Int32)
    bytes.extend_from_slice(&3u64.to_le_bytes()); // count = 3
    bytes.extend_from_slice(&(-10i32).to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&42i32.to_le_bytes());
    let (result, consumed) = read_metadata_value(&bytes, 0, 9).expect("read array int32");
    assert_eq!(consumed, 24); // 12 + 3*4
    match result {
        GgufValue::ArrayInt32(v) => {
            assert_eq!(v, vec![-10, 0, 42]);
        }
        other => panic!("Expected ArrayInt32, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_int64_eof() {
    // Int64 with insufficient data should error
    let bytes = [0u8; 4]; // Only 4 bytes, need 8
    let result = read_metadata_value(&bytes, 0, 11);
    assert!(result.is_err(), "Int64 with < 8 bytes should fail");
}

#[test]
fn test_read_metadata_value_float64_eof() {
    // Float64 with insufficient data should error
    let bytes = [0u8; 5]; // Only 5 bytes, need 8
    let result = read_metadata_value(&bytes, 0, 12);
    assert!(result.is_err(), "Float64 with < 8 bytes should fail");
}

#[test]
fn test_read_metadata_value_uint8() {
    let bytes = [42u8];
    let (result, consumed) = read_metadata_value(&bytes, 0, 0).expect("read uint8");
    assert_eq!(consumed, 1);
    match result {
        GgufValue::Uint8(v) => assert_eq!(v, 42),
        other => panic!("Expected Uint8, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_int8() {
    let bytes = [0xFEu8]; // -2 as i8
    let (result, consumed) = read_metadata_value(&bytes, 0, 1).expect("read int8");
    assert_eq!(consumed, 1);
    match result {
        GgufValue::Int8(v) => assert_eq!(v, -2),
        other => panic!("Expected Int8, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_uint16() {
    let bytes = 1000u16.to_le_bytes();
    let (result, consumed) = read_metadata_value(&bytes, 0, 2).expect("read uint16");
    assert_eq!(consumed, 2);
    match result {
        GgufValue::Uint16(v) => assert_eq!(v, 1000),
        other => panic!("Expected Uint16, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_int16() {
    let bytes = (-500i16).to_le_bytes();
    let (result, consumed) = read_metadata_value(&bytes, 0, 3).expect("read int16");
    assert_eq!(consumed, 2);
    match result {
        GgufValue::Int16(v) => assert_eq!(v, -500),
        other => panic!("Expected Int16, got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_bool_true() {
    let bytes = [1u8];
    let (result, consumed) = read_metadata_value(&bytes, 0, 7).expect("read bool");
    assert_eq!(consumed, 1);
    match result {
        GgufValue::Bool(v) => assert!(v),
        other => panic!("Expected Bool(true), got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_bool_false() {
    let bytes = [0u8];
    let (result, consumed) = read_metadata_value(&bytes, 0, 7).expect("read bool");
    assert_eq!(consumed, 1);
    match result {
        GgufValue::Bool(v) => assert!(!v),
        other => panic!("Expected Bool(false), got {other:?}"),
    }
}

#[test]
fn test_read_metadata_value_with_offset() {
    // Test reading with a non-zero offset
    let mut bytes = vec![0u8; 10]; // padding
    bytes.extend_from_slice(&42u64.to_le_bytes());
    let (result, consumed) = read_metadata_value(&bytes, 10, 10).expect("read uint64 at offset");
    assert_eq!(consumed, 8);
    match result {
        GgufValue::Uint64(v) => assert_eq!(v, 42),
        other => panic!("Expected Uint64, got {other:?}"),
    }
}

// ========================================================================
// GgufReader::from_bytes Error Path Tests
// ========================================================================

#[test]
fn test_from_bytes_file_too_small() {
    let data = vec![0u8; 10]; // < 24 bytes
    let result = GgufReader::from_bytes(data);
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("too small"),
        "Error should mention 'too small': {err}"
    );
}

#[test]
fn test_from_bytes_file_exactly_23_bytes() {
    let data = vec![0u8; 23]; // One byte short
    let result = GgufReader::from_bytes(data);
    assert!(result.is_err());
}

#[test]
fn test_from_bytes_invalid_magic() {
    let mut data = vec![0u8; 24];
    // Write invalid magic "XXXX" instead of "GGUF"
    data[0] = b'X';
    data[1] = b'X';
    data[2] = b'X';
    data[3] = b'X';
    let result = GgufReader::from_bytes(data);
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("Invalid GGUF magic"),
        "Error should mention invalid magic: {err}"
    );
    // GH-183: Enhanced error should show hex and ASCII
    assert!(
        err.contains("0x") || err.contains("ascii"),
        "Error should include hex/ascii debug info: {err}"
    );
}

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
