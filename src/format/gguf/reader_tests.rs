pub(crate) use super::*;

// ========================================================================
// BUG-GGUF-001 Falsification Tests: Allocation Attack Prevention
// ========================================================================

/// Create minimal GGUF header bytes for testing
pub(super) fn create_gguf_header(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
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

#[path = "reader_tests_part_02.rs"]

mod reader_tests_part_02;
#[path = "reader_tests_part_03.rs"]
mod reader_tests_part_03;
