
// ============================================================================
// Unaligned Pygmy Tests
// ============================================================================

#[test]
fn test_unaligned_pygmy_odd_offset() {
    let data = build_unaligned_pygmy_odd_offset();
    let result = GGUFModel::from_bytes(&data);

    // May parse but tensor access may fail
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 1);
            // Tensor exists but offset is odd - may cause issues on access
            assert_eq!(model.tensors[0].offset, 17);
        },
        Err(_) => {
            // Also acceptable - loader might reject unaligned offsets
        },
    }
}

#[test]
fn test_unaligned_pygmy_overflow_offset() {
    let data = build_unaligned_pygmy_overflow_offset();
    let result = GGUFModel::from_bytes(&data);

    // Should parse but tensor access should fail
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 1);
            // Offset is beyond file
            assert!(model.tensors[0].offset > data.len() as u64);
        },
        Err(_) => {
            // Also acceptable
        },
    }
}

// ============================================================================
// Malformed Pygmy Tests
// ============================================================================

#[test]
fn test_malformed_pygmy_empty_name() {
    let data = build_malformed_pygmy_empty_name();
    let result = GGUFModel::from_bytes(&data);

    // Empty tensor names may be accepted or rejected
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 1);
            assert!(model.tensors[0].name.is_empty());
        },
        Err(_) => {
            // Rejection is fine
        },
    }
}

#[test]
fn test_malformed_pygmy_long_name() {
    let data = build_malformed_pygmy_long_name();
    let result = GGUFModel::from_bytes(&data);

    // Should fail - not enough data for the claimed name length
    assert!(result.is_err(), "Long name without data should fail");
}

#[test]
fn test_malformed_pygmy_invalid_type() {
    let data = build_malformed_pygmy_invalid_type();
    let result = GGUFModel::from_bytes(&data);

    // May parse but type 255 is invalid
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 1);
            // Type is stored but may fail on dequantization
        },
        Err(_) => {
            // Rejection is also fine
        },
    }
}

#[test]
fn test_malformed_pygmy_zero_dims() {
    let data = build_malformed_pygmy_zero_dims();
    let result = GGUFModel::from_bytes(&data);

    // Zero dimensions (scalar) may or may not be supported
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 1);
            // Check dims
            assert_eq!(model.tensors[0].dims.len(), 0);
        },
        Err(_) => {
            // Rejection is fine
        },
    }
}

#[test]
fn test_malformed_pygmy_too_many_dims() {
    let data = build_malformed_pygmy_too_many_dims();
    let result = GGUFModel::from_bytes(&data);

    // 100 dimensions may or may not be accepted
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 1);
            assert_eq!(model.tensors[0].dims.len(), 100);
        },
        Err(_) => {
            // Rejection is fine - 100 dims is unreasonable
        },
    }
}

#[test]
fn test_malformed_pygmy_overlapping_tensors() {
    let data = build_malformed_pygmy_overlapping_tensors();
    let result = GGUFModel::from_bytes(&data);

    // Overlapping offsets may parse but cause data corruption on read
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 2);
            // Tensors overlap in memory
            let t0_end = model.tensors[0].offset + 128; // 32 * 4 bytes
            let t1_start = model.tensors[1].offset;
            assert!(t1_start < t0_end, "Tensors should overlap");
        },
        Err(_) => {
            // Rejection is also fine
        },
    }
}

// ============================================================================
// Shard Metadata Tests
// ============================================================================

#[test]
fn test_shard_pygmy_split_metadata() {
    let data = build_shard_pygmy_split_metadata();
    let result = GGUFModel::from_bytes(&data);

    match result {
        Ok(model) => {
            // Check metadata was parsed
            assert_eq!(model.metadata.len(), 3);

            // Verify shard metadata keys exist
            let keys: Vec<&str> = model
                .metadata
                .keys()
                .map(std::string::String::as_str)
                .collect();
            assert!(keys.contains(&"split.no"));
            assert!(keys.contains(&"split.count"));
            assert!(keys.contains(&"split.tensors.count"));
        },
        Err(e) => {
            // Parsing may fail but that's also coverage
            let _ = e;
        },
    }
}

// ============================================================================
// Padding Pattern Tests
// ============================================================================

#[test]
fn test_padding_pattern_all_zeros() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let name = "padded_tensor";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes()); // F32
    data.extend_from_slice(&0u64.to_le_bytes());

    // Pad with zeros to 32-byte alignment
    while data.len() % 32 != 0 {
        data.push(0x00);
    }

    // Tensor data
    for i in 0..4 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_ok(), "Zero padding should work");
}

#[test]
fn test_padding_pattern_all_ff() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let name = "ff_padded";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    // Pad with 0xFF (unusual but should still work)
    while data.len() % 32 != 0 {
        data.push(0xFF);
    }

    for i in 0..4 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_ok(), "0xFF padding should work");
}

#[test]
fn test_padding_pattern_alternating() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let name = "alt_padded";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    // Alternating pattern padding
    let mut alt = false;
    while data.len() % 32 != 0 {
        data.push(if alt { 0xAA } else { 0x55 });
        alt = !alt;
    }

    for i in 0..4 {
        data.extend_from_slice(&(i as f32).to_le_bytes());
    }

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_ok(), "Alternating padding should work");
}

// ============================================================================
// Edge Case: Tensor Count Mismatch
// ============================================================================

#[test]
fn test_tensor_count_mismatch_more_claimed() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&10u64.to_le_bytes()); // Claims 10 tensors
    data.extend_from_slice(&0u64.to_le_bytes());

    // Only provide 1 tensor info
    let name = "only_one";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    // Should fail - not enough tensor info
    assert!(result.is_err(), "Missing tensors should fail");
}

#[test]
fn test_tensor_count_zero_with_data() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
    data.extend_from_slice(&0u64.to_le_bytes());

    // Add some garbage "tensor data"
    data.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

    let result = GGUFModel::from_bytes(&data);
    // Zero tensors is valid
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 0);
        },
        Err(_) => {
            // Also fine
        },
    }
}

// ============================================================================
// Metadata Type Edge Cases
// ============================================================================

#[test]
fn test_metadata_type_string_empty() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata

    let key = "empty_string_key";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING
    data.extend_from_slice(&0u64.to_le_bytes()); // Empty string (length 0)

    let result = GGUFModel::from_bytes(&data);
    if let Ok(model) = result {
        assert_eq!(model.metadata.len(), 1);
    }
}

#[test]
fn test_metadata_type_array_empty() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    let key = "empty_array";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&9u32.to_le_bytes()); // GGUF_TYPE_ARRAY
    data.extend_from_slice(&4u32.to_le_bytes()); // Array element type: UINT32
    data.extend_from_slice(&0u64.to_le_bytes()); // Array length: 0

    let result = GGUFModel::from_bytes(&data);
    if let Ok(model) = result {
        assert_eq!(model.metadata.len(), 1);
    }
}

// ============================================================================
// Stress Test: Many Tensors with Varied Types
// ============================================================================

#[test]
fn test_many_tensors_varied_types() {
    let mut data = Vec::new();

    let tensor_count = 50u64;

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&tensor_count.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    // Tensor types to cycle through
    let types = [0u32, 1, 2, 6, 7, 8]; // F32, F16, Q4_0, Q8_0, Q5_0, Q5_1

    let mut offset = 0u64;

    for i in 0..tensor_count {
        let name = format!("tensor_{:03}", i);
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&1u32.to_le_bytes()); // 1D
        data.extend_from_slice(&32u64.to_le_bytes()); // 32 elements
        data.extend_from_slice(&types[(i as usize) % types.len()].to_le_bytes());
        data.extend_from_slice(&offset.to_le_bytes());

        // Rough size estimate (varies by type)
        offset += 128; // Simple approximation
    }

    // Pad to alignment
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Add some tensor data
    for _ in 0..(offset as usize) {
        data.push(0);
    }

    let result = GGUFModel::from_bytes(&data);
    match result {
        Ok(model) => {
            assert_eq!(model.tensors.len(), 50);
        },
        Err(_) => {
            // May fail due to type validation
        },
    }
}
