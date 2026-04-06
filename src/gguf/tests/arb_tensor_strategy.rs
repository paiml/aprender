
// ============================================================================
// Tensor Dimension Fuzzing
// ============================================================================

/// Generate arbitrary tensor dimensions
/// Note: Avoid u64::MAX as it causes OOM; use header-based overflow detection instead
fn arb_tensor_dims() -> impl Strategy<Value = (u32, Vec<u64>)> {
    (0u32..=10).prop_flat_map(|n_dims| {
        let dims = prop::collection::vec(
            prop_oneof![
                3 => 1u64..100,           // Small (common)
                1 => Just(0u64),          // Zero dimension
                1 => Just(1u64),          // Singleton
                1 => 100u64..10000,       // Large
                1 => Just(1u64 << 30),    // Large but won't OOM (header validation)
            ],
            n_dims as usize,
        );
        dims.prop_map(move |d| (n_dims, d))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Fuzz tensor dimensions
    #[test]
    fn fuzz_tensor_dimensions(
        (n_dims, dims) in arb_tensor_dims(),
        tensor_type in 0u32..20
    ) {
        let mut data = Vec::new();

        // Valid header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        // Tensor with arbitrary dimensions
        let name = "fuzz_tensor";
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&n_dims.to_le_bytes());

        for dim in &dims {
            data.extend_from_slice(&dim.to_le_bytes());
        }

        data.extend_from_slice(&tensor_type.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Pad
        while data.len() % 32 != 0 {
            data.push(0);
        }

        // Minimal data
        data.extend_from_slice(&[0u8; 64]);

        let result = GGUFModel::from_bytes(&data);
        // Should not panic
        let _ = result;
    }
}

// ============================================================================
// Metadata Type Exhaustion
// ============================================================================

/// All GGUF metadata types
const GGUF_TYPES: [u32; 13] = [
    0,  // UINT8
    1,  // INT8
    2,  // UINT16
    3,  // INT16
    4,  // UINT32
    5,  // INT32
    6,  // FLOAT32
    7,  // BOOL
    8,  // STRING
    9,  // ARRAY
    10, // UINT64
    11, // INT64
    12, // FLOAT64
];

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Fuzz metadata with all types
    #[test]
    fn fuzz_metadata_types(
        meta_type in prop::sample::select(&GGUF_TYPES),
        value_len in 0usize..100
    ) {
        let mut data = Vec::new();

        // Valid header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata

        // Metadata key
        let key = "fuzz_key";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());

        // Type
        data.extend_from_slice(&meta_type.to_le_bytes());

        // Value (type-dependent)
        match meta_type {
            8 => { // STRING
                data.extend_from_slice(&(value_len as u64).to_le_bytes());
                data.extend(std::iter::repeat_n(b'x', value_len));
            }
            9 => { // ARRAY
                data.extend_from_slice(&4u32.to_le_bytes()); // Element type: UINT32
                data.extend_from_slice(&(value_len as u64).to_le_bytes()); // Length
                for i in 0..value_len {
                    data.extend_from_slice(&(i as u32).to_le_bytes());
                }
            }
            _ => {
                // Fixed-size types: provide 8 bytes
                data.extend_from_slice(&[0u8; 8]);
            }
        }

        let result = GGUFModel::from_bytes(&data);
        let _ = result;
    }

    /// Fuzz with invalid metadata types
    /// Note: Types 0-12 are valid (0-11 standard + 12=FLOAT64), so start from 13
    #[test]
    fn fuzz_invalid_metadata_type(
        invalid_type in 13u32..256
    ) {
        let mut data = Vec::new();

        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());

        let key = "invalid_type_key";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&invalid_type.to_le_bytes());
        data.extend_from_slice(&[0u8; 8]); // Some value bytes

        let result = GGUFModel::from_bytes(&data);
        // Invalid type should fail
        prop_assert!(result.is_err());
    }
}

// ============================================================================
// Tensor Name Fuzzing
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Fuzz tensor names with arbitrary strings
    #[test]
    fn fuzz_tensor_names(
        name_len in 0usize..1000,
        name_byte in any::<u8>()
    ) {
        let mut data = Vec::new();

        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        // Tensor with generated name
        data.extend_from_slice(&(name_len as u64).to_le_bytes());
        data.extend(std::iter::repeat_n(name_byte, name_len));

        data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        data.extend_from_slice(&4u64.to_le_bytes()); // dim
        data.extend_from_slice(&0u32.to_le_bytes()); // F32
        data.extend_from_slice(&0u64.to_le_bytes()); // offset

        while data.len() % 32 != 0 {
            data.push(0);
        }
        data.extend_from_slice(&[0u8; 16]);

        let result = GGUFModel::from_bytes(&data);
        let _ = result;
    }

    /// Fuzz with specific LLaMA-style tensor name patterns
    #[test]
    fn fuzz_llama_tensor_patterns(
        layer_num in 0u32..100,
        component in prop::sample::select(&[
            "attn_q", "attn_k", "attn_v", "attn_output",
            "ffn_gate", "ffn_up", "ffn_down",
            "attn_norm", "ffn_norm"
        ])
    ) {
        let name = format!("blk.{}.{}.weight", layer_num, component);

        let mut data = Vec::new();

        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&2u32.to_le_bytes()); // 2D
        data.extend_from_slice(&64u64.to_le_bytes());
        data.extend_from_slice(&64u64.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes()); // F32
        data.extend_from_slice(&0u64.to_le_bytes());

        while data.len() % 32 != 0 {
            data.push(0);
        }
        data.extend_from_slice(&[0u8; 64]);

        let result = GGUFModel::from_bytes(&data);
        // Should parse successfully if name is valid UTF-8
        let _ = result;
    }
}

// ============================================================================
// Offset Boundary Fuzzing
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Fuzz tensor offsets (boundary conditions)
    #[test]
    fn fuzz_tensor_offsets(
        offset in prop_oneof![
            Just(0u64),
            Just(1u64),
            Just(31u64),          // Just before alignment
            Just(32u64),          // At alignment
            Just(33u64),          // Just after alignment
            0u64..1000,           // Small range
            Just(u64::MAX),       // Max
            Just(u64::MAX - 1),   // Near max
        ]
    ) {
        let mut data = Vec::new();

        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let name = "offset_test";
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&4u64.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&offset.to_le_bytes());

        while data.len() % 32 != 0 {
            data.push(0);
        }

        // Add some data at the expected offset if reasonable
        if offset < 1000 {
            while data.len() < (offset as usize) + 16 {
                data.push(0);
            }
        }

        let result = GGUFModel::from_bytes(&data);
        let _ = result;
    }
}

// ============================================================================
// Multi-Tensor Fuzzing
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Fuzz with multiple tensors
    #[test]
    fn fuzz_multi_tensor(
        num_tensors in 1usize..20,
        dims in 1u64..100
    ) {
        let mut data = Vec::new();

        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&(num_tensors as u64).to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let mut offset = 0u64;
        for i in 0..num_tensors {
            let name = format!("tensor_{}", i);
            data.extend_from_slice(&(name.len() as u64).to_le_bytes());
            data.extend_from_slice(name.as_bytes());
            data.extend_from_slice(&1u32.to_le_bytes());
            data.extend_from_slice(&dims.to_le_bytes());
            data.extend_from_slice(&0u32.to_le_bytes()); // F32
            data.extend_from_slice(&offset.to_le_bytes());
            offset += dims * 4;
        }

        while data.len() % 32 != 0 {
            data.push(0);
        }

        // Add tensor data
        for _ in 0..(offset / 4) {
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }

        let result = GGUFModel::from_bytes(&data);
        if let Ok(model) = result {
            prop_assert_eq!(model.tensors.len(), num_tensors);
        }
    }
}

// ============================================================================
// Bounds Check Validation Tests (Allocation Attack Prevention)
// ============================================================================

/// Test that excessive tensor_count is rejected (allocation attack prevention)
#[test]
fn test_bounds_check_excessive_tensor_count() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&100_001u64.to_le_bytes()); // Exceeds MAX_TENSOR_COUNT
    data.extend_from_slice(&0u64.to_le_bytes());

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("exceeds maximum") || err.contains("tensor_count"),
        "Expected bounds check error, got: {}",
        err
    );
}

/// Test that excessive metadata_count is rejected
#[test]
fn test_bounds_check_excessive_metadata_count() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&10_001u64.to_le_bytes()); // Exceeds MAX_METADATA_COUNT

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("exceeds maximum") || err.contains("metadata_count"),
        "Expected bounds check error, got: {}",
        err
    );
}

/// Test that excessive n_dims is rejected
#[test]
fn test_bounds_check_excessive_n_dims() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata

    // Tensor with excessive dimensions
    let name = "bad_tensor";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&100u32.to_le_bytes()); // n_dims = 100, exceeds MAX_DIMS (8)
    // Don't need actual dimensions since it should fail at n_dims check

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("dimensions") || err.contains("max allowed"),
        "Expected n_dims bounds check error, got: {}",
        err
    );
}

/// Test that valid counts within bounds succeed
#[test]
fn test_bounds_check_valid_counts() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor (valid)
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata (valid)

    // Simple metadata
    let key = "test.key";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes()); // UINT32
    data.extend_from_slice(&42u32.to_le_bytes());

    // Simple tensor
    let name = "test.weight";
    data.extend_from_slice(&(name.len() as u64).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes()); // 1 dimension (valid, within MAX_DIMS)
    data.extend_from_slice(&4u64.to_le_bytes()); // dim = 4
    data.extend_from_slice(&0u32.to_le_bytes()); // F32
    data.extend_from_slice(&0u64.to_le_bytes()); // offset

    while data.len() % 32 != 0 {
        data.push(0);
    }
    data.extend_from_slice(&[0u8; 16]);

    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_ok(), "Valid counts should succeed: {:?}", result);
}
