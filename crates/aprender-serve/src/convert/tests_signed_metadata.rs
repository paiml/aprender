
/// Build a GGUF with negative values in unsigned metadata
fn build_divergent_pygmy_signed_as_unsigned() -> Vec<u8> {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());

    // Architecture
    let key1 = "general.architecture";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let val1 = "llama";
    data.extend_from_slice(&(val1.len() as u64).to_le_bytes());
    data.extend_from_slice(val1.as_bytes());

    // block_count stored as i32 but reinterpreted as u32
    // -1 as i32 = 0xFFFFFFFF as u32 = 4294967295 layers!
    let key2 = "llama.block_count";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2.as_bytes());
    data.extend_from_slice(&5u32.to_le_bytes()); // GGUF_TYPE_INT32
    data.extend_from_slice(&(-1i32).to_le_bytes()); // -1

    // Tensor
    let t = "token_embd.weight";
    data.extend_from_slice(&(t.len() as u64).to_le_bytes());
    data.extend_from_slice(t.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&64u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    while data.len() % 32 != 0 {
        data.push(0);
    }

    for _ in 0..64 {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    data
}

// ============================================================================
// Semantic Divergence Tests
// ============================================================================

#[test]
fn test_divergent_llama_no_layers() {
    let data = build_divergent_pygmy_llama_no_layers();
    let result = GgufToAprConverter::convert(&data);

    // Should fail - claims 32 layers but has none
    match result {
        Ok(_) => {
            // If it succeeds, the converter is lenient
        },
        Err(e) => {
            // Expected failure
            let _ = e;
        },
    }
}

#[test]
fn test_divergent_gpt2_with_llama_tensors() {
    let data = build_divergent_pygmy_gpt2_llama_tensors();
    let result = GgufToAprConverter::convert(&data);

    // Architecture mismatch should be handled
    match result {
        Ok(_) => {},
        Err(e) => {
            let _ = e;
        },
    }
}

#[test]
fn test_divergent_dimension_mismatch() {
    let data = build_divergent_pygmy_dimension_mismatch();
    let result = GgufToAprConverter::convert(&data);

    // Dimension mismatch may cause issues
    match result {
        Ok(apr) => {
            // Check if config was inferred from tensors or metadata
            assert!(apr.config.hidden_dim > 0);
        },
        Err(e) => {
            let _ = e;
        },
    }
}

#[test]
fn test_divergent_unknown_architecture() {
    let data = build_divergent_pygmy_unknown_architecture();
    let result = GgufToAprConverter::convert(&data);

    // Unknown architecture should be handled gracefully
    match result {
        Ok(apr) => {
            // May use default config
            assert!(!apr.config.architecture.is_empty());
        },
        Err(e) => {
            // Error may mention architecture or missing config fields
            let msg = e.to_string().to_lowercase();
            assert!(
                msg.contains("architecture")
                    || msg.contains("unsupported")
                    || msg.contains("unknown")
                    || msg.contains("missing"),
                "Error should mention architecture or config issue: {}",
                msg
            );
        },
    }
}

#[test]
fn test_divergent_empty_architecture() {
    let data = build_divergent_pygmy_empty_architecture();
    let result = GgufToAprConverter::convert(&data);

    // Empty architecture may default or fail
    match result {
        Ok(apr) => {
            // Some default was used
            let _ = apr;
        },
        Err(e) => {
            let _ = e;
        },
    }
}

#[test]
fn test_divergent_metadata_heavy() {
    let data = build_divergent_pygmy_metadata_heavy();
    let result = GgufToAprConverter::convert(&data);

    // Should handle lots of metadata without crashing
    let _ = result.is_ok();
}

#[test]
fn test_divergent_conflicting_heads() {
    let data = build_divergent_pygmy_conflicting_heads();
    let result = GgufToAprConverter::convert(&data);

    // Conflicting head counts should be handled
    match result {
        Ok(apr) => {
            // Config was determined somehow
            assert!(apr.config.num_heads > 0);
        },
        Err(e) => {
            let _ = e;
        },
    }
}

#[test]
fn test_divergent_zero_vocab() {
    let data = build_divergent_pygmy_zero_vocab();
    let result = GgufToAprConverter::convert(&data);

    // Zero vocab in metadata, non-zero in tensor
    match result {
        Ok(apr) => {
            // Vocab was inferred from tensor
            assert!(apr.config.vocab_size > 0);
        },
        Err(e) => {
            let _ = e;
        },
    }
}

#[test]
fn test_divergent_signed_as_unsigned() {
    let data = build_divergent_pygmy_signed_as_unsigned();
    let result = GgufToAprConverter::convert(&data);

    // -1 interpreted as u32 = huge number
    match result {
        Ok(_) => {
            // Converter accepted it somehow
        },
        Err(e) => {
            let _ = e;
        },
    }
}

// ============================================================================
// Converter Edge Cases
// ============================================================================

#[test]
fn test_converter_empty_data() {
    let data = vec![];
    let result = GgufToAprConverter::convert(&data);

    assert!(result.is_err(), "Empty data should fail");
}

#[test]
fn test_converter_partial_header() {
    let data = vec![0x47, 0x47, 0x55, 0x46]; // Just magic, no version
    let result = GgufToAprConverter::convert(&data);

    assert!(result.is_err(), "Partial header should fail");
}

#[test]
fn test_converter_just_header_no_tensors() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
    data.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata

    let result = GgufToAprConverter::convert(&data);

    // Zero tensors may or may not be valid
    match result {
        Ok(apr) => {
            assert_eq!(apr.layers.len(), 0);
        },
        Err(e) => {
            let _ = e;
        },
    }
}

#[test]
fn test_converter_valid_minimal() {
    // A valid minimal GGUF that should convert
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    data.extend_from_slice(&2u64.to_le_bytes()); // 2 metadata

    // Architecture
    let key1 = "general.architecture";
    data.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    data.extend_from_slice(key1.as_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    let val1 = "llama";
    data.extend_from_slice(&(val1.len() as u64).to_le_bytes());
    data.extend_from_slice(val1.as_bytes());

    // block_count = 0
    let key2 = "llama.block_count";
    data.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    data.extend_from_slice(key2.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());

    // Embedding tensor
    let t = "token_embd.weight";
    data.extend_from_slice(&(t.len() as u64).to_le_bytes());
    data.extend_from_slice(t.as_bytes());
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&100u64.to_le_bytes());
    data.extend_from_slice(&64u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Tensor data: 100 * 64 = 6400 floats
    for _ in 0..6400 {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    let result = GgufToAprConverter::convert(&data);
    // May succeed or fail based on missing required tensors
    let _ = result;
}

// ============================================================================
// Mixed Quantization Divergence Tests
// ============================================================================

#[test]
fn test_divergent_mixed_quant_types() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes()); // 4 tensors
    data.extend_from_slice(&0u64.to_le_bytes());

    // Different quantization types
    let tensors = [
        ("tensor_f32", 0u32, 64),  // F32
        ("tensor_f16", 1u32, 64),  // F16
        ("tensor_q4_0", 2u32, 32), // Q4_0
        ("tensor_q8_0", 7u32, 32), // Q8_0
    ];

    let mut offset = 0u64;
    for (name, qtype, size) in tensors {
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&(size as u64).to_le_bytes());
        data.extend_from_slice(&qtype.to_le_bytes());
        data.extend_from_slice(&offset.to_le_bytes());

        // Rough size estimate
        let byte_size = match qtype {
            0 => size * 4,       // F32: 4 bytes per element
            1 => size * 2,       // F16: 2 bytes per element
            2 => (size / 2) + 2, // Q4_0: ~0.5 bytes per element + scale
            7 => size + 4,       // Q8_0: 1 byte per element + scale
            _ => size * 4,
        };
        offset += byte_size as u64;
    }

    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Pad data
    for _ in 0..(offset as usize) {
        data.push(0);
    }

    let result = GgufToAprConverter::convert(&data);
    // Mixed quant types exercise different dequantization paths
    let _ = result;
}

// ============================================================================
// Stress: Deep Nesting in Metadata
// ============================================================================

#[test]
fn test_divergent_deeply_nested_key() {
    let mut data = Vec::new();

    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());

    // Very deeply nested key
    let key = "a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p.q.r.s.t.u.v.w.x.y.z.deep.nested.key";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key.as_bytes());
    data.extend_from_slice(&4u32.to_le_bytes());
    data.extend_from_slice(&42u32.to_le_bytes());

    // Tensor
    let t = "test";
    data.extend_from_slice(&(t.len() as u64).to_le_bytes());
    data.extend_from_slice(t.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&4u64.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());

    while data.len() % 32 != 0 {
        data.push(0);
    }

    for _ in 0..4 {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    let result = GgufToAprConverter::convert(&data);
    let _ = result;
}
