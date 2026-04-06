
// =============================================================================
// Data Offset and Alignment Tests
// =============================================================================

#[test]
fn test_safetensors_data_offsets() {
    // Verify data_offsets in metadata are correct
    let f32_data1 = vec![1.0f32; 4]; // 16 bytes
    let f32_data2 = vec![2.0f32; 8]; // 32 bytes

    let data = SafetensorsBuilder::new()
        .add_f32_tensor("first", &[2, 2], &f32_data1)
        .add_f32_tensor("second", &[4, 2], &f32_data2)
        .build();

    let header_len = u64::from_le_bytes(data[0..8].try_into().expect("type conversion")) as usize;
    let json_str = std::str::from_utf8(&data[8..8 + header_len]).expect("valid UTF-8");

    // Parse JSON to verify offsets
    let metadata: serde_json::Value = serde_json::from_str(json_str).expect("valid JSON");

    let first_offsets = &metadata["first"]["data_offsets"];
    assert_eq!(first_offsets[0], 0);
    assert_eq!(first_offsets[1], 16);

    let second_offsets = &metadata["second"]["data_offsets"];
    assert_eq!(second_offsets[0], 16);
    assert_eq!(second_offsets[1], 48);
}

#[test]
fn test_apr_64_byte_alignment() {
    // APR data should be 64-byte aligned
    let f32_data = create_f32_embedding_data(10, 8); // 320 bytes

    let data = AprBuilder::new()
        .add_f32_tensor("tensor", &[10, 8], &f32_data)
        .build();

    // Header is 64 bytes
    assert!(data.len() >= 64);

    // Data offset should be 64-byte aligned
    let data_offset = u64::from_le_bytes(data[32..40].try_into().unwrap_or([0; 8]));
    assert_eq!(data_offset % 64, 0);
}

// =============================================================================
// Metadata Serialization Tests
// =============================================================================

#[test]
fn test_apr_metadata_json_serialization() {
    let data = AprBuilder::new()
        .architecture("mistral")
        .hidden_dim(4096)
        .num_layers(32)
        .build();

    // Read metadata offset and size from header
    let metadata_offset = u64::from_le_bytes(data[12..20].try_into().expect("type conversion")) as usize;
    let metadata_size = u32::from_le_bytes(data[20..24].try_into().expect("type conversion")) as usize;

    // Extract and parse metadata JSON
    let json_bytes = &data[metadata_offset..metadata_offset + metadata_size];
    let json_str = std::str::from_utf8(json_bytes).expect("valid UTF-8");
    let metadata: serde_json::Value = serde_json::from_str(json_str).expect("valid JSON");

    assert_eq!(metadata["architecture"], "mistral");
    assert_eq!(metadata["hidden_dim"], 4096);
    assert_eq!(metadata["num_layers"], 32);
}
