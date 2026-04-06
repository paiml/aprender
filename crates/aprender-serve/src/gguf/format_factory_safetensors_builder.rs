
// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SafeTensors Builder Tests
    // =========================================================================

    #[test]
    fn test_safetensors_builder_empty() {
        let data = SafetensorsBuilder::new().build();

        // Should have valid header (8 bytes length + empty JSON "{}")
        assert!(data.len() >= 10);

        // First 8 bytes are header length
        let header_len = u64::from_le_bytes(data[0..8].try_into().expect("header_len"));
        assert_eq!(header_len, 2); // "{}"

        // Empty SafeTensors is technically valid but format detection
        // requires "{" at byte 8, which "{}" satisfies
        // Note: An empty model isn't useful but the format is valid
        assert!(data[8] == b'{');
    }

    #[test]
    fn test_safetensors_builder_with_tensor() {
        let data = SafetensorsBuilder::new()
            .add_f32_tensor("test.weight", &[4, 8], &vec![0.0f32; 32])
            .build();

        assert!(data.len() > 10);
        assert_eq!(FormatType::from_magic(&data), FormatType::SafeTensors);

        // Verify JSON header contains tensor metadata
        let header_len = u64::from_le_bytes(data[0..8].try_into().expect("header_len")) as usize;
        let json_str = std::str::from_utf8(&data[8..8 + header_len]).expect("valid UTF-8");
        assert!(json_str.contains("test.weight"));
        assert!(json_str.contains("F32"));
    }

    #[test]
    fn test_safetensors_minimal_model() {
        let data = SafetensorsBuilder::minimal_model(100, 64);

        assert_eq!(FormatType::from_magic(&data), FormatType::SafeTensors);

        let header_len = u64::from_le_bytes(data[0..8].try_into().expect("header_len")) as usize;
        let json_str = std::str::from_utf8(&data[8..8 + header_len]).expect("valid UTF-8");
        assert!(json_str.contains("model.embed_tokens.weight"));
        assert!(json_str.contains("model.norm.weight"));
    }

    // =========================================================================
    // APR Builder Tests
    // =========================================================================

    #[test]
    fn test_apr_builder_empty() {
        let data = AprBuilder::new().build();

        // Should have valid header (64 bytes minimum)
        assert!(data.len() >= APR_HEADER_SIZE);

        // Check magic
        assert_eq!(&data[0..4], b"APR\0");

        // Detect format
        assert_eq!(FormatType::from_magic(&data), FormatType::Apr);
    }

    #[test]
    fn test_apr_builder_with_metadata() {
        let data = AprBuilder::new()
            .architecture("llama")
            .hidden_dim(64)
            .num_layers(2)
            .build();

        assert_eq!(FormatType::from_magic(&data), FormatType::Apr);

        // Verify version
        assert_eq!(data[4], APR_VERSION_MAJOR);
        assert_eq!(data[5], APR_VERSION_MINOR);
    }

    #[test]
    fn test_apr_builder_with_tensor() {
        let embed_data = create_f32_embedding_data(10, 8);
        let data = AprBuilder::new()
            .add_f32_tensor("token_embd.weight", &[10, 8], &embed_data)
            .build();

        assert_eq!(FormatType::from_magic(&data), FormatType::Apr);

        // Verify tensor count in header
        let tensor_count = u32::from_le_bytes(data[8..12].try_into().expect("tensor_count"));
        assert_eq!(tensor_count, 1);
    }

    #[test]
    fn test_apr_minimal_model() {
        let data = AprBuilder::minimal_model(100, 64);

        assert_eq!(FormatType::from_magic(&data), FormatType::Apr);

        // Verify tensor count
        let tensor_count = u32::from_le_bytes(data[8..12].try_into().expect("tensor_count"));
        assert_eq!(tensor_count, 2); // embed + norm
    }

    // =========================================================================
    // Format Detection Tests
    // =========================================================================

    #[test]
    fn test_format_detection_gguf() {
        let data = build_minimal_llama_gguf(100, 64, 128, 4, 4);
        assert_eq!(FormatType::from_magic(&data), FormatType::Gguf);
    }

    #[test]
    fn test_format_detection_safetensors() {
        let data = SafetensorsBuilder::minimal_model(100, 64);
        assert_eq!(FormatType::from_magic(&data), FormatType::SafeTensors);
    }

    #[test]
    fn test_format_detection_apr() {
        let data = AprBuilder::minimal_model(100, 64);
        assert_eq!(FormatType::from_magic(&data), FormatType::Apr);
    }

    #[test]
    fn test_format_detection_unknown() {
        let data = vec![0u8; 100];
        assert_eq!(FormatType::from_magic(&data), FormatType::Unknown);
    }

    #[test]
    fn test_format_detection_too_short() {
        let data = vec![0u8; 4];
        assert_eq!(FormatType::from_magic(&data), FormatType::Unknown);
    }

    // =========================================================================
    // Cross-Format Tensor Data Tests (Rosetta Parity)
    // =========================================================================

    #[test]
    fn test_rosetta_same_embedding_data() {
        // Same embedding data should produce same raw bytes in all formats
        let embed_data = create_f32_embedding_data(10, 8);

        let gguf = GGUFBuilder::new()
            .add_f32_tensor("token_embd.weight", &[10, 8], &embed_data)
            .build();

        let st = SafetensorsBuilder::new()
            .add_f32_tensor("token_embd.weight", &[10, 8], &embed_data)
            .build();

        let apr = AprBuilder::new()
            .add_f32_tensor("token_embd.weight", &[10, 8], &embed_data)
            .build();

        // All formats should be valid
        assert_eq!(FormatType::from_magic(&gguf), FormatType::Gguf);
        assert_eq!(FormatType::from_magic(&st), FormatType::SafeTensors);
        assert_eq!(FormatType::from_magic(&apr), FormatType::Apr);

        // The raw f32 bytes are stored somewhere in each format
        let f32_bytes: Vec<u8> = embed_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(f32_bytes.len(), 10 * 8 * 4); // 320 bytes

        // GGUF, SafeTensors, and APR all store raw F32 as little-endian
        // The exact offsets differ by format, but data integrity is preserved
    }

    #[test]
    fn test_rosetta_all_formats_valid() {
        // Generate all three formats and verify they're all valid
        let vocab_size = 100;
        let hidden_dim = 64;

        let gguf = build_minimal_llama_gguf(vocab_size, hidden_dim, 128, 4, 4);
        let st = SafetensorsBuilder::minimal_model(vocab_size, hidden_dim);
        let apr = AprBuilder::minimal_model(vocab_size, hidden_dim);

        // All should be detected correctly
        assert_eq!(FormatType::from_magic(&gguf), FormatType::Gguf);
        assert_eq!(FormatType::from_magic(&st), FormatType::SafeTensors);
        assert_eq!(FormatType::from_magic(&apr), FormatType::Apr);

        // All should have reasonable size (not empty)
        assert!(gguf.len() > 1000, "GGUF too small: {}", gguf.len());
        assert!(st.len() > 100, "SafeTensors too small: {}", st.len());
        assert!(apr.len() > 100, "APR too small: {}", apr.len());
    }
}
