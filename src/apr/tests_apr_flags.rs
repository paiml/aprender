//! Additional tests for APR module - Part 02
//!
//! This file contains comprehensive tests for uncovered code paths in mod.rs.
//! Focus: APR format loading edge cases, tensor extraction, error handling.

#[cfg(test)]
mod tests {
    use crate::apr::*;

    #[test]
    fn test_apr_flags_sharded_flag() {
        let flags = AprFlags::new(AprFlags::SHARDED);
        // SHARDED flag (0x0010) - indicates model split across files
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());
        assert!(!flags.is_quantized());
    }

    #[test]
    fn test_apr_flags_all_flags_combined() {
        // Combine all flags
        let all_flags = AprFlags::LZ4_COMPRESSED
            | AprFlags::ZSTD_COMPRESSED
            | AprFlags::ENCRYPTED
            | AprFlags::SIGNED
            | AprFlags::SHARDED
            | AprFlags::QUANTIZED
            | AprFlags::HAS_VOCAB;

        let flags = AprFlags::new(all_flags);
        assert!(flags.is_compressed());
        assert!(flags.is_lz4());
        assert!(flags.is_zstd());
        assert!(flags.is_encrypted());
        assert!(flags.is_quantized());
        assert!(flags.has_vocab());
    }

    #[test]
    fn test_apr_flags_constants_values() {
        assert_eq!(AprFlags::LZ4_COMPRESSED, 0x0001);
        assert_eq!(AprFlags::ZSTD_COMPRESSED, 0x0002);
        assert_eq!(AprFlags::ENCRYPTED, 0x0004);
        assert_eq!(AprFlags::SIGNED, 0x0008);
        assert_eq!(AprFlags::SHARDED, 0x0010);
        assert_eq!(AprFlags::QUANTIZED, 0x0020);
        assert_eq!(AprFlags::HAS_VOCAB, 0x0200);
    }

    // =========================================================================
    // Dequantization Edge Cases
    // =========================================================================

    #[test]
    fn test_dequantize_q8_0_with_negative_values() {
        // Q8_0 with negative int8 values
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00;
        bytes[1] = 0x3C; // Scale = 1.0
                         // Set some values as negative (using two's complement)
        bytes[2] = 0xFF; // -1 as i8
        bytes[3] = 0xFE; // -2 as i8
        bytes[4] = 0x80; // -128 as i8

        let result = crate::apr::dequantize_q8_0(&bytes, 32);
        assert_eq!(result.len(), 32);
        // -1 * 1.0 = -1.0
        assert!((result[0] + 1.0).abs() < 0.5);
    }

    #[test]
    fn test_dequantize_f16_large_positive_value() {
        // Large positive f16 value (close to max: 65504)
        // f16 for ~32768 = 0x7800
        let bytes = vec![0x00, 0x78];
        let result = crate::apr::dequantize_f16(&bytes, 1);
        assert_eq!(result.len(), 1);
        assert!(result[0] > 30000.0);
    }

    #[test]
    fn test_dequantize_f16_large_negative_value() {
        // Large negative f16 value
        // f16 for ~-32768 = 0xF800
        let bytes = vec![0x00, 0xF8];
        let result = crate::apr::dequantize_f16(&bytes, 1);
        assert_eq!(result.len(), 1);
        assert!(result[0] < -30000.0);
    }

    // =========================================================================
    // f16_to_f32 Additional Edge Cases
    // =========================================================================

    #[test]
    fn test_f16_to_f32_max_subnormal_value() {
        // Maximum subnormal f16 = 0x03FF (exp=0, mantissa=0x3FF)
        let result = crate::apr::f16_to_f32(0x03FF);
        assert!(result > 0.0);
        // Max subnormal is approximately 6.0975e-5
        assert!(result < 1e-4);
    }

    #[test]
    fn test_f16_to_f32_negative_subnormal_value() {
        // Negative subnormal f16 = 0x8001 (sign=1, exp=0, mantissa=1)
        let result = crate::apr::f16_to_f32(0x8001);
        assert!(result < 0.0);
        assert!(result > -1e-4);
    }

    #[test]
    fn test_f16_to_f32_quiet_nan_value() {
        // Quiet NaN (exp=31, mantissa high bit set)
        let result = crate::apr::f16_to_f32(0x7E00);
        assert!(result.is_nan());
    }

    #[test]
    fn test_f16_to_f32_signaling_nan_value() {
        // Signaling NaN (exp=31, mantissa low bit set)
        let result = crate::apr::f16_to_f32(0x7C01);
        assert!(result.is_nan());
    }

    // =========================================================================
    // load_tokenizer_from_sibling Tests
    // =========================================================================

    #[test]
    fn test_load_tokenizer_from_sibling_nonexistent_path() {
        use std::path::Path;
        let result = AprV2Model::load_tokenizer_from_sibling(Path::new("/nonexistent/model.apr"));
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_no_tokenizer_file() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let model_path = dir.path().join("model.apr");

        // No tokenizer.json file exists
        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);
        assert!(result.is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_with_valid_tokenizer() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let tokenizer_path = dir.path().join("tokenizer.json");

        // Create a minimal tokenizer.json
        let tokenizer_json = r#"{
            "model": {
                "vocab": {
                    "hello": 0,
                    "world": 1
                },
                "merges": []
            },
            "added_tokens": [
                {"id": 0, "content": "<s>"},
                {"id": 1, "content": "</s>"}
            ]
        }"#;

        std::fs::write(&tokenizer_path, tokenizer_json).expect("write tokenizer");

        let model_path = dir.path().join("model.apr");
        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);

        assert!(result.is_some());
        let (vocab, bos, eos) = result.expect("tokenizer should load");
        assert!(!vocab.is_empty());
        // Added tokens should be recognized
        assert!(bos.is_some() || eos.is_some());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_load_tokenizer_from_sibling_malformed_json() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("create temp dir");
        let tokenizer_path = dir.path().join("tokenizer.json");

        // Write invalid JSON
        std::fs::write(&tokenizer_path, "{ invalid json }").expect("write tokenizer");

        let model_path = dir.path().join("model.apr");
        let result = AprV2Model::load_tokenizer_from_sibling(&model_path);

        // Should return None for invalid JSON
        assert!(result.is_none());
    }

    // =========================================================================
    // encode_text Tests
    // =========================================================================

    #[test]
    fn test_encode_text_nonexistent_tokenizer() {
        use std::path::Path;
        let result = AprV2Model::encode_text(Path::new("/nonexistent/model.apr"), "test");
        assert!(result.is_none());
    }

    // =========================================================================
    // load_tokenizer Tests
    // =========================================================================

    #[test]
    fn test_load_tokenizer_nonexistent_path() {
        use std::path::Path;
        let result = AprV2Model::load_tokenizer(Path::new("/nonexistent/model.apr"));
        assert!(result.is_none());
    }

    // =========================================================================
    // Header Checksum Field Test
    // =========================================================================

    #[test]
    fn test_apr_header_checksum_field_preservation() {
        let mut data = vec![0u8; HEADER_SIZE];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset
        data[40..44].copy_from_slice(&0xDEADBEEFu32.to_le_bytes()); // checksum

        let header = AprHeader::from_bytes(&data).expect("parse header");
        assert_eq!(header.checksum, 0xDEADBEEF);
    }

    // =========================================================================
    // Constants Verification
    // =========================================================================

    #[test]
    fn test_module_constants() {
        assert_eq!(HEADER_SIZE, 64);
        assert_eq!(ALIGNMENT, 64);
        assert_eq!(MAGIC_PREFIX, [0x41, 0x50, 0x52]); // "APR"
        assert_eq!(MAGIC, [0x41, 0x50, 0x52, 0x00]); // "APR\0"
    }

    // =========================================================================
    // ModelData Tests
    // =========================================================================

    #[test]
    fn test_model_data_from_vec_empty() {
        let data = ModelData::from_vec(vec![]);
        assert!(data.is_empty());
        assert_eq!(data.len(), 0);
        assert!(!data.is_mmap());
    }

    #[test]
    fn test_model_data_from_vec_nonempty() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);
        assert!(!data.is_empty());
        assert_eq!(data.len(), 5);
        assert_eq!(data.as_slice(), &[1, 2, 3, 4, 5]);
        assert!(!data.is_mmap());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_model_data_open_mmap() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"test data for mmap").expect("write data");

        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");
        assert!(!model_data.is_empty());
        assert!(model_data.is_mmap());
        assert_eq!(model_data.len(), 18); // "test data for mmap".len()
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_model_data_open_mmap_nonexistent() {
        let result = ModelData::open_mmap("/nonexistent/path/file.apr");
        assert!(result.is_err());
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_model_data_release_cpu_pages_heap() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4]);
        // release_cpu_pages should be no-op for heap data
        let result = data.release_cpu_pages();
        assert!(result.is_ok());
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_model_data_advise_sequential_heap() {
        let data = ModelData::from_vec(vec![1, 2, 3, 4]);
        // advise_sequential should be no-op for heap data
        let result = data.advise_sequential();
        assert!(result.is_ok());
    }

    // =========================================================================
    // TensorEntry Debug Trait
    // =========================================================================

    #[test]
    fn test_tensor_entry_debug_format() {
        let entry = TensorEntry {
            name: "test.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            offset: 0,
            size: 800,
        };
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("test.weight"));
        assert!(debug_str.contains("F32"));
    }

    // =========================================================================
    // AprHeader Debug Trait
    // =========================================================================

    #[test]
    fn test_apr_header_debug_format() {
        let mut data = vec![0u8; HEADER_SIZE];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        let header = AprHeader::from_bytes(&data).expect("parse header");
        let debug_str = format!("{:?}", header);
        assert!(debug_str.contains("AprHeader"));
    }

    // =========================================================================
    // AprMetadata is_transformer Edge Cases
    // =========================================================================

    #[test]
    fn test_apr_metadata_is_transformer_all_none() {
        let meta = AprMetadata::default();
        assert!(!meta.is_transformer());
    }

    #[test]
    fn test_apr_metadata_is_transformer_partial_fields() {
        // Only hidden_size set - not a transformer
        let meta = AprMetadata {
            hidden_size: Some(256),
            ..Default::default()
        };
        assert!(!meta.is_transformer());

        // hidden_size + num_layers - still not a transformer
        let meta2 = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            ..Default::default()
        };
        assert!(!meta2.is_transformer());

        // hidden_size + num_layers + num_heads - still need vocab_size
        let meta3 = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: Some(8),
            ..Default::default()
        };
        assert!(!meta3.is_transformer());
    }

    #[test]
    fn test_apr_metadata_is_transformer_complete() {
        // All required fields set
        let meta = AprMetadata {
            hidden_size: Some(256),
            num_layers: Some(4),
            num_heads: Some(8),
            vocab_size: Some(32000),
            ..Default::default()
        };
        assert!(meta.is_transformer());
    }

    // =========================================================================
    // AprV2Model from_bytes Error Paths
    // =========================================================================

    #[test]
    fn test_apr_v2_model_from_bytes_compressed_lz4() {
        let mut data = vec![0u8; HEADER_SIZE + 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        // Set LZ4 compressed flag
        data[6] = (AprFlags::LZ4_COMPRESSED & 0xFF) as u8;
        data[7] = (AprFlags::LZ4_COMPRESSED >> 8) as u8;

        let result = AprV2Model::from_bytes(data);
        // Compressed models should error (decompression not supported in from_bytes)
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_from_bytes_compressed_zstd() {
        let mut data = vec![0u8; HEADER_SIZE + 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        // Set ZSTD compressed flag
        data[6] = (AprFlags::ZSTD_COMPRESSED & 0xFF) as u8;
        data[7] = (AprFlags::ZSTD_COMPRESSED >> 8) as u8;

        let result = AprV2Model::from_bytes(data);
        // Compressed models should error
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_v2_model_from_bytes_encrypted() {
        let mut data = vec![0u8; HEADER_SIZE + 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        // Set encrypted flag
        data[6] = (AprFlags::ENCRYPTED & 0xFF) as u8;
        data[7] = (AprFlags::ENCRYPTED >> 8) as u8;

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.to_lowercase().contains("encrypt"));
    }
include!("tests_format_and_mapped_model.rs");
}
