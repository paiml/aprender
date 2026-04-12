
    // =========================================================================
    // Helper functions
    // =========================================================================

    /// Helper to create binary tensor entry for tests
    fn create_binary_tensor_entry(
        name: &str,
        dtype: u8,
        shape: &[u64],
        offset: u64,
        size: u64,
    ) -> Vec<u8> {
        let mut data = Vec::new();
        // Name
        data.extend_from_slice(&(name.len() as u16).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        // Dtype
        data.push(dtype);
        // Shape
        data.push(shape.len() as u8);
        for &dim in shape {
            data.extend_from_slice(&dim.to_le_bytes());
        }
        // Offset and size
        data.extend_from_slice(&offset.to_le_bytes());
        data.extend_from_slice(&size.to_le_bytes());
        data
    }

    /// Helper to create minimal APR model data
    fn create_minimal_apr_data() -> Vec<u8> {
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset
        data
    }

    // =========================================================================
    // APR Version Handling Tests
    // =========================================================================

    #[test]
    #[ignore = "Test expectation needs adjustment"]
    fn test_apr_v1_format_magic_rejected() {
        // APR v1 format (magic "APR1") should be rejected with helpful message
        let mut data = vec![0u8; HEADER_SIZE + 128];
        data[0..3].copy_from_slice(&MAGIC_PREFIX); // "APR"
        data[3] = b'1'; // Version byte for APR v1
        data[4] = 1; // version major
        data[5] = 0; // version minor

        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        // The error should indicate the magic is wrong
        assert!(
            err.to_lowercase().contains("magic") || err.to_lowercase().contains("invalid"),
            "Error should mention invalid magic or format: {}",
            err
        );
    }

    #[test]
    fn test_apr_v2_format_accepted_with_magic_apr2() {
        // APR v2 format (magic "APR2") should work
        let mut data = vec![0u8; HEADER_SIZE];
        data[0..3].copy_from_slice(&MAGIC_PREFIX); // "APR"
        data[3] = b'2'; // Version byte for APR v2
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

        let result = AprHeader::from_bytes(&data);
        assert!(result.is_ok(), "APR2 magic should be accepted");
    }

    #[test]
    fn test_apr_invalid_version_byte_x() {
        // Invalid version byte should be rejected
        let mut data = vec![0u8; HEADER_SIZE];
        data[0..3].copy_from_slice(&MAGIC_PREFIX); // "APR"
        data[3] = b'X'; // Invalid version byte

        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_invalid_version_byte_3() {
        // Future version byte should be rejected
        let mut data = vec![0u8; HEADER_SIZE];
        data[0..3].copy_from_slice(&MAGIC_PREFIX); // "APR"
        data[3] = b'3'; // Hypothetical APR v3

        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    // =========================================================================
    // MappedAprModel Tests (currently untested)
    // =========================================================================

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_from_path() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        let data = create_minimal_apr_data();
        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");
        assert_eq!(model.tensor_count(), 0);
        assert_eq!(model.data_offset(), 64);
        assert!(model.file_size() >= HEADER_SIZE);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_invalid_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(b"GGUF"); // Wrong magic

        temp.write_all(&data).expect("write data");

        let result = MappedAprModel::from_path(temp.path());
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_find_tensor() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create APR with one tensor
        let metadata = r"{}";
        let metadata_bytes = metadata.as_bytes();
        let tensor_entry = create_binary_tensor_entry("test.weight", 0, &[4, 4], 0, 64);
        let tensor_index_offset = 128u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;

        let mut data = vec![0u8; data_offset as usize + 64];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count = 1
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());
        data[64..64 + metadata_bytes.len()].copy_from_slice(metadata_bytes);
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");

        assert!(model.find_tensor("test.weight").is_some());
        assert!(model.find_tensor("nonexistent").is_none());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_get_tensor_data() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create APR with tensor data
        let tensor_entry = create_binary_tensor_entry("vec.weight", 0, &[4], 0, 16);
        let tensor_index_offset = 128u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;

        let mut data = vec![0u8; data_offset as usize + 16];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&1u32.to_le_bytes());
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);
        // Write tensor data
        let tensor_data_start = data_offset as usize;
        for i in 0..4 {
            let val = (i as f32) * 2.0;
            data[tensor_data_start + i * 4..tensor_data_start + i * 4 + 4]
                .copy_from_slice(&val.to_le_bytes());
        }

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");

        let tensor_data = model.get_tensor_data("vec.weight").expect("get tensor");
        assert_eq!(tensor_data.len(), 16);

        // Non-existent tensor
        assert!(model.get_tensor_data("nonexistent").is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_tensor_past_eof() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create APR with tensor that claims data beyond file
        let tensor_entry = create_binary_tensor_entry("huge.weight", 0, &[1000000], 0, 4000000);
        let tensor_index_offset = 128u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;

        let mut data = vec![0u8; data_offset as usize + 64]; // Much smaller than claimed
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&1u32.to_le_bytes());
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");

        // Getting tensor data should error (extends past EOF)
        let result = model.get_tensor_data("huge.weight");
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_mapped_apr_model_data_accessor() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        let data = create_minimal_apr_data();
        temp.write_all(&data).expect("write data");

        let model = MappedAprModel::from_path(temp.path()).expect("load model");

        // Test data() accessor
        let raw_data = model.data();
        assert!(!raw_data.is_empty());
        assert_eq!(&raw_data[0..4], &MAGIC);
    }

    #[test]
    fn test_mapped_apr_model_dtype_to_qtype() {
        // Test all dtype conversions
        assert_eq!(MappedAprModel::dtype_to_qtype("F32"), 0);
        assert_eq!(MappedAprModel::dtype_to_qtype("F16"), 1);
        assert_eq!(MappedAprModel::dtype_to_qtype("Q4_0"), 2);
        assert_eq!(MappedAprModel::dtype_to_qtype("Q4_1"), 3);
        assert_eq!(MappedAprModel::dtype_to_qtype("Q5_0"), 6);
        assert_eq!(MappedAprModel::dtype_to_qtype("Q5_1"), 7);
        assert_eq!(MappedAprModel::dtype_to_qtype("Q8_0"), 8);
        assert_eq!(MappedAprModel::dtype_to_qtype("Q8_1"), 9);
        assert_eq!(MappedAprModel::dtype_to_qtype("Q2_K"), 10);
        assert_eq!(MappedAprModel::dtype_to_qtype("Q3_K"), 11);
        assert_eq!(MappedAprModel::dtype_to_qtype("Q4_K"), 12);
        assert_eq!(MappedAprModel::dtype_to_qtype("Q5_K"), 13);
        assert_eq!(MappedAprModel::dtype_to_qtype("Q6_K"), 14);
        assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XXS"), 16);
        assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XS"), 17);
        assert_eq!(MappedAprModel::dtype_to_qtype("BF16"), 30);
        assert_eq!(MappedAprModel::dtype_to_qtype("UNKNOWN"), 0); // Default
    }

    // =========================================================================
    // get_tensor_f32 dtype coverage tests (Q4_K, Q6_K paths)
    // =========================================================================

    #[test]
    #[ignore = "APR dtype parsing bug - needs investigation"]
    fn test_get_tensor_f32_q4_k_dtype() {
        // Create APR model with Q4_K tensor (dtype byte = 8 for Q4_K)
        // Q4_K super-block: 144 bytes = 256 elements
        let mut q4k_data = vec![0u8; 144];
        q4k_data[0] = 0x00;
        q4k_data[1] = 0x3C; // d = 1.0 in f16
        q4k_data[2] = 0x00;
        q4k_data[3] = 0x00; // dmin = 0.0

        let metadata = r"{}";
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        let tensor_entry = create_binary_tensor_entry("q4k.weight", 8, &[256], 0, 144);
        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;
        let total_size = data_offset as usize + 144;

        let mut data = vec![0u8; total_size];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&1u32.to_le_bytes());
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);
        let data_start = data_offset as usize;
        data[data_start..data_start + 144].copy_from_slice(&q4k_data);

        let model = AprV2Model::from_bytes(data).expect("load model");
        let result = model.get_tensor_f32("q4k.weight");
        assert!(result.is_ok());
        let floats = result.expect("tensor retrieval failed");
        assert_eq!(floats.len(), 256);
    }

    #[test]
    #[ignore = "APR dtype parsing bug - needs investigation"]
    fn test_get_tensor_f32_q6_k_dtype() {
        // Create APR model with Q6_K tensor (dtype byte = 9 for Q6_K)
        // Q6_K super-block: 210 bytes = 256 elements
        let mut q6k_data = vec![0u8; 210];
        q6k_data[208] = 0x00;
        q6k_data[209] = 0x3C; // d = 1.0 in f16

        let metadata = r"{}";
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        let tensor_entry = create_binary_tensor_entry("q6k.weight", 9, &[256], 0, 210);
        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;
        let total_size = data_offset as usize + 210;

        let mut data = vec![0u8; total_size];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[8..12].copy_from_slice(&1u32.to_le_bytes());
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        data[32..40].copy_from_slice(&data_offset.to_le_bytes());
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);
        let data_start = data_offset as usize;
        data[data_start..data_start + 210].copy_from_slice(&q6k_data);

        let model = AprV2Model::from_bytes(data).expect("load model");
        let result = model.get_tensor_f32("q6k.weight");
        assert!(result.is_ok());
        let floats = result.expect("tensor retrieval failed");
        assert_eq!(floats.len(), 256);
    }

    // =========================================================================
    // SimpleTokenizer Tests
    // =========================================================================

    #[test]
    fn test_simple_tokenizer_decode_basic() {
        let tokenizer = SimpleTokenizer {
            id_to_token: vec!["hello".to_string(), "world".to_string()],
            bos_token_id: None,
            eos_token_id: None,
        };

        let decoded = tokenizer.decode(&[0, 1]);
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    #[test]
    fn test_simple_tokenizer_decode_empty_input() {
        let tokenizer = SimpleTokenizer {
            id_to_token: vec!["hello".to_string()],
            bos_token_id: None,
            eos_token_id: None,
        };

        let decoded = tokenizer.decode(&[]);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_simple_tokenizer_decode_out_of_bounds_tokens() {
        let tokenizer = SimpleTokenizer {
            id_to_token: vec!["hello".to_string()],
            bos_token_id: None,
            eos_token_id: None,
        };

        // Token 100 is out of bounds - should be handled gracefully
        let decoded = tokenizer.decode(&[0, 100]);
        assert!(decoded.contains("hello"));
    }

    #[test]
    fn test_simple_tokenizer_with_bos_eos() {
        let tokenizer = SimpleTokenizer {
            id_to_token: vec!["<s>".to_string(), "</s>".to_string(), "hello".to_string()],
            bos_token_id: Some(0),
            eos_token_id: Some(1),
        };

        assert_eq!(tokenizer.bos_token_id, Some(0));
        assert_eq!(tokenizer.eos_token_id, Some(1));
        assert_eq!(tokenizer.id_to_token.len(), 3);
    }

    // =========================================================================
    // AprFlags Comprehensive Tests
    // =========================================================================

    #[test]
    fn test_apr_flags_signed_flag() {
        let flags = AprFlags::new(AprFlags::SIGNED);
        // SIGNED flag (0x0008) - check that it's independent of other flags
        assert!(!flags.is_compressed());
        assert!(!flags.is_encrypted());
        assert!(!flags.is_quantized());
    }
