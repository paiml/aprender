
    #[test]
    fn test_magic_constant() {
        // ONE format: APR\0
        assert_eq!(MAGIC, [0x41, 0x50, 0x52, 0x00]);
        assert_eq!(&MAGIC, b"APR\0");
    }

    #[test]
    fn test_header_from_bytes_too_small() {
        let data = vec![0u8; 10];
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_from_bytes_invalid_magic() {
        let mut data = vec![0u8; HEADER_SIZE];
        data[0..4].copy_from_slice(b"GGUF");
        let result = AprHeader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_from_bytes_valid() {
        let mut data = vec![0u8; HEADER_SIZE];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[8..12].copy_from_slice(&10u32.to_le_bytes()); // tensor_count
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset

        let header = AprHeader::from_bytes(&data).expect("should parse");
        assert_eq!(header.magic, MAGIC);
        assert_eq!(header.version, (2, 0));
        assert_eq!(header.tensor_count, 10);
    }

    #[test]
    fn test_flags() {
        let flags = AprFlags::new(0x0007);
        assert!(flags.is_compressed());
        assert!(flags.is_encrypted());

        let flags2 = AprFlags::new(0x0020);
        assert!(flags2.is_quantized());
        assert!(!flags2.is_compressed());
    }

    #[test]
    fn test_detect_format_by_extension() {
        assert_eq!(detect_format("/fake/model.apr"), "apr");
        assert_eq!(detect_format("/fake/model.gguf"), "gguf");
        assert_eq!(detect_format("/fake/model.safetensors"), "safetensors");
    }

    // APR v2 binary tensor index format tests

    /// Helper to create binary tensor entry
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

    #[test]
    fn test_tensor_entry_from_binary_valid() {
        let data = create_binary_tensor_entry(
            "model.embed_tokens.weight",
            0,
            &[32000, 2048],
            0,
            262144000,
        );
        let (entry, consumed) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.name, "model.embed_tokens.weight");
        assert_eq!(entry.dtype, "F32");
        assert_eq!(entry.shape, vec![32000, 2048]);
        assert_eq!(entry.offset, 0);
        assert_eq!(entry.size, 262144000);
        assert_eq!(consumed, data.len());
    }

    #[test]
    fn test_tensor_entry_from_binary_f16() {
        let data = create_binary_tensor_entry(
            "layer.0.attn.q_proj.weight",
            1,
            &[2048, 2048],
            1024,
            8388608,
        );
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.dtype, "F16");
        assert_eq!(entry.shape, vec![2048, 2048]);
    }

    #[test]
    fn test_tensor_entry_from_binary_bf16() {
        // GH-191: byte 30 is BF16 in GGML dtype mapping (was byte 2 before GH-191 fix)
        let data = create_binary_tensor_entry("lm_head.weight", 30, &[32000, 2048], 512, 131072000);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.dtype, "BF16");
    }

    #[test]
    fn test_tensor_entry_from_binary_q4_0() {
        // GH-191: byte 2 is Q4_0 in GGML dtype mapping
        let data = create_binary_tensor_entry("quantized.weight", 2, &[1024, 1024], 0, 1048576);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.dtype, "Q4_0");
    }

    #[test]
    fn test_tensor_entry_from_binary_q4_1() {
        // GH-191: byte 3 is Q4_1 in GGML dtype mapping
        let data = create_binary_tensor_entry("quantized.weight", 3, &[1024, 1024], 0, 1048576);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.dtype, "Q4_1");
    }

    #[test]
    fn test_tensor_entry_from_binary_1d() {
        let data = create_binary_tensor_entry("model.norm.weight", 0, &[2048], 0, 8192);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.shape, vec![2048]);
        assert_eq!(entry.element_count(), 2048);
    }

    #[test]
    fn test_tensor_entry_from_binary_3d() {
        let data = create_binary_tensor_entry("conv.weight", 0, &[64, 3, 7], 0, 5376);
        let (entry, _) = TensorEntry::from_binary(&data).expect("should parse");

        assert_eq!(entry.shape, vec![64, 3, 7]);
        assert_eq!(entry.element_count(), 64 * 3 * 7);
    }

    #[test]
    fn test_tensor_entry_from_binary_too_short() {
        let data = vec![0u8; 2];
        let result = TensorEntry::from_binary(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_entry_from_binary_truncated_name() {
        let mut data = Vec::new();
        data.extend_from_slice(&100u16.to_le_bytes()); // name_len = 100
        data.extend_from_slice(b"short"); // Only 5 bytes of name
        let result = TensorEntry::from_binary(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_entry_from_binary_truncated_shape() {
        let mut data = Vec::new();
        data.extend_from_slice(&4u16.to_le_bytes()); // name_len
        data.extend_from_slice(b"test");
        data.push(0); // dtype
        data.push(2); // ndim = 2
        data.extend_from_slice(&1024u64.to_le_bytes()); // first dim only
        // Missing second dim, offset, size
        let result = TensorEntry::from_binary(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_entry_element_count() {
        let entry = TensorEntry {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![32, 64, 128],
            offset: 0,
            size: 0,
        };
        assert_eq!(entry.element_count(), 32 * 64 * 128);
    }

    #[test]
    fn test_tensor_entry_element_count_scalar() {
        let entry = TensorEntry {
            name: "scalar".to_string(),
            dtype: "F32".to_string(),
            shape: vec![],
            offset: 0,
            size: 0,
        };
        assert_eq!(entry.element_count(), 1);
    }

    #[test]
    fn test_multiple_tensor_entries_sequential() {
        let mut data = Vec::new();
        data.extend(create_binary_tensor_entry("tensor1", 0, &[100], 0, 400));
        data.extend(create_binary_tensor_entry(
            "tensor2",
            1,
            &[200, 300],
            400,
            120000,
        ));
        data.extend(create_binary_tensor_entry("tensor3", 2, &[50], 120400, 100));

        let mut pos = 0;
        let mut entries = Vec::new();

        while pos < data.len() {
            let (entry, consumed) = TensorEntry::from_binary(&data[pos..]).expect("should parse");
            entries.push(entry);
            pos += consumed;
        }

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, "tensor1");
        assert_eq!(entries[1].name, "tensor2");
        assert_eq!(entries[2].name, "tensor3");
        assert_eq!(entries[1].shape, vec![200, 300]);
    }

    // =========================================================================
    // Compression Tests (GH-35)
    // =========================================================================

    #[test]
    fn test_flags_lz4() {
        let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED);
        assert!(flags.is_lz4());
        assert!(!flags.is_zstd());
        assert!(flags.is_compressed());
    }

    #[test]
    fn test_flags_zstd() {
        let flags = AprFlags::new(AprFlags::ZSTD_COMPRESSED);
        assert!(!flags.is_lz4());
        assert!(flags.is_zstd());
        assert!(flags.is_compressed());
    }

    #[test]
    fn test_flags_no_compression() {
        let flags = AprFlags::new(0);
        assert!(!flags.is_lz4());
        assert!(!flags.is_zstd());
        assert!(!flags.is_compressed());
    }

    #[cfg(not(feature = "apr-compression"))]
    #[test]
    fn test_compressed_file_requires_feature() {
        // Create a minimal APR v2 header with LZ4 flag
        let mut data = vec![0u8; HEADER_SIZE + 100];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&(AprFlags::LZ4_COMPRESSED).to_le_bytes()); // LZ4 flag
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("apr-compression"),
            "Error should mention feature: {}",
            err_msg
        );
    }

    // ============ Additional coverage tests ============

    /// Helper to create a complete valid APR v2 model
    fn create_test_apr_model() -> Vec<u8> {
        let metadata = r#"{"architecture":"test","vocab_size":100,"hidden_size":64}"#;
        let metadata_bytes = metadata.as_bytes();
        let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

        // Binary tensor index entry for "test.weight"
        let tensor_entry = create_binary_tensor_entry("test.weight", 0, &[4, 4], 0, 64);

        let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
        let data_offset = tensor_index_offset + tensor_entry.len() as u64;
        let data_size = 64usize; // 16 floats * 4 bytes

        let total_size = data_offset as usize + data_size;
        let mut data = vec![0u8; total_size];

        // Header
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count = 1
        data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes()); // metadata_size
        data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&data_offset.to_le_bytes()); // data_offset

        // Metadata
        data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

        // Tensor index
        let idx_start = tensor_index_offset as usize;
        data[idx_start..idx_start + tensor_entry.len()].copy_from_slice(&tensor_entry);

        // Tensor data (16 floats)
        let data_start = data_offset as usize;
        for i in 0..16 {
            let val = i as f32;
            data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }

        data
    }

    #[test]
    fn test_apr_model_tensor_count() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");
        assert_eq!(model.tensor_count(), 1);
    }

    #[test]
    fn test_apr_model_tensor_names() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");
        let names = model.tensor_names();
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "test.weight");
    }

    #[test]
    fn test_apr_model_metadata() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");
        let meta = model.metadata();
        assert_eq!(meta.vocab_size, Some(100));
        assert_eq!(meta.hidden_size, Some(64));
    }

    #[test]
    fn test_apr_model_get_tensor() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let tensor = model.get_tensor("test.weight");
        assert!(tensor.is_some());
        let entry = tensor.expect("APR operation failed");
        assert_eq!(entry.shape, vec![4, 4]);
        assert_eq!(entry.dtype, "F32");

        // Non-existent tensor
        assert!(model.get_tensor("nonexistent").is_none());
    }

    #[test]
    fn test_apr_model_get_tensor_f32() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let floats = model.get_tensor_f32("test.weight").expect("should get f32");
        assert_eq!(floats.len(), 16);
        assert_eq!(floats[0], 0.0);
        assert_eq!(floats[1], 1.0);
        assert_eq!(floats[15], 15.0);
    }

    #[test]
    fn test_apr_model_get_tensor_f32_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let result = model.get_tensor_f32("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_model_get_tensor_bytes() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let bytes = model
            .get_tensor_bytes("test.weight")
            .expect("should get bytes");
        assert_eq!(bytes.len(), 64); // 16 floats * 4 bytes
    }

    #[test]
    fn test_apr_model_get_tensor_bytes_not_found() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        let result = model.get_tensor_bytes("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_model_estimated_parameters() {
        let data = create_test_apr_model();
        let model = AprV2Model::from_bytes(data).expect("should load");

        // 4 x 4 = 16 parameters
        assert_eq!(model.estimated_parameters(), 16);
    }

    #[test]
    fn test_apr_flags_lz4() {
        let flags = AprFlags::new(AprFlags::LZ4_COMPRESSED);
        assert!(flags.is_lz4());
        assert!(flags.is_compressed());
        assert!(!flags.is_zstd());
    }

    #[test]
    fn test_apr_flags_zstd() {
        let flags = AprFlags::new(AprFlags::ZSTD_COMPRESSED);
        assert!(flags.is_zstd());
        assert!(flags.is_compressed());
        assert!(!flags.is_lz4());
    }
