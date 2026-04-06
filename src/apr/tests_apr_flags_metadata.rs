
    #[test]
    fn test_apr_flags_has_vocab() {
        let flags = AprFlags::new(AprFlags::HAS_VOCAB);
        assert!(flags.has_vocab());
        assert!(!flags.is_quantized());
    }

    #[test]
    fn test_apr_metadata_is_transformer() {
        // is_transformer() requires hidden_size, num_layers, num_heads, vocab_size all Some
        let mut meta = AprMetadata::default();
        assert!(!meta.is_transformer()); // all None

        // Set all required fields
        meta.hidden_size = Some(1024);
        meta.num_layers = Some(12);
        meta.num_heads = Some(16);
        meta.vocab_size = Some(32000);
        assert!(meta.is_transformer());

        // Missing one field
        meta.hidden_size = None;
        assert!(!meta.is_transformer());
    }

    // Version test removed - ONE format, no versioning

    #[test]
    fn test_apr_model_encrypted_error() {
        let mut data = vec![0u8; HEADER_SIZE + 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&AprFlags::ENCRYPTED.to_le_bytes()); // flags = encrypted

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Encrypted"));
    }

    #[test]
    fn test_apr_model_truncated_metadata() {
        let mut data = vec![0u8; 100]; // Too small for metadata
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset = 64
        data[20..24].copy_from_slice(&1000u32.to_le_bytes()); // metadata_size = 1000 (larger than file)

        let result = AprV2Model::from_bytes(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn test_is_apr_file() {
        // is_apr_file reads the file and checks for APR\0 magic bytes
        // Non-existent files return false
        assert!(!is_apr_file("/nonexistent/model.apr"));
        assert!(!is_apr_file("/nonexistent/model.gguf"));

        // Create temp file with APR magic
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("test_apr_file.apr");
        {
            let mut f = std::fs::File::create(&path).expect("create temp file");
            f.write_all(&MAGIC).expect("write magic");
            f.write_all(&[0u8; 60]).expect("write padding");
        }
        assert!(is_apr_file(&path));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_detect_format_unknown() {
        assert_eq!(detect_format("/path/model.bin"), "unknown");
        assert_eq!(detect_format("/path/model.pt"), "unknown");
    }

    #[test]
    fn test_bpe_tokenizer_encode_decode() {
        // Create vocab with ASCII characters
        let id_to_token: Vec<String> = (0u8..128)
            .map(|i| String::from_utf8(vec![i]).unwrap_or_default())
            .collect();

        let token_to_id: HashMap<String, u32> = id_to_token
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        let tokenizer = BpeTokenizer {
            token_to_id,
            id_to_token,
            merge_rules: vec![],
            bos_id: Some(1),
            eos_id: Some(2),
            special_tokens: HashMap::new(),
        };

        // Encode simple ASCII
        let ids = tokenizer.encode("hi");
        assert!(!ids.is_empty());

        // Decode back
        let decoded = tokenizer.decode(&ids);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_decode_tokens() {
        let vocab: Vec<String> = vec!["hello".to_string(), " ".to_string(), "world".to_string()];

        let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_decode_tokens_with_unknown() {
        let vocab: Vec<String> = vec!["a".to_string(), "b".to_string()];

        // Token 99 is out of bounds - decode_tokens formats as [id]
        let result = AprV2Model::decode_tokens(&vocab, &[0, 99, 1]);
        assert!(result.contains('a'));
        assert!(result.contains('b'));
        assert!(result.contains("[99]")); // Unknown tokens formatted as [id]
    }

    // =========================================================================
    // ModelData Tests (Memory-Mapped Model Loading)
    // =========================================================================

    #[test]
    fn test_model_data_from_vec() {
        let data = vec![1u8, 2, 3, 4, 5];
        let model_data = ModelData::from_vec(data.clone());

        assert_eq!(model_data.as_slice(), &data);
        assert_eq!(model_data.len(), 5);
        assert!(!model_data.is_empty());
        assert!(!model_data.is_mmap());
    }

    #[test]
    fn test_model_data_from_vec_empty() {
        let model_data = ModelData::from_vec(vec![]);

        assert!(model_data.is_empty());
        assert_eq!(model_data.len(), 0);
        assert!(!model_data.is_mmap());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_model_data_open_mmap() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"test mmap data").expect("write data");

        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");

        assert_eq!(model_data.as_slice(), b"test mmap data");
        assert_eq!(model_data.len(), 14);
        assert!(!model_data.is_empty());
        assert!(model_data.is_mmap());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_model_data_open_mmap_nonexistent() {
        let result = ModelData::open_mmap("/nonexistent/path/model.apr");
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_model_data_open_mmap_empty_file() {
        use tempfile::NamedTempFile;

        let temp = NamedTempFile::new().expect("create temp file");
        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");

        assert!(model_data.is_empty());
        assert_eq!(model_data.len(), 0);
        assert!(model_data.is_mmap());
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_model_data_release_cpu_pages_mmap() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"test release pages").expect("write data");

        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");

        // Should not error
        model_data.release_cpu_pages().expect("release pages");
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_model_data_release_cpu_pages_heap() {
        let model_data = ModelData::from_vec(vec![1, 2, 3, 4, 5]);

        // Should be no-op for heap data
        model_data
            .release_cpu_pages()
            .expect("release pages (no-op)");
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_model_data_advise_sequential_mmap() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"sequential access test")
            .expect("write data");

        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");

        // Should not error
        model_data.advise_sequential().expect("advise sequential");
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_model_data_advise_sequential_heap() {
        let model_data = ModelData::from_vec(vec![1, 2, 3]);

        // Should be no-op for heap data
        model_data
            .advise_sequential()
            .expect("advise sequential (no-op)");
    }

    #[test]
    fn test_model_data_debug() {
        let model_data = ModelData::from_vec(vec![1, 2, 3]);
        let debug_str = format!("{:?}", model_data);
        assert!(debug_str.contains("Heap"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_model_data_mmap_debug() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");
        temp.write_all(b"debug test").expect("write data");

        let model_data = ModelData::open_mmap(temp.path()).expect("open mmap");
        let debug_str = format!("{:?}", model_data);
        assert!(debug_str.contains("Mmap"));
    }

    // =========================================================================
    // AprV2Model mmap Integration Tests
    // =========================================================================

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_apr_model_load_uses_mmap_for_uncompressed() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a valid APR v2 file (uncompressed)
        let mut temp = NamedTempFile::new().expect("create temp file");

        // Write a minimal valid APR v2 file
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0 (uncompressed)
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

        temp.write_all(&data).expect("write data");

        let model = AprV2Model::load(temp.path()).expect("load model");

        // Should use mmap for uncompressed files
        assert!(model.is_mmap(), "Uncompressed model should use mmap");
    }

    #[test]
    fn test_apr_model_from_bytes_uses_heap() {
        // Create a minimal valid APR v2 data
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2; // version major
        data[5] = 0; // version minor
        data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0
        data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
        data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
        data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
        data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

        let model = AprV2Model::from_bytes(data).expect("load model");

        // from_bytes always uses heap
        assert!(!model.is_mmap(), "from_bytes should use heap, not mmap");
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    #[test]
    fn test_apr_model_release_cpu_pages() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a valid APR v2 file
        let mut temp = NamedTempFile::new().expect("create temp file");

        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 2;
        data[5] = 0;
        data[6..8].copy_from_slice(&0u16.to_le_bytes());
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..20].copy_from_slice(&64u64.to_le_bytes());
        data[20..24].copy_from_slice(&0u32.to_le_bytes());
        data[24..32].copy_from_slice(&64u64.to_le_bytes());
        data[32..40].copy_from_slice(&64u64.to_le_bytes());

        temp.write_all(&data).expect("write data");

        let model = AprV2Model::load(temp.path()).expect("load model");

        // Should not error
        model.release_cpu_pages().expect("release pages");
    }

    #[test]
    fn test_apr_model_load_nonexistent_file() {
        let result = AprV2Model::load("/nonexistent/path/model.apr");
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_apr_model_load_invalid_magic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        // Write invalid magic
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(b"GGUF"); // Wrong magic
        data[4] = 2;
        data[5] = 0;

        temp.write_all(&data).expect("write data");

        let result = AprV2Model::load(temp.path());
        assert!(result.is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_apr_model_load_truncated_header() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp = NamedTempFile::new().expect("create temp file");

        // Write truncated file (less than HEADER_SIZE)
        temp.write_all(&[0u8; 10]).expect("write data");

        let result = AprV2Model::load(temp.path());
        assert!(result.is_err());
    }

    // =========================================================================
    // F16 Conversion Tests
    // =========================================================================

    #[test]
    fn test_f16_to_f32_zero() {
        let zero: u16 = 0x0000;
        let result = crate::apr::f16_to_f32(zero);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_f16_to_f32_one() {
        let one: u16 = 0x3C00; // 1.0 in f16
        let result = crate::apr::f16_to_f32(one);
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_negative_one() {
        let neg_one: u16 = 0xBC00; // -1.0 in f16
        let result = crate::apr::f16_to_f32(neg_one);
        assert!((result + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_inf() {
        let pos_inf: u16 = 0x7C00;
        let result = crate::apr::f16_to_f32(pos_inf);
        assert!(result.is_infinite() && result > 0.0);

        let neg_inf: u16 = 0xFC00;
        let result = crate::apr::f16_to_f32(neg_inf);
        assert!(result.is_infinite() && result < 0.0);
    }

    #[test]
    fn test_f16_to_f32_nan() {
        let nan: u16 = 0x7C01; // NaN (exp=31, mantissa!=0)
        let result = crate::apr::f16_to_f32(nan);
        assert!(result.is_nan());
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        let subnormal: u16 = 0x0001; // Smallest positive subnormal
        let result = crate::apr::f16_to_f32(subnormal);
        assert!(result > 0.0 && result < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_negative_zero() {
        let neg_zero: u16 = 0x8000;
        let result = crate::apr::f16_to_f32(neg_zero);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_negative());
    }
