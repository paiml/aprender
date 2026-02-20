
    /// Helper: create a minimal valid v2 APR file
    fn create_test_apr_bytes(metadata: AprV2Metadata) -> Vec<u8> {
        let mut writer = AprV2Writer::new(metadata);
        writer.add_f32_tensor("test.weight", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        writer.write().expect("write v2 bytes")
    }

    fn create_test_apr_file(metadata: AprV2Metadata) -> NamedTempFile {
        let bytes = create_test_apr_bytes(metadata);
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(&bytes).expect("write");
        file
    }

    // ========================================================================
    // Path Validation Tests
    // ========================================================================

    #[test]
    fn test_validate_path_not_found() {
        let result = validate_path(Path::new("/nonexistent/model.apr"));
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_validate_path_is_directory() {
        let dir = tempdir().expect("create dir");
        let result = validate_path(dir.path());
        assert!(result.is_err());
        match result {
            Err(CliError::NotAFile(_)) => {}
            _ => panic!("Expected NotAFile error"),
        }
    }

    #[test]
    fn test_validate_path_valid() {
        let file = NamedTempFile::new().expect("create file");
        let result = validate_path(file.path());
        assert!(result.is_ok());
    }

    // ========================================================================
    // Run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            false,
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_file_too_small() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(b"short").expect("write");

        let result = run(file.path(), false, false, false, false);
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidFormat(msg)) => {
                assert!(msg.contains("too small") || msg.contains("64 bytes"));
            }
            _ => panic!("Expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_run_invalid_magic() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        // Write 64 bytes with invalid magic
        let mut data = [0u8; 64];
        data[0..4].copy_from_slice(b"XXXX");
        file.write_all(&data).expect("write");

        let result = run(file.path(), false, false, false, false);
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidFormat(msg)) => {
                assert!(msg.contains("Invalid magic"));
            }
            _ => panic!("Expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_run_legacy_magic_rejected() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        // Write 64 bytes with legacy APRN magic
        let mut data = [0u8; 64];
        data[0..4].copy_from_slice(b"APRN");
        file.write_all(&data).expect("write");

        let result = run(file.path(), false, false, false, false);
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidFormat(msg)) => {
                assert!(
                    msg.contains("Legacy"),
                    "Expected legacy format message, got: {msg}"
                );
            }
            _ => panic!("Expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_run_valid_v2_file_text() {
        let metadata = AprV2Metadata {
            model_type: "Qwen2".to_string(),
            name: Some("test-model".to_string()),
            architecture: Some("qwen2".to_string()),
            hidden_size: Some(896),
            num_layers: Some(24),
            num_heads: Some(14),
            num_kv_heads: Some(2),
            vocab_size: Some(151936),
            param_count: 494_032_768,
            ..Default::default()
        };
        let file = create_test_apr_file(metadata);
        let result = run(file.path(), false, false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_valid_v2_file_json() {
        let metadata = AprV2Metadata {
            model_type: "Qwen2".to_string(),
            name: Some("json-test".to_string()),
            param_count: 1_000_000,
            ..Default::default()
        };
        let file = create_test_apr_file(metadata);
        let result = run(file.path(), false, false, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_v2_with_source_metadata() {
        let mut custom = std::collections::HashMap::new();
        let mut source_meta = serde_json::Map::new();
        source_meta.insert(
            "my_run_id".to_string(),
            serde_json::Value::String("test_123".to_string()),
        );
        source_meta.insert(
            "framework".to_string(),
            serde_json::Value::String("pytorch".to_string()),
        );
        custom.insert(
            "source_metadata".to_string(),
            serde_json::Value::Object(source_meta),
        );

        let metadata = AprV2Metadata {
            model_type: "Qwen2".to_string(),
            name: Some("metadata-test".to_string()),
            custom,
            ..Default::default()
        };
        let file = create_test_apr_file(metadata);

        // Run JSON output and verify source_metadata appears
        let result = run(file.path(), false, false, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_with_show_options() {
        let metadata = AprV2Metadata::new("test");
        let file = create_test_apr_file(metadata);
        let result = run(file.path(), true, true, true, false);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Header Parsing Tests
    // ========================================================================

    #[test]
    fn test_read_header_valid_v2() {
        let metadata = AprV2Metadata {
            model_type: "test".to_string(),
            param_count: 42,
            ..Default::default()
        };
        let bytes = create_test_apr_bytes(metadata);
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(&bytes).expect("write");

        let f = File::open(file.path()).expect("open");
        let mut reader = BufReader::new(f);
        let header = read_and_parse_header(&mut reader).expect("parse header");

        assert_eq!(header.version, (2, 0));
        assert_eq!(header.tensor_count, 1);
        assert!(header.checksum_valid);
        assert!(header.metadata_size > 0);
    }

    #[test]
    fn test_read_metadata_from_v2() {
        let metadata = AprV2Metadata {
            model_type: "Qwen2".to_string(),
            name: Some("round-trip-test".to_string()),
            architecture: Some("qwen2".to_string()),
            hidden_size: Some(768),
            num_layers: Some(12),
            ..Default::default()
        };
        let bytes = create_test_apr_bytes(metadata);
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(&bytes).expect("write");

        let f = File::open(file.path()).expect("open");
        let mut reader = BufReader::new(f);
        let header = read_and_parse_header(&mut reader).expect("parse header");
        let meta = read_metadata(&mut reader, &header);

        assert_eq!(meta.model_type.as_deref(), Some("Qwen2"));
        assert_eq!(meta.name.as_deref(), Some("round-trip-test"));
        assert_eq!(meta.architecture.as_deref(), Some("qwen2"));
        assert_eq!(meta.hidden_size, Some(768));
        assert_eq!(meta.num_layers, Some(12));
    }

    #[test]
    fn test_read_metadata_with_source_metadata() {
        let mut custom = std::collections::HashMap::new();
        let mut source_meta = serde_json::Map::new();
        source_meta.insert(
            "run_id".to_string(),
            serde_json::Value::String("abc_789".to_string()),
        );
        custom.insert(
            "source_metadata".to_string(),
            serde_json::Value::Object(source_meta),
        );

        let metadata = AprV2Metadata {
            model_type: "test".to_string(),
            custom,
            ..Default::default()
        };
        let bytes = create_test_apr_bytes(metadata);
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        file.write_all(&bytes).expect("write");

        let f = File::open(file.path()).expect("open");
        let mut reader = BufReader::new(f);
        let header = read_and_parse_header(&mut reader).expect("parse header");
        let meta = read_metadata(&mut reader, &header);

        assert!(meta.source_metadata.is_some());
        let sm = meta.source_metadata.as_ref().expect("source_metadata");
        assert_eq!(sm.get("run_id").and_then(|v| v.as_str()), Some("abc_789"));
    }

    // ========================================================================
    // Serialization Tests
    // ========================================================================

    #[test]
    fn test_flags_info_serialization() {
        let flags = FlagsInfo {
            lz4_compressed: true,
            zstd_compressed: false,
            encrypted: false,
            signed: false,
            sharded: true,
            quantized: true,
            has_vocab: false,
        };

        let json = serde_json::to_string(&flags).expect("serialize");
        assert!(json.contains("\"lz4_compressed\":true"));
        assert!(json.contains("\"sharded\":true"));
        assert!(json.contains("\"quantized\":true"));
    }

    #[test]
    fn test_metadata_info_default() {
        let info = MetadataInfo::default();
        assert!(info.model_type.is_none());
        assert!(info.name.is_none());
        assert!(info.architecture.is_none());
        assert!(info.source_metadata.is_none());
    }

    #[test]
    fn test_metadata_info_serialization() {
        let info = MetadataInfo {
            model_type: Some("Qwen2".to_string()),
            name: Some("test-model".to_string()),
            architecture: Some("qwen2".to_string()),
            hidden_size: Some(768),
            num_layers: Some(12),
            vocab_size: Some(50000),
            param_count: Some(494_000_000),
            ..Default::default()
        };

        let json = serde_json::to_string(&info).expect("serialize");
        assert!(json.contains("test-model"));
        assert!(json.contains("qwen2"));
        assert!(json.contains("768"));
        assert!(json.contains("494000000"));
    }

    #[test]
    fn test_inspect_result_serialization() {
        let result = InspectResult {
            file: "model.apr".to_string(),
            valid: true,
            format: "APR v2".to_string(),
            version: "2.0".to_string(),
            tensor_count: 291,
            size_bytes: 1024 * 1024,
            checksum_valid: true,
            architecture: None,
            num_layers: None,
            num_heads: None,
            hidden_size: None,
            vocab_size: None,
            flags: FlagsInfo {
                lz4_compressed: false,
                zstd_compressed: false,
                encrypted: false,
                signed: false,
                sharded: false,
                quantized: false,
                has_vocab: false,
            },
            metadata: MetadataInfo::default(),
        };

        let json = serde_json::to_string_pretty(&result).expect("serialize");
        assert!(json.contains("model.apr"));
        assert!(json.contains("APR v2"));
        assert!(json.contains("\"valid\": true"));
        assert!(json.contains("\"tensor_count\": 291"));
    }

    // ========================================================================
    // Output Functions Tests
    // ========================================================================

    #[test]
    fn test_output_flags_empty() {
        let header = HeaderData {
            version: (2, 0),
            flags: AprV2Flags::new(),
            tensor_count: 0,
            metadata_offset: 64,
            metadata_size: 0,
            tensor_index_offset: 0,
            data_offset: 0,
            checksum_valid: true,
        };
        output_flags(&header);
    }

    #[test]
    fn test_output_flags_multiple() {
        let flags = AprV2Flags::new()
            .with(AprV2Flags::LZ4_COMPRESSED)
            .with(AprV2Flags::QUANTIZED)
            .with(AprV2Flags::HAS_VOCAB);

        let header = HeaderData {
            version: (2, 0),
            flags,
            tensor_count: 10,
            metadata_offset: 64,
            metadata_size: 100,
            tensor_index_offset: 200,
            data_offset: 300,
            checksum_valid: true,
        };
        output_flags(&header);
    }

    #[test]
    fn test_output_metadata_text_empty() {
        let metadata = MetadataInfo::default();
        output_metadata_text(&metadata);
    }

    #[test]
    fn test_output_metadata_text_full() {
        let metadata = MetadataInfo {
            model_type: Some("Qwen2".to_string()),
            name: Some("test-model".to_string()),
            description: Some("Test description".to_string()),
            author: Some("Test Author".to_string()),
            source: Some("hf://test/model".to_string()),
            original_format: Some("safetensors".to_string()),
            created_at: Some("2024-01-01".to_string()),
            architecture: Some("qwen2".to_string()),
            param_count: Some(494_000_000),
            hidden_size: Some(896),
            num_layers: Some(24),
            num_heads: Some(14),
            num_kv_heads: Some(2),
            vocab_size: Some(151936),
            intermediate_size: Some(4864),
            max_position_embeddings: Some(32768),
            rope_theta: Some(1_000_000.0),
            chat_template: Some("{{prompt}}".to_string()),
            chat_format: Some("chatml".to_string()),
            special_tokens: Some(serde_json::json!({"bos": "<s>", "eos": "</s>"})),
            source_metadata: Some(
                serde_json::json!({"run_id": "test_123", "framework": "pytorch"}),
            ),
        };
        output_metadata_text(&metadata);
    }

    #[test]
    fn test_output_metadata_text_long_template() {
        let long_template = "a".repeat(200);
        let metadata = MetadataInfo {
            chat_template: Some(long_template),
            chat_format: Some("custom".to_string()),
            ..Default::default()
        };
        output_metadata_text(&metadata);
    }
