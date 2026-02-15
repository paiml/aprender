
    #[test]
    fn test_output_architecture() {
        let metadata = MetadataInfo {
            architecture: Some("qwen2".to_string()),
            param_count: Some(7_000_000_000),
            hidden_size: Some(4096),
            num_layers: Some(32),
            ..Default::default()
        };
        output_architecture(&metadata);
    }

    #[test]
    fn test_output_architecture_empty() {
        let metadata = MetadataInfo::default();
        // Should not print anything
        output_architecture(&metadata);
    }

    #[test]
    fn test_output_json_v2() {
        let header = HeaderData {
            version: (2, 0),
            flags: AprV2Flags::new().with(AprV2Flags::QUANTIZED),
            tensor_count: 291,
            metadata_offset: 64,
            metadata_size: 1024,
            tensor_index_offset: 2048,
            data_offset: 4096,
            checksum_valid: true,
        };
        let metadata = MetadataInfo {
            model_type: Some("Qwen2".to_string()),
            ..Default::default()
        };
        output_json(Path::new("test.apr"), 1024, &header, metadata);
    }

    #[test]
    fn test_output_text_v2() {
        let header = HeaderData {
            version: (2, 0),
            flags: AprV2Flags::new(),
            tensor_count: 1,
            metadata_offset: 64,
            metadata_size: 100,
            tensor_index_offset: 200,
            data_offset: 300,
            checksum_valid: true,
        };
        let metadata = MetadataInfo::default();
        output_text(
            Path::new("test.apr"),
            512,
            &header,
            &metadata,
            false,
            false,
            false,
        );
    }

    #[test]
    fn test_output_text_with_options() {
        let header = HeaderData {
            version: (2, 0),
            flags: AprV2Flags::new(),
            tensor_count: 5,
            metadata_offset: 64,
            metadata_size: 200,
            tensor_index_offset: 300,
            data_offset: 500,
            checksum_valid: true,
        };
        let metadata = MetadataInfo::default();
        output_text(
            Path::new("test.apr"),
            1024,
            &header,
            &metadata,
            true,
            true,
            true,
        );
    }

    // ========================================================================
    // Format Helper Tests
    // ========================================================================

    #[test]
    fn test_format_param_count() {
        assert_eq!(format_param_count(500), "500");
        assert_eq!(format_param_count(1_500), "1.5K (1500)");
        assert_eq!(format_param_count(494_000_000), "494.0M (494000000)");
        assert_eq!(format_param_count(7_000_000_000), "7.0B (7000000000)");
    }

    #[test]
    fn test_flags_from_header() {
        let header = HeaderData {
            version: (2, 0),
            flags: AprV2Flags::new()
                .with(AprV2Flags::LZ4_COMPRESSED)
                .with(AprV2Flags::QUANTIZED),
            tensor_count: 0,
            metadata_offset: 64,
            metadata_size: 0,
            tensor_index_offset: 0,
            data_offset: 0,
            checksum_valid: true,
        };
        let flags = flags_from_header(&header);
        assert!(flags.lz4_compressed);
        assert!(flags.quantized);
        assert!(!flags.encrypted);
        assert!(!flags.sharded);
    }

    // ================================================================
    // Audit #3 fix: Real GGUF/SafeTensors dispatch tests
    // These exercise run_rosetta_inspect() with valid data.
    // ================================================================

    /// Build a minimal valid GGUF file for inspect tests.
    fn build_inspect_gguf() -> NamedTempFile {
        use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
        use std::io::BufWriter;

        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let mut writer = BufWriter::new(&file);

        let tensors = vec![
            GgufTensor {
                name: "token_embd.weight".to_string(),
                shape: vec![4, 8],
                dtype: GgmlType::F32,
                data: vec![0u8; 4 * 8 * 4],
            },
            GgufTensor {
                name: "blk.0.attn_q.weight".to_string(),
                shape: vec![8, 8],
                dtype: GgmlType::F32,
                data: vec![0u8; 8 * 8 * 4],
            },
        ];

        let metadata = vec![
            (
                "general.architecture".to_string(),
                GgufValue::String("llama".to_string()),
            ),
            (
                "general.name".to_string(),
                GgufValue::String("test-model".to_string()),
            ),
            ("llama.block_count".to_string(), GgufValue::Uint32(1)),
            ("llama.embedding_length".to_string(), GgufValue::Uint32(8)),
        ];

        export_tensors_to_gguf(&mut writer, &tensors, &metadata).expect("write GGUF");
        drop(writer);
        file
    }

    /// Build a minimal valid SafeTensors file for inspect tests.
    fn build_inspect_safetensors() -> NamedTempFile {
        let tensors: Vec<(&str, Vec<usize>, usize)> = vec![
            ("model.embed_tokens.weight", vec![8, 4], 32),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4], 16),
            ("lm_head.weight", vec![8, 4], 32),
        ];

        let mut data_bytes = Vec::new();
        let mut header_map = serde_json::Map::new();
        let mut offset = 0usize;

        for (name, shape, n_elements) in &tensors {
            let byte_len = n_elements * 4; // f32 = 4 bytes
            let end = offset + byte_len;

            let mut entry = serde_json::Map::new();
            entry.insert("dtype".to_string(), serde_json::json!("F32"));
            entry.insert("shape".to_string(), serde_json::json!(shape));
            entry.insert("data_offsets".to_string(), serde_json::json!([offset, end]));
            header_map.insert(name.to_string(), serde_json::Value::Object(entry));

            data_bytes.extend(vec![0u8; byte_len]);
            offset = end;
        }

        let header_json = serde_json::to_string(&header_map).expect("serialize header");
        let header_len = header_json.len() as u64;

        let mut file_data = Vec::new();
        file_data.extend_from_slice(&header_len.to_le_bytes());
        file_data.extend_from_slice(header_json.as_bytes());
        file_data.extend_from_slice(&data_bytes);

        let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        file.write_all(&file_data).expect("write safetensors");
        file
    }

    #[test]
    fn test_run_valid_gguf_inspect() {
        let file = build_inspect_gguf();
        let result = run(file.path(), false, false, false, false);
        assert!(result.is_ok(), "inspect on valid GGUF failed: {result:?}");
    }

    #[test]
    fn test_run_valid_gguf_inspect_json() {
        let file = build_inspect_gguf();
        let result = run(file.path(), false, false, false, true);
        assert!(
            result.is_ok(),
            "inspect JSON on valid GGUF failed: {result:?}"
        );
    }

    #[test]
    fn test_run_valid_safetensors_inspect() {
        let file = build_inspect_safetensors();
        let result = run(file.path(), false, false, false, false);
        assert!(
            result.is_ok(),
            "inspect on valid SafeTensors failed: {result:?}"
        );
    }

    #[test]
    fn test_run_valid_safetensors_inspect_json() {
        let file = build_inspect_safetensors();
        let result = run(file.path(), false, false, false, true);
        assert!(
            result.is_ok(),
            "inspect JSON on valid SafeTensors failed: {result:?}"
        );
    }

    #[test]
    fn test_rosetta_inspect_dispatch_gguf() {
        // Directly test run_rosetta_inspect() is reachable
        let file = build_inspect_gguf();
        let result = run_rosetta_inspect(file.path(), false);
        assert!(
            result.is_ok(),
            "run_rosetta_inspect GGUF failed: {result:?}"
        );
    }

    #[test]
    fn test_rosetta_inspect_dispatch_safetensors() {
        let file = build_inspect_safetensors();
        let result = run_rosetta_inspect(file.path(), true);
        assert!(
            result.is_ok(),
            "run_rosetta_inspect SafeTensors JSON failed: {result:?}"
        );
    }
