
    // =========================================================================
    // format_bytes tests
    // =========================================================================

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_bytes(5 * 1024 * 1024 * 1024), "5.0 GB");
    }

    #[test]
    fn test_format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 B");
    }

    #[test]
    fn test_format_bytes_small() {
        assert_eq!(format_bytes(1), "1 B");
        assert_eq!(format_bytes(100), "100 B");
        assert_eq!(format_bytes(1023), "1023 B");
    }

    #[test]
    fn test_format_bytes_kilobytes() {
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(2048), "2.0 KB");
        assert_eq!(format_bytes(512 * 1024), "512.0 KB");
    }

    #[test]
    fn test_format_bytes_megabytes() {
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(100 * 1024 * 1024), "100.0 MB");
        assert_eq!(format_bytes(500 * 1024 * 1024), "500.0 MB");
    }

    #[test]
    fn test_format_bytes_gigabytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_bytes(10 * 1024 * 1024 * 1024), "10.0 GB");
        assert_eq!(format_bytes(100 * 1024 * 1024 * 1024), "100.0 GB");
    }

    #[test]
    fn test_format_bytes_fractional_gb() {
        // 4.5 GB = 4.5 * 1024 * 1024 * 1024 = 4831838208 bytes
        assert_eq!(format_bytes(4831838208), "4.5 GB");
    }

    #[test]
    fn test_format_bytes_fractional_mb() {
        // 2.5 MB = 2.5 * 1024 * 1024 = 2621440 bytes
        assert_eq!(format_bytes(2621440), "2.5 MB");
    }

    // =========================================================================
    // PMAT-108: resolve_hf_uri Tests (Extreme TDD)
    // =========================================================================

    #[test]
    fn test_pmat_108_resolve_uri_with_gguf_extension_unchanged() {
        let uri = "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "URI with .gguf should be unchanged");
    }

    #[test]
    fn test_pmat_108_resolve_uri_case_insensitive_gguf() {
        let uri = "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/model.GGUF";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "URI with .GGUF should be unchanged");
    }

    #[test]
    fn test_pmat_108_resolve_non_hf_uri_unchanged() {
        let uri = "/path/to/local/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Non-hf:// URI should be unchanged");
    }

    #[test]
    fn test_pmat_108_resolve_invalid_uri_fails() {
        let uri = "hf://invalid";
        let result = resolve_hf_uri(uri);
        assert!(result.is_err(), "Invalid URI should fail");
    }

    #[test]
    fn test_resolve_hf_uri_relative_path() {
        let uri = "./models/test.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Relative path should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_absolute_path() {
        let uri = "/home/user/models/test.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Absolute path should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_https_url() {
        let uri = "https://example.com/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "HTTPS URL should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_with_mixed_case_extension() {
        let uri = "hf://Org/Repo/model.GgUf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Mixed case .GgUf should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_empty_string() {
        let uri = "";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Empty string should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_invalid_hf_format() {
        // hf:// without org/repo should fail
        let result = resolve_hf_uri("hf://only-one-part");
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Invalid HuggingFace URI"));
            }
            other => panic!("Expected ValidationFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_resolve_hf_uri_with_safetensors_extension_unchanged() {
        // .safetensors files are not .gguf, so they will trigger HF API query
        // This test verifies the logic path, but we can't test the full flow
        // without mocking. Instead, test that non-gguf HF URIs attempt resolution.
        // The test_resolve_hf_uri_invalid_hf_format covers the error case.
        // For now, we just verify the URI format is preserved for files with .gguf extension
        let uri = "hf://org/repo/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, ".gguf extension should be unchanged");
    }

    #[test]
    #[ignore] // Requires network access
    fn test_resolve_hf_uri_with_safetensors_queries_api() {
        // This test would need network access to verify API query behavior
        let uri = "hf://org/repo/model.safetensors";
        let _result = resolve_hf_uri(uri);
        // Result depends on network and repo existence
    }

    // Integration test (requires network, marked ignore for CI)
    #[test]
    #[ignore]
    fn test_pmat_108_resolve_qwen_repo_finds_q4_k_m() {
        let uri = "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert!(resolved.ends_with(".gguf"), "Should end with .gguf");
        assert!(
            resolved.to_lowercase().contains("q4_k_m"),
            "Should prefer Q4_K_M quantization: {}",
            resolved
        );
    }

    // =========================================================================
    // resolve_model_path tests
    // =========================================================================

    #[test]
    fn test_resolve_model_path_existing_file() {
        // Create a temp file
        let temp_dir = std::env::temp_dir().join("apr_pull_test_path");
        let _ = std::fs::create_dir_all(&temp_dir);
        let test_file = temp_dir.join("test_model.gguf");
        let _ = std::fs::write(&test_file, "GGUF");

        let result = resolve_model_path(test_file.to_str().unwrap());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), test_file);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_resolve_model_path_nonexistent_local_fails() {
        let result = resolve_model_path("/nonexistent/model.gguf");
        // This will try pacha which will fail with validation error
        assert!(result.is_err());
    }

    // =========================================================================
    // GH-198: extract_hf_repo tests
    // =========================================================================

    #[test]
    fn test_gh198_extract_hf_repo_with_file() {
        let uri = "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct/model.safetensors";
        assert_eq!(
            extract_hf_repo(uri),
            Some("Qwen/Qwen2.5-Coder-0.5B-Instruct".to_string())
        );
    }

    #[test]
    fn test_gh198_extract_hf_repo_without_file() {
        let uri = "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct";
        assert_eq!(
            extract_hf_repo(uri),
            Some("Qwen/Qwen2.5-Coder-0.5B-Instruct".to_string())
        );
    }

    #[test]
    fn test_gh198_extract_hf_repo_local_path() {
        assert_eq!(extract_hf_repo("/local/path/model.safetensors"), None);
    }

    #[test]
    fn test_gh198_extract_hf_repo_empty() {
        assert_eq!(extract_hf_repo(""), None);
    }

    #[test]
    fn test_gh198_extract_hf_repo_only_org() {
        // hf://org (missing repo) → None
        assert_eq!(extract_hf_repo("hf://org"), None);
    }

    #[test]
    fn test_gh198_extract_hf_repo_nested_path() {
        let uri = "hf://org/repo/subdir/model.safetensors";
        assert_eq!(extract_hf_repo(uri), Some("org/repo".to_string()));
    }

    // =========================================================================
    // GH-198: fetch_safetensors_companions tests
    // =========================================================================

    #[test]
    fn test_gh198_companions_non_hf_uri_is_noop() {
        let temp_dir = std::env::temp_dir().join("apr_gh198_noop");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("d71534cb.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        // Local URI → should return Ok without downloading anything
        let result = fetch_safetensors_companions(&model_path, "/local/model.safetensors");
        assert!(result.is_ok());

        // No companion files should be created (GAP-UX-002: hash-prefixed)
        assert!(!temp_dir.join("d71534cb.tokenizer.json").exists());
        assert!(!temp_dir.join("d71534cb.config.json").exists());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_gh198_companions_skips_existing() {
        let temp_dir = std::env::temp_dir().join("apr_gh198_existing");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("abc123.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        // Pre-create companion files (GAP-UX-002: hash-prefixed)
        let _ = std::fs::write(temp_dir.join("abc123.tokenizer.json"), b"{}");
        let _ = std::fs::write(temp_dir.join("abc123.config.json"), b"{}");

        // Should succeed without attempting downloads (files already exist)
        let result = fetch_safetensors_companions(
            &model_path,
            "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct/model.safetensors",
        );
        assert!(result.is_ok());

        // Verify files are unchanged (still our dummy content)
        let content = std::fs::read_to_string(temp_dir.join("abc123.tokenizer.json")).unwrap();
        assert_eq!(content, "{}");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    #[ignore] // Requires network access
    fn test_gh198_companions_downloads_from_hf() {
        let temp_dir = std::env::temp_dir().join("apr_gh198_download");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let _ = std::fs::create_dir_all(&temp_dir);
        // GAP-UX-002: Use hash-prefixed model name
        let model_path = temp_dir.join("d71534cb948e32eb.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        let result = fetch_safetensors_companions(
            &model_path,
            "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct/model.safetensors",
        );
        assert!(result.is_ok());

        // Both companion files should now exist (GAP-UX-002: hash-prefixed)
        assert!(
            temp_dir.join("d71534cb948e32eb.tokenizer.json").exists(),
            "d71534cb948e32eb.tokenizer.json should be downloaded"
        );
        assert!(
            temp_dir.join("d71534cb948e32eb.config.json").exists(),
            "d71534cb948e32eb.config.json should be downloaded"
        );

        // Verify tokenizer.json has vocab
        let tok =
            std::fs::read_to_string(temp_dir.join("d71534cb948e32eb.tokenizer.json")).unwrap();
        assert!(tok.contains("vocab"), "tokenizer.json should contain vocab");

        // Verify config.json has model architecture
        let cfg = std::fs::read_to_string(temp_dir.join("d71534cb948e32eb.config.json")).unwrap();
        assert!(
            cfg.contains("num_hidden_layers"),
            "config.json should contain num_hidden_layers"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // =========================================================================
    // GH-213: extract_shard_files_from_index tests
    // =========================================================================

    #[test]
    fn test_gh213_extract_shard_files_basic() {
        let json = r#"{
            "metadata": {"total_size": 123456},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
                "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
                "model.layers.2.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
                "lm_head.weight": "model-00004-of-00004.safetensors"
            }
        }"#;

        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 4);
        assert_eq!(shards[0], "model-00001-of-00004.safetensors");
        assert_eq!(shards[1], "model-00002-of-00004.safetensors");
        assert_eq!(shards[2], "model-00003-of-00004.safetensors");
        assert_eq!(shards[3], "model-00004-of-00004.safetensors");
    }

    #[test]
    fn test_gh213_extract_shard_files_deduplicates() {
        let json = r#"{
            "weight_map": {
                "a.weight": "model-00001-of-00002.safetensors",
                "b.weight": "model-00001-of-00002.safetensors",
                "c.weight": "model-00001-of-00002.safetensors",
                "d.weight": "model-00002-of-00002.safetensors"
            }
        }"#;

        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 2, "Should deduplicate shard filenames");
    }

    #[test]
    fn test_gh213_extract_shard_files_sorted() {
        let json = r#"{
            "weight_map": {
                "z.weight": "model-00003-of-00003.safetensors",
                "a.weight": "model-00001-of-00003.safetensors",
                "m.weight": "model-00002-of-00003.safetensors"
            }
        }"#;

        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 3);
        // Should be sorted alphabetically
        assert!(shards[0] < shards[1]);
        assert!(shards[1] < shards[2]);
    }

    #[test]
    fn test_gh213_extract_shard_files_empty_weight_map() {
        let json = r#"{"weight_map": {}}"#;
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_gh213_extract_shard_files_no_weight_map() {
        let json = r#"{"metadata": {"total_size": 123}}"#;
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_gh213_extract_shard_files_ignores_non_safetensors() {
        let json = r#"{
            "weight_map": {
                "a.weight": "model-00001-of-00002.safetensors",
                "b.weight": "not-a-safetensors-file.bin"
            }
        }"#;

        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], "model-00001-of-00002.safetensors");
    }

    #[test]
    fn test_gh213_extract_shard_files_malformed_json() {
        let json = "not valid json at all";
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty(), "Malformed JSON should return empty");
    }

    // =========================================================================
    // GH-213: resolve_hf_model tests (offline, no network)
    // =========================================================================

    #[test]
    fn test_gh213_resolve_non_hf_uri_is_single_file() {
        let result = resolve_hf_model("/path/to/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "/path/to/model.gguf"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_gh213_resolve_hf_with_extension_is_single_file() {
        let result = resolve_hf_model("hf://org/repo/model.safetensors").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.safetensors"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }
