
    #[test]
    fn test_resolve_hf_model_bare_org_repo_with_apr() {
        let result = resolve_hf_model("org/repo/model.apr").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.apr"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_relative_path_with_dots() {
        // ../path should NOT be normalized to hf://
        let result = resolve_hf_model("../models/test.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "../models/test.gguf"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_http_url() {
        let result = resolve_hf_model("http://example.com/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "http://example.com/model.gguf"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_ftp_url() {
        let result = resolve_hf_model("ftp://example.com/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "ftp://example.com/model.gguf"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_empty_org_fails() {
        // "hf:///repo" → parts = ["", "repo"] → len >= 2 but first is empty
        // The function proceeds with empty org, which goes to API call → fails
        let result = resolve_hf_model("hf:///repo");
        // This triggers a network call with empty org, which will fail
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_hf_model_hf_with_single_part_fails() {
        let result = resolve_hf_model("hf://onlyorg");
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Invalid HuggingFace URI"));
            }
            Err(other) => panic!("Expected ValidationFailed, got: {}", other),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    }

    #[test]
    fn test_resolve_hf_model_bare_empty_parts() {
        // "/" alone → parts = ["", ""], both empty → does NOT normalize
        let result = resolve_hf_model("/").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "/"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_bare_single_slash() {
        // "a/" → parts = ["a", ""], parts[1].is_empty() → no normalization
        let result = resolve_hf_model("a/").unwrap();
        // parts[1] is empty, so bare org/repo normalization skipped
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "a/"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    // =========================================================================
    // ShardManifest serialization/deserialization tests
    // =========================================================================

    #[test]
    fn test_shard_manifest_serialize_deserialize() {
        let mut files = HashMap::new();
        files.insert(
            "model-00001-of-00002.safetensors".to_string(),
            FileChecksum {
                size: 5_000_000_000,
                blake3: "abc123def456".to_string(),
            },
        );
        files.insert(
            "model-00002-of-00002.safetensors".to_string(),
            FileChecksum {
                size: 3_000_000_000,
                blake3: "789xyz000111".to_string(),
            },
        );

        let manifest = ShardManifest {
            version: 1,
            repo: "Qwen/Qwen2.5-Coder-3B-Instruct".to_string(),
            files,
        };

        let json = serde_json::to_string_pretty(&manifest).expect("serialize");
        let deserialized: ShardManifest = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.version, 1);
        assert_eq!(deserialized.repo, "Qwen/Qwen2.5-Coder-3B-Instruct");
        assert_eq!(deserialized.files.len(), 2);

        let shard1 = deserialized
            .files
            .get("model-00001-of-00002.safetensors")
            .expect("shard1");
        assert_eq!(shard1.size, 5_000_000_000);
        assert_eq!(shard1.blake3, "abc123def456");

        let shard2 = deserialized
            .files
            .get("model-00002-of-00002.safetensors")
            .expect("shard2");
        assert_eq!(shard2.size, 3_000_000_000);
        assert_eq!(shard2.blake3, "789xyz000111");
    }

    #[test]
    fn test_shard_manifest_empty_files() {
        let manifest = ShardManifest {
            version: 1,
            repo: "org/repo".to_string(),
            files: HashMap::new(),
        };

        let json = serde_json::to_string(&manifest).expect("serialize");
        let deserialized: ShardManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.version, 1);
        assert!(deserialized.files.is_empty());
    }

    #[test]
    fn test_file_checksum_serialize_deserialize() {
        let checksum = FileChecksum {
            size: 1_234_567_890,
            blake3: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
        };

        let json = serde_json::to_string(&checksum).expect("serialize");
        let deserialized: FileChecksum = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.size, 1_234_567_890);
        assert_eq!(
            deserialized.blake3,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        );
    }

    #[test]
    fn test_shard_manifest_version_zero() {
        let manifest = ShardManifest {
            version: 0,
            repo: "test/repo".to_string(),
            files: HashMap::new(),
        };
        let json = serde_json::to_string(&manifest).expect("serialize");
        assert!(json.contains("\"version\":0"));
    }

    #[test]
    fn test_shard_manifest_large_version() {
        let manifest = ShardManifest {
            version: u32::MAX,
            repo: "test/repo".to_string(),
            files: HashMap::new(),
        };
        let json = serde_json::to_string(&manifest).expect("serialize");
        let deserialized: ShardManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.version, u32::MAX);
    }

    #[test]
    fn test_file_checksum_zero_size() {
        let checksum = FileChecksum {
            size: 0,
            blake3: "empty".to_string(),
        };
        let json = serde_json::to_string(&checksum).expect("serialize");
        let deserialized: FileChecksum = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.size, 0);
    }

    #[test]
    fn test_file_checksum_max_u64_size() {
        let checksum = FileChecksum {
            size: u64::MAX,
            blake3: "huge".to_string(),
        };
        let json = serde_json::to_string(&checksum).expect("serialize");
        let deserialized: FileChecksum = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.size, u64::MAX);
    }

    // =========================================================================
    // resolve_hf_uri: backward-compat wrapper edge cases
    // =========================================================================

    #[test]
    fn test_resolve_hf_uri_with_apr_extension() {
        let uri = "hf://org/repo/model.apr";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri);
    }

    #[test]
    fn test_resolve_hf_uri_with_pt_extension() {
        let uri = "hf://org/repo/model.pt";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri);
    }

    #[test]
    fn test_resolve_hf_uri_bare_org_repo_gguf() {
        // "org/repo/file.gguf" → normalizes to "hf://org/repo/file.gguf"
        let resolved = resolve_hf_uri("org/repo/file.gguf").unwrap();
        assert_eq!(resolved, "hf://org/repo/file.gguf");
    }

    #[test]
    fn test_resolve_hf_uri_dot_relative_path() {
        let uri = "./some/dir/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "dot-relative path should not be normalized");
    }

    #[test]
    fn test_resolve_hf_uri_dot_dot_relative_path() {
        let uri = "../parent/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(
            resolved, uri,
            "parent-relative path should not be normalized"
        );
    }

    #[test]
    fn test_resolve_hf_uri_just_a_word() {
        // Single word with no slashes: not normalized, returned as SingleFile
        let resolved = resolve_hf_uri("model").unwrap();
        assert_eq!(resolved, "model");
    }

    // =========================================================================
    // fetch_safetensors_companions: path edge cases (offline)
    // =========================================================================

    #[test]
    fn test_fetch_companions_empty_uri() {
        let temp_dir = std::env::temp_dir().join("apr_companion_empty_uri");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("hash123.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        // Empty URI → not hf:// → noop
        let result = fetch_safetensors_companions(&model_path, "");
        assert!(result.is_ok());
        assert!(!temp_dir.join("hash123.tokenizer.json").exists());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fetch_companions_model_stem_extraction() {
        let temp_dir = std::env::temp_dir().join("apr_companion_stem");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Model with complex hash stem
        let model_path = temp_dir.join("e910cab26ae116eb.converted.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        // Pre-create companion files with the full stem (without .safetensors)
        // The stem is "e910cab26ae116eb.converted"
        let _ = std::fs::write(
            temp_dir.join("e910cab26ae116eb.converted.tokenizer.json"),
            b"{}",
        );
        let _ = std::fs::write(
            temp_dir.join("e910cab26ae116eb.converted.config.json"),
            b"{}",
        );

        // Should succeed — files already exist
        let result = fetch_safetensors_companions(&model_path, "hf://org/repo/model.safetensors");
        assert!(result.is_ok());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fetch_companions_https_uri_noop() {
        let temp_dir = std::env::temp_dir().join("apr_companion_https");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("model.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        // https:// URI — extract_hf_repo returns None → noop
        let result =
            fetch_safetensors_companions(&model_path, "https://example.com/model.safetensors");
        assert!(result.is_ok());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // =========================================================================
    // resolve_hf_model: URI normalization edge cases
    // =========================================================================

    #[test]
    fn test_resolve_hf_model_double_slash_bare_path() {
        // "a//b" → parts = ["a", "", "b"], parts[1].is_empty() → no normalization
        let result = resolve_hf_model("a//b").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "a//b"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_with_query_string() {
        // URI with query params in extension — extension check uses Path
        // Path::extension sees "gguf?rev=main" which doesn't match .gguf
        // So it falls through to the HF API query path which fails with network error
        let result = resolve_hf_model("hf://org/repo/model.gguf?rev=main");
        // The result should be an error since the extension is not recognized
        // and the API query for "org/repo" will fail (doesn't exist)
        match result {
            Ok(ResolvedModel::SingleFile(_)) => {
                // Extension was detected somehow — acceptable
            }
            Err(CliError::NetworkError(_)) => {
                // Expected: API query failed since "org/repo" doesn't exist
            }
            Err(CliError::ValidationFailed(_)) => {
                // Also acceptable: no files found
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_resolve_hf_model_unicode_in_path() {
        // Unicode org/repo should be normalized
        // Will fail at API call since repo doesn't exist
        let result = resolve_hf_model("org-\u{00e9}/repo-\u{00fc}/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => {
                assert!(s.starts_with("hf://"), "Should be normalized: {}", s);
                assert!(s.ends_with("model.gguf"));
            }
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_spaces_in_path() {
        // Spaces in path — should not crash
        let result = resolve_hf_model("org name/repo name/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => {
                assert!(s.starts_with("hf://"));
                assert!(s.ends_with("model.gguf"));
            }
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    // =========================================================================
    // NEW: format_bytes additional edge cases
    // =========================================================================

    #[test]
    fn test_format_bytes_one_byte() {
        assert_eq!(format_bytes(1), "1 B");
    }

    #[test]
    fn test_format_bytes_exactly_two_kb() {
        assert_eq!(format_bytes(2 * 1024), "2.0 KB");
    }

    #[test]
    fn test_format_bytes_just_above_mb() {
        // 1 MB + 1 byte
        assert_eq!(format_bytes(1_048_577), "1.0 MB");
    }

    #[test]
    fn test_format_bytes_just_above_gb() {
        // 1 GB + 1 byte
        assert_eq!(format_bytes(1_073_741_825), "1.0 GB");
    }

    #[test]
    fn test_format_bytes_terabyte_range() {
        // 1 TB = 1024 GB — batuta-common has TB unit
        let tb = 1024_u64 * 1024 * 1024 * 1024;
        let result = format_bytes(tb);
        assert_eq!(result, "1.0 TB");
    }

    #[test]
    fn test_format_bytes_10_tb() {
        let ten_tb = 10 * 1024_u64 * 1024 * 1024 * 1024;
        let result = format_bytes(ten_tb);
        assert_eq!(result, "10.0 TB");
    }

    #[test]
    fn test_format_bytes_exact_256_mb() {
        assert_eq!(format_bytes(256 * 1024 * 1024), "256.0 MB");
    }

    #[test]
    fn test_format_bytes_1b_model_size() {
        // ~600 MB typical for 1B Q4_K_M
        assert_eq!(format_bytes(629_145_600), "600.0 MB");
    }

    #[test]
    fn test_format_bytes_13b_model_size() {
        // ~7.4 GB typical for 13B Q4_K_M
        assert_eq!(format_bytes(7_945_689_498), "7.4 GB");
    }

    #[test]
    fn test_format_bytes_half_kb() {
        assert_eq!(format_bytes(512), "512 B");
    }
