
    // =========================================================================
    // NEW: extract_hf_repo additional edge cases
    // =========================================================================

    #[test]
    fn test_extract_hf_repo_trailing_slash_after_repo() {
        // "hf://org/repo/" → parts = ["org", "repo", ""], len >= 2 → Some("org/repo")
        assert_eq!(
            extract_hf_repo("hf://org/repo/"),
            Some("org/repo".to_string())
        );
    }

    #[test]
    fn test_extract_hf_repo_with_multiple_trailing_slashes() {
        let uri = "hf://org/repo///";
        assert_eq!(extract_hf_repo(uri), Some("org/repo".to_string()));
    }

    #[test]
    fn test_extract_hf_repo_just_hf_no_colon_slash() {
        // "hf" without "://" → strip_prefix fails → None
        assert_eq!(extract_hf_repo("hf"), None);
    }

    #[test]
    fn test_extract_hf_repo_hf_colon_no_slashes() {
        assert_eq!(extract_hf_repo("hf:org/repo"), None);
    }

    #[test]
    fn test_extract_hf_repo_hf_single_slash() {
        assert_eq!(extract_hf_repo("hf:/org/repo"), None);
    }

    #[test]
    fn test_extract_hf_repo_numeric_org_and_repo() {
        let uri = "hf://12345/67890/model.safetensors";
        assert_eq!(extract_hf_repo(uri), Some("12345/67890".to_string()));
    }

    #[test]
    fn test_extract_hf_repo_hyphenated_names() {
        let uri = "hf://my-org/my-awesome-model-v2/weights.safetensors";
        assert_eq!(
            extract_hf_repo(uri),
            Some("my-org/my-awesome-model-v2".to_string())
        );
    }

    #[test]
    fn test_extract_hf_repo_underscored_names() {
        let uri = "hf://my_org/my_model_v2";
        assert_eq!(extract_hf_repo(uri), Some("my_org/my_model_v2".to_string()));
    }

    #[test]
    fn test_extract_hf_repo_very_long_names() {
        let long_org = "a".repeat(100);
        let long_repo = "b".repeat(200);
        let uri = format!("hf://{}/{}/model.safetensors", long_org, long_repo);
        assert_eq!(
            extract_hf_repo(&uri),
            Some(format!("{}/{}", long_org, long_repo))
        );
    }

    #[test]
    fn test_extract_hf_repo_with_at_symbol() {
        // Some HF repos use @ for versions
        let uri = "hf://org/repo@main/model.safetensors";
        assert_eq!(extract_hf_repo(uri), Some("org/repo@main".to_string()));
    }

    #[test]
    fn test_extract_hf_repo_empty_after_prefix() {
        // "hf://" → path = "" → parts = [""], len < 2 → None
        assert_eq!(extract_hf_repo("hf://"), None);
    }

    #[test]
    fn test_extract_hf_repo_single_char_org_and_repo() {
        assert_eq!(extract_hf_repo("hf://a/b"), Some("a/b".to_string()));
    }

    // =========================================================================
    // NEW: extract_shard_files_from_index additional edge cases
    // =========================================================================

    #[test]
    fn test_extract_shard_files_truncated_json() {
        // JSON that's cut off mid-stream
        let json = r#"{"weight_map": {"a.weight": "model-00001.safe"#;
        let shards = extract_shard_files_from_index(json);
        assert!(
            shards.is_empty(),
            "Truncated JSON should produce no results"
        );
    }

    #[test]
    fn test_extract_shard_files_unicode_tensor_names() {
        let json = r#"{
            "weight_map": {
                "模型.层.0.权重": "model-00001-of-00001.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], "model-00001-of-00001.safetensors");
    }

    #[test]
    fn test_extract_shard_files_colons_in_tensor_names() {
        // Tensor names with colons (like "model:layers:0:weight") use rfind(':')
        // so the last colon determines the split point
        let json = r#"{
            "weight_map": {
                "model:layers:0:weight": "shard-001.safetensors",
                "model:layers:1:weight": "shard-002.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 2);
    }

    #[test]
    fn test_extract_shard_files_empty_json_object() {
        let json = "{}";
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_extract_shard_files_null_json() {
        let json = "null";
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_extract_shard_files_array_instead_of_object() {
        let json = r#"[{"weight_map": {"a": "model.safetensors"}}]"#;
        // weight_map is inside an array element — the string search still finds it
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 1);
    }

    #[test]
    fn test_extract_shard_files_weight_map_with_empty_value() {
        let json = r#"{"weight_map": {"a.weight": ""}}"#;
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty(), "Empty filename should be excluded");
    }

    #[test]
    fn test_extract_shard_files_weight_map_value_not_safetensors() {
        let json = r#"{"weight_map": {"a.weight": "model.gguf"}}"#;
        let shards = extract_shard_files_from_index(json);
        assert!(
            shards.is_empty(),
            "Non-safetensors files should be excluded"
        );
    }

    #[test]
    fn test_extract_shard_files_large_model_40_shards() {
        // Simulate a very large model with 40 shards
        let mut entries = Vec::new();
        for i in 1..=200 {
            let shard_num = (i % 40) + 1;
            entries.push(format!(
                "\"tensor_{}\": \"model-{:05}-of-00040.safetensors\"",
                i, shard_num
            ));
        }
        let json = format!("{{\"weight_map\": {{{}}}}}", entries.join(",\n"));
        let shards = extract_shard_files_from_index(&json);
        assert_eq!(shards.len(), 40);
        assert_eq!(shards[0], "model-00001-of-00040.safetensors");
        assert_eq!(shards[39], "model-00040-of-00040.safetensors");
    }

    #[test]
    fn test_extract_shard_files_weight_map_appears_in_metadata() {
        // "weight_map" string appears in metadata as well — should find the right one
        let json = r#"{
            "metadata": {"description": "This model has a weight_map section"},
            "weight_map": {
                "a.weight": "actual-shard.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        // The first occurrence of "weight_map" is in metadata description (as string content),
        // but the parser looks for `"weight_map"` and then the next `{`
        // The first `"weight_map"` found is inside the metadata string value, and the next `{` after it
        // would be the actual weight_map object. This is a known edge case of string-based parsing.
        // The result depends on the exact JSON layout.
        assert!(
            !shards.is_empty(),
            "Should find shards from the actual weight_map"
        );
    }

    #[test]
    fn test_extract_shard_files_sorted_alphanumerically() {
        let json = r#"{
            "weight_map": {
                "z": "shard-c.safetensors",
                "y": "shard-a.safetensors",
                "x": "shard-b.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 3);
        assert_eq!(shards[0], "shard-a.safetensors");
        assert_eq!(shards[1], "shard-b.safetensors");
        assert_eq!(shards[2], "shard-c.safetensors");
    }

    #[test]
    fn test_extract_shard_files_weight_map_no_opening_brace() {
        // "weight_map" key exists but is followed by a string, not object
        let json = r#"{"weight_map": "just a string, no object here"}"#;
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_extract_shard_files_multiple_weight_map_keys() {
        // Technically invalid JSON (duplicate keys), but tests parser resilience
        // The string search finds the first "weight_map"
        let json = r#"{
            "weight_map": {"a": "first.safetensors"},
            "weight_map": {"b": "second.safetensors"}
        }"#;
        let shards = extract_shard_files_from_index(json);
        // First weight_map is found; second is ignored
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], "first.safetensors");
    }

    // =========================================================================
    // NEW: ResolvedModel enum tests
    // =========================================================================

    #[test]
    fn test_resolved_model_single_file_debug() {
        let model = ResolvedModel::SingleFile("test.gguf".to_string());
        let debug = format!("{:?}", model);
        assert!(debug.contains("SingleFile"));
        assert!(debug.contains("test.gguf"));
    }

    #[test]
    fn test_resolved_model_sharded_debug() {
        let model = ResolvedModel::Sharded {
            org: "Qwen".to_string(),
            repo: "Qwen2.5-Coder-3B".to_string(),
            shard_files: vec!["shard-001.safetensors".to_string()],
        };
        let debug = format!("{:?}", model);
        assert!(debug.contains("Sharded"));
        assert!(debug.contains("Qwen"));
        assert!(debug.contains("Qwen2.5-Coder-3B"));
        assert!(debug.contains("shard-001.safetensors"));
    }

    #[test]
    fn test_resolved_model_sharded_empty_shard_files() {
        let model = ResolvedModel::Sharded {
            org: "org".to_string(),
            repo: "repo".to_string(),
            shard_files: vec![],
        };
        match model {
            ResolvedModel::Sharded { shard_files, .. } => {
                assert!(shard_files.is_empty());
            }
            _ => panic!("Expected Sharded"),
        }
    }

    // =========================================================================
    // NEW: ShardManifest/FileChecksum additional tests
    // =========================================================================

    #[test]
    fn test_shard_manifest_deserialize_unknown_fields() {
        // Forward compatibility: extra fields should be ignored (serde default)
        let json = r#"{"version": 2, "repo": "org/repo", "files": {}, "extra_field": "ignored"}"#;
        let manifest: ShardManifest =
            serde_json::from_str(json).expect("unknown fields should be ignored by default");
        assert_eq!(manifest.version, 2);
    }

    #[test]
    fn test_shard_manifest_missing_required_field() {
        // Missing "repo" field should fail deserialization
        let json = r#"{"version": 1, "files": {}}"#;
        assert!(serde_json::from_str::<ShardManifest>(json).is_err());
    }

    #[test]
    fn test_file_checksum_missing_blake3_field() {
        // Missing "blake3" should fail
        let json = r#"{"size": 100}"#;
        assert!(serde_json::from_str::<FileChecksum>(json).is_err());
    }

    #[test]
    fn test_file_checksum_missing_size_field() {
        // Missing "size" should fail
        let json = r#"{"blake3": "abc123"}"#;
        assert!(serde_json::from_str::<FileChecksum>(json).is_err());
    }

    #[test]
    fn test_shard_manifest_special_chars_in_repo() {
        let manifest = ShardManifest {
            version: 1,
            repo: "org/repo-with.dots_and-dashes".to_string(),
            files: HashMap::new(),
        };
        let json = serde_json::to_string(&manifest).expect("serialize");
        let deserialized: ShardManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.repo, "org/repo-with.dots_and-dashes");
    }

    #[test]
    fn test_shard_manifest_many_files() {
        let mut files = HashMap::new();
        for i in 0..100 {
            files.insert(
                format!("model-{:05}-of-00100.safetensors", i + 1),
                FileChecksum {
                    size: 5_000_000_000 + i as u64,
                    blake3: format!("hash_{:05}", i),
                },
            );
        }
        let manifest = ShardManifest {
            version: 1,
            repo: "big/model".to_string(),
            files,
        };
        let json = serde_json::to_string(&manifest).expect("serialize");
        let deserialized: ShardManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.files.len(), 100);
    }

    #[test]
    fn test_file_checksum_unicode_blake3() {
        // blake3 field is a string — should handle any string content
        let checksum = FileChecksum {
            size: 42,
            blake3: "hash_with_\u{00e9}moji".to_string(),
        };
        let json = serde_json::to_string(&checksum).expect("serialize");
        let deserialized: FileChecksum = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.blake3, "hash_with_\u{00e9}moji");
    }

    // =========================================================================
    // NEW: resolve_hf_model additional URI edge cases
    // =========================================================================

    #[test]
    fn test_resolve_hf_model_bare_org_repo_with_pt_extension() {
        // "org/repo/model.pt" → normalizes to "hf://org/repo/model.pt" → SingleFile
        let result = resolve_hf_model("org/repo/model.pt").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.pt"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_file_url_scheme() {
        // "file:///path/to/model" has "://" so NOT normalized
        let result = resolve_hf_model("file:///path/to/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "file:///path/to/model.gguf"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_s3_url_scheme() {
        // "s3://bucket/key" has "://" → not normalized, not hf:// → SingleFile
        let result = resolve_hf_model("s3://bucket/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "s3://bucket/model.gguf"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_just_dot() {
        // "." starts with '.' → NOT normalized
        let result = resolve_hf_model(".").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "."),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_empty_string() {
        let result = resolve_hf_model("").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, ""),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_gguf_mixed_case_extension() {
        let result = resolve_hf_model("hf://org/repo/model.GGuF").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.GGuF"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile for .GGuF"),
        }
    }

    #[test]
    fn test_resolve_hf_model_pt_mixed_case() {
        let result = resolve_hf_model("hf://org/repo/model.PT").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.PT"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    // =========================================================================
    // NEW: resolve_hf_uri backward-compat wrapper additional tests
    // =========================================================================

    #[test]
    fn test_resolve_hf_uri_single_file_with_safetensors() {
        // .safetensors has a known extension → SingleFile passthrough
        let uri = "hf://org/repo/model.safetensors";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri);
    }
