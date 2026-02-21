
    #[test]
    fn test_gh213_resolve_hf_invalid_uri_fails() {
        let result = resolve_hf_model("hf://invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_gh213_resolve_bare_org_repo_normalizes() {
        // "Qwen/Qwen2.5-Coder-3B-Instruct" should be treated as "hf://Qwen/Qwen2.5-Coder-3B-Instruct"
        // Can't test full resolution without network, but verify it doesn't return as SingleFile unchanged
        let result = resolve_hf_model("Qwen/FakeRepo");
        // Will fail with network error (repo doesn't exist), which proves it tried HF API
        assert!(
            result.is_err(),
            "Bare org/repo should attempt HF resolution"
        );
    }

    #[test]
    fn test_gh213_resolve_bare_org_repo_with_gguf_extension() {
        // "org/repo/file.gguf" should normalize to "hf://org/repo/file.gguf" → SingleFile
        let result = resolve_hf_model("org/repo/model.gguf").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => {
                assert_eq!(s, "hf://org/repo/model.gguf");
            }
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_gh213_resolve_bare_single_component_unchanged() {
        // "justAName" (no slash) should not be normalized, stays as local path
        let result = resolve_hf_model("justAName").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "justAName"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_gh213_resolve_relative_path_not_normalized() {
        // "./path/to/model" should NOT be treated as org/repo
        let result = resolve_hf_model("./path/to/model").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "./path/to/model"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_gh213_resolve_absolute_path_not_normalized() {
        // "/home/user/model" should NOT be treated as org/repo
        let result = resolve_hf_model("/home/user/model").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "/home/user/model"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    // Integration tests (require network, marked ignore for CI)
    #[test]
    #[ignore]
    fn test_gh213_resolve_small_model_is_single_file() {
        // 0.5B model has a single model.safetensors
        let result = resolve_hf_model("hf://Qwen/Qwen2.5-Coder-0.5B-Instruct").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => {
                assert!(
                    s.contains("model.safetensors"),
                    "Should resolve to model.safetensors: {}",
                    s
                );
            }
            ResolvedModel::Sharded { .. } => panic!("0.5B should be single file, not sharded"),
        }
    }

    #[test]
    #[ignore]
    fn test_gh213_resolve_large_model_is_sharded() {
        // 3B+ models use sharded SafeTensors
        let result = resolve_hf_model("hf://Qwen/Qwen2.5-Coder-3B-Instruct").unwrap();
        match result {
            ResolvedModel::Sharded {
                org,
                repo,
                shard_files,
            } => {
                assert_eq!(org, "Qwen");
                assert_eq!(repo, "Qwen2.5-Coder-3B-Instruct");
                assert!(
                    shard_files.len() > 1,
                    "3B model should have multiple shards, got {}",
                    shard_files.len()
                );
                // All shards should end with .safetensors
                for f in &shard_files {
                    assert!(
                        f.ends_with(".safetensors"),
                        "Shard should be .safetensors: {}",
                        f
                    );
                }
            }
            ResolvedModel::SingleFile(s) => {
                panic!("3B should be sharded, got SingleFile({})", s)
            }
        }
    }

    #[test]
    #[ignore]
    fn test_gh213_resolve_7b_model_is_sharded() {
        let result = resolve_hf_model("hf://Qwen/Qwen2.5-Coder-7B-Instruct").unwrap();
        match result {
            ResolvedModel::Sharded { shard_files, .. } => {
                assert!(
                    shard_files.len() > 1,
                    "7B model should have multiple shards, got {}",
                    shard_files.len()
                );
            }
            ResolvedModel::SingleFile(s) => {
                panic!("7B should be sharded, got SingleFile({})", s)
            }
        }
    }

    // =========================================================================
    // format_bytes: exhaustive boundary tests
    // =========================================================================

    #[test]
    fn test_format_bytes_boundary_just_below_kb() {
        assert_eq!(format_bytes(1023), "1023 B");
    }

    #[test]
    fn test_format_bytes_boundary_exact_kb() {
        assert_eq!(format_bytes(1024), "1.0 KB");
    }

    #[test]
    fn test_format_bytes_boundary_just_above_kb() {
        assert_eq!(format_bytes(1025), "1.0 KB");
    }

    #[test]
    fn test_format_bytes_boundary_just_below_mb() {
        // 1 MB - 1 byte = 1048575 bytes → KB range
        assert_eq!(format_bytes(1_048_575), "1024.0 KB");
    }

    #[test]
    fn test_format_bytes_boundary_exact_mb() {
        assert_eq!(format_bytes(1_048_576), "1.0 MB");
    }

    #[test]
    fn test_format_bytes_boundary_just_below_gb() {
        // 1 GB - 1 byte = 1073741823 bytes → MB range
        assert_eq!(format_bytes(1_073_741_823), "1024.0 MB");
    }

    #[test]
    fn test_format_bytes_boundary_exact_gb() {
        assert_eq!(format_bytes(1_073_741_824), "1.0 GB");
    }

    #[test]
    fn test_format_bytes_large_gb() {
        // 100 GB
        assert_eq!(format_bytes(107_374_182_400), "100.0 GB");
    }

    #[test]
    fn test_format_bytes_u64_max() {
        // u64::MAX should not panic, gives some large TB value
        let result = format_bytes(u64::MAX);
        assert!(
            result.contains("TB"),
            "u64::MAX should be in TB range: {}",
            result
        );
    }

    #[test]
    fn test_format_bytes_fractional_kb() {
        // 1.5 KB = 1536 bytes
        assert_eq!(format_bytes(1536), "1.5 KB");
    }

    #[test]
    fn test_format_bytes_7b_model_size() {
        // ~4.1 GB typical for 7B Q4_K_M
        assert_eq!(format_bytes(4_402_341_888), "4.1 GB");
    }

    // =========================================================================
    // extract_hf_repo: comprehensive edge cases
    // =========================================================================

    #[test]
    fn test_extract_hf_repo_just_prefix() {
        // "hf://" with nothing after → parts = [""], len < 2
        assert_eq!(extract_hf_repo("hf://"), None);
    }

    #[test]
    fn test_extract_hf_repo_single_slash_after_prefix() {
        // "hf://org/" → parts = ["org", ""], but parts[1] is empty
        // Still returns Some because len >= 2 and format just joins
        let result = extract_hf_repo("hf://org/");
        assert_eq!(result, Some("org/".to_string()));
    }

    #[test]
    fn test_extract_hf_repo_with_multiple_nested_paths() {
        // Deep nesting: only org/repo extracted
        let uri = "hf://org/repo/subdir1/subdir2/model.safetensors";
        assert_eq!(extract_hf_repo(uri), Some("org/repo".to_string()));
    }

    #[test]
    fn test_extract_hf_repo_wrong_scheme() {
        assert_eq!(extract_hf_repo("https://huggingface.co/org/repo"), None);
    }

    #[test]
    fn test_extract_hf_repo_no_scheme() {
        assert_eq!(extract_hf_repo("org/repo/model.gguf"), None);
    }

    #[test]
    fn test_extract_hf_repo_hf_prefix_case_sensitive() {
        // "HF://" (uppercase) should not match
        assert_eq!(extract_hf_repo("HF://org/repo"), None);
    }

    #[test]
    fn test_extract_hf_repo_special_chars_in_name() {
        let uri = "hf://TheBloke/Llama-2-7B-GGUF/model.gguf";
        assert_eq!(
            extract_hf_repo(uri),
            Some("TheBloke/Llama-2-7B-GGUF".to_string())
        );
    }

    #[test]
    fn test_extract_hf_repo_dots_in_name() {
        let uri = "hf://org/model.name.v2/file.safetensors";
        assert_eq!(extract_hf_repo(uri), Some("org/model.name.v2".to_string()));
    }

    // =========================================================================
    // extract_shard_files_from_index: comprehensive edge cases
    // =========================================================================

    #[test]
    fn test_extract_shard_files_single_shard() {
        let json = r#"{"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], "model-00001-of-00001.safetensors");
    }

    #[test]
    fn test_extract_shard_files_many_shards() {
        // 6 shards with heavy deduplication
        let json = r#"{
            "weight_map": {
                "a": "model-00001-of-00006.safetensors",
                "b": "model-00001-of-00006.safetensors",
                "c": "model-00002-of-00006.safetensors",
                "d": "model-00003-of-00006.safetensors",
                "e": "model-00004-of-00006.safetensors",
                "f": "model-00005-of-00006.safetensors",
                "g": "model-00005-of-00006.safetensors",
                "h": "model-00006-of-00006.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 6);
        assert_eq!(shards[0], "model-00001-of-00006.safetensors");
        assert_eq!(shards[5], "model-00006-of-00006.safetensors");
    }

    #[test]
    fn test_extract_shard_files_empty_string() {
        let shards = extract_shard_files_from_index("");
        assert!(shards.is_empty());
    }

    #[test]
    fn test_extract_shard_files_no_weight_map_key() {
        let json = r#"{"other_key": {"a": "file.safetensors"}}"#;
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_extract_shard_files_weight_map_not_object() {
        // weight_map is a string, not object — should not crash
        let json = r#"{"weight_map": "not an object"}"#;
        let shards = extract_shard_files_from_index(json);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_extract_shard_files_mixed_extensions() {
        // Only .safetensors files should be included
        let json = r#"{
            "weight_map": {
                "a": "model-00001.safetensors",
                "b": "model-00002.bin",
                "c": "model-00003.pt",
                "d": "model-00004.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 2);
        assert!(shards.contains(&"model-00001.safetensors".to_string()));
        assert!(shards.contains(&"model-00004.safetensors".to_string()));
    }

    #[test]
    fn test_extract_shard_files_nested_braces() {
        // JSON with nested braces in metadata before weight_map
        let json = r#"{
            "metadata": {"nested": {"deep": "value"}},
            "weight_map": {
                "a.weight": "shard-001.safetensors",
                "b.weight": "shard-002.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 2);
    }

    #[test]
    fn test_extract_shard_files_whitespace_in_values() {
        // Extra whitespace and newlines around values
        let json = r#"{
            "weight_map": {
                "a.weight":   "  model-00001.safetensors  "  ,
                "b.weight":
                    "model-00002.safetensors"
            }
        }"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 2);
    }

    #[test]
    fn test_extract_shard_files_values_with_path_separators() {
        // Filenames shouldn't have path separators, but test robustness
        let json = r#"{"weight_map": {"a": "subdir/model.safetensors"}}"#;
        let shards = extract_shard_files_from_index(json);
        // Contains "/" so not matching simple pattern, but the function does string trim
        // It checks ends_with(".safetensors")
        assert_eq!(shards.len(), 1);
    }

    #[test]
    fn test_extract_shard_files_real_qwen_format() {
        // Realistic index.json fragment from Qwen2.5-Coder-3B-Instruct
        let json = r#"{
  "metadata": {
    "total_size": 6534782976
  },
  "weight_map": {
    "lm_head.weight": "model-00002-of-00002.safetensors",
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.mlp.up_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.post_attention_layernorm.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00002.safetensors",
    "model.layers.35.self_attn.v_proj.weight": "model-00002-of-00002.safetensors",
    "model.norm.weight": "model-00002-of-00002.safetensors"
  }
}"#;
        let shards = extract_shard_files_from_index(json);
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0], "model-00001-of-00002.safetensors");
        assert_eq!(shards[1], "model-00002-of-00002.safetensors");
    }

    // =========================================================================
    // resolve_hf_model: offline URI normalization & extension detection
    // =========================================================================

    #[test]
    fn test_resolve_hf_model_with_apr_extension() {
        let result = resolve_hf_model("hf://org/repo/model.apr").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.apr"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile for .apr"),
        }
    }

    #[test]
    fn test_resolve_hf_model_with_pt_extension() {
        let result = resolve_hf_model("hf://org/repo/model.pt").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.pt"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile for .pt"),
        }
    }

    #[test]
    fn test_resolve_hf_model_case_insensitive_safetensors() {
        let result = resolve_hf_model("hf://org/repo/model.SafeTensors").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.SafeTensors"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }

    #[test]
    fn test_resolve_hf_model_with_mixed_case_apr() {
        let result = resolve_hf_model("hf://org/repo/model.APR").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.APR"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile for .APR"),
        }
    }

    #[test]
    fn test_resolve_hf_model_bare_org_repo_with_safetensors() {
        // "org/repo/model.safetensors" → "hf://org/repo/model.safetensors" → SingleFile
        let result = resolve_hf_model("org/repo/model.safetensors").unwrap();
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "hf://org/repo/model.safetensors"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile"),
        }
    }
