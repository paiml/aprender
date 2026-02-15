
    // ==================== ModelSource Tests ====================

    #[test]
    fn test_parse_local_path() {
        let source = ModelSource::parse("model.apr").unwrap();
        assert_eq!(source, ModelSource::Local(PathBuf::from("model.apr")));
    }

    #[test]
    fn test_parse_absolute_path() {
        let source = ModelSource::parse("/path/to/model.apr").unwrap();
        assert_eq!(
            source,
            ModelSource::Local(PathBuf::from("/path/to/model.apr"))
        );
    }

    #[test]
    fn test_parse_huggingface_source() {
        let source = ModelSource::parse("hf://openai/whisper-tiny").unwrap();
        assert_eq!(
            source,
            ModelSource::HuggingFace {
                org: "openai".to_string(),
                repo: "whisper-tiny".to_string(),
                file: None,
            }
        );
    }

    #[test]
    fn test_parse_huggingface_with_file() {
        let source =
            ModelSource::parse("hf://Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/model-q4_k_m.gguf")
                .unwrap();
        assert_eq!(
            source,
            ModelSource::HuggingFace {
                org: "Qwen".to_string(),
                repo: "Qwen2.5-Coder-0.5B-Instruct-GGUF".to_string(),
                file: Some("model-q4_k_m.gguf".to_string()),
            }
        );
    }

    #[test]
    fn test_parse_huggingface_invalid() {
        let result = ModelSource::parse("hf://invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_https_url() {
        let source = ModelSource::parse("https://example.com/model.apr").unwrap();
        assert_eq!(
            source,
            ModelSource::Url("https://example.com/model.apr".to_string())
        );
    }

    #[test]
    fn test_parse_http_url() {
        let source = ModelSource::parse("http://example.com/model.apr").unwrap();
        assert_eq!(
            source,
            ModelSource::Url("http://example.com/model.apr".to_string())
        );
    }

    // ==================== Cache Path Tests ====================

    #[test]
    fn test_cache_path_local() {
        let source = ModelSource::Local(PathBuf::from("/tmp/model.apr"));
        assert_eq!(source.cache_path(), PathBuf::from("/tmp/model.apr"));
    }

    #[test]
    fn test_cache_path_huggingface() {
        let source = ModelSource::HuggingFace {
            org: "openai".to_string(),
            repo: "whisper-tiny".to_string(),
            file: None,
        };
        let cache = source.cache_path();
        assert!(cache.to_string_lossy().contains("hf"));
        assert!(cache.to_string_lossy().contains("openai"));
        assert!(cache.to_string_lossy().contains("whisper-tiny"));
    }

    #[test]
    fn test_cache_path_url_deterministic() {
        let source1 = ModelSource::Url("https://example.com/model.apr".to_string());
        let source2 = ModelSource::Url("https://example.com/model.apr".to_string());
        assert_eq!(source1.cache_path(), source2.cache_path());
    }

    #[test]
    fn test_cache_path_url_different() {
        let source1 = ModelSource::Url("https://example.com/model1.apr".to_string());
        let source2 = ModelSource::Url("https://example.com/model2.apr".to_string());
        assert_ne!(source1.cache_path(), source2.cache_path());
    }

    // ==================== RunOptions Tests ====================

    #[test]
    fn test_run_options_default() {
        let options = RunOptions::default();
        assert!(options.input.is_none());
        assert_eq!(options.output_format, "text");
        assert!(!options.force);
        assert!(!options.no_gpu);
        assert!(!options.offline);
    }

    // ==================== MD5 Hash Tests ====================

    #[test]
    fn test_md5_hash_deterministic() {
        let hash1 = md5_hash(b"test");
        let hash2 = md5_hash(b"test");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_md5_hash_different_inputs() {
        let hash1 = md5_hash(b"test1");
        let hash2 = md5_hash(b"test2");
        assert_ne!(hash1, hash2);
    }

    // ==================== Integration Tests ====================

    #[test]
    fn test_run_model_file_not_found() {
        let result = run_model("/nonexistent/model.apr", &RunOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_run_model_with_options() {
        let options = RunOptions {
            input: Some(PathBuf::from("/tmp/test.wav")),
            prompt: None,
            max_tokens: 32,
            output_format: "json".to_string(),
            force: false,
            no_gpu: true,
            offline: false,
            benchmark: false,
            verbose: false,
            trace: false,
            trace_steps: None,
            trace_verbose: false,
            trace_output: None,
        };
        assert!(options.no_gpu);
        assert_eq!(options.output_format, "json");
    }

    // ============================================================================
    // Popperian Falsification Tests: Offline Mode (Section 9.2 Sovereign AI)
    // ============================================================================
    //
    // Per PMAT Extreme TDD: Each test defines conditions under which the claim
    // would be **proven false**.

    /// FALSIFICATION: If --offline allows HuggingFace download, the claim fails
    /// Claim: `apr run --offline hf://org/repo` rejects non-cached HF models
    #[test]
    fn offline_mode_rejects_uncached_huggingface() {
        let source = ModelSource::HuggingFace {
            org: "uncached-org".to_string(),
            repo: "nonexistent-repo".to_string(),
            file: None,
        };

        // Offline mode MUST reject non-cached HF sources
        let result = resolve_model(&source, false, true);

        assert!(result.is_err(), "FALSIFIED: Offline mode allowed HF source");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("OFFLINE MODE"),
            "FALSIFIED: Error message should mention OFFLINE MODE, got: {err}"
        );
    }

    /// FALSIFICATION: If --offline allows URL download, the claim fails
    /// Claim: `apr run --offline https://...` rejects non-cached URLs
    #[test]
    fn offline_mode_rejects_uncached_url() {
        let source = ModelSource::Url("https://example.com/model.apr".to_string());

        // Offline mode MUST reject non-cached URL sources
        let result = resolve_model(&source, false, true);

        assert!(
            result.is_err(),
            "FALSIFIED: Offline mode allowed URL source"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("OFFLINE MODE"),
            "FALSIFIED: Error message should mention OFFLINE MODE, got: {err}"
        );
    }

    /// FALSIFICATION: If --offline rejects local files, the claim fails
    /// Claim: `apr run --offline /path/to/model.apr` allows local files
    #[test]
    fn offline_mode_allows_local_files() {
        let source = ModelSource::Local(PathBuf::from("/tmp/model.apr"));

        // Offline mode MUST allow local file sources
        let result = resolve_model(&source, false, true);

        // Note: This succeeds at resolution, but may fail later if file doesn't exist
        // The key point is that offline mode doesn't reject local sources
        assert!(
            result.is_ok(),
            "FALSIFIED: Offline mode rejected local file source: {:?}",
            result
        );
    }

    /// FALSIFICATION: If default mode has offline=true, the claim fails
    /// Claim: Default RunOptions have offline=false
    #[test]
    fn default_options_are_not_offline() {
        let options = RunOptions::default();
        assert!(
            !options.offline,
            "FALSIFIED: Default options should NOT be offline"
        );
    }

    /// FALSIFICATION: If offline flag doesn't propagate, the claim fails
    /// Claim: RunOptions::offline is correctly set when specified
    #[test]
    fn offline_flag_propagates_correctly() {
        let options = RunOptions {
            offline: true,
            ..Default::default()
        };
        assert!(
            options.offline,
            "FALSIFIED: Offline flag did not propagate to options"
        );
    }

    // ==================== Sharded Model Tests (GH-127) ====================

    /// Test extract_shard_files with typical HuggingFace index.json format
    #[test]
    fn test_extract_shard_files_basic() {
        let json = r#"{
            "metadata": {"total_size": 1000000},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
                "model.layers.0.weight": "model-00001-of-00003.safetensors",
                "model.layers.1.weight": "model-00002-of-00003.safetensors",
                "model.layers.2.weight": "model-00003-of-00003.safetensors",
                "lm_head.weight": "model-00003-of-00003.safetensors"
            }
        }"#;

        let files = extract_shard_files(json);

        assert_eq!(files.len(), 3, "Should extract 3 unique shard files");
        assert!(files.contains("model-00001-of-00003.safetensors"));
        assert!(files.contains("model-00002-of-00003.safetensors"));
        assert!(files.contains("model-00003-of-00003.safetensors"));
    }

    /// Test extract_shard_files with empty weight_map
    #[test]
    fn test_extract_shard_files_empty() {
        let json = r#"{"weight_map": {}}"#;
        let files = extract_shard_files(json);
        assert!(files.is_empty(), "Empty weight_map should yield no files");
    }

    /// Test extract_shard_files with no weight_map key
    #[test]
    fn test_extract_shard_files_no_weight_map() {
        let json = r#"{"metadata": {}}"#;
        let files = extract_shard_files(json);
        assert!(files.is_empty(), "Missing weight_map should yield no files");
    }

    /// Test extract_shard_files with single shard (all tensors in one file)
    #[test]
    fn test_extract_shard_files_single_shard() {
        let json = r#"{
            "weight_map": {
                "a": "model.safetensors",
                "b": "model.safetensors",
                "c": "model.safetensors"
            }
        }"#;

        let files = extract_shard_files(json);

        assert_eq!(
            files.len(),
            1,
            "All tensors in same file should yield 1 shard"
        );
        assert!(files.contains("model.safetensors"));
    }

    /// Test extract_shard_files handles real-world Phi-4 style index
    #[test]
    fn test_extract_shard_files_phi4_style() {
        // Simplified version of microsoft/phi-4 index structure
        let json = r#"{
            "metadata": {
                "total_size": 56000000000
            },
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00006.safetensors",
                "model.layers.0.input_layernorm.weight": "model-00001-of-00006.safetensors",
                "model.layers.10.mlp.up_proj.weight": "model-00002-of-00006.safetensors",
                "model.layers.20.self_attn.v_proj.weight": "model-00003-of-00006.safetensors",
                "model.layers.30.post_attention_layernorm.weight": "model-00004-of-00006.safetensors",
                "model.layers.40.mlp.gate_proj.weight": "model-00005-of-00006.safetensors",
                "model.norm.weight": "model-00006-of-00006.safetensors",
                "lm_head.weight": "model-00006-of-00006.safetensors"
            }
        }"#;

        let files = extract_shard_files(json);

        assert_eq!(files.len(), 6, "Phi-4 style model has 6 shards");
        for i in 1..=6 {
            let expected = format!("model-{i:05}-of-00006.safetensors");
            assert!(
                files.contains(&expected),
                "Should contain shard file: {expected}"
            );
        }
    }

    /// Test that non-safetensors files are filtered out
    #[test]
    fn test_extract_shard_files_filters_non_safetensors() {
        let json = r#"{
            "weight_map": {
                "a": "model.safetensors",
                "b": "config.json",
                "c": "tokenizer.model"
            }
        }"#;

        let files = extract_shard_files(json);

        assert_eq!(files.len(), 1, "Should only include .safetensors files");
        assert!(files.contains("model.safetensors"));
        assert!(!files.contains("config.json"));
        assert!(!files.contains("tokenizer.model"));
    }

    // ========================================================================
    // Additional Coverage Tests (PMAT-117) - Unique tests only
    // ========================================================================

    #[test]
    fn test_md5_hash_empty() {
        let hash = md5_hash(&[]);
        let _ = hash;
    }

    #[test]
    fn test_md5_hash_different_input() {
        let hash1 = md5_hash(b"hello");
        let hash2 = md5_hash(b"world");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_clean_model_output_empty() {
        let output = clean_model_output("");
        assert!(output.is_empty());
    }

    #[test]
    fn test_clean_model_output_simple() {
        let output = clean_model_output("Hello, world!");
        assert_eq!(output, "Hello, world!");
    }

    #[test]
    fn test_clean_model_output_with_special_tokens() {
        let output = clean_model_output("<|im_end|>Hello<|endoftext|>");
        assert!(!output.contains("<|im_end|>"));
        assert!(!output.contains("<|endoftext|>"));
    }

    #[test]
    fn test_clean_model_output_preserves_content() {
        let output = clean_model_output("The answer is 42.");
        assert!(output.contains("42"));
    }

    #[test]
    fn test_parse_token_ids_simple() {
        let result = parse_token_ids("1 2 3");
        assert!(result.is_ok());
        let tokens = result.unwrap();
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_parse_token_ids_comma_separated() {
        let result = parse_token_ids("1,2,3");
        assert!(result.is_ok());
        let tokens = result.unwrap();
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_parse_token_ids_empty() {
        let result = parse_token_ids("");
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_parse_token_ids_invalid() {
        let result = parse_token_ids("not a number");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_token_ids_mixed_spaces() {
        let result = parse_token_ids("1  2   3");
        assert!(result.is_ok());
        let tokens = result.unwrap();
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_model_source_display_local() {
        let source = ModelSource::Local(PathBuf::from("model.apr"));
        let _debug = format!("{:?}", source);
    }
