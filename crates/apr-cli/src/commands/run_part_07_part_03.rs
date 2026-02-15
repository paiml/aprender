
    #[test]
    fn test_model_source_display_hf() {
        let source = ModelSource::HuggingFace {
            org: "test".to_string(),
            repo: "model".to_string(),
            file: Some("model.gguf".to_string()),
        };
        let _debug = format!("{:?}", source);
    }

    #[test]
    fn test_model_source_clone() {
        let source = ModelSource::Url("https://example.com/model.apr".to_string());
        let cloned = source.clone();
        assert_eq!(source, cloned);
    }

    #[test]
    fn test_run_options_custom() {
        let options = RunOptions {
            max_tokens: 100,
            benchmark: true,
            verbose: true,
            ..Default::default()
        };
        assert_eq!(options.max_tokens, 100);
        assert!(options.benchmark);
    }

    #[test]
    fn test_run_result_debug() {
        let result = RunResult {
            text: "Hello".to_string(),
            duration_secs: 0.1,
            cached: true,
            tokens_generated: Some(5),
            tok_per_sec: None,
            used_gpu: None,
            generated_tokens: None,
        };
        let _debug = format!("{:?}", result);
        assert_eq!(result.tokens_generated, Some(5));
    }

    #[test]
    fn test_extract_shard_files_empty_json() {
        let json = "{}";
        let files = extract_shard_files(json);
        assert!(files.is_empty());
    }

    #[test]
    fn test_extract_shard_files_invalid_json() {
        let json = "not valid json";
        let files = extract_shard_files(json);
        assert!(files.is_empty());
    }

    #[test]
    fn test_cache_path_url_contains_urls_dir() {
        let source = ModelSource::Url("https://example.com/model.apr".to_string());
        let cache = source.cache_path();
        assert!(cache.to_string_lossy().contains("urls"));
    }

    #[test]
    fn test_cache_path_hf_with_file() {
        let source = ModelSource::HuggingFace {
            org: "test".to_string(),
            repo: "model".to_string(),
            file: Some("model-q4.gguf".to_string()),
        };
        let cache = source.cache_path();
        assert!(cache.to_string_lossy().contains("test"));
        assert!(cache.to_string_lossy().contains("model"));
    }

    #[test]
    fn test_find_model_in_dir_returns_dir_if_no_model() {
        let result = find_model_in_dir(Path::new("/nonexistent/directory"));
        // Returns Ok with the directory path if no model found
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), PathBuf::from("/nonexistent/directory"));
    }

    #[test]
    fn test_glob_first_no_match() {
        let result = glob_first(Path::new("/nonexistent/*.gguf"));
        assert!(result.is_none());
    }

    #[test]
    fn test_format_prediction_output_single() {
        use std::time::Duration;
        let options = RunOptions::default();
        let result =
            format_prediction_output(&[0.9, 0.05, 0.05], Duration::from_millis(100), &options);
        assert!(result.is_ok());
    }

    #[test]
    fn test_format_prediction_output_json() {
        use std::time::Duration;
        let options = RunOptions {
            output_format: "json".to_string(),
            ..Default::default()
        };
        let result = format_prediction_output(&[0.5, 0.5], Duration::from_millis(50), &options);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("predictions"));
    }

    #[test]
    fn test_format_prediction_output_empty() {
        use std::time::Duration;
        let options = RunOptions::default();
        let result = format_prediction_output(&[], Duration::from_millis(10), &options);
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_model_local_returns_path() {
        let source = ModelSource::Local(PathBuf::from("/nonexistent/model.apr"));
        let result = resolve_model(&source, false, false);
        // Local paths return Ok (existence check happens later)
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), PathBuf::from("/nonexistent/model.apr"));
    }

    #[test]
    fn test_resolve_model_offline_hf() {
        let source = ModelSource::HuggingFace {
            org: "test".to_string(),
            repo: "model".to_string(),
            file: None,
        };
        let result = resolve_model(&source, false, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_find_cached_model_not_exists() {
        let result = find_cached_model("nonexistent_org", "nonexistent_repo", None);
        assert!(result.is_none());
    }

    #[test]
    fn test_run_model_invalid_source() {
        let options = RunOptions::default();
        let result = run_model("hf://", &options);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_input_features_none() {
        let result = parse_input_features(None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_parse_input_features_file_not_found() {
        let path = PathBuf::from("/nonexistent/input.wav");
        let result = parse_input_features(Some(&path));
        assert!(result.is_err());
    }

    #[test]
    fn test_run_options_with_trace() {
        let options = RunOptions {
            trace: true,
            trace_verbose: true,
            trace_steps: Some(vec!["embed".to_string()]),
            ..Default::default()
        };
        assert!(options.trace);
        assert!(options.trace_verbose);
    }

    #[test]
    fn test_run_result_clone() {
        let result = RunResult {
            text: "Test".to_string(),
            duration_secs: 1.0,
            cached: false,
            tokens_generated: None,
            tok_per_sec: None,
            used_gpu: None,
            generated_tokens: None,
        };
        let cloned = result.clone();
        assert_eq!(result.text, cloned.text);
    }

    // ========================================================================
    // clean_model_output: ChatML marker stripping (bug class: partial strip)
    // ========================================================================

    /// Verify the assistant prefix with trailing newline is stripped.
    /// Bug class: off-by-one in marker list omitting the newline variant.
    #[test]
    fn clean_model_output_strips_assistant_prefix_with_newline() {
        let raw = "<|im_start|>assistant\nThe answer is 42.";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "The answer is 42.");
    }

    /// Verify multiple distinct markers in a single string are all removed.
    /// Bug class: first-match-only replacement instead of replace-all.
    #[test]
    fn clean_model_output_strips_all_markers_simultaneously() {
        let raw = "<|im_start|>assistant\nHello<|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello");
    }

    /// Verify repeated occurrences of the same marker are all stripped.
    /// Bug class: replace() only removing first occurrence (not the case
    /// in Rust, but the test documents the invariant).
    #[test]
    fn clean_model_output_strips_repeated_markers() {
        let raw = "<|im_end|>text<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "text");
    }

    /// Verify that leading/trailing whitespace around markers is trimmed.
    /// Bug class: markers removed but residual whitespace left behind.
    #[test]
    fn clean_model_output_trims_whitespace_after_removal() {
        let raw = "  <|im_end|>  \n  Hello  \n  <|endoftext|>  ";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello");
    }

    /// Verify that text containing partial marker-like sequences is preserved.
    /// Bug class: overly greedy regex stripping content that looks similar.
    #[test]
    fn clean_model_output_preserves_partial_marker_text() {
        let raw = "Use <|tag|> for formatting";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Use <|tag|> for formatting");
    }

    /// Verify Unicode content is preserved through marker stripping.
    /// Bug class: byte-level replacement corrupting multi-byte chars.
    #[test]
    fn clean_model_output_preserves_unicode() {
        let raw = "<|im_start|>assistant\n\u{1f600} Hello \u{00e9}\u{00e8}<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "\u{1f600} Hello \u{00e9}\u{00e8}");
    }

    // ========================================================================
    // ModelSource::parse edge cases
    // ========================================================================

    /// HuggingFace paths with file at parts[2] containing a dot.
    /// Verifies the file detection triggers on dots in the third segment.
    #[test]
    fn parse_hf_file_with_dot_in_third_segment() {
        let source = ModelSource::parse("hf://org/repo/model-q4.gguf").expect("should parse");
        match source {
            ModelSource::HuggingFace { org, repo, file } => {
                assert_eq!(org, "org");
                assert_eq!(repo, "repo");
                assert_eq!(file, Some("model-q4.gguf".to_string()));
            }
            other => panic!("Expected HuggingFace, got {other:?}"),
        }
    }

    /// HuggingFace paths with subdirectory (no dot in parts[2]) treat it as
    /// non-file. Documents current behavior: file detection requires a dot
    /// in the third path segment specifically.
    /// Bug class: subdirectory path silently dropped instead of joined.
    #[test]
    fn parse_hf_subdir_without_dot_is_not_file() {
        let source = ModelSource::parse("hf://org/repo/subdir/model.gguf").expect("should parse");
        match source {
            ModelSource::HuggingFace { file, .. } => {
                // "subdir" has no dot, so file detection does not trigger
                assert_eq!(
                    file, None,
                    "Third segment without dot should not trigger file detection"
                );
            }
            other => panic!("Expected HuggingFace, got {other:?}"),
        }
    }

    /// HuggingFace with exactly two path segments and no file (no dot).
    /// Bug class: third segment without a dot being treated as a file.
    #[test]
    fn parse_hf_three_segments_no_extension() {
        let source = ModelSource::parse("hf://org/repo/branch").expect("should parse");
        match source {
            ModelSource::HuggingFace { file, .. } => {
                assert_eq!(
                    file, None,
                    "Segment without dot should not be treated as file"
                );
            }
            other => panic!("Expected HuggingFace, got {other:?}"),
        }
    }

    /// Empty string should parse as a local path (not panic or error).
    /// Bug class: unwrap on empty string in strip_prefix.
    #[test]
    fn parse_empty_string_is_local() {
        let source = ModelSource::parse("").expect("should parse");
        assert_eq!(source, ModelSource::Local(PathBuf::from("")));
    }

    /// Path with dots but no scheme should be local, not URL.
    /// Bug class: "model.v2.apr" misinterpreted as URL scheme.
    #[test]
    fn parse_dotted_filename_is_local() {
        let source = ModelSource::parse("model.v2.safetensors").expect("should parse");
        assert_eq!(
            source,
            ModelSource::Local(PathBuf::from("model.v2.safetensors"))
        );
    }

    // ========================================================================
    // md5_hash: avalanche and distribution properties
    // ========================================================================

    /// Single-bit difference must produce different hash (avalanche property).
    /// Bug class: hash function ignoring low bits of input bytes.
    #[test]
    fn md5_hash_single_byte_difference() {
        let h1 = md5_hash(b"aaaa");
        let h2 = md5_hash(b"aaab");
        assert_ne!(h1, h2, "Single byte change must produce different hash");
        // Verify reasonable bit spread (at least 8 bits differ)
        let diff_bits = (h1 ^ h2).count_ones();
        assert!(
            diff_bits >= 8,
            "Expected avalanche effect (>=8 bits differ), got {diff_bits}"
        );
    }

    /// Hash of all-zero bytes should not be zero (weak hash detection).
    /// Bug class: XOR-only hash returning zero for zero input.
    #[test]
    fn md5_hash_zero_bytes_nonzero() {
        let h = md5_hash(&[0u8; 100]);
        assert_ne!(h, 0, "Hash of zero bytes must not be zero");
    }

    /// Hash should be order-dependent (not a commutative operation).
    /// Bug class: hash treating input as a multiset rather than sequence.
    #[test]
    fn md5_hash_order_dependent() {
        let h1 = md5_hash(b"ab");
        let h2 = md5_hash(b"ba");
        assert_ne!(h1, h2, "Hash must be order-dependent");
    }

    /// Long input should not overflow or panic.
    /// Bug class: integer overflow in accumulator.
    #[test]
    fn md5_hash_large_input() {
        let data = vec![0xFFu8; 10_000];
        let h = md5_hash(&data);
        let _ = h; // No panic = pass
    }

    // ========================================================================
    // extract_shard_files: malformed JSON edge cases
    // ========================================================================

    /// Keys containing colons should not confuse the colon-based splitting.
    /// Bug class: rfind(':') matching inside tensor name instead of delimiter.
    #[test]
    fn extract_shard_files_colon_in_key() {
        let json = r#"{
            "weight_map": {
                "model:layer:0.weight": "shard-00001.safetensors"
            }
        }"#;
        let files = extract_shard_files(json);
        assert_eq!(files.len(), 1);
        assert!(files.contains("shard-00001.safetensors"));
    }

    /// Whitespace-heavy formatting should not break parsing.
    /// Bug class: trim not handling \r\n on Windows-style JSON.
    #[test]
    fn extract_shard_files_crlf_formatting() {
        let json = "{\r\n  \"weight_map\": {\r\n    \"a\": \"model-00001.safetensors\"\r\n  }\r\n}";
        let files = extract_shard_files(json);
        assert_eq!(files.len(), 1);
    }

    // ========================================================================
    // parse_token_ids: format handling
    // ========================================================================

    /// JSON array format: [1, 2, 3]
    /// Bug class: JSON path not triggered without leading bracket.
    #[test]
    fn parse_token_ids_json_array() {
        let result = parse_token_ids("[1, 2, 3]").expect("should parse JSON array");
        assert_eq!(result, vec![1, 2, 3]);
    }

    /// Tab-separated values (TSV format).
    /// Bug class: only comma and space as separators, missing tab.
    #[test]
    fn parse_token_ids_tab_separated() {
        let result = parse_token_ids("10\t20\t30").expect("should parse TSV");
        assert_eq!(result, vec![10, 20, 30]);
    }

    /// Newline-separated token IDs (one per line).
    /// Bug class: newline not in separator list.
    #[test]
    fn parse_token_ids_newline_separated() {
        let result = parse_token_ids("100\n200\n300").expect("should parse newlines");
        assert_eq!(result, vec![100, 200, 300]);
    }

    /// Token IDs with leading/trailing whitespace.
    /// Bug class: parse::<u32>() failing on untrimmed strings.
    #[test]
    fn parse_token_ids_with_padding() {
        let result = parse_token_ids("  42 , 43 , 44  ").expect("should handle padding");
        assert_eq!(result, vec![42, 43, 44]);
    }

    /// Maximum u32 token ID should not overflow.
    /// Bug class: using u16 or i32 instead of u32 for token IDs.
    #[test]
    fn parse_token_ids_max_u32() {
        let input = format!("{}", u32::MAX);
        let result = parse_token_ids(&input).expect("should parse max u32");
        assert_eq!(result, vec![u32::MAX]);
    }
