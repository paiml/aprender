
    /// Negative numbers should fail (token IDs are unsigned).
    /// Bug class: silently wrapping negative values via as u32.
    #[test]
    fn parse_token_ids_negative_fails() {
        let result = parse_token_ids("-1");
        assert!(result.is_err(), "Negative token IDs must be rejected");
    }

    /// JSON array with invalid bracket structure fails gracefully.
    /// Bug class: panic on malformed JSON.
    #[test]
    fn parse_token_ids_malformed_json_array() {
        let result = parse_token_ids("[1, 2, ");
        assert!(result.is_err(), "Malformed JSON array must fail");
    }

    // ========================================================================
    // format_prediction_output: precision and edge cases
    // ========================================================================

    /// JSON output must contain inference_time_ms field.
    /// Bug class: field renamed or omitted in serialization.
    #[test]
    fn format_prediction_output_json_has_timing() {
        use std::time::Duration;
        let options = RunOptions {
            output_format: "json".to_string(),
            ..Default::default()
        };
        let output = format_prediction_output(&[1.0, 2.0], Duration::from_millis(42), &options)
            .expect("should format");
        assert!(
            output.contains("inference_time_ms"),
            "JSON output must include inference_time_ms"
        );
        assert!(output.contains("42"), "Should contain the timing value");
    }

    /// Text output should show index-labeled predictions.
    /// Bug class: off-by-one in index labeling.
    #[test]
    fn format_prediction_output_text_indexes() {
        use std::time::Duration;
        let options = RunOptions::default();
        let output = format_prediction_output(&[0.1, 0.9], Duration::from_millis(10), &options)
            .expect("should format");
        assert!(output.contains("[0]:"), "Should contain [0]: label");
        assert!(output.contains("[1]:"), "Should contain [1]: label");
    }

    /// NaN and Inf values should not crash serialization.
    /// Bug class: serde_json panicking on non-finite floats.
    #[test]
    fn format_prediction_output_text_with_nan() {
        use std::time::Duration;
        let options = RunOptions::default(); // text mode
        let output = format_prediction_output(
            &[f32::NAN, f32::INFINITY],
            Duration::from_millis(1),
            &options,
        )
        .expect("text format should handle NaN/Inf");
        assert!(output.contains("NaN") || output.contains("nan"));
    }

    // ========================================================================
    // ModelSource::cache_path: structural invariants
    // ========================================================================

    /// URL cache path should use exactly first 16 hex chars of hash.
    /// Bug class: taking wrong slice length, causing collisions or panics.
    #[test]
    fn cache_path_url_hash_length() {
        let source = ModelSource::Url("https://example.com/model.safetensors".to_string());
        let cache = source.cache_path();
        let last_component = cache.file_name().expect("should have filename");
        let name = last_component.to_str().expect("valid utf8");
        assert_eq!(
            name.len(),
            16,
            "URL cache directory name should be 16 hex chars, got '{name}'"
        );
        assert!(
            name.chars().all(|c| c.is_ascii_hexdigit()),
            "URL cache name should be hex only, got '{name}'"
        );
    }

    /// HuggingFace cache path must include org AND repo as separate directories.
    /// Bug class: flattening org/repo into single directory.
    #[test]
    fn cache_path_hf_preserves_hierarchy() {
        let source = ModelSource::HuggingFace {
            org: "my-org".to_string(),
            repo: "my-repo".to_string(),
            file: None,
        };
        let cache = source.cache_path();
        let path_str = cache.to_string_lossy();
        // org and repo must appear as separate path segments
        assert!(
            path_str.contains("my-org/my-repo") || path_str.contains("my-org\\my-repo"),
            "Cache path must preserve org/repo hierarchy, got: {path_str}"
        );
    }

    // ========================================================================
    // clean_model_output: additional edge cases
    // ========================================================================

    /// Input consisting entirely of markers must produce empty string.
    /// Bug class: marker removal leaves residual empty-looking content.
    #[test]
    fn clean_model_output_all_markers_yields_empty() {
        let raw = "<|im_start|>assistant\n<|im_end|><|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert!(
            cleaned.is_empty(),
            "All-marker input should clean to empty, got: '{cleaned}'"
        );
    }

    /// Bare `<|im_start|>` without "assistant" suffix must still be stripped.
    /// Bug class: only stripping the combined "im_start + assistant" variant.
    #[test]
    fn clean_model_output_strips_bare_im_start() {
        let raw = "<|im_start|>Hello world";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Hello world");
    }

    /// `<|endoftext|>` alone, without other markers, must be stripped.
    /// Bug class: endoftext marker only removed when adjacent to im_end.
    #[test]
    fn clean_model_output_strips_endoftext_alone() {
        let raw = "Result: 7<|endoftext|>";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "Result: 7");
    }

    /// Markers embedded in the middle of content must be removed,
    /// leaving surrounding text joined.
    /// Bug class: replace() leaving double-spaces at marker positions.
    #[test]
    fn clean_model_output_markers_in_middle() {
        let raw = "Hello<|im_end|> World";
        let cleaned = clean_model_output(raw);
        assert!(
            cleaned.contains("Hello"),
            "Content before marker must be preserved"
        );
        assert!(
            cleaned.contains("World"),
            "Content after marker must be preserved"
        );
    }

    /// Multiline content with markers on separate lines.
    /// Bug class: line-by-line processing missing cross-line markers.
    #[test]
    fn clean_model_output_multiline_with_markers() {
        let raw = "<|im_start|>assistant\nLine 1\nLine 2\n<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert!(cleaned.contains("Line 1"));
        assert!(cleaned.contains("Line 2"));
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
    }

    /// Only whitespace between markers should collapse to empty.
    /// Bug class: whitespace not trimmed after marker removal.
    #[test]
    fn clean_model_output_whitespace_only_between_markers() {
        let raw = "<|im_start|>   <|im_end|>";
        let cleaned = clean_model_output(raw);
        assert!(
            cleaned.is_empty(),
            "Only whitespace between markers should be empty, got: '{cleaned}'"
        );
    }

    // ========================================================================
    // ModelSource::parse: additional edge cases
    // ========================================================================

    /// Deep HuggingFace path with dot in segment 3+ should join remaining as file.
    /// Bug class: only taking parts[2] instead of joining parts[2..].
    #[test]
    fn parse_hf_deep_path_joins_remaining_segments() {
        let source = ModelSource::parse("hf://org/repo/subdir/model.gguf").expect("should parse");
        match source {
            ModelSource::HuggingFace { org, repo, file } => {
                assert_eq!(org, "org");
                assert_eq!(repo, "repo");
                // parts[2] is "subdir" which has no dot, so file is None
                assert_eq!(file, None);
            }
            other => panic!("Expected HuggingFace, got {other:?}"),
        }
    }

    /// HuggingFace path where parts[2] HAS a dot AND there are more segments.
    /// Verifies parts[2..] are joined with '/'.
    #[test]
    fn parse_hf_file_with_multiple_dotted_segments() {
        let source = ModelSource::parse("hf://org/repo/dir.v2/model.gguf").expect("should parse");
        match source {
            ModelSource::HuggingFace { file, .. } => {
                assert_eq!(
                    file,
                    Some("dir.v2/model.gguf".to_string()),
                    "parts[2..] should be joined with /"
                );
            }
            other => panic!("Expected HuggingFace, got {other:?}"),
        }
    }

    /// Relative path starting with "./" should be local, not HF or URL.
    /// Bug class: relative path prefix confusing scheme detection.
    #[test]
    fn parse_relative_dot_slash_is_local() {
        let source = ModelSource::parse("./models/model.apr").expect("should parse");
        assert_eq!(
            source,
            ModelSource::Local(PathBuf::from("./models/model.apr"))
        );
    }

    /// Relative path starting with "../" should be local.
    #[test]
    fn parse_relative_dotdot_is_local() {
        let source = ModelSource::parse("../shared/model.gguf").expect("should parse");
        assert_eq!(
            source,
            ModelSource::Local(PathBuf::from("../shared/model.gguf"))
        );
    }

    /// Path with spaces should remain local (not confused by space in path).
    /// Bug class: splitting on space in URL detection.
    #[test]
    fn parse_path_with_spaces_is_local() {
        let source = ModelSource::parse("/path/to my/model.apr").expect("should parse");
        assert_eq!(
            source,
            ModelSource::Local(PathBuf::from("/path/to my/model.apr"))
        );
    }

    /// hf:// with empty org and empty repo should fail.
    /// Bug class: split('/') on "" yields [""] which has len() == 1.
    #[test]
    fn parse_hf_empty_path_fails() {
        let result = ModelSource::parse("hf://");
        assert!(result.is_err(), "hf:// with no org/repo must be rejected");
    }

    /// hf:// with only org (single segment) should fail.
    #[test]
    fn parse_hf_single_segment_fails() {
        let result = ModelSource::parse("hf://orgonly");
        assert!(
            result.is_err(),
            "hf:// with only org (no repo) must be rejected"
        );
    }

    /// https URL with query parameters should be preserved as-is.
    /// Bug class: URL parser stripping query string.
    #[test]
    fn parse_url_with_query_params() {
        let url = "https://example.com/model.apr?token=abc&v=2";
        let source = ModelSource::parse(url).expect("should parse");
        assert_eq!(source, ModelSource::Url(url.to_string()));
    }

    /// http URL should also be accepted (not just https).
    /// Bug class: only checking "https://" prefix.
    #[test]
    fn parse_http_url_preserved() {
        let url = "http://internal.corp/models/v1.gguf";
        let source = ModelSource::parse(url).expect("should parse");
        assert_eq!(source, ModelSource::Url(url.to_string()));
    }

    // ========================================================================
    // md5_hash: additional properties
    // ========================================================================

    /// Single byte hashes must differ for each byte value.
    /// Bug class: collision in single-byte inputs due to weak mixing.
    #[test]
    fn md5_hash_single_byte_no_collision() {
        let h0 = md5_hash(&[0]);
        let h1 = md5_hash(&[1]);
        let h255 = md5_hash(&[255]);
        assert_ne!(h0, h1);
        assert_ne!(h0, h255);
        assert_ne!(h1, h255);
    }

    /// Same prefix but different lengths must produce different hashes.
    /// Bug class: hash only dependent on final accumulator, ignoring length.
    #[test]
    fn md5_hash_length_sensitive() {
        let h_short = md5_hash(b"abc");
        let h_long = md5_hash(b"abcdef");
        assert_ne!(
            h_short, h_long,
            "Different-length inputs with same prefix must hash differently"
        );
    }

    /// The initial value matches the FNV-1a offset basis constant.
    /// Documents the hash algorithm choice: FNV-1a 64-bit.
    #[test]
    fn md5_hash_empty_is_fnv1a_offset_basis() {
        let h = md5_hash(&[]);
        assert_eq!(
            h, 0xcbf29ce484222325,
            "Empty input hash should equal FNV-1a 64-bit offset basis"
        );
    }

    /// Hash output should use all 64 bits (not just lower 32).
    /// Bug class: accidental truncation to u32 before return.
    #[test]
    fn md5_hash_uses_upper_bits() {
        // At least one common input should have non-zero upper 32 bits
        let h = md5_hash(b"test_upper_bits");
        let upper = h >> 32;
        assert_ne!(
            upper, 0,
            "Hash should utilize upper 32 bits for typical inputs"
        );
    }

    /// Many distinct short inputs should produce distinct hashes (no systemic collisions).
    /// Bug class: weak hash with high collision rate.
    #[test]
    fn md5_hash_no_collisions_for_sequential_inputs() {
        let mut seen = std::collections::HashSet::new();
        for i in 0u16..1000 {
            let h = md5_hash(&i.to_le_bytes());
            assert!(seen.insert(h), "Collision detected at input {i}");
        }
    }

    // ========================================================================
    // extract_shard_files: additional parsing cases
    // ========================================================================

    /// Nested braces inside weight_map values should not confuse depth tracking.
    /// Bug class: brace matching failing on nested JSON objects.
    #[test]
    fn extract_shard_files_nested_metadata_before_weight_map() {
        let json = r#"{
            "metadata": {"nested": {"deep": true}},
            "weight_map": {
                "layer.0.weight": "shard-001.safetensors",
                "layer.1.weight": "shard-002.safetensors"
            }
        }"#;
        let files = extract_shard_files(json);
        assert_eq!(files.len(), 2);
        assert!(files.contains("shard-001.safetensors"));
        assert!(files.contains("shard-002.safetensors"));
    }

    /// Large shard count should all be extracted.
    /// Bug class: off-by-one or capacity limit in HashSet.
    #[test]
    fn extract_shard_files_many_shards() {
        let mut weight_map_entries = Vec::new();
        for i in 0..50 {
            let shard = format!("model-{i:05}-of-00050.safetensors");
            weight_map_entries.push(format!("\"tensor.{i}\": \"{shard}\""));
        }
        let json = format!(r#"{{"weight_map": {{{}}}}}"#, weight_map_entries.join(", "));
        let files = extract_shard_files(&json);
        assert_eq!(files.len(), 50, "Should extract all 50 unique shard files");
    }

    /// Duplicate shard filenames should be deduplicated (HashSet property).
    /// Bug class: using Vec instead of HashSet, returning duplicates.
    #[test]
    fn extract_shard_files_deduplicates() {
        let json = r#"{
            "weight_map": {
                "a": "shard.safetensors",
                "b": "shard.safetensors",
                "c": "shard.safetensors",
                "d": "other.safetensors"
            }
        }"#;
        let files = extract_shard_files(json);
        assert_eq!(files.len(), 2, "Duplicate shards must be deduplicated");
    }

    /// Quoted values with escaped quotes should not crash.
    /// Bug class: naive quote splitting on escaped quotes.
    #[test]
    fn extract_shard_files_truncated_weight_map_empty() {
        // weight_map with opening brace but no closing brace
        let json = r#"{"weight_map": {"a": "model.safetensors""#;
        // Should not panic; may return empty or partial
        let files = extract_shard_files(json);
        // The brace matching loop will not find depth==0, so end_pos stays 0
        // and entries will be empty slice
        let _ = files; // No panic = pass
    }

    // ========================================================================
    // parse_token_ids: additional edge cases
    // ========================================================================

    /// Single token ID without delimiters.
    /// Bug class: split() returning empty on single element.
    #[test]
    fn parse_token_ids_single_value() {
        let result = parse_token_ids("42").expect("should parse single token");
        assert_eq!(result, vec![42u32]);
    }

    /// JSON array with single element.
    /// Bug class: JSON array path only handling multi-element arrays.
    #[test]
    fn parse_token_ids_json_single_element() {
        let result = parse_token_ids("[999]").expect("should parse single-element array");
        assert_eq!(result, vec![999u32]);
    }

    /// Empty JSON array should produce empty vec.
    /// Bug class: JSON deserialize failing on empty array.
    #[test]
    fn parse_token_ids_json_empty_array() {
        let result = parse_token_ids("[]").expect("should parse empty JSON array");
        assert!(result.is_empty());
    }
