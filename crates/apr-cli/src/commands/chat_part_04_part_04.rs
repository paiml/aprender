
    #[test]
    fn test_clean_chat_response_im_end_alone() {
        let raw = "Some text<|im_end|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Some text");
    }

    #[test]
    fn test_clean_chat_response_endoftext_midstream() {
        let raw = "Hello<|endoftext|> world";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_chat_response_assistant_without_newline() {
        let raw = "<|im_start|>assistantHello";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello");
    }

    #[test]
    fn test_clean_chat_response_multiple_endoftext() {
        let raw = "Text<|endoftext|><|endoftext|>more";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Textmore");
    }

    #[test]
    fn test_clean_chat_response_all_markers_combined() {
        let raw = "<|im_start|>assistant\n<|im_start|>Hello <|endoftext|>world<|im_end|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    // =========================================================================
    // clean_chat_response: whitespace normalization
    // =========================================================================

    #[test]
    fn test_clean_chat_response_many_spaces() {
        let raw = "Hello     world     test";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world test");
    }

    #[test]
    fn test_clean_chat_response_only_whitespace() {
        let raw = "   \t   \n   ";
        let cleaned = clean_chat_response(raw);
        assert!(cleaned.is_empty() || cleaned.trim().is_empty());
    }

    #[test]
    fn test_clean_chat_response_newlines_preserved() {
        let raw = "Line1\nLine2\nLine3";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Line1\nLine2\nLine3");
    }

    #[test]
    fn test_clean_chat_response_leading_newline_trimmed() {
        let raw = "\nHello";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello");
    }

    #[test]
    fn test_clean_chat_response_trailing_newline_trimmed() {
        let raw = "Hello\n";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello");
    }

    // =========================================================================
    // detect_format: additional edge cases
    // =========================================================================

    #[test]
    fn test_detect_format_empty_filename() {
        let path = Path::new("");
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_just_extension_apr() {
        // Path is literally ".apr"
        let path = Path::new(".apr");
        // file_stem is "" for ".apr", extension is "apr" on some platforms
        // but actually Path::new(".apr").extension() returns None on Unix
        // because ".apr" is treated as a hidden file with no extension
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_mixed_case_gguf() {
        let path = Path::new("/models/test.GGUF");
        // Case-sensitive: "GGUF" != "gguf"
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_mixed_case_safetensors() {
        let path = Path::new("/models/test.SafeTensors");
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_with_spaces_in_path() {
        let path = Path::new("/my models/test model.gguf");
        assert_eq!(detect_format(path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_unicode_path() {
        let path = Path::new("/modelos/modelo.apr");
        assert_eq!(detect_format(path), ModelFormat::Apr);
    }

    #[test]
    fn test_detect_format_root_path() {
        let path = Path::new("/model.safetensors");
        assert_eq!(detect_format(path), ModelFormat::SafeTensors);
    }

    #[test]
    fn test_detect_format_current_dir() {
        let path = Path::new("model.apr");
        assert_eq!(detect_format(path), ModelFormat::Apr);
    }

    // =========================================================================
    // detect_format_from_bytes: additional edge cases (inference only)
    // =========================================================================

    #[cfg(feature = "inference")]
    mod inference_edge_cases {
        use super::*;

        #[test]
        fn test_detect_format_from_bytes_exactly_8_bytes() {
            // Exactly 8 bytes, not matching any known magic
            let data = [0u8; 8];
            // header_size = 0, so SafeTensors check fails (header_size > 0)
            assert_eq!(detect_format_from_bytes(&data), ModelFormat::Demo);
        }

        #[test]
        fn test_detect_format_from_bytes_safetensors_header_zero() {
            // SafeTensors with header_size = 0 should NOT match
            let mut data = vec![0u8; 16];
            data[0..8].copy_from_slice(&0u64.to_le_bytes());
            assert_eq!(detect_format_from_bytes(&data), ModelFormat::Demo);
        }

        #[test]
        fn test_detect_format_from_bytes_safetensors_header_huge() {
            // SafeTensors with header_size >= 100_000_000 should NOT match
            let mut data = vec![0u8; 16];
            data[0..8].copy_from_slice(&100_000_000u64.to_le_bytes());
            assert_eq!(detect_format_from_bytes(&data), ModelFormat::Demo);
        }

        #[test]
        fn test_detect_format_from_bytes_safetensors_header_boundary() {
            // SafeTensors with header_size = 99_999_999 should match
            let mut data = vec![0u8; 16];
            data[0..8].copy_from_slice(&99_999_999u64.to_le_bytes());
            assert_eq!(detect_format_from_bytes(&data), ModelFormat::SafeTensors);
        }

        #[test]
        fn test_detect_format_from_bytes_safetensors_header_one() {
            // SafeTensors with header_size = 1 should match
            let mut data = vec![0u8; 16];
            data[0..8].copy_from_slice(&1u64.to_le_bytes());
            assert_eq!(detect_format_from_bytes(&data), ModelFormat::SafeTensors);
        }

        #[test]
        fn test_detect_format_from_bytes_7_bytes() {
            // 7 bytes (less than 8) should return Demo
            let data = b"APR2xxx";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Demo);
        }

        #[test]
        fn test_detect_format_from_bytes_apr_takes_priority_over_safetensors() {
            // "APRN" starts with bytes that could also be a valid u64 header size
            // APR magic should be detected first
            let data = b"APRNxxxxxxxx0000";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Apr);
        }

        #[test]
        fn test_detect_format_from_bytes_gguf_takes_priority_over_safetensors() {
            // "GGUF" starts with bytes that could also be a valid u64 header size
            // GGUF magic should be detected first
            let data = b"GGUFxxxxxxxx0000";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Gguf);
        }

        #[test]
        fn test_detect_format_from_bytes_all_0xff() {
            // All 0xFF bytes: header_size would be u64::MAX which is > 100_000_000
            let data = [0xFFu8; 16];
            assert_eq!(detect_format_from_bytes(&data), ModelFormat::Demo);
        }

        #[test]
        fn test_detect_format_from_bytes_safetensors_typical_sizes() {
            // Typical SafeTensors header sizes
            for size in [256u64, 1024, 8192, 65536, 1_000_000, 50_000_000] {
                let mut data = vec![0u8; 16];
                data[0..8].copy_from_slice(&size.to_le_bytes());
                assert_eq!(
                    detect_format_from_bytes(&data),
                    ModelFormat::SafeTensors,
                    "Expected SafeTensors for header_size={}",
                    size
                );
            }
        }
    }

    // =========================================================================
    // ChatConfig: comprehensive field combination tests
    // =========================================================================

    #[test]
    fn test_chat_config_default_trace_is_false() {
        let config = ChatConfig::default();
        assert!(!config.trace);
    }

    #[test]
    fn test_chat_config_default_trace_output_is_none() {
        let config = ChatConfig::default();
        assert!(config.trace_output.is_none());
    }

    #[test]
    fn test_chat_config_all_fields_set() {
        let config = ChatConfig {
            temperature: 1.5,
            top_p: 0.95,
            max_tokens: 2048,
            system: Some("Expert mode".to_string()),
            inspect: true,
            force_cpu: true,
            trace: true,
            trace_output: Some(PathBuf::from("/tmp/all_fields.json")),
        };
        assert!((config.temperature - 1.5).abs() < f32::EPSILON);
        assert!((config.top_p - 0.95).abs() < f32::EPSILON);
        assert_eq!(config.max_tokens, 2048);
        assert_eq!(config.system.as_deref(), Some("Expert mode"));
        assert!(config.inspect);
        assert!(config.force_cpu);
        assert!(config.trace);
        assert_eq!(
            config.trace_output.as_ref().map(|p| p.to_str().unwrap()),
            Some("/tmp/all_fields.json")
        );
    }

    #[test]
    fn test_chat_config_zero_temperature() {
        // Greedy decoding
        let config = ChatConfig {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(config.temperature, 0.0);
    }

    #[test]
    fn test_chat_config_max_tokens_zero() {
        let config = ChatConfig {
            max_tokens: 0,
            ..Default::default()
        };
        assert_eq!(config.max_tokens, 0);
    }

    #[test]
    fn test_chat_config_max_tokens_large() {
        let config = ChatConfig {
            max_tokens: usize::MAX,
            ..Default::default()
        };
        assert_eq!(config.max_tokens, usize::MAX);
    }

    #[test]
    fn test_chat_config_empty_system_prompt() {
        let config = ChatConfig {
            system: Some(String::new()),
            ..Default::default()
        };
        assert_eq!(config.system.as_deref(), Some(""));
    }

    #[test]
    fn test_chat_config_long_system_prompt() {
        let long_prompt = "a".repeat(10_000);
        let config = ChatConfig {
            system: Some(long_prompt.clone()),
            ..Default::default()
        };
        assert_eq!(config.system.as_deref(), Some(long_prompt.as_str()));
    }

    #[test]
    fn test_chat_config_trace_output_relative_path() {
        let config = ChatConfig {
            trace: true,
            trace_output: Some(PathBuf::from("relative/trace.json")),
            ..Default::default()
        };
        assert!(!config.trace_output.as_ref().unwrap().is_absolute());
    }

    // =========================================================================
    // run() additional error path tests
    // =========================================================================

    #[test]
    fn test_run_nonexistent_path_without_trace() {
        let path = Path::new("/definitely/not/a/real/path/model.apr");
        let result = run(
            path, 0.7, 0.9, 512, None, false, false, false, None, false, None, "info", false,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            CliError::FileNotFound(p) => {
                assert_eq!(p, PathBuf::from("/definitely/not/a/real/path/model.apr"));
            }
            other => panic!("Expected FileNotFound, got: {:?}", other),
        }
    }

    #[test]
    fn test_run_nonexistent_safetensors() {
        let path = Path::new("/no/such/model.safetensors");
        let result = run(
            path, 0.5, 0.8, 256, None, false, false, false, None, false, None, "info", false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_nonexistent_apr() {
        let path = Path::new("/no/such/model.apr");
        let result = run(
            path, 1.0, 1.0, 1024, None, true, true, false, None, false, None, "warn", false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_nonexistent_with_all_trace_options() {
        let path = Path::new("/no/such/model.gguf");
        let steps = vec![
            "tokenize".to_string(),
            "embed".to_string(),
            "attention".to_string(),
            "ffn".to_string(),
            "sample".to_string(),
            "decode".to_string(),
        ];
        let result = run(
            path,
            0.3,
            0.95,
            128,
            Some("System prompt"),
            true,
            false,
            true,
            Some(&steps),
            true,
            Some(PathBuf::from("/tmp/full_trace.json")),
            "debug",
            true,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_nonexistent_no_system_inspect_off() {
        let path = Path::new("/no/model.bin");
        let result = run(
            path, 0.7, 0.9, 512, None, false, false, false, None, false, None, "info", false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_nonexistent_with_empty_trace_steps() {
        let path = Path::new("/no/model.gguf");
        let steps: Vec<String> = vec![];
        let result = run(
            path,
            0.7,
            0.9,
            512,
            None,
            false,
            false,
            true,
            Some(&steps),
            false,
            None,
            "info",
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_nonexistent_trace_without_output() {
        let path = Path::new("/no/model.apr");
        let result = run(
            path, 0.7, 0.9, 512, None, false, false, true,  // trace enabled
            None,  // no trace steps
            false, // not verbose
            None,  // no trace output
            "info", false, // no profile
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_nonexistent_with_profile_only() {
        let path = Path::new("/no/model.gguf");
        let result = run(
            path, 0.7, 0.9, 512, None, false, false,
            true, // trace must be on for profile to print
            None, false, None, "info", true, // profile enabled
        );
        assert!(result.is_err());
    }
