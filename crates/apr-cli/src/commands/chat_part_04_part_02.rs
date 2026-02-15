
    #[test]
    fn test_chat_config_default() {
        let config = ChatConfig::default();
        assert!((config.temperature - 0.7).abs() < 0.01);
        assert!((config.top_p - 0.9).abs() < 0.01);
        assert_eq!(config.max_tokens, 512);
        assert!(config.system.is_none());
        assert!(!config.inspect);
        // F-GPU-134b: Default to GPU (force_cpu = false)
        assert!(
            !config.force_cpu,
            "F-GPU-134b: force_cpu should default to false"
        );
    }

    #[test]
    fn test_chat_config_with_system_prompt() {
        let config = ChatConfig {
            system: Some("You are a helpful assistant.".to_string()),
            ..Default::default()
        };
        assert!(config.system.is_some());
        assert_eq!(
            config.system.as_ref().unwrap(),
            "You are a helpful assistant."
        );
    }

    #[test]
    fn test_chat_config_trace_settings() {
        let config = ChatConfig {
            trace: true,
            trace_output: Some(PathBuf::from("/tmp/trace.json")),
            ..Default::default()
        };
        assert!(config.trace);
        assert_eq!(
            config.trace_output.as_ref().unwrap().to_str().unwrap(),
            "/tmp/trace.json"
        );
    }

    #[test]
    fn test_chat_config_force_cpu() {
        let config = ChatConfig {
            force_cpu: true,
            ..Default::default()
        };
        assert!(config.force_cpu);
    }

    // F-PIPE-166b: Test tokenizer artifact cleaning
    #[test]
    fn test_clean_chat_response_removes_chatml_markers() {
        let raw = "<|im_start|>assistant\nHello world<|im_end|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_chat_response_removes_bpe_artifacts() {
        // Ġ (U+0120) is used in GPT-2/BPE tokenizers for space prefix
        let raw = "HelloĠworld";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_chat_response_normalizes_repeated_punctuation() {
        let raw = "Wow!!!!!";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Wow!!!");

        let raw2 = "Really??????";
        let cleaned2 = clean_chat_response(raw2);
        assert_eq!(cleaned2, "Really???");
    }

    #[test]
    fn test_clean_chat_response_normalizes_spaces() {
        let raw = "Hello   world";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_chat_response_trims_whitespace() {
        let raw = "  Hello world  ";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_chat_response_removes_endoftext() {
        let raw = "Hello<|endoftext|>world";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Helloworld");
    }

    #[test]
    fn test_clean_chat_response_newline_artifact() {
        // Ċ (U+010A) represents newline in some BPE tokenizers
        let raw = "HelloĊworld";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello\nworld");
    }

    #[test]
    fn test_clean_chat_response_strips_new_turn() {
        // If response has new question after first line, return first line only
        let raw = "4\nSuggest a fun way to learn Rust";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "4");
    }

    #[test]
    fn test_clean_chat_response_keeps_multiline_answer() {
        // Normal multiline response should be preserved
        let raw = "Here is the answer:\nLine 1\nLine 2";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Here is the answer:\nLine 1\nLine 2");
    }

    #[test]
    fn test_clean_chat_response_empty_string() {
        let raw = "";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_chat_response_only_markers() {
        let raw = "<|im_start|>assistant<|im_end|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn test_clean_chat_response_human_prompt_cutoff() {
        // If "Human:" appears after first line, cut it off
        let raw = "Yes\nHuman: What else?";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Yes");
    }

    // =========================================================================
    // ModelFormat tests
    // =========================================================================

    #[test]
    fn test_model_format_equality() {
        assert_eq!(ModelFormat::Apr, ModelFormat::Apr);
        assert_eq!(ModelFormat::Gguf, ModelFormat::Gguf);
        assert_eq!(ModelFormat::SafeTensors, ModelFormat::SafeTensors);
        assert_eq!(ModelFormat::Demo, ModelFormat::Demo);
    }

    #[test]
    fn test_model_format_inequality() {
        assert_ne!(ModelFormat::Apr, ModelFormat::Gguf);
        assert_ne!(ModelFormat::Gguf, ModelFormat::SafeTensors);
        assert_ne!(ModelFormat::SafeTensors, ModelFormat::Demo);
    }

    #[test]
    fn test_model_format_debug() {
        assert_eq!(format!("{:?}", ModelFormat::Apr), "Apr");
        assert_eq!(format!("{:?}", ModelFormat::Gguf), "Gguf");
        assert_eq!(format!("{:?}", ModelFormat::SafeTensors), "SafeTensors");
        assert_eq!(format!("{:?}", ModelFormat::Demo), "Demo");
    }

    #[test]
    fn test_model_format_clone() {
        let format = ModelFormat::Apr;
        let cloned = format;
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_model_format_copy() {
        let format = ModelFormat::Gguf;
        let copied: ModelFormat = format;
        assert_eq!(format, copied);
    }

    // =========================================================================
    // detect_format tests (Y14: format-agnostic)
    // =========================================================================

    #[test]
    fn test_detect_format_apr() {
        let path = Path::new("/models/test.apr");
        assert_eq!(detect_format(path), ModelFormat::Apr);
    }

    #[test]
    fn test_detect_format_gguf() {
        let path = Path::new("/models/test.gguf");
        assert_eq!(detect_format(path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_safetensors() {
        let path = Path::new("/models/model.safetensors");
        assert_eq!(detect_format(path), ModelFormat::SafeTensors);
    }

    #[test]
    fn test_detect_format_unknown_fallback_to_demo() {
        let path = Path::new("/models/test.bin");
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_no_extension() {
        let path = Path::new("/models/modelfile");
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_case_sensitive() {
        // Extensions are case-sensitive in Rust
        let path = Path::new("/models/test.APR");
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_nested_path() {
        let path = Path::new("/home/user/.cache/models/qwen2-0.5b.gguf");
        assert_eq!(detect_format(path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_relative_path() {
        let path = Path::new("./models/model.safetensors");
        assert_eq!(detect_format(path), ModelFormat::SafeTensors);
    }

    // =========================================================================
    // detect_format_from_bytes tests (inference feature only)
    // =========================================================================

    #[cfg(feature = "inference")]
    mod inference_tests {
        use super::*;

        #[test]
        fn test_detect_format_from_bytes_apr_v1() {
            let data = b"APRNxxxx00000000";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Apr);
        }

        #[test]
        fn test_detect_format_from_bytes_apr_v2() {
            let data = b"APR2xxxx00000000";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Apr);
        }

        #[test]
        fn test_detect_format_from_bytes_apr_null() {
            let data = b"APR\0xxxx00000000";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Apr);
        }

        #[test]
        fn test_detect_format_from_bytes_gguf() {
            let data = b"GGUFxxxx00000000";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Gguf);
        }

        #[test]
        fn test_detect_format_from_bytes_safetensors() {
            // SafeTensors: first 8 bytes are little-endian header size
            // A typical header size might be ~1000-10000 bytes
            let mut data = vec![0u8; 16];
            // Write 1000 as little-endian u64 (reasonable header size)
            data[0..8].copy_from_slice(&1000u64.to_le_bytes());
            assert_eq!(detect_format_from_bytes(&data), ModelFormat::SafeTensors);
        }

        #[test]
        fn test_detect_format_from_bytes_too_short() {
            let data = b"APR";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Demo);
        }

        #[test]
        fn test_detect_format_from_bytes_empty() {
            let data: &[u8] = &[];
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Demo);
        }

        #[test]
        fn test_detect_format_from_bytes_unknown_magic() {
            let data = b"UNKN0000\x00\x00\x00\x00\x00\x00\x00\x00";
            assert_eq!(detect_format_from_bytes(data), ModelFormat::Demo);
        }
    }

    // =========================================================================
    // run() error case tests
    // =========================================================================

    #[test]
    fn test_run_file_not_found() {
        let path = Path::new("/nonexistent/model.gguf");
        let result = run(
            path, 0.7, 0.9, 512, None, false, false, false, None, false, None, "info", false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(p)) => {
                assert_eq!(p, PathBuf::from("/nonexistent/model.gguf"));
            }
            other => panic!("Expected FileNotFound error, got {:?}", other),
        }
    }

    #[test]
    fn test_run_with_trace_config() {
        let path = Path::new("/nonexistent/model.gguf");
        // This should still fail with FileNotFound, but trace config is set
        let result = run(
            path,
            0.7,
            0.9,
            512,
            Some("You are helpful"),
            true,
            true,
            true,
            Some(&["tokenize".to_string(), "sample".to_string()]),
            true,
            Some(PathBuf::from("/tmp/trace.json")),
            "debug",
            true,
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // CommandResult tests
    // =========================================================================

    #[test]
    fn test_command_result_variants() {
        // Ensure CommandResult variants exist and can be matched
        let continue_result = CommandResult::Continue;
        let quit_result = CommandResult::Quit;

        match continue_result {
            CommandResult::Continue => {}
            CommandResult::Quit => panic!("Expected Continue"),
        }

        match quit_result {
            CommandResult::Quit => {}
            CommandResult::Continue => panic!("Expected Quit"),
        }
    }

    // =========================================================================
    // Edge case tests for clean_chat_response
    // =========================================================================

    #[test]
    fn test_clean_chat_response_mixed_markers() {
        let raw = "<|im_start|>assistant\n<|im_start|>Hello<|im_end|><|endoftext|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello");
    }

    #[test]
    fn test_clean_chat_response_repeated_dots() {
        let raw = "Hmm........ let me think";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hmm... let me think");
    }

    #[test]
    fn test_clean_chat_response_unicode_preserved() {
        let raw = "こんにちは世界";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "こんにちは世界");
    }

    #[test]
    fn test_clean_chat_response_emoji_preserved() {
        let raw = "Hello 👋 World 🌍";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello 👋 World 🌍");
    }

    #[test]
    fn test_clean_chat_response_code_block() {
        let raw = "Here is code:\n```rust\nfn main() {}\n```";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Here is code:\n```rust\nfn main() {}\n```");
    }

    #[test]
    fn test_clean_chat_response_numbered_list() {
        let raw = "Steps:\n1. First\n2. Second\n3. Third";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Steps:\n1. First\n2. Second\n3. Third");
    }

    // =========================================================================
    // format_params tests (non-inference only)
    // =========================================================================

    #[cfg(not(feature = "inference"))]
    #[test]
    fn test_format_params_billions() {
        assert_eq!(format_params(7_000_000_000), "7.0B");
        assert_eq!(format_params(1_500_000_000), "1.5B");
    }

    #[cfg(not(feature = "inference"))]
    #[test]
    fn test_format_params_millions() {
        assert_eq!(format_params(500_000_000), "500.0M");
        assert_eq!(format_params(7_000_000), "7.0M");
        assert_eq!(format_params(1_500_000), "1.5M");
    }

    #[cfg(not(feature = "inference"))]
    #[test]
    fn test_format_params_thousands() {
        assert_eq!(format_params(500_000), "500.0K");
        assert_eq!(format_params(7_000), "7.0K");
        assert_eq!(format_params(1_500), "1.5K");
    }

    #[cfg(not(feature = "inference"))]
    #[test]
    fn test_format_params_small() {
        assert_eq!(format_params(999), "999");
        assert_eq!(format_params(100), "100");
        assert_eq!(format_params(1), "1");
        assert_eq!(format_params(0), "0");
    }
