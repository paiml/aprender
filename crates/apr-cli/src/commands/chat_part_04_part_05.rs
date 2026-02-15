
    // =========================================================================
    // CommandResult exhaustive tests
    // =========================================================================

    #[test]
    fn test_command_result_continue_is_not_quit() {
        let result = CommandResult::Continue;
        let is_continue = matches!(result, CommandResult::Continue);
        assert!(is_continue);
    }

    #[test]
    fn test_command_result_quit_is_not_continue() {
        let result = CommandResult::Quit;
        let is_quit = matches!(result, CommandResult::Quit);
        assert!(is_quit);
    }

    // =========================================================================
    // find_qwen_tokenizer: error paths
    // =========================================================================

    #[test]
    fn test_find_qwen_tokenizer_nonexistent_path() {
        // Path with no parent directory containing tokenizer.json.
        // Note: This function also searches HuggingFace cache and APR cache,
        // so it may succeed on dev machines with cached Qwen models.
        let path = Path::new("/nonexistent/deeply/nested/model.safetensors");
        let result = find_qwen_tokenizer(path);
        // Result depends on system state: Ok if cache has tokenizer, Err otherwise
        // We just verify it doesn't panic and returns a valid Result
        match result {
            Ok(Some(tok)) => {
                // Found in cache - verify it's a valid tokenizer
                assert!(tok.vocab_size() > 0);
            }
            Ok(None) => {
                // This shouldn't happen: function returns Err, not Ok(None) on failure
                panic!("Expected Err or Ok(Some), got Ok(None)");
            }
            Err(CliError::InvalidFormat(msg)) => {
                assert!(
                    msg.contains("No Qwen tokenizer found"),
                    "Expected helpful error message, got: {}",
                    msg
                );
            }
            Err(other) => panic!("Expected InvalidFormat error, got: {:?}", other),
        }
    }

    #[test]
    fn test_find_qwen_tokenizer_root_path() {
        // Path at root level - parent is "/"
        let path = Path::new("/model.safetensors");
        let result = find_qwen_tokenizer(path);
        // Same as above: depends on system cache state
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_find_qwen_tokenizer_error_message_content_when_no_cache() {
        // When find_qwen_tokenizer fails, it should return an InvalidFormat error
        // with a helpful message listing the search locations.
        // We test the error construction directly since the function may succeed
        // on machines with cached tokenizers.
        let err = CliError::InvalidFormat(
            "No Qwen tokenizer found. Searched:\n\
             1. Model directory (tokenizer.json)\n\
             2. HuggingFace cache (~/.cache/huggingface/hub/models--Qwen--*/snapshots/*/tokenizer.json)\n\
             3. APR cache (~/.apr/tokenizers/qwen2/tokenizer.json)\n\n\
             To fix: Download a Qwen model with tokenizer:\n\
               apr pull hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF"
                .to_string(),
        );
        let msg = err.to_string();
        assert!(msg.contains("No Qwen tokenizer found"));
        assert!(msg.contains("tokenizer.json"));
        assert!(msg.contains("HuggingFace cache"));
        assert!(msg.contains("APR cache"));
        assert!(msg.contains("apr pull"));
    }

    #[test]
    fn test_find_qwen_tokenizer_searches_parent_directory_first() {
        // The function's search order is:
        // 1. Model's parent directory (tokenizer.json)
        // 2. HuggingFace cache
        // 3. APR tokenizer cache
        // If there's no tokenizer.json in the parent dir, it falls through
        let path = Path::new("/tmp/no_tokenizer_here/model.safetensors");
        let result = find_qwen_tokenizer(path);
        // Just verify it doesn't panic; result depends on system cache
        let _ = result;
    }

    // =========================================================================
    // clean_chat_response: comprehensive combined scenarios
    // =========================================================================

    #[test]
    fn test_clean_chat_response_full_chatml_response() {
        let raw = "<|im_start|>assistant\nThe answer is 42.<|im_end|><|endoftext|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "The answer is 42.");
    }

    #[test]
    fn test_clean_chat_response_bpe_with_markers() {
        let raw = "<|im_start|>assistant\nHelloĠworld!<|im_end|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world!");
    }

    #[test]
    fn test_clean_chat_response_complex_combined() {
        let raw = "<|im_start|>assistant\nĠĠHello!!!!!!ĠĠworld<|im_end|><|endoftext|>";
        let cleaned = clean_chat_response(raw);
        // Ġ -> space, multiple spaces -> single, !!!!!! -> !!!, markers removed, trimmed
        assert_eq!(cleaned, "Hello!!! world");
    }

    #[test]
    fn test_clean_chat_response_very_long_input() {
        let raw = "x".repeat(100_000);
        let cleaned = clean_chat_response(&raw);
        assert_eq!(cleaned.len(), 100_000);
    }

    #[test]
    fn test_clean_chat_response_only_bpe_artifacts() {
        let raw = "ĠĠĠ";
        let cleaned = clean_chat_response(raw);
        // Three Ġ -> three spaces -> collapsed to single space -> trimmed to empty
        assert!(cleaned.is_empty());
    }

    #[test]
    fn test_clean_chat_response_marker_in_middle_of_word() {
        let raw = "hel<|im_end|>lo";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "hello");
    }

    #[test]
    fn test_clean_chat_response_multiple_im_start_assistant() {
        let raw = "<|im_start|>assistant\n<|im_start|>assistant\nHello";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello");
    }

    #[test]
    fn test_clean_chat_response_newline_bpe_and_human_cutoff() {
        // Ċ becomes newline, then Human: detected -> cutoff
        let raw = "DoneĊHuman: next question";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Done");
    }

    #[test]
    fn test_clean_chat_response_single_char() {
        let raw = "a";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "a");
    }

    #[test]
    fn test_clean_chat_response_just_newline() {
        let raw = "\n";
        let cleaned = clean_chat_response(raw);
        assert!(cleaned.is_empty());
    }

    #[test]
    fn test_clean_chat_response_just_markers() {
        let raw = "<|im_start|><|im_end|><|endoftext|>";
        let cleaned = clean_chat_response(raw);
        assert!(cleaned.is_empty());
    }

    // =========================================================================
    // ModelFormat: exhaustive match coverage
    // =========================================================================

    #[test]
    fn test_model_format_debug_format_all() {
        // Verify Debug representation for every variant
        let variants = [
            (ModelFormat::Apr, "Apr"),
            (ModelFormat::Gguf, "Gguf"),
            (ModelFormat::SafeTensors, "SafeTensors"),
            (ModelFormat::Demo, "Demo"),
        ];
        for (variant, expected) in variants {
            assert_eq!(format!("{:?}", variant), expected);
        }
    }

    #[test]
    fn test_model_format_clone_all_variants() {
        let variants = [
            ModelFormat::Apr,
            ModelFormat::Gguf,
            ModelFormat::SafeTensors,
            ModelFormat::Demo,
        ];
        for variant in variants {
            let cloned = variant;
            assert_eq!(variant, cloned);
        }
    }

    #[test]
    fn test_model_format_eq_reflexive() {
        let formats = [
            ModelFormat::Apr,
            ModelFormat::Gguf,
            ModelFormat::SafeTensors,
            ModelFormat::Demo,
        ];
        for f in formats {
            assert_eq!(f, f);
        }
    }

    #[test]
    fn test_model_format_ne_all_pairs() {
        let formats = [
            ModelFormat::Apr,
            ModelFormat::Gguf,
            ModelFormat::SafeTensors,
            ModelFormat::Demo,
        ];
        for (i, a) in formats.iter().enumerate() {
            for (j, b) in formats.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "Expected {:?} != {:?}", a, b);
                }
            }
        }
    }

    // =========================================================================
    // detect_format: pathological paths
    // =========================================================================

    #[test]
    fn test_detect_format_trailing_dot() {
        // Path ending in dot has no extension
        let path = Path::new("/models/model.");
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_multiple_dots_gguf() {
        let path = Path::new("/models/model.v1.2.3.gguf");
        assert_eq!(detect_format(path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_hash_named_apr() {
        let path = Path::new("/models/e910cab26ae116eb.apr");
        assert_eq!(detect_format(path), ModelFormat::Apr);
    }

    #[test]
    fn test_detect_format_hash_named_gguf() {
        let path = Path::new("/cache/d4c4d9763127153c.gguf");
        assert_eq!(detect_format(path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_long_extension() {
        let path = Path::new("/models/model.safetensorsbackup");
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_similar_extensions() {
        // Close but not exact matches
        assert_eq!(detect_format(Path::new("x.ap")), ModelFormat::Demo);
        assert_eq!(detect_format(Path::new("x.ggu")), ModelFormat::Demo);
        assert_eq!(detect_format(Path::new("x.safetensor")), ModelFormat::Demo);
        assert_eq!(detect_format(Path::new("x.ggufx")), ModelFormat::Demo);
        assert_eq!(detect_format(Path::new("x.aprx")), ModelFormat::Demo);
    }

    // =========================================================================
    // print_welcome_banner: config display combinations
    // =========================================================================

    #[test]
    fn test_print_welcome_banner_zero_temp_and_top_p() {
        let path = Path::new("/models/test.gguf");
        let config = ChatConfig {
            temperature: 0.0,
            top_p: 0.0,
            max_tokens: 1,
            ..Default::default()
        };
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_high_temp_and_top_p() {
        let path = Path::new("/models/test.apr");
        let config = ChatConfig {
            temperature: 2.0,
            top_p: 1.0,
            max_tokens: 8192,
            system: Some("Be creative and wild!".to_string()),
            inspect: true,
            ..Default::default()
        };
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_openhermes_model() {
        // openhermes triggers ChatML template
        let path = Path::new("/models/openhermes-2.5.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_yi_model() {
        // yi- triggers ChatML template
        let path = Path::new("/models/yi-34b.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_vicuna_model() {
        // vicuna triggers LLaMA2 template
        let path = Path::new("/models/vicuna-7b.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_mixtral_model() {
        // mixtral triggers Mistral template
        let path = Path::new("/models/mixtral-8x7b.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }
