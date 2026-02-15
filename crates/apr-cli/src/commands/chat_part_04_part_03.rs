
    // =========================================================================
    // Additional ChatConfig tests
    // =========================================================================

    #[test]
    fn test_chat_config_extreme_values() {
        let config = ChatConfig {
            temperature: 0.0,
            top_p: 0.0,
            max_tokens: 1,
            system: None,
            inspect: false,
            force_cpu: false,
            trace: false,
            trace_output: None,
        };
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_p, 0.0);
        assert_eq!(config.max_tokens, 1);
    }

    #[test]
    fn test_chat_config_high_temp() {
        let config = ChatConfig {
            temperature: 2.0,
            top_p: 1.0,
            max_tokens: 4096,
            system: Some("Creative mode".to_string()),
            inspect: true,
            force_cpu: true,
            trace: true,
            trace_output: Some(PathBuf::from("/tmp/creative_trace.json")),
        };
        assert_eq!(config.temperature, 2.0);
        assert_eq!(config.max_tokens, 4096);
    }

    #[test]
    fn test_chat_config_inspect_mode() {
        let config = ChatConfig {
            inspect: true,
            ..Default::default()
        };
        assert!(config.inspect);
    }

    // =========================================================================
    // Additional clean_chat_response edge cases
    // =========================================================================

    #[test]
    fn test_clean_chat_response_deeply_nested_markers() {
        let raw = "<|im_start|><|im_start|>assistant\nTest<|im_end|><|im_end|>";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Test");
    }

    #[test]
    fn test_clean_chat_response_escaped_newlines() {
        let raw = "Line1\\nLine2";
        let cleaned = clean_chat_response(raw);
        // Should preserve escaped newlines
        assert!(cleaned.contains("\\n") || cleaned.contains('\n'));
    }

    #[test]
    fn test_clean_chat_response_tabs() {
        let raw = "Column1\tColumn2";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Column1\tColumn2");
    }

    #[test]
    fn test_clean_chat_response_markdown_headers() {
        let raw = "# Header 1\n## Header 2\n### Header 3";
        let cleaned = clean_chat_response(raw);
        assert!(cleaned.contains("# Header 1"));
        assert!(cleaned.contains("## Header 2"));
    }

    #[test]
    fn test_clean_chat_response_math_expression() {
        let raw = "E = mc²";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "E = mc²");
    }

    #[test]
    fn test_clean_chat_response_url() {
        let raw = "Visit https://example.com for more info";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Visit https://example.com for more info");
    }

    // =========================================================================
    // ModelFormat additional tests
    // =========================================================================

    #[test]
    fn test_model_format_all_variants() {
        // Ensure all variants exist
        let _apr = ModelFormat::Apr;
        let _gguf = ModelFormat::Gguf;
        let _st = ModelFormat::SafeTensors;
        let _demo = ModelFormat::Demo;
    }

    #[test]
    fn test_detect_format_double_extension() {
        let path = Path::new("/models/model.tar.gz");
        // Should detect based on last extension
        assert_eq!(detect_format(path), ModelFormat::Demo);
    }

    #[test]
    fn test_detect_format_hidden_file() {
        let path = Path::new("/models/.hidden.gguf");
        assert_eq!(detect_format(path), ModelFormat::Gguf);
    }

    #[test]
    fn test_detect_format_dot_in_name() {
        let path = Path::new("/models/model.v2.0.safetensors");
        assert_eq!(detect_format(path), ModelFormat::SafeTensors);
    }

    // =========================================================================
    // print_welcome_banner smoke tests (all format branches + config combos)
    // =========================================================================

    #[test]
    fn test_print_welcome_banner_apr_format() {
        let path = Path::new("/models/qwen2-instruct.apr");
        let config = ChatConfig::default();
        // Should not panic; exercises ModelFormat::Apr branch
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_gguf_format() {
        let path = Path::new("/models/qwen2-instruct.gguf");
        let config = ChatConfig::default();
        // Should not panic; exercises ModelFormat::Gguf branch
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_safetensors_format() {
        let path = Path::new("/models/model.safetensors");
        let config = ChatConfig::default();
        // Should not panic; exercises ModelFormat::SafeTensors branch
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_demo_format() {
        let path = Path::new("/models/model.bin");
        let config = ChatConfig::default();
        // Should not panic; exercises ModelFormat::Demo branch
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_with_system_prompt() {
        let path = Path::new("/models/test.apr");
        let config = ChatConfig {
            system: Some("You are a helpful assistant.".to_string()),
            ..Default::default()
        };
        // Exercises the system prompt display branch
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_with_inspect_mode() {
        let path = Path::new("/models/test.gguf");
        let config = ChatConfig {
            inspect: true,
            ..Default::default()
        };
        // Exercises the inspect mode display branch
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_with_system_and_inspect() {
        let path = Path::new("/models/model.safetensors");
        let config = ChatConfig {
            system: Some("Be concise".to_string()),
            inspect: true,
            ..Default::default()
        };
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_template_chatml() {
        // "qwen" in filename triggers ChatML template detection
        let path = Path::new("/models/qwen2-0.5b.apr");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_template_llama2() {
        // "llama" in filename triggers LLaMA2 template detection
        let path = Path::new("/models/llama-2-7b.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_template_mistral() {
        // "mistral" in filename triggers Mistral template detection
        let path = Path::new("/models/mistral-7b.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_template_phi() {
        // "phi-" in filename triggers Phi template detection
        let path = Path::new("/models/phi-3.safetensors");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_template_alpaca() {
        // "alpaca" in filename triggers Alpaca template detection
        let path = Path::new("/models/alpaca-7b.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_template_raw_fallback() {
        // Unknown model name falls back to Raw template
        let path = Path::new("/models/unknown-model.gguf");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_no_extension() {
        // No extension -> Demo format
        let path = Path::new("/models/modelfile");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    #[test]
    fn test_print_welcome_banner_no_stem() {
        // File with no stem (just extension)
        let path = Path::new("/models/.apr");
        let config = ChatConfig::default();
        print_welcome_banner(path, &config);
    }

    // =========================================================================
    // clean_chat_response: new-turn detection branches
    // =========================================================================

    #[test]
    fn test_clean_chat_response_what_question_cutoff() {
        let raw = "42\nWhat is the meaning of life?";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "42");
    }

    #[test]
    fn test_clean_chat_response_how_question_cutoff() {
        let raw = "Done.\nHow do you feel about that?";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Done.");
    }

    #[test]
    fn test_clean_chat_response_why_question_cutoff() {
        let raw = "Because reasons.\nWhy did the chicken cross the road?";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Because reasons.");
    }

    #[test]
    fn test_clean_chat_response_can_question_cutoff() {
        let raw = "Yes.\nCan you elaborate on that?";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Yes.");
    }

    #[test]
    fn test_clean_chat_response_im_start_in_rest_not_cutoff_after_stripping() {
        // Note: <|im_start|> markers are stripped BEFORE new-turn detection,
        // so the rest becomes "user\nNext question?" which doesn't match
        // any cutoff pattern (Suggest/What/How/Why/Can/Human:/<|im_start|>)
        let raw = "Answer here.\n<|im_start|>user\nNext question?";
        let cleaned = clean_chat_response(raw);
        // After marker stripping: "Answer here.\nuser\nNext question?"
        // "user" doesn't match cutoff patterns, so full text is preserved
        assert_eq!(cleaned, "Answer here.\nuser\nNext question?");
    }

    #[test]
    fn test_clean_chat_response_raw_im_start_text_in_rest() {
        // If the raw text literally has "<|im_start|>" that WASN'T stripped
        // (e.g., doubled markers), test the cutoff pattern
        // Actually, all <|im_start|> are stripped first, so this tests
        // that the check works on the already-cleaned text
        let raw = "Answer.\nWhat is next?";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Answer.");
    }

    #[test]
    fn test_clean_chat_response_suggest_cutoff() {
        // Already covered above, but this tests with different first-line content
        let raw = "Rust is great!\nSuggest some alternatives";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Rust is great!");
    }

    #[test]
    fn test_clean_chat_response_no_cutoff_normal_continuation() {
        // Lines that start with lowercase should NOT be cut off
        let raw = "First line\nsecond line continues";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "First line\nsecond line continues");
    }

    #[test]
    fn test_clean_chat_response_no_cutoff_numbered_continuation() {
        let raw = "Steps:\n1. Do this\n2. Do that";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Steps:\n1. Do this\n2. Do that");
    }

    // =========================================================================
    // clean_chat_response: BPE artifact edge cases
    // =========================================================================

    #[test]
    fn test_clean_chat_response_multiple_bpe_spaces() {
        // Multiple Ġ should collapse to single space after cleaning
        let raw = "HelloĠĠworld";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_chat_response_bpe_newline_u010a() {
        // U+010A character (Ċ) is newline in some BPE tokenizers
        let raw = "LineAĊLineB";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "LineA\nLineB");
    }

    #[test]
    fn test_clean_chat_response_both_bpe_artifacts() {
        // Mix of Ġ and Ċ
        let raw = "HelloĠworldĊnewĠline";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello world\nnew line");
    }

    #[test]
    fn test_clean_chat_response_literal_g_with_dot() {
        // Literal "Ġ" string (not the Unicode character) should also be replaced
        let raw = "Hello\u{0120}there";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hello there");
    }

    // =========================================================================
    // clean_chat_response: punctuation normalization edge cases
    // =========================================================================

    #[test]
    fn test_clean_chat_response_exactly_three_exclamation() {
        let raw = "Wow!!!";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Wow!!!");
    }

    #[test]
    fn test_clean_chat_response_exactly_two_exclamation() {
        let raw = "Wow!!";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Wow!!");
    }

    #[test]
    fn test_clean_chat_response_single_exclamation() {
        let raw = "Wow!";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Wow!");
    }

    #[test]
    fn test_clean_chat_response_mixed_punctuation_not_collapsed() {
        // Mixed punctuation (different chars) should NOT be collapsed
        let raw = "Really!?!?";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Really!?!?");
    }

    #[test]
    fn test_clean_chat_response_dots_exactly_three() {
        let raw = "Hmm...";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Hmm...");
    }

    #[test]
    fn test_clean_chat_response_dots_many() {
        let raw = "Wait..........";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Wait...");
    }

    #[test]
    fn test_clean_chat_response_questions_exactly_three() {
        let raw = "What???";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "What???");
    }

    #[test]
    fn test_clean_chat_response_questions_many() {
        let raw = "What????????";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "What???");
    }

    // =========================================================================
    // clean_chat_response: ChatML marker variations
    // =========================================================================

    #[test]
    fn test_clean_chat_response_im_start_alone() {
        let raw = "<|im_start|>Some text";
        let cleaned = clean_chat_response(raw);
        assert_eq!(cleaned, "Some text");
    }
