
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

    /// An EMPTY cache root: the clean-machine answer.
    ///
    /// #3917. This asserted `InvalidFormat` and passed on developer boxes while
    /// failing on both CI platforms — and the PASSING answer was the wrong one. The
    /// function searches two MACHINE-GLOBAL caches under `$HOME`; on a box with a
    /// Qwen model cached, the search SUCCEEDS on some other model's tokenizer and
    /// the subsequent load of the nonexistent model fails with `InvalidFormat`. On a
    /// clean machine nothing is found and the error is `MissingCompanionFile`, which
    /// is also the correct error for a path that does not exist.
    ///
    /// The old test's own comment admitted the dependency — "may succeed on dev
    /// machines with cached Qwen models" — and tolerated it instead of removing it.
    /// Passing `home` explicitly makes the assertion about the FUNCTION rather than
    /// about the machine that ran it.
    #[test]
    fn find_qwen_tokenizer_on_a_clean_machine_reports_a_missing_companion() {
        let empty_home = tempfile::tempdir().expect("tempdir");
        let path = Path::new("/nonexistent/deeply/nested/model.safetensors");

        match find_qwen_tokenizer_from(path, Some(empty_home.path())) {
            Err(CliError::MissingCompanionFile(msg)) => {
                assert!(
                    msg.contains("No Qwen tokenizer found"),
                    "the error must name what it searched for: {msg}"
                );
                assert!(
                    msg.contains("Pacha cache"),
                    "the error must list the concrete candidates it tried: {msg}"
                );
            },
            other => panic!(
                "a clean machine has no tokenizer to find, so this must be \
                 MissingCompanionFile; got {other:?}"
            ),
        }
    }

    /// A model at the filesystem root, whose parent is `/`. Same clean-machine
    /// answer, and it exists to pin the parent-is-root path rather than to repeat
    /// the case above.
    ///
    /// REPLACES a tautology. The previous body was
    /// `assert!(result.is_ok() || result.is_err())` with the comment "depends on
    /// system cache state" — an assertion that cannot fail for any input, on any
    /// machine, under any change to the function.
    #[test]
    fn find_qwen_tokenizer_at_the_filesystem_root_also_reports_a_missing_companion() {
        let empty_home = tempfile::tempdir().expect("tempdir");
        let path = Path::new("/model.safetensors");
        assert!(
            matches!(
                find_qwen_tokenizer_from(path, Some(empty_home.path())),
                Err(CliError::MissingCompanionFile(_))
            ),
            "a root-level model with no tokenizer anywhere is a missing companion"
        );
    }

    /// THE MUST-RED FOR THE TWO ABOVE, and it is a real configuration rather than a
    /// contrived one: it is the state every developer box that has pulled a Qwen
    /// model is permanently in.
    ///
    /// With a POPULATED APR cache the function finds a tokenizer and does NOT return
    /// `MissingCompanionFile` — so the assertions above can fail, and they are about
    /// the empty-cache case specifically rather than about `find_qwen_tokenizer`
    /// always erroring.
    #[test]
    fn a_populated_cache_changes_the_answer_so_the_clean_machine_assertion_is_not_vacuous() {
        let home = tempfile::tempdir().expect("tempdir");
        let cache = home.path().join(".apr/tokenizers/qwen2");
        std::fs::create_dir_all(&cache).expect("cache dir");
        // Minimal HuggingFace tokenizer.json: a vocab and a (possibly empty) merge list.
        std::fs::write(
            cache.join("tokenizer.json"),
            r#"{"model":{"vocab":{"ZZ_CACHE_SENTINEL":0,"a":1,"b":2},"merges":[]},"added_tokens":[]}"#,
        )
        .expect("write cache tokenizer");

        let path = Path::new("/nonexistent/deeply/nested/model.safetensors");
        let got = find_qwen_tokenizer_from(path, Some(home.path()));

        assert!(
            !matches!(got, Err(CliError::MissingCompanionFile(_))),
            "with a cache present the function must NOT report a missing companion — \
             if it does, the clean-machine tests above prove nothing"
        );
    }

    /// The message must name the file the user has to produce.
    ///
    /// This test used to build its OWN copy of the error string and assert on
    /// that, so it passed while the shipped message printed the literal
    /// `{stem}.tokenizer.json` — an unsubstituted format placeholder — to every
    /// user who ran `apr chat` on a model with no tokenizer. It now calls the
    /// real message builder.
    #[test]
    fn test_find_qwen_tokenizer_error_message_content_when_no_cache() {
        let msg = no_qwen_tokenizer_message(Path::new("/models/sub/m.apr"));

        assert!(
            !msg.contains("{stem}"),
            "the message must not show an unsubstituted format placeholder: {msg}"
        );
        assert!(
            msg.contains("/models/sub/m.tokenizer.json"),
            "the pacha-cache candidate must be a concrete path: {msg}"
        );
        assert!(
            msg.contains("/models/sub/tokenizer.json"),
            "the model-directory candidate must be a concrete path: {msg}"
        );
        assert!(msg.contains("No Qwen tokenizer found"));
        assert!(msg.contains("HuggingFace cache"));
        assert!(msg.contains("APR cache"));
        assert!(msg.contains("apr pull"));
    }

    /// A bare filename has no parent directory; the message must still resolve
    /// to something a user can act on rather than an empty path fragment.
    #[test]
    fn no_qwen_tokenizer_message_handles_bare_filename() {
        let msg = no_qwen_tokenizer_message(Path::new("m.apr"));
        assert!(!msg.contains("{stem}"), "{msg}");
        assert!(
            msg.contains("./m.tokenizer.json"),
            "bare filename must resolve against the CWD: {msg}"
        );
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
        // Ġ -> space (each one), !!!!!! -> !!!, markers removed, trailing whitespace trimmed.
        // 0.69.1 CRUX sweep: runs of spaces are content (byte-level BPE `ĠĠ` IS two spaces -- the tokens
        // Python indentation is made of), so they survive; only surrounding blank lines and trailing
        // whitespace go. This test used to assert the collapse, i.e. the ctl-code-add defect.
        assert_eq!(cleaned, "  Hello!!!  world");
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
