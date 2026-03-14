
    // =========================================================================
    // GH-355: Gated model error formatting
    // =========================================================================

    #[test]
    fn test_format_gated_model_error_without_token() {
        // Use _inner to avoid env var race conditions in parallel tests.
        let msg = format_gated_model_error_inner(
            "https://huggingface.co/api/models/google/codegemma-7b-it",
            false,
        );
        assert!(msg.contains("Access denied (HTTP 401)"), "msg: {msg}");
        assert!(msg.contains("HF_TOKEN"), "Should mention HF_TOKEN: {msg}");
        assert!(
            msg.contains("huggingface-cli login"),
            "Should mention login: {msg}"
        );
    }

    #[test]
    fn test_format_gated_model_error_with_token() {
        // Use _inner to avoid env var race conditions in parallel tests.
        let msg = format_gated_model_error_inner(
            "https://huggingface.co/api/models/google/codegemma-7b-it",
            true,
        );
        assert!(msg.contains("Access denied (HTTP 401)"), "msg: {msg}");
        assert!(
            msg.contains("lacks access"),
            "Should say token lacks access: {msg}"
        );
        assert!(
            msg.contains("Request access"),
            "Should suggest requesting access: {msg}"
        );
    }

    // =========================================================================
    // GH-355: resolve_hf_token priority
    // =========================================================================

    #[test]
    fn test_resolve_hf_token_env_var_priority() {
        let saved = std::env::var("HF_TOKEN").ok();
        std::env::set_var("HF_TOKEN", "hf_env_token");

        let token = resolve_hf_token();
        assert_eq!(token, Some("hf_env_token".to_string()));

        match saved {
            Some(t) => std::env::set_var("HF_TOKEN", t),
            None => std::env::remove_var("HF_TOKEN"),
        }
    }

    #[test]
    fn test_resolve_hf_token_empty_env_var_skipped() {
        let saved = std::env::var("HF_TOKEN").ok();
        std::env::set_var("HF_TOKEN", "");

        let token = resolve_hf_token();
        // Empty env var is skipped — may still find a file token or None
        // We just verify it doesn't return Some("")
        if let Some(ref t) = token {
            assert!(!t.is_empty(), "Should not return empty token");
        }

        match saved {
            Some(t) => std::env::set_var("HF_TOKEN", t),
            None => std::env::remove_var("HF_TOKEN"),
        }
    }

    #[test]
    fn test_hf_get_sets_auth_header_when_token_present() {
        let saved = std::env::var("HF_TOKEN").ok();
        std::env::set_var("HF_TOKEN", "hf_test_auth");

        // hf_get returns a ureq::Request — we can't inspect headers directly,
        // but we can verify it doesn't panic with a token set
        let _req = hf_get("https://example.com/test");

        match saved {
            Some(t) => std::env::set_var("HF_TOKEN", t),
            None => std::env::remove_var("HF_TOKEN"),
        }
    }

    #[test]
    fn test_hf_get_works_without_token() {
        let saved = std::env::var("HF_TOKEN").ok();
        std::env::remove_var("HF_TOKEN");

        let _req = hf_get("https://example.com/test");

        if let Some(t) = saved {
            std::env::set_var("HF_TOKEN", t);
        }
    }

    // =========================================================================
    // GH-356: Tokenizer fallback chain (download_companion_files)
    // =========================================================================

    #[test]
    fn test_download_companion_files_tokenizer_json_present() {
        let temp_dir = std::env::temp_dir().join("apr_gh356_tok_json");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");

        // Pre-create tokenizer.json and config.json
        std::fs::write(temp_dir.join("tokenizer.json"), b"{}").expect("write");
        std::fs::write(temp_dir.join("config.json"), b"{}").expect("write");

        // download_companion_files checks for cached files — these exist, so no network calls
        let result = download_companion_files(&temp_dir, "https://example.com/nonexistent", false);
        assert!(result.is_ok(), "Should succeed with tokenizer.json present: {result:?}");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_download_companion_files_tokenizer_model_fallback() {
        let temp_dir = std::env::temp_dir().join("apr_gh356_tok_model");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");

        // Only tokenizer.model and config.json — no tokenizer.json
        std::fs::write(temp_dir.join("tokenizer.model"), b"\x00\x01\x02").expect("write");
        std::fs::write(temp_dir.join("config.json"), b"{}").expect("write");

        let result = download_companion_files(&temp_dir, "https://example.com/nonexistent", false);
        assert!(
            result.is_ok(),
            "Should succeed with tokenizer.model fallback: {result:?}"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_download_companion_files_tokenizer_config_only() {
        let temp_dir = std::env::temp_dir().join("apr_gh356_tok_config_only");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");

        // Only tokenizer_config.json and config.json
        std::fs::write(temp_dir.join("tokenizer_config.json"), b"{}").expect("write");
        std::fs::write(temp_dir.join("config.json"), b"{}").expect("write");

        let result = download_companion_files(&temp_dir, "https://example.com/nonexistent", false);
        assert!(
            result.is_ok(),
            "Should succeed with tokenizer_config.json only: {result:?}"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_download_companion_files_no_tokenizer_fails() {
        let temp_dir = std::env::temp_dir().join("apr_gh356_no_tok");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");

        // Only config.json — no tokenizer at all
        std::fs::write(temp_dir.join("config.json"), b"{}").expect("write");

        let result = download_companion_files(&temp_dir, "https://example.com/nonexistent", false);
        assert!(result.is_err(), "Should fail with no tokenizer files");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("No tokenizer found"),
            "Error should mention missing tokenizer: {err_msg}"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // =========================================================================
    // GH-356: fetch_safetensors_companions with tokenizer.model
    // =========================================================================

    #[test]
    fn test_fetch_companions_includes_tokenizer_model() {
        let temp_dir = std::env::temp_dir().join("apr_gh356_companions_tok_model");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");

        let model_path = temp_dir.join("abc123.safetensors");
        std::fs::write(&model_path, b"dummy").expect("write");

        // Pre-create all companion files with hash-prefix
        std::fs::write(temp_dir.join("abc123.tokenizer.json"), b"{}").expect("write");
        std::fs::write(temp_dir.join("abc123.config.json"), b"{}").expect("write");
        std::fs::write(temp_dir.join("abc123.tokenizer_config.json"), b"{}").expect("write");
        std::fs::write(temp_dir.join("abc123.tokenizer.model"), b"\x00").expect("write");

        // All files pre-exist → no network calls
        let result = fetch_safetensors_companions(&model_path, "hf://org/repo/model.safetensors");
        assert!(result.is_ok(), "Should succeed with all companions cached: {result:?}");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fetch_companions_no_tokenizer_fails_fast() {
        let temp_dir = std::env::temp_dir().join("apr_gh356_companions_no_tok");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");

        let model_path = temp_dir.join("abc456.safetensors");
        std::fs::write(&model_path, b"dummy").expect("write");

        // Only config — no tokenizer files at all (with hash prefix)
        std::fs::write(temp_dir.join("abc456.config.json"), b"{}").expect("write");

        // Should fail fast: no tokenizer.json, tokenizer.model, or tokenizer_config.json
        let result = fetch_safetensors_companions(&model_path, "hf://org/repo/model.safetensors");
        assert!(result.is_err(), "Should fail with no tokenizer: {result:?}");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("No tokenizer found"),
            "Error should mention missing tokenizer: {err_msg}"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fetch_companions_tokenizer_model_only_passes() {
        let temp_dir = std::env::temp_dir().join("apr_gh356_companions_tok_model_only");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");

        let model_path = temp_dir.join("def789.safetensors");
        std::fs::write(&model_path, b"dummy").expect("write");

        // Only tokenizer.model — no tokenizer.json (SentencePiece fallback)
        std::fs::write(temp_dir.join("def789.config.json"), b"{}").expect("write");
        std::fs::write(temp_dir.join("def789.tokenizer.model"), b"\x00\x01").expect("write");

        let result = fetch_safetensors_companions(&model_path, "hf://org/repo/model.safetensors");
        assert!(result.is_ok(), "Should pass with tokenizer.model only: {result:?}");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // =========================================================================
    // GH-357: PyTorch .bin detection
    // =========================================================================

    #[test]
    fn test_resolve_hf_model_local_path_bypass() {
        // Local paths bypass all HF logic — no .bin detection needed
        let result = resolve_hf_model("/tmp/some/model.bin").expect("local path should work");
        match result {
            ResolvedModel::SingleFile(s) => assert_eq!(s, "/tmp/some/model.bin"),
            ResolvedModel::Sharded { .. } => panic!("Expected SingleFile for local path"),
        }
    }

    #[test]
    fn test_extract_hf_repo_with_deep_path() {
        assert_eq!(
            extract_hf_repo("hf://org/repo/subdir/model.safetensors"),
            Some("org/repo".to_string())
        );
    }

    #[test]
    fn test_extract_hf_repo_minimal() {
        assert_eq!(
            extract_hf_repo("hf://org/repo"),
            Some("org/repo".to_string())
        );
    }

    #[test]
    fn test_extract_hf_repo_not_hf() {
        assert_eq!(extract_hf_repo("https://example.com/model"), None);
        assert_eq!(extract_hf_repo("/local/path"), None);
        assert_eq!(extract_hf_repo(""), None);
    }

    #[test]
    fn test_has_known_model_extension_bin_not_recognized() {
        // .bin is NOT a known model extension — we can't load it directly
        assert!(!has_known_model_extension("model.bin"));
    }

    #[test]
    fn test_has_known_model_extension_safetensors() {
        assert!(has_known_model_extension("model.safetensors"));
    }

    #[test]
    fn test_has_known_model_extension_gguf() {
        assert!(has_known_model_extension("model.gguf"));
    }

    #[test]
    fn test_has_known_model_extension_apr() {
        assert!(has_known_model_extension("model.apr"));
    }

    #[test]
    fn test_has_known_model_extension_no_extension() {
        assert!(!has_known_model_extension("model"));
    }
