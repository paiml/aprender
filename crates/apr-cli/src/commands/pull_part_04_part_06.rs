
    #[test]
    fn test_resolve_hf_uri_with_file_scheme() {
        let uri = "file:///home/user/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri);
    }

    #[test]
    fn test_resolve_hf_uri_s3_scheme() {
        let uri = "s3://bucket/key/model.safetensors";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri);
    }

    // =========================================================================
    // NEW: fetch_safetensors_companions additional offline tests
    // =========================================================================

    #[test]
    fn test_fetch_companions_hf_single_slash_uri_noop() {
        // "hf:/onlyorg" → extract_hf_repo returns None → noop
        let temp_dir = std::env::temp_dir().join("apr_companion_hf_single_slash");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("model.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        let result = fetch_safetensors_companions(&model_path, "hf:/onlyorg");
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fetch_companions_uppercase_hf_noop() {
        // "HF://org/repo" — extract_hf_repo uses strip_prefix("hf://") which is case-sensitive
        let temp_dir = std::env::temp_dir().join("apr_companion_uppercase_hf");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("model.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        let result = fetch_safetensors_companions(&model_path, "HF://Org/Repo");
        assert!(result.is_ok());
        // No companions created (not a valid hf:// URI)
        assert!(!temp_dir.join("model.tokenizer.json").exists());
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_fetch_companions_model_without_extension() {
        // Model file with no extension → file_stem returns the whole name
        let temp_dir = std::env::temp_dir().join("apr_companion_no_ext");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("model_no_ext");
        let _ = std::fs::write(&model_path, b"dummy");

        // Pre-create companion files with the stem prefix
        let _ = std::fs::write(temp_dir.join("model_no_ext.tokenizer.json"), b"{}");
        let _ = std::fs::write(temp_dir.join("model_no_ext.config.json"), b"{}");

        let result = fetch_safetensors_companions(&model_path, "hf://org/repo/model.safetensors");
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
