pub(crate) use super::*;

// ============================================================================
// Tests
// ============================================================================
#[test]
fn test_push_options_default() {
    let opts = PushOptions::default();
    assert!(opts.commit_message.is_none());
    assert!(opts.model_card.is_none());
    assert!(opts.create_repo);
    assert!(!opts.private);
    assert_eq!(opts.filename, "model.apr");
}

#[test]
fn test_push_options_builder() {
    let card = ModelCard::new("test", "1.0.0");
    let opts = PushOptions::new()
        .with_commit_message("Test commit")
        .with_model_card(card)
        .with_create_repo(false)
        .with_private(true)
        .with_filename("custom.apr");

    assert_eq!(opts.commit_message, Some("Test commit".to_string()));
    assert!(opts.model_card.is_some());
    assert!(!opts.create_repo);
    assert!(opts.private);
    assert_eq!(opts.filename, "custom.apr");
}

#[test]
fn test_hf_hub_client_new() {
    // Should not error even without token
    let client = HfHubClient::new().expect("Should create client");
    // Token depends on environment
    assert!(client.api_base.contains("huggingface.co"));
}

#[test]
fn test_hf_hub_client_with_token() {
    let client = HfHubClient::with_token("test_token");
    assert!(client.is_authenticated());
}

#[test]
fn test_parse_repo_id_valid() {
    let result = HfHubClient::parse_repo_id("paiml/my-model");
    assert!(result.is_ok());
    let (org, name) = result.unwrap();
    assert_eq!(org, "paiml");
    assert_eq!(name, "my-model");
}

#[test]
fn test_parse_repo_id_invalid() {
    let result = HfHubClient::parse_repo_id("invalid");
    assert!(result.is_err());

    let result = HfHubClient::parse_repo_id("too/many/parts");
    assert!(result.is_err());
}

#[test]
fn test_auto_generate_card() {
    let card = HfHubClient::auto_generate_card("paiml/syscall-model", "KMeans", "1.0.0");

    assert_eq!(card.model_id, "paiml/syscall-model");
    assert_eq!(card.version, "1.0.0");
    assert_eq!(card.name, "syscall-model");
    assert_eq!(card.architecture, Some("KMeans".to_string()));
}

#[test]
fn test_push_to_hub_without_token() {
    let client = HfHubClient {
        token: None,
        cache_dir: PathBuf::from("/tmp"),
        api_base: "https://huggingface.co".to_string(),
    };

    let result = client.push_to_hub("org/repo", b"test", PushOptions::default());
    assert!(matches!(result, Err(HfHubError::MissingToken)));
}

// Note: test_push_to_hub_with_token removed - APR-PUB-001 now actually uploads
// Network tests require real HF_TOKEN or mock server

#[test]
fn test_upload_progress_percentage() {
    let progress = UploadProgress {
        bytes_sent: 50,
        total_bytes: 100,
        current_file: "test.apr".to_string(),
        files_completed: 0,
        total_files: 1,
    };
    assert!((progress.percentage() - 50.0).abs() < 0.01);

    let progress_zero = UploadProgress {
        bytes_sent: 0,
        total_bytes: 0,
        current_file: "test.apr".to_string(),
        files_completed: 0,
        total_files: 1,
    };
    assert!((progress_zero.percentage() - 100.0).abs() < 0.01);
}

#[test]
fn test_upload_result_fields() {
    let result = UploadResult {
        repo_url: "https://huggingface.co/paiml/test".to_string(),
        commit_sha: "abc123".to_string(),
        files_uploaded: vec!["model.apr".to_string(), "README.md".to_string()],
        bytes_transferred: 1024,
    };
    assert_eq!(result.files_uploaded.len(), 2);
    assert_eq!(result.bytes_transferred, 1024);
}

#[test]
fn test_push_options_with_progress_callback() {
    let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let called_clone = called.clone();
    let callback: ProgressCallback = Arc::new(move |_progress| {
        called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
    });

    let opts = PushOptions::new().with_progress_callback(callback);
    assert!(opts.progress_callback.is_some());
}

#[test]
fn test_push_options_with_max_retries() {
    let opts = PushOptions::new().with_max_retries(5);
    assert_eq!(opts.max_retries, 5);
}

#[test]
fn test_hf_hub_error_display() {
    assert_eq!(
        HfHubError::MissingToken.to_string(),
        "HF_TOKEN environment variable not set"
    );
    assert!(HfHubError::RepoNotFound("test".into())
        .to_string()
        .contains("test"));
}

// Note: test_push_saves_readme removed - APR-PUB-001 now actually uploads
// The README content is tested via model_card tests instead

#[test]
fn test_base64_encode_basic() {
    // Test basic encoding
    assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
    assert_eq!(base64_encode(b""), "");
    assert_eq!(base64_encode(b"a"), "YQ==");
    assert_eq!(base64_encode(b"ab"), "YWI=");
    assert_eq!(base64_encode(b"abc"), "YWJj");
}

#[test]
fn test_base64_encode_binary() {
    // Test binary data encoding
    let data: [u8; 4] = [0x00, 0xFF, 0x80, 0x7F];
    let encoded = base64_encode(&data);
    assert!(!encoded.is_empty());
    // Verify it's valid base64 (only contains valid chars)
    assert!(encoded
        .chars()
        .all(|c| { c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=' }));
}

// ========================================================================
// Edge Case Tests (EXTREME TDD)
// ========================================================================

#[test]
fn test_all_error_variants_display() {
    // Test all HfHubError variants have proper Display
    let errors = vec![
        HfHubError::MissingToken,
        HfHubError::NetworkError("connection failed".into()),
        HfHubError::RepoNotFound("org/repo".into()),
        HfHubError::FileNotFound("model.apr".into()),
        HfHubError::InvalidRepoId("bad-id".into()),
        HfHubError::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        )),
        HfHubError::ModelCardError("invalid format".into()),
    ];

    for err in errors {
        let msg = err.to_string();
        assert!(!msg.is_empty(), "Error display should not be empty");
    }
}

#[test]
fn test_io_error_conversion() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
    let hf_err: HfHubError = io_err.into();
    assert!(matches!(hf_err, HfHubError::IoError(_)));
    assert!(hf_err.to_string().contains("denied"));
}

#[test]
fn test_with_cache_dir() {
    let client = HfHubClient::with_token("token").with_cache_dir("/custom/cache");
    assert_eq!(client.cache_dir, PathBuf::from("/custom/cache"));
}

#[test]
fn test_parse_empty_repo_id() {
    let result = HfHubClient::parse_repo_id("");
    assert!(result.is_err());
}

#[test]
fn test_parse_repo_id_with_empty_parts() {
    // "org/" has empty name
    let result = HfHubClient::parse_repo_id("org/");
    assert!(result.is_ok()); // Still parses, just empty name

    // "/name" has empty org
    let result = HfHubClient::parse_repo_id("/name");
    assert!(result.is_ok()); // Still parses, just empty org
}

#[test]
fn test_push_to_hub_invalid_repo_id() {
    let client = HfHubClient::with_token("token");
    let result = client.push_to_hub("invalid", b"data", PushOptions::default());
    assert!(matches!(result, Err(HfHubError::InvalidRepoId(_))));
}

#[test]
fn test_push_options_all_builders() {
    let opts = PushOptions::new()
        .with_commit_message("msg")
        .with_create_repo(true)
        .with_private(true)
        .with_filename("test.apr");

    assert_eq!(opts.commit_message, Some("msg".to_string()));
    assert!(opts.create_repo);
    assert!(opts.private);
    assert_eq!(opts.filename, "test.apr");
}

#[test]
fn test_auto_generate_card_single_name() {
    // Test with repo_id that has no slash
    let card = HfHubClient::auto_generate_card("simple", "LinearRegression", "2.0.0");
    assert_eq!(card.model_id, "simple");
    assert_eq!(card.name, "simple"); // Falls back to full string
    assert_eq!(card.version, "2.0.0");
}

#[test]
fn test_is_authenticated() {
    let client_with_token = HfHubClient::with_token("secret");
    assert!(client_with_token.is_authenticated());

    let client_without = HfHubClient {
        token: None,
        cache_dir: PathBuf::from("/tmp"),
        api_base: "https://huggingface.co".to_string(),
    };
    assert!(!client_without.is_authenticated());
}

#[test]
fn test_default_cache_dir_exists() {
    let dir = HfHubClient::default_cache_dir();
    // Should contain "huggingface" and "hub" in path
    let path_str = dir.to_string_lossy();
    assert!(path_str.contains("huggingface") || path_str.contains('.'));
}

// Note: test_push_creates_model_file and test_push_auto_generates_model_card removed
// APR-PUB-001 now actually uploads to HF Hub instead of saving locally
// Network-based integration tests would require mock server or real credentials

#[test]
fn test_push_options_default_retry_settings() {
    let opts = PushOptions::default();
    assert_eq!(opts.max_retries, 3);
    assert_eq!(opts.initial_backoff_ms, 1000);
}

#[test]
fn test_upload_progress_fields() {
    let progress = UploadProgress {
        bytes_sent: 1000,
        total_bytes: 5000,
        current_file: "model.safetensors".to_string(),
        files_completed: 1,
        total_files: 3,
    };
    assert_eq!(progress.bytes_sent, 1000);
    assert_eq!(progress.total_bytes, 5000);
    assert_eq!(progress.current_file, "model.safetensors");
    assert_eq!(progress.files_completed, 1);
    assert_eq!(progress.total_files, 3);
    assert!((progress.percentage() - 20.0).abs() < 0.01);
}

#[test]
fn test_hf_hub_client_default() {
    // Test Default trait implementation
    let client = HfHubClient::default();
    assert!(client.api_base.contains("huggingface.co"));
}

#[cfg(not(feature = "hf-hub-integration"))]
#[test]
fn test_pull_without_feature() {
    let client = HfHubClient::with_token("token");
    let result = client.pull_from_hub("org/repo");
    assert!(matches!(result, Err(HfHubError::NetworkError(_))));
}

// ============================================================================
// Property Tests
// ============================================================================

mod proptests {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        /// Property: Valid repo IDs always parse successfully
        #[test]
        fn prop_valid_repo_id_parses(
            org in "[a-zA-Z][a-zA-Z0-9-]{0,30}",
            name in "[a-zA-Z][a-zA-Z0-9-]{0,30}",
        ) {
            let repo_id = format!("{org}/{name}");
            let result = HfHubClient::parse_repo_id(&repo_id);
            prop_assert!(result.is_ok());
            let (parsed_org, parsed_name) = result.expect("Valid repo ID should parse");
            prop_assert_eq!(parsed_org, org.as_str());
            prop_assert_eq!(parsed_name, name.as_str());
        }

        /// Property: Invalid repo IDs always fail
        #[test]
        fn prop_invalid_repo_id_fails(
            name in "[a-zA-Z0-9-]{1,30}",
        ) {
            // Single part (no slash)
            let result = HfHubClient::parse_repo_id(&name);
            prop_assert!(result.is_err());
        }

        /// Property: PushOptions builder is idempotent
        #[test]
        fn prop_push_options_builder_idempotent(
            msg in "[a-zA-Z0-9 ]{1,50}",
            filename in "[a-zA-Z0-9_.-]{1,20}",
        ) {
            let opts1 = PushOptions::new()
                .with_commit_message(&msg)
                .with_commit_message(&msg)
                .with_filename(&filename)
                .with_filename(&filename);

            let opts2 = PushOptions::new()
                .with_commit_message(&msg)
                .with_filename(&filename);

            prop_assert_eq!(opts1.commit_message, opts2.commit_message);
            prop_assert_eq!(opts1.filename, opts2.filename);
        }

        /// Property: Auto-generated card has correct model_id
        #[test]
        fn prop_auto_card_has_repo_id(
            org in "[a-zA-Z][a-zA-Z0-9]{0,10}",
            name in "[a-zA-Z][a-zA-Z0-9]{0,10}",
            model_type in "[A-Z][a-zA-Z]{2,15}",
        ) {
            let repo_id = format!("{org}/{name}");
            let card = HfHubClient::auto_generate_card(&repo_id, &model_type, "1.0.0");

            prop_assert_eq!(card.model_id, repo_id);
            prop_assert_eq!(card.name, name);
            prop_assert_eq!(card.architecture, Some(model_type));
        }
    }
}
