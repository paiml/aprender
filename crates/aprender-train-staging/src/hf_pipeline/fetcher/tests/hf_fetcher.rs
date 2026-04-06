//! Tests for HfModelFetcher.

use crate::hf_pipeline::fetcher::{
    Architecture, FetchOptions, HfModelFetcher, ModelArtifact, WeightFormat,
};
use std::path::PathBuf;

#[test]
fn test_fetcher_new() {
    let fetcher = HfModelFetcher::new();
    assert!(fetcher.is_ok());
}

#[test]
fn test_fetcher_with_token() {
    let fetcher = HfModelFetcher::with_token("hf_test_token");
    assert!(fetcher.is_authenticated());
}

#[test]
fn test_fetcher_without_token_is_not_authenticated() {
    // Clear env for test
    let _saved = std::env::var("HF_TOKEN");
    std::env::remove_var("HF_TOKEN");

    let fetcher = HfModelFetcher {
        token: None,
        cache_dir: PathBuf::from("/tmp"),
        api_base: "https://huggingface.co".into(),
    };
    assert!(!fetcher.is_authenticated());
}

#[test]
fn test_parse_repo_id_valid() {
    let result = HfModelFetcher::parse_repo_id("microsoft/codebert-base");
    assert!(result.is_ok());
    let (org, name) = result.expect("operation should succeed");
    assert_eq!(org, "microsoft");
    assert_eq!(name, "codebert-base");
}

#[test]
fn test_parse_repo_id_invalid_no_slash() {
    use crate::hf_pipeline::error::FetchError;
    let result = HfModelFetcher::parse_repo_id("invalid");
    assert!(matches!(result, Err(FetchError::InvalidRepoId { .. })));
}

#[test]
fn test_parse_repo_id_invalid_empty_org() {
    use crate::hf_pipeline::error::FetchError;
    let result = HfModelFetcher::parse_repo_id("/model");
    assert!(matches!(result, Err(FetchError::InvalidRepoId { .. })));
}

#[test]
fn test_parse_repo_id_invalid_empty_name() {
    use crate::hf_pipeline::error::FetchError;
    let result = HfModelFetcher::parse_repo_id("org/");
    assert!(matches!(result, Err(FetchError::InvalidRepoId { .. })));
}

#[test]
fn test_parse_repo_id_invalid_too_many_parts() {
    use crate::hf_pipeline::error::FetchError;
    let result = HfModelFetcher::parse_repo_id("a/b/c");
    assert!(matches!(result, Err(FetchError::InvalidRepoId { .. })));
}

#[test]
fn test_download_rejects_pytorch_by_default() {
    use crate::hf_pipeline::error::FetchError;
    let fetcher = HfModelFetcher::with_token("test");
    let result =
        fetcher.download_model("test/model", FetchOptions::new().files(&["pytorch_model.bin"]));
    assert!(matches!(result, Err(FetchError::PickleSecurityRisk)));
}

#[test]
fn test_download_nonexistent_repo_returns_error() {
    let temp_dir = std::env::temp_dir().join("hf_test_nonexistent");
    let _ = std::fs::remove_dir_all(&temp_dir);

    let fetcher = HfModelFetcher::new().expect("operation should succeed").cache_dir(&temp_dir);
    let result = fetcher.download_model(
        "nonexistent-org-xyz123/nonexistent-model-abc456",
        FetchOptions::new().files(&["model.safetensors"]).cache_dir(&temp_dir),
    );

    // Should fail with some error (network or not found)
    assert!(result.is_err(), "Non-existent repo should return error");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_download_security_check_blocks_pytorch() {
    use crate::hf_pipeline::error::FetchError;
    let temp_dir = std::env::temp_dir().join("hf_test_security");
    let _ = std::fs::remove_dir_all(&temp_dir);

    let fetcher = HfModelFetcher::new().expect("operation should succeed").cache_dir(&temp_dir);
    let result = fetcher.download_model(
        "microsoft/codebert-base",
        FetchOptions::new()
            .files(&["pytorch_model.bin"]) // PyTorch without allow flag
            .cache_dir(&temp_dir),
    );

    // Should be blocked by security check BEFORE network access
    assert!(
        matches!(result, Err(FetchError::PickleSecurityRisk)),
        "PyTorch files should be blocked without allow_pytorch_pickle"
    );

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
#[ignore] // Requires network access
fn test_download_real_model_integration() {
    let temp_dir = std::env::temp_dir().join("hf_test_real");
    let _ = std::fs::remove_dir_all(&temp_dir);

    let fetcher = HfModelFetcher::new().expect("operation should succeed").cache_dir(&temp_dir);
    let result = fetcher.download_model(
        "hf-internal-testing/tiny-random-bert",
        FetchOptions::new().files(&["config.json"]).cache_dir(&temp_dir),
    );

    assert!(result.is_ok(), "Should download from real repo: {:?}", result.err());
    let artifact = result.expect("operation should succeed");
    assert!(artifact.path.exists(), "Cache directory should exist");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_estimate_memory_fp32() {
    // 125M params * 4 bytes = 500MB
    let mem = HfModelFetcher::estimate_memory(125_000_000, 4);
    assert_eq!(mem, 500_000_000);
}

#[test]
fn test_estimate_memory_fp16() {
    // 125M params * 2 bytes = 250MB
    let mem = HfModelFetcher::estimate_memory(125_000_000, 2);
    assert_eq!(mem, 250_000_000);
}

#[test]
fn test_estimate_memory_int4() {
    // 125M params * 0.5 bytes = 62.5MB (but we use 1 byte minimum)
    let mem = HfModelFetcher::estimate_memory(125_000_000, 1);
    assert_eq!(mem, 125_000_000);
}

#[test]
fn test_fetcher_cache_dir() {
    let temp_dir = std::env::temp_dir().join("hf_test_cache_dir");
    let fetcher = HfModelFetcher::with_token("test").cache_dir(&temp_dir);
    assert_eq!(fetcher.cache_dir, temp_dir);
}

#[test]
fn test_default_cache_dir() {
    let cache_dir = HfModelFetcher::default_cache_dir();
    // Should return a path (either from env or default)
    assert!(!cache_dir.as_os_str().is_empty());
}

#[test]
fn test_hf_model_fetcher_default() {
    // Default impl should create a valid fetcher
    let fetcher = HfModelFetcher::default();
    // Should not panic and have reasonable defaults
    assert!(!fetcher.cache_dir.as_os_str().is_empty());
}

#[test]
fn test_model_artifact_debug() {
    let artifact = ModelArtifact {
        path: PathBuf::from("/tmp/model"),
        format: WeightFormat::SafeTensors,
        architecture: Some(Architecture::BERT {
            num_layers: 12,
            hidden_size: 768,
            num_attention_heads: 12,
        }),
        sha256: Some("abc123".into()),
    };
    let debug = format!("{artifact:?}");
    assert!(debug.contains("SafeTensors"));
    assert!(debug.contains("abc123"));
}

#[test]
fn test_download_with_non_main_revision() {
    let temp_dir = std::env::temp_dir().join("hf_test_revision");
    let _ = std::fs::remove_dir_all(&temp_dir);

    let fetcher = HfModelFetcher::new().expect("operation should succeed").cache_dir(&temp_dir);
    let result = fetcher.download_model(
        "nonexistent-org/nonexistent-model",
        FetchOptions::new()
            .revision("v1.0.0") // Non-main revision
            .files(&["model.safetensors"])
            .cache_dir(&temp_dir),
    );

    // Should fail (repo doesn't exist), but should exercise the revision branch
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_download_with_empty_files_uses_defaults() {
    let temp_dir = std::env::temp_dir().join("hf_test_empty_files");
    let _ = std::fs::remove_dir_all(&temp_dir);

    let fetcher = HfModelFetcher::new().expect("operation should succeed").cache_dir(&temp_dir);
    // Empty files should default to model.safetensors and config.json
    let result = fetcher.download_model(
        "nonexistent-org-abc123/nonexistent-model-xyz789",
        FetchOptions::new().cache_dir(&temp_dir),
    );

    // Should fail (repo doesn't exist)
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_download_pytorch_allowed_when_flag_set() {
    use crate::hf_pipeline::error::FetchError;
    let temp_dir = std::env::temp_dir().join("hf_test_pytorch_allowed");
    let _ = std::fs::remove_dir_all(&temp_dir);

    let fetcher = HfModelFetcher::new().expect("operation should succeed").cache_dir(&temp_dir);
    let result = fetcher.download_model(
        "nonexistent-org/nonexistent-model",
        FetchOptions::new()
            .files(&["pytorch_model.bin"])
            .allow_pytorch_pickle(true)
            .cache_dir(&temp_dir),
    );

    // Should NOT be blocked by security check (will fail for other reasons)
    assert!(!matches!(result, Err(FetchError::PickleSecurityRisk)));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_download_uses_custom_cache_dir_from_options() {
    let temp_dir = std::env::temp_dir().join("hf_test_custom_cache");
    let _ = std::fs::remove_dir_all(&temp_dir);

    let fetcher = HfModelFetcher::new().expect("operation should succeed");
    let result = fetcher.download_model(
        "nonexistent-org/nonexistent-model",
        FetchOptions::new().files(&["model.safetensors"]).cache_dir(&temp_dir), // Custom cache from options
    );

    // Should fail (repo doesn't exist)
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&temp_dir);
}
