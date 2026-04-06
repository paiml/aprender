//! Tests for FetchOptions builder.

use crate::hf_pipeline::fetcher::FetchOptions;
use std::path::PathBuf;

#[test]
fn test_fetch_options_default() {
    let opts = FetchOptions::default();
    assert_eq!(opts.revision, "main");
    assert!(opts.files.is_empty());
    assert!(!opts.allow_pytorch_pickle);
    assert!(opts.verify_sha256.is_none());
}

#[test]
fn test_fetch_options_builder() {
    let opts = FetchOptions::new()
        .revision("v1.0")
        .files(&["model.safetensors"])
        .allow_pytorch_pickle(true)
        .verify_sha256("abc123")
        .cache_dir("/tmp/cache");

    assert_eq!(opts.revision, "v1.0");
    assert_eq!(opts.files, vec!["model.safetensors"]);
    assert!(opts.allow_pytorch_pickle);
    assert_eq!(opts.verify_sha256, Some("abc123".into()));
    assert_eq!(opts.cache_dir, Some(PathBuf::from("/tmp/cache")));
}

#[test]
fn test_fetch_options_cache_dir_pathbuf() {
    let path = PathBuf::from("/custom/cache/dir");
    let opts = FetchOptions::new().cache_dir(path.clone());
    assert_eq!(opts.cache_dir, Some(path));
}

#[test]
fn test_fetch_options_multiple_files() {
    let opts = FetchOptions::new().files(&[
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "vocab.txt",
    ]);
    assert_eq!(opts.files.len(), 4);
    assert!(opts.files.contains(&"vocab.txt".to_string()));
}

#[test]
fn test_fetch_options_debug_and_clone() {
    let opts =
        FetchOptions::new().revision("v2.0").files(&["model.gguf"]).verify_sha256("deadbeef");
    let debug = format!("{opts:?}");
    assert!(debug.contains("v2.0"));

    let cloned = opts.clone();
    assert_eq!(cloned.revision, "v2.0");
}
