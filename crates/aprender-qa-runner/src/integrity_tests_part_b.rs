/// Verify file integrity passes with hash-prefixed config and safetensors
#[test]
fn test_file_integrity_with_hash_prefix_config() {
    let dir = TempDir::new().expect("create temp dir");
    // Simulate pacha cache: <hash>.safetensors + <hash>.config.json
    create_named_config(dir.path(), "abc123.config.json", 24, 896, 151_936);
    create_named_safetensors(dir.path(), "abc123.safetensors", 24, 896, 151_936);

    let model_file = dir.path().join("abc123.safetensors");
    let result = check_safetensors_file_integrity(&model_file);
    assert!(
        result.passed,
        "Should pass with hash-prefixed config: {:?}",
        result.errors
    );
    assert!(result.config_found);
    assert!(result.layer_count_match);
}

/// Verify integrity check uses only matching model config in shared directory
#[test]
fn test_file_integrity_ignores_other_models_in_shared_dir() {
    let dir = TempDir::new().expect("create temp dir");
    // Model A: 24 layers (the one we're checking)
    create_named_config(dir.path(), "aaa111.config.json", 24, 896, 151_936);
    create_named_safetensors(dir.path(), "aaa111.safetensors", 24, 896, 151_936);
    // Model B: 28 layers (different model in same dir — must be ignored)
    create_named_config(dir.path(), "bbb222.config.json", 28, 3584, 151_936);
    create_named_safetensors(dir.path(), "bbb222.safetensors", 28, 3584, 151_936);

    let model_file = dir.path().join("aaa111.safetensors");
    let result = check_safetensors_file_integrity(&model_file);
    assert!(
        result.passed,
        "Must use only aaa111's config and tensors, not bbb222's: {:?}",
        result.errors
    );
    assert_eq!(
        result.tensor_values.as_ref().unwrap().layer_count,
        Some(24),
        "Should see 24 layers from aaa111, not 28 from bbb222"
    );
}

/// Verify integrity check fails when no config file is found
#[test]
fn test_file_integrity_no_config_found() {
    let dir = TempDir::new().expect("create temp dir");
    // Safetensors file with no matching config
    create_named_safetensors(dir.path(), "orphan.safetensors", 12, 768, 30_000);

    let model_file = dir.path().join("orphan.safetensors");
    let result = check_safetensors_file_integrity(&model_file);
    assert!(!result.passed);
    assert!(!result.config_found);
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.contains("G0-INTEGRITY-CONFIG"))
    );
}

/// Verify integrity check falls back to plain config.json when no hash prefix
#[test]
fn test_file_integrity_falls_back_to_plain_config() {
    let dir = TempDir::new().expect("create temp dir");
    // No hash-prefixed config, but plain config.json exists
    create_test_config(dir.path(), 24, 896, 151_936);
    create_named_safetensors(dir.path(), "model.safetensors", 24, 896, 151_936);

    let model_file = dir.path().join("model.safetensors");
    let result = check_safetensors_file_integrity(&model_file);
    assert!(
        result.passed,
        "Should fall back to config.json: {:?}",
        result.errors
    );
}

/// Verify integrity check fails when config and tensor layer counts mismatch
#[test]
fn test_file_integrity_layer_mismatch() {
    let dir = TempDir::new().expect("create temp dir");
    create_named_config(dir.path(), "bad.config.json", 14, 896, 151_936);
    create_named_safetensors(dir.path(), "bad.safetensors", 24, 896, 151_936);

    let model_file = dir.path().join("bad.safetensors");
    let result = check_safetensors_file_integrity(&model_file);
    assert!(!result.passed);
    assert!(!result.layer_count_match);
    assert!(result.errors.iter().any(|e| e.contains("LAYERS")));
}

/// Verify find_config_for_model_file finds hash-prefixed config
#[test]
fn test_find_config_for_model_file_hash_prefix() {
    let dir = TempDir::new().expect("create temp dir");
    create_named_config(dir.path(), "d71534cb.config.json", 24, 896, 151_936);
    create_named_safetensors(dir.path(), "d71534cb.safetensors", 24, 896, 151_936);

    let result = find_config_for_model_file(&dir.path().join("d71534cb.safetensors"));
    assert!(result.is_some());
    assert!(
        result
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .contains("d71534cb.config.json")
    );
}

/// Verify find_config_for_model_file returns None when no config matches
#[test]
fn test_find_config_for_model_file_no_match() {
    let dir = TempDir::new().expect("create temp dir");
    create_named_safetensors(dir.path(), "noconf.safetensors", 2, 768, 30_000);

    let result = find_config_for_model_file(&dir.path().join("noconf.safetensors"));
    assert!(result.is_none());
}

/// Verify file integrity fails with invalid JSON config (parse error path)
#[test]
fn test_file_integrity_corrupt_config() {
    let dir = TempDir::new().expect("create temp dir");
    // Write corrupt config.json
    std::fs::write(dir.path().join("bad.config.json"), "not json at all").expect("write");
    create_named_safetensors(dir.path(), "bad.safetensors", 4, 128, 32000);

    let model_file = dir.path().join("bad.safetensors");
    let result = check_safetensors_file_integrity(&model_file);
    assert!(!result.passed);
    assert!(!result.config_found);
    assert!(
        result.errors.iter().any(|e| e.contains("G0-INTEGRITY-CONFIG")),
        "Should report config parse error: {:?}",
        result.errors
    );
}

/// Verify file integrity fails when safetensors file is corrupt (tensor read error)
#[test]
fn test_file_integrity_corrupt_safetensors() {
    let dir = TempDir::new().expect("create temp dir");
    // Create valid config
    create_named_config(dir.path(), "bad2.config.json", 4, 128, 32000);
    // Write corrupt safetensors file (too short)
    std::fs::write(dir.path().join("bad2.safetensors"), b"tiny").expect("write");

    let model_file = dir.path().join("bad2.safetensors");
    let result = check_safetensors_file_integrity(&model_file);
    assert!(!result.passed);
    assert!(
        result.errors.iter().any(|e| e.contains("G0-INTEGRITY-CONFIG")),
        "Should report tensor read error: {:?}",
        result.errors
    );
}

/// Verify file integrity detects hidden_size mismatch (file-mode path)
#[test]
fn test_file_integrity_hidden_size_mismatch() {
    let dir = TempDir::new().expect("create temp dir");
    // Config says hidden=4096, but tensors have 896
    create_named_config(dir.path(), "h.config.json", 4, 4096, 32000);
    create_named_safetensors(dir.path(), "h.safetensors", 4, 896, 32000);

    let model_file = dir.path().join("h.safetensors");
    let result = check_safetensors_file_integrity(&model_file);
    assert!(!result.passed);
    assert!(!result.hidden_size_match);
    assert!(result.errors.iter().any(|e| e.contains("HIDDEN")));
}

/// Verify file integrity detects vocab_size mismatch (file-mode path)
#[test]
fn test_file_integrity_vocab_size_mismatch() {
    let dir = TempDir::new().expect("create temp dir");
    // Config says vocab=50000, but tensors have 32000
    create_named_config(dir.path(), "v.config.json", 4, 896, 50000);
    create_named_safetensors(dir.path(), "v.safetensors", 4, 896, 32000);

    let model_file = dir.path().join("v.safetensors");
    let result = check_safetensors_file_integrity(&model_file);
    assert!(!result.passed);
    assert!(!result.vocab_size_match);
    assert!(result.errors.iter().any(|e| e.contains("VOCAB")));
}

/// Verify find_config_for_model_file skips hash-prefix when file is not .safetensors
#[test]
fn test_find_config_non_safetensors_extension() {
    let dir = TempDir::new().expect("create temp dir");
    // Create a .gguf file and a plain config.json
    std::fs::write(dir.path().join("model.gguf"), "fake gguf").expect("write");
    create_test_config(dir.path(), 4, 128, 32000);

    // .gguf doesn't match .safetensors suffix → skips hash-prefix, falls back to config.json
    let result = find_config_for_model_file(&dir.path().join("model.gguf"));
    assert!(result.is_some(), "Should fall back to config.json for non-.safetensors");
    assert!(
        result
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            == "config.json"
    );
}

/// Verify find_config_for_model_file returns None for non-safetensors without config.json
#[test]
fn test_find_config_non_safetensors_no_fallback() {
    let dir = TempDir::new().expect("create temp dir");
    std::fs::write(dir.path().join("model.gguf"), "fake gguf").expect("write");

    // No config.json either → returns None
    let result = find_config_for_model_file(&dir.path().join("model.gguf"));
    assert!(result.is_none(), "Should return None without any config");
}

/// Verify find_config_for_model_file prefers hash-prefix over plain config.json
#[test]
fn test_find_config_prefers_hash_prefix_over_plain() {
    let dir = TempDir::new().expect("create temp dir");
    // Both hash-prefixed and plain config exist
    create_named_config(dir.path(), "xyz.config.json", 24, 896, 151_936);
    create_test_config(dir.path(), 12, 768, 30_000);
    create_named_safetensors(dir.path(), "xyz.safetensors", 24, 896, 151_936);

    let result = find_config_for_model_file(&dir.path().join("xyz.safetensors"));
    assert!(result.is_some());
    let name = result.unwrap().file_name().unwrap().to_str().unwrap().to_string();
    assert_eq!(name, "xyz.config.json", "Should prefer hash-prefix over plain config.json");
}
