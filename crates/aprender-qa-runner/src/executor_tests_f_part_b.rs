#[test]
fn test_setup_source_links_single_file() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("model.safetensors");
    std::fs::write(&source, b"fake model data").unwrap();
    let st_dir = tmp.path().join("workspace_st");
    std::fs::create_dir_all(&st_dir).unwrap();

    let err = Executor::setup_source_links(&source, &st_dir, false);
    assert!(err.is_none(), "Expected no error, got: {err:?}");
    // The symlink should exist
    assert!(st_dir.join("model.safetensors").exists());
}

#[test]
fn test_setup_source_links_sharded() {
    let tmp = tempfile::tempdir().unwrap();
    let source_dir = tmp.path().join("source");
    std::fs::create_dir_all(&source_dir).unwrap();
    let index_file = source_dir.join("model.safetensors.index.json");
    std::fs::write(&index_file, b"{}").unwrap();
    std::fs::write(
        source_dir.join("model-00001-of-00002.safetensors"),
        b"shard1",
    )
    .unwrap();
    std::fs::write(
        source_dir.join("model-00002-of-00002.safetensors"),
        b"shard2",
    )
    .unwrap();

    let st_dir = tmp.path().join("workspace_st");
    std::fs::create_dir_all(&st_dir).unwrap();

    let err = Executor::setup_source_links(&index_file, &st_dir, true);
    assert!(err.is_none(), "Expected no error, got: {err:?}");
    // All files from source should be symlinked
    assert!(st_dir.join("model.safetensors.index.json").exists());
}

// ── resolve_sharded_index ───────────────────────────────────────────

#[test]
fn test_resolve_sharded_index_safetensors_format() {
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );
    let path = Path::new("/cache/model.safetensors.index.json");
    let result =
        Executor::resolve_sharded_index(path, "/cache/model.safetensors.index.json", &scenario);
    assert_eq!(
        result,
        Some("/cache/model.safetensors.index.json".to_string())
    );
}

#[test]
fn test_resolve_sharded_index_gguf_format() {
    let tmp = tempfile::tempdir().unwrap();
    let index = tmp.path().join("model.safetensors.index.json");
    std::fs::write(&index, b"{}").unwrap();
    // Put a gguf file in the same directory
    std::fs::write(tmp.path().join("model.gguf"), b"fake gguf").unwrap();

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_sharded_index(&index, &index.to_string_lossy(), &scenario);
    // Should find sibling .gguf file
    assert!(result.is_some());
    assert!(result.unwrap().contains("gguf"));
}

// ── resolve_file_model ──────────────────────────────────────────────

#[test]
fn test_resolve_file_model_matching_extension() {
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_file_model(
        Path::new("/path/model.gguf"),
        "/path/model.gguf",
        "gguf",
        &scenario,
    );
    assert_eq!(result, Some("/path/model.gguf".to_string()));
}

#[test]
fn test_resolve_file_model_safetensors_match() {
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_file_model(
        Path::new("/path/model.safetensors"),
        "/path/model.safetensors",
        "safetensors",
        &scenario,
    );
    assert_eq!(result, Some("/path/model.safetensors".to_string()));
}

#[test]
fn test_resolve_file_model_apr_match() {
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_file_model(
        Path::new("/path/model.apr"),
        "/path/model.apr",
        "apr",
        &scenario,
    );
    assert_eq!(result, Some("/path/model.apr".to_string()));
}

#[test]
fn test_resolve_file_model_mismatch_with_sibling() {
    let tmp = tempfile::tempdir().unwrap();
    let gguf_file = tmp.path().join("model.gguf");
    let apr_file = tmp.path().join("model.apr");
    std::fs::write(&gguf_file, b"fake gguf").unwrap();
    std::fs::write(&apr_file, b"fake apr").unwrap();

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "test".to_string(),
        0,
    );
    let result =
        Executor::resolve_file_model(&gguf_file, &gguf_file.to_string_lossy(), "gguf", &scenario);
    // Should find sibling .apr file
    assert!(result.is_some());
    assert!(result.unwrap().contains("apr"));
}

// ── resolve_directory_model ─────────────────────────────────────────

#[test]
fn test_resolve_directory_model_apr_cache_structure() {
    let tmp = tempfile::tempdir().unwrap();
    let gguf_dir = tmp.path().join("gguf");
    std::fs::create_dir_all(&gguf_dir).unwrap();
    std::fs::write(gguf_dir.join("model.gguf"), b"fake").unwrap();

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_directory_model(tmp.path(), &scenario);
    assert!(result.is_some());
    assert!(result.unwrap().contains("gguf"));
}

#[test]
fn test_resolve_directory_model_flat_hf_structure() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model.safetensors"), b"fake").unwrap();

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_directory_model(tmp.path(), &scenario);
    assert!(result.is_some());
    assert!(result.unwrap().contains("model.safetensors"));
}

#[test]
fn test_resolve_directory_model_sharded_safetensors() {
    let tmp = tempfile::tempdir().unwrap();
    let st_dir = tmp.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).unwrap();
    std::fs::write(st_dir.join("model.safetensors.index.json"), b"{}").unwrap();

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_directory_model(tmp.path(), &scenario);
    assert!(result.is_some());
    assert!(result.unwrap().contains("model.safetensors.index.json"));
}

#[test]
fn test_resolve_directory_model_nothing_found() {
    let tmp = tempfile::tempdir().unwrap();
    // Empty directory - nothing to find
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let result = Executor::resolve_directory_model(tmp.path(), &scenario);
    assert!(result.is_none());
}

// ── find_clean_model_file ───────────────────────────────────────────

#[test]
fn test_find_clean_model_file_skips_artifacts() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model-converted.gguf"), b"artifact").unwrap();
    std::fs::write(tmp.path().join("model.idem.gguf"), b"artifact").unwrap();
    std::fs::write(tmp.path().join("model.com_q4k.gguf"), b"artifact").unwrap();
    std::fs::write(tmp.path().join("model.rt_q6k.gguf"), b"artifact").unwrap();
    std::fs::write(tmp.path().join("model.gguf"), b"clean").unwrap();

    let result = Executor::find_clean_model_file(tmp.path(), "gguf");
    assert!(result.is_some());
    let found = result.unwrap();
    assert!(found.contains("model.gguf"));
    assert!(!found.contains("converted"));
    assert!(!found.contains("idem"));
    assert!(!found.contains("com_"));
    assert!(!found.contains("rt_"));
}

#[test]
fn test_find_clean_model_file_no_files() {
    let tmp = tempfile::tempdir().unwrap();
    let result = Executor::find_clean_model_file(tmp.path(), "gguf");
    assert!(result.is_none());
}

#[test]
fn test_find_clean_model_file_nonexistent_dir() {
    let result = Executor::find_clean_model_file(Path::new("/nonexistent"), "gguf");
    assert!(result.is_none());
}

// ── find_sibling_model_files ────────────────────────────────────────

#[test]
fn test_find_sibling_model_files_pacha_cache() {
    let tmp = tempfile::tempdir().unwrap();
    let model_file = tmp.path().join("abc123.safetensors");
    std::fs::write(&model_file, b"model data").unwrap();
    std::fs::write(tmp.path().join("abc123.config.json"), b"config").unwrap();
    std::fs::write(tmp.path().join("abc123.tokenizer.json"), b"tokenizer").unwrap();
    std::fs::write(tmp.path().join("other_file.txt"), b"unrelated").unwrap();

    let siblings = Executor::find_sibling_model_files(&model_file);
    assert_eq!(siblings.len(), 2);
    let canonical_names: Vec<&str> = siblings.iter().map(|(_, n)| n.as_str()).collect();
    assert!(canonical_names.contains(&"config.json"));
    assert!(canonical_names.contains(&"tokenizer.json"));
}

#[test]
fn test_find_sibling_model_files_flat_hf_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let model_file = tmp.path().join("model.safetensors");
    std::fs::write(&model_file, b"model data").unwrap();
    std::fs::write(tmp.path().join("config.json"), b"config").unwrap();
    std::fs::write(tmp.path().join("tokenizer.json"), b"tokenizer").unwrap();
    std::fs::write(tmp.path().join("random.txt"), b"unrelated").unwrap();

    let siblings = Executor::find_sibling_model_files(&model_file);
    assert!(siblings.len() >= 2);
    let canonical_names: Vec<&str> = siblings.iter().map(|(_, n)| n.as_str()).collect();
    assert!(canonical_names.contains(&"config.json"));
    assert!(canonical_names.contains(&"tokenizer.json"));
}

// ── print_fail_fast_diagnostics ─────────────────────────────────────

#[test]
fn test_print_fail_fast_diagnostics_no_model_path() {
    let config = ExecutionConfig {
        model_path: None,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let evidence = Evidence::falsified("G2-BASIC", test_scenario(), "test failure", "output", 0);
    // Should not panic, just prints to stderr
    executor.print_fail_fast_diagnostics(&evidence, "test-playbook");
}

#[test]
fn test_print_fail_fast_diagnostics_with_stderr_and_exit_code() {
    let config = ExecutionConfig {
        model_path: None,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let mut evidence =
        Evidence::falsified("G2-BASIC", test_scenario(), "test failure", "output", 0);
    evidence.stderr = Some("Error message in stderr".to_string());
    evidence.exit_code = Some(42);
    // Should print stderr and exit code
    executor.print_fail_fast_diagnostics(&evidence, "test-playbook");
}

// ── format_tensor_failure ───────────────────────────────────────────

#[test]
fn test_format_tensor_failure_without_expected_actual() {
    let result = crate::layout_contract::TensorValidationResult {
        tensor_name: "lm_head.weight".to_string(),
        rule_id: "R001".to_string(),
        passed: false,
        details: "tensor not found".to_string(),
        expected: None,
        actual: None,
    };
    let formatted = Executor::format_tensor_failure(&result);
    assert!(formatted.contains("R001"));
    assert!(formatted.contains("tensor not found"));
    assert!(!formatted.contains("Expected:"));
}

// ── scenario creation helpers (unique tests) ────────────────────────

#[test]
fn test_format_scenario_creation_apr() {
    let model_id = ModelId::new("org", "name");
    let scenario = Executor::format_scenario(&model_id, Format::Apr);
    assert_eq!(scenario.format, Format::Apr);
    assert!(scenario.prompt.contains("Format"));
}

#[test]
fn test_format_scenario_creation_gguf() {
    let model_id = ModelId::new("org", "name");
    let scenario = Executor::format_scenario(&model_id, Format::Gguf);
    assert_eq!(scenario.format, Format::Gguf);
}

#[test]
fn test_hf_parity_scenario_truncates_long_prompt() {
    let model_id = ModelId::new("org", "name");
    let long_prompt = "A".repeat(100);
    let scenario = Executor::hf_parity_scenario(&model_id, &long_prompt);
    assert_eq!(scenario.format, Format::Apr);
    // Prompt should be truncated to 40 chars
    assert!(scenario.prompt.len() < 100);
}

// ── has_safetensors_files (unique) ──────────────────────────────────

#[test]
fn test_has_safetensors_files_with_st_among_others() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("model.safetensors"), b"data").unwrap();
    std::fs::write(tmp.path().join("config.json"), b"{}").unwrap();
    std::fs::write(tmp.path().join("model.gguf"), b"gguf").unwrap();
    assert!(Executor::has_safetensors_files(tmp.path()));
}

// ── find_safetensors_dir (unique) ───────────────────────────────────

#[test]
fn test_find_safetensors_dir_prefers_subdir() {
    let tmp = tempfile::tempdir().unwrap();
    // Create both a subdir with safetensors and direct safetensors
    let st_dir = tmp.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).unwrap();
    std::fs::write(st_dir.join("model.safetensors"), b"subdir").unwrap();
    std::fs::write(tmp.path().join("model.safetensors"), b"direct").unwrap();
    let result = Executor::find_safetensors_dir(tmp.path());
    assert!(result.is_some());
    // Should prefer the subdir
    assert!(result.unwrap().to_string_lossy().contains("safetensors"));
}

// ── find_model_by_prefix (unique) ───────────────────────────────────

#[test]
fn test_find_model_by_prefix_case_insensitive() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("Qwen2.5-Coder-7b-q4k.gguf"), b"data").unwrap();
    let result = Executor::find_model_by_prefix(tmp.path(), "qwen2.5-coder-7b", "gguf");
    assert!(result.is_some());
}
