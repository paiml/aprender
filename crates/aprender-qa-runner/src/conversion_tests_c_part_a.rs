/// Verify Unknown quant type falls back to F32 tolerance
#[test]
fn test_tolerance_for_unknown_falls_back_to_f32() {
    let tol = tolerance_for(QuantType::Unknown);
    assert!((tol.atol - 1e-6).abs() < 1e-10);
}

/// Verify effective_epsilon returns default EPSILON without quant type
#[test]
fn test_effective_epsilon_without_quant() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    assert!((test.effective_epsilon() - EPSILON).abs() < f64::EPSILON);
}

/// Verify effective_epsilon returns Q4KM tolerance when quant is set
#[test]
fn test_effective_epsilon_with_quant() {
    let mut test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    test.quant_type = Some(QuantType::Q4KM);
    assert!((test.effective_epsilon() - 1e-1).abs() < 1e-10);
}

/// Verify effective_epsilon returns F32 tolerance for F32 quant type
#[test]
fn test_effective_epsilon_f32_quant() {
    let mut test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    test.quant_type = Some(QuantType::F32);
    assert!((test.effective_epsilon() - 1e-6).abs() < 1e-10);
}

// ── ConversionFailureType / TensorNaming serde tests ───────────────

/// Verify all ConversionFailureType variants survive serde round-trip
#[test]
fn test_conversion_failure_type_serde() {
    let types = [
        ConversionFailureType::TensorNameMismatch,
        ConversionFailureType::DequantizationFailure,
        ConversionFailureType::ConfigMetadataMismatch,
        ConversionFailureType::MissingArtifact,
        ConversionFailureType::InferenceFailure,
        ConversionFailureType::Unknown,
    ];
    for ft in types {
        let json = serde_json::to_string(&ft).unwrap();
        let parsed: ConversionFailureType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, ft);
    }
}

/// Verify each ConversionFailureType maps to the correct gate ID
#[test]
fn test_conversion_failure_type_gate_ids() {
    assert_eq!(
        ConversionFailureType::TensorNameMismatch.gate_id(),
        "F-CONV-TNAME-001"
    );
    assert_eq!(
        ConversionFailureType::DequantizationFailure.gate_id(),
        "F-CONV-DEQUANT-001"
    );
    assert_eq!(
        ConversionFailureType::ConfigMetadataMismatch.gate_id(),
        "F-CONV-CONFIG-001"
    );
    assert_eq!(
        ConversionFailureType::MissingArtifact.gate_id(),
        "F-CONV-MISSING-001"
    );
    assert_eq!(
        ConversionFailureType::InferenceFailure.gate_id(),
        "F-CONV-INFER-001"
    );
    assert_eq!(
        ConversionFailureType::Unknown.gate_id(),
        "F-CONV-UNKNOWN-002"
    );
}

/// Verify each ConversionFailureType maps to the correct key string
#[test]
fn test_conversion_failure_type_keys() {
    assert_eq!(
        ConversionFailureType::TensorNameMismatch.key(),
        "tensor_name_mismatch"
    );
    assert_eq!(ConversionFailureType::Unknown.key(), "unknown");
}

/// Verify all QuantType variants survive serde round-trip
#[test]
fn test_quant_type_serde() {
    let types = [
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q4KM,
        QuantType::Q5KM,
        QuantType::Q6K,
        QuantType::Q4_0,
        QuantType::Q8_0,
        QuantType::Unknown,
    ];
    for qt in types {
        let json = serde_json::to_string(&qt).unwrap();
        let parsed: QuantType = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, qt);
    }
}

/// Verify all TensorNaming variants survive serde round-trip
#[test]
fn test_tensor_naming_serde() {
    let variants = [
        TensorNaming::HuggingFace,
        TensorNaming::Gguf,
        TensorNaming::Apr,
        TensorNaming::Unknown("custom".to_string()),
    ];
    for tn in &variants {
        let json = serde_json::to_string(tn).unwrap();
        let parsed: TensorNaming = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, *tn);
    }
}

/// Verify ConversionEvidence serialization preserves failure_type and quant_type
#[test]
fn test_conversion_evidence_with_failure_type() {
    let evidence = ConversionEvidence {
        source_hash: "a".to_string(),
        converted_hash: "b".to_string(),
        max_diff: 0.5,
        diff_indices: vec![],
        source_format: Format::Gguf,
        target_format: Format::Apr,
        backend: Backend::Cpu,
        failure_type: Some(ConversionFailureType::TensorNameMismatch),
        quant_type: Some(QuantType::Q4KM),
    };
    let json = serde_json::to_string(&evidence).unwrap();
    let parsed: ConversionEvidence = serde_json::from_str(&json).unwrap();
    assert_eq!(
        parsed.failure_type,
        Some(ConversionFailureType::TensorNameMismatch)
    );
    assert_eq!(parsed.quant_type, Some(QuantType::Q4KM));
}

/// Verify optional fields default to None when absent from JSON
#[test]
fn test_conversion_evidence_default_optional_fields() {
    // Deserialize without optional fields — should default to None
    let json = r#"{
            "source_hash": "a",
            "converted_hash": "b",
            "max_diff": 0.1,
            "diff_indices": [],
            "source_format": "gguf",
            "target_format": "apr",
            "backend": "cpu"
        }"#;
    let parsed: ConversionEvidence = serde_json::from_str(json).unwrap();
    assert!(parsed.failure_type.is_none());
    assert!(parsed.quant_type.is_none());
}

/// Verify DEFAULT_TOLERANCES has entries for all 8 quant types
#[test]
fn test_default_tolerances_count() {
    assert_eq!(DEFAULT_TOLERANCES.len(), 8);
}

// =========================================================================
// Model Path Resolution Tests
// =========================================================================

/// Verify resolve_model_path finds model in APR cache subdirectory structure
#[test]
fn test_resolve_model_path_apr_cache_structure() {
    let tmp = tempfile::TempDir::new().unwrap();
    let safetensors_dir = tmp.path().join("safetensors");
    std::fs::create_dir_all(&safetensors_dir).unwrap();
    let model_file = safetensors_dir.join("model.safetensors");
    std::fs::write(&model_file, b"fake").unwrap();

    let result = resolve_model_path(tmp.path(), Format::SafeTensors);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), model_file);
}

/// Verify resolve_model_path finds model in HF cache flat structure
#[test]
fn test_resolve_model_path_hf_cache_flat_structure() {
    let tmp = tempfile::TempDir::new().unwrap();
    // HF cache has model.safetensors directly in snapshot dir (flat)
    let model_file = tmp.path().join("model.safetensors");
    std::fs::write(&model_file, b"fake").unwrap();

    let result = resolve_model_path(tmp.path(), Format::SafeTensors);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), model_file);
}

/// Verify resolve_model_path accepts a direct file path
#[test]
fn test_resolve_model_path_file_mode() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_file = tmp.path().join("model.gguf");
    std::fs::write(&model_file, b"fake").unwrap();

    let result = resolve_model_path(&model_file, Format::Gguf);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), model_file);
}

/// Verify resolve_model_path rejects file with wrong extension
#[test]
fn test_resolve_model_path_file_mode_extension_mismatch() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_file = tmp.path().join("model.gguf");
    std::fs::write(&model_file, b"fake").unwrap();

    let result = resolve_model_path(&model_file, Format::SafeTensors);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("extension mismatch")
    );
}

/// Verify resolve_model_path returns error when no matching file exists
#[test]
fn test_resolve_model_path_not_found() {
    let tmp = tempfile::TempDir::new().unwrap();
    let result = resolve_model_path(tmp.path(), Format::Apr);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("No apr file found")
    );
}

/// Verify resolve_model_path finds non-standard filenames in subdirectory
#[test]
fn test_resolve_model_path_any_extension_in_subdir() {
    let tmp = tempfile::TempDir::new().unwrap();
    let gguf_dir = tmp.path().join("gguf");
    std::fs::create_dir_all(&gguf_dir).unwrap();
    // Not model.gguf but something.gguf
    let model_file = gguf_dir.join("qwen-0.5b-q4.gguf");
    std::fs::write(&model_file, b"fake").unwrap();

    let result = resolve_model_path(tmp.path(), Format::Gguf);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), model_file);
}

/// Verify resolve_model_path finds non-standard filenames in base directory
#[test]
fn test_resolve_model_path_any_extension_in_base_dir() {
    let tmp = tempfile::TempDir::new().unwrap();
    // HF cache might have different names
    let model_file = tmp.path().join("qwen2.5-coder-0.5b.safetensors");
    std::fs::write(&model_file, b"fake").unwrap();

    let result = resolve_model_path(tmp.path(), Format::SafeTensors);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), model_file);
}

/// Verify DEFAULT_TOLERANCES covers all known quant types
#[test]
fn test_default_tolerances_all_quant_types() {
    let types: Vec<QuantType> = DEFAULT_TOLERANCES.iter().map(|t| t.quant_type).collect();
    assert!(types.contains(&QuantType::F32));
    assert!(types.contains(&QuantType::F16));
    assert!(types.contains(&QuantType::BF16));
    assert!(types.contains(&QuantType::Q4KM));
    assert!(types.contains(&QuantType::Q5KM));
    assert!(types.contains(&QuantType::Q6K));
    assert!(types.contains(&QuantType::Q4_0));
    assert!(types.contains(&QuantType::Q8_0));
}

// =========================================================================
// HuggingFace Cache Resolution Tests (HF-CACHE-001, HF-CACHE-002)
// =========================================================================

/// Verify split_hf_repo separates org/repo correctly
#[test]
fn test_split_hf_repo_with_org() {
    assert_eq!(
        split_hf_repo("Qwen/Qwen2.5-Coder-0.5B"),
        ("Qwen", "Qwen2.5-Coder-0.5B")
    );
    assert_eq!(
        split_hf_repo("meta-llama/Llama-2-7b"),
        ("meta-llama", "Llama-2-7b")
    );
}

/// Verify split_hf_repo defaults org to "unknown" when no slash present
#[test]
fn test_split_hf_repo_without_org() {
    assert_eq!(split_hf_repo("model-only"), ("unknown", "model-only"));
    assert_eq!(split_hf_repo("gpt2"), ("unknown", "gpt2"));
}

/// Verify split_hf_repo only splits on first slash
#[test]
fn test_split_hf_repo_multiple_slashes() {
    // Only splits on first slash
    assert_eq!(split_hf_repo("org/repo/extra"), ("org", "repo/extra"));
}

/// Verify split_hf_repo handles empty string input
#[test]
fn test_split_hf_repo_empty_string() {
    assert_eq!(split_hf_repo(""), ("unknown", ""));
}

/// Verify find_hf_snapshot locates snapshot directory with model file
#[test]
fn test_find_hf_snapshot_found() {
    let tmp = tempfile::TempDir::new().unwrap();
    let snapshot = tmp
        .path()
        .join("models--Test--Model")
        .join("snapshots")
        .join("abc123");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("model.safetensors"), b"fake").unwrap();

    let result = find_hf_snapshot(tmp.path(), "Test", "Model");
    assert!(result.is_some());
    assert_eq!(result.unwrap(), snapshot);
}

/// Verify find_hf_snapshot returns None when model directory is absent
#[test]
fn test_find_hf_snapshot_not_found_no_dir() {
    let tmp = tempfile::TempDir::new().unwrap();
    let result = find_hf_snapshot(tmp.path(), "Missing", "Model");
    assert!(result.is_none());
}

/// Verify find_hf_snapshot returns None when snapshot has no safetensors
#[test]
fn test_find_hf_snapshot_not_found_no_safetensors() {
    let tmp = tempfile::TempDir::new().unwrap();
    let snapshot = tmp
        .path()
        .join("models--Test--NoFile")
        .join("snapshots")
        .join("abc123");
    std::fs::create_dir_all(&snapshot).unwrap();
    // No model.safetensors file

    let result = find_hf_snapshot(tmp.path(), "Test", "NoFile");
    assert!(result.is_none());
}

/// Verify find_apr_cache locates APR cache directory
#[test]
fn test_find_apr_cache_found() {
    let tmp = tempfile::TempDir::new().unwrap();
    let apr_cache = tmp.path().join(".cache/apr-models/TestOrg/TestRepo");
    std::fs::create_dir_all(&apr_cache).unwrap();

    let result = find_apr_cache(tmp.path(), "TestOrg", "TestRepo");
    assert!(result.is_some());
    assert_eq!(result.unwrap(), apr_cache);
}

/// Verify find_apr_cache returns None when cache directory is absent
#[test]
fn test_find_apr_cache_not_found() {
    let tmp = tempfile::TempDir::new().unwrap();
    let result = find_apr_cache(tmp.path(), "Missing", "Model");
    assert!(result.is_none());
}

/// Verify resolve_hf_repo_with_dirs finds model in HF cache first
#[test]
fn test_resolve_hf_repo_with_dirs_found_in_hf_cache() {
    let tmp = tempfile::TempDir::new().unwrap();
    let snapshot = tmp
        .path()
        .join("models--Test--Model")
        .join("snapshots")
        .join("abc123");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("model.safetensors"), b"fake").unwrap();

    // Use the temp dir as both HF cache and home
    let result = resolve_hf_repo_with_dirs("Test/Model", tmp.path(), tmp.path());
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), snapshot);
}

/// Verify resolve_hf_repo_with_dirs falls back to APR cache
#[test]
fn test_resolve_hf_repo_with_dirs_found_in_apr_cache() {
    let tmp = tempfile::TempDir::new().unwrap();
    let apr_cache = tmp.path().join(".cache/apr-models/TestOrg/TestRepo");
    std::fs::create_dir_all(&apr_cache).unwrap();

    // HF cache is empty, APR cache has the model
    let hf_cache = tmp.path().join("hf_empty");
    std::fs::create_dir_all(&hf_cache).unwrap();

    let result = resolve_hf_repo_with_dirs("TestOrg/TestRepo", &hf_cache, tmp.path());
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), apr_cache);
}

/// Verify resolve_hf_repo_with_dirs returns error when model is in neither cache
#[test]
fn test_resolve_hf_repo_with_dirs_not_found() {
    let tmp = tempfile::TempDir::new().unwrap();
    let hf_cache = tmp.path().join("hf_empty");
    let home = tmp.path().join("home_empty");
    std::fs::create_dir_all(&hf_cache).unwrap();
    std::fs::create_dir_all(&home).unwrap();

    let result = resolve_hf_repo_with_dirs("Missing/Model", &hf_cache, &home);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("not found in cache"));
    assert!(err_msg.contains("Missing/Model"));
}

/// Verify resolve_hf_repo_with_dirs fails when snapshot lacks safetensors
#[test]
fn test_resolve_hf_repo_with_dirs_snapshot_without_safetensors() {
    let tmp = tempfile::TempDir::new().unwrap();
    let snapshot = tmp
        .path()
        .join("models--Test--NoSafetensors")
        .join("snapshots")
        .join("abc123");
    std::fs::create_dir_all(&snapshot).unwrap();
    // No model.safetensors

    let home = tmp.path().join("home_empty");
    std::fs::create_dir_all(&home).unwrap();

    let result = resolve_hf_repo_with_dirs("Test/NoSafetensors", tmp.path(), &home);
    assert!(result.is_err());
}
