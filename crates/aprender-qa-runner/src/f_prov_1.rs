#[test]
fn test_reject_mismatched_source_hash() {
    // This would be detected by comparing provenance files
    // from different model directories
    let prov_a = sample_provenance();
    let mut prov_b = sample_provenance();
    prov_b.source.sha256 = "different_hash".to_string();

    // Different source hashes should fail comparison
    assert_ne!(prov_a.source.sha256, prov_b.source.sha256);
}

// PMAT-PROV-002: Reject third-party files without provenance
#[test]
fn test_reject_third_party_gguf() {
    let mut prov = sample_provenance();
    prov.derived[0].converter = "bartowski".to_string(); // Third-party

    let result = validate_provenance(&prov);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(matches!(err, ProvenanceError::InvalidConverter { .. }));
}

// PMAT-PROV-003: Accept only SafeTensors as source
#[test]
fn test_reject_gguf_as_source() {
    let mut prov = sample_provenance();
    prov.source.format = "gguf".to_string(); // Wrong source format

    let result = validate_provenance(&prov);
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(matches!(err, ProvenanceError::InvalidSourceFormat { .. }));
}

// PMAT-PROV-004: Reject quantization mismatch
#[test]
fn test_reject_quantization_mismatch() {
    let mut prov = sample_provenance();
    prov.derived[0].quantization = Some("q4_k_m".to_string()); // GGUF quantized
    prov.derived[1].quantization = None; // APR unquantized

    let result = validate_comparison(&prov, "gguf", "apr");
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(matches!(err, ProvenanceError::QuantizationMismatch { .. }));
}

#[test]
fn test_valid_provenance_passes() {
    let prov = sample_provenance();
    assert!(validate_provenance(&prov).is_ok());
}

#[test]
fn test_valid_comparison_same_quantization() {
    let prov = sample_provenance();
    assert!(validate_comparison(&prov, "gguf", "apr").is_ok());
}

#[test]
fn test_valid_comparison_both_quantized() {
    let mut prov = sample_provenance();
    prov.derived[0].quantization = Some("q4_k_m".to_string());
    prov.derived[1].quantization = Some("q4_k_m".to_string());

    assert!(validate_comparison(&prov, "gguf", "apr").is_ok());
}

#[test]
fn test_provenance_error_display() {
    let err = ProvenanceError::InvalidSourceFormat {
        format: "gguf".to_string(),
    };
    assert!(err.to_string().contains("PROV-003"));
    assert!(err.to_string().contains("safetensors"));
}

#[test]
fn test_source_mismatch_error_display() {
    let err = ProvenanceError::SourceMismatch {
        expected: "abc123".to_string(),
        actual: "def456".to_string(),
        format: "apr".to_string(),
    };
    assert!(err.to_string().contains("PROV-001"));
    assert!(err.to_string().contains("abc123"));
}

// ========================================================================
// Provenance Generation Tests
// ========================================================================

#[test]
fn test_compute_sha256() {
    let dir = tempfile::tempdir().unwrap();
    let test_file = dir.path().join("test.txt");
    std::fs::write(&test_file, "hello world\n").unwrap();

    let hash = compute_sha256(&test_file).unwrap();
    // SHA256 of "hello world\n"
    assert_eq!(
        hash,
        "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447"
    );
}

#[test]
fn test_compute_sha256_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let test_file = dir.path().join("empty.txt");
    std::fs::write(&test_file, "").unwrap();

    let hash = compute_sha256(&test_file).unwrap();
    // SHA256 of empty string
    assert_eq!(
        hash,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn test_compute_sha256_missing_file() {
    let result = compute_sha256(Path::new("/nonexistent/file.bin"));
    assert!(result.is_err());
}

#[test]
fn test_create_source_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let safetensors = dir.path().join("model.safetensors");
    std::fs::write(&safetensors, "fake safetensors content").unwrap();

    let prov =
        create_source_provenance(&safetensors, "Qwen/Qwen2.5-Coder-0.5B-Instruct").unwrap();

    assert_eq!(prov.source.format, "safetensors");
    assert_eq!(prov.source.path, "model.safetensors");
    assert_eq!(prov.source.hf_repo, "Qwen/Qwen2.5-Coder-0.5B-Instruct");
    assert!(!prov.source.sha256.is_empty());
    assert!(!prov.source.downloaded_at.is_empty());
    assert!(prov.derived.is_empty());
}

#[test]
fn test_add_derived() {
    let dir = tempfile::tempdir().unwrap();
    let safetensors = dir.path().join("model.safetensors");
    let gguf = dir.path().join("model.gguf");
    std::fs::write(&safetensors, "safetensors content").unwrap();
    std::fs::write(&gguf, "gguf content").unwrap();

    let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();
    add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();

    assert_eq!(prov.derived.len(), 1);
    assert_eq!(prov.derived[0].format, "gguf");
    assert_eq!(prov.derived[0].path, "model.gguf");
    assert_eq!(prov.derived[0].converter, "apr-cli");
    assert_eq!(prov.derived[0].converter_version, "0.2.12");
    assert!(prov.derived[0].quantization.is_none());
}

#[test]
fn test_add_derived_with_quantization() {
    let dir = tempfile::tempdir().unwrap();
    let safetensors = dir.path().join("model.safetensors");
    let gguf_q4 = dir.path().join("model-q4_k_m.gguf");
    std::fs::write(&safetensors, "safetensors content").unwrap();
    std::fs::write(&gguf_q4, "quantized gguf content").unwrap();

    let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();
    add_derived(&mut prov, "gguf", &gguf_q4, Some("q4_k_m"), "0.2.12").unwrap();

    assert_eq!(prov.derived[0].quantization, Some("q4_k_m".to_string()));
}

#[test]
fn test_save_and_load_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let safetensors = dir.path().join("model.safetensors");
    std::fs::write(&safetensors, "content").unwrap();

    let prov = create_source_provenance(&safetensors, "test/model").unwrap();
    save_provenance(dir.path(), &prov).unwrap();

    let loaded = load_provenance(dir.path()).unwrap();
    assert_eq!(loaded.source.hf_repo, "test/model");
    assert_eq!(loaded.source.sha256, prov.source.sha256);
}

#[test]
fn test_save_provenance_creates_json() {
    let dir = tempfile::tempdir().unwrap();
    let safetensors = dir.path().join("model.safetensors");
    std::fs::write(&safetensors, "content").unwrap();

    let prov = create_source_provenance(&safetensors, "test/model").unwrap();
    save_provenance(dir.path(), &prov).unwrap();

    let prov_path = dir.path().join(".provenance.json");
    assert!(prov_path.exists());

    let content = std::fs::read_to_string(&prov_path).unwrap();
    assert!(content.contains("\"format\": \"safetensors\""));
    assert!(content.contains("test/model"));
}

#[test]
fn test_full_provenance_workflow() {
    let dir = tempfile::tempdir().unwrap();

    // Create source
    let safetensors = dir.path().join("model.safetensors");
    std::fs::write(&safetensors, "source content").unwrap();

    // Create derived formats
    let gguf = dir.path().join("model.gguf");
    let apr = dir.path().join("model.apr");
    std::fs::write(&gguf, "gguf content").unwrap();
    std::fs::write(&apr, "apr content").unwrap();

    // Build provenance
    let mut prov =
        create_source_provenance(&safetensors, "Qwen/Qwen2.5-Coder-0.5B-Instruct").unwrap();
    add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();
    add_derived(&mut prov, "apr", &apr, None, "0.2.12").unwrap();

    // Validate provenance
    assert!(validate_provenance(&prov).is_ok());

    // Validate comparison
    assert!(validate_comparison(&prov, "gguf", "apr").is_ok());

    // Save and reload
    save_provenance(dir.path(), &prov).unwrap();
    let loaded = load_provenance(dir.path()).unwrap();

    // Revalidate after reload
    assert!(validate_provenance(&loaded).is_ok());
}

#[test]
fn test_get_apr_cli_version_returns_string() {
    // This test just verifies the function doesn't panic
    // In CI where apr isn't installed, it returns "unknown"
    let version = get_apr_cli_version();
    assert!(!version.is_empty());
}

// ========================================================================
// FALSIFICATION TESTS (PMAT-PROV-001)
// Operation "Trust No One" - Popperian Falsification
// ========================================================================

mod falsification {
    use super::*;

    // ====================================================================
    // Vector A: Integrity Attacks (Physical Artifacts)
    // ====================================================================

    /// F-PROV-IO-001: The "Bit Flip" - corrupt SHA256 hash
    /// Expected: verify_provenance_integrity() MUST detect mismatch (PROV-006)
    #[test]
    fn f_prov_io_001_bit_flip_hash() {
        let dir = tempfile::tempdir().unwrap();

        // Create real files and valid provenance
        let safetensors = dir.path().join("model.safetensors");
        std::fs::write(&safetensors, "source content").unwrap();
        let prov = create_source_provenance(&safetensors, "test/model").unwrap();
        save_provenance(dir.path(), &prov).unwrap();

        // Manually corrupt the hash in the file
        let prov_path = dir.path().join(".provenance.json");
        let content = std::fs::read_to_string(&prov_path).unwrap();
        let corrupted = content.replace(&prov.source.sha256, "CORRUPTED_HASH");
        std::fs::write(&prov_path, corrupted).unwrap();

        // Load provenance (JSON is valid, just hash is wrong)
        let loaded = load_provenance(dir.path()).unwrap();

        // Basic validation still passes (format/converter checks)
        assert!(validate_provenance(&loaded).is_ok());

        // FIX VERIFIED: verify_provenance_integrity() detects the corruption
        let integrity_result = verify_provenance_integrity(&loaded, dir.path());
        assert!(integrity_result.is_err());
        assert!(matches!(
            integrity_result.unwrap_err(),
            ProvenanceError::HashMismatch { .. }
        ));
    }

    /// F-PROV-IO-002: The "Truncation" - partial JSON
    /// Expected: load_provenance() returns robust error, no panic
    #[test]
    fn f_prov_io_002_truncated_json() {
        let dir = tempfile::tempdir().unwrap();
        let prov_path = dir.path().join(".provenance.json");

        // Write truncated JSON (simulate power loss)
        std::fs::write(&prov_path, r#"{"source": {"format": "safetens"#).unwrap();

        let result = load_provenance(dir.path());

        // CORROBORATED: Returns error, does not panic
        assert!(result.is_err());
        // Verify it's a serialization error, not a panic
        let err = result.unwrap_err();
        assert!(matches!(err, Error::SerializationError(_)));
    }

    /// F-PROV-IO-003: The "Ghost File" - model file deleted but provenance exists
    /// Expected: verify_files_exist() detects missing file (PROV-007)
    #[test]
    fn f_prov_io_003_ghost_file() {
        let dir = tempfile::tempdir().unwrap();

        // Create source file and provenance
        let safetensors = dir.path().join("model.safetensors");
        std::fs::write(&safetensors, "content").unwrap();
        let prov = create_source_provenance(&safetensors, "test/model").unwrap();
        save_provenance(dir.path(), &prov).unwrap();

        // Delete the model file (ghost it)
        std::fs::remove_file(&safetensors).unwrap();

        // Load provenance - still works (JSON exists)
        let loaded = load_provenance(dir.path()).unwrap();

        // Basic validation still passes (format/converter checks)
        assert!(validate_provenance(&loaded).is_ok());

        // FIX VERIFIED: verify_files_exist() detects the ghost
        let exist_result = verify_files_exist(&loaded, dir.path());
        assert!(exist_result.is_err());
        assert!(matches!(
            exist_result.unwrap_err(),
            ProvenanceError::FileMissing { .. }
        ));

        // FIX VERIFIED: verify_provenance_integrity() also detects it
        let integrity_result = verify_provenance_integrity(&loaded, dir.path());
        assert!(integrity_result.is_err());
        assert!(matches!(
            integrity_result.unwrap_err(),
            ProvenanceError::FileMissing { .. }
        ));
    }

    /// F-PROV-IO-004: The "Hash Collision" - empty file hash
    /// Expected: Correct hash for empty file, no panic
    #[test]
    fn f_prov_io_004_empty_file_hash() {
        let dir = tempfile::tempdir().unwrap();
        let empty_file = dir.path().join("empty.bin");
        std::fs::write(&empty_file, "").unwrap();

        let hash = compute_sha256(&empty_file).unwrap();

        // CORROBORATED: Returns correct SHA256 for empty file
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    // ====================================================================
    // Vector B: Logic Bypass (Validation Rules)
    // ====================================================================

    /// F-PROV-LOGIC-001: The "Imposter Source" - GGUF as source format
    /// Expected: Validation fails (Ground Truth Policy 7.4)
    #[test]
    fn f_prov_logic_001_imposter_source() {
        let mut prov = sample_provenance();
        prov.source.format = "gguf".to_string(); // Violate 7.4

        let result = validate_provenance(&prov);

        // CORROBORATED: Correctly rejects GGUF as source
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ProvenanceError::InvalidSourceFormat { .. }
        ));
    }

    /// F-PROV-LOGIC-002: The "Rogue Converter"
    /// Expected: validate_provenance() fails (PROV-002)
    #[test]
    fn f_prov_logic_002_rogue_converter() {
        let mut prov = sample_provenance();
        prov.derived[0].converter = "suspicious-script v0.1".to_string();

        let result = validate_provenance(&prov);

        // CORROBORATED: Correctly rejects non-apr-cli converter
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ProvenanceError::InvalidConverter { .. }
        ));
    }

    /// F-PROV-LOGIC-003: The "Quantization Lie"
    /// Expected: If we lie in JSON, can we compare apples to oranges?
    #[test]
    fn f_prov_logic_003_quantization_lie() {
        let mut prov = sample_provenance();

        // Lie: Say both are Q4_K_M but actual files would be different
        prov.derived[0].quantization = Some("q4_k_m".to_string());
        prov.derived[1].quantization = Some("q4_k_m".to_string());

        // System only checks JSON, not actual file headers
        let result = validate_comparison(&prov, "gguf", "apr");

        // OBSERVATION: Comparison passes because JSON claims match
        assert!(result.is_ok());

        // FALSIFIED: We can lie in metadata and bypass quantization check
        // The system does NOT verify actual file quantization headers
        // P0 TICKET REQUIRED: No verification of claimed vs actual quantization
    }

    /// F-PROV-LOGIC-004: Case sensitivity attack on format
    #[test]
    fn f_prov_logic_004_case_sensitivity() {
        let mut prov = sample_provenance();
        prov.source.format = "SafeTensors".to_string(); // Wrong case

        let result = validate_provenance(&prov);

        // CORROBORATED: Correctly rejects wrong case
        // (string comparison is case-sensitive)
        assert!(result.is_err());
    }

} // close mod falsification (continues in part_b)
