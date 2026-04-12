mod falsification_b {
    use super::*;

    #[test]
    fn f_prov_logic_005_empty_converter() {
        let mut prov = sample_provenance();
        prov.derived[0].converter = String::new();

        let result = validate_provenance(&prov);

        // CORROBORATED: Empty string != "apr-cli"
        assert!(result.is_err());
    }

    /// F-PROV-LOGIC-006: Converter with whitespace prefix
    #[test]
    fn f_prov_logic_006_whitespace_converter() {
        let mut prov = sample_provenance();
        prov.derived[0].converter = " apr-cli".to_string(); // Leading space

        let result = validate_provenance(&prov);

        // CORROBORATED: " apr-cli" != "apr-cli"
        assert!(result.is_err());
    }

    // ====================================================================
    // Vector C: Workflow Sabotage
    // ====================================================================

    /// F-PROV-FLOW-001: The "Time Traveler" - future timestamp
    #[test]
    fn f_prov_flow_001_future_timestamp() {
        let mut prov = sample_provenance();
        prov.source.downloaded_at = "2099-12-31T23:59:59Z".to_string();

        // OBSERVATION: No timestamp validation exists
        let result = validate_provenance(&prov);
        assert!(result.is_ok());

        // FALSIFIED: Future timestamps accepted without warning
        // Minor issue - could indicate clock manipulation
    }

    /// F-PROV-FLOW-002: The "Race Condition" - concurrent writes
    #[test]
    fn f_prov_flow_002_race_condition() {
        use std::sync::Arc;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let dir_path = Arc::new(dir.path().to_path_buf());

        let mut handles = vec![];

        for i in 0..10 {
            let path = Arc::clone(&dir_path);
            handles.push(thread::spawn(move || {
                let mut prov = sample_provenance();
                prov.source.sha256 = format!("hash_{i}");
                save_provenance(&path, &prov).unwrap();
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Load and verify - should have ONE consistent state
        let loaded = load_provenance(&dir_path).unwrap();

        // OBSERVATION: No file locking, last writer wins
        // Result is non-deterministic but at least valid JSON
        assert!(validate_provenance(&loaded).is_ok());

        // CORROBORATED (with caveat): No corruption, but no atomicity guarantee
        // Recommend: atomic write (write to .tmp, then rename)
    }

    /// F-PROV-FLOW-003: The "Version Spoof" - empty version
    #[test]
    fn f_prov_flow_003_empty_version() {
        let dir = tempfile::tempdir().unwrap();
        let safetensors = dir.path().join("model.safetensors");
        let derived = dir.path().join("model.gguf");
        std::fs::write(&safetensors, "source").unwrap();
        std::fs::write(&derived, "derived").unwrap();

        let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();

        // Add derived with empty version
        add_derived(&mut prov, "gguf", &derived, None, "").unwrap();

        // OBSERVATION: Empty version string is accepted
        assert!(validate_provenance(&prov).is_ok());
        assert!(prov.derived[0].converter_version.is_empty());

        // FALSIFIED: No validation that version is non-empty
        // Minor issue - could indicate tampering
    }

    // ====================================================================
    // Vector D: Code Mutation (White Box)
    // ====================================================================

    /// F-PROV-CODE-001: Add same derived model twice
    /// Expected: add_derived() rejects duplicate (PROV-008)
    #[test]
    fn f_prov_code_001_duplicate_derived() {
        let dir = tempfile::tempdir().unwrap();
        let safetensors = dir.path().join("model.safetensors");
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&safetensors, "source").unwrap();
        std::fs::write(&gguf, "derived").unwrap();

        let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();

        // Add GGUF first time - succeeds
        add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();
        assert_eq!(prov.derived.len(), 1);

        // FIX VERIFIED: Second add fails with DuplicateDerived error
        let result = add_derived(&mut prov, "gguf", &gguf, None, "0.2.12");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            Error::Provenance(ProvenanceError::DuplicateDerived { .. })
        ));

        // Still only one entry
        assert_eq!(prov.derived.len(), 1);
    }

    /// F-PROV-CODE-001b: Different quantization = not a duplicate
    #[test]
    fn f_prov_code_001b_different_quantization_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let safetensors = dir.path().join("model.safetensors");
        let gguf = dir.path().join("model.gguf");
        let gguf_q4 = dir.path().join("model-q4.gguf");
        std::fs::write(&safetensors, "source").unwrap();
        std::fs::write(&gguf, "derived").unwrap();
        std::fs::write(&gguf_q4, "derived q4").unwrap();

        let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();

        // Add unquantized GGUF
        add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();

        // Add quantized GGUF - different quantization = allowed
        add_derived(&mut prov, "gguf", &gguf_q4, Some("q4_k_m"), "0.2.12").unwrap();

        // Both entries exist
        assert_eq!(prov.derived.len(), 2);
    }

    /// F-PROV-CODE-002: Feed non-provenance JSON
    #[test]
    fn f_prov_code_002_wrong_json_schema() {
        let dir = tempfile::tempdir().unwrap();
        let prov_path = dir.path().join(".provenance.json");

        // Write valid JSON but wrong schema
        std::fs::write(&prov_path, r#"{"hello": "world"}"#).unwrap();

        let result = load_provenance(dir.path());

        // CORROBORATED: Returns deserialization error
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::SerializationError(_)));
    }

    /// F-PROV-CODE-003: Unicode in fields
    #[test]
    fn f_prov_code_003_unicode_injection() {
        let mut prov = sample_provenance();
        prov.source.hf_repo = "test/模型".to_string(); // Chinese characters
        prov.derived[0].path = "model_مدل.gguf".to_string(); // Arabic

        // Should handle unicode gracefully
        let result = validate_provenance(&prov);
        assert!(result.is_ok());

        // Serialize and deserialize
        let json = serde_json::to_string(&prov).unwrap();
        let loaded: Provenance = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.source.hf_repo, "test/模型");

        // CORROBORATED: Unicode handled correctly
    }

    /// F-PROV-CODE-004: Very long strings (buffer overflow attempt)
    #[test]
    fn f_prov_code_004_long_strings() {
        let mut prov = sample_provenance();
        prov.source.sha256 = "a".repeat(1_000_000); // 1MB hash string

        // Should not panic or OOM on validation
        let result = validate_provenance(&prov);
        assert!(result.is_ok());

        // CORROBORATED: Long strings handled (though invalid hash)
        // Note: No hash format validation exists
    }

    /// Verify QuantizationMismatch error Display includes both format-quant pairs
    #[test]
    fn test_quantization_mismatch_display() {
        let err = ProvenanceError::QuantizationMismatch {
            format_a: "gguf".to_string(),
            quant_a: Some("q4_k_m".to_string()),
            format_b: "apr".to_string(),
            quant_b: None,
        };
        let msg = err.to_string();
        assert!(msg.contains("PROV-005"));
        assert!(msg.contains("gguf"));
        assert!(msg.contains("apr"));
        assert!(msg.contains("q4_k_m"));
    }

    /// Verify InvalidConverter error Display includes format and converter name
    #[test]
    fn test_invalid_converter_display() {
        let err = ProvenanceError::InvalidConverter {
            format: "gguf".to_string(),
            converter: "bartowski".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("PROV-002"));
        assert!(msg.contains("gguf"));
        assert!(msg.contains("bartowski"));
        assert!(msg.contains("apr-cli"));
    }

    /// Verify MissingProvenance error Display includes the path
    #[test]
    fn test_missing_provenance_display() {
        let err = ProvenanceError::MissingProvenance {
            path: "/model/dir".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("PROV-004"));
        assert!(msg.contains("/model/dir"));
    }

    /// Verify DuplicateDerived error Display with None quantization
    #[test]
    fn test_duplicate_derived_display_no_quant() {
        let err = ProvenanceError::DuplicateDerived {
            format: "gguf".to_string(),
            quantization: None,
        };
        let msg = err.to_string();
        assert!(msg.contains("PROV-008"));
        assert!(msg.contains("gguf"));
        assert!(msg.contains("None"));
    }

    /// Verify DuplicateDerived error Display with Some quantization
    #[test]
    fn test_duplicate_derived_display_with_quant() {
        let err = ProvenanceError::DuplicateDerived {
            format: "gguf".to_string(),
            quantization: Some("q4_k_m".to_string()),
        };
        let msg = err.to_string();
        assert!(msg.contains("PROV-008"));
        assert!(msg.contains("gguf"));
        assert!(msg.contains("q4_k_m"));
    }

    /// F-PROV-CODE-005: Null bytes in strings
    #[test]
    fn f_prov_code_005_null_bytes() {
        let mut prov = sample_provenance();
        prov.source.path = "model\0.safetensors".to_string();

        // Should handle embedded nulls
        let result = validate_provenance(&prov);
        assert!(result.is_ok());

        let json = serde_json::to_string(&prov).unwrap();
        let loaded: Provenance = serde_json::from_str(&json).unwrap();
        assert!(loaded.source.path.contains('\0'));

        // CORROBORATED: Null bytes preserved (could be path traversal risk)
    }

    /// F-PROV-CODE-006: Empty derived list
    /// Expected: validate_comparison() fails for missing formats (PROV-009)
    #[test]
    fn f_prov_code_006_empty_derived() {
        let mut prov = sample_provenance();
        prov.derived.clear();

        // Provenance with no derived formats is valid (source only)
        let result = validate_provenance(&prov);
        assert!(result.is_ok());

        // FIX VERIFIED: Comparison fails when formats don't exist
        let cmp_result = validate_comparison(&prov, "gguf", "apr");
        assert!(cmp_result.is_err());
        assert!(matches!(
            cmp_result.unwrap_err(),
            ProvenanceError::FormatNotFound { .. }
        ));
    }

    /// F-PROV-CODE-007: Comparison with non-existent format
    /// Expected: validate_comparison() fails with FormatNotFound (PROV-009)
    #[test]
    fn f_prov_code_007_phantom_format_comparison() {
        let prov = sample_provenance();

        // Compare format that doesn't exist
        let result = validate_comparison(&prov, "gguf", "phantom_format");

        // FIX VERIFIED: Returns FormatNotFound error
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ProvenanceError::FormatNotFound { format } if format == "phantom_format")
        );
    }

    // ====================================================================
    // New Tests for Verification Functions
    // ====================================================================

    /// Test verify_provenance_integrity with valid files
    #[test]
    fn test_verify_integrity_valid() {
        let dir = tempfile::tempdir().unwrap();
        let safetensors = dir.path().join("model.safetensors");
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&safetensors, "source content").unwrap();
        std::fs::write(&gguf, "gguf content").unwrap();

        let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();
        add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();

        // All files exist and hashes match
        let result = verify_provenance_integrity(&prov, dir.path());
        assert!(result.is_ok());
    }

    /// Test verify_provenance_integrity detects modified source
    #[test]
    fn test_verify_integrity_modified_source() {
        let dir = tempfile::tempdir().unwrap();
        let safetensors = dir.path().join("model.safetensors");
        std::fs::write(&safetensors, "original content").unwrap();

        let prov = create_source_provenance(&safetensors, "test/model").unwrap();

        // Modify the source file after provenance creation
        std::fs::write(&safetensors, "MODIFIED content").unwrap();

        // Integrity check fails
        let result = verify_provenance_integrity(&prov, dir.path());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ProvenanceError::HashMismatch { .. }
        ));
    }

    /// Test verify_provenance_integrity detects modified derived file
    #[test]
    fn test_verify_integrity_modified_derived() {
        let dir = tempfile::tempdir().unwrap();
        let safetensors = dir.path().join("model.safetensors");
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&safetensors, "source").unwrap();
        std::fs::write(&gguf, "original gguf").unwrap();

        let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();
        add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();

        // Modify derived file
        std::fs::write(&gguf, "TAMPERED gguf").unwrap();

        let result = verify_provenance_integrity(&prov, dir.path());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ProvenanceError::HashMismatch { .. }
        ));
    }

    /// Test verify_files_exist with all files present
    #[test]
    fn test_verify_files_exist_valid() {
        let dir = tempfile::tempdir().unwrap();
        let safetensors = dir.path().join("model.safetensors");
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&safetensors, "source").unwrap();
        std::fs::write(&gguf, "gguf").unwrap();

        let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();
        add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();

        assert!(verify_files_exist(&prov, dir.path()).is_ok());
    }

    /// Test verify_files_exist detects missing derived file
    #[test]
    fn test_verify_files_exist_missing_derived() {
        let dir = tempfile::tempdir().unwrap();
        let safetensors = dir.path().join("model.safetensors");
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&safetensors, "source").unwrap();
        std::fs::write(&gguf, "gguf").unwrap();

        let mut prov = create_source_provenance(&safetensors, "test/model").unwrap();
        add_derived(&mut prov, "gguf", &gguf, None, "0.2.12").unwrap();

        // Delete derived file
        std::fs::remove_file(&gguf).unwrap();

        let result = verify_files_exist(&prov, dir.path());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ProvenanceError::FileMissing { .. }
        ));
    }

    /// Test new error display messages
    #[test]
    fn test_new_error_displays() {
        let err = ProvenanceError::HashMismatch {
            path: "model.gguf".to_string(),
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
        };
        assert!(err.to_string().contains("PROV-006"));

        let err = ProvenanceError::FileMissing {
            path: "model.safetensors".to_string(),
        };
        assert!(err.to_string().contains("PROV-007"));

        let err = ProvenanceError::DuplicateDerived {
            format: "gguf".to_string(),
            quantization: Some("q4_k_m".to_string()),
        };
        assert!(err.to_string().contains("PROV-008"));

        let err = ProvenanceError::FormatNotFound {
            format: "phantom".to_string(),
        };
        assert!(err.to_string().contains("PROV-009"));
    }
}

