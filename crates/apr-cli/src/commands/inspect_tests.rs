// PMAT-540 Phase 5: Tests for inspect command helper functions

#[cfg(test)]
mod inspect_tests {
    use super::*;
    use std::path::Path;

    // ========================================================================
    // validate_path
    // ========================================================================

    #[test]
    fn validate_path_nonexistent() {
        let result = validate_path(Path::new("/nonexistent/model.apr"));
        assert!(result.is_err());
        match result.unwrap_err() {
            CliError::FileNotFound(_) => {}
            e => panic!("Expected FileNotFound, got {e:?}"),
        }
    }

    #[test]
    fn validate_path_directory() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let result = validate_path(dir.path());
        assert!(result.is_err());
        match result.unwrap_err() {
            CliError::NotAFile(_) => {}
            e => panic!("Expected NotAFile, got {e:?}"),
        }
    }

    #[test]
    fn validate_path_valid_file() {
        let file = tempfile::NamedTempFile::new().expect("create temp file");
        let result = validate_path(file.path());
        assert!(result.is_ok());
    }

    // ========================================================================
    // InspectResult JSON serialization
    // ========================================================================

    #[test]
    fn inspect_result_json_serialization() {
        let result = InspectResult {
            file: "model.apr".to_string(),
            valid: true,
            format: "APR v2".to_string(),
            version: "2.0".to_string(),
            tensor_count: 100,
            size_bytes: 1_000_000,
            checksum_valid: true,
            architecture: Some("llama".to_string()),
            num_layers: Some(32),
            num_heads: Some(32),
            hidden_size: Some(4096),
            vocab_size: Some(128256),
            flags: FlagsInfo {
                lz4_compressed: false,
                zstd_compressed: false,
                encrypted: false,
                signed: false,
                sharded: false,
                quantized: true,
                has_vocab: true,
            },
            metadata: MetadataInfo {
                architecture: Some("llama".to_string()),
                ..MetadataInfo::default()
            },
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("model.apr"));
        assert!(json.contains("\"valid\":true"));
        assert!(json.contains("\"tensor_count\":100"));
        assert!(json.contains("\"architecture\":\"llama\""));
        assert!(json.contains("\"quantized\":true"));
    }

    #[test]
    fn inspect_result_skips_none_fields() {
        let result = InspectResult {
            file: "test.apr".to_string(),
            valid: false,
            format: "unknown".to_string(),
            version: "0".to_string(),
            tensor_count: 0,
            size_bytes: 0,
            checksum_valid: false,
            architecture: None,
            num_layers: None,
            num_heads: None,
            hidden_size: None,
            vocab_size: None,
            flags: FlagsInfo {
                lz4_compressed: false,
                zstd_compressed: false,
                encrypted: false,
                signed: false,
                sharded: false,
                quantized: false,
                has_vocab: false,
            },
            metadata: MetadataInfo::default(),
        };
        let json = serde_json::to_string(&result).expect("serialize");
        // Top-level architecture (on InspectResult) has skip_serializing_if
        // but metadata.architecture does NOT — it serializes as null
        assert!(!json.contains("\"num_layers\""), "None num_layers should be skipped");
        assert!(!json.contains("\"hidden_size\""), "None hidden_size should be skipped");
    }

    // ========================================================================
    // FlagsInfo
    // ========================================================================

    #[test]
    fn flags_info_all_false() {
        let flags = FlagsInfo {
            lz4_compressed: false,
            zstd_compressed: false,
            encrypted: false,
            signed: false,
            sharded: false,
            quantized: false,
            has_vocab: false,
        };
        let json = serde_json::to_string(&flags).expect("serialize");
        assert!(json.contains("\"lz4_compressed\":false"));
    }
}
