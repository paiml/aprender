
// ============================================================================
// Write/Import/Lint Coverage Tests (T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests_write_import_lint {
    use super::*;
    use crate::format::gguf::{GgufModelConfig, GgufTokenizer};
    use crate::format::lint::{lint_apr_file, LintLevel};
    use crate::format::test_factory::build_pygmy_apr;
    use std::fs;
    use tempfile::TempDir;

    // ------------------------------------------------------------------------
    // GgufTokenizer tests (improve gguf/api.rs coverage)
    // ------------------------------------------------------------------------

    #[test]
    fn test_gguf_tokenizer_empty_merges() {
        let tok = GgufTokenizer {
            vocabulary: vec!["a".to_string(), "b".to_string()],
            merges: vec![],
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            architecture: None,
            model_name: None,
            ..Default::default()
        };
        assert!(tok.has_vocabulary());
        assert_eq!(tok.vocab_size(), 2);
        assert!(tok.merges.is_empty());
    }

    #[test]
    fn test_gguf_tokenizer_with_merges() {
        let tok = GgufTokenizer {
            vocabulary: vec!["hello".to_string()],
            merges: vec!["h e".to_string(), "he l".to_string(), "hel lo".to_string()],
            model_type: Some("bpe".to_string()),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            architecture: Some("llama".to_string()),
            model_name: Some("test".to_string()),
            ..Default::default()
        };
        assert_eq!(tok.merges.len(), 3);
        assert_eq!(tok.bos_token_id, Some(1));
        assert_eq!(tok.eos_token_id, Some(2));
    }

    // ------------------------------------------------------------------------
    // GgufModelConfig tests (improve gguf/api.rs coverage)
    // ------------------------------------------------------------------------

    #[test]
    fn test_gguf_model_config_partial() {
        let cfg = GgufModelConfig {
            architecture: Some("llama".to_string()),
            hidden_size: Some(2048),
            num_layers: None,
            num_heads: None,
            num_kv_heads: None,
            vocab_size: None,
            intermediate_size: None,
            max_position_embeddings: None,
            rope_theta: None,
            rms_norm_eps: None,
            rope_type: None,
        };
        assert_eq!(cfg.architecture.as_deref(), Some("llama"));
        assert_eq!(cfg.hidden_size, Some(2048));
        assert!(cfg.num_layers.is_none());
    }

    #[test]
    fn test_gguf_model_config_phi_style() {
        let cfg = GgufModelConfig {
            architecture: Some("phi".to_string()),
            hidden_size: Some(2560),
            num_layers: Some(32),
            num_heads: Some(32),
            num_kv_heads: Some(32),
            vocab_size: Some(51200),
            intermediate_size: Some(10240),
            max_position_embeddings: Some(2048),
            rope_theta: Some(10000.0),
            rms_norm_eps: Some(1e-5),
            rope_type: Some(0),
        };
        assert_eq!(cfg.architecture.as_deref(), Some("phi"));
        assert_eq!(cfg.num_heads, Some(32));
    }

    #[test]
    fn test_gguf_model_config_mistral_style() {
        let cfg = GgufModelConfig {
            architecture: Some("mistral".to_string()),
            hidden_size: Some(4096),
            num_layers: Some(32),
            num_heads: Some(32),
            num_kv_heads: Some(8), // GQA
            vocab_size: Some(32000),
            intermediate_size: Some(14336),
            max_position_embeddings: Some(32768),
            rope_theta: Some(1000000.0),
            rms_norm_eps: Some(1e-5),
            rope_type: Some(0),
        };
        assert_eq!(cfg.num_kv_heads, Some(8));
        assert!(cfg.rope_theta.unwrap() > 100000.0);
    }

    // ------------------------------------------------------------------------
    // Lint tests (improve lint/mod.rs coverage)
    // ------------------------------------------------------------------------

    #[test]
    fn test_lint_level_variants() {
        // LintLevel: Info, Warn, Error
        let levels = [LintLevel::Error, LintLevel::Warn, LintLevel::Info];
        for level in &levels {
            let debug_str = format!("{level:?}");
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_lint_apr_file_v2_format_support() {
        // lint_apr_file now supports both v1 (APRN) and v2 (APR\0) formats
        // LAYOUT-CONTRACT-001: Updated to support unified format linting
        let temp_dir = TempDir::new().expect("Create temp dir");
        let apr_path = temp_dir.path().join("v2_model.apr");

        let apr_data = build_pygmy_apr();
        fs::write(&apr_path, &apr_data).expect("Write APR");

        // V2 format should now be supported (fixed as part of LAYOUT-CONTRACT-001)
        let result = lint_apr_file(&apr_path);
        assert!(result.is_ok(), "V2 APR should now be linted successfully");
        let report = result.expect("Lint report");
        // Pygmy models have missing metadata by design
        assert!(report.warn_count > 0, "Should have metadata warnings");
    }

    #[test]
    fn test_lint_apr_file_not_found() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let missing_path = temp_dir.path().join("missing.apr");

        let result = lint_apr_file(&missing_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_lint_apr_file_invalid_magic() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let bad_path = temp_dir.path().join("bad.apr");

        // Write file with wrong magic bytes
        fs::write(&bad_path, b"BADD").expect("Write bad file");

        let result = lint_apr_file(&bad_path);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------------
    // Quantization type tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_quantization_type_all_variants() {
        // Test all actual QuantizationType variants
        let types = [
            QuantizationType::Int8,
            QuantizationType::Int4,
            QuantizationType::Fp16,
            QuantizationType::Q4K,
        ];
        for qt in &types {
            let debug_str = format!("{qt:?}");
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_quantization_type_clone_eq() {
        let qt = QuantizationType::Int8;
        let cloned = qt.clone();
        assert_eq!(qt, cloned);

        let qt2 = QuantizationType::Q4K;
        assert_ne!(qt, qt2);
    }

    #[test]
    fn test_quantization_type_fp16_variant() {
        let qt = QuantizationType::Fp16;
        assert!(format!("{qt:?}").contains("Fp16"));
    }

    #[test]
    fn test_quantization_type_q4k_variant() {
        let qt = QuantizationType::Q4K;
        assert!(format!("{qt:?}").contains("Q4K"));
    }
}
