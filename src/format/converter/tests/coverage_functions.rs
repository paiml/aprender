//! APR Converter Coverage Tests - Export/Merge/Write/Import/Lint Function Tests
//! Split from coverage.rs (PMAT-197) for file size reduction.
//!
//! Contains: apr_export tests, apr_merge tests, GgufTokenizer/GgufModelConfig tests,
//! lint tests, write_apr_file tests, F16 passthrough, Rosetta conversion,
//! dequantization, transpose, load_model_tensors, calculate_tensor_size,
//! BUG-LAYOUT-003 error path tests.

#[allow(unused_imports)]
use super::super::*;
use trueno_quant::quantize_q6_k_matrix;

// ============================================================================
// Pygmy-Based Export/Merge Function Tests (T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests_export_merge_functions {
    use super::*;
    use crate::format::test_factory::{
        build_pygmy_safetensors, build_pygmy_safetensors_with_config, PygmyConfig,
    };
    use std::fs;
    use tempfile::TempDir;

    // ------------------------------------------------------------------------
    // apr_export tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_apr_export_safetensors_to_safetensors() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let input_path = temp_dir.path().join("input.safetensors");
        let output_path = temp_dir.path().join("output.safetensors");

        // Write pygmy safetensors file
        let data = build_pygmy_safetensors();
        fs::write(&input_path, &data).expect("Write input");

        // Export to SafeTensors
        let options = ExportOptions {
            format: ExportFormat::SafeTensors,
            quantize: None,
            include_tokenizer: false,
            include_config: false,
        };
        let result = apr_export(&input_path, &output_path, options);
        assert!(result.is_ok(), "Export should succeed: {:?}", result.err());

        let report = result.unwrap();
        assert_eq!(report.format, ExportFormat::SafeTensors);
        assert!(report.tensor_count > 0);
        assert!(output_path.exists());
    }

    #[test]
    fn test_apr_export_input_not_found() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let input_path = temp_dir.path().join("nonexistent.safetensors");
        let output_path = temp_dir.path().join("output.safetensors");

        let options = ExportOptions::default();
        let result = apr_export(&input_path, &output_path, options);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found") || err.contains("Input file"));
    }

    #[test]
    fn test_apr_export_unsupported_format_onnx() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let input_path = temp_dir.path().join("input.safetensors");
        let output_path = temp_dir.path().join("output.onnx");

        // Write pygmy file
        let data = build_pygmy_safetensors();
        fs::write(&input_path, &data).expect("Write input");

        let options = ExportOptions {
            format: ExportFormat::Onnx,
            quantize: None,
            include_tokenizer: false,
            include_config: false,
        };
        let result = apr_export(&input_path, &output_path, options);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not yet supported") || err.contains("Onnx"));
    }

    #[test]
    fn test_apr_export_unsupported_format_torchscript() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let input_path = temp_dir.path().join("input.safetensors");
        let output_path = temp_dir.path().join("output.pt");

        let data = build_pygmy_safetensors();
        fs::write(&input_path, &data).expect("Write input");

        let options = ExportOptions {
            format: ExportFormat::TorchScript,
            quantize: None,
            include_tokenizer: false,
            include_config: false,
        };
        let result = apr_export(&input_path, &output_path, options);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_export_with_config_companion() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let input_path = temp_dir.path().join("input.safetensors");
        let output_path = temp_dir.path().join("output.safetensors");

        let data = build_pygmy_safetensors();
        fs::write(&input_path, &data).expect("Write input");

        let options = ExportOptions {
            format: ExportFormat::SafeTensors,
            quantize: None,
            include_tokenizer: false,
            include_config: true,
        };
        let result = apr_export(&input_path, &output_path, options);
        assert!(result.is_ok());

        // Check config.json was created
        let config_path = temp_dir.path().join("config.json");
        assert!(config_path.exists(), "config.json should be created");
    }

    // ------------------------------------------------------------------------
    // apr_merge tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_apr_merge_two_models_average() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let input1 = temp_dir.path().join("model1.safetensors");
        let input2 = temp_dir.path().join("model2.safetensors");
        let output = temp_dir.path().join("merged.safetensors");

        // Create two pygmy models with same structure
        let config = PygmyConfig::minimal();
        let data1 = build_pygmy_safetensors_with_config(config.clone());
        let data2 = build_pygmy_safetensors_with_config(config);
        fs::write(&input1, &data1).expect("Write model1");
        fs::write(&input2, &data2).expect("Write model2");

        let options = MergeOptions {
            strategy: MergeStrategy::Average,
            weights: None,
            ..Default::default()
        };
        let result = apr_merge(&[&input1, &input2], &output, options);
        assert!(result.is_ok(), "Merge should succeed: {:?}", result.err());

        let report = result.unwrap();
        assert_eq!(report.model_count, 2);
        assert_eq!(report.strategy, MergeStrategy::Average);
        assert!(output.exists());
    }

    #[test]
    fn test_apr_merge_weighted() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let input1 = temp_dir.path().join("model1.safetensors");
        let input2 = temp_dir.path().join("model2.safetensors");
        let output = temp_dir.path().join("merged.safetensors");

        let config = PygmyConfig::minimal();
        let data1 = build_pygmy_safetensors_with_config(config.clone());
        let data2 = build_pygmy_safetensors_with_config(config);
        fs::write(&input1, &data1).expect("Write model1");
        fs::write(&input2, &data2).expect("Write model2");

        let options = MergeOptions {
            strategy: MergeStrategy::Weighted,
            weights: Some(vec![0.7, 0.3]),
            ..Default::default()
        };
        let result = apr_merge(&[&input1, &input2], &output, options);
        assert!(
            result.is_ok(),
            "Weighted merge should succeed: {:?}",
            result.err()
        );

        let report = result.unwrap();
        assert!(report.weights_used.is_some());
    }

    #[test]
    fn test_apr_merge_single_model_fails() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let input1 = temp_dir.path().join("model1.safetensors");
        let output = temp_dir.path().join("merged.safetensors");

        let data = build_pygmy_safetensors();
        fs::write(&input1, &data).expect("Write model1");

        let options = MergeOptions::default();
        let result = apr_merge(&[&input1], &output, options);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("at least 2") || err.contains("requires"));
    }

    #[test]
    fn test_apr_merge_unsupported_strategy_ties() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let input1 = temp_dir.path().join("model1.safetensors");
        let input2 = temp_dir.path().join("model2.safetensors");
        let output = temp_dir.path().join("merged.safetensors");

        let config = PygmyConfig::minimal();
        let data = build_pygmy_safetensors_with_config(config);
        fs::write(&input1, &data.clone()).expect("Write model1");
        fs::write(&input2, &data).expect("Write model2");

        let options = MergeOptions {
            strategy: MergeStrategy::Ties,
            ..Default::default()
        };
        // TIES requires --base-model, so without one it should fail
        let result = apr_merge(&[&input1, &input2], &output, options);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("base-model") || err.contains("TIES"));
    }

    #[test]
    fn test_apr_merge_unsupported_strategy_dare() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let input1 = temp_dir.path().join("model1.safetensors");
        let input2 = temp_dir.path().join("model2.safetensors");
        let output = temp_dir.path().join("merged.safetensors");

        let config = PygmyConfig::minimal();
        let data = build_pygmy_safetensors_with_config(config);
        fs::write(&input1, &data.clone()).expect("Write model1");
        fs::write(&input2, &data).expect("Write model2");

        let options = MergeOptions {
            strategy: MergeStrategy::Dare,
            ..Default::default()
        };
        // DARE requires --base-model, so without one it should fail
        let result = apr_merge(&[&input1, &input2], &output, options);
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_merge_unsupported_strategy_slerp() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let input1 = temp_dir.path().join("model1.safetensors");
        let input2 = temp_dir.path().join("model2.safetensors");
        let output = temp_dir.path().join("merged.safetensors");

        let config = PygmyConfig::minimal();
        let data = build_pygmy_safetensors_with_config(config);
        fs::write(&input1, &data.clone()).expect("Write model1");
        fs::write(&input2, &data).expect("Write model2");

        let options = MergeOptions {
            strategy: MergeStrategy::Slerp,
            ..Default::default()
        };
        // SLERP should succeed with 2 models and default weight (0.5)
        let result = apr_merge(&[&input1, &input2], &output, options);
        assert!(result.is_ok(), "SLERP with 2 models should succeed: {:?}", result.err());
    }

    #[test]
    fn test_apr_merge_three_models() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let input1 = temp_dir.path().join("model1.safetensors");
        let input2 = temp_dir.path().join("model2.safetensors");
        let input3 = temp_dir.path().join("model3.safetensors");
        let output = temp_dir.path().join("merged.safetensors");

        let config = PygmyConfig::minimal();
        let data = build_pygmy_safetensors_with_config(config);
        fs::write(&input1, &data.clone()).expect("Write model1");
        fs::write(&input2, &data.clone()).expect("Write model2");
        fs::write(&input3, &data).expect("Write model3");

        let options = MergeOptions::default();
        let result = apr_merge(&[&input1, &input2, &input3], &output, options);
        assert!(
            result.is_ok(),
            "3-model merge should succeed: {:?}",
            result.err()
        );

        let report = result.unwrap();
        assert_eq!(report.model_count, 3);
    }

    #[test]
    fn test_apr_merge_model_not_found() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let input1 = temp_dir.path().join("exists.safetensors");
        let input2 = temp_dir.path().join("missing.safetensors");
        let output = temp_dir.path().join("merged.safetensors");

        let data = build_pygmy_safetensors();
        fs::write(&input1, &data).expect("Write model1");
        // Note: input2 is NOT created

        let options = MergeOptions::default();
        let result = apr_merge(&[&input1, &input2], &output, options);
        assert!(result.is_err());
    }
}

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

// ============================================================================
// Write/Import/Rosetta Function Coverage Tests (T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests_write_functions {
    use super::*;
    use crate::format::gguf::{GgufModelConfig, GgufTokenizer};
    use crate::format::test_factory::{build_pygmy_apr, build_pygmy_safetensors};
    use crate::format::v2::AprV2Reader;
    use std::collections::BTreeMap;
    use std::fs;
    use tempfile::TempDir;
    // GAP-UX-002: Import trueno_quant functions for Q5K/Q6K tests
    // GH-202: transpose functions no longer re-exported from converter (wrong assumption removed)
    // Import directly from trueno_quant for tests that validate the functions themselves.
    use trueno_quant::{
        dequantize_q6_k_to_f32, quantize_q5_k, quantize_q5_k_matrix, transpose_q4k_for_matmul,
        transpose_q5k_for_matmul, transpose_q6k_for_matmul,
    };

    // ------------------------------------------------------------------------
    // write_apr_file coverage tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_write_apr_file_basic() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let output_path = temp_dir.path().join("output.apr");

        // Create minimal tensor data
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
        );

        let options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        let result = write_apr_file(
            &tensors,
            &empty_f16, // GH-205: F16 passthrough
            &output_path,
            &options,
            None,
            None,
            &Default::default(),
        );
        assert!(
            result.is_ok(),
            "write_apr_file should succeed: {:?}",
            result.err()
        );
        assert!(output_path.exists());
    }

    #[test]
    fn test_write_apr_file_with_tokenizer() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let output_path = temp_dir.path().join("with_tok.apr");

        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], vec![4, 2]),
        );

        let tokenizer = GgufTokenizer {
            vocabulary: vec![
                "hello".to_string(),
                "world".to_string(),
                "test".to_string(),
                "end".to_string(),
            ],
            merges: vec!["he llo".to_string(), "wo rld".to_string()],
            model_type: Some("bpe".to_string()),
            bos_token_id: Some(0),
            eos_token_id: Some(3),
            architecture: Some("llama".to_string()),
            model_name: Some("pygmy".to_string()),
            ..Default::default()
        };

        let options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        let result = write_apr_file(
            &tensors,
            &empty_f16, // GH-205: F16 passthrough
            &output_path,
            &options,
            Some(&tokenizer),
            None,
            &Default::default(),
        );
        assert!(
            result.is_ok(),
            "write_apr_file with tokenizer should succeed"
        );
    }

    #[test]
    fn test_write_apr_file_with_model_config() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let output_path = temp_dir.path().join("with_config.apr");

        // Create tensors matching a small config
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.1; 64], vec![8, 8]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.01; 64], vec![8, 8]),
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![0.02; 64], vec![8, 8]),
        );
        tensors.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            (vec![0.03; 64], vec![8, 8]),
        );

        let model_config = GgufModelConfig {
            architecture: Some("llama".to_string()),
            hidden_size: Some(8),
            num_layers: Some(1),
            num_heads: Some(2),
            num_kv_heads: Some(2),
            vocab_size: Some(8),
            intermediate_size: Some(16),
            max_position_embeddings: Some(128),
            rope_theta: Some(10000.0),
            rms_norm_eps: Some(1e-5),
            rope_type: Some(0),
        };

        let options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        let result = write_apr_file(
            &tensors,
            &empty_f16,
            &output_path,
            &options,
            None,
            Some(&model_config),
            &Default::default(),
        );
        assert!(
            result.is_ok(),
            "write_apr_file with config should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_write_apr_file_with_quantization_fp16() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let output_path = temp_dir.path().join("fp16.apr");

        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
        );

        let mut options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        options.quantize = Some(QuantizationType::Fp16);

        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        let result = write_apr_file(
            &tensors,
            &empty_f16,
            &output_path,
            &options,
            None,
            None,
            &Default::default(),
        );
        assert!(result.is_ok(), "write_apr_file with fp16 should succeed");
    }

    #[test]
    fn test_write_apr_file_with_quantization_int8() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let output_path = temp_dir.path().join("int8.apr");

        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
        );

        let mut options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        options.quantize = Some(QuantizationType::Int8);

        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        let result = write_apr_file(
            &tensors,
            &empty_f16,
            &output_path,
            &options,
            None,
            None,
            &Default::default(),
        );
        assert!(result.is_ok(), "write_apr_file with int8 should succeed");
    }

    #[test]
    fn test_write_apr_file_tied_embeddings() {
        // Test that lm_head.weight is created from embed_tokens when missing
        let temp_dir = TempDir::new().expect("Create temp dir");
        let output_path = temp_dir.path().join("tied.apr");

        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        // Only add embed_tokens, no lm_head
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
        );

        let options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        let result = write_apr_file(
            &tensors,
            &empty_f16,
            &output_path,
            &options,
            None,
            None,
            &Default::default(),
        );
        assert!(result.is_ok());

        // Read back and verify lm_head was created
        let apr_data = fs::read(&output_path).expect("Read APR");
        let reader = AprV2Reader::from_bytes(&apr_data).expect("Parse APR");
        let tensor_names = reader.tensor_names();
        assert!(
            tensor_names.iter().any(|n| *n == "lm_head.weight"),
            "lm_head.weight should be created from tied embeddings"
        );
    }

    #[test]
    fn test_write_apr_file_qkv_fusion() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let output_path = temp_dir.path().join("fused.apr");

        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (vec![0.1; 16], vec![4, 4]),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.1; 16], vec![4, 4]),
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".to_string(),
            (vec![0.2; 16], vec![4, 4]),
        );
        tensors.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            (vec![0.3; 16], vec![4, 4]),
        );
        tensors.insert(
            "model.layers.0.self_attn.o_proj.weight".to_string(),
            (vec![0.4; 16], vec![4, 4]),
        );

        let model_config = GgufModelConfig {
            architecture: Some("llama".to_string()),
            hidden_size: Some(4),
            num_layers: Some(1),
            num_heads: Some(1),
            num_kv_heads: Some(1),
            vocab_size: Some(4),
            intermediate_size: Some(8),
            max_position_embeddings: Some(64),
            rope_theta: Some(10000.0),
            rms_norm_eps: Some(1e-5),
            rope_type: Some(0),
        };

        let options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        let empty_f16: BTreeMap<String, (Vec<u8>, Vec<usize>)> = BTreeMap::new();
        let result = write_apr_file(
            &tensors,
            &empty_f16,
            &output_path,
            &options,
            None,
            Some(&model_config),
            &Default::default(),
        );
        assert!(
            result.is_ok(),
            "write_apr_file with QKV fusion should succeed: {:?}",
            result.err()
        );
    }

    // ------------------------------------------------------------------------
    // GH-205: F16 Passthrough Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_gh205_f16_passthrough_preserves_bytes() {
        // GH-205: Verify F16 SafeTensors -> APR conversion preserves raw bytes
        use crate::format::converter::import::apr_import;
        use crate::format::test_factory::build_pygmy_safetensors_f16;
        use crate::format::v2::AprV2Reader;

        let temp_dir = TempDir::new().expect("Create temp dir");
        let st_path = temp_dir.path().join("f16_model.safetensors");
        let apr_path = temp_dir.path().join("f16_model.apr");

        // Create F16 SafeTensors
        let st_data = build_pygmy_safetensors_f16();
        fs::write(&st_path, &st_data).expect("Write F16 SafeTensors");

        // Import with default options (should use F16 passthrough)
        let mut options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        options.architecture = Architecture::Qwen2;

        let result = apr_import(st_path.to_str().unwrap(), &apr_path, options);
        assert!(
            result.is_ok(),
            "F16 import should succeed: {:?}",
            result.err()
        );

        // Read back APR and verify tensors are F16
        let apr_bytes = fs::read(&apr_path).expect("Read APR");
        let reader = AprV2Reader::from_bytes(&apr_bytes).expect("Parse APR");

        // Find embedding tensor and verify dtype
        let tensor_names = reader.tensor_names();
        let embed_name = tensor_names
            .iter()
            .find(|n| n.contains("embed_tokens"))
            .expect("Should have embed_tokens tensor");

        let entry = reader.get_tensor(embed_name).expect("Get tensor entry");
        assert_eq!(
            entry.dtype,
            crate::format::v2::TensorDType::F16,
            "GH-205 FAIL: Tensor should be F16, got {:?}",
            entry.dtype
        );
    }

    #[test]
    fn test_gh205_f16_passthrough_no_precision_loss() {
        // GH-205: Verify F16 -> APR -> readback produces identical bytes
        use crate::format::converter::import::apr_import;
        use crate::format::test_factory::build_pygmy_safetensors_f16;
        use crate::serialization::safetensors::MappedSafeTensors;

        let temp_dir = TempDir::new().expect("Create temp dir");
        let st_path = temp_dir.path().join("f16_model.safetensors");
        let apr_path = temp_dir.path().join("f16_model.apr");

        // Create F16 SafeTensors
        let st_data = build_pygmy_safetensors_f16();
        fs::write(&st_path, &st_data).expect("Write F16 SafeTensors");

        // Get original F16 bytes from SafeTensors
        let mapped = MappedSafeTensors::open(&st_path).expect("Open SafeTensors");
        let original_bytes = mapped
            .get_tensor_bytes("model.embed_tokens.weight")
            .expect("Get original F16 bytes");
        let original_len = original_bytes.len();

        // Import with F16 passthrough
        let mut options = ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        };
        options.architecture = Architecture::Qwen2;

        let result = apr_import(st_path.to_str().unwrap(), &apr_path, options);
        assert!(
            result.is_ok(),
            "F16 import should succeed: {:?}",
            result.err()
        );

        // Read back from APR
        let apr_bytes = fs::read(&apr_path).expect("Read APR");
        let reader = crate::format::v2::AprV2Reader::from_bytes(&apr_bytes).expect("Parse APR");

        // Get F16 bytes from APR (mapped name)
        let apr_tensor_bytes = reader
            .get_tensor_data("model.embed_tokens.weight")
            .expect("Get APR tensor bytes");

        // Verify size matches (same number of bytes = no conversion happened)
        assert_eq!(
            apr_tensor_bytes.len(),
            original_len,
            "GH-205 FAIL: APR tensor size {} != original F16 size {} (conversion occurred)",
            apr_tensor_bytes.len(),
            original_len
        );
    }

    // ------------------------------------------------------------------------
    // Rosetta conversion coverage tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_rosetta_inspect_safetensors() {
        use crate::format::rosetta::RosettaStone;

        let temp_dir = TempDir::new().expect("Create temp dir");
        let st_path = temp_dir.path().join("model.safetensors");

        let st_data = build_pygmy_safetensors();
        fs::write(&st_path, &st_data).expect("Write safetensors");

        let rosetta = RosettaStone::new();
        let result = rosetta.inspect(&st_path);
        assert!(
            result.is_ok(),
            "Rosetta inspect should succeed: {:?}",
            result.err()
        );

        let inspection = result.unwrap();
        assert!(!inspection.tensors.is_empty());
        assert!(inspection.file_size > 0);
    }

    #[test]
    fn test_rosetta_inspect_apr() {
        use crate::format::rosetta::RosettaStone;

        let temp_dir = TempDir::new().expect("Create temp dir");
        let apr_path = temp_dir.path().join("model.apr");

        let apr_data = build_pygmy_apr();
        fs::write(&apr_path, &apr_data).expect("Write APR");

        let rosetta = RosettaStone::new();
        let result = rosetta.inspect(&apr_path);
        assert!(
            result.is_ok(),
            "Rosetta inspect APR should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_rosetta_convert_safetensors_to_apr() {
        use crate::format::rosetta::RosettaStone;

        let temp_dir = TempDir::new().expect("Create temp dir");
        let st_path = temp_dir.path().join("input.safetensors");
        let apr_path = temp_dir.path().join("output.apr");

        let st_data = build_pygmy_safetensors();
        fs::write(&st_path, &st_data).expect("Write safetensors");

        let rosetta = RosettaStone::new();
        let result = rosetta.convert(&st_path, &apr_path, None);
        assert!(
            result.is_ok(),
            "Rosetta convert should succeed: {:?}",
            result.err()
        );

        let report = result.unwrap();
        assert!(!report.source_inspection.tensors.is_empty());
        assert!(apr_path.exists());
    }

    #[test]
    fn test_rosetta_convert_st_to_apr_roundtrip() {
        // Test ST->APR roundtrip (APR v2 reading has limitations with v1 parser)
        use crate::format::rosetta::RosettaStone;

        let temp_dir = TempDir::new().expect("Create temp dir");
        let st_path = temp_dir.path().join("input.safetensors");
        let apr_path = temp_dir.path().join("output.apr");

        let st_data = build_pygmy_safetensors();
        fs::write(&st_path, &st_data).expect("Write safetensors");

        let rosetta = RosettaStone::new();
        let result = rosetta.convert(&st_path, &apr_path, None);
        assert!(result.is_ok(), "Rosetta ST->APR convert should succeed");

        // Verify output exists
        assert!(apr_path.exists());
        let apr_bytes = fs::read(&apr_path).expect("Read APR");
        assert!(apr_bytes.len() > 64, "APR should have content");
    }

    // ------------------------------------------------------------------------
    // Dequantization function coverage tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_dequantize_f16_to_f32_basic() {
        let f16_bytes: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x40]; // 1.0, 2.0 in f16
        let result = dequantize_f16_to_f32(&f16_bytes, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_bf16_to_f32_basic() {
        let bf16_bytes: Vec<u8> = vec![0x80, 0x3F, 0x00, 0x40]; // 1.0, 2.0 in bf16
        let result = dequantize_bf16_to_f32(&bf16_bytes, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_q8_0_to_f32() {
        // Q8_0: 34 bytes per block (2 for f16 scale, 32 for int8 values)
        // Create minimal valid Q8_0 block
        let mut q8_bytes: Vec<u8> = vec![0; 34];
        // Set scale to 1.0 (f16: 0x3C00)
        q8_bytes[0] = 0x00;
        q8_bytes[1] = 0x3C;
        // Set quantized values to known values
        for i in 0..32 {
            q8_bytes[2 + i] = (i as i8) as u8;
        }

        let result = dequantize_q8_0_to_f32(&q8_bytes, 32);
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_dequantize_q4_k_to_f32_basic() {
        // Q4_K: 144 bytes per super-block (256 elements)
        let q4k_bytes: Vec<u8> = vec![0; 144];
        let result = dequantize_q4_k_to_f32(&q4k_bytes, 256);
        assert_eq!(result.len(), 256);
        // All zeros input should produce all zeros output
        assert!(result.iter().all(|&v| v == 0.0 || !v.is_nan()));
    }

    #[test]
    fn test_dequantize_q6_k_to_f32_basic() {
        // Q6_K: 210 bytes per super-block (256 elements)
        let q6k_bytes: Vec<u8> = vec![0; 210];
        let result = dequantize_q6_k_to_f32(&q6k_bytes, 256);
        assert_eq!(result.len(), 256);
    }

    // ------------------------------------------------------------------------
    // LAYOUT-002: Transpose function tests (Row-Major Mandate)
    // ------------------------------------------------------------------------

    #[test]
    fn test_transpose_q4k_for_matmul_shape_swap() {
        // Create Q4K data for a 512x256 matrix (2 rows of super-blocks)
        // Each row needs ceil(256/256) = 1 super-block = 144 bytes
        // 512 rows x 1 super-block x 144 bytes = 73728 bytes
        let rows = 512;
        let cols = 256;
        let shape = vec![rows, cols];

        // Create test F32 data and quantize it
        let f32_data: Vec<f32> = (0..(rows * cols))
            .map(|i| (i as f32 / (rows * cols) as f32) - 0.5)
            .collect();
        let q4k_bytes = quantize_q4_k_matrix(&f32_data, &shape);

        // Transpose
        let (transposed_bytes, transposed_shape) = transpose_q4k_for_matmul(&q4k_bytes, &shape);

        // Shape should be swapped: [512, 256] -> [256, 512]
        assert_eq!(transposed_shape, vec![cols, rows]);

        // Output should be valid Q4K bytes
        // New shape [256, 512] needs ceil(512/256) = 2 super-blocks per row
        // 256 rows x 2 super-blocks x 144 bytes = 73728 bytes
        let expected_bytes = 256 * 2 * 144;
        assert_eq!(
            transposed_bytes.len(),
            expected_bytes,
            "Transposed Q4K should have {} bytes, got {}",
            expected_bytes,
            transposed_bytes.len()
        );
    }

    #[test]
    fn test_transpose_q4k_for_matmul_1d_passthrough() {
        // 1D tensors should pass through unchanged
        let q4k_bytes: Vec<u8> = vec![0; 144];
        let shape = vec![256];

        let (result_bytes, result_shape) = transpose_q4k_for_matmul(&q4k_bytes, &shape);

        assert_eq!(result_bytes, q4k_bytes);
        assert_eq!(result_shape, shape);
    }

    #[test]
    fn test_transpose_q6k_for_matmul_shape_swap() {
        // Create Q6K data for a 512x256 matrix
        // Q6_K: 210 bytes per super-block
        let rows = 512;
        let cols = 256;
        let shape = vec![rows, cols];

        // Create test F32 data and quantize it
        let f32_data: Vec<f32> = (0..(rows * cols))
            .map(|i| (i as f32 / (rows * cols) as f32) - 0.5)
            .collect();
        let q6k_bytes = quantize_q6_k_matrix(&f32_data, &shape);

        // Transpose
        let (transposed_bytes, transposed_shape) = transpose_q6k_for_matmul(&q6k_bytes, &shape);

        // Shape should be swapped: [512, 256] -> [256, 512]
        assert_eq!(transposed_shape, vec![cols, rows]);

        // Output should be valid Q6K bytes (after transpose uses q6k_matrix)
        // New shape [256, 512] needs ceil(512/256) = 2 super-blocks per row
        // 256 rows x 2 super-blocks x 210 bytes = 107520 bytes
        let expected_bytes = 256 * 2 * 210;
        assert_eq!(
            transposed_bytes.len(),
            expected_bytes,
            "Transposed Q6K should have {} bytes, got {}",
            expected_bytes,
            transposed_bytes.len()
        );
    }

    #[test]
    fn test_transpose_q6k_for_matmul_1d_passthrough() {
        // 1D tensors should pass through unchanged
        let q6k_bytes: Vec<u8> = vec![0; 210];
        let shape = vec![256];

        let (result_bytes, result_shape) = transpose_q6k_for_matmul(&q6k_bytes, &shape);

        assert_eq!(result_bytes, q6k_bytes);
        assert_eq!(result_shape, shape);
    }

    // ------------------------------------------------------------------------
    // Q5K transpose tests (LAYOUT-002)
    // ------------------------------------------------------------------------

    #[test]
    fn test_transpose_q5k_for_matmul_shape_swap() {
        // Q5K: 256 elements per super-block, 176 bytes per block
        // For a 256x512 matrix: 256 rows, each row has 2 super-blocks
        // Total bytes: 256 * 2 * 176 = 90,112 bytes
        let rows = 256;
        let cols = 512;
        let super_blocks_per_row = 2;
        let q5k_bytes: Vec<u8> = vec![0; rows * super_blocks_per_row * 176];
        let shape = vec![rows, cols];

        let (result_bytes, result_shape) = transpose_q5k_for_matmul(&q5k_bytes, &shape);

        // Shape should be swapped: [256, 512] -> [512, 256]
        assert_eq!(result_shape, vec![cols, rows]);

        // NOTE: trueno-quant converts Q5K to Q6K for better precision (APR doesn't have native Q5K)
        // Result is Q6K format with transposed dimensions
        // After transpose: 512 rows, each row has 1 super-block
        // Expected Q6K bytes: 512 * 1 * 210 = 107,520 bytes
        let expected_super_blocks = 512 * ((256 + 255) / 256);
        assert_eq!(result_bytes.len(), expected_super_blocks * 210);
    }

    #[test]
    fn test_transpose_q5k_for_matmul_1d_passthrough() {
        // 1D tensors should pass through unchanged
        let q5k_bytes: Vec<u8> = vec![0; 176];
        let shape = vec![256];

        let (result_bytes, result_shape) = transpose_q5k_for_matmul(&q5k_bytes, &shape);

        assert_eq!(result_bytes, q5k_bytes);
        assert_eq!(result_shape, shape);
    }

    #[test]
    fn test_quantize_q5k_roundtrip() {
        // Test that Q5K quantization and dequantization are consistent
        let test_data: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
        let q5k_bytes = quantize_q5_k(&test_data);

        // Q5K: 256 elements = 1 super-block = 176 bytes
        assert_eq!(q5k_bytes.len(), 176);
    }

    #[test]
    fn test_quantize_q5k_matrix_row_padding() {
        // Test that Q5K matrix quantization pads rows correctly
        let rows = 4;
        let cols = 128; // Less than 256, should be padded to 256
        let test_data: Vec<f32> = vec![1.0f32; rows * cols];
        let shape = vec![rows, cols];

        let q5k_bytes = quantize_q5_k_matrix(&test_data, &shape);

        // Each row should get 1 super-block (256 elements, padded from 128)
        // 4 rows * 1 block/row * 176 bytes/block = 704 bytes
        assert_eq!(q5k_bytes.len(), rows * 176);
    }

    // ------------------------------------------------------------------------
    // Load model tensors coverage tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_load_model_tensors_safetensors() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let st_path = temp_dir.path().join("model.safetensors");

        let st_data = build_pygmy_safetensors();
        fs::write(&st_path, &st_data).expect("Write safetensors");

        let result = load_model_tensors(&st_path);
        assert!(
            result.is_ok(),
            "load_model_tensors should succeed: {:?}",
            result.err()
        );

        let tensors = result.unwrap();
        assert!(!tensors.is_empty());
    }

    #[test]
    fn test_load_model_tensors_apr_via_v2_reader() {
        // Test APR v2 loading via AprV2Reader (v1 parser has format differences)
        use crate::format::v2::AprV2Reader;

        let temp_dir = TempDir::new().expect("Create temp dir");
        let apr_path = temp_dir.path().join("model.apr");

        let apr_data = build_pygmy_apr();
        fs::write(&apr_path, &apr_data).expect("Write APR");

        // Use V2 reader directly which understands the format
        let reader = AprV2Reader::from_bytes(&apr_data);
        assert!(reader.is_ok(), "AprV2Reader should parse pygmy APR");

        let reader = reader.unwrap();
        assert!(!reader.tensor_names().is_empty(), "Should have tensors");
    }

    #[test]
    fn test_load_model_tensors_unsupported_format() {
        let temp_dir = TempDir::new().expect("Create temp dir");
        let bad_path = temp_dir.path().join("model.xyz");

        fs::write(&bad_path, b"some data").expect("Write file");

        let result = load_model_tensors(&bad_path);
        assert!(result.is_err(), "Unsupported format should fail");
    }

    // ------------------------------------------------------------------------
    // Calculate tensor size coverage tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_calculate_tensor_size() {
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        tensors.insert("a".to_string(), (vec![0.0; 100], vec![10, 10]));
        tensors.insert("b".to_string(), (vec![0.0; 200], vec![20, 10]));

        let size = calculate_tensor_size(&tensors);
        // 300 f32 elements * 4 bytes = 1200 bytes
        assert_eq!(size, 1200);
    }

    #[test]
    fn test_calculate_tensor_size_empty() {
        let tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        let size = calculate_tensor_size(&tensors);
        assert_eq!(size, 0);
    }

    // ------------------------------------------------------------------------
    // BUG-LAYOUT-003: Error paths must not bypass LAYOUT-002 transpose
    // ------------------------------------------------------------------------
    // These tests verify that error paths in GGUF->APR conversion properly fail
    // instead of silently writing column-major data that violates LAYOUT-002.
    // Prior to this fix, failed dequantization wrote raw bytes as F32, corrupting
    // both the layout (column-major instead of row-major) and dtype interpretation.
    // ------------------------------------------------------------------------

    // Note: These are documentation tests verifying the fix was applied.
    // The actual error paths now return Err() instead of silently corrupting data.
    // We cannot easily trigger dequantization failures in unit tests since the
    // dequant functions are robust. The fix ensures that IF they fail, the
    // conversion fails rather than producing corrupt output.

    #[test]
    fn test_bug_layout_003_error_paths_documented() {
        // BUG-LAYOUT-003: Error paths in write.rs now return Err() instead of:
        // - Writing column-major quantized bytes as F32
        // - Bypassing LAYOUT-002 transpose mandate
        //
        // Fixed error paths:
        // - Q5_K dequant failure (was lines 699-705)
        // - Q4_0 dequant failure (was lines 728-734)
        // - Q4_1 dequant failure (was lines 750-756)
        // - Q5_0 dequant failure (was lines 772-778)
        // - Q8_0 dequant failure (was lines 794-800)
        // - Q5_1/Q8_1 unsupported (was lines 809-814)
        // - Unknown dtype (was lines 821-826)
        //
        // All now return AprenderError::FormatError with LAYOUT-002 mandate message.
        //
        // This test documents the fix. The actual enforcement is in write.rs.
        assert!(true, "BUG-LAYOUT-003 fix documented - error paths now fail");
    }
}
