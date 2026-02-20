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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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

include!("coverage_functions_part_02.rs");
include!("coverage_functions_part_03.rs");
