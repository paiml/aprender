
#[cfg(test)]
mod tests {
    use super::*;
    use aprender::format::ExportReport;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ========================================================================
    // Error Path Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Some(Path::new("/nonexistent/model.apr")),
            "safetensors",
            Some(Path::new("/tmp/output.safetensors")),
            None,
            false,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_run_no_file() {
        let result = run(None, "safetensors", None, None, false, None, false);
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Model file path required"));
            }
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    #[test]
    fn test_run_no_output() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run(
            Some(file.path()),
            "safetensors",
            None,
            None,
            false,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Output path required"));
            }
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    #[test]
    fn test_run_unknown_format() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run(
            Some(file.path()),
            "unknown_format",
            Some(Path::new("/tmp/output.xyz")),
            None,
            false,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Unknown export format"));
            }
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    #[test]
    fn test_run_unknown_quantization() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run(
            Some(file.path()),
            "safetensors",
            Some(Path::new("/tmp/output.safetensors")),
            Some("unknown_quant"),
            false,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Unknown quantization"));
            }
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    #[test]
    fn test_run_unsupported_format_onnx() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run(
            Some(file.path()),
            "onnx",
            Some(Path::new("/tmp/output.onnx")),
            None,
            false,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("not yet implemented"));
            }
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    #[test]
    fn test_run_unsupported_format_openvino() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run(
            Some(file.path()),
            "openvino",
            Some(Path::new("/tmp/output.xml")),
            None,
            false,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("not yet implemented"));
            }
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    #[test]
    fn test_run_unsupported_format_coreml() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run(
            Some(file.path()),
            "coreml",
            Some(Path::new("/tmp/output.mlpackage")),
            None,
            false,
            None,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("not yet implemented"));
            }
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    // ========================================================================
    // Quantization Option Tests
    // ========================================================================

    #[test]
    fn test_parse_quantization_valid() {
        assert!(matches!(
            parse_quantization(Some("int8")),
            Ok(Some(QuantizationType::Int8))
        ));
        assert!(matches!(
            parse_quantization(Some("int4")),
            Ok(Some(QuantizationType::Int4))
        ));
        assert!(matches!(
            parse_quantization(Some("fp16")),
            Ok(Some(QuantizationType::Fp16))
        ));
        assert!(matches!(
            parse_quantization(Some("q4k")),
            Ok(Some(QuantizationType::Q4K))
        ));
        assert!(matches!(parse_quantization(None), Ok(None)));
    }

    #[test]
    fn test_parse_quantization_invalid() {
        assert!(parse_quantization(Some("unknown")).is_err());
    }

    #[test]
    fn test_run_with_int8_quantization() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run(
            Some(file.path()),
            "safetensors",
            Some(Path::new("/tmp/output.safetensors")),
            Some("int8"),
            false,
            None,
            false,
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // List Formats Tests
    // ========================================================================

    #[test]
    fn test_list_formats() {
        let result = run(None, "safetensors", None, None, true, None, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_formats_json() {
        let result = run(None, "safetensors", None, None, true, None, true);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Batch Export Tests
    // ========================================================================

    #[test]
    fn test_batch_unknown_format() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run(
            Some(file.path()),
            "safetensors",
            Some(Path::new("/tmp/exports")),
            None,
            false,
            Some("gguf,unknown"),
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_unsupported_format() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run(
            Some(file.path()),
            "safetensors",
            Some(Path::new("/tmp/exports")),
            None,
            false,
            Some("gguf,onnx"),
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("not yet supported"));
            }
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    // ========================================================================
    // Format Aliases Tests
    // ========================================================================

    #[test]
    fn test_format_aliases() {
        let aliases = format_aliases(ExportFormat::SafeTensors);
        assert!(aliases.contains(&"safetensors".to_string()));
        assert!(aliases.contains(&"st".to_string()));

        let aliases = format_aliases(ExportFormat::Mlx);
        assert!(aliases.contains(&"mlx".to_string()));

        let aliases = format_aliases(ExportFormat::OpenVino);
        assert!(aliases.contains(&"openvino".to_string()));
        assert!(aliases.contains(&"ov".to_string()));
    }

    // ========================================================================
    // Display Report Tests
    // ========================================================================

    #[test]
    fn test_display_report_basic() {
        let report = ExportReport {
            original_size: 1024 * 1024,
            exported_size: 512 * 1024,
            tensor_count: 10,
            format: ExportFormat::SafeTensors,
            quantization: None,
        };
        display_report(&report);
    }

    #[test]
    fn test_display_report_with_quantization() {
        let report = ExportReport {
            original_size: 2048 * 1024,
            exported_size: 512 * 1024,
            tensor_count: 20,
            format: ExportFormat::Gguf,
            quantization: Some(QuantizationType::Int8),
        };
        display_report(&report);
    }

    #[test]
    fn test_display_report_json() {
        let report = ExportReport {
            original_size: 1024 * 1024,
            exported_size: 512 * 1024,
            tensor_count: 10,
            format: ExportFormat::Mlx,
            quantization: None,
        };
        display_report_json(&report);
    }

    #[test]
    fn test_display_report_large_model() {
        let report = ExportReport {
            original_size: 7 * 1024 * 1024 * 1024,
            exported_size: 4 * 1024 * 1024 * 1024,
            tensor_count: 500,
            format: ExportFormat::Gguf,
            quantization: Some(QuantizationType::Int4),
        };
        display_report(&report);
    }

    // ========================================================================
    // Invalid APR File Tests
    // ========================================================================

    #[test]
    fn test_run_invalid_apr_file() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid APR file")
            .expect("write to file");
        let result = run(
            Some(file.path()),
            "safetensors",
            Some(Path::new("/tmp/output.safetensors")),
            None,
            false,
            None,
            false,
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // ExportFormat API Tests
    // ========================================================================

    #[test]
    fn test_export_format_all() {
        let all = ExportFormat::all();
        assert!(all.len() >= 7);
        assert!(all.contains(&ExportFormat::Mlx));
        assert!(all.contains(&ExportFormat::OpenVino));
        assert!(all.contains(&ExportFormat::CoreMl));
    }

    #[test]
    fn test_export_format_display_name() {
        assert_eq!(ExportFormat::SafeTensors.display_name(), "SafeTensors");
        assert_eq!(ExportFormat::Gguf.display_name(), "GGUF");
        assert_eq!(ExportFormat::Mlx.display_name(), "MLX");
        assert_eq!(ExportFormat::Onnx.display_name(), "ONNX");
        assert_eq!(ExportFormat::OpenVino.display_name(), "OpenVINO");
        assert_eq!(ExportFormat::CoreMl.display_name(), "CoreML");
    }

    #[test]
    fn test_export_format_parse_new_variants() {
        assert!(matches!("mlx".parse::<ExportFormat>(), Ok(ExportFormat::Mlx)));
        assert!(matches!(
            "openvino".parse::<ExportFormat>(),
            Ok(ExportFormat::OpenVino)
        ));
        assert!(matches!("ov".parse::<ExportFormat>(), Ok(ExportFormat::OpenVino)));
        assert!(matches!(
            "coreml".parse::<ExportFormat>(),
            Ok(ExportFormat::CoreMl)
        ));
        assert!(matches!(
            "mlpackage".parse::<ExportFormat>(),
            Ok(ExportFormat::CoreMl)
        ));
    }

    #[test]
    fn test_export_format_supported() {
        assert!(ExportFormat::SafeTensors.is_supported());
        assert!(ExportFormat::Gguf.is_supported());
        assert!(ExportFormat::Mlx.is_supported());
        assert!(!ExportFormat::Onnx.is_supported());
        assert!(!ExportFormat::OpenVino.is_supported());
        assert!(!ExportFormat::CoreMl.is_supported());
    }

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::Mlx.extension(), "mlx");
        assert_eq!(ExportFormat::OpenVino.extension(), "xml");
        assert_eq!(ExportFormat::CoreMl.extension(), "mlpackage");
    }

    // ========================================================================
    // PMAT-261: Stdout Pipe Detection Tests
    // ========================================================================

    #[test]
    fn test_stdout_pipe_detection_dash() {
        let result = run(
            Some(Path::new("/nonexistent.apr")),
            "safetensors",
            Some(Path::new("-")),
            None,
            false,
            None,
            false,
        );
        // Fails with FileNotFound (not output validation) because stdout is detected
        assert!(result.is_err());
        assert!(matches!(result, Err(CliError::FileNotFound(_))));
    }

    #[test]
    fn test_stdout_pipe_detection_dev_stdout() {
        let result = run(
            Some(Path::new("/nonexistent.apr")),
            "safetensors",
            Some(Path::new("/dev/stdout")),
            None,
            false,
            None,
            false,
        );
        assert!(result.is_err());
        assert!(matches!(result, Err(CliError::FileNotFound(_))));
    }

    #[test]
    fn test_stdout_pipe_run_export_to_stdout_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid APR file").expect("write");
        let result = run_export_to_stdout(
            file.path(),
            ExportFormat::SafeTensors,
            None,
        );
        assert!(result.is_err());
    }
}
