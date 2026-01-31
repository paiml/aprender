//! Export command implementation
//!
//! Implements APR-SPEC §4.6: Export Command
//!
//! Exports APR models to other formats for different ecosystems:
//! - SafeTensors (.safetensors) - HuggingFace ecosystem
//! - GGUF (.gguf) - llama.cpp / local inference
//! - ONNX (.onnx) - Cross-framework inference (planned)
//! - TorchScript (.pt) - PyTorch deployment (planned)

use crate::error::{CliError, Result};
use aprender::format::{apr_export, ExportFormat, ExportOptions, ExportReport, QuantizationType};
use colored::Colorize;
use humansize::{format_size, BINARY};
use std::path::Path;

/// Run the export command
pub(crate) fn run(file: &Path, format: &str, output: &Path, quantize: Option<&str>) -> Result<()> {
    // Validate input exists
    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }

    println!("{}", "=== APR Export ===".cyan().bold());
    println!();
    println!("Input:  {}", file.display());
    println!("Output: {}", output.display());
    println!("Format: {format}");

    // Parse export format
    let export_format: ExportFormat = format.parse().map_err(|_| {
        CliError::ValidationFailed(format!(
            "Unknown export format: {format}. Supported: safetensors, gguf"
        ))
    })?;

    // Check if format is supported
    if !export_format.is_supported() {
        return Err(CliError::ValidationFailed(format!(
            "Export format '{format}' is not yet supported. Use 'safetensors' or 'gguf'."
        )));
    }

    // Parse quantization option
    let quant_type = match quantize {
        Some("int8") => Some(QuantizationType::Int8),
        Some("int4") => Some(QuantizationType::Int4),
        Some("fp16") => Some(QuantizationType::Fp16),
        Some(other) => {
            return Err(CliError::ValidationFailed(format!(
                "Unknown quantization: {other}. Supported: int8, int4, fp16"
            )));
        }
        None => None,
    };

    if let Some(ref q) = quant_type {
        println!("Quantization: {q:?}");
    }
    println!();

    // Build options
    let options = ExportOptions {
        format: export_format,
        quantize: quant_type,
        ..Default::default()
    };

    // Run export
    println!("{}", "Exporting...".yellow());

    match apr_export(file, output, options) {
        Ok(report) => {
            display_report(&report);
            Ok(())
        }
        Err(e) => {
            println!();
            println!("{}", "Export failed".red().bold());
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
}

/// Display export report
fn display_report(report: &ExportReport) {
    println!();
    println!("{}", "=== Export Report ===".cyan().bold());
    println!();
    println!(
        "Original size:  {}",
        format_size(report.original_size, BINARY)
    );
    println!(
        "Exported size:  {}",
        format_size(report.exported_size, BINARY)
    );
    println!("Tensors:        {}", report.tensor_count);
    println!("Format:         {:?}", report.format);

    if let Some(ref quant) = report.quantization {
        println!("Quantization:   {quant:?}");
    }

    println!();
    println!("{}", "Export successful".green().bold());
}

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
            Path::new("/nonexistent/model.apr"),
            "safetensors",
            Path::new("/tmp/output.safetensors"),
            None,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_run_unknown_format() {
        // Create a temp file to bypass file existence check
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            file.path(),
            "unknown_format",
            Path::new("/tmp/output.xyz"),
            None,
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
            file.path(),
            "safetensors",
            Path::new("/tmp/output.safetensors"),
            Some("unknown_quant"),
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

        let result = run(file.path(), "onnx", Path::new("/tmp/output.onnx"), None);
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("not yet supported"));
            }
            _ => panic!("Expected ValidationFailed error for unsupported format"),
        }
    }

    // ========================================================================
    // Quantization Option Tests
    // ========================================================================

    #[test]
    fn test_run_with_int8_quantization() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");

        // This will fail at apr_export, but we test the quantization parsing
        let result = run(
            file.path(),
            "safetensors",
            Path::new("/tmp/output.safetensors"),
            Some("int8"),
        );
        // Will fail because file is not a valid APR, but that's after quant parsing
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_int4_quantization() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            file.path(),
            "safetensors",
            Path::new("/tmp/output.safetensors"),
            Some("int4"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_fp16_quantization() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            file.path(),
            "safetensors",
            Path::new("/tmp/output.safetensors"),
            Some("fp16"),
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // Display Report Tests
    // ========================================================================

    #[test]
    fn test_display_report_basic() {
        let report = ExportReport {
            original_size: 1024 * 1024, // 1MB
            exported_size: 512 * 1024,  // 512KB
            tensor_count: 10,
            format: ExportFormat::SafeTensors,
            quantization: None,
        };
        // Should not panic
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
    fn test_display_report_large_model() {
        let report = ExportReport {
            original_size: 7 * 1024 * 1024 * 1024, // 7GB
            exported_size: 4 * 1024 * 1024 * 1024, // 4GB
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
            file.path(),
            "safetensors",
            Path::new("/tmp/output.safetensors"),
            None,
        );
        // Should fail because file is not valid APR
        assert!(result.is_err());
    }
}
