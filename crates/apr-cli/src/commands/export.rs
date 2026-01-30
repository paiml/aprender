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
