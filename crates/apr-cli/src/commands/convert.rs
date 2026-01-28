//! Convert command implementation
//!
//! Implements APR-SPEC §4.8: Convert Command
//!
//! Applies quantization and compression to models.

use crate::error::{CliError, Result};
use aprender::format::{apr_convert, Compression, ConvertOptions, QuantizationType};
use colored::Colorize;
use humansize::{format_size, BINARY};
use std::path::Path;

/// Run the convert command
pub(crate) fn run(
    file: &Path,
    quantize: Option<&str>,
    compress: Option<&str>,
    output: &Path,
) -> Result<()> {
    // Validate input exists
    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }

    println!("{}", "=== APR Convert ===".cyan().bold());
    println!();
    println!("Input:  {}", file.display());
    println!("Output: {}", output.display());

    // Parse quantization option
    let quant_type = match quantize {
        Some("int8") => Some(QuantizationType::Int8),
        Some("int4") => Some(QuantizationType::Int4),
        Some("fp16") => Some(QuantizationType::Fp16),
        Some("q4k" | "q4_k") => Some(QuantizationType::Q4K),
        Some(other) => {
            return Err(CliError::ValidationFailed(format!(
                "Unknown quantization: {other}. Supported: int8, int4, fp16, q4k"
            )));
        }
        None => None,
    };

    // Parse compression option
    let compress_type = match compress {
        Some("none") => Some(Compression::None),
        Some("zstd" | "zstd-default") => Some(Compression::ZstdDefault),
        Some("zstd-max") => Some(Compression::ZstdMax),
        Some("lz4") => Some(Compression::Lz4),
        Some(other) => {
            return Err(CliError::ValidationFailed(format!(
                "Unknown compression: {other}. Supported: none, zstd, zstd-max, lz4"
            )));
        }
        None => None,
    };

    if let Some(ref q) = quant_type {
        println!("Quantization: {q:?}");
    } else {
        println!("Quantization: None (copy)");
    }
    if let Some(ref c) = compress_type {
        println!("Compression:  {c:?}");
    }
    println!();

    // Build options
    let options = ConvertOptions {
        quantize: quant_type,
        compress: compress_type,
        validate: true,
    };

    // Run conversion
    println!("{}", "Converting...".yellow());

    match apr_convert(file, output, options) {
        Ok(report) => {
            println!();
            println!("{}", "=== Conversion Report ===".cyan().bold());
            println!();
            println!(
                "Original size:  {}",
                format_size(report.original_size, BINARY)
            );
            println!(
                "Converted size: {}",
                format_size(report.converted_size, BINARY)
            );
            println!("Tensors:        {}", report.tensor_count);
            println!(
                "Reduction:      {} ({:.2}x)",
                report.reduction_percent(),
                report.reduction_ratio
            );
            println!();

            if report.reduction_ratio >= 1.0 {
                println!("{}", "✓ Conversion successful".green().bold());
            } else {
                println!(
                    "{}",
                    "⚠ Conversion completed (output larger than input)".yellow()
                );
            }

            Ok(())
        }
        Err(e) => {
            println!();
            println!("{}", "✗ Conversion failed".red().bold());
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
}
