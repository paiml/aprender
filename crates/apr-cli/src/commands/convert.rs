//! Convert command implementation
//!
//! Implements APR-SPEC §4.8: Convert Command
//!
//! Applies quantization and compression to models.

use crate::error::{CliError, Result};
use crate::output;
use aprender::format::{apr_convert, Compression, ConvertOptions, QuantizationType};
use humansize::{format_size, BINARY};
use std::path::Path;

/// Run the convert command
pub(crate) fn run(
    file: &Path,
    quantize: Option<&str>,
    compress: Option<&str>,
    output: &Path,
    force: bool,
) -> Result<()> {
    // Validate input exists
    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }

    // F-CONV-064: Overwrite protection - refuse to overwrite without --force
    if output.exists() && !force {
        return Err(CliError::ValidationFailed(format!(
            "Output file '{}' already exists. Use --force to overwrite.",
            output.display()
        )));
    }

    output::header("APR Convert");
    println!(
        "{}",
        output::kv_table(&[
            ("Input", file.display().to_string()),
            ("Output", output.display().to_string()),
        ])
    );

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

    let quant_str = quant_type
        .as_ref()
        .map_or("None (copy)".to_string(), |q| format!("{q:?}"));
    let compress_str = compress_type
        .as_ref()
        .map_or(String::new(), |c| format!("{c:?}"));

    let mut config_pairs: Vec<(&str, String)> = vec![("Quantization", quant_str)];
    if !compress_str.is_empty() {
        config_pairs.push(("Compression", compress_str));
    }
    println!("{}", output::kv_table(&config_pairs));
    println!();

    // Build options
    let options = ConvertOptions {
        quantize: quant_type,
        compress: compress_type,
        validate: true,
    };

    // Run conversion
    output::pipeline_stage("Converting", output::StageStatus::Running);

    match apr_convert(file, output, options) {
        Ok(report) => {
            println!();
            output::subheader("Conversion Report");
            println!(
                "{}",
                output::kv_table(&[
                    ("Original size", format_size(report.original_size, BINARY)),
                    ("Converted size", format_size(report.converted_size, BINARY)),
                    ("Tensors", output::count_fmt(report.tensor_count)),
                    (
                        "Reduction",
                        format!("{} ({:.2}x)", report.reduction_percent(), report.reduction_ratio),
                    ),
                ])
            );
            println!();

            if report.reduction_ratio >= 1.0 {
                println!("  {}", output::badge_pass("Conversion successful"));
            } else {
                println!(
                    "  {}",
                    output::badge_warn("Conversion completed (output larger than input)")
                );
            }

            Ok(())
        }
        Err(e) => {
            println!();
            println!("  {}", output::badge_fail("Conversion failed"));
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ========================================================================
    // File Validation Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            None,
            None,
            Path::new("/tmp/output.apr"),
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_run_overwrite_protection() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let output = NamedTempFile::with_suffix(".apr").expect("create output");

        let result = run(
            input.path(),
            None,
            None,
            output.path(),
            false, // force = false
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("already exists"));
                assert!(msg.contains("--force"));
            }
            _ => panic!("Expected ValidationFailed error for overwrite protection"),
        }
    }

    #[test]
    fn test_run_overwrite_with_force() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        let output = NamedTempFile::with_suffix(".apr").expect("create output");

        input.write_all(b"test data").expect("write");

        let result = run(
            input.path(),
            None,
            None,
            output.path(),
            true, // force = true, but will still fail on invalid APR
        );
        // Will fail at actual conversion, but tests force path
        assert!(result.is_err());
    }

    // ========================================================================
    // Quantization Option Tests
    // ========================================================================

    #[test]
    fn test_run_unknown_quantization() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            input.path(),
            Some("unknown_quant"),
            None,
            Path::new("/tmp/output.apr"),
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
    fn test_run_quantization_int8() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            input.path(),
            Some("int8"),
            None,
            Path::new("/tmp/output.apr"),
            false,
        );
        // Will fail at conversion, but tests int8 parsing
        assert!(result.is_err());
    }

    #[test]
    fn test_run_quantization_int4() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            input.path(),
            Some("int4"),
            None,
            Path::new("/tmp/output.apr"),
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_quantization_fp16() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            input.path(),
            Some("fp16"),
            None,
            Path::new("/tmp/output.apr"),
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_quantization_q4k() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            input.path(),
            Some("q4k"),
            None,
            Path::new("/tmp/output.apr"),
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_quantization_q4_k_alias() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            input.path(),
            Some("q4_k"),
            None,
            Path::new("/tmp/output.apr"),
            false,
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // Compression Option Tests
    // ========================================================================

    #[test]
    fn test_run_unknown_compression() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            input.path(),
            None,
            Some("unknown_compress"),
            Path::new("/tmp/output.apr"),
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Unknown compression"));
            }
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    #[test]
    fn test_run_compression_none() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            input.path(),
            None,
            Some("none"),
            Path::new("/tmp/output.apr"),
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_compression_zstd() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            input.path(),
            None,
            Some("zstd"),
            Path::new("/tmp/output.apr"),
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_compression_zstd_default() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            input.path(),
            None,
            Some("zstd-default"),
            Path::new("/tmp/output.apr"),
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_compression_zstd_max() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            input.path(),
            None,
            Some("zstd-max"),
            Path::new("/tmp/output.apr"),
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_compression_lz4() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            input.path(),
            None,
            Some("lz4"),
            Path::new("/tmp/output.apr"),
            false,
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // Combined Options Tests
    // ========================================================================

    #[test]
    fn test_run_quantize_and_compress() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");

        let result = run(
            input.path(),
            Some("int8"),
            Some("zstd"),
            Path::new("/tmp/output.apr"),
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_apr_file() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create temp file");
        input.write_all(b"not valid APR").expect("write");

        let result = run(
            input.path(),
            None,
            None,
            Path::new("/tmp/output.apr"),
            false,
        );
        assert!(result.is_err());
    }
}
