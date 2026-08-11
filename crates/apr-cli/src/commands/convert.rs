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

/// Parse quantization type from CLI string.
fn parse_quantization(s: Option<&str>) -> Result<Option<QuantizationType>> {
    match s {
        Some("int8") => Ok(Some(QuantizationType::Int8)),
        Some("int4") => Ok(Some(QuantizationType::Int4)),
        Some("fp16") => Ok(Some(QuantizationType::Fp16)),
        Some("q4k" | "q4_k") => Ok(Some(QuantizationType::Q4K)),
        Some(other) => Err(CliError::ValidationFailed(format!(
            "Unknown quantization: {other}. Supported: int8, int4, fp16, q4k"
        ))),
        None => Ok(None),
    }
}

/// Parse compression type from CLI string.
fn parse_compression(s: Option<&str>) -> Result<Option<Compression>> {
    match s {
        Some("none") => Ok(Some(Compression::None)),
        Some("zstd" | "zstd-default") => Ok(Some(Compression::ZstdDefault)),
        Some("zstd-max") => Ok(Some(Compression::ZstdMax)),
        Some("lz4") => Ok(Some(Compression::Lz4)),
        Some(other) => Err(CliError::ValidationFailed(format!(
            "Unknown compression: {other}. Supported: none, zstd, zstd-max, lz4"
        ))),
        None => Ok(None),
    }
}

/// Run the convert command
#[allow(clippy::disallowed_methods)]
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "mutating_output_contract"
)]
pub(crate) fn run(
    file: &Path,
    quantize: Option<&str>,
    compress: Option<&str>,
    output: &Path,
    force: bool,
    json_output: bool,
) -> Result<()> {
    contract_pre_format_conversion_roundtrip!();
    validate_convert_inputs(file, output, force)?;

    let quant_type = parse_quantization(quantize)?;
    let compress_type = parse_compression(compress)?;

    if quant_type.is_none() && compress_type.is_none() {
        return Err(CliError::ValidationFailed(
            "At least one of --quantize or --compress must be specified".to_string(),
        ));
    }

    if !json_output {
        print_convert_banner(file, output, quant_type.as_ref(), compress_type.as_ref());
    }

    let options = ConvertOptions {
        quantize: quant_type,
        compress: compress_type,
        validate: true,
    };

    match apr_convert(file, output, options) {
        Ok(report) => {
            print_convert_success(json_output, file, output, &report, quantize, compress);
            contract_post_format_conversion_roundtrip!(&());
            Ok(())
        }
        Err(e) => {
            if !json_output {
                println!();
                println!("  {}", output::badge_fail("Conversion failed"));
            }
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
}

fn validate_convert_inputs(file: &Path, output: &Path, force: bool) -> Result<()> {
    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }
    crate::error::refuse_overwrite(output, force)
}

fn print_convert_banner(
    file: &Path,
    output_path: &Path,
    quant_type: Option<&QuantizationType>,
    compress_type: Option<&Compression>,
) {
    output::header("APR Convert");
    println!(
        "{}",
        output::kv_table(&[
            ("Input", file.display().to_string()),
            ("Output", output_path.display().to_string()),
        ])
    );

    let quant_str = quant_type.map_or("None (copy)".to_string(), |q| format!("{q:?}"));
    let compress_str = compress_type.map_or(String::new(), |c| format!("{c:?}"));

    let mut config_pairs: Vec<(&str, String)> = vec![("Quantization", quant_str)];
    if !compress_str.is_empty() {
        config_pairs.push(("Compression", compress_str));
    }
    println!("{}", output::kv_table(&config_pairs));
    println!();
    output::pipeline_stage("Converting", output::StageStatus::Running);
}

fn print_convert_success(
    json_output: bool,
    file: &Path,
    output_path: &Path,
    report: &aprender::format::ConvertReport,
    quantize: Option<&str>,
    compress: Option<&str>,
) {
    if json_output {
        let json = serde_json::json!({
            "status": "success",
            "input": file.display().to_string(),
            "output": output_path.display().to_string(),
            "original_size": report.original_size,
            "converted_size": report.converted_size,
            "tensor_count": report.tensor_count,
            "reduction_ratio": report.reduction_ratio,
            "reduction_percent": report.reduction_percent(),
            "quantization": quantize,
            "compression": compress,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
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
                    format!(
                        "{} ({:.2}x)",
                        report.reduction_percent(),
                        report.reduction_ratio
                    ),
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            None,
            None,
            Path::new("/tmp/output.apr"),
            false,
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
        let result = run(input.path(), None, None, output.path(), false, false);
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
        let result = run(input.path(), None, None, output.path(), true, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_no_transform_rejected() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run(
            input.path(),
            None,
            None,
            Path::new("/tmp/convert-out.apr"),
            false,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("--quantize"));
                assert!(msg.contains("--compress"));
            }
            _ => panic!("Expected ValidationFailed error for no-op conversion"),
        }
    }

    #[test]
    fn test_run_unknown_quantization() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run(
            input.path(),
            Some("unknown_quant"),
            None,
            Path::new("/tmp/output.apr"),
            false,
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
            false,
        );
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
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_unknown_compression() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run(
            input.path(),
            None,
            Some("unknown_compress"),
            Path::new("/tmp/output.apr"),
            false,
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
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_quantize_and_compress() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run(
            input.path(),
            Some("int8"),
            Some("zstd"),
            Path::new("/tmp/output.apr"),
            false,
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
            false,
        );
        assert!(result.is_err());
    }
}
