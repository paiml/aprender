//! Quantize command implementation (GH-243)
//!
//! Unified quantization pipeline with support for multiple schemes
//! and output formats. Surfaces entrenar's quantization pipeline
//! through the apr CLI.
//!
//! # Example
//!
//! ```bash
//! apr quantize model.apr --scheme int4 -o model-int4.apr
//! apr quantize model.safetensors --scheme q4k --format gguf -o model.gguf
//! apr quantize model.apr --batch int4,int8,q4k -o models/
//! apr quantize model.apr --scheme int4 --plan --json
//! ```

use crate::error::{CliError, Result};
use crate::output;
use aprender::format::{apr_convert, ConvertOptions, QuantizationType};
use humansize::{format_size, BINARY};
use std::path::Path;

/// Quantization scheme selection
#[derive(Debug, Clone, Copy)]
pub enum QuantScheme {
    Int8,
    Int4,
    Fp16,
    Q4K,
}

impl std::str::FromStr for QuantScheme {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "int8" | "i8" | "q8_0" => Ok(Self::Int8),
            "int4" | "i4" | "q4_0" => Ok(Self::Int4),
            "fp16" | "f16" | "half" => Ok(Self::Fp16),
            "q4k" | "q4_k" | "q4_k_m" => Ok(Self::Q4K),
            _ => Err(format!(
                "Unknown quantization scheme: {s}. Supported: int8, int4, fp16, q4k"
            )),
        }
    }
}

impl From<QuantScheme> for QuantizationType {
    fn from(s: QuantScheme) -> Self {
        match s {
            QuantScheme::Int8 => Self::Int8,
            QuantScheme::Int4 => Self::Int4,
            QuantScheme::Fp16 => Self::Fp16,
            QuantScheme::Q4K => Self::Q4K,
        }
    }
}

/// Estimate memory requirements for quantization
fn estimate_memory(file_size: u64, scheme: QuantScheme) -> (u64, u64, f64) {
    let bits_per_weight: f64 = match scheme {
        QuantScheme::Int8 => 8.0,
        QuantScheme::Int4 => 4.0,
        QuantScheme::Fp16 => 16.0,
        QuantScheme::Q4K => 4.5,
    };
    let input_size = file_size;
    // Assume input is F32 (32 bits/weight)
    let output_size = (file_size as f64 * bits_per_weight / 32.0) as u64;
    let reduction = if output_size > 0 {
        input_size as f64 / output_size as f64
    } else {
        1.0
    };
    (input_size, output_size, reduction)
}

/// Run the quantize command
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
pub(crate) fn run(
    file: &Path,
    scheme: &str,
    output_path: &Path,
    format: Option<&str>,
    batch: Option<&str>,
    plan_only: bool,
    force: bool,
    json_output: bool,
) -> Result<()> {
    // Validate input exists
    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }

    // Handle batch mode
    if let Some(schemes) = batch {
        return run_batch(file, schemes, output_path, force, json_output);
    }

    // Parse scheme
    let quant_scheme: QuantScheme = scheme.parse().map_err(CliError::ValidationFailed)?;

    // Plan mode
    if plan_only {
        return run_plan(file, quant_scheme, format, json_output);
    }

    // F-CONV-064: Overwrite protection
    if output_path.exists() && !force {
        return Err(CliError::ValidationFailed(format!(
            "Output file '{}' already exists. Use --force to overwrite.",
            output_path.display()
        )));
    }

    if !json_output {
        output::header("APR Quantize");
        println!(
            "{}",
            output::kv_table(&[
                ("Input", file.display().to_string()),
                ("Scheme", format!("{quant_scheme:?}")),
                ("Output", output_path.display().to_string()),
            ])
        );
        println!();
    }

    // Determine output format from extension or flag
    let output_ext = format.unwrap_or_else(|| {
        output_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("apr")
    });

    match output_ext {
        "gguf" => {
            return quantize_to_gguf(file, quant_scheme, output_path, json_output);
        }
        _ => {
            // Use apr_convert for APR/SafeTensors output
        }
    }

    if !json_output {
        output::pipeline_stage("Quantizing", output::StageStatus::Running);
    }

    let options = ConvertOptions {
        quantize: Some(quant_scheme.into()),
        compress: None,
        validate: true,
    };

    match apr_convert(file, output_path, options) {
        Ok(report) => {
            if json_output {
                let json = serde_json::json!({
                    "status": "success",
                    "input": file.display().to_string(),
                    "output": output_path.display().to_string(),
                    "scheme": format!("{quant_scheme:?}"),
                    "original_size": report.original_size,
                    "quantized_size": report.converted_size,
                    "reduction_ratio": report.reduction_ratio,
                    "tensor_count": report.tensor_count,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json).unwrap_or_default()
                );
            } else {
                println!();
                output::subheader("Quantization Report");
                println!(
                    "{}",
                    output::kv_table(&[
                        ("Original size", format_size(report.original_size, BINARY)),
                        ("Quantized size", format_size(report.converted_size, BINARY)),
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
                println!("  {}", output::badge_pass("Quantization successful"));
            }
            Ok(())
        }
        Err(e) => {
            if !json_output {
                println!();
                println!("  {}", output::badge_fail("Quantization failed"));
            }
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
}

/// Run quantization in planning mode (estimate only)
#[allow(clippy::disallowed_methods)]
fn run_plan(
    file: &Path,
    scheme: QuantScheme,
    format: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let file_size = std::fs::metadata(file)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read model file: {e}")))?
        .len();

    let (input_size, output_size, reduction) = estimate_memory(file_size, scheme);
    let output_format = format.unwrap_or("apr");

    if json_output {
        let json = serde_json::json!({
            "plan": true,
            "input": file.display().to_string(),
            "input_size": input_size,
            "estimated_output_size": output_size,
            "reduction_ratio": reduction,
            "scheme": format!("{scheme:?}"),
            "output_format": output_format,
            "peak_memory_estimate": input_size + output_size,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::header("APR Quantize — Plan");
        println!(
            "{}",
            output::kv_table(&[
                ("Input", file.display().to_string()),
                ("Input size", format_size(input_size, BINARY)),
                ("Scheme", format!("{scheme:?}")),
                ("Output format", output_format.to_string()),
                ("Estimated output", format_size(output_size, BINARY)),
                ("Reduction", format!("{reduction:.2}x")),
                (
                    "Peak memory",
                    format_size(input_size + output_size, BINARY),
                ),
            ])
        );
        println!();
        println!(
            "  {} Run without --plan to execute quantization.",
            output::badge_info("INFO")
        );
    }

    Ok(())
}

/// Run batch quantization (multiple schemes)
fn run_batch(
    file: &Path,
    schemes: &str,
    output_dir: &Path,
    force: bool,
    json_output: bool,
) -> Result<()> {
    let scheme_list: Vec<&str> = schemes.split(',').map(str::trim).collect();

    if scheme_list.is_empty() {
        return Err(CliError::ValidationFailed(
            "No quantization schemes specified for batch mode".to_string(),
        ));
    }

    // Validate all schemes first
    let parsed: Vec<QuantScheme> = scheme_list
        .iter()
        .map(|s| s.parse::<QuantScheme>().map_err(CliError::ValidationFailed))
        .collect::<Result<Vec<_>>>()?;

    // Create output directory if needed
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir).map_err(|e| {
            CliError::ValidationFailed(format!(
                "Cannot create output directory '{}': {e}",
                output_dir.display()
            ))
        })?;
    }

    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    if !json_output {
        output::header("APR Quantize — Batch");
        println!("  Input: {}", file.display());
        println!("  Schemes: {}", scheme_list.join(", "));
        println!("  Output: {}/", output_dir.display());
        println!();
    }

    let mut results = Vec::new();

    for (scheme_name, scheme) in scheme_list.iter().zip(parsed.iter()) {
        let ext = file
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("apr");
        let output_file = output_dir.join(format!("{stem}-{scheme_name}.{ext}"));

        if !json_output {
            output::pipeline_stage(
                &format!("Quantizing {scheme_name}"),
                output::StageStatus::Running,
            );
        }

        let options = ConvertOptions {
            quantize: Some((*scheme).into()),
            compress: None,
            validate: true,
        };

        match apr_convert(file, &output_file, options) {
            Ok(report) => {
                if !json_output {
                    println!(
                        "    {} {} → {} ({:.2}x)",
                        output::badge_pass("OK"),
                        format_size(report.original_size, BINARY),
                        format_size(report.converted_size, BINARY),
                        report.reduction_ratio,
                    );
                }
                results.push(((*scheme_name).to_string(), true, output_file));
            }
            Err(e) => {
                if !json_output {
                    println!("    {} {e}", output::badge_fail("FAIL"));
                }
                if !force {
                    return Err(CliError::ValidationFailed(format!(
                        "Batch quantization failed at scheme {scheme_name}: {e}"
                    )));
                }
                results.push(((*scheme_name).to_string(), false, output_file));
            }
        }
    }

    if !json_output {
        println!();
        let success_count = results.iter().filter(|(_, ok, _)| *ok).count();
        println!(
            "  {} {}/{} quantizations completed",
            if success_count == results.len() {
                output::badge_pass("DONE")
            } else {
                output::badge_warn("PARTIAL")
            },
            success_count,
            results.len()
        );
    }

    Ok(())
}

/// Quantize to GGUF format (via export pipeline)
fn quantize_to_gguf(
    file: &Path,
    scheme: QuantScheme,
    output_path: &Path,
    json_output: bool,
) -> Result<()> {
    // GGUF export uses the export pipeline with quantization
    use aprender::format::{apr_export, ExportFormat, ExportOptions};

    if !json_output {
        output::pipeline_stage("Quantizing to GGUF", output::StageStatus::Running);
    }

    let options = ExportOptions {
        format: ExportFormat::Gguf,
        quantize: Some(scheme.into()),
        include_tokenizer: false,
        include_config: false,
    };

    match apr_export(file, output_path, options) {
        Ok(report) => {
            if !json_output {
                println!();
                output::subheader("GGUF Quantization Report");
                println!(
                    "{}",
                    output::kv_table(&[
                        ("Original size", format_size(report.original_size, BINARY)),
                        ("GGUF size", format_size(report.exported_size, BINARY)),
                        ("Tensors", output::count_fmt(report.tensor_count)),
                        ("Format", "GGUF".to_string()),
                    ])
                );
                println!();
                println!(
                    "  {}",
                    output::badge_pass("GGUF quantization successful")
                );
            }
            Ok(())
        }
        Err(e) => {
            if !json_output {
                println!();
                println!("  {}", output::badge_fail("GGUF quantization failed"));
            }
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_quant_scheme_parse_int8() {
        let scheme: QuantScheme = "int8".parse().expect("parse int8");
        assert!(matches!(scheme, QuantScheme::Int8));
    }

    #[test]
    fn test_quant_scheme_parse_int4() {
        let scheme: QuantScheme = "int4".parse().expect("parse int4");
        assert!(matches!(scheme, QuantScheme::Int4));
    }

    #[test]
    fn test_quant_scheme_parse_fp16() {
        let scheme: QuantScheme = "fp16".parse().expect("parse fp16");
        assert!(matches!(scheme, QuantScheme::Fp16));
    }

    #[test]
    fn test_quant_scheme_parse_q4k() {
        let scheme: QuantScheme = "q4k".parse().expect("parse q4k");
        assert!(matches!(scheme, QuantScheme::Q4K));
    }

    #[test]
    fn test_quant_scheme_parse_aliases() {
        assert!("i8".parse::<QuantScheme>().is_ok());
        assert!("i4".parse::<QuantScheme>().is_ok());
        assert!("q8_0".parse::<QuantScheme>().is_ok());
        assert!("q4_0".parse::<QuantScheme>().is_ok());
        assert!("f16".parse::<QuantScheme>().is_ok());
        assert!("half".parse::<QuantScheme>().is_ok());
        assert!("q4_k".parse::<QuantScheme>().is_ok());
        assert!("q4_k_m".parse::<QuantScheme>().is_ok());
    }

    #[test]
    fn test_quant_scheme_parse_unknown() {
        assert!("unknown".parse::<QuantScheme>().is_err());
    }

    #[test]
    fn test_quant_scheme_to_quant_type() {
        assert!(matches!(
            QuantizationType::from(QuantScheme::Int8),
            QuantizationType::Int8
        ));
        assert!(matches!(
            QuantizationType::from(QuantScheme::Q4K),
            QuantizationType::Q4K
        ));
    }

    #[test]
    fn test_estimate_memory_int4() {
        let (input, output, ratio) = estimate_memory(1_000_000, QuantScheme::Int4);
        assert_eq!(input, 1_000_000);
        assert_eq!(output, 125_000); // 4/32 = 0.125
        assert!((ratio - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_memory_fp16() {
        let (_, output, ratio) = estimate_memory(1_000_000, QuantScheme::Fp16);
        assert_eq!(output, 500_000); // 16/32 = 0.5
        assert!((ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            "int4",
            Path::new("/tmp/output.apr"),
            None,
            None,
            false,
            false,
            false,
        );
        assert!(result.is_err());
        assert!(matches!(result, Err(CliError::FileNotFound(_))));
    }

    #[test]
    fn test_run_unknown_scheme() {
        let input = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run(
            input.path(),
            "bad_scheme",
            Path::new("/tmp/output.apr"),
            None,
            None,
            false,
            false,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Unknown quantization scheme"));
            }
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_overwrite_protection() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let output = NamedTempFile::with_suffix(".apr").expect("create output");
        let result = run(
            input.path(),
            "int4",
            output.path(),
            None,
            None,
            false,
            false, // force=false
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("already exists"));
            }
            _ => panic!("Expected overwrite protection error"),
        }
    }

    #[test]
    fn test_run_plan_mode() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 1024]).expect("write");
        let result = run(
            input.path(),
            "int4",
            Path::new("/tmp/output.apr"),
            None,
            None,
            true, // plan only
            false,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_plan_json() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 1024]).expect("write");
        let result = run(
            input.path(),
            "int4",
            Path::new("/tmp/output.apr"),
            None,
            None,
            true,
            false,
            true, // json
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_invalid_apr_content() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(b"not valid APR data").expect("write");
        let result = run(
            input.path(),
            "int4",
            Path::new("/tmp/output.apr"),
            None,
            None,
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_empty_schemes() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run_batch(input.path(), "", Path::new("/tmp/batch/"), false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_invalid_scheme() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run_batch(
            input.path(),
            "int4,unknown",
            Path::new("/tmp/batch/"),
            false,
            false,
        );
        assert!(result.is_err());
    }
}
