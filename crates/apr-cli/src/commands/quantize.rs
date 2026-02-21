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
    output_path: Option<&Path>,
    format: Option<&str>,
    batch: Option<&str>,
    plan_only: bool,
    force: bool,
    json_output: bool,
) -> Result<()> {
    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }

    let quant_scheme: QuantScheme = scheme.parse().map_err(CliError::ValidationFailed)?;

    if plan_only {
        return run_plan(file, quant_scheme, format, json_output);
    }

    let output_path = output_path.ok_or_else(|| {
        CliError::ValidationFailed("--output is required (unless --plan is used)".to_string())
    })?;

    if let Some(schemes) = batch {
        return run_batch(file, schemes, output_path, force, json_output);
    }

    check_overwrite_protection(output_path, force)?;
    print_quantize_header(file, quant_scheme, output_path, json_output);

    let output_ext = resolve_output_format(format, output_path);
    if output_ext == "gguf" {
        return quantize_to_gguf(file, quant_scheme, output_path, json_output);
    }

    run_apr_quantize(file, quant_scheme, output_path, json_output)
}

/// F-CONV-064: Overwrite protection check.
fn check_overwrite_protection(output_path: &Path, force: bool) -> Result<()> {
    if output_path.exists() && !force {
        return Err(CliError::ValidationFailed(format!(
            "Output file '{}' already exists. Use --force to overwrite.",
            output_path.display()
        )));
    }
    Ok(())
}

/// Print header for quantize command (no-op in JSON mode).
fn print_quantize_header(file: &Path, scheme: QuantScheme, output_path: &Path, json: bool) {
    if !json {
        output::header("APR Quantize");
        println!(
            "{}",
            output::kv_table(&[
                ("Input", file.display().to_string()),
                ("Scheme", format!("{scheme:?}")),
                ("Output", output_path.display().to_string()),
            ])
        );
        println!();
    }
}

/// Determine output format from explicit flag or file extension.
fn resolve_output_format<'a>(format: Option<&'a str>, output_path: &'a Path) -> &'a str {
    format.unwrap_or_else(|| {
        output_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("apr")
    })
}

/// Run APR-native quantization via apr_convert.
#[allow(clippy::disallowed_methods)]
fn run_apr_quantize(
    file: &Path,
    scheme: QuantScheme,
    output_path: &Path,
    json_output: bool,
) -> Result<()> {
    if !json_output {
        output::pipeline_stage("Quantizing", output::StageStatus::Running);
    }

    let options = ConvertOptions {
        quantize: Some(scheme.into()),
        compress: None,
        validate: true,
    };

    match apr_convert(file, output_path, options) {
        Ok(report) => {
            print_apr_quantize_result(file, output_path, scheme, &report, json_output);
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

/// Print APR quantization result (JSON or human-readable).
#[allow(clippy::disallowed_methods)]
fn print_apr_quantize_result(
    file: &Path,
    output_path: &Path,
    scheme: QuantScheme,
    report: &aprender::format::ConvertReport,
    json_output: bool,
) {
    if json_output {
        let json = serde_json::json!({
            "status": "success",
            "input": file.display().to_string(),
            "output": output_path.display().to_string(),
            "scheme": format!("{scheme:?}"),
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
                ("Peak memory", format_size(input_size + output_size, BINARY),),
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

    let parsed: Vec<QuantScheme> = scheme_list
        .iter()
        .map(|s| s.parse::<QuantScheme>().map_err(CliError::ValidationFailed))
        .collect::<Result<Vec<_>>>()?;

    ensure_output_dir(output_dir)?;
    print_batch_header(file, &scheme_list, output_dir, json_output);

    let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("apr");
    let mut results = Vec::new();

    for (scheme_name, scheme) in scheme_list.iter().zip(parsed.iter()) {
        let output_file = output_dir.join(format!("{stem}-{scheme_name}.{ext}"));
        let ok = run_batch_single(file, scheme_name, *scheme, &output_file, force, json_output)?;
        results.push(ok);
    }

    print_batch_summary(&results, json_output);
    Ok(())
}

/// Ensure the batch output directory exists.
fn ensure_output_dir(output_dir: &Path) -> Result<()> {
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir).map_err(|e| {
            CliError::ValidationFailed(format!(
                "Cannot create output directory '{}': {e}",
                output_dir.display()
            ))
        })?;
    }
    Ok(())
}

/// Print batch mode header (no-op in JSON mode).
fn print_batch_header(file: &Path, schemes: &[&str], output_dir: &Path, json: bool) {
    if !json {
        output::header("APR Quantize — Batch");
        println!("  Input: {}", file.display());
        println!("  Schemes: {}", schemes.join(", "));
        println!("  Output: {}/", output_dir.display());
        println!();
    }
}

/// Run a single batch quantization step. Returns Ok(true) on success, Ok(false) on force-skip.
fn run_batch_single(
    file: &Path,
    scheme_name: &str,
    scheme: QuantScheme,
    output_file: &Path,
    force: bool,
    json_output: bool,
) -> Result<bool> {
    if !json_output {
        output::pipeline_stage(
            &format!("Quantizing {scheme_name}"),
            output::StageStatus::Running,
        );
    }

    let options = ConvertOptions {
        quantize: Some(scheme.into()),
        compress: None,
        validate: true,
    };

    match apr_convert(file, output_file, options) {
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
            Ok(true)
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
            Ok(false)
        }
    }
}

/// Print batch completion summary.
fn print_batch_summary(results: &[bool], json_output: bool) {
    if !json_output {
        println!();
        let success_count = results.iter().filter(|ok| **ok).count();
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
        ..Default::default()
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
                println!("  {}", output::badge_pass("GGUF quantization successful"));
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

include!("quantize_quant_scheme.rs");
