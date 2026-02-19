//! Export command implementation (GH-246)
//!
//! Exports APR models to other formats for different ecosystems:
//! - SafeTensors (.safetensors) - HuggingFace ecosystem
//! - GGUF (.gguf) - llama.cpp / local inference
//! - MLX (directory) - Apple Silicon inference
//! - ONNX (.onnx) - Cross-framework inference (planned)
//! - OpenVINO (.xml/.bin) - Intel inference (planned)
//! - CoreML (.mlpackage) - iOS/macOS deployment (planned)
//!
//! # Example
//!
//! ```bash
//! apr export model.apr --format gguf -o model.gguf
//! apr export model.apr --format mlx -o model-mlx/
//! apr export model.apr --batch gguf,mlx -o exports/
//! apr export --list-formats --json
//! ```

use crate::error::{CliError, Result};
use crate::output;
use aprender::format::{apr_export, ExportFormat, ExportOptions, ExportReport, QuantizationType};
use humansize::{format_size, BINARY};
use std::path::Path;

/// GH-273: Resolve export format (infer from output extension) and parse quantization.
fn resolve_export_options(
    format: &str,
    output: &Path,
    quantize: Option<&str>,
) -> Result<(ExportFormat, Option<QuantizationType>)> {
    // GH-273: Infer export format from output extension when --format is default
    let effective_format = if format == "safetensors" {
        let ext = output.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext {
            "gguf" => "gguf",
            "mlx" => "mlx",
            "onnx" => "onnx",
            _ => format,
        }
    } else {
        format
    };

    let export_format: ExportFormat = effective_format.parse().map_err(|_| {
        CliError::ValidationFailed(format!(
            "Unknown export format: {effective_format}. Use: safetensors, gguf, mlx, onnx, openvino, coreml"
        ))
    })?;

    if !export_format.is_supported() {
        return Err(CliError::ValidationFailed(format!(
            "Export format '{}' is not yet implemented. Supported: safetensors, gguf, mlx",
            export_format.display_name()
        )));
    }

    let quant_type = parse_quantization(quantize)?;
    Ok((export_format, quant_type))
}

/// Execute the export and display results.
fn execute_and_display(
    file: &Path,
    output: &Path,
    export_format: ExportFormat,
    quant_type: Option<QuantizationType>,
    json_output: bool,
) -> Result<()> {
    let options = ExportOptions {
        format: export_format,
        quantize: quant_type,
        ..Default::default()
    };

    match apr_export(file, output, options) {
        Ok(report) => {
            if json_output {
                display_report_json(&report);
            } else {
                display_report(&report);
            }
            Ok(())
        }
        Err(e) => {
            if !json_output {
                println!();
                println!("  {}", output::badge_fail("Export failed"));
            }
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
}

/// Run the export command
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
pub(crate) fn run(
    file: Option<&Path>,
    format: &str,
    output: Option<&Path>,
    quantize: Option<&str>,
    list_formats: bool,
    batch: Option<&str>,
    json_output: bool,
) -> Result<()> {
    // Handle --list-formats
    if list_formats {
        return run_list_formats(json_output);
    }

    // Require file for all other operations
    let file = file.ok_or_else(|| {
        CliError::ValidationFailed("Model file path required. Usage: apr export <FILE>".to_string())
    })?;

    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }

    // Handle --batch mode
    if let Some(batch_formats) = batch {
        return run_batch(file, batch_formats, output, quantize, json_output);
    }

    // Require output for single export
    let output = output.ok_or_else(|| {
        CliError::ValidationFailed("Output path required. Use -o <path>".to_string())
    })?;

    let (export_format, quant_type) = resolve_export_options(format, output, quantize)?;

    // PMAT-261: Detect stdout pipe output (-o - | -o /dev/stdout)
    let pipe_to_stdout = crate::pipe::is_stdout(&output.to_string_lossy());
    if pipe_to_stdout {
        return run_export_to_stdout(file, export_format, quant_type);
    }

    if !json_output {
        output::header("APR Export");
        let mut pairs = vec![
            ("Input", file.display().to_string()),
            ("Output", output.display().to_string()),
            ("Format", export_format.display_name().to_string()),
        ];
        if let Some(ref q) = quant_type {
            pairs.push(("Quantization", format!("{q:?}")));
        }
        println!("{}", output::kv_table(&pairs));
        println!();
        output::pipeline_stage("Exporting", output::StageStatus::Running);
    }

    execute_and_display(file, output, export_format, quant_type, json_output)
}

/// PMAT-261: Export to stdout — write raw bytes, no ANSI, no status messages.
///
/// Exports to a temporary file, then writes the raw bytes to stdout.
/// All status output is suppressed (binary data on stdout must be clean).
fn run_export_to_stdout(
    file: &Path,
    export_format: ExportFormat,
    quant_type: Option<QuantizationType>,
) -> Result<()> {
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join(format!("apr-export-{}.bin", std::process::id()));

    let options = ExportOptions {
        format: export_format,
        quantize: quant_type,
        ..Default::default()
    };

    let result = apr_export(file, &tmp_path, options);
    match result {
        Ok(_) => {
            let data = std::fs::read(&tmp_path).map_err(|e| {
                let _ = std::fs::remove_file(&tmp_path);
                CliError::ValidationFailed(format!("Failed to read exported file: {e}"))
            })?;
            let _ = std::fs::remove_file(&tmp_path);
            crate::pipe::write_stdout(&data)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp_path);
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
}

/// Parse quantization option string
fn parse_quantization(quantize: Option<&str>) -> Result<Option<QuantizationType>> {
    match quantize {
        Some("int8") => Ok(Some(QuantizationType::Int8)),
        Some("int4") => Ok(Some(QuantizationType::Int4)),
        Some("fp16") => Ok(Some(QuantizationType::Fp16)),
        Some("q4k") => Ok(Some(QuantizationType::Q4K)),
        Some(other) => Err(CliError::ValidationFailed(format!(
            "Unknown quantization: {other}. Supported: int8, int4, fp16, q4k"
        ))),
        None => Ok(None),
    }
}

/// List all supported export formats
#[allow(clippy::disallowed_methods)]
fn run_list_formats(json_output: bool) -> Result<()> {
    if json_output {
        let formats: Vec<serde_json::Value> = ExportFormat::all()
            .iter()
            .map(|f| {
                serde_json::json!({
                    "name": f.display_name(),
                    "extension": f.extension(),
                    "supported": f.is_supported(),
                    "parse_aliases": format_aliases(*f),
                })
            })
            .collect();
        let json = serde_json::json!({ "formats": formats });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::header("APR Export — Supported Formats");
        println!();
        for f in ExportFormat::all() {
            let status = if f.is_supported() {
                output::badge_pass("supported")
            } else {
                output::badge_info("planned")
            };
            println!(
                "  {:<14} .{:<14} {}",
                f.display_name(),
                f.extension(),
                status
            );
        }
        println!();
        println!(
            "  {} Use --format <name> to select format.",
            output::badge_info("INFO")
        );
    }
    Ok(())
}

/// Get parse aliases for a format
fn format_aliases(f: ExportFormat) -> Vec<String> {
    match f {
        ExportFormat::SafeTensors => vec!["safetensors".into(), "st".into()],
        ExportFormat::Gguf => vec!["gguf".into()],
        ExportFormat::Mlx => vec!["mlx".into()],
        ExportFormat::Onnx => vec!["onnx".into()],
        ExportFormat::OpenVino => vec!["openvino".into(), "ov".into()],
        ExportFormat::CoreMl => vec!["coreml".into(), "mlpackage".into()],
        ExportFormat::TorchScript => vec!["torchscript".into(), "pt".into(), "torch".into()],
    }
}

/// Print batch export summary (JSON or human-readable).
#[allow(clippy::disallowed_methods)]
fn print_batch_summary(
    file: &Path,
    results: &[(&str, String, ExportReport)],
    total_formats: usize,
    json_output: bool,
) {
    if json_output {
        let json_results: Vec<serde_json::Value> = results
            .iter()
            .map(|(name, path, report)| {
                serde_json::json!({
                    "format": name,
                    "output": path,
                    "original_size": report.original_size,
                    "exported_size": report.exported_size,
                    "tensor_count": report.tensor_count,
                })
            })
            .collect();
        let json = serde_json::json!({
            "batch": true,
            "input": file.display().to_string(),
            "results": json_results,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        println!();
        println!(
            "  {} Batch export complete: {}/{} formats",
            output::badge_pass("PASS"),
            results.len(),
            total_formats
        );
    }
}

/// Batch export to multiple formats
#[allow(clippy::disallowed_methods)]
fn run_batch(
    file: &Path,
    batch_formats: &str,
    output_dir: Option<&Path>,
    quantize: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let out_dir = output_dir.unwrap_or(Path::new("exports"));

    // Parse comma-separated formats
    let formats: Vec<ExportFormat> = batch_formats
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<ExportFormat>()
                .map_err(|_| CliError::ValidationFailed(format!("Unknown format in batch: {s}")))
        })
        .collect::<Result<Vec<_>>>()?;

    // Validate all are supported
    for f in &formats {
        if !f.is_supported() {
            return Err(CliError::ValidationFailed(format!(
                "Format '{}' in batch is not yet supported",
                f.display_name()
            )));
        }
    }

    let quant_type = parse_quantization(quantize)?;

    if !json_output {
        output::header("APR Export — Batch");
        output::kv("Input", file.display().to_string());
        output::kv("Output dir", out_dir.display().to_string());
        output::kv(
            "Formats",
            formats
                .iter()
                .map(ExportFormat::display_name)
                .collect::<Vec<_>>()
                .join(", "),
        );
        println!();
    }

    let mut results = Vec::new();

    for fmt in &formats {
        let ext = fmt.extension();
        let out_path = if *fmt == ExportFormat::Mlx {
            out_dir.join(format!("model-{ext}"))
        } else {
            out_dir.join(format!("model.{ext}"))
        };

        // Create parent dir
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        let options = ExportOptions {
            format: *fmt,
            quantize: quant_type,
            ..Default::default()
        };

        if !json_output {
            output::pipeline_stage(
                &format!("Exporting to {}", fmt.display_name()),
                output::StageStatus::Running,
            );
        }

        match apr_export(file, &out_path, options) {
            Ok(report) => {
                if !json_output {
                    println!(
                        "    {} → {} ({})",
                        fmt.display_name(),
                        out_path.display(),
                        format_size(report.exported_size, BINARY)
                    );
                }
                results.push((fmt.display_name(), out_path.display().to_string(), report));
            }
            Err(e) => {
                if !json_output {
                    println!(
                        "    {} {} — {}",
                        output::badge_fail("FAIL"),
                        fmt.display_name(),
                        e
                    );
                }
            }
        }
    }

    print_batch_summary(file, &results, formats.len(), json_output);

    Ok(())
}

/// Display export report (human-readable)
fn display_report(report: &ExportReport) {
    println!();
    output::subheader("Export Report");

    let mut pairs: Vec<(&str, String)> = vec![
        ("Original size", format_size(report.original_size, BINARY)),
        ("Exported size", format_size(report.exported_size, BINARY)),
        ("Tensors", output::count_fmt(report.tensor_count)),
        ("Format", report.format.display_name().to_string()),
    ];
    if let Some(ref quant) = report.quantization {
        pairs.push(("Quantization", format!("{quant:?}")));
    }

    println!("{}", output::kv_table(&pairs));
    println!();
    println!("  {}", output::badge_pass("Export successful"));
}

/// Display export report (JSON)
#[allow(clippy::disallowed_methods)]
fn display_report_json(report: &ExportReport) {
    let json = serde_json::json!({
        "status": "success",
        "original_size": report.original_size,
        "exported_size": report.exported_size,
        "tensor_count": report.tensor_count,
        "format": report.format.display_name(),
        "quantization": report.quantization.as_ref().map(|q| format!("{q:?}")),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&json).unwrap_or_default()
    );
}

include!("export_part_02.rs");
