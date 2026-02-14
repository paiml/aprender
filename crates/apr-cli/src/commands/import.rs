//! Import command implementation
//!
//! Implements APR-SPEC §13: Import/Convert Pipeline
//!
//! Downloads models from HuggingFace, converts to APR format with inline validation.

use crate::error::{CliError, Result};
use crate::output;
use aprender::format::{apr_import, Architecture, ImportOptions, Source, ValidationConfig};
use colored::Colorize;
use std::path::{Path, PathBuf};

/// Run the import command
pub(crate) fn run(
    source: &str,
    output: Option<&Path>,
    arch: Option<&str>,
    quantize: Option<&str>,
    strict: bool,
    preserve_q4k: bool,
    tokenizer: Option<&PathBuf>,
    enforce_provenance: bool,
    allow_no_config: bool,
) -> Result<()> {
    check_provenance(source, enforce_provenance)?;

    // GH-169: Derive output path from source if not provided
    let output_path = match output {
        Some(p) => p.to_path_buf(),
        None => derive_output_path(source)?,
    };
    let output = output_path.as_path();
    // PMAT-103: If preserve_q4k is set and source is a local GGUF file,
    // use realizar's Q4K converter to preserve quantization
    #[cfg(feature = "inference")]
    if preserve_q4k {
        let source_path = std::path::Path::new(source);
        if source_path.exists()
            && source_path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            return run_q4k_import(source_path, output);
        }
    }

    // BUG-IMPORT-001 FIX: Warn if preserve_q4k is used but feature not enabled
    #[cfg(not(feature = "inference"))]
    if preserve_q4k {
        eprintln!(
            "  {} --preserve-q4k requires the 'inference' feature. \
             Falling back to standard import (Q4K will be dequantized to F32).",
            output::badge_warn("WARN")
        );
    }

    // Parse and display source info
    let parsed_source = Source::parse(source)
        .map_err(|e| CliError::ValidationFailed(format!("Invalid source: {e}")))?;

    output::header("APR Import Pipeline");

    let source_desc = describe_source(&parsed_source);

    println!(
        "{}",
        output::kv_table(&[
            ("Source", source_desc),
            ("Output", output.display().to_string()),
        ])
    );
    println!();

    // Build import options
    let architecture = parse_architecture(arch)?;
    let options = ImportOptions {
        architecture,
        validation: if strict {
            ValidationConfig::Strict
        } else {
            ValidationConfig::Basic
        },
        quantize: parse_quantize(quantize)?,
        compress: None,
        strict,
        cache: true,
        tokenizer_path: tokenizer.cloned(),
        allow_no_config,
    };

    print_import_config(&options);

    // Run import pipeline
    output::pipeline_stage("Importing", output::StageStatus::Running);
    print_import_result(apr_import(source, output, options))
}

/// F-GT-001: Enforce provenance chain — reject pre-baked GGUF imports.
fn check_provenance(source: &str, enforce: bool) -> Result<()> {
    if !enforce {
        return Ok(());
    }
    let is_gguf = source.to_ascii_lowercase().ends_with(".gguf")
        || source.contains("-GGUF")
        || source.contains("-gguf");
    if is_gguf {
        return Err(CliError::ValidationFailed(
            "F-GT-001: --enforce-provenance rejects GGUF imports. \
             Use SafeTensors as the canonical source format for single-provenance testing. \
             See Section 0 of qwen2.5-coder-showcase-demo.md for rationale."
                .to_string(),
        ));
    }
    Ok(())
}

/// Describe a parsed source for display.
fn describe_source(source: &Source) -> String {
    match source {
        Source::HuggingFace { org, repo, file } => {
            let base = format!("hf://{org}/{repo}");
            file.as_ref()
                .map_or(base.clone(), |f| format!("{base}/{f}"))
        }
        Source::Local(path) => path.display().to_string(),
        Source::Url(url) => url.clone(),
    }
}

/// Parse architecture string into Architecture enum.
fn parse_architecture(arch: Option<&str>) -> Result<Architecture> {
    match arch {
        Some("whisper") => Ok(Architecture::Whisper),
        Some("llama") => Ok(Architecture::Llama),
        Some("bert") => Ok(Architecture::Bert),
        Some("qwen2") => Ok(Architecture::Qwen2),
        Some("auto") | None => Ok(Architecture::Auto),
        Some(other) => Err(CliError::ValidationFailed(format!(
            "Unknown architecture: {other}. Supported: whisper, llama, bert, qwen2, auto"
        ))),
    }
}

/// Print import configuration.
fn print_import_config(options: &ImportOptions) {
    let mut config_pairs: Vec<(&str, String)> = vec![
        ("Architecture", format!("{:?}", options.architecture)),
        ("Validation", format!("{:?}", options.validation)),
    ];
    if let Some(q) = &options.quantize {
        config_pairs.push(("Quantization", format!("{q:?}")));
    }
    println!("{}", output::kv_table(&config_pairs));
    println!();
}

/// Print import result with validation report.
fn print_import_result(
    result: std::result::Result<aprender::format::ValidationReport, aprender::error::AprenderError>,
) -> Result<()> {
    match result {
        Ok(report) => {
            println!();
            output::subheader("Validation Report");
            let grade = report.grade();
            println!(
                "{}",
                output::kv_table(&[
                    ("Score", format!("{}/100", report.total_score)),
                    ("Grade", output::grade_color(grade).to_string()),
                ])
            );
            println!();

            if report.passed(95) {
                println!("  {}", output::badge_pass("Import successful"));
            } else {
                println!("  {}", output::badge_warn("Import completed with warnings"));
            }

            Ok(())
        }
        Err(e) => {
            println!();
            println!("  {}", output::badge_fail("Import failed"));
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
}

fn parse_quantize(
    quantize: Option<&str>,
) -> Result<Option<aprender::format::converter::QuantizationType>> {
    use aprender::format::converter::QuantizationType;

    match quantize {
        None => Ok(None),
        Some("int8") => Ok(Some(QuantizationType::Int8)),
        Some("int4") => Ok(Some(QuantizationType::Int4)),
        Some("fp16") => Ok(Some(QuantizationType::Fp16)),
        Some("q4k" | "q4_k") => Ok(Some(QuantizationType::Q4K)),
        Some(other) => Err(CliError::ValidationFailed(format!(
            "Unknown quantization: {other}. Supported: int8, int4, fp16, q4k"
        ))),
    }
}

/// PMAT-103: Import GGUF file to APR with Q4K quantization preserved
///
/// This uses realizar's `GgufToAprQ4KConverter` to create an APR file
/// that preserves raw Q4K bytes for fused kernel inference.
#[cfg(feature = "inference")]
fn run_q4k_import(source: &Path, output: &Path) -> Result<()> {
    use humansize::{format_size, BINARY};
    use realizar::convert::GgufToAprQ4KConverter;

    output::header("APR Q4K Import (Fused Kernel)");
    println!(
        "{}",
        output::kv_table(&[
            ("Source", format!("{} (GGUF)", source.display())),
            ("Output", format!("{} (APR with Q4K)", output.display())),
        ])
    );
    println!();
    output::pipeline_stage("Preserving Q4K quantization", output::StageStatus::Running);

    // Use realizar's Q4K converter
    match GgufToAprQ4KConverter::convert(source, output) {
        Ok(stats) => {
            println!();
            output::subheader("Q4K Import Report");
            println!(
                "{}",
                output::kv_table(&[
                    ("Total tensors", stats.tensor_count.to_string()),
                    ("Q4K tensors", stats.q4k_tensor_count.to_string()),
                    ("Total bytes", format_size(stats.total_bytes as u64, BINARY)),
                    ("Architecture", stats.architecture.clone()),
                    ("Layers", stats.num_layers.to_string()),
                    ("Hidden size", stats.hidden_size.to_string()),
                ])
            );
            println!();
            println!("  {}", output::badge_pass("Q4K import successful"));
            println!(
                "{}",
                "  Model ready for fused kernel inference (30+ tok/s CPU target)".dimmed()
            );
            Ok(())
        }
        Err(e) => {
            println!();
            println!("  {}", output::badge_fail("Q4K import failed"));
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
}

/// Derive output .apr filename from source (GH-169)
///
/// Examples:
/// - hf://Qwen/Qwen2.5-Coder-1.5B-Instruct → Qwen2.5-Coder-1.5B-Instruct.apr
/// - hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/model.gguf → model.apr
/// - /path/to/model.gguf → model.apr
/// - /path/to/model.safetensors → model.apr
fn derive_output_path(source: &str) -> Result<PathBuf> {
    // Parse the source to extract a reasonable filename
    if let Ok(parsed) = Source::parse(source) {
        match parsed {
            Source::HuggingFace { org: _, repo, file } => {
                // If file is specified, use its stem; otherwise use repo name
                let base_name = if let Some(f) = file {
                    Path::new(&f)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or(&repo)
                        .to_string()
                } else {
                    repo
                };
                Ok(PathBuf::from(format!("{base_name}.apr")))
            }
            Source::Local(path) => {
                let stem = path.file_stem().and_then(|s| s.to_str()).ok_or_else(|| {
                    CliError::ValidationFailed("Cannot derive output name from source".into())
                })?;
                Ok(PathBuf::from(format!("{stem}.apr")))
            }
            Source::Url(url) => {
                // Extract filename from URL string
                let filename = url.rsplit('/').next().unwrap_or("model");
                let stem = Path::new(filename)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model");
                Ok(PathBuf::from(format!("{stem}.apr")))
            }
        }
    } else {
        // Fallback: try to extract filename from source string
        let path = Path::new(source);
        let stem = path.file_stem().and_then(|s| s.to_str()).ok_or_else(|| {
            CliError::ValidationFailed(
                "Cannot derive output name from source. Please specify --output.".into(),
            )
        })?;
        Ok(PathBuf::from(format!("{stem}.apr")))
    }
}

#[cfg(test)]
#[path = "import_tests.rs"]
mod tests;
