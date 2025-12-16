//! Import command implementation
//!
//! Implements APR-SPEC §13: Import/Convert Pipeline
//!
//! Downloads models from HuggingFace, converts to APR format with inline validation.

use crate::error::{CliError, Result};
use aprender::format::{apr_import, Architecture, ImportOptions, Source, ValidationConfig};
use colored::Colorize;
use std::path::Path;

/// Run the import command
pub(crate) fn run(
    source: &str,
    output: &Path,
    arch: Option<&str>,
    quantize: Option<&str>,
    force: bool,
) -> Result<()> {
    // Parse and display source info
    let parsed_source = Source::parse(source)
        .map_err(|e| CliError::ValidationFailed(format!("Invalid source: {e}")))?;

    println!("{}", "=== APR Import Pipeline ===".cyan().bold());
    println!();
    print_source_info(&parsed_source);
    println!("Output: {}", output.display());
    println!();

    // Build import options
    let architecture = match arch {
        Some("whisper") => Architecture::Whisper,
        Some("llama") => Architecture::Llama,
        Some("bert") => Architecture::Bert,
        Some("auto") | None => Architecture::Auto,
        Some(other) => {
            return Err(CliError::ValidationFailed(format!(
                "Unknown architecture: {other}. Supported: whisper, llama, bert, auto"
            )));
        }
    };

    let options = ImportOptions {
        architecture,
        validation: if force {
            ValidationConfig::Basic
        } else {
            ValidationConfig::Strict
        },
        quantize: parse_quantize(quantize)?,
        compress: None,
        force,
        cache: true,
    };

    println!("Architecture: {:?}", options.architecture);
    if let Some(q) = &options.quantize {
        println!("Quantization: {q:?}");
    }
    println!("Validation: {:?}", options.validation);
    println!();

    // Run import pipeline
    println!("{}", "Importing...".yellow());

    match apr_import(source, output, options) {
        Ok(report) => {
            println!();
            println!("{}", "=== Validation Report ===".cyan().bold());
            println!(
                "Score: {}/100 (Grade: {})",
                report.total_score,
                report.grade()
            );
            println!();

            if report.passed(95) {
                println!("{}", "✓ Import successful".green().bold());
            } else {
                println!("{}", "⚠ Import completed with warnings".yellow().bold());
            }

            Ok(())
        }
        Err(e) => {
            println!();
            println!("{}", "✗ Import failed".red().bold());
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
}

fn print_source_info(source: &Source) {
    match source {
        Source::HuggingFace { org, repo, file } => {
            println!("Source: {} (HuggingFace)", "hf://".cyan());
            println!("  Organization: {org}");
            println!("  Repository: {repo}");
            if let Some(f) = file {
                println!("  File: {f}");
            }
        }
        Source::Local(path) => {
            println!("Source: {} (Local)", path.display());
        }
        Source::Url(url) => {
            println!("Source: {url} (URL)");
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
        Some(other) => Err(CliError::ValidationFailed(format!(
            "Unknown quantization: {other}. Supported: int8, int4, fp16"
        ))),
    }
}
