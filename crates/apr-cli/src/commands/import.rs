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
    preserve_q4k: bool,
) -> Result<()> {
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

    #[allow(unused_variables)]
    let _ = preserve_q4k; // Suppress unused warning when inference feature not enabled
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
        Some("qwen2") => Architecture::Qwen2,
        Some("auto") | None => Architecture::Auto,
        Some(other) => {
            return Err(CliError::ValidationFailed(format!(
                "Unknown architecture: {other}. Supported: whisper, llama, bert, qwen2, auto"
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

    println!("{}", "=== APR Q4K Import (Fused Kernel) ===".cyan().bold());
    println!();
    println!("Source: {} (GGUF)", source.display());
    println!("Output: {} (APR with Q4K)", output.display());
    println!();
    println!(
        "{}",
        "Preserving Q4K quantization for fused kernel inference...".yellow()
    );

    // Use realizar's Q4K converter
    match GgufToAprQ4KConverter::convert(source, output) {
        Ok(stats) => {
            println!();
            println!("{}", "=== Q4K Import Report ===".cyan().bold());
            println!("Total tensors:    {}", stats.tensor_count);
            println!("Q4K tensors:      {}", stats.q4k_tensor_count);
            println!(
                "Total bytes:      {}",
                format_size(stats.total_bytes as u64, BINARY)
            );
            println!("Architecture:     {}", stats.architecture);
            println!("Layers:           {}", stats.num_layers);
            println!("Hidden size:      {}", stats.hidden_size);
            println!();
            println!("{}", "✓ Q4K import successful".green().bold());
            println!(
                "{}",
                "  Model ready for fused kernel inference (30+ tok/s CPU target)".dimmed()
            );
            Ok(())
        }
        Err(e) => {
            println!();
            println!("{}", "✗ Q4K import failed".red().bold());
            Err(CliError::ValidationFailed(e.to_string()))
        }
    }
}
