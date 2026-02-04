//! Import command implementation
//!
//! Implements APR-SPEC §13: Import/Convert Pipeline
//!
//! Downloads models from HuggingFace, converts to APR format with inline validation.

use crate::error::{CliError, Result};
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
) -> Result<()> {
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
            "{} --preserve-q4k requires the 'inference' feature. \
             Falling back to standard import (Q4K will be dequantized to F32).",
            "[WARN]".yellow()
        );
    }

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
        validation: if strict {
            ValidationConfig::Strict
        } else {
            ValidationConfig::Basic
        },
        quantize: parse_quantize(quantize)?,
        compress: None,
        strict,
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
    println!(
        "[DEBUG-CLI] About to call apr_import with source={}",
        source
    );

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
mod tests {
    use super::*;
    use aprender::format::converter::QuantizationType;

    // =========================================================================
    // derive_output_path() tests
    // =========================================================================

    #[test]
    fn test_derive_output_path_hf_repo() {
        let result = derive_output_path("hf://Qwen/Qwen2.5-Coder-1.5B-Instruct").unwrap();
        assert_eq!(result, PathBuf::from("Qwen2.5-Coder-1.5B-Instruct.apr"));
    }

    #[test]
    fn test_derive_output_path_hf_with_file() {
        let result =
            derive_output_path("hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/model-q4k.gguf")
                .unwrap();
        assert_eq!(result, PathBuf::from("model-q4k.apr"));
    }

    #[test]
    fn test_derive_output_path_local_gguf() {
        let result = derive_output_path("/path/to/model.gguf").unwrap();
        assert_eq!(result, PathBuf::from("model.apr"));
    }

    #[test]
    fn test_derive_output_path_local_safetensors() {
        let result = derive_output_path("model.safetensors").unwrap();
        assert_eq!(result, PathBuf::from("model.apr"));
    }

    #[test]
    fn test_derive_output_path_url() {
        let result = derive_output_path("https://example.com/models/qwen-1.5b.gguf").unwrap();
        assert_eq!(result, PathBuf::from("qwen-1.5b.apr"));
    }

    #[test]
    fn test_derive_output_path_url_no_extension() {
        let result = derive_output_path("https://example.com/models/mymodel").unwrap();
        assert_eq!(result, PathBuf::from("mymodel.apr"));
    }

    #[test]
    fn test_derive_output_path_hf_nested_file() {
        let result = derive_output_path("hf://openai/whisper-tiny/pytorch_model.bin").unwrap();
        assert_eq!(result, PathBuf::from("pytorch_model.apr"));
    }

    #[test]
    fn test_derive_output_path_relative_path() {
        let result = derive_output_path("./models/test.safetensors").unwrap();
        assert_eq!(result, PathBuf::from("test.apr"));
    }

    // =========================================================================
    // parse_quantize() tests
    // =========================================================================

    #[test]
    fn test_parse_quantize_none() {
        let result = parse_quantize(None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_quantize_int8() {
        let result = parse_quantize(Some("int8")).unwrap();
        assert_eq!(result, Some(QuantizationType::Int8));
    }

    #[test]
    fn test_parse_quantize_int4() {
        let result = parse_quantize(Some("int4")).unwrap();
        assert_eq!(result, Some(QuantizationType::Int4));
    }

    #[test]
    fn test_parse_quantize_fp16() {
        let result = parse_quantize(Some("fp16")).unwrap();
        assert_eq!(result, Some(QuantizationType::Fp16));
    }

    #[test]
    fn test_parse_quantize_q4k() {
        let result = parse_quantize(Some("q4k")).unwrap();
        assert_eq!(result, Some(QuantizationType::Q4K));
    }

    #[test]
    fn test_parse_quantize_q4_k_underscore() {
        let result = parse_quantize(Some("q4_k")).unwrap();
        assert_eq!(result, Some(QuantizationType::Q4K));
    }

    #[test]
    fn test_parse_quantize_unknown() {
        let result = parse_quantize(Some("q8_0"));
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Unknown quantization"));
                assert!(msg.contains("Supported: int8, int4, fp16, q4k"));
            }
            other => panic!("Expected ValidationFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_quantize_invalid() {
        let result = parse_quantize(Some("notaquant"));
        assert!(result.is_err());
    }

    // =========================================================================
    // run() error cases tests
    // =========================================================================

    #[test]
    fn test_run_unknown_architecture() {
        let result = run(
            "hf://test/model",
            Some(Path::new("output.apr")),
            Some("unknown_arch"), // Invalid architecture
            None,
            false,
            false,
        );

        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Unknown architecture"));
                assert!(msg.contains("Supported: whisper, llama, bert, qwen2, auto"));
            }
            other => panic!("Expected ValidationFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_run_with_whisper_arch() {
        // This will fail at import stage but tests architecture parsing
        let result = run(
            "hf://test/model",
            Some(Path::new("output.apr")),
            Some("whisper"),
            None,
            false,
            false,
        );

        // Will fail at network stage, not architecture parsing
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_llama_arch() {
        // This will fail at import stage but tests architecture parsing
        let result = run(
            "hf://test/model",
            Some(Path::new("output.apr")),
            Some("llama"),
            None,
            false,
            false,
        );

        // Will fail at network stage, not architecture parsing
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_bert_arch() {
        // This will fail at import stage but tests architecture parsing
        let result = run(
            "hf://test/model",
            Some(Path::new("output.apr")),
            Some("bert"),
            None,
            false,
            false,
        );

        // Will fail at network stage, not architecture parsing
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_qwen2_arch() {
        // This will fail at import stage but tests architecture parsing
        let result = run(
            "hf://test/model",
            Some(Path::new("output.apr")),
            Some("qwen2"),
            None,
            false,
            false,
        );

        // Will fail at network stage, not architecture parsing
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_auto_arch() {
        // This will fail at import stage but tests architecture parsing
        let result = run(
            "hf://test/model",
            Some(Path::new("output.apr")),
            Some("auto"),
            None,
            false,
            false,
        );

        // Will fail at network stage, not architecture parsing
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_quantize_option() {
        // This will fail at import stage but tests quantize parsing
        let result = run(
            "hf://test/model",
            Some(Path::new("output.apr")),
            None,
            Some("int8"),
            false,
            false,
        );

        // Will fail at network stage, not quantize parsing
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_force_flag() {
        // This will fail at import stage but tests force flag
        let result = run(
            "hf://test/model",
            Some(Path::new("output.apr")),
            None,
            None,
            true, // force
            false,
        );

        // Will fail at network stage
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_source() {
        // Empty source should fail
        let result = run("", Some(Path::new("output.apr")), None, None, false, false);

        assert!(result.is_err());
    }

    // =========================================================================
    // Source parsing tests (via derive_output_path)
    // =========================================================================

    #[test]
    fn test_source_parse_huggingface_basic() {
        let source = Source::parse("hf://openai/whisper-tiny").unwrap();
        match source {
            Source::HuggingFace { org, repo, file } => {
                assert_eq!(org, "openai");
                assert_eq!(repo, "whisper-tiny");
                assert!(file.is_none());
            }
            _ => panic!("Expected HuggingFace source"),
        }
    }

    #[test]
    fn test_source_parse_huggingface_with_file() {
        let source = Source::parse("hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/model.gguf").unwrap();
        match source {
            Source::HuggingFace { org, repo, file } => {
                assert_eq!(org, "Qwen");
                assert_eq!(repo, "Qwen2.5-0.5B-Instruct-GGUF");
                assert_eq!(file, Some("model.gguf".to_string()));
            }
            _ => panic!("Expected HuggingFace source"),
        }
    }

    #[test]
    fn test_source_parse_local() {
        let source = Source::parse("/path/to/model.safetensors").unwrap();
        match source {
            Source::Local(path) => {
                assert_eq!(path, PathBuf::from("/path/to/model.safetensors"));
            }
            _ => panic!("Expected Local source"),
        }
    }

    #[test]
    fn test_source_parse_url() {
        let source = Source::parse("https://example.com/model.gguf").unwrap();
        match source {
            Source::Url(url) => {
                assert_eq!(url, "https://example.com/model.gguf");
            }
            _ => panic!("Expected URL source"),
        }
    }
}
