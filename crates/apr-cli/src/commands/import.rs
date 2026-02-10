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
            None,  // tokenizer
            false, // enforce_provenance
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
            None,  // tokenizer
            false, // enforce_provenance
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
            None,  // tokenizer
            false, // enforce_provenance
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
            None,  // tokenizer
            false, // enforce_provenance
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
            None,  // tokenizer
            false, // enforce_provenance
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
            None,  // tokenizer
            false, // enforce_provenance
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
            None,  // tokenizer
            false, // enforce_provenance
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
            None,  // tokenizer
            false, // enforce_provenance
        );

        // Will fail at network stage
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_source() {
        // Empty source should fail
        let result = run(
            "",
            Some(Path::new("output.apr")),
            None,
            None,
            false,
            false,
            None,
            false, // enforce_provenance
        );

        assert!(result.is_err());
    }

    // =========================================================================
    // F-GT-001: --enforce-provenance tests
    // =========================================================================

    #[test]
    fn t_f_gt_001_enforce_provenance_rejects_gguf_source() {
        let result = run(
            "model.gguf",
            Some(Path::new("output.apr")),
            None,
            None,
            false,
            false,
            None,
            true, // enforce_provenance = ON
        );
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("F-GT-001"),
            "Error must cite F-GT-001 gate: {err_msg}"
        );
        assert!(
            err_msg.contains("provenance"),
            "Error must mention provenance: {err_msg}"
        );
    }

    #[test]
    fn t_f_gt_001_enforce_provenance_rejects_gguf_hub_pattern() {
        // Hub-style paths with -GGUF suffix should also be rejected
        let result = run(
            "hf://TheBloke/Qwen2.5-Coder-7B-GGUF",
            Some(Path::new("output.apr")),
            None,
            None,
            false,
            false,
            None,
            true, // enforce_provenance = ON
        );
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("F-GT-001"),
            "Error must cite F-GT-001 gate: {err_msg}"
        );
    }

    #[test]
    fn t_f_gt_001_no_provenance_allows_gguf() {
        // Without --enforce-provenance, GGUF should NOT be rejected
        // (it will fail for other reasons like file not found, but NOT F-GT-001)
        let result = run(
            "model.gguf",
            Some(Path::new("output.apr")),
            None,
            None,
            false,
            false,
            None,
            false, // enforce_provenance = OFF
        );
        // Should fail (file doesn't exist) but NOT with F-GT-001
        if let Err(e) = &result {
            let err_msg = format!("{e}");
            assert!(
                !err_msg.contains("F-GT-001"),
                "Without --enforce-provenance, F-GT-001 must not trigger: {err_msg}"
            );
        }
    }

    #[test]
    fn t_f_gt_001_enforce_provenance_allows_safetensors() {
        // SafeTensors source should pass provenance check (fail later for file not found)
        let result = run(
            "model.safetensors",
            Some(Path::new("output.apr")),
            None,
            None,
            false,
            false,
            None,
            true, // enforce_provenance = ON
        );
        // Should fail (file doesn't exist) but NOT with F-GT-001
        if let Err(e) = &result {
            let err_msg = format!("{e}");
            assert!(
                !err_msg.contains("F-GT-001"),
                "SafeTensors must pass provenance check: {err_msg}"
            );
        }
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
