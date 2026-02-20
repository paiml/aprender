//! Rosetta Stone command implementation
//!
//! Universal model format converter (PMAT-ROSETTA-001)
//!
//! Toyota Way principles:
//! - Genchi Genbutsu: Inspect actual tensor data before/after conversion
//! - Jidoka: Stop on any conversion anomaly
//! - Kaizen: Multi-step chains for iterative improvement

use crate::error::{CliError, Result};
use crate::output;
use aprender::format::rosetta::{
    ConversionOptions, ConversionPath, ConversionReport, FormatType, InspectionReport,
    RosettaStone, TensorInfo, VerificationReport,
};
use clap::Subcommand;
use colored::Colorize;
use std::fmt::Write;
use std::path::{Path, PathBuf};

/// Rosetta Stone subcommands
#[derive(Subcommand, Debug, Clone)]
pub enum RosettaCommands {
    /// Inspect a model file (detect format, list tensors)
    Inspect {
        /// Path to model file (GGUF, SafeTensors, or APR)
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Show hex dump of file header
        #[arg(long)]
        hexdump: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Convert between model formats
    Convert {
        /// Source model file
        #[arg(value_name = "SOURCE")]
        source: PathBuf,

        /// Target output file (format inferred from extension)
        #[arg(value_name = "TARGET")]
        target: PathBuf,

        /// Apply quantization during conversion (int8, int4, fp16)
        #[arg(long)]
        quantize: Option<String>,

        /// Verify conversion with round-trip check
        #[arg(long)]
        verify: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// External tokenizer.json for weights-only models (PMAT-232)
        #[arg(long, value_name = "TOKENIZER")]
        tokenizer: Option<PathBuf>,
    },

    /// Execute multi-step conversion chain
    Chain {
        /// Source model file
        #[arg(value_name = "SOURCE")]
        source: PathBuf,

        /// Format chain (e.g., gguf safetensors apr)
        #[arg(value_name = "FORMATS", num_args = 2..)]
        formats: Vec<String>,

        /// Working directory for intermediate files
        #[arg(long, default_value = "./rosetta-work")]
        work_dir: PathBuf,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Verify round-trip conversion preserves tensor data
    Verify {
        /// Source model file
        #[arg(value_name = "SOURCE")]
        source: PathBuf,

        /// Intermediate format for round-trip (gguf, safetensors, apr)
        #[arg(long, default_value = "safetensors")]
        intermediate: String,

        /// Tolerance for numerical differences (default: 1e-5)
        #[arg(long, default_value = "1e-5")]
        tolerance: f32,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Compare inference outputs between two models (PMAT-114)
    CompareInference {
        /// Reference model (typically GGUF)
        #[arg(value_name = "MODEL_A")]
        model_a: PathBuf,

        /// Test model (typically APR)
        #[arg(value_name = "MODEL_B")]
        model_b: PathBuf,

        /// Test prompt
        #[arg(long, default_value = "2+2=")]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value = "5")]
        max_tokens: usize,

        /// Sampling temperature (0 = greedy)
        #[arg(long, default_value = "0")]
        temperature: f32,

        /// Logit difference tolerance
        #[arg(long, default_value = "0.1")]
        tolerance: f32,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Diff tensors between two models to detect layout mismatches (GH-188)
    ///
    /// GGML convention stores weights as [in_dim, out_dim] while most ML frameworks
    /// expect [out_dim, in_dim]. This command detects such mismatches that cause
    /// garbage output (like PAD token floods - GH-186).
    DiffTensors {
        /// Reference model (typically GGUF - the one that works)
        #[arg(value_name = "MODEL_A")]
        model_a: PathBuf,

        /// Test model (typically APR - the one producing garbage)
        #[arg(value_name = "MODEL_B")]
        model_b: PathBuf,

        /// Only show tensors with dimension mismatches
        #[arg(long)]
        mismatches_only: bool,

        /// Show first N values from each tensor for comparison
        #[arg(long, default_value = "0")]
        show_values: usize,

        /// Filter tensors by name pattern (e.g., "embed", "lm_head", "layer.0")
        #[arg(long)]
        filter: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Generate per-tensor statistical fingerprints (PMAT-201, JAX-STAT-001)
    ///
    /// Computes mean, std, min, max, percentiles, and checksum for each tensor.
    /// Used to detect silent corruption that passes structural checks but produces
    /// garbage output (GH-186 class bugs).
    Fingerprint {
        /// Model file to fingerprint
        #[arg(value_name = "MODEL")]
        model: PathBuf,

        /// Second model to compare (optional - enables diff mode)
        #[arg(value_name = "MODEL_B")]
        model_b: Option<PathBuf>,

        /// Output fingerprints to JSON file
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Filter tensors by name pattern
        #[arg(long)]
        filter: Option<String>,

        /// Show detailed statistics for each tensor
        #[arg(long)]
        verbose: bool,

        /// Output as JSON (to stdout if no --output specified)
        #[arg(long)]
        json: bool,
    },

    /// Validate tensor statistics against reference or expected values (PMAT-202)
    ///
    /// Compares actual tensor statistics to reference model or stored fingerprints.
    /// Reports anomalies where values deviate by more than threshold.
    ValidateStats {
        /// Model to validate
        #[arg(value_name = "MODEL")]
        model: PathBuf,

        /// Reference model for comparison
        #[arg(long)]
        reference: Option<PathBuf>,

        /// Fingerprint JSON file for comparison
        #[arg(long)]
        fingerprints: Option<PathBuf>,

        /// Deviation threshold in standard deviations (default: 3.0)
        #[arg(long, default_value = "3.0")]
        threshold: f32,

        /// Use role-specific thresholds (stricter for LayerNorm, looser for embeddings)
        #[arg(long)]
        strict: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

/// Run the rosetta inspect subcommand
pub fn run_inspect(file: &Path, hexdump: bool, json: bool) -> Result<()> {
    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }

    let rosetta = RosettaStone::new();

    let report = rosetta
        .inspect(file)
        .map_err(|e| CliError::ValidationFailed(format!("Inspection failed: {e}")))?;

    if json {
        print_inspection_json(&report);
    } else {
        print_inspection_report(&report, hexdump);
    }

    Ok(())
}

/// Run the rosetta convert subcommand
pub fn run_convert(
    source: &Path,
    target: &Path,
    quantize: Option<&str>,
    verify: bool,
    json: bool,
    tokenizer: Option<&Path>,
) -> Result<()> {
    if !source.exists() {
        return Err(CliError::FileNotFound(source.to_path_buf()));
    }

    let options = ConversionOptions {
        quantization: quantize.map(String::from),
        verify,
        tokenizer_path: tokenizer.map(PathBuf::from),
        ..Default::default()
    };

    let rosetta = RosettaStone::with_options(options.clone());

    if !json {
        println!("{}", "=== Rosetta Stone Conversion ===".cyan().bold());
        println!();
        println!("Source: {}", source.display());
        println!("Target: {}", target.display());
        if let Some(q) = quantize {
            println!("Quantization: {q}");
        }
        println!();
    }

    // Inspect source first
    if !json {
        println!("{}", "--- Source Inspection ---".yellow());
    }
    let source_report = rosetta
        .inspect(source)
        .map_err(|e| CliError::ValidationFailed(format!("Source inspection failed: {e}")))?;

    if !json {
        print_inspection_summary(&source_report);
        println!();
        println!("{}", "Converting...".yellow());
    }

    // Perform conversion
    let report = rosetta
        .convert(source, target, Some(options))
        .map_err(|e| CliError::ValidationFailed(format!("Conversion failed: {e}")))?;

    if json {
        print_conversion_json(&report.path, &source_report, &report.target_inspection);
    } else {
        print_conversion_summary(&report);
    }

    Ok(())
}

/// Run the rosetta chain subcommand
#[allow(clippy::disallowed_methods)]
pub fn run_chain(source: &Path, formats: &[String], work_dir: &Path, json: bool) -> Result<()> {
    if !source.exists() {
        return Err(CliError::FileNotFound(source.to_path_buf()));
    }

    let chain = parse_format_chain(formats)?;
    validate_chain_no_cycles(&chain)?;

    std::fs::create_dir_all(work_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot create work directory: {e}")))?;

    if !json {
        let path = ConversionPath::chain(
            chain[0],
            chain[1..chain.len() - 1].to_vec(),
            chain[chain.len() - 1],
        );
        println!("{}", "=== Rosetta Stone Chain Conversion ===".cyan().bold());
        println!();
        println!("Source: {}", source.display());
        println!("Chain: {path}");
        println!("Work Dir: {}", work_dir.display());
        println!();
    }

    let rosetta = RosettaStone::new();
    let reports = rosetta
        .chain(source, &chain, work_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Chain conversion failed: {e}")))?;

    print_chain_results(&chain, &reports, json);
    Ok(())
}

/// Parse format strings into typed format chain.
fn parse_format_chain(formats: &[String]) -> Result<Vec<FormatType>> {
    let chain: Vec<FormatType> = formats
        .iter()
        .map(|s| match s.to_lowercase().as_str() {
            "gguf" => Ok(FormatType::Gguf),
            "safetensors" | "st" => Ok(FormatType::SafeTensors),
            "apr" => Ok(FormatType::Apr),
            other => Err(CliError::ValidationFailed(format!(
                "Unknown format: {other}. Supported: gguf, safetensors, apr"
            ))),
        })
        .collect::<Result<Vec<_>>>()?;

    if chain.len() < 2 {
        return Err(CliError::ValidationFailed(
            "Chain must have at least 2 formats".to_string(),
        ));
    }
    Ok(chain)
}

/// Validate that the conversion chain has no cycles.
fn validate_chain_no_cycles(chain: &[FormatType]) -> Result<()> {
    let path = ConversionPath::chain(
        chain[0],
        chain[1..chain.len() - 1].to_vec(),
        chain[chain.len() - 1],
    );
    if path.has_cycle() {
        return Err(CliError::ValidationFailed(
            "Conversion chain contains a cycle (repeated format in middle)".to_string(),
        ));
    }
    Ok(())
}

/// Print chain conversion results in JSON or text format.
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn print_chain_results(
    chain: &[FormatType],
    reports: &[aprender::format::rosetta::ConversionReport],
    json: bool,
) {
    if json {
        let steps: Vec<serde_json::Value> = reports
            .iter()
            .enumerate()
            .map(|(i, r)| {
                serde_json::json!({
                    "step": i + 1,
                    "path": r.path.to_string(),
                    "duration_ms": r.duration_ms,
                    "source_tensors": r.source_inspection.tensors.len(),
                    "target_tensors": r.target_inspection.tensors.len(),
                    "warnings": r.warnings,
                    "lossless": r.is_lossless()
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "chain": chain.iter().map(|f| format!("{f:?}")).collect::<Vec<_>>(),
                "steps": steps,
                "total_steps": reports.len(),
                "success": true
            })
        );
    } else {
        for (i, report) in reports.iter().enumerate() {
            println!("{}", format!("--- Step {} ---", i + 1).yellow());
            println!("Path: {}", report.path);
            println!("Duration: {}ms", report.duration_ms);
            println!(
                "Tensors: {} -> {}",
                report.source_inspection.tensors.len(),
                report.target_inspection.tensors.len()
            );
            println!();
        }
        println!("{}", "Chain conversion complete".green().bold());
    }
}

include!("inference.rs");
include!("rosetta_part_03.rs");
include!("rosetta_part_04.rs");
include!("rosetta_part_05.rs");
include!("fingerprints.rs");
include!("inference_result.rs");
include!("rosetta_part_08.rs");
