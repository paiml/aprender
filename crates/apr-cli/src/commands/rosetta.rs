//! Rosetta Stone command implementation
//!
//! Universal model format converter (PMAT-ROSETTA-001)
//!
//! Toyota Way principles:
//! - Genchi Genbutsu: Inspect actual tensor data before/after conversion
//! - Jidoka: Stop on any conversion anomaly
//! - Kaizen: Multi-step chains for iterative improvement

use crate::error::{CliError, Result};
use aprender::format::rosetta::{
    ConversionOptions, ConversionPath, FormatType, InspectionReport, RosettaStone,
    VerificationReport,
};
use clap::Subcommand;
use colored::Colorize;
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
) -> Result<()> {
    if !source.exists() {
        return Err(CliError::FileNotFound(source.to_path_buf()));
    }

    let options = ConversionOptions {
        quantization: quantize.map(String::from),
        verify,
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
        println!();
        println!("{}", "--- Target Inspection ---".yellow());
        print_inspection_summary(&report.target_inspection);
        println!();

        // Summary
        println!("{}", "=== Conversion Summary ===".cyan().bold());
        println!("Path: {}", report.path);
        println!("Duration: {}ms", report.duration_ms);
        println!(
            "Tensors: {} -> {}",
            report.source_inspection.tensors.len(),
            report.target_inspection.tensors.len()
        );

        if report.is_lossless() && report.tensor_counts_match() {
            println!();
            println!("{}", "Conversion successful".green().bold());
        } else {
            println!();
            if !report.tensor_counts_match() {
                println!(
                    "{}",
                    "Warning: Tensor count changed during conversion".yellow()
                );
            }
            if !report.is_lossless() {
                println!(
                    "{}",
                    format!("Warning: {} tensors dropped", report.dropped_tensors.len()).yellow()
                );
            }
        }
    }

    Ok(())
}

/// Run the rosetta chain subcommand
pub fn run_chain(source: &Path, formats: &[String], work_dir: &Path, json: bool) -> Result<()> {
    if !source.exists() {
        return Err(CliError::FileNotFound(source.to_path_buf()));
    }

    // Parse format strings
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

    // Check for cycles
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

    // Create work directory
    std::fs::create_dir_all(work_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot create work directory: {e}")))?;

    let rosetta = RosettaStone::new();

    if !json {
        println!("{}", "=== Rosetta Stone Chain Conversion ===".cyan().bold());
        println!();
        println!("Source: {}", source.display());
        println!("Chain: {path}");
        println!("Work Dir: {}", work_dir.display());
        println!();
    }

    let reports = rosetta
        .chain(source, &chain, work_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Chain conversion failed: {e}")))?;

    if json {
        // TODO: Print JSON output for chain
        println!("{{\"steps\": {}}}", reports.len());
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

    Ok(())
}

/// Run the rosetta verify subcommand
pub fn run_verify(source: &Path, intermediate: &str, tolerance: f32, json: bool) -> Result<()> {
    if !source.exists() {
        return Err(CliError::FileNotFound(source.to_path_buf()));
    }

    let intermediate_format = match intermediate.to_lowercase().as_str() {
        "gguf" => FormatType::Gguf,
        "safetensors" | "st" => FormatType::SafeTensors,
        "apr" => FormatType::Apr,
        other => {
            return Err(CliError::ValidationFailed(format!(
                "Unknown format: {other}. Supported: gguf, safetensors, apr"
            )));
        }
    };

    let rosetta = RosettaStone::new();

    if !json {
        println!(
            "{}",
            "=== Rosetta Stone Round-Trip Verification ==="
                .cyan()
                .bold()
        );
        println!();
        println!("Source: {}", source.display());
        println!("Intermediate: {intermediate_format}");
        println!("Tolerance: {tolerance}");
        println!();
        println!("{}", "Verifying round-trip...".yellow());
    }

    let report = rosetta
        .verify_roundtrip(source, intermediate_format)
        .map_err(|e| CliError::ValidationFailed(format!("Verification failed: {e}")))?;

    if json {
        print_verification_json(&report);
    } else {
        println!();
        println!("{}", "=== Verification Report ===".cyan().bold());
        println!("Equivalent: {}", report.is_equivalent);
        println!("Max Diff: {:.2e}", report.max_diff);
        println!("Mean Diff: {:.2e}", report.mean_diff);

        if !report.failed_tensors.is_empty() {
            println!();
            println!("{}", "Failed tensors:".red());
            for t in &report.failed_tensors {
                println!("  - {t}");
            }
        }

        println!();
        if report.passes_with_tolerance(tolerance) {
            println!("{}", "Round-trip verification PASSED".green().bold());
        } else {
            println!("{}", "Round-trip verification FAILED".red().bold());
        }
    }

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

fn print_inspection_report(report: &InspectionReport, hexdump: bool) {
    println!("{}", "=== Rosetta Stone Inspection ===".cyan().bold());
    println!();
    println!("Format: {}", report.format);
    println!("File Size: {} bytes", report.file_size);
    println!("Total Parameters: {}", report.total_params);

    if let Some(ref arch) = report.architecture {
        println!("Architecture: {arch}");
    }
    if let Some(ref quant) = report.quantization {
        println!("Quantization: {quant}");
    }

    println!();
    println!("{}", "--- Metadata ---".yellow());
    for (k, v) in &report.metadata {
        let display_v = if v.len() > 60 {
            format!("{}...", &v[..60])
        } else {
            v.clone()
        };
        println!("  {k}: {display_v}");
    }

    println!();
    println!(
        "{}",
        format!("--- Tensors ({} total) ---", report.tensors.len()).yellow()
    );
    for (i, t) in report.tensors.iter().enumerate() {
        if i < 10 || i >= report.tensors.len() - 2 {
            println!(
                "  {}: {} {:?} ({} bytes)",
                t.name, t.dtype, t.shape, t.size_bytes
            );
        } else if i == 10 {
            println!("  ... ({} more tensors) ...", report.tensors.len() - 12);
        }
    }

    if hexdump {
        println!();
        println!("{}", "--- Hexdump (first 64 bytes) ---".yellow());
        println!("  (Use 'apr hex <file>' for full hex dump)");
    }
}

fn print_inspection_summary(report: &InspectionReport) {
    println!("  Format: {}", report.format);
    println!("  File Size: {} bytes", report.file_size);
    println!("  Tensors: {}", report.tensors.len());
    println!("  Parameters: {}", report.total_params);
    if let Some(ref arch) = report.architecture {
        println!("  Architecture: {arch}");
    }
    if let Some(ref quant) = report.quantization {
        println!("  Quantization: {quant}");
    }
}

fn print_inspection_json(report: &InspectionReport) {
    // Simple JSON output
    println!("{{");
    println!("  \"format\": \"{}\",", report.format);
    println!("  \"file_size\": {},", report.file_size);
    println!("  \"total_params\": {},", report.total_params);
    println!("  \"tensor_count\": {},", report.tensors.len());
    if let Some(ref arch) = report.architecture {
        println!("  \"architecture\": \"{arch}\",");
    }
    if let Some(ref quant) = report.quantization {
        println!("  \"quantization\": \"{quant}\",");
    }
    println!("  \"metadata_keys\": {}", report.metadata.len());
    println!("}}");
}

fn print_conversion_json(
    path: &ConversionPath,
    source: &InspectionReport,
    target: &InspectionReport,
) {
    println!("{{");
    println!("  \"path\": \"{path}\",");
    println!("  \"source\": {{");
    println!("    \"format\": \"{}\",", source.format);
    println!("    \"tensors\": {}", source.tensors.len());
    println!("  }},");
    println!("  \"target\": {{");
    println!("    \"format\": \"{}\",", target.format);
    println!("    \"tensors\": {}", target.tensors.len());
    println!("  }}");
    println!("}}");
}

fn print_verification_json(report: &VerificationReport) {
    println!("{{");
    println!("  \"is_equivalent\": {},", report.is_equivalent);
    println!("  \"max_diff\": {},", report.max_diff);
    println!("  \"mean_diff\": {},", report.mean_diff);
    println!("  \"failed_tensors\": {}", report.failed_tensors.len());
    println!("}}");
}
