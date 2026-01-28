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
        // Print JSON output for chain conversion
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

/// Run the rosetta compare-inference subcommand (PMAT-114)
///
/// Compare inference outputs between two models to debug parity issues.
/// Runs the same prompt through both models and compares logits/outputs.
pub fn run_compare_inference(
    model_a: &Path,
    model_b: &Path,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    tolerance: f32,
    json: bool,
) -> Result<()> {
    if !model_a.exists() {
        return Err(CliError::FileNotFound(model_a.to_path_buf()));
    }
    if !model_b.exists() {
        return Err(CliError::FileNotFound(model_b.to_path_buf()));
    }

    if !json {
        println!(
            "{}",
            "╔══════════════════════════════════════════════════════════════════════════════╗"
                .cyan()
        );
        println!(
            "{}",
            "║                     INFERENCE COMPARISON REPORT (PMAT-114)                   ║"
                .cyan()
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        println!(
            "║ Model A: {:<66} ║",
            truncate_path(model_a.display().to_string(), 66)
        );
        println!(
            "║ Model B: {:<66} ║",
            truncate_path(model_b.display().to_string(), 66)
        );
        println!("║ Prompt: {:?}{} ║", prompt, " ".repeat(59_usize.saturating_sub(prompt.len())));
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
    }

    // Run Model A with APR_TRACE_LOGITS to capture logit data
    if !json {
        println!("{}", "Running Model A...".yellow());
    }

    let result_a = run_model_with_logits(model_a, prompt, max_tokens, temperature)?;

    if !json {
        println!("{}", "Running Model B...".yellow());
    }

    let result_b = run_model_with_logits(model_b, prompt, max_tokens, temperature)?;

    // Compare results
    let total_tokens = result_a.tokens.len().min(result_b.tokens.len());
    let mut mismatches = 0;

    if !json {
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        println!(
            "{}",
            "║                           TOKEN-BY-TOKEN COMPARISON                           ║"
                .cyan()
        );
        println!(
            "{}",
            "╠───────┬─────────────────────────────────┬────────────────────────────────┬───╣"
                .cyan()
        );
        println!(
            "{}",
            "║ Pos   │ Model A (top-1)                 │ Model B (top-1)                │ Δ ║"
                .cyan()
        );
        println!(
            "{}",
            "╠───────┼─────────────────────────────────┼────────────────────────────────┼───╣"
                .cyan()
        );
    }

    for i in 0..total_tokens {
        let token_a = result_a.tokens.get(i).copied().unwrap_or(0);
        let token_b = result_b.tokens.get(i).copied().unwrap_or(0);

        let logit_a = result_a.logits.get(i).copied().unwrap_or(0.0);
        let logit_b = result_b.logits.get(i).copied().unwrap_or(0.0);

        let matches = token_a == token_b;
        if !matches {
            mismatches += 1;
        }

        let status = if matches { "✓" } else { "✗" };
        let status_colored = if matches {
            status.green()
        } else {
            status.red()
        };

        if !json {
            println!(
                "║ {:<5} │ token={:<5} logit={:<12.2} │ token={:<5} logit={:<11.2} │ {} ║",
                i, token_a, logit_a, token_b, logit_b, status_colored
            );

            // Show top-5 if mismatch
            if !matches {
                if let Some(top5_a) = result_a.top5.get(i) {
                    println!(
                        "║       │ Top-5: {:<24} │{:<32} │   ║",
                        format!("{:?}", top5_a),
                        ""
                    );
                }
                if let Some(top5_b) = result_b.top5.get(i) {
                    println!(
                        "║       │{:<33} │ Top-5: {:<23} │   ║",
                        "",
                        format!("{:?}", top5_b)
                    );
                }
            }
        }
    }

    if !json {
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );

        // Diagnosis
        println!(
            "{}",
            "║                           DIAGNOSIS                                           ║"
                .cyan()
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );

        if mismatches == 0 {
            println!(
                "║ {:<76} ║",
                "All tokens match - models produce identical output"
                    .green()
                    .bold()
            );
        } else {
            println!(
                "║ {:<76} ║",
                format!(
                    "{}/{} tokens differ - see possible causes below",
                    mismatches, total_tokens
                )
                .yellow()
            );
            println!("║ Possible causes:                                                              ║");
            println!(
                "║   1. Precision difference (F32 vs Q4K): logit variance ~0.5                  ║"
            );
            println!(
                "║   2. RoPE type mismatch: Qwen2 needs rope_type=2 (NEOX style)                ║"
            );
            println!(
                "║   3. Missing QKV bias: check APR has qkv_proj.bias tensors                   ║"
            );
            println!(
                "║   4. LayerNorm epsilon: check rms_norm_eps matches (1e-6 for Qwen2)          ║"
            );
        }

        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );

        // Result
        let match_rate = if total_tokens > 0 {
            1.0 - (mismatches as f32 / total_tokens as f32)
        } else {
            0.0
        };

        let result_text = if mismatches == 0 {
            "RESULT: INFERENCE MATCH (100%)".green().bold().to_string()
        } else if match_rate >= (1.0 - tolerance) {
            format!(
                "RESULT: PARTIAL MATCH ({:.0}% within tolerance {:.0}%)",
                match_rate * 100.0,
                tolerance * 100.0
            )
            .yellow()
            .bold()
            .to_string()
        } else {
            format!(
                "RESULT: INFERENCE MISMATCH ({}/{} tokens = {:.0}%)",
                mismatches,
                total_tokens,
                match_rate * 100.0
            )
            .red()
            .bold()
            .to_string()
        };
        println!("║ {:<76} ║", result_text);
        println!(
            "{}",
            "╚══════════════════════════════════════════════════════════════════════════════╝"
                .cyan()
        );

        // Show actual outputs
        println!();
        println!("{}", "=== Generated Text ===".cyan().bold());
        println!("Model A: {:?}", result_a.output_text);
        println!("Model B: {:?}", result_b.output_text);
    } else {
        // JSON output
        let match_rate = if total_tokens > 0 {
            1.0 - (mismatches as f64 / total_tokens as f64)
        } else {
            0.0
        };

        println!("{{");
        println!("  \"model_a\": \"{}\",", model_a.display());
        println!("  \"model_b\": \"{}\",", model_b.display());
        println!("  \"prompt\": {:?},", prompt);
        println!("  \"total_tokens\": {},", total_tokens);
        println!("  \"mismatches\": {},", mismatches);
        println!("  \"match_rate\": {:.4},", match_rate);
        println!("  \"text_a\": {:?},", result_a.output_text);
        println!("  \"text_b\": {:?},", result_b.output_text);
        println!("  \"passed\": {}", mismatches == 0 || (1.0 - match_rate as f32) <= tolerance);
        println!("}}");
    }

    if mismatches > 0 {
        let match_rate = 1.0 - (mismatches as f32 / total_tokens.max(1) as f32);
        if match_rate < (1.0 - tolerance) {
            return Err(CliError::ValidationFailed(format!(
                "Inference mismatch: {}/{} tokens differ ({:.0}% match rate, need {:.0}%)",
                mismatches,
                total_tokens,
                match_rate * 100.0,
                (1.0 - tolerance) * 100.0
            )));
        }
    }

    Ok(())
}

/// Inference result with logit data
struct InferenceResult {
    tokens: Vec<u32>,
    logits: Vec<f32>,
    top5: Vec<Vec<u32>>,
    output_text: String,
}

/// Run a model and capture output
fn run_model_with_logits(
    model_path: &Path,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> Result<InferenceResult> {
    use std::process::{Command, Stdio};

    // Determine which command to use based on file extension
    let ext = model_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    // Use realizar for both GGUF and APR files for consistent comparison
    // Use the full path to avoid picking up old versions from PATH
    let realizar_path = std::env::var("REALIZAR_PATH")
        .unwrap_or_else(|_| "realizar".to_string());
    let output = Command::new(&realizar_path)
        .arg("run")
        .arg(model_path)
        .arg(prompt)
        .arg("--max-tokens")
        .arg(max_tokens.to_string())
        .arg("--temperature")
        .arg(temperature.to_string())
        .arg("--format")
        .arg("text")
        .env("APR_TRACE_LOGITS", "1")
        .env("NO_COLOR", "1")
        .env("TERM", "dumb")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to run realizar: {e}")))?;

    let (stdout_text, stderr_text) = (
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    );

    // Combine stdout and stderr for parsing (trace goes to both)
    let combined = format!("{}\n{}", stdout_text, stderr_text);

    let mut tokens = Vec::new();
    let mut logits = Vec::new();
    let mut top5 = Vec::new();

    // Parse PMAT-113-F trace lines
    for line in combined.lines() {
        // Handle "Selected token: X (logit: Y)" lines
        if line.contains("Selected token:") {
            if let Some(token_part) = line.split("Selected token:").nth(1) {
                let trimmed = token_part.trim();
                if let Some(paren_pos) = trimmed.find(" (") {
                    if let Ok(token_id) = trimmed[..paren_pos].parse::<u32>() {
                        tokens.push(token_id);
                    }
                    if let Some(logit_start) = trimmed.find("logit:") {
                        let logit_str = &trimmed[logit_start + 6..];
                        if let Some(end) = logit_str.find(')') {
                            if let Ok(logit) = logit_str[..end].trim().parse::<f32>() {
                                logits.push(logit);
                            }
                        }
                    }
                }
            }
        }
        // Handle "Top 5 tokens:" lines
        if line.contains("Top 5 tokens:") {
            if let Some(top5_part) = line.split("Top 5 tokens:").nth(1) {
                let mut current_top5 = Vec::new();
                for pair in top5_part.split("),") {
                    if let Some(open_paren) = pair.find('(') {
                        let inner = &pair[open_paren + 1..];
                        if let Some(comma) = inner.find(',') {
                            if let Ok(token_id) = inner[..comma].trim().parse::<u32>() {
                                current_top5.push(token_id);
                            }
                        }
                    }
                }
                if !current_top5.is_empty() {
                    top5.push(current_top5);
                }
            }
        }
    }

    // Extract output text - both GGUF and APR use realizar, so same parsing
    // Strip ANSI codes and spinner characters, get last non-empty line
    let clean = strip_ansi(&stdout_text);
    let output_text = clean
        .lines()
        .filter(|l| !l.is_empty() && !l.contains('⠋') && !l.contains('⠙') && !l.contains('⠹') && !l.contains('⠸'))
        .last()
        .unwrap_or("")
        .trim()
        .to_string();
    let _ = ext; // Suppress unused warning

    Ok(InferenceResult {
        tokens,
        logits,
        top5,
        output_text,
    })
}

/// Strip ANSI escape codes from text
fn strip_ansi(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip escape sequence
            if chars.peek() == Some(&'[') {
                chars.next(); // consume '['
                // Skip until we hit a letter
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Truncate path for display
fn truncate_path(path: String, max_len: usize) -> String {
    if path.len() <= max_len {
        path
    } else {
        format!("...{}", &path[path.len() - max_len + 3..])
    }
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
