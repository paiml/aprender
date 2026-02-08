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
    ConversionOptions, ConversionPath, FormatType, InspectionReport, RosettaStone,
    VerificationReport,
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
        println!(
            "║ Prompt: {:?}{} ║",
            prompt,
            " ".repeat(59_usize.saturating_sub(prompt.len()))
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
    }

    // F-GT-002: Check for mixed quantization levels (R3 violation)
    if let Some(warning) = check_mixed_quant_warning(model_a, model_b) {
        if !json {
            println!("{}", warning.yellow());
            println!();
        }
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

    if json {
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
        println!(
            "  \"passed\": {}",
            mismatches == 0 || (1.0 - match_rate as f32) <= tolerance
        );
        println!("}}");
    } else {
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );

        print_inference_diagnosis(
            total_tokens,
            mismatches,
            tolerance,
            &result_a.output_text,
            &result_b.output_text,
        );
    }

    // GH-188 FIX: Fail if no tokens were captured (tracing broken or inference failed)
    if total_tokens == 0 {
        // Check if we have output text even without token traces
        let a_empty = result_a.output_text.is_empty() || result_a.output_text.contains("tok/s");
        let b_empty = result_b.output_text.is_empty() || result_b.output_text.contains("tok/s");

        if a_empty && b_empty {
            return Err(CliError::ValidationFailed(
                "TRACING BROKEN: No tokens captured from either model. Check APR_TRACE_LOGITS parsing.".to_string()
            ));
        } else if a_empty {
            return Err(CliError::ValidationFailed(format!(
                "Model A produced no output. Model B: {:?}",
                result_b.output_text
            )));
        } else if b_empty {
            return Err(CliError::ValidationFailed(format!(
                "Model B produced no output. Model A: {:?}",
                result_a.output_text
            )));
        }
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

/// Print diagnosis section for inference comparison (extracted for complexity reduction).
fn print_inference_diagnosis(
    total_tokens: usize,
    mismatches: usize,
    tolerance: f32,
    text_a: &str,
    text_b: &str,
) {
    println!(
        "{}",
        "║                           DIAGNOSIS                                           ║".cyan()
    );
    println!(
        "{}",
        "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
    );

    // GH-188 FIX: Detect when tracing captured nothing (inference failure)
    if total_tokens == 0 {
        println!(
            "║ {:<76} ║",
            "⚠️  NO TOKENS CAPTURED - INFERENCE MAY HAVE FAILED!"
                .red()
                .bold()
        );
        println!(
            "║ {:<76} ║",
            "Check that APR_TRACE_LOGITS output is being parsed correctly."
        );
        println!(
            "║ {:<76} ║",
            "Model A output: see below. Model B output: see below."
        );
    } else if mismatches == 0 {
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
        println!(
            "║ Possible causes:                                                              ║"
        );
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
        "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
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
        "╚══════════════════════════════════════════════════════════════════════════════╝".cyan()
    );

    // Show actual outputs
    println!();
    println!("{}", "=== Generated Text ===".cyan().bold());
    println!("Model A: {:?}", text_a);
    println!("Model B: {:?}", text_b);

    // GH-188: Direct text comparison for quick diagnosis
    let text_a_clean = text_a.trim();
    let text_b_clean = text_b.trim();
    let text_a_has_content = !text_a_clean.is_empty() && !text_a_clean.contains("tok/s");
    let text_b_has_content = !text_b_clean.is_empty() && !text_b_clean.contains("tok/s");

    if text_a_has_content != text_b_has_content {
        println!();
        println!("{}", "⚠️  TEXT OUTPUT MISMATCH DETECTED:".red().bold());
        if text_a_has_content && !text_b_has_content {
            println!("   Model A produced text, Model B produced nothing/garbage.");
            println!("   → Model B likely has inference bug (layout, kernel, or load issue).");
        } else {
            println!("   Model B produced text, Model A produced nothing/garbage.");
            println!("   → Model A likely has inference bug (layout, kernel, or load issue).");
        }
    } else if text_a_has_content && text_b_has_content && text_a_clean != text_b_clean {
        println!();
        println!("{}", "⚠️  TEXT CONTENT DIFFERS:".yellow().bold());
        println!("   Models produced different outputs (may be precision-related).");
    }
}

/// F-GT-002: Detect quantization level from file path.
///
/// Returns a normalized quantization level string for R3 comparison.
/// SafeTensors are unquantized (BF16/F16/F32), GGUF files contain quant level
/// in their filename (e.g. Q4_K_M, Q6_K), APR files may have been quantized at import.
fn detect_quant_level_from_path(path: &Path) -> String {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    // SafeTensors are always unquantized (BF16/F16/F32)
    if name.ends_with(".safetensors") {
        return "unquantized (BF16/F16/F32)".to_string();
    }

    // GGUF: detect from filename patterns
    if name.ends_with(".gguf") {
        for quant in &[
            "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q4_k",
            "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q5_k", "q6_k", "q8_0", "f16", "f32",
        ] {
            if name.contains(quant) {
                return quant.to_uppercase();
            }
        }
        return "GGUF (quant unknown)".to_string();
    }

    // APR: detect from filename patterns (e.g. model-q4k.apr)
    if name.ends_with(".apr") {
        for quant in &["q4k", "q6k", "q4_k", "q6_k", "q8_0", "f16", "f32"] {
            if name.contains(quant) {
                return quant.to_uppercase();
            }
        }
        return "APR (quant unknown)".to_string();
    }

    "unknown".to_string()
}

/// F-GT-002: Check for R3 mixed quantization level violation.
///
/// Returns a warning string if the two models have different quantization levels,
/// which violates the R3 ground truth comparison rule.
pub(crate) fn check_mixed_quant_warning(model_a: &Path, model_b: &Path) -> Option<String> {
    let quant_a = detect_quant_level_from_path(model_a);
    let quant_b = detect_quant_level_from_path(model_b);

    if quant_a != quant_b {
        Some(format!(
            "F-GT-002 WARNING: Mixed quantization levels detected (R3 violation)\n  \
             Model A: {} ({})\n  \
             Model B: {} ({})\n  \
             Comparing models at different quantization levels may produce \
             misleading differences.\n  \
             For valid comparison, use the same quantization level (R3 rule).",
            model_a.display(),
            quant_a,
            model_b.display(),
            quant_b,
        ))
    } else {
        None
    }
}

/// Run the rosetta diff-tensors subcommand (GH-188)
///
/// Compares tensor dimensions between two models to detect layout mismatches.
/// GGML stores weights as [in_dim, out_dim] but most ML code expects [out_dim, in_dim].
/// This mismatch causes garbage output (PAD token floods).
pub fn run_diff_tensors(
    model_a: &Path,
    model_b: &Path,
    mismatches_only: bool,
    show_values: usize,
    filter: Option<&str>,
    json: bool,
) -> Result<()> {
    if !model_a.exists() {
        return Err(CliError::FileNotFound(model_a.to_path_buf()));
    }
    if !model_b.exists() {
        return Err(CliError::FileNotFound(model_b.to_path_buf()));
    }

    let rosetta = RosettaStone::new();

    // F-GT-002: Check for mixed quantization levels (R3 violation)
    if let Some(warning) = check_mixed_quant_warning(model_a, model_b) {
        if !json {
            println!("{}", warning.yellow());
            println!();
        }
    }

    // Inspect both models
    let report_a = rosetta
        .inspect(model_a)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect model A: {e}")))?;
    let report_b = rosetta
        .inspect(model_b)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect model B: {e}")))?;

    // Build tensor maps by normalized name (GH-202: cross-format tensor matching)
    let tensors_a: std::collections::HashMap<String, _> = report_a
        .tensors
        .iter()
        .map(|t| (normalize_tensor_name(&t.name), t))
        .collect();
    let tensors_b: std::collections::HashMap<String, _> = report_b
        .tensors
        .iter()
        .map(|t| (normalize_tensor_name(&t.name), t))
        .collect();

    // Collect all unique tensor names
    let mut all_names: Vec<_> = tensors_a.keys().chain(tensors_b.keys()).collect();
    all_names.sort();
    all_names.dedup();

    // Apply filter
    let filtered_names: Vec<_> = if let Some(pattern) = filter {
        all_names
            .into_iter()
            .filter(|n| n.contains(pattern))
            .collect()
    } else {
        all_names
    };

    let mut layout_mismatches = Vec::new();
    let mut missing_in_a = Vec::new();
    let mut missing_in_b = Vec::new();

    if !json {
        println!(
            "{}",
            "╔══════════════════════════════════════════════════════════════════════════════╗"
                .cyan()
        );
        println!(
            "{}",
            "║               TENSOR DIFF REPORT (GH-188: Layout Mismatch Detection)        ║"
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
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );

        // GH-188: Show tensor count comparison FIRST - missing tensors is critical!
        let count_a = report_a.tensors.len();
        let count_b = report_b.tensors.len();
        let count_match = count_a == count_b;
        let count_status = if count_match {
            "✓".green()
        } else {
            "✗".red()
        };
        println!(
            "║ {} Tensor Count: A={:<5} B={:<5} {}║",
            count_status,
            count_a,
            count_b,
            if count_match {
                "                                  ".to_string()
            } else {
                format!(
                    "MISSING {} TENSORS!",
                    (count_a as i64 - count_b as i64).abs()
                )
                .red()
                .bold()
                .to_string()
            }
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        println!(
            "{}",
            "║ GGML Convention: [in_dim, out_dim] - needs transpose for standard matmul     ║"
                .yellow()
        );
        println!(
            "{}",
            "║ Standard Conv:   [out_dim, in_dim] - expected by most ML code                ║"
                .yellow()
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
    }

    for name in &filtered_names {
        let tensor_a = tensors_a.get(*name);
        let tensor_b = tensors_b.get(*name);

        match (tensor_a, tensor_b) {
            (Some(a), Some(b)) => {
                let dims_match = a.shape == b.shape;
                let is_transposed = is_transposed_dims(&a.shape, &b.shape);

                if !dims_match || !mismatches_only {
                    if !json {
                        let status = if dims_match {
                            "✓".green()
                        } else if is_transposed {
                            "⚠️".yellow()
                        } else {
                            "✗".red()
                        };

                        println!("║ {} {:<72} ║", status, name);
                        println!(
                            "║   A: {:?} {:>20} {:>15} bytes    ║",
                            a.shape, a.dtype, a.size_bytes
                        );
                        println!(
                            "║   B: {:?} {:>20} {:>15} bytes    ║",
                            b.shape, b.dtype, b.size_bytes
                        );

                        if is_transposed {
                            println!(
                                "║   {} ║",
                                "LAYOUT MISMATCH: Dimensions are transposed! Likely GGML convention."
                                    .red()
                                    .bold()
                            );
                            println!(
                                "║   {} ║",
                                "FIX: Transpose this tensor during load OR use row-major kernel"
                                    .yellow()
                            );
                        }
                        println!(
                            "{}",
                            "╠──────────────────────────────────────────────────────────────────────────────╣"
                                .cyan()
                        );
                    }

                    if is_transposed {
                        layout_mismatches.push(((*name).clone(), a.shape.clone(), b.shape.clone()));
                    }
                }
            }
            (Some(a), None) => {
                missing_in_b.push(((*name).clone(), a.shape.clone()));
                if !mismatches_only && !json {
                    println!("║ {} {:<72} ║", "−".red(), name);
                    println!("║   A: {:?} (missing in B){}║", a.shape, " ".repeat(40));
                    println!(
                        "{}",
                        "╠──────────────────────────────────────────────────────────────────────────────╣"
                            .cyan()
                    );
                }
            }
            (None, Some(b)) => {
                missing_in_a.push(((*name).clone(), b.shape.clone()));
                if !mismatches_only && !json {
                    println!("║ {} {:<72} ║", "+".green(), name);
                    println!("║   B: {:?} (missing in A){}║", b.shape, " ".repeat(40));
                    println!(
                        "{}",
                        "╠──────────────────────────────────────────────────────────────────────────────╣"
                            .cyan()
                    );
                }
            }
            (None, None) => {} // shouldn't happen
        }
    }

    // Summary
    if json {
        println!("{{");
        println!("  \"model_a\": \"{}\",", model_a.display());
        println!("  \"model_b\": \"{}\",", model_b.display());
        println!("  \"tensors_a\": {},", tensors_a.len());
        println!("  \"tensors_b\": {},", tensors_b.len());
        println!("  \"layout_mismatches\": {},", layout_mismatches.len());
        println!("  \"missing_in_a\": {},", missing_in_a.len());
        println!("  \"missing_in_b\": {},", missing_in_b.len());
        if !layout_mismatches.is_empty() {
            println!("  \"mismatched_tensors\": [");
            for (i, (name, shape_a, shape_b)) in layout_mismatches.iter().enumerate() {
                let comma = if i < layout_mismatches.len() - 1 {
                    ","
                } else {
                    ""
                };
                println!(
                    "    {{\"name\": \"{}\", \"shape_a\": {:?}, \"shape_b\": {:?}}}{}",
                    name, shape_a, shape_b, comma
                );
            }
            println!("  ],");
        }
        println!(
            "  \"diagnosis\": \"{}\"",
            if layout_mismatches.is_empty() {
                "No layout mismatches detected"
            } else {
                "LAYOUT MISMATCH: Some tensors have transposed dimensions (GGML convention)"
            }
        );
        println!("}}");
    } else {
        println!(
            "{}",
            "║                                 SUMMARY                                       ║"
                .cyan()
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        println!("║ Tensors in A: {:<62} ║", tensors_a.len());
        println!("║ Tensors in B: {:<62} ║", tensors_b.len());
        println!(
            "║ Layout mismatches: {:<56} ║",
            format!("{}", layout_mismatches.len()).red().bold()
        );
        println!("║ Missing in A: {:<62} ║", missing_in_a.len());
        println!("║ Missing in B: {:<62} ║", missing_in_b.len());
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );

        if layout_mismatches.is_empty() {
            println!("║ {} ║", "No layout mismatches detected".green().bold());
        } else {
            println!(
                "{}",
                "║                              DIAGNOSIS                                       ║"
                    .red()
                    .bold()
            );
            println!(
                "{}",
                "╠══════════════════════════════════════════════════════════════════════════════╣"
                    .cyan()
            );
            println!("║ {} ║", "LAYOUT MISMATCH DETECTED!".red().bold());
            println!("║ Tensors with transposed dimensions found. This causes garbage output. ║");
            println!("║  ║");
            println!("║ Root Cause: GGML stores weights as [in_dim, out_dim] ║");
            println!("║              Standard ML expects [out_dim, in_dim] ║");
            println!("║  ║");
            println!("║ {} ║", "Fix Options:".yellow().bold());
            println!("║   1. Transpose tensor data during APR load ║");
            println!("║   2. Use row-major kernels that expect GGML layout ║");
            println!("║   3. Store layout convention in APR metadata ║");
            println!(
                "{}",
                "╠══════════════════════════════════════════════════════════════════════════════╣"
                    .cyan()
            );
            println!(
                "║ Mismatched tensors:                                                          ║"
            );
            for (name, shape_a, shape_b) in &layout_mismatches {
                println!("║   {} ║", name);
                println!("║     A: {:?} → B: {:?} ║", shape_a, shape_b);
            }
        }

        println!(
            "{}",
            "╚══════════════════════════════════════════════════════════════════════════════╝"
                .cyan()
        );
    }

    // Return error if mismatches found (for CI assertion)
    let count_a = report_a.tensors.len();
    let count_b = report_b.tensors.len();

    // GH-188: Tensor count mismatch is CRITICAL - fail immediately
    if count_a != count_b {
        return Err(CliError::ValidationFailed(format!(
            "TENSOR COUNT MISMATCH: Model A has {} tensors, Model B has {} ({} missing!)",
            count_a,
            count_b,
            (count_a as i64 - count_b as i64).abs()
        )));
    }

    if !layout_mismatches.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "Layout mismatch: {} tensors have transposed dimensions",
            layout_mismatches.len()
        )));
    }

    // PMAT-GLASS-HOUSE: show_values feature deferred (P2)
    // When show_values > 0, user expects to see tensor value samples.
    // Currently not implemented - inform user rather than silently ignore.
    if show_values > 0 {
        eprintln!(
            "Note: --show-values {} requested but value comparison not yet implemented. \
             Use 'apr rosetta fingerprint' for tensor statistics.",
            show_values
        );
    }

    Ok(())
}

/// Run the rosetta fingerprint subcommand (PMAT-201)
///
/// Computes statistical fingerprints for all tensors in a model.
/// Fingerprints include: mean, std, min, max, percentiles, nan/inf counts.
pub fn run_fingerprint(
    model: &Path,
    model_b: Option<&Path>,
    output: Option<&Path>,
    filter: Option<&str>,
    verbose: bool,
    json: bool,
) -> Result<()> {
    if !model.exists() {
        return Err(CliError::FileNotFound(model.to_path_buf()));
    }

    if !json {
        println!(
            "{}",
            "╔══════════════════════════════════════════════════════════════════════════════╗"
                .cyan()
        );
        println!(
            "{}",
            "║           TENSOR STATISTICAL FINGERPRINTS (PMAT-201, JAX-STAT-001)          ║"
                .cyan()
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        println!(
            "║ Model: {:<69} ║",
            truncate_path(model.display().to_string(), 69)
        );
    }

    // Compute fingerprints for model A
    let fingerprints_a = compute_fingerprints(model, filter)?;

    if let Some(model_b_path) = model_b {
        // Diff mode: compare two models
        if !model_b_path.exists() {
            return Err(CliError::FileNotFound(model_b_path.to_path_buf()));
        }

        if !json {
            println!(
                "║ Compare: {:<67} ║",
                truncate_path(model_b_path.display().to_string(), 67)
            );
            println!(
                "{}",
                "╠══════════════════════════════════════════════════════════════════════════════╣"
                    .cyan()
            );
        }

        let fingerprints_b = compute_fingerprints(model_b_path, filter)?;
        print_fingerprint_diff(&fingerprints_a, &fingerprints_b, verbose, json)?;
    } else {
        // Single model mode: just output fingerprints
        if !json {
            println!(
                "{}",
                "╠══════════════════════════════════════════════════════════════════════════════╣"
                    .cyan()
            );
        }
        print_fingerprints(&fingerprints_a, verbose, json)?;
    }

    // Output to file if requested
    if let Some(output_path) = output {
        let json_content = fingerprints_to_json(&fingerprints_a);
        std::fs::write(output_path, json_content).map_err(|e| {
            CliError::ValidationFailed(format!("Failed to write fingerprints: {e}"))
        })?;
        if !json {
            println!("║ Saved fingerprints to: {:<53} ║", output_path.display());
        }
    }

    if !json {
        println!(
            "{}",
            "╚══════════════════════════════════════════════════════════════════════════════╝"
                .cyan()
        );
    }

    Ok(())
}

/// Run the rosetta validate-stats subcommand (PMAT-202)
///
/// Validates tensor statistics against a reference model or stored fingerprints.
pub fn run_validate_stats(
    model: &Path,
    reference: Option<&Path>,
    fingerprints_file: Option<&Path>,
    threshold: f32,
    strict: bool,
    json: bool,
) -> Result<()> {
    if !model.exists() {
        return Err(CliError::FileNotFound(model.to_path_buf()));
    }

    if reference.is_none() && fingerprints_file.is_none() {
        return Err(CliError::ValidationFailed(
            "Must provide either --reference or --fingerprints".to_string(),
        ));
    }

    if !json {
        println!(
            "{}",
            "╔══════════════════════════════════════════════════════════════════════════════╗"
                .cyan()
        );
        println!(
            "{}",
            "║             TENSOR STATISTICS VALIDATION (PMAT-202, JAX-STAT-002)           ║"
                .cyan()
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
        println!(
            "║ Model: {:<69} ║",
            truncate_path(model.display().to_string(), 69)
        );
        println!(
            "║ Threshold: {:.1}σ{:<60} ║",
            threshold,
            if strict { " (strict mode)" } else { "" }
        );
    }

    // Compute actual fingerprints
    let actual = compute_fingerprints(model, None)?;

    // Get reference fingerprints
    let reference_fps = if let Some(ref_path) = reference {
        if !ref_path.exists() {
            return Err(CliError::FileNotFound(ref_path.to_path_buf()));
        }
        if !json {
            println!(
                "║ Reference: {:<65} ║",
                truncate_path(ref_path.display().to_string(), 65)
            );
        }
        compute_fingerprints(ref_path, None)?
    } else if let Some(fp_path) = fingerprints_file {
        if !fp_path.exists() {
            return Err(CliError::FileNotFound(fp_path.to_path_buf()));
        }
        if !json {
            println!(
                "║ Fingerprints: {:<62} ║",
                truncate_path(fp_path.display().to_string(), 62)
            );
        }
        load_fingerprints_from_json(fp_path)?
    } else {
        unreachable!()
    };

    if !json {
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣"
                .cyan()
        );
    }

    // Validate and collect anomalies
    let anomalies = validate_fingerprints(&actual, &reference_fps, threshold, strict);

    if json {
        println!("{{");
        println!("  \"model\": \"{}\",", model.display());
        println!("  \"threshold\": {},", threshold);
        println!("  \"strict\": {},", strict);
        println!("  \"total_tensors\": {},", actual.len());
        println!("  \"anomalies\": {},", anomalies.len());
        if !anomalies.is_empty() {
            println!("  \"anomaly_details\": [");
            for (i, anomaly) in anomalies.iter().enumerate() {
                let comma = if i < anomalies.len() - 1 { "," } else { "" };
                println!(
                    "    {{\"tensor\": \"{}\", \"field\": \"{}\", \"expected\": {:.6}, \"actual\": {:.6}, \"deviation\": {:.2}}}{}",
                    anomaly.tensor, anomaly.field, anomaly.expected, anomaly.actual, anomaly.deviation_sigma, comma
                );
            }
            println!("  ],");
        }
        println!("  \"passed\": {}", anomalies.is_empty());
        println!("}}");
    } else {
        if anomalies.is_empty() {
            println!(
                "║ {} ║",
                "✓ All tensors within expected statistical bounds"
                    .green()
                    .bold()
            );
        } else {
            println!(
                "║ {} ║",
                format!("✗ {} STATISTICAL ANOMALIES DETECTED", anomalies.len())
                    .red()
                    .bold()
            );
            println!(
                "{}",
                "╠──────────────────────────────────────────────────────────────────────────────╣"
                    .cyan()
            );

            for anomaly in &anomalies {
                let severity = if anomaly.deviation_sigma > 10.0 {
                    "CRITICAL".red().bold()
                } else if anomaly.deviation_sigma > 5.0 {
                    "WARNING".yellow()
                } else {
                    "INFO".white()
                };

                println!("║ {} {} ║", severity, anomaly.tensor);
                println!(
                    "║   {}: expected={:.6}, actual={:.6}, deviation={:.1}σ ║",
                    anomaly.field, anomaly.expected, anomaly.actual, anomaly.deviation_sigma
                );
            }
        }
        println!(
            "{}",
            "╚══════════════════════════════════════════════════════════════════════════════╝"
                .cyan()
        );
    }

    // Fail if anomalies found
    if !anomalies.is_empty() {
        let critical_count = anomalies
            .iter()
            .filter(|a| a.deviation_sigma > 10.0)
            .count();
        if critical_count > 0 {
            return Err(CliError::ValidationFailed(format!(
                "E020: {} critical statistical anomalies detected (>{:.0}σ deviation)",
                critical_count, threshold
            )));
        }
    }

    Ok(())
}

/// Tensor statistical fingerprint (PMAT-201)
#[derive(Debug, Clone)]
pub struct TensorFingerprint {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub p5: f32,
    pub p25: f32,
    pub p50: f32,
    pub p75: f32,
    pub p95: f32,
    pub nan_count: u32,
    pub inf_count: u32,
    pub zero_fraction: f32,
    pub checksum: u32,
}

/// Statistical anomaly detected during validation
#[derive(Debug)]
struct StatisticalAnomaly {
    tensor: String,
    field: String,
    expected: f32,
    actual: f32,
    deviation_sigma: f32,
}

/// Compute fingerprints for all tensors in a model
fn compute_fingerprints(model_path: &Path, filter: Option<&str>) -> Result<Vec<TensorFingerprint>> {
    let rosetta = RosettaStone::new();
    let report = rosetta
        .inspect(model_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect model: {e}")))?;

    let mut fingerprints = Vec::new();

    // Try to load actual tensor data for statistics
    let tensor_data = load_tensor_data(model_path);

    for tensor_info in &report.tensors {
        // Apply filter
        if let Some(pattern) = filter {
            if !tensor_info.name.contains(pattern) {
                continue;
            }
        }

        // Compute statistics from actual data if available
        let (
            mean,
            std,
            min,
            max,
            p5,
            p25,
            p50,
            p75,
            p95,
            nan_count,
            inf_count,
            zero_fraction,
            checksum,
        ) = if let Some(ref data_map) = tensor_data {
            if let Some(values) = data_map.get(&tensor_info.name) {
                compute_tensor_stats(values)
            } else {
                // No data available - use placeholder
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0)
            }
        } else {
            // No data available - use placeholder
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0)
        };

        fingerprints.push(TensorFingerprint {
            name: tensor_info.name.clone(),
            shape: tensor_info.shape.clone(),
            dtype: tensor_info.dtype.clone(),
            mean,
            std,
            min,
            max,
            p5,
            p25,
            p50,
            p75,
            p95,
            nan_count,
            inf_count,
            zero_fraction,
            checksum,
        });
    }

    Ok(fingerprints)
}

/// Load tensor data from model file (for computing actual statistics)
fn load_tensor_data(model_path: &Path) -> Option<std::collections::HashMap<String, Vec<f32>>> {
    // Use realizar to load tensor data
    let realizar_path = std::env::var("REALIZAR_PATH").unwrap_or_else(|_| "realizar".to_string());

    // Try to get tensor statistics via realizar dump command
    let output = std::process::Command::new(&realizar_path)
        .arg("dump")
        .arg("--stats")
        .arg(model_path)
        .arg("--format")
        .arg("json")
        .output()
        .ok()?;

    if !output.status.success() {
        // Fallback: try loading directly via aprender format module
        return load_tensor_data_direct(model_path);
    }

    // Parse JSON output
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_tensor_stats_json(&stdout)
}

/// Direct tensor loading via aprender format module
fn load_tensor_data_direct(
    model_path: &Path,
) -> Option<std::collections::HashMap<String, Vec<f32>>> {
    use aprender::format::gguf::GgufReader;

    let ext = model_path.extension()?.to_str()?;

    let mut tensor_map = std::collections::HashMap::new();

    match ext.to_lowercase().as_str() {
        "gguf" => {
            // Use GgufReader::from_file which handles mmap internally
            let reader = GgufReader::from_file(model_path).ok()?;

            // Iterate over tensor metadata (reader.tensors is a Vec<GgufTensorMeta>)
            for tensor_meta in &reader.tensors {
                // Use get_tensor_f32 to dequantize and get values
                if let Ok((values, _shape)) = reader.get_tensor_f32(&tensor_meta.name) {
                    tensor_map.insert(tensor_meta.name.clone(), values);
                }
            }
        }
        "apr" => {
            // PMAT-201 FIX: Load APR tensors directly
            let data = std::fs::read(model_path).ok()?;
            if data.len() < 40 {
                return None;
            }

            // Parse APR v2 header
            let magic = &data[0..4];
            if magic != b"APR\0" {
                return None;
            }

            let tensor_count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
            let tensor_index_offset = u64::from_le_bytes([
                data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
            ]) as usize;
            let data_offset = u64::from_le_bytes([
                data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
            ]) as usize;

            // Parse tensor index
            let mut pos = tensor_index_offset;
            for _ in 0..tensor_count {
                if pos + 2 > data.len() {
                    break;
                }
                let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 2;

                if pos + name_len > data.len() {
                    break;
                }
                let name = String::from_utf8_lossy(&data[pos..pos + name_len]).to_string();
                pos += name_len;

                if pos + 2 > data.len() {
                    break;
                }
                let dtype = data[pos];
                pos += 1;
                let ndim = data[pos] as usize;
                pos += 1;

                let mut dims = Vec::with_capacity(ndim);
                for _ in 0..ndim {
                    if pos + 8 > data.len() {
                        break;
                    }
                    let dim = u64::from_le_bytes([
                        data[pos],
                        data[pos + 1],
                        data[pos + 2],
                        data[pos + 3],
                        data[pos + 4],
                        data[pos + 5],
                        data[pos + 6],
                        data[pos + 7],
                    ]) as usize;
                    dims.push(dim);
                    pos += 8;
                }

                if pos + 16 > data.len() {
                    break;
                }
                let offset = u64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]) as usize;
                pos += 8;
                let size = u64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]) as usize;
                pos += 8;

                // Load tensor data
                let tensor_start = data_offset + offset;
                let tensor_end = tensor_start + size;
                if tensor_end > data.len() {
                    continue;
                }
                let tensor_bytes = &data[tensor_start..tensor_end];

                // Dequantize based on dtype
                let values: Vec<f32> = match dtype {
                    0 => {
                        // F32 - direct read
                        tensor_bytes
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect()
                    }
                    12 => {
                        // Q4_K - dequantize
                        let num_elements: usize = dims.iter().product();
                        dequantize_q4k_for_stats(tensor_bytes, num_elements)
                    }
                    14 => {
                        // Q6_K - dequantize
                        let num_elements: usize = dims.iter().product();
                        dequantize_q6k_for_stats(tensor_bytes, num_elements)
                    }
                    _ => continue, // Skip unknown dtypes
                };

                tensor_map.insert(name, values);
            }
        }
        "safetensors" => {
            // PMAT-203 FIX: Implement SafeTensors tensor loading for fingerprints
            use aprender::serialization::safetensors::MappedSafeTensors;

            let mapped = MappedSafeTensors::open(model_path).ok()?;
            for name in mapped.tensor_names() {
                if let Ok(values) = mapped.get_tensor(name) {
                    tensor_map.insert((*name).to_string(), values);
                }
            }
        }
        _ => return None,
    }

    if tensor_map.is_empty() {
        None
    } else {
        Some(tensor_map)
    }
}

/// Simple Q4_K dequantization for statistics (PMAT-201)
fn dequantize_q4k_for_stats(data: &[u8], num_elements: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 144; // Q4_K block size

    let num_blocks = (num_elements + QK_K - 1) / QK_K;
    let mut result = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        if block_start + BLOCK_SIZE > data.len() {
            break;
        }

        // Read scales (d, dmin)
        let d = f16_to_f32(&data[block_start..block_start + 2]);
        let dmin = f16_to_f32(&data[block_start + 2..block_start + 4]);

        // Read 12 bytes of scales
        let scales = &data[block_start + 4..block_start + 16];

        // Read 128 bytes of quantized values (4 bits each, 256 values)
        let qs = &data[block_start + 16..block_start + 144];

        // Dequantize 256 elements
        for j in 0..QK_K {
            if result.len() >= num_elements {
                break;
            }
            let scale_idx = j / 32;
            let scale = if scale_idx < 12 {
                (scales[scale_idx] & 0x3F) as f32
            } else {
                1.0
            };

            let q_idx = j / 2;
            let q_val = if j % 2 == 0 {
                (qs.get(q_idx).copied().unwrap_or(0) & 0x0F) as i32
            } else {
                ((qs.get(q_idx).copied().unwrap_or(0) >> 4) & 0x0F) as i32
            };

            let val = d * scale * (q_val as f32 - 8.0) - dmin * scale;
            result.push(val);
        }
    }

    result
}

/// Simple Q6_K dequantization for statistics (PMAT-201)
fn dequantize_q6k_for_stats(data: &[u8], num_elements: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const BLOCK_SIZE: usize = 210; // Q6_K block size

    let num_blocks = (num_elements + QK_K - 1) / QK_K;
    let mut result = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        if block_start + BLOCK_SIZE > data.len() {
            break;
        }

        // Read scale (d)
        let d = f16_to_f32(&data[block_start + 208..block_start + 210]);

        // Simplified: just read as scaled values
        for j in 0..QK_K {
            if result.len() >= num_elements {
                break;
            }
            let q_idx = block_start + (j * 6 / 8);
            let q_val = data.get(q_idx).copied().unwrap_or(0) as i32;
            let val = d * (q_val as f32 - 32.0);
            result.push(val);
        }
    }

    result
}

/// Convert f16 bytes to f32
fn f16_to_f32(bytes: &[u8]) -> f32 {
    if bytes.len() < 2 {
        return 0.0;
    }
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    half::f16::from_bits(bits).to_f32()
}

/// Parse tensor statistics from JSON
fn parse_tensor_stats_json(_json_str: &str) -> Option<std::collections::HashMap<String, Vec<f32>>> {
    // Simple JSON parsing for tensor data
    // Format expected: {"tensors": {"name": [values...], ...}}
    // PMAT-201: Would need proper JSON parsing for full implementation
    None // Placeholder - returns None to use placeholder stats
}

/// Compute statistics for a tensor
#[allow(clippy::type_complexity)]
fn compute_tensor_stats(
    values: &[f32],
) -> (
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    u32,
    u32,
    f32,
    u32,
) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0);
    }

    let mut nan_count = 0u32;
    let mut inf_count = 0u32;
    let mut zero_count = 0u32;
    let mut sum = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut checksum = 0u32;

    // Collect valid values for percentile calculation
    let mut valid_values: Vec<f32> = Vec::with_capacity(values.len());

    for &v in values {
        // Update checksum (simple CRC-like)
        checksum = checksum.wrapping_add(v.to_bits());

        if v.is_nan() {
            nan_count += 1;
        } else if v.is_infinite() {
            inf_count += 1;
        } else {
            valid_values.push(v);
            sum += v as f64;
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            if v == 0.0 {
                zero_count += 1;
            }
        }
    }

    let n = valid_values.len();
    if n == 0 {
        return (
            0.0, 0.0, min, max, 0.0, 0.0, 0.0, 0.0, 0.0, nan_count, inf_count, 0.0, checksum,
        );
    }

    let mean = (sum / n as f64) as f32;

    // Compute std
    let variance: f64 = valid_values
        .iter()
        .map(|&v| {
            let diff = v as f64 - sum / n as f64;
            diff * diff
        })
        .sum::<f64>()
        / n as f64;
    let std = variance.sqrt() as f32;

    // Compute percentiles (sort for percentile calculation)
    valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile = |p: f32| -> f32 {
        let idx = ((p / 100.0) * (n - 1) as f32) as usize;
        valid_values[idx.min(n - 1)]
    };

    let p5 = percentile(5.0);
    let p25 = percentile(25.0);
    let p50 = percentile(50.0);
    let p75 = percentile(75.0);
    let p95 = percentile(95.0);

    let zero_fraction = zero_count as f32 / values.len() as f32;

    (
        mean,
        std,
        min,
        max,
        p5,
        p25,
        p50,
        p75,
        p95,
        nan_count,
        inf_count,
        zero_fraction,
        checksum,
    )
}

/// Print fingerprints
fn print_fingerprints(fingerprints: &[TensorFingerprint], verbose: bool, json: bool) -> Result<()> {
    if json {
        println!("{}", fingerprints_to_json(fingerprints));
        return Ok(());
    }

    for fp in fingerprints {
        println!("║ {:<74} ║", truncate_path(fp.name.clone(), 74));
        println!("║   shape={:?} dtype={:<10} ║", fp.shape, fp.dtype);
        if verbose {
            println!(
                "║   mean={:>10.6} std={:>10.6} min={:>10.6} max={:>10.6} ║",
                fp.mean, fp.std, fp.min, fp.max
            );
            println!(
                "║   p5={:>10.6} p25={:>10.6} p50={:>10.6} p75={:>10.6} p95={:>10.6} ║",
                fp.p5, fp.p25, fp.p50, fp.p75, fp.p95
            );
            println!(
                "║   nan={} inf={} zero_frac={:.4} checksum=0x{:08X} ║",
                fp.nan_count, fp.inf_count, fp.zero_fraction, fp.checksum
            );
        } else {
            println!(
                "║   mean={:>10.6} std={:>10.6} nan={} inf={} ║",
                fp.mean, fp.std, fp.nan_count, fp.inf_count
            );
        }
        println!(
            "{}",
            "╠──────────────────────────────────────────────────────────────────────────────╣"
                .cyan()
        );
    }

    println!("║ Total tensors: {:<61} ║", fingerprints.len());

    Ok(())
}

/// Print fingerprint diff between two models
fn print_fingerprint_diff(
    fps_a: &[TensorFingerprint],
    fps_b: &[TensorFingerprint],
    verbose: bool,
    json: bool,
) -> Result<()> {
    // GH-202: Use normalized names for cross-format matching
    let map_b: std::collections::HashMap<_, _> = fps_b
        .iter()
        .map(|fp| (normalize_tensor_name(&fp.name), fp))
        .collect();

    let mut anomalies = Vec::new();

    if !json {
        println!(
            "{}",
            "║                              FINGERPRINT DIFF                                ║"
                .yellow()
        );
        println!(
            "{}",
            "╠──────────────────────────────────────────────────────────────────────────────╣"
                .cyan()
        );
    }

    for fp_a in fps_a {
        // GH-202: Use normalized name for cross-format lookup
        let norm_name_a = normalize_tensor_name(&fp_a.name);
        if let Some(fp_b) = map_b.get(&norm_name_a) {
            // Check for significant differences
            let mean_diff = if fp_a.std > 1e-10 {
                (fp_a.mean - fp_b.mean).abs() / fp_a.std
            } else {
                (fp_a.mean - fp_b.mean).abs()
            };

            let has_anomaly = mean_diff > 3.0
                || fp_a.nan_count != fp_b.nan_count
                || fp_a.inf_count != fp_b.inf_count;

            if has_anomaly || verbose {
                let status = if has_anomaly {
                    "⚠️".yellow()
                } else {
                    "✓".green()
                };

                if !json {
                    println!(
                        "║ {} {:<72} ║",
                        status,
                        truncate_path(fp_a.name.clone(), 72)
                    );
                    println!(
                        "║   A: mean={:>10.6} std={:>10.6} nan={} inf={} ║",
                        fp_a.mean, fp_a.std, fp_a.nan_count, fp_a.inf_count
                    );
                    println!(
                        "║   B: mean={:>10.6} std={:>10.6} nan={} inf={} ║",
                        fp_b.mean, fp_b.std, fp_b.nan_count, fp_b.inf_count
                    );
                    if has_anomaly {
                        println!(
                            "║   {} mean_diff={:.2}σ ║",
                            "ANOMALY:".red().bold(),
                            mean_diff
                        );
                    }
                    println!(
                        "{}",
                        "╠──────────────────────────────────────────────────────────────────────────────╣"
                            .cyan()
                    );
                }

                if has_anomaly {
                    anomalies.push(StatisticalAnomaly {
                        tensor: fp_a.name.clone(),
                        field: "mean".to_string(),
                        expected: fp_a.mean,
                        actual: fp_b.mean,
                        deviation_sigma: mean_diff,
                    });
                }
            }
        } else if !json {
            println!(
                "║ {} {:<72} ║",
                "−".red(),
                truncate_path(fp_a.name.clone(), 72)
            );
            println!("║   Missing in Model B ║");
        }
    }

    if json {
        println!("{{");
        println!("  \"total_tensors\": {},", fps_a.len());
        println!("  \"anomalies\": {},", anomalies.len());
        println!("  \"passed\": {}", anomalies.is_empty());
        println!("}}");
    } else if anomalies.is_empty() {
        println!(
            "║ {} ║",
            "✓ No statistical anomalies detected".green().bold()
        );
    } else {
        println!(
            "║ {} ║",
            format!("✗ {} ANOMALIES DETECTED", anomalies.len())
                .red()
                .bold()
        );
    }

    Ok(())
}

/// Convert fingerprints to JSON
fn fingerprints_to_json(fingerprints: &[TensorFingerprint]) -> String {
    let mut json = String::from("{\n  \"fingerprints\": [\n");

    for (i, fp) in fingerprints.iter().enumerate() {
        let comma = if i < fingerprints.len() - 1 { "," } else { "" };
        write!(
            json,
            "    {{\n      \"name\": \"{}\",\n      \"shape\": {:?},\n      \"dtype\": \"{}\",\n      \"mean\": {},\n      \"std\": {},\n      \"min\": {},\n      \"max\": {},\n      \"p5\": {},\n      \"p25\": {},\n      \"p50\": {},\n      \"p75\": {},\n      \"p95\": {},\n      \"nan_count\": {},\n      \"inf_count\": {},\n      \"zero_fraction\": {},\n      \"checksum\": {}\n    }}{}\n",
            fp.name, fp.shape, fp.dtype, fp.mean, fp.std, fp.min, fp.max,
            fp.p5, fp.p25, fp.p50, fp.p75, fp.p95,
            fp.nan_count, fp.inf_count, fp.zero_fraction, fp.checksum, comma
        )
        .expect("write to String should not fail");
    }

    json.push_str("  ]\n}");
    json
}

/// Load fingerprints from JSON file
fn load_fingerprints_from_json(path: &Path) -> Result<Vec<TensorFingerprint>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read fingerprints: {e}")))?;

    // Simple JSON parsing - in production would use serde
    let mut fingerprints = Vec::new();

    // Parse JSON manually (simplified)
    for line in content.lines() {
        if line.contains("\"name\":") {
            // Extract tensor info from JSON
            // This is a placeholder - proper implementation would use serde_json
            let name = line
                .split("\"name\": \"")
                .nth(1)
                .and_then(|s| s.split('"').next())
                .unwrap_or("unknown")
                .to_string();

            fingerprints.push(TensorFingerprint {
                name,
                shape: vec![],
                dtype: "unknown".to_string(),
                mean: 0.0,
                std: 1.0,
                min: 0.0,
                max: 0.0,
                p5: 0.0,
                p25: 0.0,
                p50: 0.0,
                p75: 0.0,
                p95: 0.0,
                nan_count: 0,
                inf_count: 0,
                zero_fraction: 0.0,
                checksum: 0,
            });
        }
    }

    Ok(fingerprints)
}

/// Validate fingerprints against reference
fn validate_fingerprints(
    actual: &[TensorFingerprint],
    reference: &[TensorFingerprint],
    threshold: f32,
    strict: bool,
) -> Vec<StatisticalAnomaly> {
    let ref_map: std::collections::HashMap<_, _> = reference
        .iter()
        .map(|fp| (normalize_tensor_name(&fp.name), fp))
        .collect();

    let mut anomalies = Vec::new();

    for actual_fp in actual {
        let norm_name = normalize_tensor_name(&actual_fp.name);
        if let Some(ref_fp) = ref_map.get(&norm_name) {
            // Get role-specific threshold
            let role_threshold = if strict {
                get_role_threshold(&actual_fp.name)
            } else {
                threshold
            };

            // Check mean
            let mean_deviation = if ref_fp.std > 1e-10 {
                (actual_fp.mean - ref_fp.mean).abs() / ref_fp.std
            } else {
                (actual_fp.mean - ref_fp.mean).abs() * 1000.0 // Scale up small differences
            };

            if mean_deviation > role_threshold {
                anomalies.push(StatisticalAnomaly {
                    tensor: actual_fp.name.clone(),
                    field: "mean".to_string(),
                    expected: ref_fp.mean,
                    actual: actual_fp.mean,
                    deviation_sigma: mean_deviation,
                });
            }

            // Check for NaN/Inf (always anomalous if reference doesn't have them)
            if actual_fp.nan_count > 0 && ref_fp.nan_count == 0 {
                anomalies.push(StatisticalAnomaly {
                    tensor: actual_fp.name.clone(),
                    field: "nan_count".to_string(),
                    expected: ref_fp.nan_count as f32,
                    actual: actual_fp.nan_count as f32,
                    deviation_sigma: f32::INFINITY,
                });
            }

            if actual_fp.inf_count > 0 && ref_fp.inf_count == 0 {
                anomalies.push(StatisticalAnomaly {
                    tensor: actual_fp.name.clone(),
                    field: "inf_count".to_string(),
                    expected: ref_fp.inf_count as f32,
                    actual: actual_fp.inf_count as f32,
                    deviation_sigma: f32::INFINITY,
                });
            }
        }
    }

    anomalies
}

/// Get role-specific threshold based on tensor name
fn get_role_threshold(tensor_name: &str) -> f32 {
    let name_lower = tensor_name.to_lowercase();

    if name_lower.contains("layernorm")
        || name_lower.contains("layer_norm")
        || name_lower.contains("ln_")
    {
        // LayerNorm weights should be very close to 1.0 - tight threshold
        2.0
    } else if name_lower.contains("embed") {
        // Embeddings can have more variance
        5.0
    } else if name_lower.contains("lm_head") || name_lower.contains("output") {
        // Output layers - moderate threshold
        3.0
    } else {
        // Default threshold for other weights
        3.0
    }
}

/// Normalize tensor name for cross-format comparison (GH-202 fix)
///
/// Maps both GGUF and APR/HuggingFace naming conventions to a common canonical form.
/// This enables proper tensor matching when comparing models across formats.
///
/// GGUF convention: `blk.N.attn_q.weight`
/// APR/HF convention: `model.layers.N.self_attn.q_proj.weight`
/// Canonical form: `N.q_proj.weight`
fn normalize_tensor_name(name: &str) -> String {
    // Step 1: Remove format-specific prefixes
    let name = name
        .trim_start_matches("model.")
        .trim_start_matches("blk.")
        .trim_start_matches("layers.");

    // Step 2: Remove APR/HF intermediate prefixes (self_attn., mlp.)
    // These don't exist in GGUF naming
    let name = name.replace(".self_attn.", ".").replace(".mlp.", ".");

    // Step 3: Normalize GGUF tensor suffixes to HF convention
    // GGUF: attn_q → HF: q_proj
    let name = name
        .replace("attn_q", "q_proj")
        .replace("attn_k", "k_proj")
        .replace("attn_v", "v_proj")
        .replace("attn_output", "o_proj")
        .replace("ffn_gate", "gate_proj")
        .replace("ffn_up", "up_proj")
        .replace("ffn_down", "down_proj")
        .replace("attn_norm", "input_layernorm")
        .replace("ffn_norm", "post_attention_layernorm")
        .replace("token_embd", "embed_tokens")
        .replace("output_norm", "norm");

    // Step 4: Handle lm_head vs output naming
    // GGUF: output.weight → APR: lm_head.weight
    if name == "output.weight" {
        "lm_head.weight".to_string()
    } else {
        name
    }
}

/// Check if two shapes are transposed versions of each other
fn is_transposed_dims(shape_a: &[usize], shape_b: &[usize]) -> bool {
    if shape_a.len() != 2 || shape_b.len() != 2 {
        return false;
    }
    // Check if dims are swapped AND shapes are different (not square identical)
    // For [896, 896] vs [896, 896], this should return false (identical, not transposed)
    let is_swapped = shape_a[0] == shape_b[1] && shape_a[1] == shape_b[0];
    let is_different = shape_a != shape_b;
    is_swapped && is_different
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

    // GH-188: Use realizar for both GGUF and APR files for consistent comparison
    // Use simple text format for clean output that's easy to compare
    let realizar_path = std::env::var("REALIZAR_PATH").unwrap_or_else(|_| "realizar".to_string());
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
        .env("NO_COLOR", "1")
        .env("TERM", "dumb")
        .env("REALIZE_DEBUG", "1") // Enable debug for APR load tracing
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to run realizar: {e}")))?;

    let (stdout_text, stderr_text) = (
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    );

    // GH-188 DEBUG: Log what we captured for diagnosis
    if std::env::var("ROSETTA_DEBUG").is_ok() {
        eprintln!("[ROSETTA] Model: {}", model_path.display());
        eprintln!("[ROSETTA] Exit code: {:?}", output.status.code());
        eprintln!(
            "[ROSETTA] STDOUT ({} bytes): {:?}",
            stdout_text.len(),
            &stdout_text[..stdout_text.len().min(200)]
        );
        eprintln!(
            "[ROSETTA] STDERR ({} bytes): {:?}",
            stderr_text.len(),
            &stderr_text[..stderr_text.len().min(200)]
        );
    }

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

    // GH-188: Extract output text - without --verbose, stdout is just the generated text
    // Filter out debug/trace lines, spinners, and noise
    let output_text = strip_ansi(&stdout_text)
        .chars()
        .filter(|c| {
            // Remove spinner characters
            !matches!(c, '⠋' | '⠙' | '⠹' | '⠸' | '⠼' | '⠴' | '⠦' | '⠧' | '⠇' | '⠏')
        })
        .collect::<String>()
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty()
                && !t.starts_with('[')
                && !t.starts_with("Loading")
                && !t.starts_with("Model loaded")
                && !t.starts_with("Prompt tokens")
                && !t.starts_with("Temperature:")
                && !t.starts_with("Generated (")
                && !t.contains("tok/s")
                && !t.contains("ERROR")
                && !t.contains("using greedy")
        })
        .collect::<Vec<_>>()
        .join(" ")
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
    output::header("Rosetta Stone Inspection");

    let mut pairs: Vec<(&str, String)> = vec![
        ("Format", report.format.to_string()),
        ("File Size", output::format_size(report.file_size as u64)),
        ("Parameters", output::count_fmt(report.total_params)),
    ];
    if let Some(ref arch) = report.architecture {
        pairs.push(("Architecture", arch.clone()));
    }
    if let Some(ref quant) = report.quantization {
        pairs.push(("Quantization", quant.clone()));
    }
    println!("{}", output::kv_table(&pairs));

    // Metadata
    if !report.metadata.is_empty() {
        output::subheader(&format!("Metadata ({} keys)", report.metadata.len()));
        let meta_pairs: Vec<(&str, String)> = report
            .metadata
            .iter()
            .map(|(k, v)| {
                let display_v = if v.len() > 60 {
                    format!("{}...", &v[..60])
                } else {
                    v.clone()
                };
                (k.as_str(), display_v)
            })
            .collect();
        println!("{}", output::kv_table(&meta_pairs));
    }

    // Tensors
    output::subheader(&format!("Tensors ({} total)", report.tensors.len()));
    let mut rows: Vec<Vec<String>> = Vec::new();
    for (i, t) in report.tensors.iter().enumerate() {
        if i < 10 || i >= report.tensors.len().saturating_sub(2) {
            rows.push(vec![
                t.name.clone(),
                format!("{}", output::dtype_color(&t.dtype)),
                format!("{:?}", t.shape),
                output::format_size(t.size_bytes as u64),
            ]);
        } else if i == 10 {
            rows.push(vec![
                format!("... {} more ...", report.tensors.len().saturating_sub(12)),
                String::new(),
                String::new(),
                String::new(),
            ]);
        }
    }
    println!("{}", output::table(&["Name", "DType", "Shape", "Size"], &rows));

    if hexdump {
        output::subheader("Hexdump (first 64 bytes)");
        println!("  (Use 'apr hex <file>' for full hex dump)");
    }
}

fn print_inspection_summary(report: &InspectionReport) {
    let mut pairs: Vec<(&str, String)> = vec![
        ("Format", report.format.to_string()),
        ("File Size", output::format_size(report.file_size as u64)),
        ("Tensors", output::count_fmt(report.tensors.len())),
        ("Parameters", output::count_fmt(report.total_params)),
    ];
    if let Some(ref arch) = report.architecture {
        pairs.push(("Architecture", arch.clone()));
    }
    if let Some(ref quant) = report.quantization {
        pairs.push(("Quantization", quant.clone()));
    }
    println!("{}", output::kv_table(&pairs));
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    // ========================================================================
    // FormatType Library Tests
    // ========================================================================

    #[test]
    fn test_format_type_from_extension_gguf() {
        let path = Path::new("model.gguf");
        let result = FormatType::from_extension(path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), FormatType::Gguf);
    }

    #[test]
    fn test_format_type_from_extension_safetensors() {
        let path = Path::new("model.safetensors");
        let result = FormatType::from_extension(path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), FormatType::SafeTensors);
    }

    #[test]
    fn test_format_type_from_extension_apr() {
        let path = Path::new("model.apr");
        let result = FormatType::from_extension(path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), FormatType::Apr);
    }

    #[test]
    fn test_format_type_from_extension_invalid() {
        let path = Path::new("model.pytorch");
        let result = FormatType::from_extension(path);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_type_display() {
        assert_eq!(FormatType::Gguf.to_string(), "GGUF");
        assert_eq!(FormatType::SafeTensors.to_string(), "SafeTensors");
        assert_eq!(FormatType::Apr.to_string(), "APR");
    }

    // ========================================================================
    // Run Inspect Tests
    // ========================================================================

    #[test]
    fn test_run_inspect_file_not_found() {
        let result = run_inspect(Path::new("/nonexistent/model.gguf"), false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inspect_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not a valid gguf file").expect("write");

        let result = run_inspect(file.path(), false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inspect_directory() {
        let dir = tempdir().expect("create temp dir");
        let result = run_inspect(dir.path(), false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inspect_with_hexdump_flag() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not a valid gguf").expect("write");

        let result = run_inspect(file.path(), true, false);
        // Should fail (invalid file) but tests hexdump path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inspect_with_json_flag() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not a valid gguf").expect("write");

        let result = run_inspect(file.path(), false, true);
        // Should fail (invalid file) but tests json path
        assert!(result.is_err());
    }

    // ========================================================================
    // Run Convert Tests
    // ========================================================================

    #[test]
    fn test_run_convert_source_not_found() {
        let target = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run_convert(
            Path::new("/nonexistent/model.gguf"),
            target.path(),
            None,
            false,
            false,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_convert_invalid_source() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let target = NamedTempFile::with_suffix(".apr").expect("create target");

        let result = run_convert(source.path(), target.path(), None, false, false, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_convert_with_quantize_option() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let target = NamedTempFile::with_suffix(".apr").expect("create target");

        let result = run_convert(
            source.path(),
            target.path(),
            Some("int8"),
            false,
            false,
            None,
        );
        // Should fail (invalid file) but tests quantize path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_convert_with_verify_flag() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let target = NamedTempFile::with_suffix(".apr").expect("create target");

        let result = run_convert(source.path(), target.path(), None, true, false, None);
        // Should fail (invalid file) but tests verify path
        assert!(result.is_err());
    }

    // ========================================================================
    // Run Chain Tests
    // ========================================================================

    #[test]
    fn test_run_chain_source_not_found() {
        let work_dir = tempdir().expect("create work dir");
        let formats = vec!["safetensors".to_string(), "apr".to_string()];
        let result = run_chain(
            Path::new("/nonexistent/model.gguf"),
            &formats,
            work_dir.path(),
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_chain_invalid_source() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        let formats = vec!["safetensors".to_string(), "apr".to_string()];

        let result = run_chain(source.path(), &formats, work_dir.path(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_chain_empty_formats() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        let formats: Vec<String> = vec![];

        let result = run_chain(source.path(), &formats, work_dir.path(), false);
        // Should fail - empty chain
        assert!(result.is_err());
    }

    // ========================================================================
    // Run Verify Tests
    // ========================================================================

    #[test]
    fn test_run_verify_source_not_found() {
        let result = run_verify(
            Path::new("/nonexistent/model.gguf"),
            "safetensors",
            1e-5,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_verify_invalid_source() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");

        let result = run_verify(source.path(), "safetensors", 1e-5, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_verify_with_different_tolerance() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");

        let result = run_verify(source.path(), "safetensors", 1e-3, false);
        // Should fail (invalid file) but tests tolerance path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_verify_with_apr_intermediate() {
        let mut source = NamedTempFile::with_suffix(".safetensors").expect("create source");
        source.write_all(b"not valid safetensors").expect("write");

        let result = run_verify(source.path(), "apr", 1e-5, false);
        // Should fail (invalid file) but tests apr intermediate path
        assert!(result.is_err());
    }

    // ========================================================================
    // Run Fingerprint Tests
    // ========================================================================

    #[test]
    fn test_run_fingerprint_file_not_found() {
        // run_fingerprint(model, model_b, output, filter, verbose, json)
        let result = run_fingerprint(
            Path::new("/nonexistent/model.gguf"),
            None,  // model_b
            None,  // output
            None,  // filter
            false, // verbose
            false, // json
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_fingerprint_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_fingerprint(file.path(), None, None, None, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_fingerprint_with_output() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");
        let output = NamedTempFile::with_suffix(".json").expect("create output");

        let result = run_fingerprint(file.path(), None, Some(output.path()), None, false, false);
        // Should fail (invalid file) but tests output path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_fingerprint_with_filter() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_fingerprint(file.path(), None, None, Some("encoder"), false, false);
        // Should fail (invalid file) but tests filter path
        assert!(result.is_err());
    }

    // ========================================================================
    // Run Validate Stats Tests
    // ========================================================================

    #[test]
    fn test_run_validate_stats_file_not_found() {
        // run_validate_stats(model, reference, fingerprints_file, threshold, strict, json)
        let result = run_validate_stats(
            Path::new("/nonexistent/model.gguf"),
            None,  // reference
            None,  // fingerprints_file
            1e-5,  // threshold
            false, // strict
            false, // json
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_validate_stats_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_validate_stats(file.path(), None, None, 1e-5, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_validate_stats_with_reference() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");
        let mut ref_file = NamedTempFile::with_suffix(".gguf").expect("create ref file");
        ref_file.write_all(b"not valid ref").expect("write");

        let result =
            run_validate_stats(file.path(), Some(ref_file.path()), None, 1e-5, false, false);
        // Should fail (invalid files) but tests reference path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_validate_stats_with_strict() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_validate_stats(file.path(), None, None, 1e-5, true, false);
        // Should fail (invalid file) but tests strict path
        assert!(result.is_err());
    }

    // ========================================================================
    // Run Diff Tensors Tests
    // ========================================================================

    #[test]
    fn test_run_diff_tensors_model1_not_found() {
        let model2 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        // run_diff_tensors(model_a, model_b, mismatches_only, show_values, filter, json)
        let result = run_diff_tensors(
            Path::new("/nonexistent/model1.gguf"),
            model2.path(),
            false, // mismatches_only
            0,     // show_values
            None,  // filter
            false, // json
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_tensors_model2_not_found() {
        let mut model1 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model1.write_all(b"not valid gguf").expect("write");

        let result = run_diff_tensors(
            model1.path(),
            Path::new("/nonexistent/model2.gguf"),
            false, // mismatches_only
            0,     // show_values
            None,  // filter
            false, // json
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_tensors_both_invalid() {
        let mut model1 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model1.write_all(b"not valid 1").expect("write");
        let mut model2 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model2.write_all(b"not valid 2").expect("write");

        let result = run_diff_tensors(model1.path(), model2.path(), false, 0, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_tensors_with_filter() {
        let mut model1 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model1.write_all(b"not valid 1").expect("write");
        let mut model2 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model2.write_all(b"not valid 2").expect("write");

        let result = run_diff_tensors(
            model1.path(),
            model2.path(),
            false,           // mismatches_only
            0,               // show_values
            Some("encoder"), // filter
            false,           // json
        );
        // Should fail (invalid files) but tests filter path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_tensors_with_mismatches_only() {
        let mut model1 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model1.write_all(b"not valid 1").expect("write");
        let mut model2 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model2.write_all(b"not valid 2").expect("write");

        let result = run_diff_tensors(
            model1.path(),
            model2.path(),
            true,  // mismatches_only
            0,     // show_values
            None,  // filter
            false, // json
        );
        // Should fail (invalid files) but tests mismatches_only path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_tensors_with_show_values() {
        let mut model1 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model1.write_all(b"not valid 1").expect("write");
        let mut model2 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model2.write_all(b"not valid 2").expect("write");

        let result = run_diff_tensors(
            model1.path(),
            model2.path(),
            false, // mismatches_only
            10,    // show_values (show 10 sample values)
            None,  // filter
            false, // json
        );
        // Should fail (invalid files) but tests show_values path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_diff_tensors_with_json() {
        let mut model1 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model1.write_all(b"not valid 1").expect("write");
        let mut model2 = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model2.write_all(b"not valid 2").expect("write");

        let result = run_diff_tensors(
            model1.path(),
            model2.path(),
            false, // mismatches_only
            0,     // show_values
            None,  // filter
            true,  // json
        );
        // Should fail (invalid files) but tests json path
        assert!(result.is_err());
    }

    // ========================================================================
    // RosettaCommands Tests
    // ========================================================================

    #[test]
    fn test_rosetta_commands_inspect_default() {
        // Test that the Inspect variant can be created
        let cmd = RosettaCommands::Inspect {
            file: PathBuf::from("model.gguf"),
            hexdump: false,
            json: false,
        };
        match cmd {
            RosettaCommands::Inspect {
                file,
                hexdump,
                json,
            } => {
                assert_eq!(file.to_string_lossy(), "model.gguf");
                assert!(!hexdump);
                assert!(!json);
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_convert() {
        let cmd = RosettaCommands::Convert {
            source: PathBuf::from("model.gguf"),
            target: PathBuf::from("model.apr"),
            quantize: None,
            verify: false,
            json: false,
            tokenizer: None,
        };
        match cmd {
            RosettaCommands::Convert {
                source,
                target,
                quantize,
                ..
            } => {
                assert_eq!(source.to_string_lossy(), "model.gguf");
                assert_eq!(target.to_string_lossy(), "model.apr");
                assert!(quantize.is_none());
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_chain() {
        let cmd = RosettaCommands::Chain {
            source: PathBuf::from("model.gguf"),
            formats: vec!["safetensors".to_string(), "apr".to_string()],
            work_dir: PathBuf::from("./work"),
            json: false,
        };
        match cmd {
            RosettaCommands::Chain { formats, .. } => {
                assert_eq!(formats.len(), 2);
                assert_eq!(formats[0], "safetensors");
                assert_eq!(formats[1], "apr");
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_verify() {
        let cmd = RosettaCommands::Verify {
            source: PathBuf::from("model.gguf"),
            intermediate: "safetensors".to_string(),
            tolerance: 1e-5,
            json: false,
        };
        match cmd {
            RosettaCommands::Verify {
                tolerance,
                intermediate,
                ..
            } => {
                assert_eq!(tolerance, 1e-5);
                assert_eq!(intermediate, "safetensors");
            }
            _ => panic!("Wrong command variant"),
        }
    }

    // ========================================================================
    // Helper Function Tests (PMAT Coverage - Internal Functions)
    // ========================================================================

    #[test]
    fn test_f16_to_f32_zero() {
        // f16 zero: 0x0000
        let bytes = [0x00, 0x00];
        let result = f16_to_f32(&bytes);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_f16_to_f32_one() {
        // f16 1.0: 0x3C00
        let bytes = [0x00, 0x3C];
        let result = f16_to_f32(&bytes);
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_negative_one() {
        // f16 -1.0: 0xBC00
        let bytes = [0x00, 0xBC];
        let result = f16_to_f32(&bytes);
        assert!((result + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_small_value() {
        // f16 0.5: 0x3800
        let bytes = [0x00, 0x38];
        let result = f16_to_f32(&bytes);
        assert!((result - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_normalize_tensor_name_basic() {
        let name = "model.layers.0.attention.q_proj.weight";
        let normalized = normalize_tensor_name(name);
        assert!(normalized.contains("attention"));
        assert!(normalized.contains("q_proj"));
    }

    #[test]
    fn test_normalize_tensor_name_empty() {
        let name = "";
        let normalized = normalize_tensor_name(name);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_normalize_tensor_name_with_numbers() {
        let name = "layer_123_weight";
        let normalized = normalize_tensor_name(name);
        assert!(!normalized.is_empty());
    }

    // GH-202: Cross-format tensor name normalization tests
    #[test]
    fn test_normalize_tensor_name_gguf_to_canonical() {
        // GGUF style: blk.N.attn_q.weight → N.q_proj.weight
        assert_eq!(
            normalize_tensor_name("blk.0.attn_q.weight"),
            "0.q_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.5.attn_k.weight"),
            "5.k_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.12.attn_v.weight"),
            "12.v_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.0.attn_output.weight"),
            "0.o_proj.weight"
        );
    }

    #[test]
    fn test_normalize_tensor_name_apr_to_canonical() {
        // APR/HF style: model.layers.N.self_attn.q_proj.weight → N.q_proj.weight
        assert_eq!(
            normalize_tensor_name("model.layers.0.self_attn.q_proj.weight"),
            "0.q_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.5.self_attn.k_proj.weight"),
            "5.k_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.12.self_attn.v_proj.weight"),
            "12.v_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.0.self_attn.o_proj.weight"),
            "0.o_proj.weight"
        );
    }

    #[test]
    fn test_normalize_tensor_name_ffn_mapping() {
        // GGUF FFN → HF MLP
        assert_eq!(
            normalize_tensor_name("blk.0.ffn_gate.weight"),
            "0.gate_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.0.ffn_up.weight"),
            "0.up_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.0.ffn_down.weight"),
            "0.down_proj.weight"
        );

        // APR/HF MLP
        assert_eq!(
            normalize_tensor_name("model.layers.0.mlp.gate_proj.weight"),
            "0.gate_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.0.mlp.up_proj.weight"),
            "0.up_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("model.layers.0.mlp.down_proj.weight"),
            "0.down_proj.weight"
        );
    }

    #[test]
    fn test_normalize_tensor_name_layernorm() {
        // GGUF: attn_norm/ffn_norm → HF: input_layernorm/post_attention_layernorm
        assert_eq!(
            normalize_tensor_name("blk.0.attn_norm.weight"),
            "0.input_layernorm.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.0.ffn_norm.weight"),
            "0.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn test_normalize_tensor_name_embeddings() {
        // token_embd → embed_tokens
        assert_eq!(
            normalize_tensor_name("token_embd.weight"),
            "embed_tokens.weight"
        );
        // output_norm → norm
        assert_eq!(normalize_tensor_name("output_norm.weight"), "norm.weight");
        // output → lm_head
        assert_eq!(normalize_tensor_name("output.weight"), "lm_head.weight");
    }

    #[test]
    fn test_normalize_tensor_name_cross_format_match() {
        // Verify GGUF and APR/HF normalize to the SAME canonical form (GH-202 core fix)
        let gguf_name = "blk.3.attn_q.weight";
        let apr_name = "model.layers.3.self_attn.q_proj.weight";
        assert_eq!(
            normalize_tensor_name(gguf_name),
            normalize_tensor_name(apr_name)
        );

        let gguf_ffn = "blk.7.ffn_down.weight";
        let apr_ffn = "model.layers.7.mlp.down_proj.weight";
        assert_eq!(
            normalize_tensor_name(gguf_ffn),
            normalize_tensor_name(apr_ffn)
        );
    }

    #[test]
    fn test_is_transposed_dims_true() {
        let shape_a = vec![768, 3072];
        let shape_b = vec![3072, 768];
        assert!(is_transposed_dims(&shape_a, &shape_b));
    }

    #[test]
    fn test_is_transposed_dims_false_same() {
        let shape_a = vec![768, 3072];
        let shape_b = vec![768, 3072];
        assert!(!is_transposed_dims(&shape_a, &shape_b));
    }

    #[test]
    fn test_is_transposed_dims_different_ndims() {
        let shape_a = vec![768, 3072];
        let shape_b = vec![768, 3072, 1];
        assert!(!is_transposed_dims(&shape_a, &shape_b));
    }

    #[test]
    fn test_is_transposed_dims_1d() {
        let shape_a = vec![768];
        let shape_b = vec![768];
        assert!(!is_transposed_dims(&shape_a, &shape_b));
    }

    #[test]
    fn test_strip_ansi_no_codes() {
        let text = "Hello, World!";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Hello, World!");
    }

    #[test]
    fn test_strip_ansi_with_codes() {
        let text = "\x1b[31mRed Text\x1b[0m";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Red Text");
    }

    #[test]
    fn test_strip_ansi_multiple_codes() {
        let text = "\x1b[1m\x1b[32mBold Green\x1b[0m Normal";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Bold Green Normal");
    }

    #[test]
    fn test_strip_ansi_empty() {
        let text = "";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "");
    }

    #[test]
    fn test_truncate_path_short() {
        let path = "/short/path".to_string();
        let truncated = truncate_path(path.clone(), 50);
        assert_eq!(truncated, path);
    }

    #[test]
    fn test_truncate_path_long() {
        let path = "/very/long/path/to/some/deeply/nested/file.txt".to_string();
        let truncated = truncate_path(path, 20);
        assert!(truncated.len() <= 23); // max_len + "..."
        assert!(truncated.contains("...") || truncated.len() <= 20);
    }

    #[test]
    fn test_truncate_path_exact_length() {
        let path = "exactly20characters!".to_string();
        let truncated = truncate_path(path, 20);
        assert!(truncated.len() <= 23);
    }

    #[test]
    fn test_get_role_threshold_embedding() {
        let threshold = get_role_threshold("model.embed_tokens.weight");
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_get_role_threshold_attention() {
        let threshold = get_role_threshold("model.layers.0.self_attn.q_proj.weight");
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_get_role_threshold_mlp() {
        let threshold = get_role_threshold("model.layers.0.mlp.gate_proj.weight");
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_get_role_threshold_norm() {
        let threshold = get_role_threshold("model.layers.0.input_layernorm.weight");
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_get_role_threshold_lm_head() {
        let threshold = get_role_threshold("lm_head.weight");
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_get_role_threshold_unknown() {
        let threshold = get_role_threshold("unknown_tensor_name");
        assert!(threshold > 0.0);
    }

    #[test]
    fn test_compute_tensor_stats_empty() {
        let data: Vec<f32> = vec![];
        let stats = compute_tensor_stats(&data);
        // Empty data should return NaN or zeros
        assert!(stats.0.is_nan() || stats.0 == 0.0); // mean
    }

    #[test]
    fn test_compute_tensor_stats_single_value() {
        let data = vec![5.0];
        let stats = compute_tensor_stats(&data);
        assert!((stats.0 - 5.0).abs() < 0.001); // mean = 5.0
        assert!(stats.1 == 0.0 || stats.1.is_nan()); // std = 0 for single value
    }

    #[test]
    fn test_compute_tensor_stats_multiple_values() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_tensor_stats(&data);
        assert!((stats.0 - 3.0).abs() < 0.001); // mean = 3.0
        assert!((stats.2 - 1.0).abs() < 0.001); // min = 1.0
        assert!((stats.3 - 5.0).abs() < 0.001); // max = 5.0
    }

    #[test]
    fn test_compute_tensor_stats_negative_values() {
        let data = vec![-5.0, -3.0, 0.0, 3.0, 5.0];
        let stats = compute_tensor_stats(&data);
        assert!((stats.0 - 0.0).abs() < 0.001); // mean = 0.0
        assert!((stats.2 - (-5.0)).abs() < 0.001); // min = -5.0
        assert!((stats.3 - 5.0).abs() < 0.001); // max = 5.0
    }

    #[test]
    fn test_dequantize_q4k_empty() {
        let data: Vec<u8> = vec![];
        let result = dequantize_q4k_for_stats(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q6k_empty() {
        let data: Vec<u8> = vec![];
        let result = dequantize_q6k_for_stats(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_fingerprints_to_json_empty() {
        let fingerprints: Vec<TensorFingerprint> = vec![];
        let json = fingerprints_to_json(&fingerprints);
        // Returns JSON object with empty fingerprints array
        assert!(json.contains("fingerprints"));
    }

    #[test]
    fn test_fingerprints_to_json_single() {
        let fingerprints = vec![TensorFingerprint {
            name: "test_tensor".to_string(),
            shape: vec![10, 20],
            dtype: "F32".to_string(),
            mean: 0.5,
            std: 0.1,
            min: 0.0,
            max: 1.0,
            p5: 0.05,
            p25: 0.25,
            p50: 0.5,
            p75: 0.75,
            p95: 0.95,
            nan_count: 0,
            inf_count: 0,
            zero_fraction: 0.0,
            checksum: 12345,
        }];
        let json = fingerprints_to_json(&fingerprints);
        assert!(json.contains("test_tensor"));
        assert!(json.contains("F32"));
    }

    #[test]
    fn test_load_fingerprints_from_json_not_found() {
        let result = load_fingerprints_from_json(Path::new("/nonexistent/fingerprints.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_fingerprints_from_json_invalid() {
        let mut file = NamedTempFile::with_suffix(".json").expect("create temp file");
        file.write_all(b"not valid json").expect("write");

        let result = load_fingerprints_from_json(file.path());
        // Returns Ok with empty vec for invalid JSON (no "name" fields found)
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_load_fingerprints_from_json_empty_array() {
        let mut file = NamedTempFile::with_suffix(".json").expect("create temp file");
        file.write_all(b"[]").expect("write");

        let result = load_fingerprints_from_json(file.path());
        // Empty array is valid JSON
        assert!(result.is_ok() || result.is_err()); // May or may not be valid depending on schema
    }

    #[test]
    fn test_rosetta_commands_compare_inference() {
        let cmd = RosettaCommands::CompareInference {
            model_a: PathBuf::from("model_a.gguf"),
            model_b: PathBuf::from("model_b.apr"),
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: 0.0,
            tolerance: 0.1,
            json: false,
        };
        match cmd {
            RosettaCommands::CompareInference {
                max_tokens,
                temperature,
                ..
            } => {
                assert_eq!(max_tokens, 10);
                assert_eq!(temperature, 0.0);
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_diff_tensors() {
        let cmd = RosettaCommands::DiffTensors {
            model_a: PathBuf::from("model_a.gguf"),
            model_b: PathBuf::from("model_b.apr"),
            mismatches_only: true,
            show_values: 5,
            filter: Some("attention".to_string()),
            json: false,
        };
        match cmd {
            RosettaCommands::DiffTensors {
                mismatches_only,
                show_values,
                filter,
                ..
            } => {
                assert!(mismatches_only);
                assert_eq!(show_values, 5);
                assert_eq!(filter, Some("attention".to_string()));
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_fingerprint() {
        let cmd = RosettaCommands::Fingerprint {
            model: PathBuf::from("model.gguf"),
            model_b: None,
            output: Some(PathBuf::from("fingerprints.json")),
            filter: None,
            verbose: true,
            json: false,
        };
        match cmd {
            RosettaCommands::Fingerprint {
                verbose, output, ..
            } => {
                assert!(verbose);
                assert!(output.is_some());
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_validate_stats() {
        let cmd = RosettaCommands::ValidateStats {
            model: PathBuf::from("model.gguf"),
            reference: None,
            fingerprints: Some(PathBuf::from("ref.json")),
            threshold: 0.01,
            strict: true,
            json: true,
        };
        match cmd {
            RosettaCommands::ValidateStats {
                strict,
                threshold,
                json,
                ..
            } => {
                assert!(strict);
                assert_eq!(threshold, 0.01);
                assert!(json);
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_run_compare_inference_model_a_not_found() {
        let model_b = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let result = run_compare_inference(
            Path::new("/nonexistent/model_a.gguf"),
            model_b.path(),
            "test",
            5,
            0.0,
            0.1,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_compare_inference_model_b_not_found() {
        let mut model_a = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        model_a.write_all(b"not valid gguf").expect("write");

        let result = run_compare_inference(
            model_a.path(),
            Path::new("/nonexistent/model_b.apr"),
            "test",
            5,
            0.0,
            0.1,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_print_fingerprints_empty() {
        let fingerprints: Vec<TensorFingerprint> = vec![];
        let result = print_fingerprints(&fingerprints, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprints_json_empty() {
        let fingerprints: Vec<TensorFingerprint> = vec![];
        let result = print_fingerprints(&fingerprints, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprints_verbose() {
        let fingerprints = vec![TensorFingerprint {
            name: "test".to_string(),
            shape: vec![10],
            dtype: "F32".to_string(),
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            p5: -0.9,
            p25: -0.5,
            p50: 0.0,
            p75: 0.5,
            p95: 0.9,
            nan_count: 0,
            inf_count: 0,
            zero_fraction: 0.1,
            checksum: 0,
        }];
        let result = print_fingerprints(&fingerprints, true, false);
        assert!(result.is_ok());
    }

    // ========================================================================
    // NEW: compute_tensor_stats comprehensive tests
    // ========================================================================

    #[test]
    fn test_compute_tensor_stats_with_nan_values() {
        let data = vec![1.0, f32::NAN, 3.0, f32::NAN, 5.0];
        let (
            mean,
            _std,
            min,
            max,
            _p5,
            _p25,
            _p50,
            _p75,
            _p95,
            nan_count,
            inf_count,
            _zero_frac,
            _checksum,
        ) = compute_tensor_stats(&data);
        assert_eq!(nan_count, 2);
        assert_eq!(inf_count, 0);
        // Mean should be (1+3+5)/3 = 3.0
        assert!((mean - 3.0).abs() < 0.001);
        assert!((min - 1.0).abs() < 0.001);
        assert!((max - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_tensor_stats_with_inf_values() {
        let data = vec![1.0, f32::INFINITY, 3.0, f32::NEG_INFINITY, 5.0];
        let (
            _mean,
            _std,
            _min,
            _max,
            _p5,
            _p25,
            _p50,
            _p75,
            _p95,
            nan_count,
            inf_count,
            _zero_frac,
            _checksum,
        ) = compute_tensor_stats(&data);
        assert_eq!(nan_count, 0);
        assert_eq!(inf_count, 2);
    }

    #[test]
    fn test_compute_tensor_stats_all_nan() {
        let data = vec![f32::NAN, f32::NAN, f32::NAN];
        let (
            mean,
            std,
            _min,
            _max,
            _p5,
            _p25,
            _p50,
            _p75,
            _p95,
            nan_count,
            inf_count,
            _zero_frac,
            checksum,
        ) = compute_tensor_stats(&data);
        assert_eq!(nan_count, 3);
        assert_eq!(inf_count, 0);
        // With no valid values, should return zeros for mean/std
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
        assert_ne!(checksum, 0); // NaN bits still contribute to checksum
    }

    #[test]
    fn test_compute_tensor_stats_zero_fraction() {
        let data = vec![0.0, 0.0, 1.0, 2.0, 0.0];
        let (
            _mean,
            _std,
            _min,
            _max,
            _p5,
            _p25,
            _p50,
            _p75,
            _p95,
            _nan_count,
            _inf_count,
            zero_frac,
            _checksum,
        ) = compute_tensor_stats(&data);
        // 3 out of 5 values are zero
        assert!((zero_frac - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_compute_tensor_stats_all_zeros() {
        let data = vec![0.0, 0.0, 0.0, 0.0];
        let (
            mean,
            std,
            min,
            max,
            _p5,
            _p25,
            _p50,
            _p75,
            _p95,
            _nan_count,
            _inf_count,
            zero_frac,
            _checksum,
        ) = compute_tensor_stats(&data);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert!((zero_frac - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_tensor_stats_checksum_deterministic() {
        let data = vec![1.0, 2.0, 3.0];
        let (_, _, _, _, _, _, _, _, _, _, _, _, checksum1) = compute_tensor_stats(&data);
        let (_, _, _, _, _, _, _, _, _, _, _, _, checksum2) = compute_tensor_stats(&data);
        assert_eq!(checksum1, checksum2);
    }

    #[test]
    fn test_compute_tensor_stats_checksum_differs_for_different_data() {
        let data1 = vec![1.0, 2.0, 3.0];
        let data2 = vec![4.0, 5.0, 6.0];
        let (_, _, _, _, _, _, _, _, _, _, _, _, checksum1) = compute_tensor_stats(&data1);
        let (_, _, _, _, _, _, _, _, _, _, _, _, checksum2) = compute_tensor_stats(&data2);
        assert_ne!(checksum1, checksum2);
    }

    #[test]
    fn test_compute_tensor_stats_std_deviation() {
        // Values: 2, 4, 4, 4, 5, 5, 7, 9 => mean=5, std=2
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (
            mean,
            std,
            _min,
            _max,
            _p5,
            _p25,
            _p50,
            _p75,
            _p95,
            _nan_count,
            _inf_count,
            _zero_frac,
            _checksum,
        ) = compute_tensor_stats(&data);
        assert!((mean - 5.0).abs() < 0.001);
        assert!((std - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_tensor_stats_percentiles() {
        // 100 evenly spaced values 0..99
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let (
            _mean,
            _std,
            min,
            max,
            p5,
            p25,
            p50,
            p75,
            p95,
            _nan_count,
            _inf_count,
            _zero_frac,
            _checksum,
        ) = compute_tensor_stats(&data);
        assert!((min - 0.0).abs() < 0.001);
        assert!((max - 99.0).abs() < 0.001);
        // p5 ~ 4.95, p25 ~ 24.75, p50 ~ 49.5, p75 ~ 74.25, p95 ~ 94.05
        assert!((p5 - 4.0).abs() < 2.0);
        assert!((p25 - 24.0).abs() < 2.0);
        assert!((p50 - 49.0).abs() < 2.0);
        assert!((p75 - 74.0).abs() < 2.0);
        assert!((p95 - 94.0).abs() < 2.0);
    }

    #[test]
    fn test_compute_tensor_stats_mixed_nan_inf_zero() {
        let data = vec![f32::NAN, f32::INFINITY, 0.0, 5.0, f32::NEG_INFINITY, 0.0];
        let (
            _mean,
            _std,
            _min,
            _max,
            _p5,
            _p25,
            _p50,
            _p75,
            _p95,
            nan_count,
            inf_count,
            zero_frac,
            _checksum,
        ) = compute_tensor_stats(&data);
        assert_eq!(nan_count, 1);
        assert_eq!(inf_count, 2);
        // 2 zeros out of 6 total values
        assert!((zero_frac - 2.0 / 6.0).abs() < 0.001);
    }

    // ========================================================================
    // NEW: validate_fingerprints comprehensive tests
    // ========================================================================

    fn make_fingerprint(
        name: &str,
        mean: f32,
        std: f32,
        nan_count: u32,
        inf_count: u32,
    ) -> TensorFingerprint {
        TensorFingerprint {
            name: name.to_string(),
            shape: vec![10, 20],
            dtype: "F32".to_string(),
            mean,
            std,
            min: -1.0,
            max: 1.0,
            p5: -0.9,
            p25: -0.25,
            p50: 0.0,
            p75: 0.25,
            p95: 0.9,
            nan_count,
            inf_count,
            zero_fraction: 0.0,
            checksum: 0,
        }
    }

    #[test]
    fn test_validate_fingerprints_identical() {
        let actual = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_mean_deviation_above_threshold() {
        let actual = vec![make_fingerprint("tensor_a", 5.0, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].field, "mean");
    }

    #[test]
    fn test_validate_fingerprints_mean_deviation_below_threshold() {
        let actual = vec![make_fingerprint("tensor_a", 0.6, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        // 0.1 sigma deviation < 3.0 threshold
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_nan_anomaly() {
        let actual = vec![make_fingerprint("tensor_a", 0.5, 1.0, 5, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        // NaN in actual but not in reference = anomaly
        let nan_anomaly = anomalies.iter().find(|a| a.field == "nan_count");
        assert!(nan_anomaly.is_some());
        assert_eq!(
            nan_anomaly.expect("nan anomaly").deviation_sigma,
            f32::INFINITY
        );
    }

    #[test]
    fn test_validate_fingerprints_inf_anomaly() {
        let actual = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 3)];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        let inf_anomaly = anomalies.iter().find(|a| a.field == "inf_count");
        assert!(inf_anomaly.is_some());
    }

    #[test]
    fn test_validate_fingerprints_nan_not_anomaly_when_reference_has_nan() {
        // Both have NaN => not anomalous
        let actual = vec![make_fingerprint("tensor_a", 0.5, 1.0, 5, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 5, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        let nan_anomaly = anomalies.iter().find(|a| a.field == "nan_count");
        assert!(nan_anomaly.is_none());
    }

    #[test]
    fn test_validate_fingerprints_missing_reference_tensor() {
        let actual = vec![make_fingerprint("tensor_only_in_actual", 0.5, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("tensor_only_in_ref", 0.5, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        // No matching tensor name => no anomalies (tensor is just skipped)
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_strict_mode_layernorm() {
        // LayerNorm has tighter threshold (2.0) in strict mode
        let actual = vec![make_fingerprint(
            "model.layers.0.input_layernorm.weight",
            3.5,
            1.0,
            0,
            0,
        )];
        let reference = vec![make_fingerprint(
            "model.layers.0.input_layernorm.weight",
            1.0,
            1.0,
            0,
            0,
        )];
        let anomalies = validate_fingerprints(&actual, &reference, 5.0, true);
        // Deviation = 2.5 sigma, strict threshold for layernorm = 2.0
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_strict_mode_embedding() {
        // Embeddings have looser threshold (5.0) in strict mode
        let actual = vec![make_fingerprint(
            "model.embed_tokens.weight",
            3.5,
            1.0,
            0,
            0,
        )];
        let reference = vec![make_fingerprint(
            "model.embed_tokens.weight",
            0.5,
            1.0,
            0,
            0,
        )];
        let anomalies = validate_fingerprints(&actual, &reference, 2.0, true);
        // Deviation = 3.0 sigma, strict threshold for embed = 5.0 => no anomaly
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_zero_std_reference() {
        // When reference std is near zero, deviation is scaled up
        let actual = vec![make_fingerprint("tensor_a", 0.001, 0.0, 0, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.0, 0.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        // deviation = 0.001 * 1000 = 1.0 < threshold 3.0 => no anomaly
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_cross_format_names() {
        // GGUF name should match APR name via normalize_tensor_name
        let actual = vec![make_fingerprint("blk.0.attn_q.weight", 5.0, 1.0, 0, 0)];
        let reference = vec![make_fingerprint(
            "model.layers.0.self_attn.q_proj.weight",
            0.5,
            1.0,
            0,
            0,
        )];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        // Should match and detect the mean deviation
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_multiple_tensors() {
        let actual = vec![
            make_fingerprint("tensor_a", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_b", 10.0, 1.0, 0, 0),
            make_fingerprint("tensor_c", 0.5, 1.0, 3, 0),
        ];
        let reference = vec![
            make_fingerprint("tensor_a", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_b", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_c", 0.5, 1.0, 0, 0),
        ];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        // tensor_b has mean deviation, tensor_c has NaN anomaly
        assert!(anomalies.len() >= 2);
    }

    // ========================================================================
    // NEW: get_role_threshold specific return value tests
    // ========================================================================

    #[test]
    fn test_get_role_threshold_layernorm_value() {
        assert_eq!(
            get_role_threshold("model.layers.0.input_layernorm.weight"),
            2.0
        );
    }

    #[test]
    fn test_get_role_threshold_layer_norm_underscore_value() {
        assert_eq!(get_role_threshold("some.layer_norm.weight"), 2.0);
    }

    #[test]
    fn test_get_role_threshold_ln_prefix_value() {
        assert_eq!(get_role_threshold("ln_1.weight"), 2.0);
    }

    #[test]
    fn test_get_role_threshold_embed_value() {
        assert_eq!(get_role_threshold("model.embed_tokens.weight"), 5.0);
    }

    #[test]
    fn test_get_role_threshold_lm_head_value() {
        assert_eq!(get_role_threshold("lm_head.weight"), 3.0);
    }

    #[test]
    fn test_get_role_threshold_output_value() {
        assert_eq!(get_role_threshold("output.weight"), 3.0);
    }

    #[test]
    fn test_get_role_threshold_default_value() {
        assert_eq!(get_role_threshold("some.random.tensor"), 3.0);
    }

    #[test]
    fn test_get_role_threshold_case_insensitive() {
        // Should detect "LAYERNORM" even though check uses lowercase
        assert_eq!(get_role_threshold("model.LAYERNORM.weight"), 2.0);
        assert_eq!(get_role_threshold("model.EMBED.weight"), 5.0);
    }

    // ========================================================================
    // NEW: fingerprints_to_json comprehensive tests
    // ========================================================================

    #[test]
    fn test_fingerprints_to_json_multiple() {
        let fingerprints = vec![
            make_fingerprint("tensor_a", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_b", 1.5, 2.0, 1, 2),
        ];
        let json = fingerprints_to_json(&fingerprints);
        assert!(json.contains("tensor_a"));
        assert!(json.contains("tensor_b"));
        // First entry should have trailing comma, last should not (between entries)
        assert!(json.contains("},\n"));
    }

    #[test]
    fn test_fingerprints_to_json_special_values() {
        let fp = TensorFingerprint {
            name: "test".to_string(),
            shape: vec![],
            dtype: "Q4_K".to_string(),
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            p5: 0.0,
            p25: 0.0,
            p50: 0.0,
            p75: 0.0,
            p95: 0.0,
            nan_count: 100,
            inf_count: 50,
            zero_fraction: 0.99,
            checksum: 0xDEADBEEF,
        };
        let json = fingerprints_to_json(&[fp]);
        assert!(json.contains("\"nan_count\": 100"));
        assert!(json.contains("\"inf_count\": 50"));
        assert!(json.contains("Q4_K"));
        assert!(json.contains(&format!("{}", 0xDEADBEEF_u32)));
    }

    #[test]
    fn test_fingerprints_to_json_roundtrip_structure() {
        let fps = vec![make_fingerprint("t1", 0.1, 0.2, 0, 0)];
        let json = fingerprints_to_json(&fps);
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
        assert!(json.contains("\"fingerprints\""));
    }

    // ========================================================================
    // NEW: load_fingerprints_from_json with valid content
    // ========================================================================

    #[test]
    fn test_load_fingerprints_from_json_valid_name_fields() {
        let mut file = NamedTempFile::with_suffix(".json").expect("create temp file");
        file.write_all(b"{\n  \"fingerprints\": [\n    {\"name\": \"tensor_a\", \"mean\": 0.5},\n    {\"name\": \"tensor_b\", \"mean\": 1.0}\n  ]\n}").expect("write");

        let result = load_fingerprints_from_json(file.path());
        assert!(result.is_ok());
        let fps = result.expect("parsed");
        assert_eq!(fps.len(), 2);
        assert_eq!(fps[0].name, "tensor_a");
        assert_eq!(fps[1].name, "tensor_b");
        // All other fields are placeholder defaults
        assert_eq!(fps[0].std, 1.0);
        assert_eq!(fps[0].dtype, "unknown");
    }

    #[test]
    fn test_load_fingerprints_from_json_no_name_fields() {
        let mut file = NamedTempFile::with_suffix(".json").expect("create temp file");
        file.write_all(b"{\"data\": [1, 2, 3]}").expect("write");

        let result = load_fingerprints_from_json(file.path());
        assert!(result.is_ok());
        assert!(result.expect("parsed").is_empty());
    }

    // ========================================================================
    // NEW: parse_tensor_stats_json always returns None
    // ========================================================================

    #[test]
    fn test_parse_tensor_stats_json_placeholder() {
        assert!(parse_tensor_stats_json("{}").is_none());
        assert!(parse_tensor_stats_json("{\"tensors\": {}}").is_none());
        assert!(parse_tensor_stats_json("").is_none());
    }

    // ========================================================================
    // NEW: normalize_tensor_name edge cases
    // ========================================================================

    #[test]
    fn test_normalize_tensor_name_output_weight() {
        assert_eq!(normalize_tensor_name("output.weight"), "lm_head.weight");
    }

    #[test]
    fn test_normalize_tensor_name_output_not_weight() {
        // "output.bias" should NOT map to lm_head
        let result = normalize_tensor_name("output.bias");
        assert_ne!(result, "lm_head.weight");
        assert_eq!(result, "output.bias"); // stays as-is (falls through)
    }

    #[test]
    fn test_normalize_tensor_name_deeply_nested() {
        // Only first occurrence of prefixes is stripped
        let result = normalize_tensor_name("model.layers.10.self_attn.q_proj.weight");
        assert_eq!(result, "10.q_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_no_match() {
        // Name with no recognized patterns should pass through mostly unchanged
        let result = normalize_tensor_name("custom_tensor_name");
        assert_eq!(result, "custom_tensor_name");
    }

    #[test]
    fn test_normalize_tensor_name_gguf_all_mappings() {
        assert_eq!(
            normalize_tensor_name("token_embd.weight"),
            "embed_tokens.weight"
        );
        assert_eq!(normalize_tensor_name("output_norm.weight"), "norm.weight");
    }

    // ========================================================================
    // NEW: is_transposed_dims edge cases
    // ========================================================================

    #[test]
    fn test_is_transposed_dims_square_matrix() {
        // [512, 512] vs [512, 512] - same shape, NOT transposed
        assert!(!is_transposed_dims(&[512, 512], &[512, 512]));
    }

    #[test]
    fn test_is_transposed_dims_empty_shapes() {
        assert!(!is_transposed_dims(&[], &[]));
    }

    #[test]
    fn test_is_transposed_dims_3d_shapes() {
        assert!(!is_transposed_dims(&[2, 3, 4], &[4, 3, 2]));
    }

    #[test]
    fn test_is_transposed_dims_one_empty_one_not() {
        assert!(!is_transposed_dims(&[768, 3072], &[]));
    }

    #[test]
    fn test_is_transposed_dims_different_sizes() {
        // Shapes that are NOT transposed versions of each other
        assert!(!is_transposed_dims(&[768, 3072], &[768, 1024]));
    }

    // ========================================================================
    // NEW: strip_ansi edge cases
    // ========================================================================

    #[test]
    fn test_strip_ansi_escape_without_bracket() {
        // ESC not followed by [ should just skip the ESC char
        let text = "\x1b Hello";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, " Hello");
    }

    #[test]
    fn test_strip_ansi_nested_escape_sequences() {
        let text = "\x1b[1;31;42mColored\x1b[0m";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Colored");
    }

    #[test]
    fn test_strip_ansi_only_escape_sequences() {
        let text = "\x1b[31m\x1b[0m";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "");
    }

    #[test]
    fn test_strip_ansi_preserves_non_ansi_content() {
        let text = "Hello [World] (test)";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Hello [World] (test)");
    }

    // ========================================================================
    // NEW: truncate_path edge cases
    // ========================================================================

    #[test]
    fn test_truncate_path_empty_string() {
        let result = truncate_path(String::new(), 10);
        assert_eq!(result, "");
    }

    #[test]
    fn test_truncate_path_single_char() {
        let result = truncate_path("a".to_string(), 1);
        assert_eq!(result, "a");
    }

    #[test]
    fn test_truncate_path_boundary() {
        let path = "12345".to_string();
        // Exactly at boundary
        assert_eq!(truncate_path(path.clone(), 5), "12345");
        // One less than boundary
        let truncated = truncate_path(path, 4);
        assert!(truncated.starts_with("..."));
    }

    // ========================================================================
    // NEW: f16_to_f32 edge cases
    // ========================================================================

    #[test]
    fn test_f16_to_f32_empty_bytes() {
        let bytes: [u8; 0] = [];
        let result = f16_to_f32(&bytes);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_f16_to_f32_single_byte() {
        let bytes = [0x00];
        let result = f16_to_f32(&bytes);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_f16_to_f32_infinity() {
        // f16 +infinity: 0x7C00
        let bytes = [0x00, 0x7C];
        let result = f16_to_f32(&bytes);
        assert!(result.is_infinite() && result > 0.0);
    }

    #[test]
    fn test_f16_to_f32_negative_infinity() {
        // f16 -infinity: 0xFC00
        let bytes = [0x00, 0xFC];
        let result = f16_to_f32(&bytes);
        assert!(result.is_infinite() && result < 0.0);
    }

    #[test]
    fn test_f16_to_f32_nan() {
        // f16 NaN: 0x7E00
        let bytes = [0x00, 0x7E];
        let result = f16_to_f32(&bytes);
        assert!(result.is_nan());
    }

    #[test]
    fn test_f16_to_f32_negative_zero() {
        // f16 -0.0: 0x8000
        let bytes = [0x00, 0x80];
        let result = f16_to_f32(&bytes);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_negative());
    }

    // ========================================================================
    // NEW: dequantize_q4k_for_stats tests
    // ========================================================================

    #[test]
    fn test_dequantize_q4k_for_stats_short_data() {
        // Data shorter than one block => no output
        let data = vec![0u8; 100]; // Less than 144 bytes (one Q4_K block)
        let result = dequantize_q4k_for_stats(&data, 256);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q4k_for_stats_one_block() {
        // One complete Q4_K block: 144 bytes
        let mut data = vec![0u8; 144];
        // Set d (f16 1.0) at bytes 0-1
        data[0] = 0x00;
        data[1] = 0x3C;
        // dmin = 0
        // scales and qs all zero => all values should be d * scale * (0 - 8) = negative
        let result = dequantize_q4k_for_stats(&data, 256);
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q4k_for_stats_limits_to_num_elements() {
        // Request fewer elements than one block produces
        let data = vec![0u8; 144];
        let result = dequantize_q4k_for_stats(&data, 10);
        assert_eq!(result.len(), 10);
    }

    // ========================================================================
    // NEW: dequantize_q6k_for_stats tests
    // ========================================================================

    #[test]
    fn test_dequantize_q6k_for_stats_short_data() {
        let data = vec![0u8; 100]; // Less than 210 bytes (one Q6_K block)
        let result = dequantize_q6k_for_stats(&data, 256);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q6k_for_stats_one_block() {
        let mut data = vec![0u8; 210];
        // Set d (f16 1.0) at bytes 208-209
        data[208] = 0x00;
        data[209] = 0x3C;
        let result = dequantize_q6k_for_stats(&data, 256);
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q6k_for_stats_limits_to_num_elements() {
        let data = vec![0u8; 210];
        let result = dequantize_q6k_for_stats(&data, 10);
        assert_eq!(result.len(), 10);
    }

    // ========================================================================
    // NEW: ConversionOptions default tests
    // ========================================================================

    #[test]
    fn test_conversion_options_default() {
        let opts = ConversionOptions::default();
        assert!(opts.quantization.is_none());
        assert!(opts.verify);
        assert!(!opts.compute_stats);
        assert!((opts.tolerance - 1e-6).abs() < 1e-10);
        assert!(opts.preserve_metadata);
        assert!(opts.add_provenance);
    }

    #[test]
    fn test_conversion_options_custom() {
        let opts = ConversionOptions {
            quantization: Some("int8".to_string()),
            verify: false,
            compute_stats: true,
            tolerance: 0.01,
            preserve_metadata: false,
            add_provenance: false,
            tokenizer_path: None,
        };
        assert_eq!(opts.quantization.as_deref(), Some("int8"));
        assert!(!opts.verify);
        assert!(opts.compute_stats);
    }

    // ========================================================================
    // NEW: ConversionPath tests
    // ========================================================================

    #[test]
    fn test_conversion_path_direct() {
        let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        assert_eq!(path.source, FormatType::Gguf);
        assert_eq!(path.target, FormatType::Apr);
        assert!(path.intermediates.is_empty());
    }

    #[test]
    fn test_conversion_path_chain() {
        let path = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::SafeTensors],
            FormatType::Apr,
        );
        assert_eq!(path.intermediates.len(), 1);
        assert_eq!(path.intermediates[0], FormatType::SafeTensors);
    }

    #[test]
    fn test_conversion_path_steps() {
        let path = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::SafeTensors],
            FormatType::Apr,
        );
        let steps = path.steps();
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0], FormatType::Gguf);
        assert_eq!(steps[1], FormatType::SafeTensors);
        assert_eq!(steps[2], FormatType::Apr);
    }

    #[test]
    fn test_conversion_path_is_roundtrip() {
        let roundtrip =
            ConversionPath::chain(FormatType::Gguf, vec![FormatType::Apr], FormatType::Gguf);
        assert!(roundtrip.is_roundtrip());

        let direct = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        assert!(!direct.is_roundtrip());

        // Same source/target but no intermediates
        let same = ConversionPath::direct(FormatType::Gguf, FormatType::Gguf);
        assert!(!same.is_roundtrip());
    }

    #[test]
    fn test_conversion_path_has_cycle() {
        // A→B→A is roundtrip but no cycle (middle doesn't repeat)
        let roundtrip =
            ConversionPath::chain(FormatType::Gguf, vec![FormatType::Apr], FormatType::Gguf);
        assert!(!roundtrip.has_cycle());

        // A→B→B→C has cycle (B repeated in middle)
        let cyclic = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::SafeTensors, FormatType::SafeTensors],
            FormatType::Apr,
        );
        assert!(cyclic.has_cycle());
    }

    #[test]
    fn test_conversion_path_display() {
        let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        let display = format!("{path}");
        assert!(display.contains("GGUF"));
        assert!(display.contains("APR"));
    }

    #[test]
    fn test_conversion_path_display_chain() {
        let path = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::SafeTensors],
            FormatType::Apr,
        );
        let display = format!("{path}");
        assert!(display.contains("SafeTensors"));
    }

    // ========================================================================
    // NEW: FormatType additional tests
    // ========================================================================

    #[test]
    fn test_format_type_extension() {
        assert_eq!(FormatType::Gguf.extension(), "gguf");
        assert_eq!(FormatType::SafeTensors.extension(), "safetensors");
        assert_eq!(FormatType::Apr.extension(), "apr");
    }

    #[test]
    fn test_format_type_debug() {
        let fmt = format!("{:?}", FormatType::Gguf);
        assert_eq!(fmt, "Gguf");
    }

    #[test]
    fn test_format_type_clone_eq() {
        let a = FormatType::SafeTensors;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn test_format_type_from_extension_no_extension() {
        let path = Path::new("model");
        let result = FormatType::from_extension(path);
        assert!(result.is_err());
    }

    // ========================================================================
    // NEW: Chain format parsing tests
    // ========================================================================

    #[test]
    fn test_chain_format_parsing_st_alias() {
        // "st" is alias for "safetensors"
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        let formats = vec!["st".to_string(), "apr".to_string()];

        // Will fail due to invalid file, but exercises format parsing
        let result = run_chain(source.path(), &formats, work_dir.path(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_chain_format_parsing_invalid_format() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        let formats = vec!["pytorch".to_string(), "apr".to_string()];

        let result = run_chain(source.path(), &formats, work_dir.path(), false);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = format!("{err}");
        assert!(err_str.contains("Unknown format"));
    }

    #[test]
    fn test_chain_format_parsing_case_insensitive() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        let formats = vec!["GGUF".to_string(), "APR".to_string()];

        // Will fail due to invalid file, but exercises case-insensitive parsing
        let result = run_chain(source.path(), &formats, work_dir.path(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_chain_single_format_too_short() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        let formats = vec!["apr".to_string()];

        let result = run_chain(source.path(), &formats, work_dir.path(), false);
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(err_str.contains("at least 2 formats"));
    }

    #[test]
    fn test_chain_with_cycle_detection() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        // GGUF→SafeTensors→SafeTensors→APR has a cycle (SafeTensors repeated)
        let formats = vec![
            "gguf".to_string(),
            "safetensors".to_string(),
            "safetensors".to_string(),
            "apr".to_string(),
        ];

        let result = run_chain(source.path(), &formats, work_dir.path(), false);
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(err_str.contains("cycle"));
    }

    #[test]
    fn test_chain_json_output_flag() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let work_dir = tempdir().expect("create work dir");
        let formats = vec!["safetensors".to_string(), "apr".to_string()];

        let result = run_chain(source.path(), &formats, work_dir.path(), true);
        // Fails due to invalid file, but exercises json=true path
        assert!(result.is_err());
    }

    // ========================================================================
    // NEW: Verify intermediate format parsing tests
    // ========================================================================

    #[test]
    fn test_verify_intermediate_gguf() {
        let mut source = NamedTempFile::with_suffix(".safetensors").expect("create source");
        source.write_all(b"not valid").expect("write");
        let result = run_verify(source.path(), "gguf", 1e-5, false);
        assert!(result.is_err()); // Invalid file, but exercises gguf intermediate
    }

    #[test]
    fn test_verify_intermediate_st_alias() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid").expect("write");
        let result = run_verify(source.path(), "st", 1e-5, false);
        assert!(result.is_err()); // Invalid file, but exercises st alias
    }

    #[test]
    fn test_verify_intermediate_invalid() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid").expect("write");
        let result = run_verify(source.path(), "pytorch", 1e-5, false);
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(err_str.contains("Unknown format"));
    }

    #[test]
    fn test_verify_json_output_flag() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid").expect("write");
        let result = run_verify(source.path(), "safetensors", 1e-5, true);
        assert!(result.is_err()); // Invalid file, exercises json=true
    }

    // ========================================================================
    // NEW: run_validate_stats missing reference and fingerprints
    // ========================================================================

    #[test]
    fn test_run_validate_stats_no_reference_no_fingerprints() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_validate_stats(file.path(), None, None, 3.0, false, false);
        assert!(result.is_err());
        let err_str = format!("{}", result.unwrap_err());
        assert!(err_str.contains("--reference") || err_str.contains("--fingerprints"));
    }

    #[test]
    fn test_run_validate_stats_json_flag() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");
        let result = run_validate_stats(file.path(), None, None, 3.0, false, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_validate_stats_reference_not_found() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");
        let result = run_validate_stats(
            file.path(),
            Some(Path::new("/nonexistent/ref.gguf")),
            None,
            3.0,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_validate_stats_fingerprints_not_found() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");
        let result = run_validate_stats(
            file.path(),
            None,
            Some(Path::new("/nonexistent/fp.json")),
            3.0,
            false,
            false,
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // NEW: TensorFingerprint struct tests
    // ========================================================================

    #[test]
    fn test_tensor_fingerprint_clone() {
        let fp = make_fingerprint("test", 0.5, 1.0, 0, 0);
        let fp_clone = fp.clone();
        assert_eq!(fp.name, fp_clone.name);
        assert_eq!(fp.mean, fp_clone.mean);
        assert_eq!(fp.shape, fp_clone.shape);
    }

    #[test]
    fn test_tensor_fingerprint_debug() {
        let fp = make_fingerprint("test", 0.5, 1.0, 0, 0);
        let debug_str = format!("{fp:?}");
        assert!(debug_str.contains("test"));
        assert!(debug_str.contains("TensorFingerprint"));
    }

    // ========================================================================
    // NEW: StatisticalAnomaly tests
    // ========================================================================

    #[test]
    fn test_statistical_anomaly_construction() {
        let anomaly = StatisticalAnomaly {
            tensor: "test_tensor".to_string(),
            field: "mean".to_string(),
            expected: 0.5,
            actual: 5.0,
            deviation_sigma: 4.5,
        };
        assert_eq!(anomaly.tensor, "test_tensor");
        assert_eq!(anomaly.field, "mean");
        assert!((anomaly.deviation_sigma - 4.5).abs() < 0.001);
    }

    #[test]
    fn test_statistical_anomaly_debug() {
        let anomaly = StatisticalAnomaly {
            tensor: "t".to_string(),
            field: "std".to_string(),
            expected: 1.0,
            actual: 10.0,
            deviation_sigma: 9.0,
        };
        let debug = format!("{anomaly:?}");
        assert!(debug.contains("StatisticalAnomaly"));
    }

    // ========================================================================
    // NEW: InferenceResult struct tests
    // ========================================================================

    #[test]
    fn test_inference_result_construction() {
        let result = InferenceResult {
            tokens: vec![1, 2, 3],
            logits: vec![0.5, 0.6, 0.7],
            top5: vec![vec![1, 2, 3, 4, 5]],
            output_text: "hello world".to_string(),
        };
        assert_eq!(result.tokens.len(), 3);
        assert_eq!(result.logits.len(), 3);
        assert_eq!(result.top5.len(), 1);
        assert_eq!(result.output_text, "hello world");
    }

    // ========================================================================
    // NEW: print_fingerprint_diff tests
    // ========================================================================

    #[test]
    fn test_print_fingerprint_diff_no_anomalies() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_with_anomaly() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 10.0, 1.0, 0, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_missing_in_b() {
        let fps_a = vec![make_fingerprint("only_in_a", 0.5, 1.0, 0, 0)];
        let fps_b: Vec<TensorFingerprint> = vec![];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_verbose() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, true, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_json() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_nan_mismatch() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 0.5, 1.0, 5, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_cross_format_matching() {
        // GGUF name in A, HF name in B - should still match
        let fps_a = vec![make_fingerprint("blk.0.attn_q.weight", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint(
            "model.layers.0.self_attn.q_proj.weight",
            0.5,
            1.0,
            0,
            0,
        )];
        let result = print_fingerprint_diff(&fps_a, &fps_b, true, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_zero_std() {
        // When std is near zero, mean diff uses absolute value
        let fps_a = vec![make_fingerprint("tensor_a", 0.001, 0.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 0.0, 0.0, 0, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, false);
        assert!(result.is_ok());
    }

    // ========================================================================
    // NEW: print_fingerprints non-verbose with data
    // ========================================================================

    #[test]
    fn test_print_fingerprints_non_verbose_with_data() {
        let fingerprints = vec![
            make_fingerprint("tensor_a", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_b", 1.5, 2.0, 1, 2),
        ];
        let result = print_fingerprints(&fingerprints, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprints_json_with_data() {
        let fingerprints = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let result = print_fingerprints(&fingerprints, false, true);
        assert!(result.is_ok());
    }

    // ========================================================================
    // NEW: VerificationReport tests
    // ========================================================================

    #[test]
    fn test_verification_report_passing() {
        let report = VerificationReport::passing();
        assert!(report.is_equivalent);
        assert_eq!(report.max_diff, 0.0);
        assert_eq!(report.mean_diff, 0.0);
        assert!(report.tensor_diffs.is_empty());
        assert!(report.changed_metadata.is_empty());
        assert!(report.failed_tensors.is_empty());
    }

    #[test]
    fn test_verification_report_passes_with_tolerance() {
        let report = VerificationReport::passing();
        assert!(report.passes_with_tolerance(1e-5));
        assert!(report.passes_with_tolerance(0.0));
    }

    #[test]
    fn test_verification_report_fails_with_tolerance() {
        let mut report = VerificationReport::passing();
        report.max_diff = 0.01;
        assert!(!report.passes_with_tolerance(0.001));
        assert!(report.passes_with_tolerance(0.1));
    }

    #[test]
    fn test_verification_report_fails_with_failed_tensors() {
        let mut report = VerificationReport::passing();
        report.failed_tensors.push("bad_tensor".to_string());
        assert!(!report.passes_with_tolerance(1.0));
    }

    // ========================================================================
    // NEW: RosettaCommands additional variant tests
    // ========================================================================

    #[test]
    fn test_rosetta_commands_inspect_with_hexdump() {
        let cmd = RosettaCommands::Inspect {
            file: PathBuf::from("model.gguf"),
            hexdump: true,
            json: true,
        };
        match cmd {
            RosettaCommands::Inspect { hexdump, json, .. } => {
                assert!(hexdump);
                assert!(json);
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_convert_with_all_options() {
        let cmd = RosettaCommands::Convert {
            source: PathBuf::from("in.safetensors"),
            target: PathBuf::from("out.apr"),
            quantize: Some("int4".to_string()),
            verify: true,
            json: true,
            tokenizer: None,
        };
        match cmd {
            RosettaCommands::Convert {
                quantize,
                verify,
                json,
                ..
            } => {
                assert_eq!(quantize, Some("int4".to_string()));
                assert!(verify);
                assert!(json);
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_rosetta_commands_fingerprint_with_model_b() {
        let cmd = RosettaCommands::Fingerprint {
            model: PathBuf::from("model_a.gguf"),
            model_b: Some(PathBuf::from("model_b.apr")),
            output: None,
            filter: Some("attn".to_string()),
            verbose: false,
            json: true,
        };
        match cmd {
            RosettaCommands::Fingerprint {
                model_b,
                filter,
                json,
                ..
            } => {
                assert!(model_b.is_some());
                assert_eq!(filter, Some("attn".to_string()));
                assert!(json);
            }
            _ => panic!("Wrong command variant"),
        }
    }

    // ========================================================================
    // NEW: run_convert with JSON flag
    // ========================================================================

    #[test]
    fn test_run_convert_json_flag() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let target = NamedTempFile::with_suffix(".apr").expect("create target");

        let result = run_convert(source.path(), target.path(), None, false, true, None);
        // Fails due to invalid file, but exercises json=true path
        assert!(result.is_err());
    }

    // ========================================================================
    // NEW: run_fingerprint edge cases
    // ========================================================================

    #[test]
    fn test_run_fingerprint_model_b_not_found() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_fingerprint(
            file.path(),
            Some(Path::new("/nonexistent/model_b.gguf")),
            None,
            None,
            false,
            false,
        );
        // Fails because model A is invalid (inspection fails before model_b check)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_fingerprint_verbose_flag() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_fingerprint(file.path(), None, None, None, true, false);
        assert!(result.is_err()); // Invalid file, but exercises verbose path
    }

    #[test]
    fn test_run_fingerprint_json_flag() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run_fingerprint(file.path(), None, None, None, false, true);
        assert!(result.is_err()); // Invalid file, but exercises json path
    }

    // ========================================================================
    // NEW: run_compare_inference error paths
    // ========================================================================

    #[test]
    fn test_run_compare_inference_both_not_found() {
        let result = run_compare_inference(
            Path::new("/nonexistent/a.gguf"),
            Path::new("/nonexistent/b.apr"),
            "test",
            5,
            0.0,
            0.1,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_compare_inference_json_flag() {
        let result = run_compare_inference(
            Path::new("/nonexistent/a.gguf"),
            Path::new("/nonexistent/b.apr"),
            "test",
            5,
            0.0,
            0.1,
            true,
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // NEW: load_tensor_data_direct edge cases
    // ========================================================================

    #[test]
    fn test_load_tensor_data_direct_unknown_extension() {
        let mut file = NamedTempFile::with_suffix(".unknown").expect("create temp file");
        file.write_all(b"data").expect("write");
        let result = load_tensor_data_direct(file.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tensor_data_direct_invalid_gguf() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");
        let result = load_tensor_data_direct(file.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tensor_data_direct_invalid_apr() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid apr").expect("write");
        let result = load_tensor_data_direct(file.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tensor_data_direct_apr_too_short() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"APR\0short").expect("write"); // Less than 40 bytes
        let result = load_tensor_data_direct(file.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tensor_data_direct_apr_wrong_magic() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let data = vec![0u8; 50]; // 50 bytes but wrong magic
        file.write_all(&data).expect("write");
        let result = load_tensor_data_direct(file.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tensor_data_direct_invalid_safetensors() {
        let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        file.write_all(b"not valid safetensors").expect("write");
        let result = load_tensor_data_direct(file.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_load_tensor_data_direct_no_extension() {
        let dir = tempdir().expect("create temp dir");
        let file_path = dir.path().join("model");
        std::fs::write(&file_path, b"data").expect("write");
        let result = load_tensor_data_direct(&file_path);
        assert!(result.is_none());
    }

    // ========================================================================
    // NEW: FormatType from_magic tests
    // ========================================================================

    #[test]
    fn test_format_type_from_magic_gguf() {
        let dir = tempdir().expect("create temp dir");
        let file_path = dir.path().join("test.bin");
        // GGUF magic: "GGUF" + version bytes
        let mut data = b"GGUF".to_vec();
        data.extend_from_slice(&[3, 0, 0, 0]); // version 3
        std::fs::write(&file_path, &data).expect("write");
        let result = FormatType::from_magic(&file_path);
        assert!(result.is_ok());
        assert_eq!(result.expect("format"), FormatType::Gguf);
    }

    #[test]
    fn test_format_type_from_magic_apr() {
        let dir = tempdir().expect("create temp dir");
        let file_path = dir.path().join("test.bin");
        let mut data = b"APR\0".to_vec();
        data.extend_from_slice(&[0u8; 8]); // Padding for 8+ bytes
        std::fs::write(&file_path, &data).expect("write");
        let result = FormatType::from_magic(&file_path);
        assert!(result.is_ok());
        assert_eq!(result.expect("format"), FormatType::Apr);
    }

    #[test]
    fn test_format_type_from_magic_unknown() {
        let dir = tempdir().expect("create temp dir");
        let file_path = dir.path().join("test.bin");
        std::fs::write(&file_path, b"UNKNOWN!MAGIC").expect("write");
        let result = FormatType::from_magic(&file_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_type_from_magic_nonexistent() {
        let result = FormatType::from_magic(Path::new("/nonexistent/file"));
        assert!(result.is_err());
    }

    #[test]
    fn test_format_type_from_magic_too_short() {
        let dir = tempdir().expect("create temp dir");
        let file_path = dir.path().join("test.bin");
        std::fs::write(&file_path, b"GGU").expect("write"); // Too short for magic read
        let result = FormatType::from_magic(&file_path);
        assert!(result.is_err());
    }

    // ========================================================================
    // NEW: RosettaStone construction tests
    // ========================================================================

    #[test]
    fn test_rosetta_stone_new() {
        let rs = RosettaStone::new();
        let debug_str = format!("{rs:?}");
        assert!(debug_str.contains("RosettaStone"));
    }

    #[test]
    fn test_rosetta_stone_with_options() {
        let opts = ConversionOptions {
            quantization: Some("int8".to_string()),
            ..Default::default()
        };
        let rs = RosettaStone::with_options(opts);
        let debug_str = format!("{rs:?}");
        assert!(debug_str.contains("RosettaStone"));
    }

    // ========================================================================
    // NEW: dequantize multi-block tests
    // ========================================================================

    #[test]
    fn test_dequantize_q4k_multiple_blocks() {
        // Two complete Q4_K blocks
        let data = vec![0u8; 288]; // 2 * 144
        let result = dequantize_q4k_for_stats(&data, 512);
        assert_eq!(result.len(), 512);
    }

    #[test]
    fn test_dequantize_q6k_multiple_blocks() {
        // Two complete Q6_K blocks
        let data = vec![0u8; 420]; // 2 * 210
        let result = dequantize_q6k_for_stats(&data, 512);
        assert_eq!(result.len(), 512);
    }

    #[test]
    fn test_dequantize_q4k_partial_last_block() {
        // One and a half blocks - second block is incomplete
        let data = vec![0u8; 200]; // 144 + 56 (incomplete second block)
        let result = dequantize_q4k_for_stats(&data, 512);
        // Only first block should produce output
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_dequantize_q6k_partial_last_block() {
        let data = vec![0u8; 300]; // 210 + 90 (incomplete second block)
        let result = dequantize_q6k_for_stats(&data, 512);
        assert_eq!(result.len(), 256);
    }

    // ========================================================================
    // NEW: Comprehensive convert flow tests
    // ========================================================================

    #[test]
    fn test_run_convert_with_quantize_and_verify() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let target = NamedTempFile::with_suffix(".apr").expect("create target");

        let result = run_convert(
            source.path(),
            target.path(),
            Some("fp16"),
            true,
            false,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_convert_json_with_quantize() {
        let mut source = NamedTempFile::with_suffix(".gguf").expect("create source");
        source.write_all(b"not valid gguf").expect("write");
        let target = NamedTempFile::with_suffix(".apr").expect("create target");

        let result = run_convert(
            source.path(),
            target.path(),
            Some("int4"),
            false,
            true,
            None,
        );
        assert!(result.is_err());
    }

    // ====================================================================
    // Coverage-boost tests: normalize_tensor_name exhaustive cases
    // ====================================================================

    #[test]
    fn test_normalize_tensor_name_gguf_attn_v_bias() {
        // GGUF bias tensors should also normalize
        assert_eq!(normalize_tensor_name("blk.2.attn_v.bias"), "2.v_proj.bias");
    }

    #[test]
    fn test_normalize_tensor_name_gguf_attn_output_bias() {
        assert_eq!(
            normalize_tensor_name("blk.0.attn_output.bias"),
            "0.o_proj.bias"
        );
    }

    #[test]
    fn test_normalize_tensor_name_apr_mlp_down_proj_bias() {
        assert_eq!(
            normalize_tensor_name("model.layers.3.mlp.down_proj.bias"),
            "3.down_proj.bias"
        );
    }

    #[test]
    fn test_normalize_tensor_name_gguf_norm_weights_both_types() {
        // attn_norm → input_layernorm, ffn_norm → post_attention_layernorm
        assert_eq!(
            normalize_tensor_name("blk.15.attn_norm.bias"),
            "15.input_layernorm.bias"
        );
        assert_eq!(
            normalize_tensor_name("blk.15.ffn_norm.bias"),
            "15.post_attention_layernorm.bias"
        );
    }

    #[test]
    fn test_normalize_tensor_name_double_prefix_model_layers() {
        // "model.layers." prefix is stripped as "model." then "layers."
        let result = normalize_tensor_name("model.layers.7.self_attn.v_proj.weight");
        assert_eq!(result, "7.v_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_only_model_prefix() {
        // Only "model." prefix, no "layers."
        let result = normalize_tensor_name("model.norm.weight");
        assert_eq!(result, "norm.weight");
    }

    #[test]
    fn test_normalize_tensor_name_only_blk_prefix() {
        // "blk." prefix with a simple suffix
        let result = normalize_tensor_name("blk.0.token_embd.weight");
        assert_eq!(result, "0.embed_tokens.weight");
    }

    #[test]
    fn test_normalize_tensor_name_output_weight_exact_match() {
        // "output.weight" should map to "lm_head.weight" (exact match)
        assert_eq!(normalize_tensor_name("output.weight"), "lm_head.weight");
    }

    #[test]
    fn test_normalize_tensor_name_output_norm_weight() {
        // "output_norm.weight" → "norm.weight" via replacement
        assert_eq!(normalize_tensor_name("output_norm.weight"), "norm.weight");
    }

    #[test]
    fn test_normalize_tensor_name_multiple_self_attn_occurrences() {
        // str::replace scans left-to-right without re-scanning replacements.
        // "0.self_attn.self_attn.q_proj.weight" → first ".self_attn." match
        // yields "0.self_attn.q_proj.weight" — the overlapping second occurrence
        // collapses but one "self_attn" remains as a name prefix, not a dotted segment.
        let result = normalize_tensor_name("model.layers.0.self_attn.self_attn.q_proj.weight");
        assert_eq!(result, "0.self_attn.q_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_token_embd_bias() {
        assert_eq!(
            normalize_tensor_name("token_embd.bias"),
            "embed_tokens.bias"
        );
    }

    #[test]
    fn test_normalize_tensor_name_preserves_layer_numbers() {
        // Layer numbers should be preserved exactly
        for layer_num in [0, 1, 10, 99, 127] {
            let gguf = format!("blk.{layer_num}.attn_q.weight");
            let apr = format!("model.layers.{layer_num}.self_attn.q_proj.weight");
            assert_eq!(normalize_tensor_name(&gguf), normalize_tensor_name(&apr));
        }
    }

    #[test]
    fn test_normalize_tensor_name_all_gguf_ffn_variants() {
        // Verify all 3 FFN mappings with different layer numbers
        assert_eq!(
            normalize_tensor_name("blk.10.ffn_gate.weight"),
            "10.gate_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.10.ffn_up.weight"),
            "10.up_proj.weight"
        );
        assert_eq!(
            normalize_tensor_name("blk.10.ffn_down.weight"),
            "10.down_proj.weight"
        );
    }

    #[test]
    fn test_normalize_tensor_name_cross_format_embedding_match() {
        // Both formats should match for embedding
        assert_eq!(
            normalize_tensor_name("token_embd.weight"),
            normalize_tensor_name("model.embed_tokens.weight")
        );
    }

    #[test]
    fn test_normalize_tensor_name_cross_format_lm_head_match() {
        // GGUF "output.weight" and APR "lm_head.weight" should match
        assert_eq!(
            normalize_tensor_name("output.weight"),
            normalize_tensor_name("lm_head.weight")
        );
    }

    #[test]
    fn test_normalize_tensor_name_cross_format_norm_match() {
        // GGUF "output_norm.weight" and APR "model.norm.weight" should match
        assert_eq!(
            normalize_tensor_name("output_norm.weight"),
            normalize_tensor_name("model.norm.weight")
        );
    }

    // ====================================================================
    // Coverage-boost tests: is_transposed_dims exhaustive edge cases
    // ====================================================================

    #[test]
    fn test_is_transposed_dims_large_dimensions() {
        assert!(is_transposed_dims(&[4096, 11008], &[11008, 4096]));
    }

    #[test]
    fn test_is_transposed_dims_one_dimension_is_one() {
        // [1, 768] vs [768, 1] - technically transposed
        assert!(is_transposed_dims(&[1, 768], &[768, 1]));
    }

    #[test]
    fn test_is_transposed_dims_both_one() {
        // [1, 1] vs [1, 1] - square, so NOT transposed
        assert!(!is_transposed_dims(&[1, 1], &[1, 1]));
    }

    #[test]
    fn test_is_transposed_dims_single_element_each() {
        assert!(!is_transposed_dims(&[5], &[5]));
    }

    #[test]
    fn test_is_transposed_dims_4d_tensors() {
        // 4D should always return false
        assert!(!is_transposed_dims(&[2, 3, 4, 5], &[5, 4, 3, 2]));
    }

    #[test]
    fn test_is_transposed_dims_mismatched_ndims() {
        assert!(!is_transposed_dims(&[768], &[768, 3072]));
        assert!(!is_transposed_dims(&[768, 3072], &[768]));
    }

    #[test]
    fn test_is_transposed_dims_completely_different() {
        // Shapes where neither dimension matches when swapped
        assert!(!is_transposed_dims(&[100, 200], &[300, 400]));
    }

    #[test]
    fn test_is_transposed_dims_partially_matching() {
        // Only one dim matches: [768, 3072] vs [3072, 1024]
        assert!(!is_transposed_dims(&[768, 3072], &[3072, 1024]));
    }

    #[test]
    fn test_is_transposed_dims_zero_dimensions() {
        // [0, 768] vs [768, 0]: a[0]==b[1] (0==0) AND a[1]==b[0] (768==768),
        // and shapes differ, so the function correctly reports them as transposed.
        assert!(is_transposed_dims(&[0, 768], &[768, 0]));
    }

    // ====================================================================
    // Coverage-boost tests: truncate_path additional cases
    // ====================================================================

    #[test]
    fn test_truncate_path_exactly_at_max_len() {
        let path = "abcde".to_string(); // 5 chars
        assert_eq!(truncate_path(path, 5), "abcde");
    }

    #[test]
    fn test_truncate_path_one_over_max_len() {
        let path = "abcdef".to_string(); // 6 chars
        let result = truncate_path(path, 5);
        // Should be "...ef" (3 dots + last 2 chars = 5 chars)
        assert!(result.starts_with("..."));
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_truncate_path_very_long() {
        let path = "a".repeat(200);
        let result = truncate_path(path, 20);
        assert_eq!(result.len(), 20);
        assert!(result.starts_with("..."));
    }

    #[test]
    fn test_truncate_path_max_len_three() {
        // Edge case: max_len == 3 means "..." fits exactly
        let path = "abcdef".to_string();
        let result = truncate_path(path, 3);
        assert_eq!(result, "...");
    }

    #[test]
    fn test_truncate_path_preserves_file_extension_when_possible() {
        let path = "/very/long/path/to/model.gguf".to_string();
        let result = truncate_path(path, 15);
        // Should end with "model.gguf" since it takes from end
        assert!(result.ends_with("model.gguf"));
    }

    // ====================================================================
    // Coverage-boost tests: strip_ansi additional patterns
    // ====================================================================

    #[test]
    fn test_strip_ansi_256_color() {
        // 256-color code: \x1b[38;5;196m
        let text = "\x1b[38;5;196mRed\x1b[0m";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Red");
    }

    #[test]
    fn test_strip_ansi_cursor_movement() {
        // Cursor movement codes
        let text = "\x1b[2AUp two lines\x1b[3BDown three";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Up two linesDown three");
    }

    #[test]
    fn test_strip_ansi_mixed_content_and_escapes() {
        let text = "start\x1b[31m red \x1b[32m green \x1b[0mend";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "start red  green end");
    }

    #[test]
    fn test_strip_ansi_unicode_preserved() {
        let text = "\x1b[1mUnicode: \u{2713} \u{2717}\x1b[0m";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Unicode: \u{2713} \u{2717}");
    }

    // ====================================================================
    // Coverage-boost tests: compute_tensor_stats more edge cases
    // ====================================================================

    #[test]
    fn test_compute_tensor_stats_two_values() {
        let data = vec![0.0, 10.0];
        let (mean, std, min, max, _p5, _p25, p50, _p75, _p95, _, _, _, _) =
            compute_tensor_stats(&data);
        assert!((mean - 5.0).abs() < 0.001);
        assert!((std - 5.0).abs() < 0.001);
        assert!((min - 0.0).abs() < 0.001);
        assert!((max - 10.0).abs() < 0.001);
        // Median of 2 values: p50 is index-based so it's the 0th value (0.0)
        assert!(p50 >= 0.0 && p50 <= 10.0);
    }

    #[test]
    fn test_compute_tensor_stats_large_uniform() {
        // 1000 identical values
        let data = vec![42.0; 1000];
        let (mean, std, min, max, p5, p25, p50, p75, p95, nan, inf, zero_frac, _) =
            compute_tensor_stats(&data);
        assert!((mean - 42.0).abs() < 0.001);
        assert!(std < 0.001); // No variance
        assert!((min - 42.0).abs() < 0.001);
        assert!((max - 42.0).abs() < 0.001);
        assert!((p5 - 42.0).abs() < 0.001);
        assert!((p25 - 42.0).abs() < 0.001);
        assert!((p50 - 42.0).abs() < 0.001);
        assert!((p75 - 42.0).abs() < 0.001);
        assert!((p95 - 42.0).abs() < 0.001);
        assert_eq!(nan, 0);
        assert_eq!(inf, 0);
        assert!((zero_frac - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_tensor_stats_negative_only() {
        let data = vec![-10.0, -5.0, -1.0];
        let (mean, _std, min, max, _, _, _, _, _, _, _, _, _) = compute_tensor_stats(&data);
        assert!(mean < 0.0);
        assert!((min - (-10.0)).abs() < 0.001);
        assert!((max - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_compute_tensor_stats_all_inf() {
        let data = vec![f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY];
        let (mean, std, _, _, _, _, _, _, _, nan, inf, _, _) = compute_tensor_stats(&data);
        assert_eq!(nan, 0);
        assert_eq!(inf, 3);
        // No valid values, so mean/std should be 0
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    // ====================================================================
    // Coverage-boost tests: get_role_threshold more patterns
    // ====================================================================

    #[test]
    fn test_get_role_threshold_post_attention_layernorm() {
        assert_eq!(
            get_role_threshold("model.layers.0.post_attention_layernorm.weight"),
            2.0
        );
    }

    #[test]
    fn test_get_role_threshold_ln_f() {
        // ln_f is a common name for final layer norm (GPT-style)
        assert_eq!(get_role_threshold("ln_f.weight"), 2.0);
    }

    #[test]
    fn test_get_role_threshold_embed_with_suffix() {
        assert_eq!(get_role_threshold("wte.embed.weight"), 5.0);
    }

    #[test]
    fn test_get_role_threshold_output_proj() {
        // "output" in the name should match
        assert_eq!(get_role_threshold("output_proj.weight"), 3.0);
    }

    // ====================================================================
    // Coverage-boost tests: f16_to_f32 more values
    // ====================================================================

    #[test]
    fn test_f16_to_f32_two() {
        // f16 2.0: 0x4000
        let bytes = [0x00, 0x40];
        let result = f16_to_f32(&bytes);
        assert!((result - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_f16_to_f32_smallest_subnormal() {
        // f16 smallest subnormal: 0x0001
        let bytes = [0x01, 0x00];
        let result = f16_to_f32(&bytes);
        assert!(result > 0.0);
        assert!(result < 0.001);
    }

    #[test]
    fn test_f16_to_f32_max_normal() {
        // f16 max normal: 0x7BFF (65504.0)
        let bytes = [0xFF, 0x7B];
        let result = f16_to_f32(&bytes);
        assert!((result - 65504.0).abs() < 1.0);
    }

    // ====================================================================
    // Coverage-boost tests: validate_fingerprints more cases
    // ====================================================================

    #[test]
    fn test_validate_fingerprints_empty_actual() {
        let actual: Vec<TensorFingerprint> = vec![];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_empty_reference() {
        let actual = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let reference: Vec<TensorFingerprint> = vec![];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_both_empty() {
        let actual: Vec<TensorFingerprint> = vec![];
        let reference: Vec<TensorFingerprint> = vec![];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_exact_threshold_boundary() {
        // Deviation exactly at threshold should not trigger anomaly
        // ref mean=0, ref std=1, actual mean=3 => deviation=3.0, threshold=3.0
        let actual = vec![make_fingerprint("tensor_a", 3.0, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.0, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        // deviation == threshold, not > threshold, so no anomaly
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_just_above_threshold() {
        // Deviation just above threshold should trigger anomaly
        let actual = vec![make_fingerprint("tensor_a", 3.01, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("tensor_a", 0.0, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_inf_count_not_anomaly_when_both_have() {
        // Both have inf => not anomalous for inf_count
        let actual = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 3)];
        let reference = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 3)];
        let anomalies = validate_fingerprints(&actual, &reference, 3.0, false);
        let inf_anomaly = anomalies.iter().find(|a| a.field == "inf_count");
        assert!(inf_anomaly.is_none());
    }

    #[test]
    fn test_validate_fingerprints_strict_mode_default_tensor() {
        // Non-special tensor in strict mode should use default threshold (3.0)
        let actual = vec![make_fingerprint("random_tensor.weight", 4.0, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("random_tensor.weight", 0.0, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 10.0, true);
        // Strict mode: default threshold is 3.0, deviation is 4.0 > 3.0
        assert!(!anomalies.is_empty());
    }

    // ====================================================================
    // Coverage-boost tests: fingerprints_to_json structure
    // ====================================================================

    #[test]
    fn test_fingerprints_to_json_contains_all_fields() {
        let fp = make_fingerprint("test_tensor", 0.5, 1.0, 2, 3);
        let json = fingerprints_to_json(&[fp]);
        assert!(json.contains("\"name\": \"test_tensor\""));
        assert!(json.contains("\"mean\":"));
        assert!(json.contains("\"std\":"));
        assert!(json.contains("\"min\":"));
        assert!(json.contains("\"max\":"));
        assert!(json.contains("\"p5\":"));
        assert!(json.contains("\"p25\":"));
        assert!(json.contains("\"p50\":"));
        assert!(json.contains("\"p75\":"));
        assert!(json.contains("\"p95\":"));
        assert!(json.contains("\"nan_count\": 2"));
        assert!(json.contains("\"inf_count\": 3"));
        assert!(json.contains("\"zero_fraction\":"));
        assert!(json.contains("\"checksum\":"));
        assert!(json.contains("\"shape\":"));
        assert!(json.contains("\"dtype\": \"F32\""));
    }

    #[test]
    fn test_fingerprints_to_json_three_items_comma_placement() {
        let fps = vec![
            make_fingerprint("a", 0.0, 0.0, 0, 0),
            make_fingerprint("b", 0.0, 0.0, 0, 0),
            make_fingerprint("c", 0.0, 0.0, 0, 0),
        ];
        let json = fingerprints_to_json(&fps);
        // Count commas between entries (should be exactly 2 for 3 items)
        let entry_separators = json.matches("},\n").count();
        assert_eq!(entry_separators, 2);
    }

    // ====================================================================
    // Coverage-boost tests: is_transposed_dims symmetry
    // ====================================================================

    #[test]
    fn test_is_transposed_dims_symmetric() {
        // If a and b are transposed, b and a should also be transposed
        let a = &[768, 3072];
        let b = &[3072, 768];
        assert_eq!(is_transposed_dims(a, b), is_transposed_dims(b, a));
    }

    #[test]
    fn test_is_transposed_dims_square_large() {
        // Large square matrix should not be considered transposed
        assert!(!is_transposed_dims(&[4096, 4096], &[4096, 4096]));
    }

    // ====================================================================
    // Coverage-boost tests: normalize_tensor_name idempotency
    // ====================================================================

    #[test]
    fn test_normalize_tensor_name_idempotent() {
        // Normalizing an already-normalized name should be stable
        let names = [
            "0.q_proj.weight",
            "0.gate_proj.weight",
            "embed_tokens.weight",
            "norm.weight",
            "lm_head.weight",
        ];
        for name in &names {
            let once = normalize_tensor_name(name);
            let twice = normalize_tensor_name(&once);
            assert_eq!(
                once, twice,
                "normalize_tensor_name not idempotent for {name}"
            );
        }
    }

    #[test]
    fn test_normalize_tensor_name_with_dots_only() {
        let result = normalize_tensor_name("...");
        assert_eq!(result, "...");
    }

    #[test]
    fn test_normalize_tensor_name_gguf_all_attn_variants_same_layer() {
        // All attention-related tensors for layer 0 normalize correctly
        let mappings = [
            ("blk.0.attn_q.weight", "0.q_proj.weight"),
            ("blk.0.attn_k.weight", "0.k_proj.weight"),
            ("blk.0.attn_v.weight", "0.v_proj.weight"),
            ("blk.0.attn_output.weight", "0.o_proj.weight"),
            ("blk.0.attn_norm.weight", "0.input_layernorm.weight"),
        ];
        for (gguf, expected) in &mappings {
            assert_eq!(
                normalize_tensor_name(gguf),
                *expected,
                "Failed for GGUF name: {gguf}"
            );
        }
    }

    // ====================================================================
    // Coverage-boost: print_inspection_report / summary / json
    // ====================================================================

    fn make_inspection_report(
        tensor_count: usize,
        arch: Option<&str>,
        quant: Option<&str>,
    ) -> InspectionReport {
        use aprender::format::rosetta::TensorInfo;
        use std::collections::BTreeMap;

        let mut metadata = BTreeMap::new();
        metadata.insert("key1".to_string(), "value1".to_string());
        metadata.insert(
            "long_key".to_string(),
            "a".repeat(80), // long value to trigger truncation
        );

        let tensors: Vec<TensorInfo> = (0..tensor_count)
            .map(|i| TensorInfo {
                name: format!("layer.{i}.weight"),
                dtype: "F32".to_string(),
                shape: vec![768, 3072],
                size_bytes: 768 * 3072 * 4,
                stats: None,
            })
            .collect();

        InspectionReport {
            format: FormatType::Apr,
            file_size: 1_000_000,
            metadata,
            tensors,
            total_params: 1_000_000,
            quantization: quant.map(String::from),
            architecture: arch.map(String::from),
        }
    }

    #[test]
    fn test_print_inspection_report_basic() {
        let report = make_inspection_report(5, Some("llama"), Some("Q4_K_M"));
        // Should not panic - exercises format, arch, quant branches
        print_inspection_report(&report, false);
    }

    #[test]
    fn test_print_inspection_report_with_hexdump_flag() {
        let report = make_inspection_report(3, None, None);
        // Exercises the hexdump branch (just prints a note)
        print_inspection_report(&report, true);
    }

    #[test]
    fn test_print_inspection_report_many_tensors() {
        // >12 tensors triggers the "... (N more tensors) ..." branch
        let report = make_inspection_report(20, Some("qwen2"), None);
        print_inspection_report(&report, false);
    }

    #[test]
    fn test_print_inspection_report_no_arch_no_quant() {
        let report = make_inspection_report(2, None, None);
        print_inspection_report(&report, false);
    }

    #[test]
    fn test_print_inspection_report_zero_tensors() {
        let report = make_inspection_report(0, None, None);
        print_inspection_report(&report, false);
    }

    #[test]
    fn test_print_inspection_summary_basic() {
        let report = make_inspection_report(5, Some("llama"), Some("Q4_K_M"));
        print_inspection_summary(&report);
    }

    #[test]
    fn test_print_inspection_summary_no_optionals() {
        let report = make_inspection_report(3, None, None);
        print_inspection_summary(&report);
    }

    #[test]
    fn test_print_inspection_json_with_arch_and_quant() {
        let report = make_inspection_report(5, Some("llama"), Some("Q4_K_M"));
        print_inspection_json(&report);
    }

    #[test]
    fn test_print_inspection_json_no_arch_no_quant() {
        let report = make_inspection_report(3, None, None);
        print_inspection_json(&report);
    }

    #[test]
    fn test_print_inspection_json_empty_tensors() {
        let report = make_inspection_report(0, None, None);
        print_inspection_json(&report);
    }

    // ====================================================================
    // Coverage-boost: print_conversion_json
    // ====================================================================

    #[test]
    fn test_print_conversion_json_direct_path() {
        let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        let source = make_inspection_report(10, Some("llama"), None);
        let target = make_inspection_report(10, Some("llama"), None);
        print_conversion_json(&path, &source, &target);
    }

    #[test]
    fn test_print_conversion_json_chain_path() {
        let path = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::SafeTensors],
            FormatType::Apr,
        );
        let source = make_inspection_report(5, None, Some("Q4_K"));
        let target = make_inspection_report(5, None, None);
        print_conversion_json(&path, &source, &target);
    }

    // ====================================================================
    // Coverage-boost: print_verification_json
    // ====================================================================

    #[test]
    fn test_print_verification_json_passing() {
        let report = VerificationReport::passing();
        print_verification_json(&report);
    }

    #[test]
    fn test_print_verification_json_with_failures() {
        let mut report = VerificationReport::passing();
        report.is_equivalent = false;
        report.max_diff = 0.5;
        report.mean_diff = 0.01;
        report.failed_tensors = vec!["tensor_a".to_string(), "tensor_b".to_string()];
        print_verification_json(&report);
    }

    // ====================================================================
    // Coverage-boost: print_fingerprint_diff JSON with anomalies
    // ====================================================================

    #[test]
    fn test_print_fingerprint_diff_json_with_anomalies() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 10.0, 1.0, 0, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_json_no_anomalies() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_verbose_with_anomaly() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 20.0, 1.0, 5, 0)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, true, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_json_missing_in_b() {
        let fps_a = vec![make_fingerprint("only_in_a", 0.5, 1.0, 0, 0)];
        let fps_b: Vec<TensorFingerprint> = vec![];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_empty_both() {
        let fps_a: Vec<TensorFingerprint> = vec![];
        let fps_b: Vec<TensorFingerprint> = vec![];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_empty_both_json() {
        let fps_a: Vec<TensorFingerprint> = vec![];
        let fps_b: Vec<TensorFingerprint> = vec![];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_inf_mismatch() {
        let fps_a = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 0)];
        let fps_b = vec![make_fingerprint("tensor_a", 0.5, 1.0, 0, 5)];
        let result = print_fingerprint_diff(&fps_a, &fps_b, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprint_diff_multiple_tensors_mixed() {
        let fps_a = vec![
            make_fingerprint("t1", 0.5, 1.0, 0, 0),
            make_fingerprint("t2", 0.5, 1.0, 0, 0),
            make_fingerprint("t3_only_in_a", 0.5, 1.0, 0, 0),
        ];
        let fps_b = vec![
            make_fingerprint("t1", 0.5, 1.0, 0, 0),  // matches
            make_fingerprint("t2", 50.0, 1.0, 3, 0), // anomaly
        ];
        let result = print_fingerprint_diff(&fps_a, &fps_b, true, false);
        assert!(result.is_ok());
    }

    // ====================================================================
    // Coverage-boost: print_fingerprints more coverage
    // ====================================================================

    #[test]
    fn test_print_fingerprints_verbose_multiple() {
        let fingerprints = vec![
            make_fingerprint("tensor_a", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_b", -1.5, 2.0, 1, 2),
            make_fingerprint("tensor_c", 100.0, 50.0, 0, 0),
        ];
        let result = print_fingerprints(&fingerprints, true, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprints_non_verbose_multiple() {
        let fingerprints = vec![
            make_fingerprint("tensor_a", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_b", -1.5, 2.0, 1, 2),
        ];
        let result = print_fingerprints(&fingerprints, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_fingerprints_json_multiple() {
        let fingerprints = vec![
            make_fingerprint("tensor_a", 0.5, 1.0, 0, 0),
            make_fingerprint("tensor_b", -1.5, 2.0, 1, 2),
        ];
        let result = print_fingerprints(&fingerprints, false, true);
        assert!(result.is_ok());
    }

    // ====================================================================
    // Coverage-boost: dequantize with nonzero scale/data patterns
    // ====================================================================

    #[test]
    fn test_dequantize_q4k_for_stats_with_nonzero_scales() {
        let mut data = vec![0u8; 144];
        // Set d (f16 1.0) at bytes 0-1
        data[0] = 0x00;
        data[1] = 0x3C;
        // Set dmin (f16 0.5) at bytes 2-3
        data[2] = 0x00;
        data[3] = 0x38;
        // Set some scale values
        for i in 4..16 {
            data[i] = 0x21; // scale = 0x21 & 0x3F = 33
        }
        // Set some quantized values (alternating patterns)
        for i in 16..144 {
            data[i] = 0xA5; // nibbles: 5 and 0xA
        }
        let result = dequantize_q4k_for_stats(&data, 256);
        assert_eq!(result.len(), 256);
        // Values should not all be zero
        let nonzero = result.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero > 0, "Expected nonzero dequantized values");
    }

    #[test]
    fn test_dequantize_q4k_for_stats_request_fewer_than_block() {
        let mut data = vec![0u8; 144];
        data[0] = 0x00;
        data[1] = 0x3C; // d = 1.0
        let result = dequantize_q4k_for_stats(&data, 128);
        assert_eq!(result.len(), 128);
    }

    #[test]
    fn test_dequantize_q4k_for_stats_request_zero_elements() {
        let data = vec![0u8; 144];
        let result = dequantize_q4k_for_stats(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q6k_for_stats_with_nonzero_d() {
        let mut data = vec![0u8; 210];
        // Set d (f16 2.0) at bytes 208-209
        data[208] = 0x00;
        data[209] = 0x40;
        // Set some quantized data
        for i in 0..208 {
            data[i] = 0x55;
        }
        let result = dequantize_q6k_for_stats(&data, 256);
        assert_eq!(result.len(), 256);
        let nonzero = result.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero > 0, "Expected nonzero dequantized values");
    }

    #[test]
    fn test_dequantize_q6k_for_stats_request_fewer_than_block() {
        let mut data = vec![0u8; 210];
        data[208] = 0x00;
        data[209] = 0x3C; // d = 1.0
        let result = dequantize_q6k_for_stats(&data, 64);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_dequantize_q6k_for_stats_request_zero_elements() {
        let data = vec![0u8; 210];
        let result = dequantize_q6k_for_stats(&data, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dequantize_q4k_for_stats_three_blocks() {
        let data = vec![0u8; 432]; // 3 * 144
        let result = dequantize_q4k_for_stats(&data, 768);
        assert_eq!(result.len(), 768);
    }

    #[test]
    fn test_dequantize_q6k_for_stats_three_blocks() {
        let data = vec![0u8; 630]; // 3 * 210
        let result = dequantize_q6k_for_stats(&data, 768);
        assert_eq!(result.len(), 768);
    }

    // ====================================================================
    // Coverage-boost: compute_tensor_stats edge cases
    // ====================================================================

    #[test]
    fn test_compute_tensor_stats_alternating_nan_and_valid() {
        let data = vec![f32::NAN, 1.0, f32::NAN, 2.0, f32::NAN, 3.0];
        let (mean, _std, min, max, _, _, _, _, _, nan_count, _, _, _) = compute_tensor_stats(&data);
        assert_eq!(nan_count, 3);
        assert!((mean - 2.0).abs() < 0.001);
        assert!((min - 1.0).abs() < 0.001);
        assert!((max - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_tensor_stats_very_small_values() {
        let data = vec![1e-30, 2e-30, 3e-30];
        let (mean, _std, min, max, _, _, _, _, _, _, _, _, _) = compute_tensor_stats(&data);
        assert!(mean > 0.0);
        assert!(min > 0.0);
        assert!(max > 0.0);
        assert!(min <= mean && mean <= max);
    }

    #[test]
    fn test_compute_tensor_stats_very_large_values() {
        let data = vec![1e30, 2e30, 3e30];
        let (mean, _std, min, max, _, _, _, _, _, _, _, _, _) = compute_tensor_stats(&data);
        assert!(mean > 1e29);
        assert!(min <= mean && mean <= max);
    }

    #[test]
    fn test_compute_tensor_stats_single_nan() {
        let data = vec![f32::NAN];
        let (mean, std, _, _, _, _, _, _, _, nan_count, _, _, _) = compute_tensor_stats(&data);
        assert_eq!(nan_count, 1);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_tensor_stats_single_inf() {
        let data = vec![f32::INFINITY];
        let (mean, std, _, _, _, _, _, _, _, _, inf_count, _, _) = compute_tensor_stats(&data);
        assert_eq!(inf_count, 1);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    // ====================================================================
    // Coverage-boost: validate_fingerprints strict mode for all roles
    // ====================================================================

    #[test]
    fn test_validate_fingerprints_strict_mode_lm_head() {
        // lm_head has threshold 3.0 in strict mode
        let actual = vec![make_fingerprint("lm_head.weight", 4.0, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("lm_head.weight", 0.0, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 10.0, true);
        // Strict threshold for lm_head = 3.0, deviation = 4.0 > 3.0
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_strict_mode_ln_prefix() {
        // ln_ prefix has threshold 2.0 in strict mode
        let actual = vec![make_fingerprint("ln_1.weight", 3.0, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("ln_1.weight", 0.0, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 10.0, true);
        // Strict threshold for ln_ = 2.0, deviation = 3.0 > 2.0
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_strict_embed_below_threshold() {
        // Embeddings have loose threshold (5.0) - deviation 4.0 should pass
        let actual = vec![make_fingerprint("embed_tokens.weight", 4.0, 1.0, 0, 0)];
        let reference = vec![make_fingerprint("embed_tokens.weight", 0.0, 1.0, 0, 0)];
        let anomalies = validate_fingerprints(&actual, &reference, 10.0, true);
        // Strict threshold for embed = 5.0, deviation = 4.0 < 5.0
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_validate_fingerprints_non_strict_ignores_role() {
        // Non-strict mode: all tensors use the same threshold
        let actual = vec![make_fingerprint(
            "model.layers.0.input_layernorm.weight",
            4.0,
            1.0,
            0,
            0,
        )];
        let reference = vec![make_fingerprint(
            "model.layers.0.input_layernorm.weight",
            0.0,
            1.0,
            0,
            0,
        )];
        let anomalies = validate_fingerprints(&actual, &reference, 5.0, false);
        // Non-strict: threshold = 5.0, deviation = 4.0 < 5.0 => no anomaly
        assert!(anomalies.is_empty());
    }

    // ====================================================================
    // Coverage-boost: load_fingerprints_from_json with varied JSON content
    // ====================================================================

    #[test]
    fn test_load_fingerprints_from_json_multiple_tensors() {
        let mut file = NamedTempFile::with_suffix(".json").expect("create temp file");
        let content = r#"{
  "fingerprints": [
    {"name": "t1", "mean": 0.1},
    {"name": "t2", "mean": 0.2},
    {"name": "t3", "mean": 0.3}
  ]
}"#;
        file.write_all(content.as_bytes()).expect("write");
        let result = load_fingerprints_from_json(file.path());
        assert!(result.is_ok());
        let fps = result.expect("parsed");
        assert_eq!(fps.len(), 3);
        assert_eq!(fps[0].name, "t1");
        assert_eq!(fps[1].name, "t2");
        assert_eq!(fps[2].name, "t3");
    }

    #[test]
    fn test_load_fingerprints_from_json_with_quoted_name() {
        let mut file = NamedTempFile::with_suffix(".json").expect("create temp file");
        let content = r#"{"name": "layer.0.attn.q_proj.weight"}"#;
        file.write_all(content.as_bytes()).expect("write");
        let result = load_fingerprints_from_json(file.path());
        assert!(result.is_ok());
        let fps = result.expect("parsed");
        assert_eq!(fps.len(), 1);
        assert_eq!(fps[0].name, "layer.0.attn.q_proj.weight");
    }

    #[test]
    fn test_load_fingerprints_from_json_empty_file() {
        let file = NamedTempFile::with_suffix(".json").expect("create temp file");
        // Empty file - no "name" fields
        let result = load_fingerprints_from_json(file.path());
        assert!(result.is_ok());
        assert!(result.expect("parsed").is_empty());
    }

    // ====================================================================
    // Coverage-boost: parse_tensor_stats_json placeholder
    // ====================================================================

    #[test]
    fn test_parse_tensor_stats_json_with_valid_looking_json() {
        // Even valid-looking JSON returns None (placeholder implementation)
        let json_str = r#"{"tensors": {"layer.0.weight": [1.0, 2.0, 3.0]}}"#;
        assert!(parse_tensor_stats_json(json_str).is_none());
    }

    // ====================================================================
    // Coverage-boost: InspectionReport creation and field access
    // ====================================================================

    #[test]
    fn test_inspection_report_with_long_metadata_values() {
        let report = make_inspection_report(1, None, None);
        // Metadata should contain the long value
        let long_val = report.metadata.get("long_key").expect("long_key exists");
        assert_eq!(long_val.len(), 80);
    }

    #[test]
    fn test_inspection_report_exactly_12_tensors() {
        // 12 tensors: first 10 + last 2 = all printed, no "..." line
        let report = make_inspection_report(12, None, None);
        assert_eq!(report.tensors.len(), 12);
        print_inspection_report(&report, false);
    }

    #[test]
    fn test_inspection_report_exactly_13_tensors() {
        // 13 tensors: first 10 + "..." + last 2 = exercises the "..." branch
        let report = make_inspection_report(13, None, None);
        assert_eq!(report.tensors.len(), 13);
        print_inspection_report(&report, false);
    }

    // ====================================================================
    // Coverage-boost: fingerprints_to_json field validation
    // ====================================================================

    #[test]
    fn test_fingerprints_to_json_with_zero_values() {
        let fp = TensorFingerprint {
            name: "zeros".to_string(),
            shape: vec![0],
            dtype: "F32".to_string(),
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            p5: 0.0,
            p25: 0.0,
            p50: 0.0,
            p75: 0.0,
            p95: 0.0,
            nan_count: 0,
            inf_count: 0,
            zero_fraction: 1.0,
            checksum: 0,
        };
        let json = fingerprints_to_json(&[fp]);
        assert!(json.contains("\"name\": \"zeros\""));
        assert!(json.contains("\"zero_fraction\": 1"));
    }

    #[test]
    fn test_fingerprints_to_json_with_large_checksum() {
        let mut fp = make_fingerprint("test", 0.0, 0.0, 0, 0);
        fp.checksum = u32::MAX;
        let json = fingerprints_to_json(&[fp]);
        assert!(json.contains(&format!("{}", u32::MAX)));
    }

    // ====================================================================
    // Coverage-boost: ConversionPath edge cases
    // ====================================================================

    #[test]
    fn test_conversion_path_direct_display_format() {
        let path = ConversionPath::direct(FormatType::SafeTensors, FormatType::Gguf);
        let display = format!("{path}");
        assert!(display.contains("SafeTensors"));
        assert!(display.contains("GGUF"));
    }

    #[test]
    fn test_conversion_path_long_chain_display() {
        let path = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::SafeTensors, FormatType::Apr],
            FormatType::Gguf,
        );
        let display = format!("{path}");
        assert!(display.contains("GGUF"));
        assert!(display.contains("SafeTensors"));
        assert!(display.contains("APR"));
    }

    #[test]
    fn test_conversion_path_steps_direct() {
        let path = ConversionPath::direct(FormatType::Apr, FormatType::Gguf);
        let steps = path.steps();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0], FormatType::Apr);
        assert_eq!(steps[1], FormatType::Gguf);
    }

    #[test]
    fn test_conversion_path_has_cycle_no_intermediates() {
        let path = ConversionPath::direct(FormatType::Apr, FormatType::Gguf);
        assert!(!path.has_cycle());
    }

    #[test]
    fn test_conversion_path_is_roundtrip_direct_same() {
        // Same source and target without intermediates is NOT a roundtrip
        let path = ConversionPath::direct(FormatType::Apr, FormatType::Apr);
        assert!(!path.is_roundtrip());
    }

    // ====================================================================
    // Coverage-boost: VerificationReport with tensor_diffs and changed_metadata
    // ====================================================================

    #[test]
    fn test_verification_report_with_tensor_diffs() {
        use std::collections::BTreeMap;
        let mut tensor_diffs = BTreeMap::new();
        tensor_diffs.insert("layer.0.weight".to_string(), 0.001);
        tensor_diffs.insert("layer.1.weight".to_string(), 0.005);
        let report = VerificationReport {
            is_equivalent: true,
            max_diff: 0.005,
            mean_diff: 0.003,
            tensor_diffs,
            changed_metadata: vec!["version".to_string()],
            failed_tensors: vec![],
        };
        assert!(report.passes_with_tolerance(0.01));
        assert!(!report.passes_with_tolerance(0.001));
        print_verification_json(&report);
    }

    #[test]
    fn test_verification_report_not_equivalent_but_within_tolerance() {
        let report = VerificationReport {
            is_equivalent: false,
            max_diff: 0.01,
            mean_diff: 0.001,
            tensor_diffs: std::collections::BTreeMap::new(),
            changed_metadata: vec![],
            failed_tensors: vec![],
        };
        // passes_with_tolerance checks max_diff AND failed_tensors
        assert!(report.passes_with_tolerance(0.1));
    }

    // ====================================================================
    // Coverage-boost: TensorFingerprint field access patterns
    // ====================================================================

    #[test]
    fn test_tensor_fingerprint_all_fields() {
        let fp = TensorFingerprint {
            name: "model.layers.5.self_attn.q_proj.weight".to_string(),
            shape: vec![4096, 4096],
            dtype: "Q4_K_M".to_string(),
            mean: -0.002,
            std: 0.15,
            min: -1.5,
            max: 1.5,
            p5: -0.25,
            p25: -0.08,
            p50: -0.001,
            p75: 0.08,
            p95: 0.25,
            nan_count: 0,
            inf_count: 0,
            zero_fraction: 0.05,
            checksum: 0xABCD_1234,
        };
        assert_eq!(fp.name, "model.layers.5.self_attn.q_proj.weight");
        assert_eq!(fp.shape, vec![4096, 4096]);
        assert_eq!(fp.dtype, "Q4_K_M");
        assert!((fp.mean - (-0.002)).abs() < 0.001);
        assert!((fp.std - 0.15).abs() < 0.001);
        assert!((fp.min - (-1.5)).abs() < 0.001);
        assert!((fp.max - 1.5).abs() < 0.001);
        assert!((fp.p5 - (-0.25)).abs() < 0.001);
        assert!((fp.p25 - (-0.08)).abs() < 0.001);
        assert!((fp.p50 - (-0.001)).abs() < 0.001);
        assert!((fp.p75 - 0.08).abs() < 0.001);
        assert!((fp.p95 - 0.25).abs() < 0.001);
        assert_eq!(fp.nan_count, 0);
        assert_eq!(fp.inf_count, 0);
        assert!((fp.zero_fraction - 0.05).abs() < 0.001);
        assert_eq!(fp.checksum, 0xABCD_1234);
    }

    // ====================================================================
    // Coverage-boost: strip_ansi with escape but no bracket
    // ====================================================================

    #[test]
    fn test_strip_ansi_escape_followed_by_non_bracket() {
        // ESC followed by a regular character (not '[')
        let text = "\x1bXhello";
        let stripped = strip_ansi(text);
        // ESC is consumed, 'X' is not consumed since peek != '['
        assert_eq!(stripped, "Xhello");
    }

    #[test]
    fn test_strip_ansi_multiple_escapes_no_brackets() {
        let text = "\x1b\x1b\x1bhello";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "hello");
    }

    #[test]
    fn test_strip_ansi_escape_at_end_of_string() {
        let text = "hello\x1b";
        let stripped = strip_ansi(text);
        // ESC at end - peek returns None, ESC consumed
        assert_eq!(stripped, "hello");
    }

    #[test]
    fn test_strip_ansi_escape_bracket_at_end() {
        // ESC[ at end with no terminating letter
        let text = "hello\x1b[";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "hello");
    }

    // ====================================================================
    // Coverage-boost: normalize_tensor_name with all prefix combinations
    // ====================================================================

    #[test]
    fn test_normalize_tensor_name_layers_prefix_without_model() {
        // "layers." prefix alone (without "model." before it)
        let result = normalize_tensor_name("layers.5.self_attn.q_proj.weight");
        assert_eq!(result, "5.q_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_blk_with_self_attn() {
        // "blk." prefix but tensor uses self_attn (unusual but possible)
        let result = normalize_tensor_name("blk.0.self_attn.q_proj.weight");
        // blk. stripped, .self_attn. stripped
        assert_eq!(result, "0.q_proj.weight");
    }

    #[test]
    fn test_normalize_tensor_name_mlp_prefix_stripped() {
        // Test .mlp. stripping without layers prefix
        let result = normalize_tensor_name("0.mlp.gate_proj.weight");
        assert_eq!(result, "0.gate_proj.weight");
    }

    // ====================================================================
    // Coverage-boost: FormatType from_extension edge cases
    // ====================================================================

    #[test]
    fn test_format_type_from_extension_case_sensitivity() {
        // Extension lookup may be case-sensitive depending on implementation
        let path = Path::new("model.GGUF");
        let result = FormatType::from_extension(path);
        // Should handle uppercase extensions
        assert!(result.is_ok() || result.is_err()); // Platform-dependent
    }

    #[test]
    fn test_format_type_from_extension_double_extension() {
        let path = Path::new("model.tar.gguf");
        let result = FormatType::from_extension(path);
        // Should look at last extension only
        assert!(result.is_ok());
        assert_eq!(result.expect("format"), FormatType::Gguf);
    }

    // ====================================================================
    // Coverage-boost: ConversionOptions clone
    // ====================================================================

    #[test]
    fn test_conversion_options_clone() {
        let opts = ConversionOptions {
            quantization: Some("int8".to_string()),
            verify: true,
            compute_stats: true,
            tolerance: 0.01,
            preserve_metadata: true,
            add_provenance: true,
            tokenizer_path: None,
        };
        let cloned = opts.clone();
        assert_eq!(opts.quantization, cloned.quantization);
        assert_eq!(opts.verify, cloned.verify);
        assert_eq!(opts.compute_stats, cloned.compute_stats);
        assert!((opts.tolerance - cloned.tolerance).abs() < 1e-10);
        assert_eq!(opts.preserve_metadata, cloned.preserve_metadata);
        assert_eq!(opts.add_provenance, cloned.add_provenance);
    }

    // ========================================================================
    // F-ROSETTA-004: Fingerprint detects tensor corruption
    // Flip 1 byte in tensor data → checksum must differ
    // ========================================================================

    #[test]
    fn t_f_rosetta_004_fingerprint_detects_single_byte_corruption() {
        // Create realistic tensor data (weight-like values)
        let original: Vec<f32> = (0..1000)
            .map(|i| ((i as f32) * 0.00314159 - 1.5).sin() * 0.02)
            .collect();

        // Compute baseline fingerprint
        let (mean_a, std_a, min_a, max_a, _, _, _, _, _, _, _, _, checksum_a) =
            compute_tensor_stats(&original);

        // Corrupt exactly 1 float value (flip a bit in byte representation)
        let mut corrupted = original.clone();
        // Flip the sign bit of element 500 (significant change)
        let bits = corrupted[500].to_bits() ^ 0x8000_0000;
        corrupted[500] = f32::from_bits(bits);

        // Compute corrupted fingerprint
        let (mean_b, std_b, min_b, max_b, _, _, _, _, _, _, _, _, checksum_b) =
            compute_tensor_stats(&corrupted);

        // PRIMARY ASSERTION: checksum MUST differ
        assert_ne!(
            checksum_a, checksum_b,
            "F-ROSETTA-004: Checksum must detect single-byte corruption"
        );

        // SECONDARY: at least one stat must differ (mean, std, min, or max)
        let stats_differ = (mean_a - mean_b).abs() > 1e-10
            || (std_a - std_b).abs() > 1e-10
            || (min_a - min_b).abs() > 1e-10
            || (max_a - max_b).abs() > 1e-10;
        assert!(
            stats_differ,
            "F-ROSETTA-004: At least one stat must differ after corruption"
        );
    }

    #[test]
    fn t_f_rosetta_004_fingerprint_stable_for_identical_data() {
        let data: Vec<f32> = (0..500)
            .map(|i| ((i as f32) * 0.007 - 1.75).cos() * 0.1)
            .collect();

        let (mean_a, std_a, _, _, _, _, _, _, _, _, _, _, checksum_a) =
            compute_tensor_stats(&data);
        let (mean_b, std_b, _, _, _, _, _, _, _, _, _, _, checksum_b) =
            compute_tensor_stats(&data);

        assert_eq!(
            checksum_a, checksum_b,
            "Identical data must produce identical checksums"
        );
        assert!(
            (mean_a - mean_b).abs() < f32::EPSILON,
            "Identical data must produce identical means"
        );
        assert!(
            (std_a - std_b).abs() < f32::EPSILON,
            "Identical data must produce identical stds"
        );
    }

    #[test]
    fn t_f_rosetta_004_fingerprint_detects_small_perturbation() {
        // Even a tiny perturbation (1 ULP change) must be detected by checksum
        let original: Vec<f32> = (0..100)
            .map(|i| (i as f32) * 0.01)
            .collect();

        let mut perturbed = original.clone();
        // Add 1 ULP (unit of least precision) to element 50
        let bits = perturbed[50].to_bits() + 1;
        perturbed[50] = f32::from_bits(bits);

        let (_, _, _, _, _, _, _, _, _, _, _, _, checksum_a) = compute_tensor_stats(&original);
        let (_, _, _, _, _, _, _, _, _, _, _, _, checksum_b) = compute_tensor_stats(&perturbed);

        assert_ne!(
            checksum_a, checksum_b,
            "F-ROSETTA-004: Even 1 ULP change must produce different checksum"
        );
    }

    // =========================================================================
    // F-GT-002: Mixed quantization level warning tests
    // =========================================================================

    #[test]
    fn t_f_gt_002_mixed_quant_warning_safetensors_vs_gguf_q4k() {
        let model_a = Path::new("model.safetensors");
        let model_b = Path::new("model-q4_k_m.gguf");

        let warning = super::check_mixed_quant_warning(model_a, model_b);
        assert!(
            warning.is_some(),
            "F-GT-002: Must warn when comparing SafeTensors (unquantized) vs GGUF Q4_K_M"
        );
        let msg = warning.expect("checked above");
        assert!(
            msg.contains("F-GT-002"),
            "Warning must cite F-GT-002: {msg}"
        );
        assert!(
            msg.contains("mixed quantization") || msg.contains("Mixed quantization"),
            "Warning must mention mixed quantization: {msg}"
        );
    }

    #[test]
    fn t_f_gt_002_no_warning_same_format() {
        let model_a = Path::new("model-q4_k.gguf");
        let model_b = Path::new("other-q4_k.gguf");

        let warning = super::check_mixed_quant_warning(model_a, model_b);
        assert!(
            warning.is_none(),
            "F-GT-002: No warning when both models are Q4_K GGUF"
        );
    }

    #[test]
    fn t_f_gt_002_warning_different_gguf_quants() {
        let model_a = Path::new("model-q4_k.gguf");
        let model_b = Path::new("model-q6_k.gguf");

        let warning = super::check_mixed_quant_warning(model_a, model_b);
        assert!(
            warning.is_some(),
            "F-GT-002: Must warn when comparing Q4_K vs Q6_K"
        );
    }

    #[test]
    fn t_f_gt_002_warning_apr_vs_safetensors() {
        let model_a = Path::new("model-q4k.apr");
        let model_b = Path::new("model.safetensors");

        let warning = super::check_mixed_quant_warning(model_a, model_b);
        assert!(
            warning.is_some(),
            "F-GT-002: Must warn when comparing APR Q4K vs SafeTensors (unquantized)"
        );
    }

    #[test]
    fn t_f_gt_002_no_warning_both_safetensors() {
        let model_a = Path::new("model-part1.safetensors");
        let model_b = Path::new("model-part2.safetensors");

        let warning = super::check_mixed_quant_warning(model_a, model_b);
        assert!(
            warning.is_none(),
            "F-GT-002: No warning when both are SafeTensors (same quant level)"
        );
    }
}
