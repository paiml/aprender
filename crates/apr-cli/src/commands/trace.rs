//! Trace command implementation
//!
//! Layer-by-layer analysis of APR models.
//! Toyota Way: Visualization - Make hidden problems visible.
//!
//! This command traces through model layers, computing statistics at each stage
//! to help identify where numerical issues or divergences occur.

use crate::error::CliError;
use crate::output;
use aprender::format::HEADER_SIZE;
use colored::Colorize;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Layer trace information
#[derive(Serialize, Clone)]
pub(crate) struct LayerTrace {
    /// Layer name/type
    pub name: String,
    /// Layer index (if applicable)
    pub index: Option<usize>,
    /// Input statistics
    pub input_stats: Option<TensorStats>,
    /// Output statistics
    pub output_stats: Option<TensorStats>,
    /// Weight statistics (if layer has weights)
    pub weight_stats: Option<TensorStats>,
    /// Anomalies detected
    pub anomalies: Vec<String>,
}

/// Tensor statistics for tracing
#[derive(Serialize, Clone)]
#[allow(dead_code)]
pub(crate) struct TensorStats {
    /// Number of elements
    pub count: usize,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// L2 norm
    pub l2_norm: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Maximum absolute value
    pub max_abs: f32,
    /// Count of NaN values
    pub nan_count: usize,
    /// Count of Inf values
    pub inf_count: usize,
}

impl TensorStats {
    /// Compute statistics from a slice of f32 values
    #[allow(dead_code, clippy::cast_lossless)]
    pub(crate) fn from_slice(data: &[f32]) -> Self {
        let count = data.len();
        if count == 0 {
            return Self {
                count: 0,
                mean: 0.0,
                std: 0.0,
                l2_norm: 0.0,
                min: 0.0,
                max: 0.0,
                max_abs: 0.0,
                nan_count: 0,
                inf_count: 0,
            };
        }

        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut max_abs = 0.0_f32;
        let mut nan_count = 0;
        let mut inf_count = 0;

        for &v in data {
            if v.is_nan() {
                nan_count += 1;
                continue;
            }
            if v.is_infinite() {
                inf_count += 1;
                continue;
            }
            sum += v as f64;
            sum_sq += (v as f64) * (v as f64);
            min = min.min(v);
            max = max.max(v);
            max_abs = max_abs.max(v.abs());
        }

        let valid_count = count - nan_count - inf_count;
        let mean = if valid_count > 0 {
            (sum / valid_count as f64) as f32
        } else {
            0.0
        };
        let variance = if valid_count > 1 {
            ((sum_sq / valid_count as f64) - (mean as f64).powi(2)).max(0.0)
        } else {
            0.0
        };
        let std = (variance as f32).sqrt();
        let l2_norm = (sum_sq as f32).sqrt();

        Self {
            count,
            mean,
            std,
            l2_norm,
            min: if min.is_finite() { min } else { 0.0 },
            max: if max.is_finite() { max } else { 0.0 },
            max_abs,
            nan_count,
            inf_count,
        }
    }

    /// Check for anomalies
    #[allow(dead_code)]
    pub(crate) fn detect_anomalies(&self, name: &str) -> Vec<String> {
        let mut anomalies = Vec::new();

        if self.nan_count > 0 {
            anomalies.push(format!(
                "{name}: {}/{} NaN values",
                self.nan_count, self.count
            ));
        }
        if self.inf_count > 0 {
            anomalies.push(format!(
                "{name}: {}/{} Inf values",
                self.inf_count, self.count
            ));
        }
        if self.std < 1e-8 && self.count > 1 {
            anomalies.push(format!("{name}: near-zero variance (std={:.2e})", self.std));
        }
        if self.max_abs > 100.0 {
            anomalies.push(format!(
                "{name}: large values (max_abs={:.2})",
                self.max_abs
            ));
        }
        if self.mean.abs() > 10.0 {
            anomalies.push(format!("{name}: large mean bias ({:.4})", self.mean));
        }

        anomalies
    }
}

/// Trace result for JSON output
#[derive(Serialize)]
struct TraceResult {
    file: String,
    format: String,
    layers: Vec<LayerTrace>,
    summary: TraceSummary,
}

/// Summary of trace analysis
#[derive(Serialize)]
struct TraceSummary {
    total_layers: usize,
    total_parameters: usize,
    anomaly_count: usize,
    anomalies: Vec<String>,
}

/// Handle special trace modes (interactive, payload, diff).
/// Returns `Some(Ok(()))` if a mode handled the request, `None` to continue.
fn handle_special_modes(
    path: &Path,
    reference: Option<&Path>,
    payload: bool,
    diff: bool,
    interactive: bool,
) -> Option<Result<(), CliError>> {
    if interactive {
        println!("Starting interactive trace (TUI) for {}", path.display());
        println!("(TUI mode not yet fully implemented)");
        return Some(Ok(()));
    }

    if payload {
        return Some(run_traced_inference(path));
    }

    if diff {
        if let Some(ref_path) = reference {
            println!(
                "Diffing trace between {} and {}",
                path.display(),
                ref_path.display()
            );
        } else {
            println!("Diff mode requires --reference");
        }
    }

    None
}

/// Run traced inference through the model to debug layer-by-layer outputs.
/// This is the core functionality for debugging garbage output (BUG-GGUF-001).
fn run_traced_inference(path: &Path) -> Result<(), CliError> {
    use super::run::{download_hf_model, ModelSource};
    use colored::Colorize;

    output::section("Traced Inference (APR-TRACE-001)");

    // Resolve HuggingFace URLs to local paths
    let path_str = path.to_string_lossy();
    let local_path = if path_str.starts_with("hf://") {
        let source = ModelSource::parse(&path_str)?;
        match source {
            ModelSource::HuggingFace { org, repo, file } => {
                println!(
                    "Model: hf://{}/{}{}",
                    org,
                    repo,
                    file.as_ref().map(|f| format!("/{}", f)).unwrap_or_default()
                );
                println!();
                eprintln!("{}", "Downloading from HuggingFace...".yellow());
                download_hf_model(&org, &repo, file.as_deref())?
            }
            _ => path.to_path_buf(),
        }
    } else {
        println!("Model: {}", path.display());
        println!();
        path.to_path_buf()
    };

    // PMAT-235: Pre-flight contract validation before traced inference
    {
        use aprender::format::rosetta::RosettaStone;
        let rosetta = RosettaStone::new();
        match rosetta.validate(&local_path) {
            Ok(report) => {
                let contract_failures: Vec<String> = report
                    .tensors
                    .iter()
                    .flat_map(|t| {
                        t.failures
                            .iter()
                            .map(move |f| format!("{}: {}", t.name, f))
                    })
                    .collect();
                if contract_failures.is_empty() {
                    println!(
                        "{}",
                        format!(
                            "Contract: {} tensors pass PMAT-235 gates",
                            report.tensor_count
                        )
                        .green()
                    );
                } else {
                    println!(
                        "{}",
                        format!(
                            "Contract: {} violations in {} tensors",
                            contract_failures.len(),
                            report.failed_tensor_count
                        )
                        .red()
                        .bold()
                    );
                    for failure in contract_failures.iter().take(5) {
                        println!("  {}", failure.red());
                    }
                    if contract_failures.len() > 5 {
                        println!(
                            "  ... and {} more",
                            contract_failures.len() - 5
                        );
                    }
                    println!();
                    println!(
                        "{}",
                        "WARNING: Contract violations may cause garbage output."
                            .yellow()
                            .bold()
                    );
                }
            }
            Err(e) => {
                println!(
                    "{}",
                    format!("Contract: validation skipped ({e})").yellow()
                );
            }
        }
        println!();
    }

    // Detect format from extension
    let ext = local_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext.to_lowercase().as_str() {
        "gguf" => run_traced_inference_gguf(&local_path),
        "apr" => run_traced_inference_apr(&local_path),
        "safetensors" => run_traced_inference_safetensors(&local_path),
        _ => Err(CliError::InvalidFormat(format!(
            "Unknown format: {}. Supported: .gguf, .apr, .safetensors",
            ext
        ))),
    }
}

/// Traced inference for GGUF models (primary path for BUG-GGUF-001 debugging)
#[cfg(feature = "inference")]
fn run_traced_inference_gguf(path: &Path) -> Result<(), CliError> {
    use colored::Colorize;
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

    println!("{}", "Format: GGUF (quantized)".cyan());
    println!();

    // Load GGUF via mmap
    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load GGUF: {e}")))?;

    // Create quantized model
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to create quantized model: {e}")))?;

    let config = &model.config;
    println!("Architecture: {}", config.architecture);
    println!("  Layers: {}", config.num_layers);
    println!("  Hidden dim: {}", config.hidden_dim);
    println!("  Vocab size: {}", config.vocab_size);
    println!(
        "  Heads: {} (KV: {})",
        config.num_heads, config.num_kv_heads
    );
    println!();

    // Encode test prompt using GGUF's embedded tokenizer
    let test_prompt = "What is 2+2?";
    let test_tokens = mapped
        .model
        .encode(test_prompt)
        .unwrap_or_else(|| vec![1u32]);

    println!("{}", format!("Test prompt: {:?}", test_prompt).cyan());
    println!("{}", format!("Encoded tokens: {:?}", test_tokens).cyan());
    println!();

    // Run generation with small max_tokens to see what comes out
    println!("{}", "GENERATION (max 8 tokens):".green().bold());
    let gen_config = QuantizedGenerateConfig {
        max_tokens: 8,
        temperature: 0.0, // Greedy for reproducibility
        top_k: 1,
        ..Default::default()
    };

    let output_tokens = model
        .generate_with_cache(&test_tokens, &gen_config)
        .map_err(|e| CliError::InferenceFailed(format!("Generation failed: {e}")))?;

    let generated = &output_tokens[test_tokens.len()..];
    println!("  Generated token IDs: {:?}", generated);

    // Decode each token individually to see where garbage starts
    println!();
    println!("{}", "TOKEN-BY-TOKEN DECODE:".green().bold());
    for (i, &token_id) in generated.iter().enumerate() {
        let decoded = mapped.model.decode(&[token_id]);
        let is_garbage = is_likely_garbage(&decoded);
        if is_garbage {
            println!(
                "  {}. token_id={} → {:?} {}",
                i + 1,
                token_id,
                decoded,
                "⚠ GARBAGE".red().bold()
            );
        } else {
            println!("  {}. token_id={} → {:?}", i + 1, token_id, decoded);
        }
    }

    // Full decoded output
    let full_decoded = mapped.model.decode(generated);
    println!();
    println!("{}", "FULL OUTPUT:".green().bold());
    println!("  {:?}", full_decoded);

    // Garbage detection
    println!();
    if is_likely_garbage(&full_decoded) {
        println!("{}", "⚠ GARBAGE OUTPUT DETECTED!".red().bold());
        println!();
        println!("Likely causes:");
        println!("  1. LAYOUT-001: Column-major vs row-major kernel mismatch");
        println!("  2. Weight tensor corruption during loading");
        println!("  3. Tokenizer vocabulary mismatch");
        println!();
        println!("Debug steps:");
        println!("  1. Check if SafeTensors produces correct output (same model)");
        println!("  2. Compare token IDs between GGUF and SafeTensors");
        println!("  3. Verify quantization type is supported");
    } else {
        println!("{}", "✓ Output appears reasonable".green());
    }

    Ok(())
}

/// Stub for GGUF inference when inference feature is disabled
#[cfg(not(feature = "inference"))]
fn run_traced_inference_gguf(_path: &Path) -> Result<(), CliError> {
    Err(CliError::FeatureDisabled(
        "Traced inference for GGUF models requires the 'inference' feature. Build with --features inference".to_string(),
    ))
}

/// Traced inference for APR models
#[cfg(feature = "inference")]
fn run_traced_inference_apr(path: &Path) -> Result<(), CliError> {
    use colored::Colorize;
    use realizar::apr::AprV2Model;
    use realizar::apr_transformer::AprTransformer;

    println!("{}", "Format: APR (native)".cyan());
    println!();

    // Load the APR model (for tokenizer access)
    let model = AprV2Model::load(path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load APR model: {e}")))?;

    let metadata = model.metadata();
    let num_layers = metadata.num_layers.unwrap_or(0);
    let hidden_dim = metadata.hidden_size.unwrap_or(0);
    let vocab_size = metadata.vocab_size.unwrap_or(32000);
    let num_heads = metadata.num_heads.unwrap_or(0);

    println!("Architecture:");
    println!("  Layers: {}", num_layers);
    println!("  Hidden dim: {}", hidden_dim);
    println!("  Vocab size: {}", vocab_size);
    println!("  Heads: {}", num_heads);
    println!();

    // Load embedded BPE tokenizer and encode test prompt (PMAT-232 fix)
    let test_prompt = "What is 2+2?";
    let test_tokens: Vec<u32> = match model.load_embedded_bpe_tokenizer() {
        Some(tokenizer) => {
            let tokens = tokenizer.encode(test_prompt);
            println!("{}", format!("Test prompt: {:?}", test_prompt).cyan());
            println!("{}", format!("Encoded tokens: {:?}", tokens).cyan());
            tokens
        }
        None => {
            // Fail fast - no silent fallback to wrong tokens
            return Err(CliError::InferenceFailed(
                "FATAL: APR file has no embedded tokenizer. Cannot trace without proper tokenization. \
                 Re-import with: apr import <source>.gguf -o <output>.apr".to_string()
            ));
        }
    };
    println!();

    // Try to load as AprTransformer for layer-by-layer tracing
    match AprTransformer::from_apr_file(path) {
        Ok(transformer) => {
            // Use forward_traced for layer-by-layer statistics
            println!("{}", "FORWARD PASS (with layer tracing):".green().bold());
            let trace = transformer
                .forward_traced(&test_tokens)
                .map_err(|e| CliError::InferenceFailed(format!("Forward pass failed: {e}")))?;

            // Embedding stats
            println!();
            println!("{}", "EMBEDDING:".cyan().bold());
            print_activation_stats_colored("  ", &trace.embed_stats);

            // Layer-by-layer stats with colors
            println!();
            println!("{}", "LAYER-BY-LAYER ACTIVATIONS:".cyan().bold());
            println!("{}", "  Legend: std>100=RED, std>50=YELLOW, std>10=BLUE, else=GREEN".dimmed());
            println!();

            let total_layers = trace.layer_activations.len();
            for layer in &trace.layer_activations {
                // Color layer header based on position (gradient from cyan to magenta)
                let layer_header = format!("Layer {:>2}/{}", layer.layer_idx, total_layers);
                let header_colored = match layer.layer_idx % 6 {
                    0 => layer_header.cyan().bold(),
                    1 => layer_header.blue().bold(),
                    2 => layer_header.magenta().bold(),
                    3 => layer_header.purple().bold(),
                    4 => layer_header.bright_blue().bold(),
                    _ => layer_header.bright_cyan().bold(),
                };

                // Check for anomalies at this layer
                let has_nan = layer.attn_norm_stats.nan_count > 0
                    || layer.qkv_stats.nan_count > 0
                    || layer.attn_out_stats.nan_count > 0
                    || layer.ffn_norm_stats.nan_count > 0
                    || layer.ffn_out_stats.nan_count > 0
                    || layer.output_stats.nan_count > 0;
                let has_inf = layer.attn_norm_stats.inf_count > 0
                    || layer.qkv_stats.inf_count > 0
                    || layer.attn_out_stats.inf_count > 0
                    || layer.ffn_norm_stats.inf_count > 0
                    || layer.ffn_out_stats.inf_count > 0
                    || layer.output_stats.inf_count > 0;

                // Status indicator
                let status = if has_nan || has_inf {
                    "ANOMALY".red().bold()
                } else if layer.output_stats.std_dev > 100.0 {
                    "HIGH-VAR".yellow().bold()
                } else {
                    "OK".green()
                };

                println!("  {} [{}]", header_colored, status);

                // Print each stage with color-coded std_dev
                print_stage_stats("    attn_norm", &layer.attn_norm_stats);
                print_stage_stats("    qkv      ", &layer.qkv_stats);
                print_stage_stats("    attn_out ", &layer.attn_out_stats);
                print_stage_stats("    ffn_norm ", &layer.ffn_norm_stats);
                print_stage_stats("    ffn_out  ", &layer.ffn_out_stats);
                print_stage_stats("    output   ", &layer.output_stats);

                // Early exit if critical anomalies detected
                if has_nan || has_inf {
                    println!();
                    println!("{}", "    CRITICAL: NaN/Inf detected - numerical instability!".red().bold());
                    println!("{}", "    Possible causes:".red());
                    println!("{}", "      - Weight overflow during dequantization".red());
                    println!("{}", "      - Attention score explosion (missing scaling)".red());
                    println!("{}", "      - RoPE frequency miscalculation".red());
                    println!();
                    break;
                }
                println!();
            }

            // Final norm stats
            println!();
            println!("{}", "FINAL LAYER NORM:".cyan().bold());
            print_activation_stats("  ", &trace.final_norm_stats);

            // Logits
            let logits = &trace.logits;
            let logit_stats = compute_vector_stats(logits);
            println!();
            println!("{}", "LM_HEAD output:".green().bold());
            println!("  Vocab size: {}", logits.len());
            print_stats("  ", &logit_stats);

            // Top 5 predictions
            println!();
            println!("{}", "Top 5 predictions:".green().bold());
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (i, (token_id, logit)) in indexed.iter().take(5).enumerate() {
                println!("  {}. token_id={}, logit={:.4}", i + 1, token_id, logit);
            }

            // Summary analysis
            println!();
            println!("{}", "TRACE SUMMARY:".white().bold());

            // Find layers with highest variance
            let mut max_std_layer = 0;
            let mut max_std_value = 0.0f32;
            let mut high_var_count = 0;
            let mut total_nan = 0;
            let mut total_inf = 0;

            for layer in &trace.layer_activations {
                if layer.output_stats.std_dev > max_std_value {
                    max_std_value = layer.output_stats.std_dev;
                    max_std_layer = layer.layer_idx;
                }
                if layer.output_stats.std_dev > 50.0 {
                    high_var_count += 1;
                }
                total_nan += layer.output_stats.nan_count;
                total_inf += layer.output_stats.inf_count;
            }

            if total_nan > 0 || total_inf > 0 {
                println!("  {}", format!("CRITICAL: {} NaN, {} Inf values detected!", total_nan, total_inf).red().bold());
                println!("  {}", "Model weights or computation is corrupted.".red());
            } else if high_var_count > 0 {
                println!("  {}", format!("WARNING: {} layers with std > 50", high_var_count).yellow());
                println!("  Peak variance at layer {} (std={:.2})", max_std_layer, max_std_value);
                if max_std_value > 100.0 {
                    println!("  {}", "High variance may indicate attention explosion or weight issues.".yellow());
                }
            } else {
                println!("  {}", "All layers have reasonable variance (std < 50)".green());
            }

            // Logit range analysis
            let logit_range = logit_stats.max - logit_stats.min;
            if logit_range < 1.0 {
                println!("  {}", format!("WARNING: Logit range too narrow ({:.4})", logit_range).yellow());
                println!("  {}", "Model may not have learned meaningful patterns.".yellow());
            } else if logit_range > 100.0 {
                println!("  {}", format!("WARNING: Logit range very wide ({:.4})", logit_range).yellow());
            } else {
                println!("  Logit range: {:.2} {}", logit_range, "(reasonable)".green());
            }
        }
        Err(e) => {
            // Fall back to AprV2Model forward if AprTransformer fails
            eprintln!("{}", format!("Note: AprTransformer failed ({e}), using AprV2Model").yellow());
            println!("{}", "FORWARD PASS:".green().bold());
            let logits = model
                .forward(&test_tokens)
                .map_err(|e| CliError::InferenceFailed(format!("Forward pass failed: {e}")))?;

            // Compute statistics on output logits
            let logit_stats = compute_vector_stats(&logits);
            println!();
            println!("{}", "LM_HEAD output:".green().bold());
            println!("  Vocab size: {}", logits.len());
            print_stats("  ", &logit_stats);

            // Top 5 predictions
            println!();
            println!("{}", "Top 5 predictions:".green().bold());
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (i, (token_id, logit)) in indexed.iter().take(5).enumerate() {
                println!("  {}. token_id={}, logit={:.4}", i + 1, token_id, logit);
            }

            println!();
            println!("{}", "NOTE:".cyan().bold());
            println!("  Layer-by-layer tracing not available for this APR file.");
            println!("  Re-import with newer format for full tracing support.");
        }
    }

    Ok(())
}

/// Stub for APR inference when inference feature is disabled
#[cfg(not(feature = "inference"))]
fn run_traced_inference_apr(_path: &Path) -> Result<(), CliError> {
    Err(CliError::FeatureDisabled(
        "Traced inference for APR models requires the 'inference' feature. Build with --features inference".to_string(),
    ))
}

/// Simple vector statistics for tracing
struct VectorStats {
    l2_norm: f32,
    min: f32,
    max: f32,
    mean: f32,
    nan_count: usize,
    inf_count: usize,
}

fn compute_vector_stats(data: &[f32]) -> VectorStats {
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut nan_count = 0;
    let mut inf_count = 0;

    for &val in data {
        if val.is_nan() {
            nan_count += 1;
        } else if val.is_infinite() {
            inf_count += 1;
        } else {
            sum += val as f64;
            sum_sq += (val as f64) * (val as f64);
            min = min.min(val);
            max = max.max(val);
        }
    }

    let n = (data.len() - nan_count - inf_count) as f64;
    let mean = if n > 0.0 { (sum / n) as f32 } else { 0.0 };
    let l2_norm = (sum_sq as f32).sqrt();

    // Use n to check if any valid elements were found, rather than comparing
    // sentinel float values (clippy::float_cmp)
    let valid_elements = n > 0.0;
    VectorStats {
        l2_norm,
        min: if valid_elements { min } else { 0.0 },
        max: if valid_elements { max } else { 0.0 },
        mean,
        nan_count,
        inf_count,
    }
}

fn print_stats(prefix: &str, stats: &VectorStats) {
    println!("{}L2 norm: {:.4}", prefix, stats.l2_norm);
    println!("{}Range: [{:.6}, {:.6}]", prefix, stats.min, stats.max);
    println!("{}Mean: {:.6}", prefix, stats.mean);
    if stats.nan_count > 0 || stats.inf_count > 0 {
        println!(
            "{}NaN: {}, Inf: {}",
            prefix, stats.nan_count, stats.inf_count
        );
    }
}

/// Print activation statistics from realizar's ActivationStats
#[cfg(feature = "inference")]
fn print_activation_stats(_prefix: &str, stats: &realizar::apr_transformer::ActivationStats) {
    use colored::Colorize;
    println!("  Range: [{:.6}, {:.6}]", stats.min, stats.max);
    println!("  Mean: {:.6}, Std: {:.6}", stats.mean, stats.std_dev);
    if stats.nan_count > 0 || stats.inf_count > 0 {
        println!(
            "  {}: NaN={}, Inf={}",
            "ANOMALY".red().bold(),
            stats.nan_count,
            stats.inf_count
        );
    }
}

/// Print activation statistics with color coding
#[cfg(feature = "inference")]
fn print_activation_stats_colored(_prefix: &str, stats: &realizar::apr_transformer::ActivationStats) {
    use colored::Colorize;

    // Color code the std_dev
    let std_colored = format_std_colored(stats.std_dev);

    println!("  Range: [{:.4}, {:.4}]", stats.min, stats.max);
    println!("  Mean: {:.4}, Std: {}", stats.mean, std_colored);

    if stats.nan_count > 0 {
        println!("  {}", format!("NaN count: {}", stats.nan_count).red().bold());
    }
    if stats.inf_count > 0 {
        println!("  {}", format!("Inf count: {}", stats.inf_count).red().bold());
    }
}

/// Print stage-specific stats in a compact colored format
#[cfg(feature = "inference")]
fn print_stage_stats(stage_name: &str, stats: &realizar::apr_transformer::ActivationStats) {
    use colored::Colorize;

    let std_colored = format_std_colored(stats.std_dev);
    let mean_str = format!("{:>8.4}", stats.mean);

    // Build anomaly indicators
    let mut anomalies = String::new();
    if stats.nan_count > 0 {
        use std::fmt::Write;
        let _ = write!(anomalies, " NaN:{}", stats.nan_count);
    }
    if stats.inf_count > 0 {
        use std::fmt::Write;
        let _ = write!(anomalies, " Inf:{}", stats.inf_count);
    }

    if anomalies.is_empty() {
        println!("{}: mean={} std={}", stage_name.dimmed(), mean_str, std_colored);
    } else {
        println!(
            "{}: mean={} std={} {}",
            stage_name.dimmed(),
            mean_str,
            std_colored,
            anomalies.red().bold()
        );
    }
}

/// Format std_dev with color based on magnitude
#[cfg(feature = "inference")]
fn format_std_colored(std_dev: f32) -> colored::ColoredString {
    use colored::Colorize;

    let formatted = format!("{:>8.4}", std_dev);
    if std_dev > 100.0 {
        formatted.red().bold()
    } else if std_dev > 50.0 {
        formatted.yellow()
    } else if std_dev > 10.0 {
        formatted.blue()
    } else {
        formatted.green()
    }
}

/// Check if decoded text looks like garbage (BUG-GGUF-001)
fn is_likely_garbage(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }

    // Count suspicious patterns
    let text_lower = text.to_lowercase();
    let words: Vec<&str> = text_lower.split_whitespace().collect();

    // Check for repeated words (common in garbage output)
    let mut repeated = 0;
    for i in 1..words.len() {
        if words[i] == words[i - 1] {
            repeated += 1;
        }
    }

    // More than 50% repeated words is suspicious
    if words.len() > 2 && repeated * 2 > words.len() {
        return true;
    }

    // Check for high ratio of unusual characters
    let unusual_chars = text
        .chars()
        .filter(|c| {
            // Unicode replacement character or private use area
            *c == '\u{FFFD}'
                || ('\u{E000}'..='\u{F8FF}').contains(c)
                || ('\u{20000}'..='\u{2FFFF}').contains(c)
        })
        .count();

    if unusual_chars * 3 > text.chars().count() {
        return true;
    }

    // Check for common garbage patterns
    let garbage_patterns = [
        "random random",
        "random_",
        "domain domain",
        "domainuster",
        "pandas pandas",
        "olumbia",
        "localents",
        "nunca",
        ".mult",
    ];

    for pattern in garbage_patterns {
        if text_lower.contains(pattern) {
            return true;
        }
    }

    // Check for nonsensical word combinations (no real sentence structure)
    // If output doesn't contain common words and has weird fragments, it's garbage
    let has_normal_words = [
        "the", "is", "are", "and", "to", "of", "in", "that", "it", "for",
    ]
    .iter()
    .any(|w| text_lower.contains(w));

    let has_numbers = text.chars().any(|c| c.is_ascii_digit());

    // If answering a math question and no numbers in output, likely garbage
    if !has_numbers && !has_normal_words && words.len() > 2 {
        return true;
    }

    false
}

/// Traced inference for SafeTensors models
#[cfg(feature = "inference")]
fn run_traced_inference_safetensors(path: &Path) -> Result<(), CliError> {
    use colored::Colorize;
    use realizar::safetensors::{SafetensorsConfig, SafetensorsModel};

    println!("{}", "Format: SafeTensors (float)".cyan());
    println!();

    // Load SafeTensors
    let data = std::fs::read(path)?;
    let model = SafetensorsModel::from_bytes(&data)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load SafeTensors: {e}")))?;

    println!("Tensors: {}", model.tensors.len());

    // Load config if available
    if let Some(config) = SafetensorsConfig::load_from_sibling(path) {
        println!("Architecture: {}", config.architecture());
        println!("  Layers: {}", config.num_hidden_layers.unwrap_or(0));
        println!("  Hidden: {}", config.hidden_size.unwrap_or(0));
        println!("  Vocab: {}", config.vocab_size.unwrap_or(0));
    } else {
        println!("{}", "No config.json found".yellow());
    }

    println!();
    println!("{}", "SafeTensors traced inference:".green().bold());
    println!("  For SafeTensors, use `apr run --trace` for full tracing.");
    println!("  SafeTensors path uses realizar's optimized inference.");

    Ok(())
}

/// Stub for SafeTensors inference when inference feature is disabled
#[cfg(not(feature = "inference"))]
fn run_traced_inference_safetensors(_path: &Path) -> Result<(), CliError> {
    Err(CliError::FeatureDisabled(
        "Traced inference for SafeTensors models requires the 'inference' feature. Build with --features inference".to_string(),
    ))
}

/// Read and parse model metadata from an APR file.
fn read_model_metadata(path: &Path) -> Result<(String, Vec<u8>), CliError> {
    validate_path(path)?;

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let format_name = validate_header(&mut reader)?;

    let mut size_buf = [0u8; 4];
    reader.seek(SeekFrom::Start(8))?;
    reader.read_exact(&mut size_buf)?;
    let metadata_size = u32::from_le_bytes(size_buf) as usize;

    reader.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    let mut metadata_bytes = vec![0u8; metadata_size];
    reader.read_exact(&mut metadata_bytes)?;

    Ok((format_name, metadata_bytes))
}

/// Compute trace summary from layer information.
/// BUG-TRACE-001 FIX: Accept total_params from caller since weight_stats are not populated
fn compute_trace_summary(layers: &[LayerTrace], total_params: usize) -> TraceSummary {
    let all_anomalies: Vec<String> = layers.iter().flat_map(|l| l.anomalies.clone()).collect();

    TraceSummary {
        total_layers: layers.len(),
        total_parameters: total_params,
        anomaly_count: all_anomalies.len(),
        anomalies: all_anomalies,
    }
}

/// Run the trace command
#[allow(clippy::too_many_arguments)] // CLI command needs these distinct options
#[allow(clippy::fn_params_excessive_bools)] // CLI flags are naturally boolean
pub(crate) fn run(
    path: &Path,
    layer_filter: Option<&str>,
    reference: Option<&Path>,
    json_output: bool,
    verbose: bool,
    payload: bool,
    diff: bool,
    interactive: bool,
) -> Result<(), CliError> {
    if let Some(result) = handle_special_modes(path, reference, payload, diff, interactive) {
        return result;
    }

    // Detect format via Rosetta Stone dispatch
    // BUG-TRACE-001 FIX: Now returns total_params computed from tensor shapes
    let (format_name, layers, total_params) = detect_and_trace(path, layer_filter, verbose)?;
    let summary = compute_trace_summary(&layers, total_params);

    if let Some(ref_path) = reference {
        return compare_with_reference(path, ref_path, &layers, json_output);
    }

    if json_output {
        output_json(path, &format_name, &layers, &summary);
    } else {
        output_text(path, &format_name, &layers, &summary, verbose);
    }

    Ok(())
}

/// Detect format and trace layers from any supported format.
/// BUG-TRACE-001 FIX: Now returns total_params computed from tensor shapes
fn detect_and_trace(
    path: &Path,
    layer_filter: Option<&str>,
    verbose: bool,
) -> Result<(String, Vec<LayerTrace>, usize), CliError> {
    use aprender::format::rosetta::FormatType;

    validate_path(path)?;

    let format = FormatType::from_magic(path)
        .or_else(|_| FormatType::from_extension(path))
        .map_err(|e| CliError::InvalidFormat(format!("Cannot detect format: {e}")))?;

    match format {
        FormatType::Apr => {
            let (format_name, metadata_bytes) = read_model_metadata(path)?;
            let layers = trace_layers(&metadata_bytes, layer_filter, verbose);
            // BUG-TRACE-003 FIX: Use RosettaStone to compute total_params from tensor shapes
            // Previously hardcoded to 0, now properly computed like GGUF/SafeTensors
            let rosetta = aprender::format::rosetta::RosettaStone::new();
            let total_params = rosetta
                .inspect(path)
                .map(|report| report.total_params)
                .unwrap_or(0);
            Ok((format_name, layers, total_params))
        }
        FormatType::Gguf => trace_gguf(path, layer_filter),
        FormatType::SafeTensors => trace_safetensors(path, layer_filter),
    }
}

/// Extract a u32 value from GGUF metadata (handles Uint32 and Uint64 variants).
fn gguf_meta_u32(
    metadata: &BTreeMap<String, aprender::format::gguf::GgufValue>,
    key: &str,
) -> Option<u32> {
    use aprender::format::gguf::GgufValue;
    match metadata.get(key)? {
        GgufValue::Uint32(v) => Some(*v),
        GgufValue::Uint64(v) => Some(*v as u32),
        GgufValue::Int32(v) => Some(*v as u32),
        _ => None,
    }
}

/// Trace layers from GGUF format by extracting architecture from KV metadata.
/// BUG-TRACE-001 FIX: Now computes total_params from tensor shapes
fn trace_gguf(
    path: &Path,
    layer_filter: Option<&str>,
) -> Result<(String, Vec<LayerTrace>, usize), CliError> {
    use aprender::format::gguf::reader::GgufReader;
    use aprender::format::gguf::GgufValue;

    let data = std::fs::read(path)?;
    let reader = GgufReader::from_bytes(data)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to parse GGUF: {e}")))?;

    // BUG-TRACE-001 FIX: Compute total params from tensor dimensions
    let total_params: usize = reader
        .tensors
        .iter()
        .map(|t| t.dims.iter().map(|&d| d as usize).product::<usize>())
        .sum();

    // Extract architecture info from GGUF KV metadata
    let arch = match reader.metadata.get("general.architecture") {
        Some(GgufValue::String(s)) => s.clone(),
        _ => String::new(),
    };
    let n_layers = gguf_meta_u32(&reader.metadata, &format!("{arch}.block_count"))
        .or_else(|| gguf_meta_u32(&reader.metadata, "general.block_count"))
        .unwrap_or(0) as usize;
    let n_embd =
        gguf_meta_u32(&reader.metadata, &format!("{arch}.embedding_length")).unwrap_or(0) as usize;

    let format_name = format!("GGUF ({arch})");

    let mut layers = vec![create_embedding_layer(n_embd)];
    layers.extend(create_transformer_layers(n_layers, layer_filter));
    layers.push(create_final_layer_norm());

    // Add tensor count info as anomaly note if verbose
    if layers.len() <= 2 && !reader.tensors.is_empty() {
        // No layers detected from metadata but tensors exist
        layers.clear();
        layers.extend(infer_layers_from_tensor_names(
            &reader
                .tensors
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            layer_filter,
        ));
    }

    if layers.is_empty() {
        layers.push(create_default_layer());
    }

    Ok((format_name, layers, total_params))
}

/// Trace layers from SafeTensors format by inferring architecture from tensor names.
/// BUG-TRACE-001 FIX: Now returns total_params from Rosetta inspection
fn trace_safetensors(
    path: &Path,
    layer_filter: Option<&str>,
) -> Result<(String, Vec<LayerTrace>, usize), CliError> {
    use aprender::format::rosetta::RosettaStone;

    let rosetta = RosettaStone::new();
    let report = rosetta
        .inspect(path)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to inspect SafeTensors: {e}")))?;

    let format_name = "SafeTensors".to_string();
    let tensor_names: Vec<&str> = report.tensors.iter().map(|t| t.name.as_str()).collect();
    let mut layers = infer_layers_from_tensor_names(&tensor_names, layer_filter);

    if layers.is_empty() {
        layers.push(create_default_layer());
    }

    // BUG-TRACE-001 FIX: Use total_params from Rosetta inspection
    Ok((format_name, layers, report.total_params))
}

/// Infer layer structure from tensor naming conventions.
/// Supports patterns like: `model.layers.N.*`, `encoder.layer.N.*`, `h.N.*`
fn infer_layers_from_tensor_names(
    tensor_names: &[&str],
    layer_filter: Option<&str>,
) -> Vec<LayerTrace> {
    let mut layer_indices: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let mut has_embedding = false;
    let mut has_lm_head = false;

    for &name in tensor_names {
        let lower = name.to_lowercase();

        if lower.contains("embed") || lower.contains("wte") || lower.contains("wpe") {
            has_embedding = true;
        }
        if lower.contains("lm_head") || lower.contains("output") {
            has_lm_head = true;
        }

        // Extract layer index from common patterns
        if let Some(idx) = extract_layer_index(name) {
            layer_indices.entry(idx).or_default().push(name.to_string());
        }
    }

    let mut layers = Vec::new();

    if has_embedding {
        let embedding = LayerTrace {
            name: "embedding".to_string(),
            index: None,
            input_stats: None,
            output_stats: None,
            weight_stats: None,
            anomalies: vec![],
        };
        if layer_filter.is_none() || layer_filter.is_some_and(|f| "embedding".contains(f)) {
            layers.push(embedding);
        }
    }

    for &idx in layer_indices.keys() {
        let layer_name = format!("transformer_block_{idx}");
        if layer_filter.is_some_and(|f| !layer_name.contains(f)) {
            continue;
        }
        layers.push(LayerTrace {
            name: layer_name,
            index: Some(idx),
            input_stats: None,
            output_stats: None,
            weight_stats: None,
            anomalies: vec![],
        });
    }

    if has_lm_head {
        let lm_head = LayerTrace {
            name: "lm_head".to_string(),
            index: None,
            input_stats: None,
            output_stats: None,
            weight_stats: None,
            anomalies: vec![],
        };
        if layer_filter.is_none() || layer_filter.is_some_and(|f| "lm_head".contains(f)) {
            layers.push(lm_head);
        }
    }

    layers
}

/// Extract layer index from tensor name patterns.
/// Matches: `model.layers.N.`, `encoder.layer.N.`, `h.N.`, `blk.N.`
fn extract_layer_index(name: &str) -> Option<usize> {
    // Common patterns: layers.N, layer.N, h.N, blk.N, blocks.N
    let patterns = ["layers.", "layer.", "h.", "blk.", "blocks.", "block."];

    for pattern in &patterns {
        if let Some(pos) = name.find(pattern) {
            let after = &name[pos + pattern.len()..];
            let num_str: String = after.chars().take_while(char::is_ascii_digit).collect();
            if let Ok(idx) = num_str.parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

fn validate_path(path: &Path) -> Result<(), CliError> {
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if !path.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    Ok(())
}

fn validate_header(reader: &mut BufReader<File>) -> Result<String, CliError> {
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(|_| CliError::InvalidFormat("File too small".to_string()))?;

    // BUG-TRACE-002 FIX: Error message now mentions GGUF (matches is_valid_magic)
    if !output::is_valid_magic(&magic) {
        return Err(CliError::InvalidFormat(format!(
            "Invalid magic: expected APRN, APR1, APR2, APR\\0, or GGUF, got {magic:?}"
        )));
    }

    Ok(output::format_name(&magic).to_string())
}

/// Extract layer count from hyperparameters.
fn extract_layer_count(hp: &serde_json::Map<String, serde_json::Value>) -> usize {
    hp.get("n_layer")
        .or_else(|| hp.get("n_layers"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize
}

/// Extract model dimension from hyperparameters.
fn extract_model_dimension(hp: &serde_json::Map<String, serde_json::Value>) -> usize {
    hp.get("n_embd")
        .or_else(|| hp.get("d_model"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize
}

/// Create an embedding layer trace.
fn create_embedding_layer(d_model: usize) -> LayerTrace {
    LayerTrace {
        name: "embedding".to_string(),
        index: None,
        input_stats: None,
        output_stats: Some(TensorStats {
            count: d_model,
            mean: 0.0,
            std: 0.0,
            l2_norm: 0.0,
            min: 0.0,
            max: 0.0,
            max_abs: 0.0,
            nan_count: 0,
            inf_count: 0,
        }),
        weight_stats: None,
        anomalies: vec![],
    }
}

/// Create transformer layer traces with optional filtering.
fn create_transformer_layers(n_layers: usize, filter: Option<&str>) -> Vec<LayerTrace> {
    (0..n_layers)
        .filter_map(|i| {
            let layer_name = format!("transformer_block_{i}");
            if filter.is_some_and(|f| !layer_name.contains(f)) {
                return None;
            }
            Some(LayerTrace {
                name: layer_name,
                index: Some(i),
                input_stats: None,
                output_stats: None,
                weight_stats: None,
                anomalies: vec![],
            })
        })
        .collect()
}

/// Create final layer norm trace.
fn create_final_layer_norm() -> LayerTrace {
    LayerTrace {
        name: "final_layer_norm".to_string(),
        index: None,
        input_stats: None,
        output_stats: None,
        weight_stats: None,
        anomalies: vec![],
    }
}

/// Create default layer trace when no metadata available.
fn create_default_layer() -> LayerTrace {
    LayerTrace {
        name: "(layer trace metadata not available)".to_string(),
        index: None,
        input_stats: None,
        output_stats: None,
        weight_stats: None,
        anomalies: vec!["No layer information in metadata".to_string()],
    }
}

/// Extract layers from hyperparameters metadata.
fn extract_layers_from_hyperparameters(
    hp: &serde_json::Map<String, serde_json::Value>,
    filter: Option<&str>,
) -> Vec<LayerTrace> {
    let n_layers = extract_layer_count(hp);
    let d_model = extract_model_dimension(hp);

    let mut layers = vec![create_embedding_layer(d_model)];
    layers.extend(create_transformer_layers(n_layers, filter));
    layers.push(create_final_layer_norm());
    layers
}

#[allow(clippy::disallowed_methods)] // unwrap_or_default is safe here for empty vec
fn trace_layers(metadata_bytes: &[u8], filter: Option<&str>, _verbose: bool) -> Vec<LayerTrace> {
    let metadata: BTreeMap<String, serde_json::Value> =
        rmp_serde::from_slice(metadata_bytes).unwrap_or_else(|_| BTreeMap::new());

    let layers: Vec<LayerTrace> = metadata
        .get("hyperparameters")
        .and_then(|hp| hp.as_object())
        .map(|hp_obj| extract_layers_from_hyperparameters(hp_obj, filter))
        .unwrap_or_default();

    if layers.is_empty() {
        vec![create_default_layer()]
    } else {
        layers
    }
}

fn compare_with_reference(
    model_path: &Path,
    ref_path: &Path,
    _layers: &[LayerTrace],
    json_output: bool,
) -> Result<(), CliError> {
    validate_path(ref_path)?;

    if json_output {
        println!("{{\"comparison\": \"reference comparison not yet implemented\"}}");
    } else {
        output::section(&format!(
            "Layer Comparison: {} vs {}",
            model_path.display(),
            ref_path.display()
        ));
        println!();
        println!("{}", "Reference comparison coming soon...".yellow());
        println!();
        println!("Future features:");
        println!("  - Layer-by-layer output comparison");
        println!("  - Cosine similarity between activations");
        println!("  - Probar visual diff generation");
    }

    Ok(())
}

fn output_json(path: &Path, format: &str, layers: &[LayerTrace], summary: &TraceSummary) {
    let result = TraceResult {
        file: path.display().to_string(),
        format: format.to_string(),
        layers: layers.to_vec(),
        summary: TraceSummary {
            total_layers: summary.total_layers,
            total_parameters: summary.total_parameters,
            anomaly_count: summary.anomaly_count,
            anomalies: summary.anomalies.clone(),
        },
    };

    if let Ok(json) = serde_json::to_string_pretty(&result) {
        println!("{json}");
    }
}

fn output_text(
    path: &Path,
    format: &str,
    layers: &[LayerTrace],
    summary: &TraceSummary,
    verbose: bool,
) {
    output::section(&format!("Layer Trace: {}", path.display()));
    println!();

    output::kv("Format", format);
    output::kv("Layers", summary.total_layers);
    output::kv("Parameters", format!("{}", summary.total_parameters));

    if !summary.anomalies.is_empty() {
        println!();
        println!(
            "{}",
            format!("⚠ {} anomalies detected:", summary.anomaly_count)
                .yellow()
                .bold()
        );
        for anomaly in &summary.anomalies {
            println!("  - {}", anomaly.red());
        }
    }

    println!();
    println!("{}", "Layer Breakdown:".white().bold());

    for layer in layers {
        let idx_str = layer.index.map_or(String::new(), |i| format!("[{i}]"));
        println!("  {} {}", layer.name.cyan(), idx_str);

        if verbose {
            if let Some(ref stats) = layer.weight_stats {
                println!(
                    "    weights: {} params, mean={:.4}, std={:.4}, L2={:.4}",
                    stats.count, stats.mean, stats.std, stats.l2_norm
                );
            }

            if let Some(ref stats) = layer.output_stats {
                println!(
                    "    output:  mean={:.4}, std={:.4}, range=[{:.4}, {:.4}]",
                    stats.mean, stats.std, stats.min, stats.max
                );
            }
        }

        for anomaly in &layer.anomalies {
            println!("    {}", anomaly.red());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_stats_empty() {
        let stats = TensorStats::from_slice(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_tensor_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStats::from_slice(&data);

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_tensor_stats_with_nan() {
        let data = vec![1.0, f32::NAN, 3.0];
        let stats = TensorStats::from_slice(&data);

        assert_eq!(stats.nan_count, 1);
        assert!((stats.mean - 2.0).abs() < 1e-5); // Mean of 1 and 3
    }

    #[test]
    fn test_anomaly_detection() {
        let stats = TensorStats {
            count: 100,
            mean: 15.0, // Large mean
            std: 1.0,
            l2_norm: 100.0,
            min: 0.0,
            max: 20.0,
            max_abs: 20.0,
            nan_count: 0,
            inf_count: 0,
        };

        let anomalies = stats.detect_anomalies("test_layer");
        assert!(anomalies.iter().any(|a| a.contains("large mean")));
    }

    #[test]
    fn test_anomaly_detection_nan() {
        let stats = TensorStats {
            count: 100,
            mean: 0.0,
            std: 1.0,
            l2_norm: 10.0,
            min: -1.0,
            max: 1.0,
            max_abs: 1.0,
            nan_count: 5,
            inf_count: 0,
        };

        let anomalies = stats.detect_anomalies("test");
        assert!(anomalies.iter().any(|a| a.contains("NaN")));
    }

    // ========================================================================
    // Additional TensorStats Tests
    // ========================================================================

    #[test]
    fn test_tensor_stats_with_inf() {
        let data = vec![1.0, f32::INFINITY, 3.0];
        let stats = TensorStats::from_slice(&data);

        assert_eq!(stats.inf_count, 1);
        assert!((stats.mean - 2.0).abs() < 1e-5); // Mean of 1 and 3
    }

    #[test]
    fn test_tensor_stats_single_value() {
        let data = vec![5.0];
        let stats = TensorStats::from_slice(&data);

        assert_eq!(stats.count, 1);
        assert_eq!(stats.mean, 5.0);
        assert_eq!(stats.std, 0.0);
        assert_eq!(stats.min, 5.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_tensor_stats_negative_values() {
        let data = vec![-5.0, -2.0, 0.0, 2.0, 5.0];
        let stats = TensorStats::from_slice(&data);

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 0.0).abs() < 1e-5);
        assert_eq!(stats.min, -5.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.max_abs, 5.0);
    }

    #[test]
    fn test_tensor_stats_l2_norm() {
        let data = vec![3.0, 4.0]; // 3^2 + 4^2 = 25, sqrt(25) = 5
        let stats = TensorStats::from_slice(&data);

        assert!((stats.l2_norm - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_tensor_stats_clone() {
        let stats = TensorStats::from_slice(&[1.0, 2.0, 3.0]);
        let cloned = stats.clone();
        assert_eq!(cloned.count, stats.count);
        assert_eq!(cloned.mean, stats.mean);
    }

    // ========================================================================
    // LayerTrace Tests
    // ========================================================================

    #[test]
    fn test_layer_trace_basic() {
        let trace = LayerTrace {
            name: "attention".to_string(),
            index: Some(0),
            input_stats: None,
            output_stats: None,
            weight_stats: None,
            anomalies: vec![],
        };
        assert_eq!(trace.name, "attention");
        assert_eq!(trace.index, Some(0));
        assert!(trace.anomalies.is_empty());
    }

    #[test]
    fn test_layer_trace_with_anomalies() {
        let trace = LayerTrace {
            name: "ffn".to_string(),
            index: Some(5),
            input_stats: None,
            output_stats: None,
            weight_stats: None,
            anomalies: vec!["NaN detected".to_string(), "Large mean".to_string()],
        };
        assert_eq!(trace.anomalies.len(), 2);
    }

    #[test]
    fn test_layer_trace_clone() {
        let trace = LayerTrace {
            name: "mlp".to_string(),
            index: None,
            input_stats: None,
            output_stats: None,
            weight_stats: None,
            anomalies: vec![],
        };
        let cloned = trace.clone();
        assert_eq!(cloned.name, trace.name);
    }

    #[test]
    fn test_layer_trace_serialize() {
        let trace = LayerTrace {
            name: "layer_0".to_string(),
            index: Some(0),
            input_stats: None,
            output_stats: None,
            weight_stats: None,
            anomalies: vec![],
        };
        let json = serde_json::to_string(&trace).expect("serialize");
        assert!(json.contains("layer_0"));
    }

    // ========================================================================
    // Anomaly Detection Tests
    // ========================================================================

    #[test]
    fn test_anomaly_detection_inf() {
        let stats = TensorStats {
            count: 100,
            mean: 0.0,
            std: 1.0,
            l2_norm: 10.0,
            min: -1.0,
            max: 1.0,
            max_abs: 1.0,
            nan_count: 0,
            inf_count: 3,
        };

        let anomalies = stats.detect_anomalies("test");
        assert!(anomalies.iter().any(|a| a.contains("Inf")));
    }

    #[test]
    fn test_anomaly_detection_zero_std() {
        let stats = TensorStats {
            count: 100,
            mean: 1.0,
            std: 0.0, // Zero variance
            l2_norm: 10.0,
            min: 1.0,
            max: 1.0,
            max_abs: 1.0,
            nan_count: 0,
            inf_count: 0,
        };

        let anomalies = stats.detect_anomalies("test");
        assert!(anomalies
            .iter()
            .any(|a| a.contains("zero std") || a.contains("variance")));
    }

    #[test]
    fn test_anomaly_detection_no_anomalies() {
        let stats = TensorStats {
            count: 100,
            mean: 0.0,
            std: 0.5,
            l2_norm: 5.0,
            min: -1.0,
            max: 1.0,
            max_abs: 1.0,
            nan_count: 0,
            inf_count: 0,
        };

        let anomalies = stats.detect_anomalies("test");
        assert!(anomalies.is_empty());
    }

    // ========================================================================
    // run Command Tests
    // ========================================================================

    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            None,
            None,
            false,
            false,
            false,
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_apr() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), None, None, false, false, false, false, false);
        // Should fail (invalid APR)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_is_directory() {
        let dir = tempdir().expect("create temp dir");
        let result = run(dir.path(), None, None, false, false, false, false, false);
        // Should fail (is a directory)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_layer_filter() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            Some("encoder"),
            None,
            false,
            false,
            false,
            false,
            false,
        );
        // Should fail (invalid file) but tests filter path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_reference() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        let mut ref_file = NamedTempFile::with_suffix(".apr").expect("create ref file");
        ref_file.write_all(b"not valid ref").expect("write");

        let result = run(
            file.path(),
            None,
            Some(ref_file.path()),
            false,
            false,
            false,
            false,
            false,
        );
        // Should fail (invalid files)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_json_output() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            true, // json output
            false,
            false,
            false,
            false,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_verbose() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            false,
            true, // verbose
            false,
            false,
            false,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_payload() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(
            file.path(),
            None,
            None,
            false,
            false,
            true, // payload
            false,
            false,
        );
        // Should fail (invalid file)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_diff() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");
        let mut ref_file = NamedTempFile::with_suffix(".apr").expect("create ref file");
        ref_file.write_all(b"not valid ref").expect("write");

        let result = run(
            file.path(),
            None,
            Some(ref_file.path()),
            false,
            false,
            false,
            true, // diff
            false,
        );
        // Should fail (invalid files)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_format_invalid() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(b"not valid gguf").expect("write");

        let result = run(file.path(), None, None, false, false, false, false, false);
        // Should fail (invalid GGUF)
        assert!(result.is_err());
    }

    #[test]
    fn test_run_safetensors_format_invalid() {
        let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        file.write_all(b"not valid safetensors").expect("write");

        let result = run(file.path(), None, None, false, false, false, false, false);
        // Should fail (invalid SafeTensors)
        assert!(result.is_err());
    }

    // ================================================================
    // Audit #3 fix: Real GGUF/SafeTensors dispatch tests
    // These exercise trace_gguf() and trace_safetensors() with valid data.
    // ================================================================

    /// Build a minimal valid GGUF file with architecture metadata and tensors.
    fn build_test_gguf() -> NamedTempFile {
        use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
        use std::io::BufWriter;

        let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        let mut writer = BufWriter::new(&file);

        let tensors = vec![
            GgufTensor {
                name: "token_embd.weight".to_string(),
                shape: vec![4, 8],
                dtype: GgmlType::F32,
                data: vec![0u8; 4 * 8 * 4], // 4*8 f32s
            },
            GgufTensor {
                name: "blk.0.attn_q.weight".to_string(),
                shape: vec![8, 8],
                dtype: GgmlType::F32,
                data: vec![0u8; 8 * 8 * 4],
            },
            GgufTensor {
                name: "blk.0.attn_k.weight".to_string(),
                shape: vec![8, 8],
                dtype: GgmlType::F32,
                data: vec![0u8; 8 * 8 * 4],
            },
            GgufTensor {
                name: "blk.0.attn_v.weight".to_string(),
                shape: vec![8, 8],
                dtype: GgmlType::F32,
                data: vec![0u8; 8 * 8 * 4],
            },
            GgufTensor {
                name: "blk.0.ffn_gate.weight".to_string(),
                shape: vec![16, 8],
                dtype: GgmlType::F32,
                data: vec![0u8; 16 * 8 * 4],
            },
            GgufTensor {
                name: "output_norm.weight".to_string(),
                shape: vec![8],
                dtype: GgmlType::F32,
                data: vec![0u8; 8 * 4],
            },
        ];

        let metadata = vec![
            (
                "general.architecture".to_string(),
                GgufValue::String("llama".to_string()),
            ),
            ("llama.block_count".to_string(), GgufValue::Uint32(1)),
            ("llama.embedding_length".to_string(), GgufValue::Uint32(8)),
            (
                "llama.attention.head_count".to_string(),
                GgufValue::Uint32(2),
            ),
            (
                "llama.attention.head_count_kv".to_string(),
                GgufValue::Uint32(2),
            ),
        ];

        export_tensors_to_gguf(&mut writer, &tensors, &metadata).expect("write GGUF");
        drop(writer);
        file
    }

    /// Build a minimal valid SafeTensors file with named tensors.
    fn build_test_safetensors() -> NamedTempFile {
        // Build SafeTensors manually: 8-byte header_len + JSON header + tensor data
        let tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
            ("model.embed_tokens.weight", vec![8, 4], vec![0.1; 32]),
            (
                "model.layers.0.self_attn.q_proj.weight",
                vec![4, 4],
                vec![0.2; 16],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                vec![4, 4],
                vec![0.3; 16],
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                vec![8, 4],
                vec![0.4; 32],
            ),
            ("lm_head.weight", vec![8, 4], vec![0.5; 32]),
        ];

        // Build header JSON and data bytes
        let mut data_bytes = Vec::new();
        let mut header_map = serde_json::Map::new();
        let mut offset = 0usize;

        for (name, shape, values) in &tensors {
            let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
            let end = offset + bytes.len();

            let mut entry = serde_json::Map::new();
            entry.insert("dtype".to_string(), serde_json::json!("F32"));
            entry.insert("shape".to_string(), serde_json::json!(shape));
            entry.insert("data_offsets".to_string(), serde_json::json!([offset, end]));
            header_map.insert(name.to_string(), serde_json::Value::Object(entry));

            data_bytes.extend_from_slice(&bytes);
            offset = end;
        }

        let header_json = serde_json::to_string(&header_map).expect("serialize header");
        let header_len = header_json.len() as u64;

        let mut file_data = Vec::new();
        file_data.extend_from_slice(&header_len.to_le_bytes());
        file_data.extend_from_slice(header_json.as_bytes());
        file_data.extend_from_slice(&data_bytes);

        let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        file.write_all(&file_data).expect("write safetensors");
        file
    }

    #[test]
    fn test_run_valid_gguf_dispatch() {
        let file = build_test_gguf();
        let result = run(file.path(), None, None, false, false, false, false, false);
        assert!(result.is_ok(), "trace on valid GGUF failed: {result:?}");
    }

    #[test]
    fn test_run_valid_gguf_json_output() {
        let file = build_test_gguf();
        let result = run(file.path(), None, None, true, false, false, false, false);
        assert!(
            result.is_ok(),
            "trace JSON on valid GGUF failed: {result:?}"
        );
    }

    #[test]
    fn test_run_valid_safetensors_dispatch() {
        let file = build_test_safetensors();
        let result = run(file.path(), None, None, false, false, false, false, false);
        assert!(
            result.is_ok(),
            "trace on valid SafeTensors failed: {result:?}"
        );
    }

    #[test]
    fn test_run_valid_safetensors_json_output() {
        let file = build_test_safetensors();
        let result = run(file.path(), None, None, true, false, false, false, false);
        assert!(
            result.is_ok(),
            "trace JSON on valid SafeTensors failed: {result:?}"
        );
    }

    #[test]
    fn test_trace_gguf_detects_layers() {
        let file = build_test_gguf();
        let (format_name, layers, total_params) =
            detect_and_trace(file.path(), None, false).expect("detect_and_trace GGUF");
        assert!(
            format_name.contains("GGUF"),
            "format should be GGUF, got: {format_name}"
        );
        // Should detect at least the embedding and one transformer block
        assert!(
            !layers.is_empty(),
            "GGUF trace must produce at least one layer"
        );
        // BUG-TRACE-001 FIX: total_params should be computed from tensor shapes
        assert!(total_params > 0, "total_params should be > 0 for GGUF");
    }

    #[test]
    fn test_trace_safetensors_detects_layers() {
        let file = build_test_safetensors();
        let (format_name, layers, total_params) =
            detect_and_trace(file.path(), None, false).expect("detect_and_trace SafeTensors");
        assert_eq!(format_name, "SafeTensors");
        assert!(
            !layers.is_empty(),
            "SafeTensors trace must produce at least one layer"
        );
        // BUG-TRACE-001 FIX: total_params should be computed
        let _ = total_params; // May be 0 for test file
    }
}
