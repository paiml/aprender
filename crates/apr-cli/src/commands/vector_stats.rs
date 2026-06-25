
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
    // C-16 (Meyer DbC): 0 = unknown, no architecture-specific magic number.
    let vocab_size = metadata.vocab_size.unwrap_or(0);
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
            println!("{}", "FORWARD PASS (with layer tracing):".green().bold());
            let trace = transformer
                .forward_traced(&test_tokens)
                .map_err(|e| CliError::InferenceFailed(format!("Forward pass failed: {e}")))?;

            println!();
            println!("{}", "EMBEDDING:".cyan().bold());
            print_activation_stats_colored("  ", &trace.embed_stats);

            print_layer_activations(&trace.layer_activations);

            println!();
            println!("{}", "FINAL LAYER NORM:".cyan().bold());
            print_activation_stats("  ", &trace.final_norm_stats);

            print_logit_predictions(&trace.logits);
            print_trace_summary(&trace.layer_activations, &trace.logits);
        }
        Err(e) => {
            eprintln!(
                "{}",
                format!("Note: AprTransformer failed ({e}), using AprV2Model").yellow()
            );
            let logits = model
                .forward(&test_tokens)
                .map_err(|e| CliError::InferenceFailed(format!("Forward pass failed: {e}")))?;

            print_logit_predictions(&logits);

            println!();
            println!("{}", "NOTE:".cyan().bold());
            println!("  Layer-by-layer tracing not available for this APR file.");
            println!("  Re-import with newer format for full tracing support.");
        }
    }

    Ok(())
}

/// Print layer-by-layer activation stats with anomaly detection.
///
/// Made `pub(crate)` for SHIP-007 §26.4 P3: GGUF dispatch in
/// `commands/trace.rs::run_traced_inference_gguf` reuses this for
/// the APR-vs-GGUF layer-3 ffn_swigl bisection comparison.
#[cfg(feature = "inference")]
pub(crate) fn print_layer_activations(layers: &[realizar::apr_transformer::LayerActivation]) {
    use colored::Colorize;

    println!();
    println!("{}", "LAYER-BY-LAYER ACTIVATIONS:".cyan().bold());
    println!(
        "{}",
        "  Legend: std>100=RED, std>50=YELLOW, std>10=BLUE, else=GREEN".dimmed()
    );
    println!();

    let total_layers = layers.len();
    for layer in layers {
        let layer_header = format!("Layer {:>2}/{}", layer.layer_idx, total_layers);
        let header_colored = match layer.layer_idx % 6 {
            0 => layer_header.cyan().bold(),
            1 => layer_header.blue().bold(),
            2 => layer_header.magenta().bold(),
            3 => layer_header.purple().bold(),
            4 => layer_header.bright_blue().bold(),
            _ => layer_header.bright_cyan().bold(),
        };

        let has_nan = layer_has_nan(layer);
        let has_inf = layer_has_inf(layer);

        let status = if has_nan || has_inf {
            "ANOMALY".red().bold()
        } else if layer.output_stats.std_dev > 100.0 {
            "HIGH-VAR".yellow().bold()
        } else {
            "OK".green()
        };

        println!("  {} [{}]", header_colored, status);

        print_stage_stats("    attn_norm", &layer.attn_norm_stats);
        print_stage_stats("    qkv      ", &layer.qkv_stats);
        print_stage_stats("    attn_out ", &layer.attn_out_stats);
        print_stage_stats("    ffn_norm ", &layer.ffn_norm_stats);
        // Per contracts/trace-ffn-sub-block-v1.yaml OBL-SUB-FFN-006:
        // sub-FFN telemetry lines emitted in computation order between
        // ffn_norm and ffn_out. Lines suppressed when all 4 sub-stats
        // are default-zero (GPU path until OBL-SUB-FFN-008 lands).
        if layer.ffn_gate_stats.count > 0 || layer.ffn_up_stats.count > 0 {
            print_stage_stats("    ffn_gate ", &layer.ffn_gate_stats);
            print_stage_stats("    ffn_up   ", &layer.ffn_up_stats);
            print_stage_stats("    ffn_silu ", &layer.ffn_silu_gate_stats);
            print_stage_stats("    ffn_swigl", &layer.ffn_swiglu_inner_stats);
        }
        print_stage_stats("    ffn_out  ", &layer.ffn_out_stats);
        print_stage_stats("    output   ", &layer.output_stats);

        if has_nan || has_inf {
            println!();
            println!(
                "{}",
                "    CRITICAL: NaN/Inf detected - numerical instability!"
                    .red()
                    .bold()
            );
            println!("{}", "    Possible causes:".red());
            println!("{}", "      - Weight overflow during dequantization".red());
            println!(
                "{}",
                "      - Attention score explosion (missing scaling)".red()
            );
            println!("{}", "      - RoPE frequency miscalculation".red());
            println!();
            break;
        }
        println!();
    }
}

/// Check if a layer has any NaN values.
#[cfg(feature = "inference")]
fn layer_has_nan(layer: &realizar::apr_transformer::LayerActivation) -> bool {
    layer.attn_norm_stats.nan_count > 0
        || layer.qkv_stats.nan_count > 0
        || layer.attn_out_stats.nan_count > 0
        || layer.ffn_norm_stats.nan_count > 0
        || layer.ffn_out_stats.nan_count > 0
        || layer.output_stats.nan_count > 0
}

/// Check if a layer has any Inf values.
#[cfg(feature = "inference")]
fn layer_has_inf(layer: &realizar::apr_transformer::LayerActivation) -> bool {
    layer.attn_norm_stats.inf_count > 0
        || layer.qkv_stats.inf_count > 0
        || layer.attn_out_stats.inf_count > 0
        || layer.ffn_norm_stats.inf_count > 0
        || layer.ffn_out_stats.inf_count > 0
        || layer.output_stats.inf_count > 0
}

/// Print logit predictions with top-5 tokens.
#[cfg(feature = "inference")]
pub(crate) fn print_logit_predictions(logits: &[f32]) {
    use colored::Colorize;

    let logit_stats = compute_vector_stats(logits);
    println!();
    println!("{}", "LM_HEAD output:".green().bold());
    println!("  Vocab size: {}", logits.len());
    print_stats("  ", &logit_stats);

    println!();
    println!("{}", "Top 5 predictions:".green().bold());
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (i, (token_id, logit)) in indexed.iter().take(5).enumerate() {
        println!("  {}. token_id={}, logit={:.4}", i + 1, token_id, logit);
    }
}

/// Print trace summary analysis (variance, NaN/Inf, logit range).
#[cfg(feature = "inference")]
pub(crate) fn print_trace_summary(
    layers: &[realizar::apr_transformer::LayerActivation],
    logits: &[f32],
) {
    use colored::Colorize;

    println!();
    println!("{}", "TRACE SUMMARY:".white().bold());

    let mut max_std_layer = 0;
    let mut max_std_value = 0.0f32;
    let mut high_var_count = 0;
    let mut total_nan = 0;
    let mut total_inf = 0;

    for layer in layers {
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
        println!(
            "  {}",
            format!(
                "CRITICAL: {} NaN, {} Inf values detected!",
                total_nan, total_inf
            )
            .red()
            .bold()
        );
        println!("  {}", "Model weights or computation is corrupted.".red());
    } else if high_var_count > 0 {
        println!(
            "  {}",
            format!("WARNING: {} layers with std > 50", high_var_count).yellow()
        );
        println!(
            "  Peak variance at layer {} (std={:.2})",
            max_std_layer, max_std_value
        );
        if max_std_value > 100.0 {
            println!(
                "  {}",
                "High variance may indicate attention explosion or weight issues.".yellow()
            );
        }
    } else {
        println!(
            "  {}",
            "All layers have reasonable variance (std < 50)".green()
        );
    }

    let logit_stats = compute_vector_stats(logits);
    let logit_range = logit_stats.max - logit_stats.min;
    if logit_range < 1.0 {
        println!(
            "  {}",
            format!("WARNING: Logit range too narrow ({:.4})", logit_range).yellow()
        );
        println!(
            "  {}",
            "Model may not have learned meaningful patterns.".yellow()
        );
    } else if logit_range > 100.0 {
        println!(
            "  {}",
            format!("WARNING: Logit range very wide ({:.4})", logit_range).yellow()
        );
    } else {
        println!(
            "  Logit range: {:.2} {}",
            logit_range,
            "(reasonable)".green()
        );
    }
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
pub(crate) fn print_activation_stats(_prefix: &str, stats: &realizar::apr_transformer::ActivationStats) {
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
pub(crate) fn print_activation_stats_colored(
    _prefix: &str,
    stats: &realizar::apr_transformer::ActivationStats,
) {
    use colored::Colorize;

    // Color code the std_dev
    let std_colored = format_std_colored(stats.std_dev);

    println!("  Range: [{:.4}, {:.4}]", stats.min, stats.max);
    println!("  Mean: {:.4}, Std: {}", stats.mean, std_colored);

    if stats.nan_count > 0 {
        println!(
            "  {}",
            format!("NaN count: {}", stats.nan_count).red().bold()
        );
    }
    if stats.inf_count > 0 {
        println!(
            "  {}",
            format!("Inf count: {}", stats.inf_count).red().bold()
        );
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
        println!(
            "{}: mean={} std={}",
            stage_name.dimmed(),
            mean_str,
            std_colored
        );
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

// =============================================================================
// PMAT-125 B4: compute_vector_stats / format_std_colored coverage
// =============================================================================

#[cfg(test)]
mod vector_stats_tests {
    use super::*;

    #[test]
    fn test_compute_vector_stats_basic() {
        let stats = compute_vector_stats(&[1.0, 2.0, 3.0, 4.0]);
        assert!((stats.mean - 2.5).abs() < 1e-5);
        assert!((stats.min - 1.0).abs() < 1e-5);
        assert!((stats.max - 4.0).abs() < 1e-5);
        assert!((stats.l2_norm - 30.0_f32.sqrt()).abs() < 1e-3);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);
    }

    #[test]
    fn test_compute_vector_stats_empty() {
        let stats = compute_vector_stats(&[]);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.l2_norm, 0.0);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.inf_count, 0);
    }

    #[test]
    fn test_compute_vector_stats_counts_nan_and_inf() {
        let data = [1.0, f32::NAN, 2.0, f32::INFINITY, f32::NEG_INFINITY, 3.0];
        let stats = compute_vector_stats(&data);
        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.inf_count, 2);
        assert!((stats.mean - 2.0).abs() < 1e-5);
        assert!((stats.min - 1.0).abs() < 1e-5);
        assert!((stats.max - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_vector_stats_all_nan_no_valid() {
        let data = [f32::NAN, f32::NAN];
        let stats = compute_vector_stats(&data);
        assert_eq!(stats.nan_count, 2);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
    }

    #[test]
    fn test_compute_vector_stats_negative_values() {
        let stats = compute_vector_stats(&[-5.0, -1.0, -3.0]);
        assert!((stats.min - (-5.0)).abs() < 1e-5);
        assert!((stats.max - (-1.0)).abs() < 1e-5);
        assert!((stats.mean - (-3.0)).abs() < 1e-5);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn test_format_std_colored_thresholds() {
        for v in [5.0_f32, 25.0, 75.0, 150.0] {
            let s = format_std_colored(v);
            assert!(s.to_string().contains(&format!("{v:.4}")));
        }
    }
}
