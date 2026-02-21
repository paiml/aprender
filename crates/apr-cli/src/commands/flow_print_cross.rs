
/// Print cross-attention flow (critical for debugging Posterior Collapse)
#[allow(clippy::too_many_lines)] // Visual debugging output requires detailed formatting
fn print_cross_attention_flow(
    reader: Option<&AprReader>,
    tensor_names: &[String],
    layer_filter: Option<&str>,
    verbose: bool,
) {
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!("{}", "                   CROSS-ATTENTION DATA FLOW".bold());
    println!(
        "{}",
        "    (Posterior Collapse occurs when decoder ignores this)".dimmed()
    );
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!();

    // Find cross-attention layers
    let cross_attn_layers: Vec<_> = tensor_names
        .iter()
        .filter(|n| n.contains("encoder_attn") || n.contains("cross_attn"))
        .filter(|n| n.contains("q_proj.weight"))
        .filter(|n| layer_filter.map_or(true, |f| n.contains(f)))
        .collect();

    if cross_attn_layers.is_empty() {
        println!("{}", "No cross-attention layers found".yellow());
        return;
    }

    println!(
        "Found {} cross-attention Q projections\n",
        cross_attn_layers.len().to_string().green()
    );

    for q_weight_name in &cross_attn_layers {
        let layer_prefix = q_weight_name
            .strip_suffix(".q_proj.weight")
            .unwrap_or(q_weight_name);

        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│  {}  │", layer_prefix.cyan().bold());
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│                                                             │");
        println!(
            "│   {} [seq_len, d_model]              │",
            "encoder_output".green()
        );
        println!("│          │                                                  │");
        println!("│          ├───────────────┬───────────────┐                  │");
        println!("│          ▼               ▼               │                  │");
        println!("│    ┌──────────┐    ┌──────────┐          │                  │");
        println!(
            "│    │ {} │    │ {} │          │                  │",
            "w_k".yellow(),
            "w_v".yellow()
        );
        println!("│    └────┬─────┘    └────┬─────┘          │                  │");
        println!("│         ▼               ▼               │                  │");
        println!("│    K [seq, d]      V [seq, d]           │                  │");
        println!("│         │               │               │                  │");
        println!(
            "│   {} [dec_len, d_model]    │               │                  │",
            "decoder_hidden".magenta()
        );
        println!("│          │               │               │                  │");
        println!("│          ▼               │               │                  │");
        println!("│    ┌──────────┐          │               │                  │");
        println!(
            "│    │ {} │          │               │                  │",
            "w_q".yellow()
        );
        println!("│    └────┬─────┘          │               │                  │");
        println!("│         ▼               │               │                  │");
        println!("│    Q [dec, d]           │               │                  │");
        println!("│         │               │               │                  │");
        println!("│         └───────┬───────┘               │                  │");
        println!("│                 ▼                       │                  │");
        println!("│    ┌────────────────────────┐           │                  │");
        println!("│    │ Q @ K^T / √d_k         │           │                  │");
        println!("│    │ = scores [dec, seq]    │           │                  │");
        println!("│    └───────────┬────────────┘           │                  │");
        println!("│                ▼                        │                  │");
        println!("│    ┌────────────────────────┐           │                  │");
        println!("│    │ softmax(scores)        │           │                  │");
        println!(
            "│    │ = attn_weights         │ ◄─ {} │",
            "CRITICAL".red().bold()
        );
        println!("│    └───────────┬────────────┘   If uniform → Collapse     │");
        println!("│                │                        │                  │");
        println!("│                └────────────────────────┘                  │");
        println!("│                ▼                                           │");
        println!("│    ┌────────────────────────┐                              │");
        println!("│    │ attn_weights @ V       │                              │");
        println!("│    │ = context [dec, d]     │                              │");
        println!("│    └───────────┬────────────┘                              │");
        println!("│                ▼                                           │");
        println!("│    ┌──────────┐                                            │");
        println!(
            "│    │ {} │                                            │",
            "w_o".yellow()
        );
        println!("│    └────┬─────┘                                            │");
        println!("│         ▼                                                  │");
        println!("│    output [dec_len, d_model]                               │");
        println!("│                                                             │");
        println!("└─────────────────────────────────────────────────────────────┘");

        if verbose {
            // Show weight statistics
            for suffix in &[
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "out_proj.weight",
            ] {
                let tensor_name = format!("{layer_prefix}.{suffix}");
                if let Some(Ok(data)) = reader.map(|r| r.read_tensor_f32(&tensor_name)) {
                    let (min, max, mean, std) = compute_stats(&data);
                    println!(
                        "  {}: min={:.4} max={:.4} mean={:.4} std={:.4}",
                        suffix.dimmed(),
                        min,
                        max,
                        mean,
                        std
                    );
                }
            }
        }

        println!();
    }

    // Diagnostic hint
    println!("{}", "─".repeat(65));
    println!("{}", "DIAGNOSTIC HINT:".yellow().bold());
    println!("If attention weights are nearly uniform (max ≈ 1/seq_len),");
    println!("decoder is ignoring encoder output → Posterior Collapse");
    println!();
    println!("Check with: apr trace model.apr --layer cross_attn --payload");
}

/// Print self-attention flow
fn print_self_attention_flow(
    _reader: Option<&AprReader>,
    _tensor_names: &[String],
    _layer_filter: Option<&str>,
    _verbose: bool,
) {
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!("{}", "                   SELF-ATTENTION DATA FLOW".bold());
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!();

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!(
        "│  {}                                        │",
        "SELF-ATTENTION".cyan().bold()
    );
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                                                             │");
    println!("│   input [seq_len, d_model]                                  │");
    println!("│          │                                                  │");
    println!("│          ├───────────────┬───────────────┐                  │");
    println!("│          ▼               ▼               ▼                  │");
    println!("│    ┌──────────┐    ┌──────────┐    ┌──────────┐             │");
    println!(
        "│    │ {} │    │ {} │    │ {} │             │",
        "w_q".yellow(),
        "w_k".yellow(),
        "w_v".yellow()
    );
    println!("│    └────┬─────┘    └────┬─────┘    └────┬─────┘             │");
    println!("│         ▼               ▼               ▼                  │");
    println!("│    Q [seq, d]      K [seq, d]      V [seq, d]              │");
    println!("│         │               │               │                  │");
    println!("│         └───────┬───────┘               │                  │");
    println!("│                 ▼                       │                  │");
    println!("│    ┌────────────────────────┐           │                  │");
    println!("│    │ Q @ K^T / √d_k         │           │                  │");
    println!("│    │ + causal_mask          │ (decoder only)               │");
    println!("│    └───────────┬────────────┘           │                  │");
    println!("│                ▼                        │                  │");
    println!("│    ┌────────────────────────┐           │                  │");
    println!("│    │ softmax(scores)        │           │                  │");
    println!("│    └───────────┬────────────┘           │                  │");
    println!("│                │                        │                  │");
    println!("│                └────────────────────────┘                  │");
    println!("│                ▼                                           │");
    println!("│    ┌────────────────────────┐                              │");
    println!("│    │ attn_weights @ V       │                              │");
    println!("│    └───────────┬────────────┘                              │");
    println!("│                ▼                                           │");
    println!("│    ┌──────────┐                                            │");
    println!(
        "│    │ {} │                                            │",
        "w_o".yellow()
    );
    println!("│    └────┬─────┘                                            │");
    println!("│         ▼                                                  │");
    println!("│    output [seq_len, d_model]                               │");
    println!("│                                                             │");
    println!("└─────────────────────────────────────────────────────────────┘");
}

/// Print FFN flow
fn print_ffn_flow(
    _reader: Option<&AprReader>,
    _tensor_names: &[String],
    _layer_filter: Option<&str>,
    _verbose: bool,
) {
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!("{}", "                   FEED-FORWARD NETWORK FLOW".bold());
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!();

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!(
        "│  {}                                                  │",
        "FFN".yellow().bold()
    );
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│                                                             │");
    println!("│   input [seq_len, d_model]                                  │");
    println!("│          │                                                  │");
    println!("│          ▼                                                  │");
    println!("│    ┌──────────────────────────┐                             │");
    println!(
        "│    │ {} [d_model → d_ff]       │                             │",
        "fc1".yellow()
    );
    println!("│    └───────────┬──────────────┘                             │");
    println!("│                ▼                                           │");
    println!("│    ┌──────────────────────────┐                             │");
    println!(
        "│    │ {} (x * sigmoid(1.702*x)) │                             │",
        "GELU".cyan()
    );
    println!("│    └───────────┬──────────────┘                             │");
    println!("│                ▼                                           │");
    println!("│    hidden [seq_len, d_ff]                                   │");
    println!("│                │                                           │");
    println!("│                ▼                                           │");
    println!("│    ┌──────────────────────────┐                             │");
    println!(
        "│    │ {} [d_ff → d_model]       │                             │",
        "fc2".yellow()
    );
    println!("│    └───────────┬──────────────┘                             │");
    println!("│                ▼                                           │");
    println!("│    output [seq_len, d_model]                                │");
    println!("│                                                             │");
    println!("└─────────────────────────────────────────────────────────────┘");
}

/// Compute basic statistics
fn compute_stats(data: &[f32]) -> (f32, f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;

    for &x in data {
        min = min.min(x);
        max = max.max(x);
        sum += f64::from(x);
    }

    let mean = (sum / data.len() as f64) as f32;

    let variance: f32 = (data
        .iter()
        .map(|&x| {
            let diff = f64::from(x) - f64::from(mean);
            diff * diff
        })
        .sum::<f64>()
        / data.len() as f64) as f32;

    let std = variance.sqrt();

    (min, max, mean, std)
}
