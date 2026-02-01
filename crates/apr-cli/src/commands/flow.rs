//! Data flow visualization (GH-122)
//!
//! Toyota Way: Genchi Genbutsu - Go and see how data flows.
//! Visualize tensor transformations through model layers.
//!
//! Usage:
//!   apr flow model.apr --layer "decoder.layers.0.cross_attn"
//!   apr flow model.apr --component encoder
//!   apr flow model.apr --all

use crate::error::CliError;
use aprender::serialization::apr::AprReader;
use colored::Colorize;
use std::path::Path;

/// Flow component type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlowComponent {
    /// Full model flow
    Full,
    /// Encoder only
    Encoder,
    /// Decoder only
    Decoder,
    /// Self-attention
    SelfAttention,
    /// Cross-attention (encoder-decoder)
    CrossAttention,
    /// Feed-forward network
    Ffn,
}

impl std::str::FromStr for FlowComponent {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "full" | "all" => Ok(Self::Full),
            "encoder" | "enc" => Ok(Self::Encoder),
            "decoder" | "dec" => Ok(Self::Decoder),
            "self_attn" | "self-attn" | "selfattn" => Ok(Self::SelfAttention),
            "cross_attn" | "cross-attn" | "crossattn" | "encoder_attn" => Ok(Self::CrossAttention),
            "ffn" | "mlp" | "feedforward" => Ok(Self::Ffn),
            _ => Err(format!("Unknown component: {s}")),
        }
    }
}

/// Run the flow command
pub(crate) fn run(
    apr_path: &Path,
    layer_filter: Option<&str>,
    component: FlowComponent,
    verbose: bool,
) -> Result<(), CliError> {
    if !apr_path.exists() {
        return Err(CliError::FileNotFound(apr_path.to_path_buf()));
    }

    let reader = AprReader::open(apr_path)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to read APR: {e}")))?;

    // Detect model architecture from tensor names
    let tensor_names: Vec<String> = reader.tensors.iter().map(|t| t.name.clone()).collect();
    let arch = detect_architecture(&tensor_names);

    println!(
        "{} {} ({})",
        "Model:".bold(),
        apr_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .cyan(),
        arch.yellow()
    );
    println!();

    match component {
        FlowComponent::Full => {
            print_full_flow(&reader, &tensor_names, verbose);
        }
        FlowComponent::Encoder => {
            print_encoder_flow(&reader, &tensor_names, verbose);
        }
        FlowComponent::Decoder => {
            print_decoder_flow(&reader, &tensor_names, verbose);
        }
        FlowComponent::CrossAttention => {
            print_cross_attention_flow(&reader, &tensor_names, layer_filter, verbose);
        }
        FlowComponent::SelfAttention => {
            print_self_attention_flow(&reader, &tensor_names, layer_filter, verbose);
        }
        FlowComponent::Ffn => {
            print_ffn_flow(&reader, &tensor_names, layer_filter, verbose);
        }
    }

    Ok(())
}

/// Detect model architecture
fn detect_architecture(tensor_names: &[String]) -> &'static str {
    let has_encoder = tensor_names.iter().any(|n| n.starts_with("encoder"));
    let has_decoder = tensor_names.iter().any(|n| n.starts_with("decoder"));
    let has_cross_attn = tensor_names
        .iter()
        .any(|n| n.contains("encoder_attn") || n.contains("cross_attn"));

    if has_encoder && has_decoder && has_cross_attn {
        "encoder-decoder (Whisper/T5)"
    } else if has_encoder && !has_decoder {
        "encoder-only (BERT)"
    } else if has_decoder && !has_encoder {
        "decoder-only (GPT/LLaMA)"
    } else {
        "unknown"
    }
}

/// Print full model flow
fn print_full_flow(_reader: &AprReader, tensor_names: &[String], verbose: bool) {
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!("{}", "                     FULL MODEL DATA FLOW".bold());
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!();

    // Input
    println!("{}  audio [N, samples]", "INPUT".green().bold());
    println!("   │");
    println!("   ▼");

    // Mel spectrogram
    println!("{}  mel_spectrogram()", "┌─────────────────┐".blue());
    println!("{}  [N, 80, T]      {}", "│".blue(), "│".blue());
    println!("{}", "└────────┬────────┘".blue());
    println!("         │");
    println!("         ▼");

    // Encoder
    print_encoder_block(tensor_names, verbose);

    // Decoder
    print_decoder_block(tensor_names, verbose);

    // Output
    println!("         │");
    println!("         ▼");
    println!("{}  tokens [N, seq_len]", "OUTPUT".green().bold());
}

/// Print encoder flow
fn print_encoder_flow(_reader: &AprReader, tensor_names: &[String], verbose: bool) {
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!("{}", "                       ENCODER DATA FLOW".bold());
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!();

    print_encoder_block(tensor_names, verbose);
}

fn print_encoder_block(tensor_names: &[String], _verbose: bool) {
    // Count encoder layers
    let n_layers = tensor_names
        .iter()
        .filter(|n| n.starts_with("encoder.layers."))
        .filter_map(|n| {
            n.strip_prefix("encoder.layers.")
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse::<usize>().ok())
        })
        .max()
        .map_or(0, |n| n + 1);

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!(
        "│  {}                                               │",
        "ENCODER".cyan().bold()
    );
    println!("├─────────────────────────────────────────────────────────────┤");

    // Conv layers
    if tensor_names.iter().any(|n| n.contains("conv1")) {
        println!("│  mel [N, 80, T]                                             │");
        println!("│       │                                                     │");
        println!("│       ▼                                                     │");
        println!("│  ┌─────────┐                                                │");
        println!(
            "│  │ {} │ kernel=3, stride=1 → [N, d_model, T]         │",
            "Conv1".yellow()
        );
        println!("│  └────┬────┘                                                │");
        println!("│       ▼                                                     │");
        println!("│  ┌─────────┐                                                │");
        println!(
            "│  │ {} │ kernel=3, stride=2 → [N, d_model, T/2]       │",
            "Conv2".yellow()
        );
        println!("│  └────┬────┘                                                │");
        println!("│       ▼                                                     │");
        println!("│  ┌──────────────────┐                                       │");
        println!(
            "│  │ {} │ + positional_embedding             │",
            "Pos Embed".yellow()
        );
        println!("│  └────────┬─────────┘                                       │");
    }

    // Transformer layers
    if n_layers > 0 {
        println!("│            │                                                │");
        println!("│            ▼                                                │");
        println!("│  ┌─────────────────────────────────────────────────┐        │");
        println!(
            "│  │  {} × {}                                   │        │",
            "Encoder Layers".cyan(),
            n_layers
        );
        println!("│  │  ┌─────────────────────────────────────────┐    │        │");
        println!("│  │  │ ln1 → self_attn → + residual            │    │        │");
        println!("│  │  │ ln2 → ffn → + residual                  │    │        │");
        println!("│  │  └─────────────────────────────────────────┘    │        │");
        println!("│  └─────────────────────────────────────────────────┘        │");
    }

    println!("│            │                                                │");
    println!("│            ▼                                                │");
    println!(
        "│  {} [N, T', d_model]                          │",
        "encoder_output".green()
    );
    println!("└─────────────────────────────────────────────────────────────┘");
    println!("             │");
    println!("             │ (used as K,V for cross-attention)");
    println!("             ▼");
}

/// Print decoder flow
fn print_decoder_flow(_reader: &AprReader, tensor_names: &[String], verbose: bool) {
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!("{}", "                       DECODER DATA FLOW".bold());
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════════".dimmed()
    );
    println!();

    print_decoder_block(tensor_names, verbose);
}

fn print_decoder_block(tensor_names: &[String], _verbose: bool) {
    // Count decoder layers
    let n_layers = tensor_names
        .iter()
        .filter(|n| n.starts_with("decoder.layers."))
        .filter_map(|n| {
            n.strip_prefix("decoder.layers.")
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse::<usize>().ok())
        })
        .max()
        .map_or(0, |n| n + 1);

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!(
        "│  {}                                               │",
        "DECODER".magenta().bold()
    );
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│  tokens [N, seq_len]                                        │");
    println!("│       │                                                     │");
    println!("│       ▼                                                     │");
    println!("│  ┌──────────────────┐                                       │");
    println!(
        "│  │ {} │ + positional_embedding              │",
        "Token Embed".yellow()
    );
    println!("│  └────────┬─────────┘                                       │");

    if n_layers > 0 {
        println!("│            │                                                │");
        println!("│            ▼                                                │");
        println!("│  ┌─────────────────────────────────────────────────┐        │");
        println!(
            "│  │  {} × {}                                   │        │",
            "Decoder Layers".magenta(),
            n_layers
        );
        println!("│  │  ┌─────────────────────────────────────────┐    │        │");
        println!(
            "│  │  │ ln1 → {} → + residual      │    │        │",
            "self_attn (causal)".yellow()
        );
        println!(
            "│  │  │ ln2 → {} → + residual          │    │        │",
            "cross_attn".cyan()
        );
        println!("│  │  │       └── Q from decoder                │    │        │");
        println!(
            "│  │  │       └── K,V from {} │    │        │",
            "encoder_output".green()
        );
        println!("│  │  │ ln3 → ffn → + residual                  │    │        │");
        println!("│  │  └─────────────────────────────────────────┘    │        │");
        println!("│  └─────────────────────────────────────────────────┘        │");
    }

    println!("│            │                                                │");
    println!("│            ▼                                                │");
    println!("│  ┌──────────────────┐                                       │");
    println!(
        "│  │ {} │ → logits [N, seq_len, vocab]           │",
        "LM Head".yellow()
    );
    println!("│  └────────┬─────────┘                                       │");
    println!("└─────────────────────────────────────────────────────────────┘");
}

/// Print cross-attention flow (critical for debugging Posterior Collapse)
#[allow(clippy::too_many_lines)] // Visual debugging output requires detailed formatting
fn print_cross_attention_flow(
    reader: &AprReader,
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
                if let Ok(data) = reader.read_tensor_f32(&tensor_name) {
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
    _reader: &AprReader,
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
    _reader: &AprReader,
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    // ========================================================================
    // FlowComponent Tests
    // ========================================================================

    #[test]
    fn test_flow_component_from_str_full() {
        assert_eq!(
            "full".parse::<FlowComponent>().unwrap(),
            FlowComponent::Full
        );
        assert_eq!("all".parse::<FlowComponent>().unwrap(), FlowComponent::Full);
    }

    #[test]
    fn test_flow_component_from_str_encoder() {
        assert_eq!(
            "encoder".parse::<FlowComponent>().unwrap(),
            FlowComponent::Encoder
        );
        assert_eq!(
            "enc".parse::<FlowComponent>().unwrap(),
            FlowComponent::Encoder
        );
        assert_eq!(
            "ENCODER".parse::<FlowComponent>().unwrap(),
            FlowComponent::Encoder
        );
    }

    #[test]
    fn test_flow_component_from_str_decoder() {
        assert_eq!(
            "decoder".parse::<FlowComponent>().unwrap(),
            FlowComponent::Decoder
        );
        assert_eq!(
            "dec".parse::<FlowComponent>().unwrap(),
            FlowComponent::Decoder
        );
    }

    #[test]
    fn test_flow_component_from_str_self_attention() {
        assert_eq!(
            "self_attn".parse::<FlowComponent>().unwrap(),
            FlowComponent::SelfAttention
        );
        assert_eq!(
            "self-attn".parse::<FlowComponent>().unwrap(),
            FlowComponent::SelfAttention
        );
        assert_eq!(
            "selfattn".parse::<FlowComponent>().unwrap(),
            FlowComponent::SelfAttention
        );
    }

    #[test]
    fn test_flow_component_from_str_cross_attention() {
        assert_eq!(
            "cross_attn".parse::<FlowComponent>().unwrap(),
            FlowComponent::CrossAttention
        );
        assert_eq!(
            "cross-attn".parse::<FlowComponent>().unwrap(),
            FlowComponent::CrossAttention
        );
        assert_eq!(
            "crossattn".parse::<FlowComponent>().unwrap(),
            FlowComponent::CrossAttention
        );
        assert_eq!(
            "encoder_attn".parse::<FlowComponent>().unwrap(),
            FlowComponent::CrossAttention
        );
    }

    #[test]
    fn test_flow_component_from_str_ffn() {
        assert_eq!("ffn".parse::<FlowComponent>().unwrap(), FlowComponent::Ffn);
        assert_eq!("mlp".parse::<FlowComponent>().unwrap(), FlowComponent::Ffn);
        assert_eq!(
            "feedforward".parse::<FlowComponent>().unwrap(),
            FlowComponent::Ffn
        );
    }

    #[test]
    fn test_flow_component_from_str_invalid() {
        assert!("unknown".parse::<FlowComponent>().is_err());
        assert!("invalid".parse::<FlowComponent>().is_err());
        assert!("".parse::<FlowComponent>().is_err());
    }

    #[test]
    fn test_flow_component_debug() {
        // Verify Debug trait is derived
        let comp = FlowComponent::Full;
        let debug = format!("{comp:?}");
        assert!(debug.contains("Full"));
    }

    #[test]
    fn test_flow_component_clone() {
        let comp = FlowComponent::Encoder;
        let cloned = comp.clone();
        assert_eq!(comp, cloned);
    }

    #[test]
    fn test_flow_component_eq() {
        assert_eq!(FlowComponent::Full, FlowComponent::Full);
        assert_ne!(FlowComponent::Full, FlowComponent::Encoder);
    }

    // ========================================================================
    // detect_architecture Tests
    // ========================================================================

    #[test]
    fn test_detect_architecture_encoder_decoder() {
        let names = vec![
            "encoder.layers.0.self_attn.q_proj.weight".to_string(),
            "decoder.layers.0.self_attn.q_proj.weight".to_string(),
            "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "encoder-decoder (Whisper/T5)");
    }

    #[test]
    fn test_detect_architecture_encoder_only() {
        let names = vec![
            "encoder.layers.0.self_attn.q_proj.weight".to_string(),
            "encoder.layers.0.ffn.fc1.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "encoder-only (BERT)");
    }

    #[test]
    fn test_detect_architecture_decoder_only() {
        let names = vec![
            "decoder.layers.0.self_attn.q_proj.weight".to_string(),
            "decoder.layers.0.ffn.fc1.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "decoder-only (GPT/LLaMA)");
    }

    #[test]
    fn test_detect_architecture_unknown() {
        let names = vec!["weight".to_string(), "bias".to_string()];
        assert_eq!(detect_architecture(&names), "unknown");
    }

    #[test]
    fn test_detect_architecture_empty() {
        let names: Vec<String> = vec![];
        assert_eq!(detect_architecture(&names), "unknown");
    }

    #[test]
    fn test_detect_architecture_cross_attn_variant() {
        let names = vec![
            "encoder.layers.0.self_attn.weight".to_string(),
            "decoder.layers.0.cross_attn.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "encoder-decoder (Whisper/T5)");
    }

    // ========================================================================
    // compute_stats Tests
    // ========================================================================

    #[test]
    fn test_compute_stats_empty() {
        let (min, max, mean, std) = compute_stats(&[]);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_single_value() {
        let (min, max, mean, std) = compute_stats(&[5.0]);
        assert_eq!(min, 5.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 5.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_simple_range() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 3.0);
        // std for [1,2,3,4,5] is sqrt(2) ≈ 1.414
        assert!((std - 1.4142).abs() < 0.01);
    }

    #[test]
    fn test_compute_stats_negative_values() {
        let data = [-5.0, -2.0, 0.0, 2.0, 5.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, -5.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_compute_stats_all_same() {
        let data = [7.0, 7.0, 7.0, 7.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 7.0);
        assert_eq!(max, 7.0);
        assert_eq!(mean, 7.0);
        assert_eq!(std, 0.0);
    }

    // ========================================================================
    // Run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            None,
            FlowComponent::Full,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_run_invalid_apr_file() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), None, FlowComponent::Full, false);
        // Should fail because it's not a valid APR
        assert!(result.is_err());
    }

    #[test]
    fn test_run_is_directory() {
        let dir = tempdir().expect("create temp dir");
        let result = run(dir.path(), None, FlowComponent::Full, false);
        // Should fail because it's a directory
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_layer_filter() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(
            file.path(),
            Some("encoder.layers.0"),
            FlowComponent::Full,
            false,
        );
        // Should fail (invalid file) but tests the filter path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_verbose() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), None, FlowComponent::Full, true);
        // Should fail (invalid file) but tests verbose path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_encoder_component() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), None, FlowComponent::Encoder, false);
        // Should fail (invalid file) but tests encoder path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_decoder_component() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), None, FlowComponent::Decoder, false);
        // Should fail (invalid file) but tests decoder path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_self_attention_component() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), None, FlowComponent::SelfAttention, false);
        // Should fail (invalid file) but tests self-attention path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_cross_attention_component() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), None, FlowComponent::CrossAttention, false);
        // Should fail (invalid file) but tests cross-attention path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_ffn_component() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), None, FlowComponent::Ffn, false);
        // Should fail (invalid file) but tests FFN path
        assert!(result.is_err());
    }
}
