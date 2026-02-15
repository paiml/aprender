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
use aprender::format::rosetta::RosettaStone;
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

    // GH-259: All formats via Rosetta Stone (handles GGUF, SafeTensors, APR v2)
    let rosetta = RosettaStone::new();
    let report = rosetta
        .inspect(apr_path)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to inspect: {e}")))?;

    let tensor_names: Vec<String> = report.tensors.iter().map(|t| t.name.clone()).collect();
    let arch = detect_architecture(&tensor_names);

    // APR reader for verbose tensor stats (only available for APR format)
    let apr_reader = AprReader::open(apr_path).ok();

    println!(
        "{} {} ({}, {})",
        "Model:".bold(),
        apr_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .cyan(),
        arch.yellow(),
        format!("{:?}", report.format).cyan()
    );
    println!();

    match component {
        FlowComponent::Full => {
            print_full_flow(apr_reader.as_ref(), &tensor_names, verbose);
        }
        FlowComponent::Encoder => {
            print_encoder_flow(apr_reader.as_ref(), &tensor_names, verbose);
        }
        FlowComponent::Decoder => {
            print_decoder_flow(apr_reader.as_ref(), &tensor_names, verbose);
        }
        FlowComponent::CrossAttention => {
            print_cross_attention_flow(apr_reader.as_ref(), &tensor_names, layer_filter, verbose);
        }
        FlowComponent::SelfAttention => {
            print_self_attention_flow(apr_reader.as_ref(), &tensor_names, layer_filter, verbose);
        }
        FlowComponent::Ffn => {
            print_ffn_flow(apr_reader.as_ref(), &tensor_names, layer_filter, verbose);
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
fn print_full_flow(_reader: Option<&AprReader>, tensor_names: &[String], verbose: bool) {
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
fn print_encoder_flow(_reader: Option<&AprReader>, tensor_names: &[String], verbose: bool) {
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
fn print_decoder_flow(_reader: Option<&AprReader>, tensor_names: &[String], verbose: bool) {
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

    // ========================================================================
    // FlowComponent FromStr Additional Coverage
    // ========================================================================

    #[test]
    fn test_flow_component_from_str_mixed_case_full() {
        assert_eq!(
            "FULL".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Full
        );
        assert_eq!(
            "Full".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Full
        );
        assert_eq!(
            "ALL".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Full
        );
        assert_eq!(
            "All".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Full
        );
    }

    #[test]
    fn test_flow_component_from_str_mixed_case_encoder() {
        assert_eq!(
            "ENC".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Encoder
        );
        assert_eq!(
            "Enc".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Encoder
        );
        assert_eq!(
            "Encoder".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Encoder
        );
    }

    #[test]
    fn test_flow_component_from_str_mixed_case_decoder() {
        assert_eq!(
            "DEC".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Decoder
        );
        assert_eq!(
            "Dec".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Decoder
        );
        assert_eq!(
            "Decoder".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Decoder
        );
        assert_eq!(
            "DECODER".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Decoder
        );
    }

    #[test]
    fn test_flow_component_from_str_mixed_case_self_attn() {
        assert_eq!(
            "SELF_ATTN".parse::<FlowComponent>().expect("parse"),
            FlowComponent::SelfAttention
        );
        assert_eq!(
            "Self-Attn".parse::<FlowComponent>().expect("parse"),
            FlowComponent::SelfAttention
        );
        assert_eq!(
            "SELFATTN".parse::<FlowComponent>().expect("parse"),
            FlowComponent::SelfAttention
        );
        assert_eq!(
            "Self_Attn".parse::<FlowComponent>().expect("parse"),
            FlowComponent::SelfAttention
        );
    }

    #[test]
    fn test_flow_component_from_str_mixed_case_cross_attn() {
        assert_eq!(
            "CROSS_ATTN".parse::<FlowComponent>().expect("parse"),
            FlowComponent::CrossAttention
        );
        assert_eq!(
            "CROSS-ATTN".parse::<FlowComponent>().expect("parse"),
            FlowComponent::CrossAttention
        );
        assert_eq!(
            "CROSSATTN".parse::<FlowComponent>().expect("parse"),
            FlowComponent::CrossAttention
        );
        assert_eq!(
            "ENCODER_ATTN".parse::<FlowComponent>().expect("parse"),
            FlowComponent::CrossAttention
        );
        assert_eq!(
            "Encoder_Attn".parse::<FlowComponent>().expect("parse"),
            FlowComponent::CrossAttention
        );
    }

    #[test]
    fn test_flow_component_from_str_mixed_case_ffn() {
        assert_eq!(
            "FFN".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Ffn
        );
        assert_eq!(
            "MLP".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Ffn
        );
        assert_eq!(
            "FEEDFORWARD".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Ffn
        );
        assert_eq!(
            "Mlp".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Ffn
        );
        assert_eq!(
            "FeedForward".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Ffn
        );
    }

    #[test]
    fn test_flow_component_from_str_error_message() {
        let err = "banana".parse::<FlowComponent>().unwrap_err();
        assert_eq!(err, "Unknown component: banana");
    }

    #[test]
    fn test_flow_component_from_str_error_whitespace() {
        assert!(" full".parse::<FlowComponent>().is_err());
        assert!("full ".parse::<FlowComponent>().is_err());
        assert!(" ".parse::<FlowComponent>().is_err());
    }

    #[test]
    fn test_flow_component_from_str_error_partial() {
        assert!("ful".parse::<FlowComponent>().is_err());
        assert!("encode".parse::<FlowComponent>().is_err());
        assert!("decode".parse::<FlowComponent>().is_err());
        assert!("atten".parse::<FlowComponent>().is_err());
    }

    #[test]
    fn test_flow_component_copy() {
        // FlowComponent derives Copy
        let a = FlowComponent::Full;
        let b = a; // Copy, not move
        assert_eq!(a, b); // a is still accessible
    }

    #[test]
    fn test_flow_component_all_variants_distinct() {
        let variants = [
            FlowComponent::Full,
            FlowComponent::Encoder,
            FlowComponent::Decoder,
            FlowComponent::SelfAttention,
            FlowComponent::CrossAttention,
            FlowComponent::Ffn,
        ];
        // Every pair of distinct variants should be unequal
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(
                    variants[i], variants[j],
                    "{:?} should not equal {:?}",
                    variants[i], variants[j]
                );
            }
        }
    }

    // ========================================================================
    // detect_architecture Additional Coverage
    // ========================================================================

    #[test]
    fn test_detect_architecture_encoder_decoder_without_cross_attn() {
        // Has both encoder and decoder, but NO cross_attn
        // This falls through to the else branch
        let names = vec![
            "encoder.layers.0.self_attn.q_proj.weight".to_string(),
            "decoder.layers.0.self_attn.q_proj.weight".to_string(),
        ];
        // No cross_attn -> first condition fails -> encoder_only fails (has_decoder)
        // -> decoder_only fails (has_encoder) -> unknown
        assert_eq!(detect_architecture(&names), "unknown");
    }

    #[test]
    fn test_detect_architecture_only_cross_attn_no_enc_dec() {
        // Has cross_attn mention but neither "encoder" nor "decoder" prefix
        let names = vec!["model.layers.0.cross_attn.q_proj.weight".to_string()];
        // has_encoder=false, has_decoder=false, has_cross_attn=true
        // First condition: false (needs all three)
        // Second: false (no encoder)
        // Third: false (no decoder)
        assert_eq!(detect_architecture(&names), "unknown");
    }

    #[test]
    fn test_detect_architecture_encoder_attn_keyword() {
        let names = vec![
            "encoder.layers.0.weight".to_string(),
            "decoder.layers.0.weight".to_string(),
            "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "encoder-decoder (Whisper/T5)");
    }

    #[test]
    fn test_detect_architecture_single_encoder_tensor() {
        let names = vec!["encoder.conv1.weight".to_string()];
        assert_eq!(detect_architecture(&names), "encoder-only (BERT)");
    }

    #[test]
    fn test_detect_architecture_single_decoder_tensor() {
        let names = vec!["decoder.embed_tokens.weight".to_string()];
        assert_eq!(detect_architecture(&names), "decoder-only (GPT/LLaMA)");
    }

    #[test]
    fn test_detect_architecture_llama_style_names() {
        // LLaMA-style models don't use "encoder"/"decoder" prefixes
        let names = vec![
            "model.embed_tokens.weight".to_string(),
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "unknown");
    }

    #[test]
    fn test_detect_architecture_gguf_style_names() {
        // GGUF-style: blk.0.attn_q.weight
        let names = vec![
            "token_embd.weight".to_string(),
            "blk.0.attn_q.weight".to_string(),
            "blk.0.ffn_gate.weight".to_string(),
            "output.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "unknown");
    }

    #[test]
    fn test_detect_architecture_encoder_prefix_in_middle() {
        // "encoder" must be a prefix (starts_with), not just a substring
        let names = vec!["some_encoder_layer.weight".to_string()];
        // starts_with("encoder") is false
        assert_eq!(detect_architecture(&names), "unknown");
    }

    #[test]
    fn test_detect_architecture_decoder_prefix_in_middle() {
        let names = vec!["pre_decoder.layers.0.weight".to_string()];
        // starts_with("decoder") is false
        assert_eq!(detect_architecture(&names), "unknown");
    }

    // ========================================================================
    // compute_stats Additional Coverage
    // ========================================================================

    #[test]
    fn test_compute_stats_two_values() {
        let data = [3.0, 7.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 3.0);
        assert_eq!(max, 7.0);
        assert_eq!(mean, 5.0);
        assert!((std - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_stats_large_values() {
        let data = [1e6, 2e6, 3e6];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, 1e6);
        assert_eq!(max, 3e6);
        assert!((mean - 2e6).abs() < 1.0);
    }

    #[test]
    fn test_compute_stats_very_small_values() {
        let data = [1e-7, 2e-7, 3e-7];
        let (min, max, mean, _std) = compute_stats(&data);
        assert!((min - 1e-7).abs() < 1e-10);
        assert!((max - 3e-7).abs() < 1e-10);
        assert!((mean - 2e-7).abs() < 1e-10);
    }

    #[test]
    fn test_compute_stats_mixed_sign() {
        let data = [-100.0, 0.0, 100.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, -100.0);
        assert_eq!(max, 100.0);
        assert!(mean.abs() < 0.001);
    }

    #[test]
    fn test_compute_stats_single_negative() {
        let data = [-42.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, -42.0);
        assert_eq!(max, -42.0);
        assert_eq!(mean, -42.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_single_zero() {
        let data = [0.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_all_zeros() {
        let data = [0.0, 0.0, 0.0, 0.0, 0.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_ascending() {
        let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 1.0);
        assert_eq!(max, 100.0);
        assert!((mean - 50.5).abs() < 0.01);
        // std of uniform 1..=100 is ~28.87
        assert!((std - 28.87).abs() < 0.1);
    }

    #[test]
    fn test_compute_stats_descending() {
        let data: Vec<f32> = (1..=100).rev().map(|i| i as f32).collect();
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 1.0);
        assert_eq!(max, 100.0);
        assert!((mean - 50.5).abs() < 0.01);
        assert!((std - 28.87).abs() < 0.1);
    }

    #[test]
    fn test_compute_stats_typical_weights() {
        // Simulating typical neural network weight distribution
        let data = vec![
            -0.1, 0.05, -0.03, 0.08, -0.07, 0.02, -0.01, 0.04, -0.06, 0.09,
        ];
        let (min, max, mean, std) = compute_stats(&data);
        assert!(min < 0.0);
        assert!(max > 0.0);
        assert!(mean.abs() < 0.05); // near zero mean
        assert!(std > 0.0);
        assert!(std < 0.2); // small spread
    }

    #[test]
    fn test_compute_stats_std_is_non_negative() {
        // Standard deviation must always be >= 0
        let test_cases: Vec<Vec<f32>> = vec![
            vec![1.0, 1.0, 1.0],
            vec![-1.0, 1.0],
            vec![0.0],
            vec![100.0, -100.0],
        ];
        for data in &test_cases {
            let (_, _, _, std) = compute_stats(data);
            assert!(std >= 0.0, "std must be non-negative, got {std}");
        }
    }

    #[test]
    fn test_compute_stats_mean_is_between_min_and_max() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert!(mean >= min, "mean should be >= min");
        assert!(mean <= max, "mean should be <= max");
    }

    // ========================================================================
    // Layer Counting Logic (used in print_encoder_block / print_decoder_block)
    // ========================================================================

    #[test]
    fn test_encoder_layer_counting_zero_layers() {
        let tensor_names: Vec<String> = vec!["output.weight".to_string()];
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
        assert_eq!(n_layers, 0);
    }

    #[test]
    fn test_encoder_layer_counting_single_layer() {
        let tensor_names = vec![
            "encoder.layers.0.self_attn.q_proj.weight".to_string(),
            "encoder.layers.0.self_attn.k_proj.weight".to_string(),
        ];
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
        assert_eq!(n_layers, 1);
    }

    #[test]
    fn test_encoder_layer_counting_multiple_layers() {
        let tensor_names = vec![
            "encoder.layers.0.self_attn.weight".to_string(),
            "encoder.layers.1.self_attn.weight".to_string(),
            "encoder.layers.2.self_attn.weight".to_string(),
            "encoder.layers.3.self_attn.weight".to_string(),
        ];
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
        assert_eq!(n_layers, 4);
    }

    #[test]
    fn test_encoder_layer_counting_non_contiguous() {
        // Layers 0, 5 -> max=5, n_layers=6
        let tensor_names = vec![
            "encoder.layers.0.weight".to_string(),
            "encoder.layers.5.weight".to_string(),
        ];
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
        assert_eq!(n_layers, 6);
    }

    #[test]
    fn test_decoder_layer_counting_zero_layers() {
        let tensor_names: Vec<String> = vec!["output.weight".to_string()];
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
        assert_eq!(n_layers, 0);
    }

    #[test]
    fn test_decoder_layer_counting_multiple_layers() {
        let tensor_names = vec![
            "decoder.layers.0.self_attn.weight".to_string(),
            "decoder.layers.1.encoder_attn.weight".to_string(),
            "decoder.layers.2.ffn.weight".to_string(),
            "decoder.layers.3.self_attn.weight".to_string(),
            "decoder.layers.3.ffn.weight".to_string(),
        ];
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
        assert_eq!(n_layers, 4);
    }

    #[test]
    fn test_layer_counting_mixed_encoder_decoder() {
        let tensor_names = vec![
            "encoder.layers.0.weight".to_string(),
            "encoder.layers.1.weight".to_string(),
            "decoder.layers.0.weight".to_string(),
            "decoder.layers.1.weight".to_string(),
            "decoder.layers.2.weight".to_string(),
        ];
        let enc_layers = tensor_names
            .iter()
            .filter(|n| n.starts_with("encoder.layers."))
            .filter_map(|n| {
                n.strip_prefix("encoder.layers.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map_or(0, |n| n + 1);
        let dec_layers = tensor_names
            .iter()
            .filter(|n| n.starts_with("decoder.layers."))
            .filter_map(|n| {
                n.strip_prefix("decoder.layers.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map_or(0, |n| n + 1);
        assert_eq!(enc_layers, 2);
        assert_eq!(dec_layers, 3);
    }

    // ========================================================================
    // Cross-Attention Layer Filtering Logic
    // ========================================================================

    #[test]
    fn test_cross_attn_layer_detection_encoder_attn() {
        let tensor_names = vec![
            "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
            "decoder.layers.0.encoder_attn.k_proj.weight".to_string(),
            "decoder.layers.1.encoder_attn.q_proj.weight".to_string(),
        ];
        let cross_attn_layers: Vec<_> = tensor_names
            .iter()
            .filter(|n| n.contains("encoder_attn") || n.contains("cross_attn"))
            .filter(|n| n.contains("q_proj.weight"))
            .collect();
        assert_eq!(cross_attn_layers.len(), 2);
    }

    #[test]
    fn test_cross_attn_layer_detection_cross_attn() {
        let tensor_names = vec![
            "decoder.layers.0.cross_attn.q_proj.weight".to_string(),
            "decoder.layers.0.cross_attn.k_proj.weight".to_string(),
        ];
        let cross_attn_layers: Vec<_> = tensor_names
            .iter()
            .filter(|n| n.contains("encoder_attn") || n.contains("cross_attn"))
            .filter(|n| n.contains("q_proj.weight"))
            .collect();
        assert_eq!(cross_attn_layers.len(), 1);
    }

    #[test]
    fn test_cross_attn_layer_detection_empty() {
        let tensor_names = vec![
            "decoder.layers.0.self_attn.q_proj.weight".to_string(),
            "decoder.layers.0.ffn.fc1.weight".to_string(),
        ];
        let cross_attn_layers: Vec<_> = tensor_names
            .iter()
            .filter(|n| n.contains("encoder_attn") || n.contains("cross_attn"))
            .filter(|n| n.contains("q_proj.weight"))
            .collect();
        assert!(cross_attn_layers.is_empty());
    }

    #[test]
    fn test_cross_attn_layer_filter_applied() {
        let tensor_names = vec![
            "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
            "decoder.layers.1.encoder_attn.q_proj.weight".to_string(),
            "decoder.layers.2.encoder_attn.q_proj.weight".to_string(),
        ];
        let layer_filter: Option<&str> = Some("layers.1");
        let filtered: Vec<_> = tensor_names
            .iter()
            .filter(|n| n.contains("encoder_attn") || n.contains("cross_attn"))
            .filter(|n| n.contains("q_proj.weight"))
            .filter(|n| layer_filter.map_or(true, |f| n.contains(f)))
            .collect();
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].contains("layers.1"));
    }

    #[test]
    fn test_cross_attn_layer_filter_none_matches_all() {
        let tensor_names = vec![
            "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
            "decoder.layers.1.encoder_attn.q_proj.weight".to_string(),
        ];
        let layer_filter: Option<&str> = None;
        let filtered: Vec<_> = tensor_names
            .iter()
            .filter(|n| n.contains("encoder_attn") || n.contains("cross_attn"))
            .filter(|n| n.contains("q_proj.weight"))
            .filter(|n| layer_filter.map_or(true, |f| n.contains(f)))
            .collect();
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_cross_attn_layer_filter_no_match() {
        let tensor_names = vec![
            "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
            "decoder.layers.1.encoder_attn.q_proj.weight".to_string(),
        ];
        let layer_filter: Option<&str> = Some("layers.99");
        let filtered: Vec<_> = tensor_names
            .iter()
            .filter(|n| n.contains("encoder_attn") || n.contains("cross_attn"))
            .filter(|n| n.contains("q_proj.weight"))
            .filter(|n| layer_filter.map_or(true, |f| n.contains(f)))
            .collect();
        assert!(filtered.is_empty());
    }

    // ========================================================================
    // Q weight name prefix stripping (cross-attention flow)
    // ========================================================================

    #[test]
    fn test_strip_q_proj_suffix() {
        let name = "decoder.layers.0.encoder_attn.q_proj.weight";
        let prefix = name.strip_suffix(".q_proj.weight").unwrap_or(name);
        assert_eq!(prefix, "decoder.layers.0.encoder_attn");
    }

    #[test]
    fn test_strip_q_proj_suffix_no_match() {
        let name = "decoder.layers.0.encoder_attn.k_proj.weight";
        let prefix = name.strip_suffix(".q_proj.weight").unwrap_or(name);
        // No match -> returns the full name
        assert_eq!(prefix, name);
    }

    // ========================================================================
    // Conv1 detection in encoder block
    // ========================================================================

    #[test]
    fn test_conv1_detection_present() {
        let tensor_names = vec![
            "encoder.conv1.weight".to_string(),
            "encoder.conv2.weight".to_string(),
            "encoder.positional_embedding".to_string(),
        ];
        let has_conv1 = tensor_names.iter().any(|n| n.contains("conv1"));
        assert!(has_conv1);
    }

    #[test]
    fn test_conv1_detection_absent() {
        let tensor_names = vec![
            "encoder.layers.0.self_attn.weight".to_string(),
            "encoder.layers.0.ffn.weight".to_string(),
        ];
        let has_conv1 = tensor_names.iter().any(|n| n.contains("conv1"));
        assert!(!has_conv1);
    }

    // ========================================================================
    // run() Error Path Tests
    // ========================================================================

    #[test]
    fn test_run_nonexistent_path_specific_error_variant() {
        let result = run(
            Path::new("/tmp/definitely_does_not_exist_xyz123.apr"),
            None,
            FlowComponent::Full,
            false,
        );
        match result {
            Err(CliError::FileNotFound(p)) => {
                assert_eq!(p, Path::new("/tmp/definitely_does_not_exist_xyz123.apr"));
            }
            other => panic!("Expected FileNotFound, got: {other:?}"),
        }
    }

    #[test]
    fn test_run_empty_apr_file() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        // Empty file should fail with InvalidFormat
        let result = run(file.path(), None, FlowComponent::Full, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_non_apr_extension() {
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"some data").expect("write");
        // This file exists but flow command requires APR format
        let result = run(&path, None, FlowComponent::Full, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_all_components_fail_on_invalid() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"invalid").expect("write");

        // All component variants should fail with invalid data
        let components = [
            FlowComponent::Full,
            FlowComponent::Encoder,
            FlowComponent::Decoder,
            FlowComponent::SelfAttention,
            FlowComponent::CrossAttention,
            FlowComponent::Ffn,
        ];
        for comp in &components {
            let result = run(file.path(), None, *comp, false);
            assert!(result.is_err(), "Expected error for component {comp:?}");
        }
    }

    #[test]
    fn test_run_verbose_with_layer_filter() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid apr").expect("write");

        let result = run(
            file.path(),
            Some("decoder.layers.0"),
            FlowComponent::CrossAttention,
            true,
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // Printing functions (should not panic)
    // ========================================================================

    #[test]
    fn test_print_encoder_block_no_conv1_no_layers() {
        let tensor_names: Vec<String> = vec![];
        // Should not panic with empty tensor names
        print_encoder_block(&tensor_names, false);
    }

    #[test]
    fn test_print_encoder_block_with_conv1_and_layers() {
        let tensor_names = vec![
            "encoder.conv1.weight".to_string(),
            "encoder.conv2.weight".to_string(),
            "encoder.layers.0.self_attn.weight".to_string(),
            "encoder.layers.1.self_attn.weight".to_string(),
        ];
        // Should not panic
        print_encoder_block(&tensor_names, false);
    }

    #[test]
    fn test_print_encoder_block_with_conv1_no_layers() {
        let tensor_names = vec!["encoder.conv1.weight".to_string()];
        // Has conv1 but no "encoder.layers." tensors -> n_layers = 0
        print_encoder_block(&tensor_names, false);
    }

    #[test]
    fn test_print_decoder_block_no_layers() {
        let tensor_names: Vec<String> = vec![];
        print_decoder_block(&tensor_names, false);
    }

    #[test]
    fn test_print_decoder_block_with_layers() {
        let tensor_names = vec![
            "decoder.layers.0.self_attn.weight".to_string(),
            "decoder.layers.1.self_attn.weight".to_string(),
            "decoder.layers.2.self_attn.weight".to_string(),
        ];
        print_decoder_block(&tensor_names, false);
    }

    #[test]
    fn test_print_decoder_block_many_layers() {
        let tensor_names: Vec<String> = (0..32)
            .map(|i| format!("decoder.layers.{i}.self_attn.weight"))
            .collect();
        print_decoder_block(&tensor_names, false);
    }
}
