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

include!("flow_part_02.rs");
include!("flow_part_03.rs");
