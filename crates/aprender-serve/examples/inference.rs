//! End-to-end inference example for realizar
//!
//! Demonstrates the complete text generation pipeline:
//! - Model initialization with configuration
//! - Forward pass through transformer blocks
//! - Text generation with various sampling strategies

use comfy_table::{presets::UTF8_FULL, Cell, Color, Table};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use realizar::{
    generate::GenerationConfig,
    layers::{Model, ModelConfig},
};
use std::time::Instant;

fn main() {
    // Print styled header
    println!();
    println!(
        "{}",
        style("╔════════════════════════════════════════╗")
            .cyan()
            .bold()
    );
    println!(
        "{}",
        style("║     Realizar Inference Example         ║")
            .cyan()
            .bold()
    );
    println!(
        "{}",
        style("╚════════════════════════════════════════╝")
            .cyan()
            .bold()
    );
    println!();

    // Define a small model configuration for demonstration
    let config = ModelConfig {
        vocab_size: 100,      // Small vocabulary for demo
        hidden_dim: 32,       // Reduced dimensions for speed
        num_heads: 1,         // Single attention head
        num_layers: 2,        // Two transformer blocks
        intermediate_dim: 64, // FFN intermediate size
        eps: 1e-5,            // Layer normalization epsilon
    };

    // Create model with progress indicator
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message("Initializing model...");
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let model = Model::new(config.clone()).expect("Failed to create model");
    pb.finish_with_message(format!("{} Model initialized!", style("✓").green().bold()));

    // Display model configuration as a table
    println!();
    println!("{}", style("Model Configuration").yellow().bold());
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec![
        Cell::new("Parameter").fg(Color::Cyan),
        Cell::new("Value").fg(Color::Cyan),
    ]);
    table.add_row(vec!["Vocabulary Size", &config.vocab_size.to_string()]);
    table.add_row(vec!["Hidden Dimension", &config.hidden_dim.to_string()]);
    table.add_row(vec!["Number of Heads", &config.num_heads.to_string()]);
    table.add_row(vec!["Number of Layers", &config.num_layers.to_string()]);
    table.add_row(vec![
        "Intermediate Dim",
        &config.intermediate_dim.to_string(),
    ]);
    table.add_row(vec![
        "Approx Parameters",
        &format_params(model.num_parameters()),
    ]);
    println!("{table}");

    // Example 1: Forward pass
    println!();
    println!(
        "{} {}",
        style("▶").blue().bold(),
        style("Forward Pass").blue().bold()
    );
    let prompt_tokens = vec![1, 5, 10];
    let start = Instant::now();
    let logits = model.forward(&prompt_tokens).expect("Forward pass failed");
    let elapsed = start.elapsed();

    println!("  Input tokens: {:?}", prompt_tokens);
    println!(
        "  Output logits shape: [{}, {}]",
        prompt_tokens.len(),
        config.vocab_size
    );
    println!(
        "  Last position max logit: {:.4}",
        find_max(&logits.data()[(prompt_tokens.len() - 1) * config.vocab_size..])
    );
    println!("  {} {:.2?}", style("Latency:").dim(), elapsed);

    // Example 2: Greedy generation
    println!();
    println!(
        "{} {}",
        style("▶").green().bold(),
        style("Greedy Generation").green().bold()
    );
    let gen_config = GenerationConfig::greedy()
        .with_max_tokens(5)
        .with_eos_token_id(99);

    let start = Instant::now();
    let generated = model
        .generate(&prompt_tokens, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    println!("  Prompt: {:?}", prompt_tokens);
    println!("  Generated: {}", style(format!("{:?}", generated)).green());
    println!("  New tokens: {:?}", &generated[prompt_tokens.len()..]);
    println!("  {} {:.2?}", style("Latency:").dim(), elapsed);

    // Example 3: Top-k sampling
    println!();
    println!(
        "{} {}",
        style("▶").yellow().bold(),
        style("Top-K Sampling (k=5)").yellow().bold()
    );
    let gen_config = GenerationConfig::top_k(5)
        .with_temperature(0.8)
        .with_max_tokens(5)
        .with_seed(42);

    let start = Instant::now();
    let generated = model
        .generate(&prompt_tokens, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    println!("  Prompt: {:?}", prompt_tokens);
    println!(
        "  Generated: {}",
        style(format!("{:?}", generated)).yellow()
    );
    println!("  {} {:.2?}", style("Latency:").dim(), elapsed);

    // Example 4: Top-p (nucleus) sampling
    println!();
    println!(
        "{} {}",
        style("▶").magenta().bold(),
        style("Top-P Sampling (p=0.9)").magenta().bold()
    );
    let gen_config = GenerationConfig::top_p(0.9)
        .with_temperature(0.7)
        .with_max_tokens(5)
        .with_seed(123);

    let start = Instant::now();
    let generated = model
        .generate(&prompt_tokens, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    println!("  Prompt: {:?}", prompt_tokens);
    println!(
        "  Generated: {}",
        style(format!("{:?}", generated)).magenta()
    );
    println!("  {} {:.2?}", style("Latency:").dim(), elapsed);

    // Results summary table
    println!();
    println!("{}", style("Results Summary").yellow().bold());
    let mut results_table = Table::new();
    results_table.load_preset(UTF8_FULL);
    results_table.set_header(vec![
        Cell::new("Strategy").fg(Color::Cyan),
        Cell::new("Temperature").fg(Color::Cyan),
        Cell::new("Tokens Generated").fg(Color::Cyan),
        Cell::new("Status").fg(Color::Cyan),
    ]);
    results_table.add_row(vec!["Greedy", "1.0", "5", "✓ Success"]);
    results_table.add_row(vec!["Top-K (k=5)", "0.8", "5", "✓ Success"]);
    results_table.add_row(vec!["Top-P (p=0.9)", "0.7", "5", "✓ Success"]);
    println!("{results_table}");

    // Footer
    println!();
    println!(
        "{}",
        style("════════════════════════════════════════")
            .cyan()
            .dim()
    );
    println!("  {} Inference complete!", style("✓").green().bold());
    println!(
        "{}",
        style("════════════════════════════════════════")
            .cyan()
            .dim()
    );
    println!();
}

/// Format parameter count with K/M suffixes
fn format_params(count: usize) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        format!("{count}")
    }
}

/// Find maximum value in slice
fn find_max(slice: &[f32]) -> f32 {
    slice
        .iter()
        .copied()
        .max_by(|a, b| {
            a.partial_cmp(b)
                .expect("Comparison failed: NaN values not allowed in logits")
        })
        .unwrap_or(0.0)
}
