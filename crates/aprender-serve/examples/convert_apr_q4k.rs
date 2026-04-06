//! APR Q4K Conversion Example
//!
//! Converts a GGUF Q4K model to APR Q4K format preserving quantization.
//!
//! Run with: cargo run --release --example convert_apr_q4k

use realizar::convert::{GgufToAprQ4KConverter, Q4KConversionStats};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("APR Q4K Converter");
    println!("=================");
    println!();

    let gguf_path = std::env::var("GGUF_PATH").unwrap_or_else(|_| {
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
            .to_string()
    });

    let output_path = std::env::var("OUTPUT_PATH")
        .unwrap_or_else(|_| "/tmp/qwen2.5-coder-1.5b-q4k.apr".to_string());

    if !Path::new(&gguf_path).exists() {
        eprintln!("GGUF file not found: {}", gguf_path);
        eprintln!("Set GGUF_PATH environment variable to specify the input file.");
        return;
    }

    println!("Input:  {}", gguf_path);
    println!("Output: {}", output_path);
    println!();

    let start = Instant::now();
    match GgufToAprQ4KConverter::convert(Path::new(&gguf_path), Path::new(&output_path)) {
        Ok(stats) => {
            let elapsed = start.elapsed();
            println!("Conversion successful!");
            println!();
            print_stats(&stats);
            println!();
            println!("Time: {:.2}s", elapsed.as_secs_f64());

            // Verify output file
            if let Ok(metadata) = std::fs::metadata(&output_path) {
                println!("Output size: {:.2} MB", metadata.len() as f64 / 1_000_000.0);
            }
        },
        Err(e) => {
            eprintln!("Conversion failed: {}", e);
        },
    }
}

fn print_stats(stats: &Q4KConversionStats) {
    println!("Model Statistics:");
    println!("  Architecture:    {}", stats.architecture);
    println!("  Hidden size:     {}", stats.hidden_size);
    println!("  Num layers:      {}", stats.num_layers);
    println!("  Total tensors:   {}", stats.tensor_count);
    println!(
        "  Q4K tensors:     {} ({:.1}%)",
        stats.q4k_tensor_count,
        stats.q4k_tensor_count as f64 / stats.tensor_count as f64 * 100.0
    );
    println!(
        "  Total bytes:     {:.2} MB",
        stats.total_bytes as f64 / 1_000_000.0
    );
}
