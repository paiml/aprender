//! Phi Model Import from HuggingFace
//!
//! Demonstrates importing Microsoft's smallest Phi model from HuggingFace Hub
//! and converting it to APR format.
//!
//! # Model: microsoft/phi-1_5
//!
//! Phi-1.5 is a 1.3B parameter transformer model designed for code and
//! natural language understanding. It's the smallest official Phi model.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example phi_hf_import
//! ```
//!
//! # Actual Import (requires network)
//!
//! ```bash
//! apr import hf://microsoft/phi-1_5 -o phi-1.5.apr --arch llama
//! ```

use aprender::format::{
    apr_import, Architecture, ImportOptions, Source, ValidationConfig,
};
use std::path::PathBuf;

fn main() {
    println!("=== Phi-1.5 HuggingFace Import Demo ===\n");

    // Model information
    println!("Model: microsoft/phi-1_5");
    println!("  Parameters: 1.3B");
    println!("  Architecture: Transformer (LLaMA-style)");
    println!("  Context length: 2048 tokens");
    println!("  Training: Textbooks, code, synthetic data");
    println!("  License: MIT");
    println!();

    // Parse HuggingFace source
    let source_str = "hf://microsoft/phi-1_5";
    println!("=== Parsing HuggingFace Source ===\n");

    match Source::parse(source_str) {
        Ok(source) => {
            println!("Source parsed successfully:");
            match &source {
                Source::HuggingFace { org, repo, file } => {
                    println!("  Type: HuggingFace Hub");
                    println!("  Organization: {org}");
                    println!("  Repository: {repo}");
                    if let Some(f) = file {
                        println!("  File: {f}");
                    }
                }
                Source::Local(path) => println!("  Local: {}", path.display()),
                Source::Url(url) => println!("  URL: {url}"),
            }
        }
        Err(e) => {
            println!("Failed to parse source: {e}");
            return;
        }
    }
    println!();

    // Configure import options
    println!("=== Import Configuration ===\n");

    let options = ImportOptions {
        architecture: Architecture::Llama, // Phi uses LLaMA-style architecture
        validation: ValidationConfig::Strict,
        quantize: None, // Keep FP16 for now
        compress: None,
        force: false,
        cache: true,
    };

    println!("Architecture: {:?}", options.architecture);
    println!("Validation: {:?}", options.validation);
    println!("Quantization: None (FP16)");
    println!("Caching: Enabled");
    println!();

    // Demonstrate the import pipeline (dry run)
    println!("=== Import Pipeline (Dry Run) ===\n");
    println!("Steps that would execute:");
    println!("  1. Resolve HuggingFace repository: microsoft/phi-1_5");
    println!("  2. Download model.safetensors.index.json");
    println!("  3. Parse weight map for sharded files");
    println!("  4. Download each shard (model-00001-of-00002.safetensors, ...)");
    println!("  5. Map tensor names to APR canonical format");
    println!("  6. Validate tensor shapes and dtypes");
    println!("  7. Write APR v2 format output");
    println!("  8. Generate validation report");
    println!();

    // Expected tensor layout
    println!("=== Expected Tensor Layout ===\n");
    println!("Phi-1.5 tensors (LLaMA-style naming):");
    println!("  model.embed_tokens.weight         [51200, 2048]");
    println!("  model.layers.0.self_attn.q_proj   [2048, 2048]");
    println!("  model.layers.0.self_attn.k_proj   [2048, 2048]");
    println!("  model.layers.0.self_attn.v_proj   [2048, 2048]");
    println!("  model.layers.0.self_attn.o_proj   [2048, 2048]");
    println!("  model.layers.0.mlp.gate_proj      [2048, 8192]");
    println!("  model.layers.0.mlp.up_proj        [2048, 8192]");
    println!("  model.layers.0.mlp.down_proj      [8192, 2048]");
    println!("  model.layers.0.input_layernorm    [2048]");
    println!("  model.layers.0.post_attention_layernorm [2048]");
    println!("  ... (24 layers total)");
    println!("  model.norm.weight                 [2048]");
    println!("  lm_head.weight                    [51200, 2048]");
    println!();

    // Size estimates
    println!("=== Size Estimates ===\n");
    let vocab_size = 51200_usize;
    let hidden_size = 2048_usize;
    let num_layers = 24_usize;
    let intermediate_size = 8192_usize;

    // FP16 = 2 bytes per param
    let embed_params = vocab_size * hidden_size;
    let layer_params = hidden_size * hidden_size * 4  // QKV + O
        + hidden_size * intermediate_size * 3; // gate, up, down
    let total_params = embed_params * 2 + layer_params * num_layers;
    let fp16_size_mb = (total_params * 2) as f64 / 1024.0 / 1024.0;
    let int4_size_mb = fp16_size_mb / 4.0;

    println!("Estimated parameters: ~{:.1}B", total_params as f64 / 1e9);
    println!("FP16 size: ~{:.0} MB", fp16_size_mb);
    println!("INT4 size: ~{:.0} MB", int4_size_mb);
    println!();

    // Quantization options
    println!("=== Quantization Options ===\n");
    println!("For smaller model sizes, use --quantize:");
    println!("  apr import hf://microsoft/phi-1_5 -o phi.apr --quantize int8  # ~50% size");
    println!("  apr import hf://microsoft/phi-1_5 -o phi.apr --quantize int4  # ~25% size");
    println!("  apr import hf://microsoft/phi-1_5 -o phi.apr --quantize fp16  # Keep FP16");
    println!();

    // Actual import attempt (will fail without network/model)
    println!("=== Attempting Import (Demo) ===\n");

    let output_path = PathBuf::from("/tmp/phi-1.5-demo.apr");
    println!("Output path: {}", output_path.display());
    println!();

    // Note: This will fail in demo mode as we don't have network access
    // In real usage, this would download and convert the model
    match apr_import(source_str, &output_path, options) {
        Ok(report) => {
            println!("Import successful!");
            println!("  Score: {}/100 (Grade: {})", report.total_score, report.grade());
            println!("  Checks passed: {}", report.checks.iter().filter(|c| c.status.is_pass()).count());
        }
        Err(e) => {
            println!("Import not available in demo mode:");
            println!("  Error: {e}");
            println!();
            println!("To actually import, run:");
            println!("  apr import hf://microsoft/phi-1_5 -o phi-1.5.apr");
        }
    }
    println!();

    // Alternative tiny models
    println!("=== Alternative Tiny Phi-like Models ===\n");
    println!("Other small instruction-following models:");
    println!("  microsoft/phi-1_5          1.3B params (MIT)");
    println!("  microsoft/phi-2            2.7B params (MIT)");
    println!("  TinyLlama/TinyLlama-1.1B   1.1B params (Apache 2.0)");
    println!("  Qwen/Qwen2-0.5B-Instruct   0.5B params (Apache 2.0)");
    println!();
    println!("For the absolute smallest instruct model, use Qwen2-0.5B!");

    println!("\n=== Demo Complete ===");
}
