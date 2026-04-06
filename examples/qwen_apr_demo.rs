//! Qwen2.5-Coder-0.5B .apr Serving Demo
//!
//! Demonstrates the complete workflow:
//! 1. Load GGUF model
//! 2. Convert to APR format
//! 3. Run inference with both GGUF and APR
//!
//! Usage:
//!   cargo run --example qwen_apr_demo --release
//!
//! Or with custom model:
//!   cargo run --example qwen_apr_demo --release -- /path/to/qwen.gguf

use realizar::apr_transformer::QuantizedAprTransformerQ4;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};
use std::env;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let model_path = args.get(1).map_or(
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-0.5b-instruct-q4_0.gguf",
        |s| s.as_str(),
    );

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Qwen2.5-Coder-0.5B APR Serving Demo (Sovereign AI Stack)     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // Step 1: Load GGUF Model
    // ========================================================================
    println!("📦 Step 1: Loading GGUF model...");
    let start = Instant::now();

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().ok_or("No vocabulary")?;

    println!("   ✓ Model: Qwen2.5-Coder-0.5B-Instruct");
    println!("   ✓ Architecture: {}", gguf_model.config().architecture);
    println!("   ✓ Hidden dim: {}", gguf_model.config().hidden_dim);
    println!("   ✓ Layers: {}", gguf_model.config().num_layers);
    println!("   ✓ Vocab size: {}", gguf_model.config().vocab_size);
    println!("   ✓ Loaded in {:.2}s\n", start.elapsed().as_secs_f32());

    // ========================================================================
    // Step 2: Convert to APR Q4_0 Format
    // ========================================================================
    println!("🔄 Step 2: Converting to APR Q4_0 format...");
    let start = Instant::now();

    let apr_model = QuantizedAprTransformerQ4::from_gguf(&gguf_model);

    println!(
        "   ✓ Conversion completed in {:.2}s\n",
        start.elapsed().as_secs_f32()
    );

    // ========================================================================
    // Step 3: Run Inference with GGUF (baseline)
    // ========================================================================
    println!("🚀 Step 3: Running GGUF inference (baseline)...");

    let prompt = "def fibonacci(n):";
    let prompt_tokens = mapped.model.encode(prompt).ok_or("Encoding failed")?;
    println!("   Prompt: '{}'", prompt);
    println!("   Tokens: {:?}", prompt_tokens);

    // Create KV cache
    let max_seq_len = 256;
    let head_dim = gguf_model.config().hidden_dim / gguf_model.config().num_heads;
    let kv_dim = gguf_model.config().num_kv_heads * head_dim;
    let mut cache = OwnedQuantizedKVCache::new(gguf_model.config().num_layers, kv_dim, max_seq_len);

    // Prefill
    let start = Instant::now();
    let mut logits = Vec::new();
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        logits = gguf_model.forward_single_with_cache(tok, &mut cache, pos)?;
    }
    let prefill_time = start.elapsed();

    // Generate 30 tokens
    let mut generated = prompt_tokens.clone();
    let gen_start = Instant::now();
    for i in 0..30 {
        let (best_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("test");

        generated.push(best_idx as u32);
        let pos = prompt_tokens.len() + i;
        logits = gguf_model.forward_single_with_cache(best_idx as u32, &mut cache, pos)?;
    }
    let gen_time = gen_start.elapsed();

    // Decode output
    let mut output = String::new();
    for &tok in &generated {
        if (tok as usize) < vocab.len() {
            let tok_str = &vocab[tok as usize];
            output.push_str(&tok_str.replace("▁", " ").replace('\u{0120}', " "));
        }
    }

    println!("\n   Generated code (GGUF):");
    println!("   ─────────────────────────────────────────");
    for line in output.lines().take(10) {
        println!("   {}", line);
    }
    println!("   ─────────────────────────────────────────");
    println!("   ✓ Prefill: {:.1}ms", prefill_time.as_secs_f64() * 1000.0);
    println!(
        "   ✓ Generation: {:.1} tok/s ({:.1}ms total)\n",
        30.0 / gen_time.as_secs_f64(),
        gen_time.as_secs_f64() * 1000.0
    );

    // ========================================================================
    // Step 4: Run Inference with APR Q4_0
    // ========================================================================
    println!("🚀 Step 4: Running APR Q4_0 inference...");

    // Warmup
    for _ in 0..3 {
        let _ = apr_model.forward(&[1]);
    }

    // Create APR KV cache
    let mut apr_cache = apr_model.create_kv_cache();

    // Prefill
    let start = Instant::now();
    let mut apr_logits = Vec::new();
    for &tok in &prompt_tokens {
        apr_logits = apr_model.forward_with_cache(&[tok], &mut apr_cache)?;
    }
    let apr_prefill = start.elapsed();

    // Generate
    let mut apr_generated = prompt_tokens;
    let gen_start = Instant::now();
    for _ in 0..30 {
        let (best_idx, _) = apr_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("test");

        apr_generated.push(best_idx as u32);
        apr_logits = apr_model.forward_with_cache(&[best_idx as u32], &mut apr_cache)?;
    }
    let apr_gen_time = gen_start.elapsed();

    // Decode APR output
    let mut apr_output = String::new();
    for &tok in &apr_generated {
        if (tok as usize) < vocab.len() {
            let tok_str = &vocab[tok as usize];
            apr_output.push_str(&tok_str.replace("▁", " ").replace('\u{0120}', " "));
        }
    }

    println!("\n   Generated code (APR Q4_0):");
    println!("   ─────────────────────────────────────────");
    for line in apr_output.lines().take(10) {
        println!("   {}", line);
    }
    println!("   ─────────────────────────────────────────");
    println!("   ✓ Prefill: {:.1}ms", apr_prefill.as_secs_f64() * 1000.0);
    println!(
        "   ✓ Generation: {:.1} tok/s ({:.1}ms total)\n",
        30.0 / apr_gen_time.as_secs_f64(),
        apr_gen_time.as_secs_f64() * 1000.0
    );

    // ========================================================================
    // Summary
    // ========================================================================
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         Summary                                   ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║ GGUF Throughput:     {:>6.1} tok/s                                ║",
        30.0 / gen_time.as_secs_f64()
    );
    println!(
        "║ APR Q4_0 Throughput: {:>6.1} tok/s                                ║",
        30.0 / apr_gen_time.as_secs_f64()
    );
    println!(
        "║ Speedup:             {:>6.2}x                                     ║",
        gen_time.as_secs_f64() / apr_gen_time.as_secs_f64()
    );
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║ Model: Qwen2.5-Coder-0.5B-Instruct (Q4_0 quantization)           ║");
    println!("║ Format: .apr (Aprender native, WASM-compatible)                  ║");
    println!("║ Stack: Pure Rust Sovereign AI (trueno + realizar)                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    Ok(())
}
