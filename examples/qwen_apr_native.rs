//! Qwen2-0.5B Native APR Format Demo
//!
//! Demonstrates creating and loading a Qwen2-0.5B-Instruct model in native
//! APR v2 format. This is the "north star" reference model for the aprender
//! browser inference demo.
//!
//! # Model: Qwen/Qwen2-0.5B-Instruct
//!
//! The smallest Qwen2 model, ideal for edge deployment and browser inference.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example qwen_apr_native
//! ```
//!
//! # Production Import
//!
//! ```bash
//! apr import hf://Qwen/Qwen2-0.5B-Instruct -o qwen2-0.5b.apr
//! apr convert qwen2-0.5b.apr --quantize int4 -o qwen2-0.5b-int4.apr
//! ```

use aprender::demo::{
    BrowserCompatibility, DemoMetrics, PerplexityChecker, Qwen2Config, Qwen2Tokenizer,
};
use aprender::loading::{Backend, LoadConfig, LoadingMode, VerificationLevel};
use aprender::serialization::apr::AprWriter;
use serde_json::json;

fn main() {
    println!("=== Qwen2-0.5B Native APR Format Demo ===\n");

    // Part 1: Model configuration
    model_config_demo();

    // Part 2: Create APR file
    create_apr_demo();

    // Part 3: Loading strategies
    loading_demo();

    // Part 4: Tokenizer demo
    tokenizer_demo();

    // Part 5: Browser deployment
    browser_demo();

    println!("\n=== Demo Complete ===");
    println!("Qwen2-0.5B is your north star for browser inference!");
}

fn model_config_demo() {
    println!("--- Part 1: Model Configuration ---\n");

    let config = Qwen2Config::qwen2_0_5b_instruct();

    println!("Qwen2-0.5B-Instruct Architecture:");
    println!("  Hidden size:       {}", config.hidden_size);
    println!("  Attention heads:   {}", config.num_attention_heads);
    println!("  KV heads (GQA):    {}", config.num_kv_heads);
    println!("  Layers:            {}", config.num_layers);
    println!("  Vocabulary:        {} tokens", config.vocab_size);
    println!("  Context length:    {} tokens", config.max_seq_len);
    println!("  FFN intermediate:  {}", config.intermediate_size);
    println!("  RoPE theta:        {:.0}", config.rope_theta);
    println!();

    // Size estimates
    println!("Size Estimates:");
    println!(
        "  FP16 size:    {:.1} MB",
        config.model_size_fp16() as f64 / 1024.0 / 1024.0
    );
    println!(
        "  INT4 size:    {:.1} MB",
        config.model_size_int4() as f64 / 1024.0 / 1024.0
    );
    println!(
        "  KV cache (2K): {:.1} MB",
        config.kv_cache_size(2048) as f64 / 1024.0 / 1024.0
    );
    println!(
        "  KV cache (8K): {:.1} MB",
        config.kv_cache_size(8192) as f64 / 1024.0 / 1024.0
    );
    println!();
}

fn create_apr_demo() {
    println!("--- Part 2: Creating APR File ---\n");

    let config = Qwen2Config::qwen2_0_5b_instruct();
    let output_path = "/tmp/qwen2-0.5b-demo.apr";

    println!("Creating minimal Qwen2 APR file...");
    println!("  Output: {}", output_path);
    println!();

    // Create APR writer
    let mut writer = AprWriter::new();

    // Model metadata
    writer.set_metadata("model_type", json!("qwen2"));
    writer.set_metadata("model_name", json!("Qwen2-0.5B-Instruct"));
    writer.set_metadata("hidden_size", json!(config.hidden_size));
    writer.set_metadata("num_attention_heads", json!(config.num_attention_heads));
    writer.set_metadata("num_kv_heads", json!(config.num_kv_heads));
    writer.set_metadata("num_layers", json!(config.num_layers));
    writer.set_metadata("vocab_size", json!(config.vocab_size));
    writer.set_metadata("max_seq_len", json!(config.max_seq_len));
    writer.set_metadata("intermediate_size", json!(config.intermediate_size));
    writer.set_metadata("rope_theta", json!(config.rope_theta));

    // Embedding layer (demo: tiny subset for illustration)
    // Real model has [151936, 896] but we use tiny demo size
    let demo_vocab = 1000_usize;
    let demo_hidden = 64_usize;
    let demo_intermediate = 128_usize;
    let demo_layers = 2_usize;

    println!("APR Tensor Layout (demo subset):");

    // Token embeddings
    let embed_size = demo_vocab * demo_hidden;
    writer.add_tensor_f32(
        "model.embed_tokens.weight",
        vec![demo_vocab, demo_hidden],
        &vec![0.01_f32; embed_size],
    );
    println!("  model.embed_tokens.weight      [{}, {}]", demo_vocab, demo_hidden);

    // Transformer layers
    for i in 0..demo_layers {
        let prefix = format!("model.layers.{}", i);
        let head_dim = demo_hidden / 2; // Simplified

        // Self-attention
        let qkv_size = demo_hidden * demo_hidden;
        writer.add_tensor_f32(
            &format!("{}.self_attn.q_proj.weight", prefix),
            vec![demo_hidden, demo_hidden],
            &vec![0.01_f32; qkv_size],
        );
        writer.add_tensor_f32(
            &format!("{}.self_attn.k_proj.weight", prefix),
            vec![head_dim, demo_hidden],
            &vec![0.01_f32; head_dim * demo_hidden],
        );
        writer.add_tensor_f32(
            &format!("{}.self_attn.v_proj.weight", prefix),
            vec![head_dim, demo_hidden],
            &vec![0.01_f32; head_dim * demo_hidden],
        );
        writer.add_tensor_f32(
            &format!("{}.self_attn.o_proj.weight", prefix),
            vec![demo_hidden, demo_hidden],
            &vec![0.01_f32; qkv_size],
        );

        // FFN (SwiGLU style: gate_proj, up_proj, down_proj)
        writer.add_tensor_f32(
            &format!("{}.mlp.gate_proj.weight", prefix),
            vec![demo_intermediate, demo_hidden],
            &vec![0.01_f32; demo_intermediate * demo_hidden],
        );
        writer.add_tensor_f32(
            &format!("{}.mlp.up_proj.weight", prefix),
            vec![demo_intermediate, demo_hidden],
            &vec![0.01_f32; demo_intermediate * demo_hidden],
        );
        writer.add_tensor_f32(
            &format!("{}.mlp.down_proj.weight", prefix),
            vec![demo_hidden, demo_intermediate],
            &vec![0.01_f32; demo_hidden * demo_intermediate],
        );

        // Layer norms
        writer.add_tensor_f32(
            &format!("{}.input_layernorm.weight", prefix),
            vec![demo_hidden],
            &vec![1.0_f32; demo_hidden],
        );
        writer.add_tensor_f32(
            &format!("{}.post_attention_layernorm.weight", prefix),
            vec![demo_hidden],
            &vec![1.0_f32; demo_hidden],
        );

        println!(
            "  model.layers.{}.self_attn.*    [Q:{}x{}, KV:{}x{}]",
            i, demo_hidden, demo_hidden, head_dim, demo_hidden
        );
        println!(
            "  model.layers.{}.mlp.*          [{}x{}, {}x{}]",
            i, demo_intermediate, demo_hidden, demo_hidden, demo_intermediate
        );
    }

    // Final norm
    writer.add_tensor_f32(
        "model.norm.weight",
        vec![demo_hidden],
        &vec![1.0_f32; demo_hidden],
    );
    println!("  model.norm.weight              [{}]", demo_hidden);

    // LM head (tied with embeddings in real model)
    writer.add_tensor_f32(
        "lm_head.weight",
        vec![demo_vocab, demo_hidden],
        &vec![0.01_f32; embed_size],
    );
    println!("  lm_head.weight                 [{}, {}]", demo_vocab, demo_hidden);

    // Write file
    match writer.write(output_path) {
        Ok(()) => {
            println!();
            println!("APR file created successfully!");
            if let Ok(metadata) = std::fs::metadata(output_path) {
                println!("  Size: {:.2} KB", metadata.len() as f64 / 1024.0);
            }
        }
        Err(e) => {
            println!();
            println!("Failed to write APR file: {}", e);
        }
    }
    println!();
}

fn loading_demo() {
    println!("--- Part 3: Loading Strategies ---\n");

    // Different deployment configurations
    println!("Loading Configuration for Different Targets:\n");

    // Browser/WASM
    let wasm_config = LoadConfig::wasm();
    println!("Browser (WASM):");
    println!("  Mode: {:?}", wasm_config.mode);
    println!("  Verification: {:?}", wasm_config.verification);
    println!(
        "  Max Memory: {} MB",
        wasm_config.max_memory_bytes.unwrap_or(0) / (1024 * 1024)
    );
    println!("  Streaming: {}", wasm_config.streaming);

    // Edge device
    let edge_config = LoadConfig::embedded(256 * 1024 * 1024); // 256MB budget
    println!("\nEdge Device (256MB budget):");
    println!("  Mode: {:?}", edge_config.mode);
    println!("  Verification: {:?}", edge_config.verification);
    println!("  Backend: {:?}", edge_config.backend);

    // Server
    let server_config = LoadConfig::server();
    println!("\nServer (full resources):");
    println!("  Mode: {:?}", server_config.mode);
    println!(
        "  Verification: {:?}",
        server_config.verification
    );
    println!(
        "  Backend: {:?} (SIMD: {})",
        server_config.backend,
        server_config.backend.supports_simd()
    );

    // Custom config for mobile
    let mobile_config = LoadConfig::new()
        .with_mode(LoadingMode::Streaming)
        .with_max_memory(128 * 1024 * 1024) // 128MB
        .with_verification(VerificationLevel::ChecksumOnly)
        .with_backend(Backend::Embedded);

    println!("\nMobile (custom, 128MB budget):");
    println!("  Mode: {:?}", mobile_config.mode);
    println!("  Verification: {:?}", mobile_config.verification);
    println!("  Backend: {:?}", mobile_config.backend);
    println!();
}

fn tokenizer_demo() {
    println!("--- Part 4: Tokenizer Demo ---\n");

    let tokenizer = Qwen2Tokenizer::new();

    println!("Qwen2 Special Tokens:");
    println!("  BOS: {} (im_start)", tokenizer.special_tokens.bos_id);
    println!("  EOS: {} (im_end)", tokenizer.special_tokens.eos_id);
    println!("  PAD: {}", tokenizer.special_tokens.pad_id);
    println!("  <|im_start|>: {}", tokenizer.special_tokens.im_start_id);
    println!("  <|im_end|>: {}", tokenizer.special_tokens.im_end_id);
    println!();

    // Instruction formatting
    println!("Instruction Formatting:");
    let prompts = [
        "What is the capital of France?",
        "Write a haiku about Rust.",
        "Explain quantum computing.",
    ];

    for prompt in &prompts {
        let formatted = tokenizer.format_instruction(prompt);
        println!("\nUser: \"{}\"", prompt);
        println!("Formatted:");
        for line in formatted.lines() {
            println!("  {}", line);
        }
    }
    println!();
}

fn browser_demo() {
    println!("--- Part 5: Browser Deployment ---\n");

    // Browser compatibility
    let compat = BrowserCompatibility::default();
    println!("Minimum Browser Versions:");
    println!(
        "  Chrome: {} (WASM SIMD: {})",
        compat.chrome_min,
        compat.supports_chrome(91)
    );
    println!(
        "  Firefox: {} (WASM SIMD: {})",
        compat.firefox_min,
        compat.supports_firefox(89)
    );
    println!(
        "  Safari: {} (WASM Threads: {})",
        compat.safari_min,
        compat.supports_safari(15)
    );
    println!();

    // Performance targets
    println!("Performance Targets (INT4 quantized):");
    let metrics = DemoMetrics {
        load_time_ms: 3500,
        first_token_ms: 1200,
        tokens_per_sec: 18.5,
        peak_memory_bytes: 350 * 1024 * 1024, // ~350MB for INT4
        tokens_generated: 100,
    };

    let checks = [
        ("Load time < 5s", metrics.load_time_ms < 5000),
        ("First token < 2s", metrics.first_token_ms < 2000),
        ("Speed >= 15 tok/s", metrics.tokens_per_sec >= 15.0),
        ("Memory < 512 MB", metrics.peak_memory_bytes < 512 * 1024 * 1024),
    ];

    for (check, passed) in &checks {
        let status = if *passed { "PASS" } else { "FAIL" };
        println!("  [{}] {}", status, check);
    }
    println!("  All targets: {}", if metrics.meets_targets() { "MET" } else { "NOT MET" });
    println!();

    // Perplexity check
    let fp16_ppl = 8.5;
    let int4_ppl = 9.2;
    let checker = PerplexityChecker::new(fp16_ppl);
    println!("Quantization Quality:");
    println!("  FP16 perplexity: {:.2}", fp16_ppl);
    println!("  INT4 perplexity: {:.2}", int4_ppl);
    println!(
        "  Degradation: {:.1}% (max 15%)",
        checker.degradation_pct(int4_ppl)
    );
    println!(
        "  Acceptable: {}",
        if checker.is_acceptable(int4_ppl) { "YES" } else { "NO" }
    );
    println!();

    // Production pipeline
    println!("Production Pipeline:");
    println!("  1. apr import hf://Qwen/Qwen2-0.5B-Instruct -o qwen.apr");
    println!("  2. apr convert qwen.apr --quantize int4 -o qwen-int4.apr");
    println!("  3. apr validate qwen-int4.apr --quality");
    println!("  4. apr compile qwen-int4.apr --target wasm32 -o qwen.wasm");
    println!("  5. Deploy to CDN and serve via web workers");
}
