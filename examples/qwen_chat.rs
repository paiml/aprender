//! Qwen2 Chat Demo (Section O4)
//!
//! CLI demonstration of the Qwen2-0.5B model configuration and tokenization.
//! This example shows the model setup that would be used for browser inference.
//!
//! # Usage
//! ```bash
//! cargo run --example qwen_chat
//! ```

use aprender::demo::{
    Qwen2Config, Qwen2Tokenizer, DemoMetrics, PerplexityChecker, BrowserCompatibility,
};

fn main() {
    println!("=== Qwen2-0.5B Chat Demo ===\n");

    // Load model configuration
    let config = Qwen2Config::qwen2_0_5b_instruct();
    println!("Model Configuration:");
    println!("  Model: Qwen2-0.5B-Instruct");
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Attention heads: {}", config.num_attention_heads);
    println!("  KV heads: {}", config.num_kv_heads);
    println!("  Layers: {}", config.num_layers);
    println!("  Vocabulary size: {}", config.vocab_size);
    println!("  Max sequence length: {}", config.max_seq_len);
    println!("  Intermediate size: {}", config.intermediate_size);
    println!();

    // Initialize tokenizer
    let tokenizer = Qwen2Tokenizer::new();
    println!("Tokenizer Configuration:");
    println!("  BOS token ID: {}", tokenizer.special_tokens.bos_id);
    println!("  EOS token ID: {}", tokenizer.special_tokens.eos_id);
    println!("  PAD token ID: {}", tokenizer.special_tokens.pad_id);
    println!();

    // Demonstrate formatting
    println!("=== Instruction Formatting Demo ===");
    let test_prompts = [
        "Hello, how are you?",
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
    ];

    for prompt in &test_prompts {
        let formatted = tokenizer.format_instruction(prompt);
        println!("\nPrompt: \"{}\"", prompt);
        println!("  Formatted: {}", formatted.replace('\n', "\\n"));
    }

    // Performance metrics
    println!("\n=== Performance Metrics Demo ===");
    let metrics = DemoMetrics {
        load_time_ms: 3500,
        first_token_ms: 1200,
        tokens_per_sec: 18.5,
        peak_memory_bytes: 450 * 1024 * 1024,
        tokens_generated: 100,
    };

    println!("Demo Metrics:");
    println!("  Load time: {} ms", metrics.load_time_ms);
    println!("  First token latency: {} ms", metrics.first_token_ms);
    println!("  Generation speed: {:.1} tokens/sec", metrics.tokens_per_sec);
    println!("  Peak memory: {:.1} MB", metrics.peak_memory_bytes as f64 / 1024.0 / 1024.0);

    // Validate against targets
    println!("\n=== Target Validation ===");
    let validations = [
        ("Load time < 5s", metrics.load_time_ms < 5000),
        ("First token < 2s", metrics.first_token_ms < 2000),
        ("Speed >= 15 tok/s", metrics.tokens_per_sec >= 15.0),
        ("Memory < 512 MB", metrics.peak_memory_bytes < 512 * 1024 * 1024),
    ];

    for (check, passed) in &validations {
        let status = if *passed { "✓ PASS" } else { "✗ FAIL" };
        println!("  {} {}", status, check);
    }

    println!("\n  All targets met: {}", metrics.meets_targets());

    // Perplexity validation
    println!("\n=== Perplexity Validation ===");
    let fp16_perplexity = 8.5;
    let int4_perplexity = 9.2;
    let checker = PerplexityChecker::new(fp16_perplexity);

    println!("  FP16 perplexity: {:.2}", fp16_perplexity);
    println!("  INT4 perplexity: {:.2}", int4_perplexity);
    println!("  Degradation: {:.1}%", checker.degradation_pct(int4_perplexity));
    println!("  Within tolerance: {}", if checker.is_acceptable(int4_perplexity) { "YES" } else { "NO" });

    // Browser compatibility
    println!("\n=== Browser Compatibility ===");
    let compat = BrowserCompatibility::default();

    println!("Minimum Browser Versions:");
    println!("  Chrome: {} (supported: {})", compat.chrome_min, compat.supports_chrome(120));
    println!("  Firefox: {} (supported: {})", compat.firefox_min, compat.supports_firefox(120));
    println!("  Safari: {} (supported: {})", compat.safari_min, compat.supports_safari(17));

    // Model size estimates
    println!("\n=== Model Size Estimates ===");
    println!("  FP16 size: {:.1} MB", config.model_size_fp16() as f64 / 1024.0 / 1024.0);
    println!("  INT4 size: {:.1} MB", config.model_size_int4() as f64 / 1024.0 / 1024.0);
    println!("  KV cache (2K seq): {:.1} MB", config.kv_cache_size(2048) as f64 / 1024.0 / 1024.0);

    // Simulated generation
    println!("\n=== Simulated Generation ===");
    println!("User: What is Rust?");
    println!("\nAssistant: Rust is a systems programming language that focuses on safety,");
    println!("concurrency, and performance. Key features include:");
    println!("- Memory safety without garbage collection");
    println!("- Zero-cost abstractions");
    println!("- Fearless concurrency");
    println!("- Rich type system with ownership model");
    println!("\n(This is a simulated response - actual inference requires model weights)");

    println!("\n=== Demo Complete ===");
    println!("Qwen2-0.5B is configured for browser inference!");
}
