//! CUDA-accelerated chat completions example
//!
//! Demonstrates high-performance GPU inference using `OwnedQuantizedModelCuda`
//! with pre-uploaded weights for maximum throughput.
//!
//! # Performance
//!
//! | Mode | Throughput | Memory |
//! |------|------------|--------|
//! | CPU baseline | ~15 tok/s | 1.1 GB |
//! | GPU (lazy) | ~83 tok/s | 1.5 GB |
//! | GPU (preloaded) | ~755 tok/s | 1.9 GB |
//!
//! # Usage
//!
//! ```bash
//! cargo run --features cuda --example cuda_chat_completions -- /path/to/model.gguf
//! ```
//!
//! # Requirements
//!
//! - NVIDIA GPU with CUDA support
//! - GGUF model file (e.g., Qwen2.5-Coder-1.5B-Instruct-Q4_K_M)

use realizar::cuda::CudaExecutor;
use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig,
};
use std::env;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       CUDA Chat Completions Example                          ║");
    println!("║       High-Performance GPU Inference with Pre-uploaded Weights║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Get model path from args
    let args: Vec<String> = env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        println!(
            "Usage: cargo run --features cuda --example cuda_chat_completions -- <model.gguf>"
        );
        println!();
        println!("Example models:");
        println!("  - qwen2.5-coder-1.5b-instruct-q4_k_m.gguf (recommended)");
        println!("  - tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
        return;
    };

    // Check CUDA availability
    if !CudaExecutor::is_available() {
        println!("❌ CUDA not available. Cannot run GPU example.");
        return;
    }

    let num_devices = CudaExecutor::num_devices();
    println!("✅ CUDA available: {} device(s)", num_devices);

    // Get device info
    let executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(err) => {
            println!("❌ Failed to create CUDA executor: {}", err);
            return;
        },
    };

    let device_name = executor.device_name().unwrap_or_default();
    let (vram_free, vram_total) = executor.memory_info().unwrap_or((0, 0));
    println!("   Device: {}", device_name);
    println!(
        "   VRAM: {} MB ({} MB free)",
        vram_total / 1024 / 1024,
        vram_free / 1024 / 1024
    );
    println!();

    // Load model
    println!("Loading model: {}", model_path);
    let start = Instant::now();

    let mapped = match MappedGGUFModel::from_path(model_path) {
        Ok(m) => m,
        Err(e) => {
            println!("❌ Failed to load model: {}", e);
            return;
        },
    };

    let cpu_model = match OwnedQuantizedModel::from_mapped(&mapped) {
        Ok(m) => m,
        Err(e) => {
            println!("❌ Failed to build model: {}", e);
            return;
        },
    };

    println!("   Model loaded in {:.2?}", start.elapsed());
    println!("   Layers: {}", cpu_model.config().num_layers);
    println!("   Hidden dim: {}", cpu_model.config().hidden_dim);
    println!("   Vocab size: {}", cpu_model.config().vocab_size);
    println!();

    // Create CUDA model with pre-uploaded weights
    println!("Creating CUDA model with pre-uploaded weights...");
    let start = Instant::now();

    let mut cuda_model = match OwnedQuantizedModelCuda::new(cpu_model, 0) {
        Ok(m) => m,
        Err(e) => {
            println!("❌ Failed to create CUDA model: {}", e);
            return;
        },
    };

    // Pre-upload weights to GPU
    let bytes_uploaded = match cuda_model.preload_weights_gpu() {
        Ok(bytes) => bytes,
        Err(e) => {
            println!("❌ Failed to preload weights: {}", e);
            return;
        },
    };

    println!(
        "   Uploaded {} MB to GPU in {:.2?}",
        bytes_uploaded / 1024 / 1024,
        start.elapsed()
    );
    println!();

    // Get vocabulary for decoding
    println!("Loading tokenizer...");
    let vocab = match mapped.model.vocabulary() {
        Some(v) => v,
        None => {
            println!("❌ Failed to load vocabulary");
            return;
        },
    };
    println!("   Vocab size: {}", vocab.len());
    println!();

    // Format chat message with ChatML template
    let system_msg = "You are a helpful coding assistant.";
    let user_msg = "Write a Rust function to calculate the factorial of a number.";

    let prompt = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        system_msg, user_msg
    );

    println!("Prompt:");
    println!("  System: {}", system_msg);
    println!("  User: {}", user_msg);
    println!();

    // Tokenize using GGUFModel's encode method
    let input_ids = match mapped.model.encode(&prompt) {
        Some(ids) => ids,
        None => {
            println!("❌ Failed to encode prompt");
            return;
        },
    };
    println!("Tokenized: {} tokens", input_ids.len());
    println!();

    // Configure generation
    let config = QuantizedGenerateConfig {
        max_tokens: 100,
        temperature: 0.7,
        top_k: 40,
        stop_tokens: vec![151645], // <|im_end|> for Qwen models
        trace: false,
        ..Default::default()
    };

    // Generate with GPU
    println!("Generating response (GPU)...");
    let start = Instant::now();

    let output_ids = match cuda_model.generate_gpu_resident(&input_ids, &config) {
        Ok(ids) => ids,
        Err(e) => {
            println!("❌ Generation failed: {}", e);
            return;
        },
    };

    let elapsed = start.elapsed();
    let new_tokens = output_ids.len() - input_ids.len();
    let toks_per_sec = new_tokens as f64 / elapsed.as_secs_f64();

    // Decode output using GGUFModel's decode method
    let response = mapped.model.decode(&output_ids[input_ids.len()..]);

    println!();
    println!("Response:");
    println!("─────────────────────────────────────────────────────────────────");
    println!("{}", response.trim());
    println!("─────────────────────────────────────────────────────────────────");
    println!();

    // Print stats
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PERFORMANCE SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Tokens generated: {}", new_tokens);
    println!("  Latency: {:.2?}", elapsed);
    println!("  Throughput: {:.1} tok/s", toks_per_sec);
    println!();

    // Compare to baselines
    let ollama_baseline = 333.0; // tok/s
    let speedup = toks_per_sec / ollama_baseline;
    println!(
        "  vs Ollama ({:.0} tok/s): {:.2}x",
        ollama_baseline, speedup
    );

    if speedup >= 2.0 {
        println!("  ✅ Exceeds 2x Ollama parity!");
    } else if speedup >= 1.0 {
        println!("  ✅ Matches Ollama parity!");
    } else {
        println!("  ⚠️  Below Ollama baseline (optimize batch size)");
    }

    println!();
    println!("Tip: For higher throughput, use batched inference with M=16");
    println!("     Expected: 850+ tok/s (2.9x Ollama)");
}
