//! Qwen2-0.5B Real Inference Demo
//!
//! This example runs actual inference with real Qwen2-0.5B-Instruct model weights
//! and the proper BPE tokenizer.
//!
//! # Prerequisites
//!
//! 1. Download the tokenizer (one-time):
//! ```bash
//! mkdir -p ~/.cache/qwen2
//! curl -L -o ~/.cache/qwen2/tokenizer.json \
//!   "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/tokenizer.json"
//! ```
//!
//! 2. Download the model weights:
//! ```bash
//! # Option A: Using hf-hub (Rust-native)
//! cargo install hf-hub
//! hf download Qwen/Qwen2-0.5B-Instruct --include "model.safetensors"
//!
//! # Option B: Using APR CLI
//! apr import hf://Qwen/Qwen2-0.5B-Instruct -o qwen2-0.5b.apr
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --example qwen_inference --release
//! ```
//!
//! # Popperian Falsification Tests
//!
//! This demo is validated by Section S of the EOY 2025 specification:
//! - S1: Tokenizer loads from tokenizer.json ✓
//! - S2: Tokenizer round-trips ASCII correctly ✓
//! - S4: Model loads without OOM ✓
//! - S5: Model loads 219 tensors ✓

use aprender::demo::Qwen2Config;
use aprender::models::Qwen2Model;
use aprender::text::bpe::{load_from_json, BpeTokenizer};
use std::time::Instant;

fn main() {
    println!("=== Qwen2-0.5B Real Inference Demo ===\n");

    // Find and load tokenizer
    let tokenizer = match load_tokenizer() {
        Some(t) => t,
        None => {
            eprintln!("ERROR: Could not find tokenizer.json");
            eprintln!();
            eprintln!("Download the tokenizer:");
            eprintln!("  mkdir -p ~/.cache/qwen2");
            eprintln!("  curl -L -o ~/.cache/qwen2/tokenizer.json \\");
            eprintln!("    \"https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/tokenizer.json\"");
            std::process::exit(1);
        }
    };
    println!(
        "Tokenizer loaded: {} tokens in vocabulary\n",
        tokenizer.vocab_size()
    );

    // Find model weights
    let model_path = find_model_weights();
    let model_path = match model_path {
        Some(p) => p,
        None => {
            eprintln!("ERROR: Could not find Qwen2-0.5B-Instruct model weights.");
            eprintln!();
            eprintln!("Download using hf-hub (Rust-native):");
            eprintln!("  cargo install hf-hub");
            eprintln!("  hf download Qwen/Qwen2-0.5B-Instruct --include model.safetensors");
            eprintln!();
            eprintln!("Or use the APR CLI:");
            eprintln!("  apr import hf://Qwen/Qwen2-0.5B-Instruct -o qwen2-0.5b.apr");
            std::process::exit(1);
        }
    };

    println!("Model weights: {}\n", model_path.display());

    // Load configuration
    let config = Qwen2Config::qwen2_0_5b_instruct();
    println!("Model: Qwen2-0.5B-Instruct");
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Layers: {}", config.num_layers);
    println!("  Vocab size: {}", config.vocab_size);
    println!();

    // Load model weights using uninitialized model to avoid OOM
    // This uses placeholder tensors (1 element each) instead of full allocation
    println!("Loading model weights (memory-efficient mode)...");
    let load_start = Instant::now();

    let mut model = Qwen2Model::new_uninitialized(&config);
    match model.load_from_safetensors(&model_path) {
        Ok(n) => println!("Loaded {} weight tensors", n),
        Err(e) => {
            eprintln!("ERROR: Failed to load model: {}", e);
            std::process::exit(1);
        }
    }
    model.eval();

    let load_time = load_start.elapsed();
    println!("Model loaded in {:.2}s\n", load_time.as_secs_f32());

    // Test prompts with Qwen2 chat format
    let prompts = [
        "What is 2+2?",
        "Hello, my name is",
        "The capital of France is",
    ];

    for prompt in &prompts {
        println!("─────────────────────────────────────────");
        println!("User: {}", prompt);
        println!();

        // Format prompt with Qwen2 chat template
        let formatted = format_chat_prompt(prompt);

        // Tokenize with proper BPE
        let input_ids = tokenizer.encode(&formatted);
        println!("Input tokens: {} tokens", input_ids.len());

        // Generate
        let gen_start = Instant::now();
        let output_ids = model.generate(
            &input_ids, 32,  // max_new_tokens
            0.7, // temperature
            0.9, // top_p (not used currently)
        );
        let gen_time = gen_start.elapsed();

        let new_tokens = output_ids.len() - input_ids.len();
        let tokens_per_sec = if gen_time.as_secs_f32() > 0.0 {
            new_tokens as f32 / gen_time.as_secs_f32()
        } else {
            0.0
        };

        // Decode output with proper BPE
        let response = tokenizer.decode(&output_ids[input_ids.len()..]);

        println!("Assistant: {}", response.trim());
        println!();
        println!(
            "Generated {} tokens in {:.2}s ({:.1} tok/s)",
            new_tokens,
            gen_time.as_secs_f32(),
            tokens_per_sec
        );
        println!();
    }

    println!("─────────────────────────────────────────");
    println!("\n=== Inference Complete ===");
}

/// Load tokenizer from common locations
fn load_tokenizer() -> Option<BpeTokenizer> {
    let candidates = [
        // Custom cache location
        home_dir().map(|h| h.join(".cache/qwen2/tokenizer.json")),
        // HuggingFace cache
        home_dir()
            .map(|h| h.join(".cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots")),
    ];

    for candidate in candidates.iter().flatten() {
        // Direct file
        if candidate.exists() && candidate.is_file() {
            if let Ok(json) = std::fs::read_to_string(candidate) {
                if let Ok(tokenizer) = load_from_json(&json) {
                    println!("Tokenizer: {}", candidate.display());
                    return Some(tokenizer);
                }
            }
        }

        // Check in snapshot subdirectories
        if candidate.is_dir() {
            if let Ok(entries) = std::fs::read_dir(candidate) {
                for entry in entries.flatten() {
                    let path = entry.path().join("tokenizer.json");
                    if path.exists() {
                        if let Ok(json) = std::fs::read_to_string(&path) {
                            if let Ok(tokenizer) = load_from_json(&json) {
                                println!("Tokenizer: {}", path.display());
                                return Some(tokenizer);
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

/// Find model weights in common locations
fn find_model_weights() -> Option<std::path::PathBuf> {
    let candidates = [
        // HuggingFace cache (Linux)
        home_dir()
            .map(|h| h.join(".cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots")),
        // Custom download location
        home_dir().map(|h| h.join(".cache/qwen2-0.5b")),
        // Current directory
        Some(std::path::PathBuf::from(".")),
    ];

    for candidate in candidates.iter().flatten() {
        // Check for direct model.safetensors
        let direct = candidate.join("model.safetensors");
        if direct.exists() {
            return Some(direct);
        }

        // Check in snapshot subdirectories
        if candidate.is_dir() {
            if let Ok(entries) = std::fs::read_dir(candidate) {
                for entry in entries.flatten() {
                    let path = entry.path().join("model.safetensors");
                    if path.exists() {
                        return Some(path);
                    }
                }
            }
        }
    }

    None
}

/// Format prompt with Qwen2 chat template
fn format_chat_prompt(user_message: &str) -> String {
    // Qwen2 chat template:
    // <|im_start|>system
    // You are a helpful assistant.<|im_end|>
    // <|im_start|>user
    // {message}<|im_end|>
    // <|im_start|>assistant
    format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n",
        user_message
    )
}

/// Get home directory
fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var("HOME").ok().map(std::path::PathBuf::from)
}
