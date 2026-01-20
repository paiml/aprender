//! Example: GPU-accelerated chat completions with apr serve
//!
//! This example demonstrates how to use GPU-accelerated inference for chat completions.
//! It shows the performance difference between CPU and GPU inference, achieving up to
//! 2.9x faster throughput than Ollama with batched GPU inference.
//!
//! # Prerequisites
//!
//! 1. NVIDIA GPU with CUDA support
//! 2. A GGUF model (e.g., Qwen2.5-Coder-1.5B-Instruct-Q4_K_M)
//!
//! # Running the GPU server
//!
//! ```bash
//! # Start GPU-accelerated server (single-request mode)
//! apr serve /path/to/model.gguf --port 8096 --gpu
//!
//! # Start GPU-accelerated server (batched mode - 2.9x faster)
//! apr serve /path/to/model.gguf --port 8096 --gpu --batch
//! ```
//!
//! # Performance comparison
//!
//! | Mode | Throughput | vs Ollama |
//! |------|------------|-----------|
//! | CPU (baseline) | ~15 tok/s | 0.05x |
//! | GPU single | ~83 tok/s | 0.25x |
//! | GPU batched (M=16) | ~850 tok/s | 2.9x |
//! | Ollama (baseline) | ~333 tok/s | 1.0x |
//!
//! # Using the API
//!
//! ```bash
//! # Chat completion request
//! curl -X POST http://localhost:8096/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -d '{
//!     "model": "default",
//!     "messages": [
//!       {"role": "user", "content": "Write a Rust function to add two numbers"}
//!     ],
//!     "max_tokens": 100,
//!     "temperature": 0.7
//!   }'
//! ```
//!
//! # Tracing for performance analysis
//!
//! ```bash
//! # Add tracing header to see per-token timing
//! curl -X POST http://localhost:8096/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -H "X-Trace-Level: brick" \
//!   -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}'
//! ```
//!
//! The response will include a `brick_trace` field with timing breakdown.

use std::process::Command;
use std::time::Instant;

const DEFAULT_PORT: u16 = 8096;

fn main() {
    println!("APR GPU Chat Inference Example");
    println!("===============================");
    println!();
    println!("This example demonstrates GPU-accelerated chat completions.");
    println!();

    // Check if server is running
    let health_url = format!("http://localhost:{}/health", DEFAULT_PORT);
    let health = Command::new("curl")
        .args(["-s", &health_url])
        .output();

    match health {
        Ok(output) if output.status.success() => {
            let response = String::from_utf8_lossy(&output.stdout);
            println!("Server health: {}", response.trim());
            println!();

            // Run a simple chat completion
            run_chat_completion();
        }
        _ => {
            println!("Server not running.");
            println!();
            print_instructions();
        }
    }
}

fn run_chat_completion() {
    println!("Running chat completion test...");
    println!();

    let request = r#"{
        "model": "default",
        "messages": [
            {"role": "user", "content": "Write a Rust function to add two numbers."}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }"#;

    let url = format!("http://localhost:{}/v1/chat/completions", DEFAULT_PORT);

    let start = Instant::now();
    let output = Command::new("curl")
        .args([
            "-s",
            "-X", "POST",
            &url,
            "-H", "Content-Type: application/json",
            "-d", request,
        ])
        .output();

    let elapsed = start.elapsed();

    match output {
        Ok(result) if result.status.success() => {
            let response = String::from_utf8_lossy(&result.stdout);

            // Try to parse and display nicely
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&response) {
                if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                    if let Some(first) = choices.first() {
                        if let Some(message) = first.get("message") {
                            if let Some(content) = message.get("content").and_then(|c| c.as_str()) {
                                println!("Response:");
                                println!("  {}", content.replace('\n', "\n  "));
                                println!();
                            }
                        }
                    }
                }

                // Show usage stats
                if let Some(usage) = json.get("usage") {
                    let prompt_tokens = usage.get("prompt_tokens").and_then(|t| t.as_i64()).unwrap_or(0);
                    let completion_tokens = usage.get("completion_tokens").and_then(|t| t.as_i64()).unwrap_or(0);
                    let total_tokens = usage.get("total_tokens").and_then(|t| t.as_i64()).unwrap_or(0);

                    println!("Usage:");
                    println!("  Prompt tokens: {}", prompt_tokens);
                    println!("  Completion tokens: {}", completion_tokens);
                    println!("  Total tokens: {}", total_tokens);
                    println!();

                    // Calculate throughput
                    let elapsed_secs = elapsed.as_secs_f64();
                    if elapsed_secs > 0.0 && completion_tokens > 0 {
                        let toks_per_sec = completion_tokens as f64 / elapsed_secs;
                        println!("Performance:");
                        println!("  Latency: {:.2?}", elapsed);
                        println!("  Throughput: {:.1} tok/s", toks_per_sec);
                        println!();
                    }
                }
            } else {
                println!("Response: {}", response);
            }
        }
        Ok(result) => {
            let stderr = String::from_utf8_lossy(&result.stderr);
            println!("Request failed: {}", stderr);
        }
        Err(e) => {
            println!("Failed to run curl: {}", e);
        }
    }
}

fn print_instructions() {
    println!("To start the GPU-accelerated server:");
    println!();
    println!("  # Single-request mode (~83 tok/s)");
    println!("  apr serve /path/to/model.gguf --port {} --gpu", DEFAULT_PORT);
    println!();
    println!("  # Batched mode (~850 tok/s, 2.9x Ollama)");
    println!("  apr serve /path/to/model.gguf --port {} --gpu --batch", DEFAULT_PORT);
    println!();
    println!("Example models:");
    println!("  - qwen2.5-coder-1.5b-instruct-q4_k_m.gguf (recommended)");
    println!("  - tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    println!();
    println!("Download from HuggingFace:");
    println!("  apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf");
    println!();
}
