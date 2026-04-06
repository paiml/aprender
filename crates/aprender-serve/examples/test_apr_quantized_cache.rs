//! Test APR model loading with quantized weight caching
use realizar::apr::{AprV2Model, AprV2ModelCuda};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: test_apr_quantized_cache <model.apr>");
        std::process::exit(1);
    });

    println!("Loading APR model: {}", path);
    let start = std::time::Instant::now();

    // First load the APR model
    let apr_model = match AprV2Model::load(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load APR model: {e}");
            std::process::exit(1);
        },
    };
    println!("APR model loaded in {:.2}ms", start.elapsed().as_millis());
    println!("Tensors: {}", apr_model.tensor_count());

    // Now wrap with CUDA executor
    let cuda_start = std::time::Instant::now();
    match AprV2ModelCuda::with_max_seq_len(apr_model, 0, 4096) {
        Ok(model) => {
            println!(
                "CUDA init + weight caching in {:.2}ms",
                cuda_start.elapsed().as_millis()
            );
            println!("Device: {}", model.device_name());
            println!("Cached weight MB: {}", model.cached_weight_mb());
            println!("Weights cached: {}", model.weights_cached());

            // Test multiple forward passes to measure steady-state throughput
            let mut model = model;

            // Warmup: compiles kernels and captures CUDA graph
            println!("\nWarmup pass (kernel compilation + CUDA graph capture)...");
            let embed_start = std::time::Instant::now();
            // Do 3 warmup tokens to fully capture graph and warm caches
            for _ in 0..3 {
                let _ = model.forward_cuda_to_token(151644u32);
            }
            println!(
                "Warmup (3 tokens): {:.2}ms",
                embed_start.elapsed().as_millis()
            );

            // Reset KV cache for clean benchmark
            println!("Resetting KV cache for benchmark...");
            model.reset_kv_cache();

            // Increased token count for more stable measurement
            let num_tokens = 100;
            println!(
                "\nGenerating {} tokens with CUDA graph replay...",
                num_tokens
            );
            let mut total_time_ms = 0.0;
            let mut tokens = vec![151644u32]; // BOS token

            for i in 0..num_tokens {
                let fwd_start = std::time::Instant::now();
                // Use forward_cuda_to_token which uses GPU argmax (4 bytes D2H vs 600KB)
                match model.forward_cuda_to_token(tokens[tokens.len() - 1]) {
                    Ok(next_token) => {
                        let fwd_time_ms = fwd_start.elapsed().as_secs_f64() * 1000.0;
                        tokens.push(next_token);

                        if i == 0 {
                            println!("Token 1 (GPU argmax): {:.2}ms", fwd_time_ms);
                        }
                        // Count all tokens for steady-state (graph is already captured)
                        total_time_ms += fwd_time_ms;
                    },
                    Err(e) => {
                        eprintln!("Forward pass failed at token {}: {e}", i + 1);
                        break;
                    },
                }
            }

            // Calculate throughput (all tokens use CUDA graph replay)
            if total_time_ms > 0.0 {
                let tok_per_sec = (num_tokens as f64) / (total_time_ms / 1000.0);
                println!("\n=== Benchmark Results ===");
                println!(
                    "CUDA graph replay ({} tokens): {:.2}ms avg/token",
                    num_tokens,
                    total_time_ms / num_tokens as f64
                );
                println!("Throughput: {:.1} tok/s", tok_per_sec);
                println!("Target: >= 240 tok/s (2x Ollama)");
                println!("Generated tokens: {:?}", &tokens[1..]);
            }
        },
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            std::process::exit(1);
        },
    }
}
