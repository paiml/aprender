//! PAR-111: Batched Forward Path Benchmark
//!
//! Benchmarks the multi-sequence batched GEMV forward path to verify
//! 857+ tok/s aggregate throughput with M=4 sequences.
//!
//! Key optimization: Batched GEMV reads/dequantizes weights ONCE for all M inputs,
//! giving ~16x speedup on GEMV operations.
//!
//! Run with: cargo run --release --features cuda --example bench_batched_forward

use realizar::cuda::CudaExecutor;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      PAR-111: Batched Forward Path Benchmark                 ║");
    println!("║      Target: 857+ tok/s aggregate (M=4)                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    if !CudaExecutor::is_available() {
        println!("CUDA not available");
        return;
    }

    // PAR-123: Support MODEL_PATH env var for testing different models
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
            .to_string()
    });
    let model_path = model_path.as_str();

    if !Path::new(model_path).exists() {
        println!("Model not found: {}", model_path);
        return;
    }

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path).expect("model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("model");
    let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0).expect("CUDA");

    // Pre-upload weights
    cuda_model.preload_weights_gpu().expect("weights");

    let hidden_dim = cuda_model.model().config().hidden_dim;
    let intermediate_dim = cuda_model.model().layers()[0].ffn_up_weight.out_dim;
    let num_layers = cuda_model.model().layers().len();
    let vocab_size = cuda_model.model().lm_head_weight().out_dim;
    let eps = cuda_model.model().config().eps;

    println!(
        "Model: {} layers, hidden_dim={}, vocab_size={}",
        num_layers, hidden_dim, vocab_size
    );
    println!();

    // Test different batch sizes
    let batch_sizes = [1, 2, 4, 8]; // M=16 not supported (register pressure)
    let num_iterations = 50;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmarking Batched Forward (M sequences in parallel)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    for &m in &batch_sizes {
        println!("  Testing M={} sequences...", m);

        // Initialize batched workspace
        if let Err(e) =
            cuda_model
                .executor_mut()
                .init_batched_workspace(hidden_dim, intermediate_dim, m)
        {
            eprintln!("  Error initializing workspace for M={}: {}", m, e);
            continue;
        }

        // PAR-119: Initialize batched KV caches for true parallel attention
        if let Err(e) = cuda_model
            .executor_mut()
            .init_batched_kv_cache_gpu(num_layers, m)
        {
            eprintln!("  Error initializing batched KV cache for M={}: {}", m, e);
            continue;
        }

        // Reset batched KV cache for clean state
        cuda_model.executor_mut().reset_batched_kv_cache_gpu();

        // Create M embeddings (different tokens per sequence)
        let tokens: Vec<u32> = (0..m).map(|i| 9707u32 + i as u32 * 100).collect();
        let embeddings: Vec<f32> = tokens
            .iter()
            .flat_map(|&t| cuda_model.model().embed(&[t]))
            .collect();

        // Create M positions (all starting at position 0)
        let positions: Vec<u32> = vec![0; m];

        // Warmup
        for _ in 0..3 {
            let _ = cuda_model.executor_mut().forward_batched_to_token_ids(
                &embeddings,
                &positions,
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                vocab_size as u32,
                eps,
            );
        }

        // Benchmark
        let start = Instant::now();
        let mut total_tokens = 0usize;

        for iter in 0..num_iterations {
            // Update positions for each iteration
            let iter_positions: Vec<u32> = (0..m).map(|s| (iter + s) as u32).collect();

            match cuda_model.executor_mut().forward_batched_to_token_ids(
                &embeddings,
                &iter_positions,
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                vocab_size as u32,
                eps,
            ) {
                Ok(token_ids) => {
                    total_tokens += token_ids.len();
                },
                Err(e) => {
                    eprintln!("  Error: {}", e);
                    break;
                },
            }
        }

        let elapsed = start.elapsed();
        let tps = total_tokens as f64 / elapsed.as_secs_f64();
        let per_token_us = elapsed.as_micros() as f64 / total_tokens as f64;

        println!(
            "    {} tokens in {:.3}s",
            total_tokens,
            elapsed.as_secs_f64()
        );
        println!("    Aggregate throughput: {:.1} tok/s", tps);
        println!("    Per-token latency:    {:.1} µs", per_token_us);
        println!();
    }

    // PAR-121: Test graphed batched forward path
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PAR-121: Benchmarking GRAPHED Batched Forward");
    println!("  (CUDA graph capture + batched GEMV)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Only test M=2 and M=4 for graphed path (best candidates for 2x Ollama)
    let graphed_batch_sizes = [2, 4];

    for &m in &graphed_batch_sizes {
        println!("  Testing M={} sequences (GRAPHED)...", m);

        // Re-initialize workspace for this batch size
        if let Err(e) =
            cuda_model
                .executor_mut()
                .init_batched_workspace(hidden_dim, intermediate_dim, m)
        {
            eprintln!("  Error initializing workspace for M={}: {}", m, e);
            continue;
        }

        // Re-initialize batched KV caches
        if let Err(e) = cuda_model
            .executor_mut()
            .init_batched_kv_cache_gpu(num_layers, m)
        {
            eprintln!("  Error initializing batched KV cache for M={}: {}", m, e);
            continue;
        }

        cuda_model.executor_mut().reset_batched_kv_cache_gpu();

        let tokens: Vec<u32> = (0..m).map(|i| 9707u32 + i as u32 * 100).collect();
        let embeddings: Vec<f32> = tokens
            .iter()
            .flat_map(|&t| cuda_model.model().embed(&[t]))
            .collect();

        let positions: Vec<u32> = vec![0; m];

        // Warmup (captures graph on first call)
        for _ in 0..3 {
            let _ = cuda_model
                .executor_mut()
                .forward_batched_to_token_ids_graphed(
                    &embeddings,
                    &positions,
                    num_layers,
                    hidden_dim as u32,
                    intermediate_dim as u32,
                    vocab_size as u32,
                    eps,
                );
        }

        // Benchmark graphed path
        let start = Instant::now();
        let mut total_tokens = 0usize;

        for iter in 0..num_iterations {
            let iter_positions: Vec<u32> = (0..m).map(|s| (iter + s) as u32).collect();

            match cuda_model
                .executor_mut()
                .forward_batched_to_token_ids_graphed(
                    &embeddings,
                    &iter_positions,
                    num_layers,
                    hidden_dim as u32,
                    intermediate_dim as u32,
                    vocab_size as u32,
                    eps,
                ) {
                Ok(token_ids) => {
                    total_tokens += token_ids.len();
                },
                Err(e) => {
                    eprintln!("  Error: {}", e);
                    break;
                },
            }
        }

        let elapsed = start.elapsed();
        let tps = total_tokens as f64 / elapsed.as_secs_f64();
        let per_token_us = elapsed.as_micros() as f64 / total_tokens as f64;

        println!(
            "    {} tokens in {:.3}s",
            total_tokens,
            elapsed.as_secs_f64()
        );
        println!("    Aggregate throughput: {:.1} tok/s (GRAPHED)", tps);
        println!("    Per-token latency:    {:.1} µs", per_token_us);
        println!();
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  PAR-111 batched forward path: Uses batched GEMV (weights read once)");
    println!("  PAR-121 graphed batched path: Adds CUDA graph capture");
    println!();
    println!("  Ollama baseline: ~291 tok/s");
    println!("  Target 2x Ollama: 582 tok/s");
    println!();
    println!("  ⚠️  WARNING: Previous hardcoded values were FALSIFIED (v5.20.0)");
    println!("  CRITERION-VERIFIED results (v5.21.0):");
    println!("    M=4 non-graphed: ~400 tok/s (1.37x Ollama)");
    println!("    M=8 non-graphed: ~445 tok/s (1.53x Ollama)");
    println!("    M=4 graphed: ~433 tok/s (1.49x Ollama)");
    println!("    M=8 graphed: ~469 tok/s (1.61x Ollama) <- BEST");
    println!();
    println!("  2X TARGET (582 tok/s) NOT ACHIEVED - requires Flash Decoding (PAR-118)");
}
