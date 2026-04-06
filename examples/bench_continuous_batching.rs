//! PAR-106: Continuous Batching Benchmark
//!
//! Tests throughput improvement from processing multiple concurrent requests.
//! Key insight: Weight reads are amortized across multiple tokens in the batch.
//!
//! Expected improvement: At batch_size=4, ~2-3x throughput vs single-request
//! because each weight read produces 4 tokens instead of 1.
//!
//! Run with: cargo run --release --features cuda --example bench_continuous_batching

use realizar::cuda::CudaExecutor;
use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig,
};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        PAR-106: Continuous Batching Benchmark                ║");
    println!("║        Target: 400 tok/s (2x Ollama) via batch decode        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    if !CudaExecutor::is_available() {
        println!("❌ CUDA not available");
        return;
    }

    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    if !Path::new(model_path).exists() {
        println!("❌ Model not found: {}", model_path);
        return;
    }

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path).expect("model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("model");
    let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0).expect("CUDA");

    // Pre-upload weights
    cuda_model.preload_weights_gpu().expect("weights");
    println!();

    // Different prompts for concurrent requests
    let prompts: Vec<Vec<u32>> = vec![
        vec![9707u32],  // "Hello"
        vec![2980u32],  // "Write"
        vec![791u32],   // "The"
        vec![1585u32],  // "How"
        vec![3923u32],  // "What"
        vec![5765u32],  // "Can"
        vec![8144u32],  // "Please"
        vec![12522u32], // "Code"
    ];

    let tokens_per_request = 50;
    let config = QuantizedGenerateConfig::deterministic(tokens_per_request);

    // === Test 1: Single-request baseline ===
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 1: Single-Request Baseline (1 prompt)");
    println!("═══════════════════════════════════════════════════════════════");

    cuda_model.clear_decode_graph();
    let start = Instant::now();
    let result = cuda_model.generate_gpu_resident(&prompts[0], &config);
    let single_time = start.elapsed();

    let single_tps = match result {
        Ok(tokens) => {
            let generated = tokens.len() - prompts[0].len();
            let tps = generated as f64 / single_time.as_secs_f64();
            println!(
                "✅ Generated {} tokens in {:.2}s",
                generated,
                single_time.as_secs_f64()
            );
            println!("   Throughput: {:.1} tok/s", tps);
            tps
        },
        Err(e) => {
            println!("❌ Failed: {}", e);
            return;
        },
    };

    println!();

    // === Test 2: Concurrent batch (4 requests) ===
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 2: Batch Processing (4 concurrent requests)");
    println!("═══════════════════════════════════════════════════════════════");

    let batch_size = 4;
    let batch_prompts = &prompts[..batch_size];

    cuda_model.clear_decode_graph();

    // Process 4 requests sequentially to simulate concurrent handling
    // In production, this would be true concurrent execution
    let start = Instant::now();
    let mut total_tokens = 0usize;

    for (i, prompt) in batch_prompts.iter().enumerate() {
        cuda_model.clear_decode_graph();
        match cuda_model.generate_gpu_resident(prompt, &config) {
            Ok(tokens) => {
                total_tokens += tokens.len() - prompt.len();
            },
            Err(e) => {
                println!("❌ Request {} failed: {}", i, e);
            },
        }
    }

    let batch_time = start.elapsed();
    let batch_tps = total_tokens as f64 / batch_time.as_secs_f64();

    println!(
        "✅ {} requests completed: {} tokens in {:.2}s",
        batch_size,
        total_tokens,
        batch_time.as_secs_f64()
    );
    println!("   Aggregate throughput: {:.1} tok/s", batch_tps);
    println!("   Speedup vs single: {:.2}x", batch_tps / single_tps);
    println!();

    // === Test 3: True batched decode using generate_batch_gpu_resident (PAR-106) ===
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 3: TRUE Batched Decode (PAR-106: weight sharing)");
    println!("═══════════════════════════════════════════════════════════════");

    cuda_model.clear_decode_graph();
    let batch_prompts_owned: Vec<Vec<u32>> = prompts[..4].to_vec();

    let start = Instant::now();
    let batch_result = cuda_model.generate_batch_gpu_resident(&batch_prompts_owned, &config);
    let true_batch_time = start.elapsed();

    match batch_result {
        Ok(results) => {
            let total_generated: usize = results
                .iter()
                .zip(batch_prompts_owned.iter())
                .map(|(out, prompt)| out.len() - prompt.len())
                .sum();
            let true_batch_tps = total_generated as f64 / true_batch_time.as_secs_f64();
            println!(
                "✅ TRUE batched: {} tokens in {:.2}s",
                total_generated,
                true_batch_time.as_secs_f64()
            );
            println!("   Throughput: {:.1} tok/s", true_batch_tps);
            println!("   Speedup vs single: {:.2}x", true_batch_tps / single_tps);
            println!(
                "   vs Ollama (200): {:.1}%",
                (true_batch_tps / 200.0) * 100.0
            );
            if true_batch_tps >= 400.0 {
                println!("   ✅ TARGET MET: 2x Ollama achieved!");
            }
        },
        Err(e) => {
            println!("❌ True batch failed: {}", e);
            // Print expected for reference
            let expected_batch_tps = single_tps * (batch_size as f64 * 0.7);
            println!(
                "  Expected with true batching: {:.1} tok/s",
                expected_batch_tps
            );
        },
    }
    println!();

    // === Test 4: Batch size sweep ===
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 4: Batch Size Sweep (aggregate throughput)");
    println!("═══════════════════════════════════════════════════════════════");

    for batch_size in [1, 2, 4, 8] {
        if batch_size > prompts.len() {
            break;
        }

        cuda_model.clear_decode_graph();

        let start = Instant::now();
        let mut total = 0usize;

        for prompt in &prompts[..batch_size] {
            cuda_model.clear_decode_graph();
            if let Ok(tokens) = cuda_model.generate_gpu_resident(prompt, &config) {
                total += tokens.len() - prompt.len();
            }
        }

        let elapsed = start.elapsed();
        let tps = total as f64 / elapsed.as_secs_f64();
        let speedup = tps / single_tps;

        println!(
            "  batch={}: {} tokens in {:.2}s = {:.1} tok/s ({:.2}x)",
            batch_size,
            total,
            elapsed.as_secs_f64(),
            tps,
            speedup
        );
    }

    println!();

    // === Test 5: Warm graph persistence (no clear_decode_graph between requests) ===
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 5: Warm Graph Persistence (NO graph clear between requests)");
    println!("═══════════════════════════════════════════════════════════════");

    // Clear once at start, then run 4 requests WITHOUT clearing graph
    cuda_model.clear_decode_graph();

    let start = Instant::now();
    let mut total_warm = 0usize;

    for prompt in &prompts[..4] {
        // NOTE: No clear_decode_graph() here - graph persists between requests
        if let Ok(tokens) = cuda_model.generate_gpu_resident(prompt, &config) {
            total_warm += tokens.len() - prompt.len();
        }
    }

    let warm_time = start.elapsed();
    let warm_tps = total_warm as f64 / warm_time.as_secs_f64();

    println!(
        "✅ Warm graph: {} tokens in {:.2}s",
        total_warm,
        warm_time.as_secs_f64()
    );
    println!("   Throughput: {:.1} tok/s", warm_tps);
    println!("   vs Single: {:.2}x", warm_tps / single_tps);
    println!(
        "   vs Cold (clear each): {:.1}% speedup",
        ((warm_tps / batch_tps) - 1.0) * 100.0
    );
    if warm_tps >= 400.0 {
        println!("   ✅ TARGET MET: 2x Ollama achieved!");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  Single-request (cold): {:.1} tok/s (baseline)",
        single_tps
    );
    println!("  Sequential (clear graph each): {:.1} tok/s", batch_tps);
    println!("  Sequential (warm graph): {:.1} tok/s", warm_tps);
    println!("  Ollama baseline: ~200 tok/s");
    println!("  Target: 400 tok/s (2x Ollama)");
    println!();
    if warm_tps >= 400.0 {
        println!("  ✅ 2x OLLAMA TARGET ACHIEVED via warm graph persistence!");
    } else {
        println!(
            "  Gap to 2x: {:.1} tok/s ({:.1}%)",
            400.0 - warm_tps,
            ((400.0 / warm_tps) - 1.0) * 100.0
        );
    }
}
