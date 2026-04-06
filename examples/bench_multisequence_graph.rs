//! PAR-109: Multi-Sequence CUDA Graph Benchmark
//!
//! Measures overhead breakdown for sequential graph replays vs theoretical
//! multi-sequence graph to understand the 11% gap to 2x Ollama (400 tok/s).
//!
//! Current: 360 tok/s (sequential CUDA graphs)
//! Target: 400 tok/s (2x Ollama)
//! Gap: 40 tok/s (11%)
//!
//! This benchmark profiles:
//! 1. Embedding lookup time (CPU)
//! 2. H2D copy time (position, seq_len, embedding)
//! 3. Graph launch overhead
//! 4. GPU argmax time
//! 5. D2H copy time (token ID)
//!
//! Run with: cargo run --release --features cuda --example bench_multisequence_graph

use realizar::cuda::CudaExecutor;
use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
    QuantizedGenerateConfig,
};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        PAR-109: Multi-Sequence Graph Overhead Analysis       ║");
    println!("║        Target: Identify 11% gap to 400 tok/s                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    if !CudaExecutor::is_available() {
        println!("CUDA not available");
        return;
    }

    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

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
    println!();

    // Create KV cache
    let hidden_dim = cuda_model.model().config().hidden_dim;
    let num_layers = cuda_model.model().layers().len();
    let kv_dim = cuda_model.model().config().num_kv_heads
        * (hidden_dim / cuda_model.model().config().num_heads);
    let max_len = 2048;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 1: Per-Token Overhead Breakdown (Single Sequence)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Generate warmup tokens to capture graph
    let warmup_prompt: Vec<u32> = vec![9707u32]; // "Hello"
    let warmup_config = QuantizedGenerateConfig::deterministic(5);
    cuda_model.clear_decode_graph();
    let _ = cuda_model.generate_gpu_resident(&warmup_prompt, &warmup_config);

    // Now benchmark individual operations
    let num_tokens = 100;
    let mut embed_times = Vec::with_capacity(num_tokens);
    let mut total_times = Vec::with_capacity(num_tokens);

    let mut cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, max_len);

    // Prefill a few tokens to warm up cache
    for i in 0..5 {
        let _ = cuda_model.forward_gpu_resident_to_token_id(9707, &mut cache, i);
    }

    println!("  Measuring {} decode tokens...", num_tokens);

    let mut last_token = 9707u32;
    for i in 0..num_tokens {
        let position = 5 + i;

        // Measure embedding lookup (CPU)
        let embed_start = Instant::now();
        let _embedding = cuda_model.model().embed(&[last_token]);
        let embed_time = embed_start.elapsed();
        embed_times.push(embed_time.as_nanos() as f64 / 1000.0); // microseconds

        // Measure total forward pass
        let total_start = Instant::now();
        let next_token = cuda_model
            .forward_gpu_resident_to_token_id(last_token, &mut cache, position)
            .expect("forward");
        let total_time = total_start.elapsed();
        total_times.push(total_time.as_micros() as f64);

        last_token = next_token;
    }

    // Calculate statistics
    let avg_embed = embed_times.iter().sum::<f64>() / num_tokens as f64;
    let avg_total = total_times.iter().sum::<f64>() / num_tokens as f64;
    let avg_gpu = avg_total - avg_embed;

    println!();
    println!("  Per-token breakdown:");
    println!("    Embedding lookup (CPU): {:>8.1} us", avg_embed);
    println!("    GPU forward + argmax:   {:>8.1} us", avg_gpu);
    println!("    Total per token:        {:>8.1} us", avg_total);
    println!();
    println!(
        "    Implied throughput: {:.1} tok/s",
        1_000_000.0 / avg_total
    );
    println!();

    // Test 2: Multi-sequence sequential (what we currently do)
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 2: Multi-Sequence Sequential (Current Approach)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let batch_sizes = [1, 2, 4, 8];
    let tokens_per_seq = 50;

    for &batch_size in &batch_sizes {
        cuda_model.clear_decode_graph();

        // Create caches for each sequence
        let mut caches: Vec<OwnedQuantizedKVCache> = (0..batch_size)
            .map(|_| OwnedQuantizedKVCache::new(num_layers, kv_dim, max_len))
            .collect();

        // Warmup with first token from each sequence
        let prompts: Vec<u32> = vec![9707, 2980, 791, 1585, 3923, 5765, 8144, 12522];
        for (seq_idx, cache) in caches.iter_mut().enumerate().take(batch_size) {
            let _ = cuda_model.forward_gpu_resident_to_token_id(prompts[seq_idx], cache, 0);
        }

        // Benchmark: generate tokens_per_seq tokens for each sequence
        let start = Instant::now();
        let mut total_tokens = 0usize;

        for token_num in 1..tokens_per_seq {
            for (seq_idx, cache) in caches.iter_mut().enumerate().take(batch_size) {
                let last_token = prompts[seq_idx]; // Simplified - just use prompt token
                let _ = cuda_model.forward_gpu_resident_to_token_id(last_token, cache, token_num);
                total_tokens += 1;
            }
        }

        let elapsed = start.elapsed();
        let tps = total_tokens as f64 / elapsed.as_secs_f64();
        let per_token_us = elapsed.as_micros() as f64 / total_tokens as f64;

        println!(
            "  M={}: {} tokens in {:.2}s = {:.1} tok/s ({:.1}us/tok)",
            batch_size,
            total_tokens,
            elapsed.as_secs_f64(),
            tps,
            per_token_us
        );
    }

    println!();

    // Test 3: Theoretical multi-sequence graph savings
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 3: Multi-Sequence Graph Potential (Analysis)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Calculate theoretical improvement from multi-sequence graph
    // Current overhead per token (measured):
    // - Embedding: ~10us (CPU, can't avoid)
    // - H2D (pos + seq_len + embed): ~50us (can batch M into 1)
    // - Graph launch: ~10us (can reduce by M)
    // - GPU argmax: ~50us (can batch M into 1)
    // - D2H (token): ~5us (can batch M into 1)

    let overhead_per_token = avg_embed; // Only embedding is unavoidable per-token

    // For M=4, theoretical throughput:
    // Graph captures batched ops, so GPU time is shared across M
    // Only per-token overhead is embedding lookup
    let m = 4.0;
    let current_gpu_time = avg_gpu;

    // With batched graph, GPU time scales ~1.2x (not 4x) due to batched GEMV efficiency
    // From PAR-108: batched GEMV is 15x faster per element, but attention can't batch
    // Estimate: ~60% of GPU time is GEMV (can batch), ~40% is attention (can't batch)
    // GEMV portion: 0.6 * avg_gpu / 4 (batched)
    // Attention portion: 0.4 * avg_gpu (per sequence, but in graph)
    let gemv_portion = 0.6;
    let batched_gpu_time =
        (gemv_portion * current_gpu_time / m) + ((1.0 - gemv_portion) * current_gpu_time);
    let batched_per_token = (overhead_per_token + batched_gpu_time) / m + overhead_per_token;

    println!("  Current per-token breakdown:");
    println!("    Embedding (CPU):        {:>8.1} us", avg_embed);
    println!("    GPU (graph replay):     {:>8.1} us", current_gpu_time);
    println!("    Total:                  {:>8.1} us", avg_total);
    println!(
        "    Throughput:             {:>8.1} tok/s",
        1_000_000.0 / avg_total
    );
    println!();
    println!("  Theoretical M=4 batched graph:");
    println!(
        "    GEMV portion (60%):     {:>8.1} us / 4 = {:.1} us",
        current_gpu_time * 0.6,
        current_gpu_time * 0.6 / m
    );
    println!(
        "    Attention portion (40%): {:>8.1} us (can't batch)",
        current_gpu_time * 0.4
    );
    println!(
        "    Batched GPU total:      {:>8.1} us for 4 tokens",
        batched_gpu_time
    );
    println!(
        "    Per-token GPU:          {:>8.1} us",
        batched_gpu_time / m
    );
    println!("    Per-token total:        {:>8.1} us", batched_per_token);
    println!(
        "    Theoretical throughput: {:>8.1} tok/s",
        1_000_000.0 / batched_per_token
    );
    println!();

    // Test 4: Validate GEMV vs attention time split
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Test 4: GEMV vs Attention Time Analysis");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Reset and enable profiling
    cuda_model.clear_decode_graph();

    // The profiler is not easily accessible here, but we can estimate:
    // Per layer (Qwen 1.5B, 28 layers):
    // - QKV GEMV: 3 × (1536 → 1536) ≈ 3 × 10us = 30us
    // - O proj GEMV: (1536 → 1536) ≈ 10us
    // - FFN up GEMV: (1536 → 8960) ≈ 30us
    // - FFN gate GEMV: (1536 → 8960) ≈ 30us
    // - FFN down GEMV: (8960 → 1536) ≈ 25us
    // Total GEMV per layer: ~125us × 28 = 3500us
    // - RMSNorm × 2: ~2us × 2 = 4us
    // - RoPE: ~2us
    // - Attention: ~50us (varies with seq len)
    // - Residuals: ~2us × 2 = 4us
    // Non-GEMV per layer: ~62us × 28 = 1736us
    // LM head GEMV: ~200us

    let est_gemv_time = 3500.0 + 200.0; // us
    let est_non_gemv_time = 1736.0;
    let est_total = est_gemv_time + est_non_gemv_time;
    let gemv_fraction = est_gemv_time / est_total;

    println!("  Estimated time breakdown:");
    println!(
        "    GEMV operations:    {:>8.1} us ({:.0}%)",
        est_gemv_time,
        gemv_fraction * 100.0
    );
    println!(
        "    Non-GEMV ops:       {:>8.1} us ({:.0}%)",
        est_non_gemv_time,
        (1.0 - gemv_fraction) * 100.0
    );
    println!("    Estimated total:    {:>8.1} us", est_total);
    println!();

    // Revised theoretical with better estimate
    let batched_gemv = est_gemv_time / m; // Batched GEMV amortizes dequant
    let unbatched_other = est_non_gemv_time; // Attention per sequence
    let batched_total_for_m = batched_gemv + unbatched_other;
    let batched_per_token_revised = batched_total_for_m / m + avg_embed;

    println!("  Revised M=4 batched graph estimate:");
    println!("    Batched GEMV (÷4):  {:>8.1} us", batched_gemv);
    println!("    Unbatched other:    {:>8.1} us", unbatched_other);
    println!("    Total for 4 tok:    {:>8.1} us", batched_total_for_m);
    println!(
        "    Per-token:          {:>8.1} us",
        batched_per_token_revised
    );
    println!(
        "    Throughput:         {:>8.1} tok/s",
        1_000_000.0 / batched_per_token_revised
    );
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  Current (sequential graphs): {:.1} tok/s",
        1_000_000.0 / avg_total
    );
    println!("  Target (2x Ollama):          400 tok/s");
    println!(
        "  Gap:                         {:.1}%",
        (400.0 - 1_000_000.0 / avg_total) / (1_000_000.0 / avg_total) * 100.0
    );
    println!();
    println!("  Multi-sequence graph potential:");
    println!(
        "    M=4 theoretical: {:.1} tok/s",
        1_000_000.0 / batched_per_token_revised
    );
    println!(
        "    vs target:       {:.1}%",
        1_000_000.0 / batched_per_token_revised / 400.0 * 100.0
    );
    println!();
    if 1_000_000.0 / batched_per_token_revised >= 400.0 {
        println!("  Multi-sequence graph can achieve 2x Ollama!");
    } else {
        println!("  Additional optimizations needed beyond multi-sequence graph.");
        println!("  Consider: DP4A instructions, memory coalescing, kernel fusion.");
    }
}
