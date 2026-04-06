//! Profile cached forward pass to find hot paths
//!
//! Run with: cargo run --release --example profile_cached_forward -- <model.gguf>

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};
use std::{env, time::Instant};

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map_or(
        "/home/noah/.cache/gguf/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        |s| s.as_str(),
    );

    println!("=== Cached Forward Pass Profiling ===\n");

    // Load model
    let start = Instant::now();
    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let load_time = start.elapsed();

    let config = model.config();
    let model_name = path.split('/').next_back().unwrap_or(path);

    println!("Model: {}", model_name);
    println!("Load time: {:.2?}", load_time);
    println!(
        "Config: {} layers, {} hidden, {} heads, {} kv_heads",
        config.num_layers, config.hidden_dim, config.num_heads, config.num_kv_heads
    );
    println!(
        "Intermediate: {}, Vocab: {}",
        config.intermediate_dim, config.vocab_size
    );
    println!();

    // Create KV cache
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;
    let max_seq_len = 128;
    let mut cache = OwnedQuantizedKVCache::new(config.num_layers, kv_dim, max_seq_len);

    // Warmup
    println!("Warming up (10 tokens)...");
    for pos in 0..10 {
        let _ = model.forward_single_with_cache(1, &mut cache, pos);
    }

    // Reset cache for profiling
    cache = OwnedQuantizedKVCache::new(config.num_layers, kv_dim, max_seq_len);

    // Profile individual token generation
    let num_profile_tokens = 50;
    let mut token_times = Vec::with_capacity(num_profile_tokens);

    println!("\nProfiling {} tokens...", num_profile_tokens);
    for pos in 0..num_profile_tokens {
        let start = Instant::now();
        let _ = model.forward_single_with_cache(1, &mut cache, pos);
        token_times.push(start.elapsed());
    }

    // Analyze timing
    println!("\n=== Per-Token Latency ===");
    println!("Position | Time (ms) | Cumulative");
    println!("---------+-----------+-----------");

    let mut cumulative = std::time::Duration::ZERO;
    for (i, &time) in token_times.iter().enumerate() {
        cumulative += time;
        if i < 10 || i == num_profile_tokens - 1 || i % 10 == 9 {
            println!(
                "{:>8} | {:>9.2} | {:>9.2}",
                i,
                time.as_secs_f64() * 1000.0,
                cumulative.as_secs_f64() * 1000.0
            );
        }
    }

    // Statistics
    let times_us: Vec<u128> = token_times.iter().map(|t| t.as_micros()).collect();
    let min = *times_us.iter().min().expect("test");
    let max = *times_us.iter().max().expect("test");
    let sum: u128 = times_us.iter().sum();
    let avg = sum / times_us.len() as u128;

    let mut sorted = times_us;
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];
    let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];

    println!("\n=== Latency Statistics ===");
    println!("Min:    {:>8} µs ({:.2} ms)", min, min as f64 / 1000.0);
    println!(
        "Median: {:>8} µs ({:.2} ms)",
        median,
        median as f64 / 1000.0
    );
    println!("Avg:    {:>8} µs ({:.2} ms)", avg, avg as f64 / 1000.0);
    println!("P95:    {:>8} µs ({:.2} ms)", p95, p95 as f64 / 1000.0);
    println!("Max:    {:>8} µs ({:.2} ms)", max, max as f64 / 1000.0);

    let tok_per_sec = 1_000_000.0 / median as f64;
    println!("\nThroughput: {:.1} tok/s (median)", tok_per_sec);

    // Memory bandwidth analysis
    println!("\n=== Memory Bandwidth Analysis ===");

    // Q4_0: 0.5 bytes per weight (4 bits)
    let bytes_per_weight = 0.5_f64;
    let hidden = config.hidden_dim as f64;
    let layers = config.num_layers as f64;
    let inter = config.intermediate_dim as f64;
    let vocab = config.vocab_size as f64;
    let kv_heads = config.num_kv_heads as f64;
    let num_heads = config.num_heads as f64;
    let head_d = hidden / num_heads;

    // Per-layer weight sizes
    // QKV: hidden -> (hidden + 2*kv_dim) for GQA
    let qkv_weights = hidden * (hidden + 2.0 * kv_heads * head_d) * bytes_per_weight;
    let out_weights = hidden * hidden * bytes_per_weight;

    // FFN: Check if SwiGLU (3 matrices) or GELU (2 matrices)
    // TinyLlama uses SwiGLU: up, gate, down
    let ffn_weights = 3.0 * hidden * inter * bytes_per_weight; // SwiGLU

    let layer_weights = qkv_weights + out_weights + ffn_weights;
    let total_layer_weights = layer_weights * layers;
    let lm_head_weights = hidden * vocab * bytes_per_weight;
    let _embed_weights = vocab * hidden * bytes_per_weight;
    let total_weights = total_layer_weights + lm_head_weights; // embed often shared

    println!("Weight sizes (Q4_0):");
    println!("  Per layer:");
    println!("    QKV:    {:.2} MB", qkv_weights / 1e6);
    println!("    Out:    {:.2} MB", out_weights / 1e6);
    println!("    FFN:    {:.2} MB", ffn_weights / 1e6);
    println!("    Total:  {:.2} MB", layer_weights / 1e6);
    println!(
        "  All {} layers: {:.2} MB",
        layers as i32,
        total_layer_weights / 1e6
    );
    println!("  LM head:      {:.2} MB", lm_head_weights / 1e6);
    println!("  Total:        {:.2} MB", total_weights / 1e6);

    // Estimate memory bandwidth
    let actual_time_s = median as f64 / 1_000_000.0;
    let bytes_read = total_weights;
    let bandwidth_gbps = bytes_read / 1e9 / actual_time_s;

    println!("\nBandwidth utilization:");
    println!("  Bytes read per token: {:.2} MB", bytes_read / 1e6);
    println!("  Time per token:       {:.2} ms", actual_time_s * 1000.0);
    println!("  Achieved bandwidth:   {:.1} GB/s", bandwidth_gbps);

    // Compare to theoretical
    let ddr4_bandwidth = 50.0; // ~50 GB/s for DDR4-3200
    let ddr5_bandwidth = 80.0; // ~80 GB/s for DDR5
    println!(
        "\n  vs DDR4 (50 GB/s):    {:.1}%",
        100.0 * bandwidth_gbps / ddr4_bandwidth
    );
    println!(
        "  vs DDR5 (80 GB/s):    {:.1}%",
        100.0 * bandwidth_gbps / ddr5_bandwidth
    );

    // Theoretical minimum latency
    let theoretical_min_ms = (bytes_read / 1e9) / ddr4_bandwidth * 1000.0;
    let overhead_pct =
        (actual_time_s * 1000.0 - theoretical_min_ms) / (actual_time_s * 1000.0) * 100.0;

    println!("\nRoofline (@ 50 GB/s DDR4):");
    println!("  Theoretical min:  {:.2} ms", theoretical_min_ms);
    println!("  Actual:           {:.2} ms", actual_time_s * 1000.0);
    println!("  Overhead:         {:.1}%", overhead_pct);

    if overhead_pct > 50.0 {
        println!("\n=== Bottleneck Analysis ===");
        println!("High overhead ({:.0}%) suggests:", overhead_pct);
        println!("  - Compute-bound operations (attention, softmax)");
        println!("  - Thread synchronization overhead (Rayon)");
        println!("  - Cache misses on weight access");
        println!("  - Memory allocation in hot paths");
    }

    // Scaling analysis
    println!("\n=== Scaling with Cache Length ===");
    println!("Attention cost scales O(seq_len) per token");

    let first_10_avg: f64 = token_times[0..10]
        .iter()
        .map(|t| t.as_secs_f64())
        .sum::<f64>()
        / 10.0;
    let last_10_avg: f64 = token_times[40..50]
        .iter()
        .map(|t| t.as_secs_f64())
        .sum::<f64>()
        / 10.0;
    let scaling = last_10_avg / first_10_avg;

    println!("  First 10 tokens avg: {:.2} ms", first_10_avg * 1000.0);
    println!("  Last 10 tokens avg:  {:.2} ms", last_10_avg * 1000.0);
    println!("  Scaling factor:      {:.2}x", scaling);

    if scaling > 1.5 {
        println!("\n  Attention is a significant cost - consider FlashAttention");
    } else {
        println!("\n  Attention scales well - matmul is likely the bottleneck");
    }
}
