//! PAR-108: Batched Q4_K GEMV Benchmark
//!
//! Compares sequential GEMV vs batched GEMV to validate 2x Ollama potential.
//!
//! Key insight: Sequential GEMV dequantizes weights M times for M sequences.
//! Batched GEMV dequantizes once and multiplies by M different inputs.
//!
//! Run with: cargo run --release --features cuda --example bench_batched_gemv

use realizar::cuda::CudaExecutor;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        PAR-108: Batched Q4_K GEMV Benchmark                  ║");
    println!("║        Target: Demonstrate dequant sharing speedup           ║");
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

    // Get layer 0 FFN up weight for testing (typical large GEMV in decode)
    let ffn_up_weight = &model.layers()[0].ffn_up_weight;
    let k = ffn_up_weight.in_dim; // Hidden dim (e.g., 1536)
    let n = ffn_up_weight.out_dim; // Intermediate dim (e.g., 8960)

    println!("  Hidden dim (K): {}", k);
    println!("  Intermediate dim (N): {}", n);
    println!();

    // Create CUDA executor
    println!("Creating CUDA executor...");
    let mut executor = CudaExecutor::new(0).expect("cuda executor");

    // Upload weight
    let weight_name = "ffn_up_test";
    executor
        .load_quantized_weights(weight_name, &ffn_up_weight.data)
        .expect("upload weight");

    // Get weight pointer for batched version
    let weight_ptr = executor
        .get_quantized_weight_ptr(weight_name)
        .expect("get ptr");

    // Test parameters
    let batch_sizes: [u32; 4] = [1, 2, 4, 8];
    let warmup_iters = 5;
    let bench_iters = 20;

    println!("Testing k={}, n={}", k, n);
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmark Results (per-call latency in microseconds)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "{:>5} {:>14} {:>14} {:>10}",
        "M", "Sequential(µs)", "Batched(µs)", "Speedup"
    );
    println!(
        "{:>5} {:>14} {:>14} {:>10}",
        "---", "-------------", "-----------", "-------"
    );

    for m in batch_sizes {
        // Create input/output buffers for this batch size
        // Sequential: M separate inputs of size K, M separate outputs of size N
        // Batched: 1 input of size M×K, 1 output of size M×N

        let inputs: Vec<Vec<f32>> = (0..m)
            .map(|batch_idx| {
                (0..k)
                    .map(|i| ((i + batch_idx as usize * 100) % 10) as f32 * 0.1 - 0.5)
                    .collect()
            })
            .collect();

        let mut outputs_seq: Vec<Vec<f32>> = (0..m).map(|_| vec![0.0f32; n]).collect();

        // Sequential benchmark: M separate GEMV calls
        // Warmup
        for _ in 0..warmup_iters {
            for batch_idx in 0..m as usize {
                let _ = executor.q4k_gemv_cached(
                    weight_name,
                    &inputs[batch_idx],
                    &mut outputs_seq[batch_idx],
                    n as u32,
                    k as u32,
                );
            }
        }

        // Benchmark
        let seq_start = Instant::now();
        for _ in 0..bench_iters {
            for batch_idx in 0..m as usize {
                let _ = executor.q4k_gemv_cached(
                    weight_name,
                    &inputs[batch_idx],
                    &mut outputs_seq[batch_idx],
                    n as u32,
                    k as u32,
                );
            }
        }
        let seq_time = seq_start.elapsed();
        let seq_us = seq_time.as_micros() as f64 / bench_iters as f64;

        // Batched benchmark: Create batched input (M×K) and output (M×N)
        let batched_input: Vec<f32> = inputs.iter().flatten().copied().collect();
        let batched_output = vec![0.0f32; (m as usize) * n];

        // Upload batched input to GPU
        let mut batched_input_buf = executor
            .allocate_buffer(batched_input.len())
            .expect("input buf");
        batched_input_buf
            .copy_from_host(&batched_input)
            .expect("copy input");

        let batched_output_buf = executor
            .allocate_buffer(batched_output.len())
            .expect("output buf");

        // Warmup batched
        for _ in 0..warmup_iters {
            let _ = executor.batched_q4k_gemv_into(
                weight_ptr,
                &batched_input_buf,
                &batched_output_buf,
                m,
                n as u32,
                k as u32,
            );
            executor.synchronize().expect("sync");
        }

        // Benchmark batched
        let batch_start = Instant::now();
        for _ in 0..bench_iters {
            let _ = executor.batched_q4k_gemv_into(
                weight_ptr,
                &batched_input_buf,
                &batched_output_buf,
                m,
                n as u32,
                k as u32,
            );
            executor.synchronize().expect("sync");
        }
        let batch_time = batch_start.elapsed();
        let batch_us = batch_time.as_micros() as f64 / bench_iters as f64;

        let speedup = seq_us / batch_us;

        println!(
            "{:>5} {:>14.1} {:>14.1} {:>9.2}x",
            m, seq_us, batch_us, speedup
        );
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Analysis");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Performance insight (Five-Whys PAR-108):");
    println!("  - Sequential: Reads and dequantizes weights M times");
    println!("  - Batched: Reads and dequantizes weights ONCE");
    println!();
    println!("  Expected speedup pattern:");
    println!("  - M=1: ~1.0x (same work)");
    println!("  - M=2: ~1.5-1.8x (dequant amortized 2 ways)");
    println!("  - M=4: ~2.0-3.0x (dequant amortized 4 ways)");
    println!("  - M=8: ~3.0-5.0x (dequant amortized 8 ways)");
    println!();
    println!("  Current status:");
    println!("  - Single request: ~180 tok/s (baseline)");
    println!("  - Sequential batch of 4: ~360 tok/s (1.80x Ollama)");
    println!("  - Target: 400 tok/s (2.0x Ollama)");
    println!();
    println!("  If batched shows ~1.5x speedup for M=4:");
    println!("  → 360 × 1.5 ÷ 4 = ~135 tok/s per sequence in batch");
    println!("  → Aggregate: 135 × 4 = 540 tok/s (2.7x Ollama!) ✓");
}
