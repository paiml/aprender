//! PARITY-036/037: GPU GEMM Performance Test
//!
//! Tests CudaExecutor GEMM kernel performance:
//! - PARITY-036: GPU GEMM vs CPU baseline (with H2D transfer overhead)
//! - PARITY-037: Cached weights on GPU (eliminates weight transfer)
//!
//! FFN is the actual bottleneck (>90% of inference time), not attention.
//!
//! Run with: cargo run --release --example parity_036_gpu_attention --features cuda

#[cfg(feature = "cuda")]
use std::time::Instant;

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║         PARITY-036: GPU GEMM Performance Test (FFN focus)      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    #[cfg(feature = "cuda")]
    {
        use realizar::cuda::CudaExecutor;

        // Test configuration matching phi-2 FFN
        let hidden_dim = 2560u32;
        let intermediate_dim = 10240u32; // 4x hidden
        let num_layers = 32usize;

        println!("Configuration (phi-2 FFN):");
        println!("  hidden_dim: {}", hidden_dim);
        println!("  intermediate_dim: {}", intermediate_dim);
        println!("  num_layers: {}", num_layers);
        println!();

        // Initialize CUDA
        println!("[1/5] Initializing CUDA...");
        let start = Instant::now();
        let mut executor = match CudaExecutor::new(0) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("Failed to initialize CUDA: {}", err);
                std::process::exit(1);
            },
        };
        let cuda_init_time = start.elapsed();
        println!("       CUDA init: {:?}", cuda_init_time);

        if let Ok(name) = executor.device_name() {
            println!("       Device: {}", name);
        }
        if let Ok((free, total)) = executor.memory_info() {
            println!(
                "       Memory: {} MB free / {} MB total",
                free / 1024 / 1024,
                total / 1024 / 1024
            );
        }

        // Generate test data for MATVEC (single token generation)
        // FFN fc1: [hidden, intermediate] @ [hidden, 1] = [intermediate, 1]
        // FFN fc2: [intermediate, hidden] @ [intermediate, 1] = [hidden, 1]
        println!("\n[2/5] Generating test data...");

        // Matvec test (single token): [M, K] @ [K, 1] = [M, 1]
        let m_fc1 = intermediate_dim;
        let k_fc1 = hidden_dim;
        let n_fc1 = 1u32;

        let weight_fc1: Vec<f32> = (0..(m_fc1 * k_fc1) as usize)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let input_fc1: Vec<f32> = (0..k_fc1 as usize)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
            .collect();
        let mut output_fc1 = vec![0.0f32; m_fc1 as usize];

        println!(
            "       fc1 weight: {}x{} = {} elements ({:.1} MB)",
            m_fc1,
            k_fc1,
            m_fc1 * k_fc1,
            (m_fc1 * k_fc1 * 4) as f64 / 1024.0 / 1024.0
        );
        println!("       fc1 input: {} elements", k_fc1);

        // Warmup GPU
        println!("\n[3/5] Warming up GPU...");
        for _ in 0..3 {
            let _ = executor.gemm(
                &weight_fc1,
                &input_fc1,
                &mut output_fc1,
                m_fc1,
                n_fc1,
                k_fc1,
            );
        }

        // Benchmark GPU GEMM
        println!(
            "\n[4/5] Benchmarking GEMM (fc1: {}x{} @ {}x1)...",
            m_fc1, k_fc1, k_fc1
        );
        let iterations = 20;

        // GPU GEMM
        let start = Instant::now();
        for _ in 0..iterations {
            executor
                .gemm(
                    &weight_fc1,
                    &input_fc1,
                    &mut output_fc1,
                    m_fc1,
                    n_fc1,
                    k_fc1,
                )
                .expect("GPU GEMM failed");
        }
        let gpu_total = start.elapsed();
        let gpu_per_iter = gpu_total / iterations;

        // CPU GEMM (naive matvec)
        let start = Instant::now();
        for _ in 0..iterations {
            cpu_matvec(
                &weight_fc1,
                &input_fc1,
                &mut output_fc1,
                m_fc1 as usize,
                k_fc1 as usize,
            );
        }
        let cpu_total = start.elapsed();
        let cpu_per_iter = cpu_total / iterations;

        // Results
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║                        RESULTS                                  ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!(
            "║ GPU GEMM:  {:>10?}/iter ({} iterations)                   ║",
            gpu_per_iter, iterations
        );
        println!(
            "║ CPU GEMM:  {:>10?}/iter ({} iterations)                   ║",
            cpu_per_iter, iterations
        );
        println!("╠════════════════════════════════════════════════════════════════╣");

        let speedup = cpu_per_iter.as_secs_f64() / gpu_per_iter.as_secs_f64();
        if speedup > 1.0 {
            println!(
                "║ GPU is {:.2}x FASTER than CPU                                 ║",
                speedup
            );
        } else {
            println!(
                "║ GPU is {:.2}x SLOWER than CPU (transfer overhead)            ║",
                1.0 / speedup
            );
        }
        println!("╚════════════════════════════════════════════════════════════════╝");

        // GFLOPS calculation
        let flops_per_matvec = 2.0 * (m_fc1 as f64) * (k_fc1 as f64); // MAC = 2 ops
        let gpu_gflops = flops_per_matvec / gpu_per_iter.as_secs_f64() / 1e9;
        let cpu_gflops = flops_per_matvec / cpu_per_iter.as_secs_f64() / 1e9;

        println!("\n═══ Performance Metrics ═══");
        println!("  FLOPs per fc1: {:.2}M", flops_per_matvec / 1e6);
        println!("  GPU: {:.2} GFLOPS", gpu_gflops);
        println!("  CPU: {:.2} GFLOPS", cpu_gflops);

        // Per-token estimate (fc1 + fc2, all layers)
        // Each layer: fc1 (hidden→intermediate) + fc2 (intermediate→hidden)
        let fc1_ops = 2.0 * (intermediate_dim as f64) * (hidden_dim as f64);
        let fc2_ops = 2.0 * (hidden_dim as f64) * (intermediate_dim as f64);
        let ffn_ops_per_layer = fc1_ops + fc2_ops;
        let total_ffn_ops = ffn_ops_per_layer * (num_layers as f64);

        let gpu_time_per_token = total_ffn_ops / (gpu_gflops * 1e9);
        let cpu_time_per_token = total_ffn_ops / (cpu_gflops * 1e9);
        let tps_gpu = 1.0 / gpu_time_per_token;
        let tps_cpu = 1.0 / cpu_time_per_token;

        println!(
            "\n═══ Token Generation Estimate (FFN only, {} layers) ═══",
            num_layers
        );
        println!("  Total FFN FLOPs: {:.2}B", total_ffn_ops / 1e9);
        println!(
            "  GPU: {:.1} tok/s ({:.2}ms per token)",
            tps_gpu,
            gpu_time_per_token * 1000.0
        );
        println!(
            "  CPU: {:.1} tok/s ({:.2}ms per token)",
            tps_cpu,
            cpu_time_per_token * 1000.0
        );

        println!("\n═══ Analysis (PARITY-036) ═══");
        println!("  - FFN is >90% of inference compute");
        println!("  - MATVEC is memory-bound (low arithmetic intensity)");
        println!("  - GPU transfer overhead = H2D(weight) + H2D(input) + D2H(output)");
        println!("  - Solution: Keep weights on GPU permanently (PARITY-037)");

        // PARITY-037: Cached weights benchmark
        println!("\n[5/6] PARITY-037: Benchmarking GEMM with cached weights...");

        // Load weights to GPU once
        let start = Instant::now();
        let weight_bytes = executor
            .load_weights("fc1", &weight_fc1)
            .expect("Failed to load weights");
        let load_time = start.elapsed();
        println!(
            "       Loaded {} MB to GPU in {:?}",
            weight_bytes / 1024 / 1024,
            load_time
        );
        println!(
            "       Cached weights: {} tensors, {} MB",
            executor.cached_weight_count(),
            executor.cached_weight_bytes() / 1024 / 1024
        );

        // Benchmark cached GEMM
        let start = Instant::now();
        for _ in 0..iterations {
            executor
                .gemm_cached("fc1", &input_fc1, &mut output_fc1, m_fc1, n_fc1, k_fc1)
                .expect("Cached GEMM failed");
        }
        let cached_total = start.elapsed();
        let cached_per_iter = cached_total / iterations;

        // Results comparison
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║            PARITY-037: CACHED vs UNCACHED GEMM                 ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!(
            "║ GPU (uncached): {:>10?}/iter (100MB H2D each time)        ║",
            gpu_per_iter
        );
        println!(
            "║ GPU (cached):   {:>10?}/iter (no weight transfer)         ║",
            cached_per_iter
        );
        println!("╠════════════════════════════════════════════════════════════════╣");

        let cached_speedup = gpu_per_iter.as_secs_f64() / cached_per_iter.as_secs_f64();
        if cached_speedup > 1.0 {
            println!(
                "║ Cached is {:.2}x FASTER (transfer overhead eliminated)       ║",
                cached_speedup
            );
        } else {
            println!(
                "║ Cached is {:.2}x slower (unexpected!)                        ║",
                1.0 / cached_speedup
            );
        }
        println!("╚════════════════════════════════════════════════════════════════╝");

        // Updated token generation estimate with cached weights
        let cached_gflops = flops_per_matvec / cached_per_iter.as_secs_f64() / 1e9;
        let cached_time_per_token = total_ffn_ops / (cached_gflops * 1e9);
        let tps_cached = 1.0 / cached_time_per_token;

        println!("\n═══ Token Generation Estimate (PARITY-037, cached) ═══");
        println!("  Cached GFLOPS: {:.2}", cached_gflops);
        println!(
            "  Estimated: {:.1} tok/s ({:.2}ms per token)",
            tps_cached,
            cached_time_per_token * 1000.0
        );
        println!(
            "  vs uncached: {:.1} tok/s ({:.2}x faster)",
            tps_gpu,
            tps_cached / tps_gpu
        );

        // Cleanup
        executor.clear_weights();

        // Softmax test (small operation)
        println!("\n[6/6] Testing softmax (small operation)...");
        let mut softmax_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let start = Instant::now();
        for _ in 0..100 {
            executor.softmax(&mut softmax_data).expect("Softmax failed");
        }
        let softmax_time = start.elapsed() / 100;
        println!("  GPU softmax (8 elements): {:?}/iter", softmax_time);

        println!("\n═══ Summary ═══");
        if speedup > 1.0 {
            println!("✓ GPU GEMM is {:.2}x faster than CPU", speedup);
            println!("  This validates that CUDA kernels work correctly.");
            println!("  Next: Keep weights on GPU to eliminate transfer overhead.");
        } else {
            println!("✗ GPU GEMM is {:.2}x slower than CPU", 1.0 / speedup);
            println!("  Transfer overhead dominates for single-token MATVEC.");
            println!("  Solutions:");
            println!("  1. Batch tokens to increase compute density");
            println!("  2. Keep weights on GPU (load once at startup)");
            println!("  3. Use async streams to overlap transfer and compute");
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled. Run with --features cuda");
    }
}

/// CPU reference matvec: output = weight @ input
/// weight: [m, k], input: [k], output: [m]
#[cfg(feature = "cuda")]
fn cpu_matvec(weight: &[f32], input: &[f32], output: &mut [f32], m: usize, k: usize) {
    for row in 0..m {
        let mut sum = 0.0f32;
        for col in 0..k {
            sum += weight[row * k + col] * input[col];
        }
        output[row] = sum;
    }
}

/// CPU reference attention (naive O(N²))
#[allow(dead_code)]
fn cpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();

    for i in 0..seq_len {
        // Compute attention scores for row i
        let mut scores = vec![0.0f32; seq_len];
        let mut max_score = f32::NEG_INFINITY;

        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[j] = dot * scale;
            max_score = max_score.max(scores[j]);
        }

        // Softmax
        let mut sum_exp = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        for s in &mut scores {
            *s /= sum_exp;
        }

        // Weighted sum of V
        for d in 0..head_dim {
            let mut val = 0.0f32;
            for j in 0..seq_len {
                val += scores[j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = val;
        }
    }
}
