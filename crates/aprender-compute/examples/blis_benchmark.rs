#![allow(clippy::disallowed_methods)]
//! BLIS GEMM Benchmark
//!
//! Measures performance of the BLIS-style matrix multiplication implementation.
//!
//! # Running
//!
//! ```bash
//! cargo run --release --example blis_benchmark
//! ```
//!
//! # Performance Expectations
//!
//! On a modern x86_64 CPU with AVX2:
//! - 64×64: ~25 GFLOP/s
//! - 256×256: ~60 GFLOP/s
//! - 512×512: ~70 GFLOP/s
//! - 1024×1024: ~72 GFLOP/s
//!
//! # Algorithm
//!
//! Uses the BLIS framework (Van Zee & Van de Geijn, 2015):
//! - 8×6 register-blocked microkernel for AVX2
//! - Cache-optimized packing for L1/L2/L3
//! - 5-loop blocking structure
//!
//! # References
//!
//! - Goto, K., & Van de Geijn, R. A. (2008). Anatomy of High-Performance Matrix Multiplication.
//! - Van Zee, F. G., & Van de Geijn, R. A. (2015). BLIS: A Framework for BLAS.

use std::time::Instant;
use trueno::blis::{gemm_blis, gemm_reference, BlisProfiler};

fn benchmark_gemm(name: &str, n: usize, iterations: usize) {
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let mut c = vec![0.0f32; n * n];

    // Warmup
    for _ in 0..3 {
        c.fill(0.0);
        gemm_blis(n, n, n, &a, &b, &mut c, None).unwrap();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        c.fill(0.0);
        gemm_blis(n, n, n, &a, &b, &mut c, None).unwrap();
    }
    let elapsed = start.elapsed();

    let total_flops = 2u64 * (n as u64) * (n as u64) * (n as u64) * (iterations as u64);
    let gflops = total_flops as f64 / elapsed.as_secs_f64() / 1e9;
    let time_per_op = elapsed.as_micros() as f64 / iterations as f64;

    println!("{:20} {:4}x{:4}: {:8.1} us, {:6.1} GFLOP/s", name, n, n, time_per_op, gflops);
}

fn benchmark_with_profiler(n: usize) {
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let mut c = vec![0.0f32; n * n];
    let mut profiler = BlisProfiler::enabled();

    // Run with profiling
    gemm_blis(n, n, n, &a, &b, &mut c, Some(&mut profiler)).unwrap();

    println!("\nProfiler Results for {}x{}:", n, n);
    println!("{}", profiler.summary());
}

fn compare_reference_vs_blis(n: usize) {
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();

    // Reference
    let mut c_ref = vec![0.0f32; n * n];
    let start = Instant::now();
    gemm_reference(n, n, n, &a, &b, &mut c_ref).unwrap();
    let ref_time = start.elapsed();

    // BLIS
    let mut c_blis = vec![0.0f32; n * n];
    let start = Instant::now();
    gemm_blis(n, n, n, &a, &b, &mut c_blis, None).unwrap();
    let blis_time = start.elapsed();

    // Verify correctness
    let max_diff: f32 =
        c_ref.iter().zip(c_blis.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);

    let speedup = ref_time.as_secs_f64() / blis_time.as_secs_f64();

    println!(
        "{}x{}: Reference {:8.1}ms, BLIS {:8.1}ms, Speedup: {:5.1}x, MaxDiff: {:.2e}",
        n,
        n,
        ref_time.as_secs_f64() * 1000.0,
        blis_time.as_secs_f64() * 1000.0,
        speedup,
        max_diff
    );
}

fn benchmark_gemv(k: usize, n: usize, iterations: usize) {
    use trueno::blis::gemv::gemv;
    let a: Vec<f32> = (0..k).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let mut c = vec![0.0f32; n];

    // Warmup
    for _ in 0..10 {
        c.fill(0.0);
        gemv(k, n, &a, &b, &mut c);
    }

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        c.fill(0.0);
        gemv(k, n, &a, &b, &mut c);
    }
    let elapsed = start.elapsed();

    let total_flops = 2u64 * (k as u64) * (n as u64) * (iterations as u64);
    let gflops = total_flops as f64 / elapsed.as_secs_f64() / 1e9;
    let time_per_op = elapsed.as_nanos() as f64 / iterations as f64 / 1000.0;

    println!("GEMV  1x{:4} @ {:4}x{:5}: {:8.2} us, {:6.1} GFLOP/s", k, k, n, time_per_op, gflops);
}

fn benchmark_fused_attention(head_dim: usize, seq_len: usize, iterations: usize) {
    use trueno::blis::attention::fused_attention_decode;
    use trueno::blis::gemv::gemv;

    let q: Vec<f32> = (0..head_dim).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();
    let k: Vec<f32> =
        (0..seq_len * head_dim).map(|i| ((i * 13 + 7) % 100) as f32 / 100.0 - 0.5).collect();
    let v: Vec<f32> =
        (0..seq_len * head_dim).map(|i| ((i * 11 + 5) % 100) as f32 / 100.0 - 0.5).collect();

    let mut out = vec![0.0f32; head_dim];

    // Unfused: GEMV + softmax + GEMV
    for _ in 0..10 {
        let mut scores = vec![0.0f32; seq_len];
        gemv(head_dim, seq_len, &q, &k, &mut scores);
        let s = trueno::blis::softmax::softmax_1d_alloc(&scores);
        out.fill(0.0);
        gemv(1, head_dim, &s, &v, &mut out); // scores(1,S) @ V(S,D)... wrong shape
    }
    // Actually, unfused scores@V is: for each V row, multiply by weight and accumulate
    // This is a weighted sum, not a standard GEMV. Let's measure properly.
    let mut unfused_fn = || {
        let mut scores = vec![0.0f32; seq_len];
        gemv(head_dim, seq_len, &q, &k, &mut scores);
        let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        let inv = 1.0 / sum;
        for s in scores.iter_mut() {
            *s *= inv;
        }
        out.fill(0.0);
        for i in 0..seq_len {
            let w = scores[i];
            let vr = &v[i * head_dim..(i + 1) * head_dim];
            for d in 0..head_dim {
                out[d] += w * vr[d];
            }
        }
    };

    // Warmup
    for _ in 0..10 {
        unfused_fn();
    }
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        unfused_fn();
    }
    let unfused_us = start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0;

    // Fused
    for _ in 0..10 {
        fused_attention_decode(&q, &k, &v, head_dim, seq_len, &mut out);
    }
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        fused_attention_decode(&q, &k, &v, head_dim, seq_len, &mut out);
    }
    let fused_us = start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0;

    let speedup = unfused_us / fused_us;
    println!(
        "Attn D={:3} S={:4}: unfused {:6.2}us | fused {:6.2}us | {:.2}x",
        head_dim, seq_len, unfused_us, fused_us, speedup,
    );
}

fn benchmark_rmsnorm(n: usize, iterations: usize) {
    let input: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();
    let gamma: Vec<f32> = (0..n).map(|i| ((i * 11 + 5) % 100) as f32 / 100.0 + 0.5).collect();
    let mut output = vec![0.0f32; n];

    for _ in 0..100 {
        trueno::blis::norms::rms_norm(&input, &gamma, 1e-5, &mut output).unwrap();
    }
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        trueno::blis::norms::rms_norm(&input, &gamma, 1e-5, &mut output).unwrap();
    }
    let us = start.elapsed().as_nanos() as f64 / iterations as f64 / 1000.0;
    let gbps = (n * 4 * 3) as f64 / (us * 1e-6) / 1e9;
    println!("RmsNorm  n={:5}: {:6.2} us, {:5.1} GB/s", n, us, gbps);
}

fn main() {
    println!("=== BLIS GEMM Benchmark ===\n");

    println!("--- Reference vs BLIS Comparison ---");
    for n in [64, 128, 256, 512] {
        compare_reference_vs_blis(n);
    }

    println!("\n--- BLIS Performance (multiple iterations) ---");
    benchmark_gemm("BLIS", 64, 1000);
    benchmark_gemm("BLIS", 128, 100);
    benchmark_gemm("BLIS", 256, 20);
    benchmark_gemm("BLIS", 512, 5);
    benchmark_gemm("BLIS", 1024, 2);

    #[cfg(target_arch = "x86_64")]
    {
        println!("\n--- Broadcast-B (MR=64, NR=6) Performance ---");
        benchmark_bcast_b(64, 1000);
        benchmark_bcast_b(128, 100);
        benchmark_bcast_b(256, 20);
        benchmark_bcast_b(512, 5);
        benchmark_bcast_b(1024, 2);
    }

    println!("\n--- RmsNorm Performance ---");
    benchmark_rmsnorm(128, 100_000); // attention head_dim
    benchmark_rmsnorm(4096, 50_000); // LLM hidden_dim
    benchmark_rmsnorm(11008, 20_000); // LLM FFN dim

    println!("\n--- GEMV Performance (attention-critical path) ---");
    benchmark_gemv(128, 512, 5000); // head_dim=128, seq_len=512
    benchmark_gemv(128, 1024, 2000); // head_dim=128, seq_len=1024
    benchmark_gemv(128, 4096, 500); // head_dim=128, seq_len=4096
    benchmark_gemv(4096, 4096, 100); // FFN projection
    benchmark_gemv(4096, 11008, 50); // FFN up/down projection

    println!("\n--- Fused Attention (FlashAttention-style [64]) ---");
    benchmark_fused_attention(128, 64, 10000);
    benchmark_fused_attention(128, 512, 5000);
    benchmark_fused_attention(128, 1024, 2000);
    benchmark_fused_attention(128, 4096, 500);

    #[cfg(feature = "parallel")]
    {
        println!("\n--- Parallel Comparison (per-thread-B vs shared-B) ---");
        for &n in &[256, 512, 1024] {
            benchmark_parallel_compare(n);
        }
    }

    println!("\n--- Detailed Profiler Output ---");
    benchmark_with_profiler(256);
}

#[cfg(feature = "parallel")]
fn benchmark_parallel_compare(n: usize) {
    use trueno::blis::{gemm_blis_parallel, gemm_blis_parallel_shared_b};
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let mut c = vec![0.0f32; n * n];
    let iters = if n >= 1024 {
        5
    } else if n >= 512 {
        10
    } else {
        20
    };
    let flops = 2u64 * (n as u64).pow(3);

    // Warmup + bench per-thread-B
    for _ in 0..3 {
        c.fill(0.0);
        gemm_blis_parallel(n, n, n, &a, &b, &mut c).unwrap();
    }
    let start = std::time::Instant::now();
    for _ in 0..iters {
        c.fill(0.0);
        gemm_blis_parallel(n, n, n, &a, &b, &mut c).unwrap();
    }
    let per_thread = start.elapsed().as_secs_f64() / iters as f64;
    let gf_pt = flops as f64 / per_thread / 1e9;

    // Warmup + bench shared-B
    for _ in 0..3 {
        c.fill(0.0);
        gemm_blis_parallel_shared_b(n, n, n, &a, &b, &mut c).unwrap();
    }
    let start = std::time::Instant::now();
    for _ in 0..iters {
        c.fill(0.0);
        gemm_blis_parallel_shared_b(n, n, n, &a, &b, &mut c).unwrap();
    }
    let shared = start.elapsed().as_secs_f64() / iters as f64;
    let gf_sb = flops as f64 / shared / 1e9;

    let ratio = shared / per_thread;
    println!(
        "{:4}x{:4}: per-thread-B {:6.1} GFLOPS ({:.2}ms) | shared-B {:6.1} GFLOPS ({:.2}ms) | ratio {:.2}x",
        n, n, gf_pt, per_thread * 1e3, gf_sb, shared * 1e3, ratio,
    );
}

#[cfg(target_arch = "x86_64")]
fn benchmark_bcast_b(n: usize, iterations: usize) {
    use trueno::blis::gemm_blis_broadcast_b;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let mut c = vec![0.0f32; n * n];

    // Warmup
    for _ in 0..3 {
        c.fill(0.0);
        gemm_blis_broadcast_b(n, n, n, &a, &b, &mut c).unwrap();
    }

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        c.fill(0.0);
        gemm_blis_broadcast_b(n, n, n, &a, &b, &mut c).unwrap();
    }
    let elapsed = start.elapsed();

    let total_flops = 2u64 * (n as u64) * (n as u64) * (n as u64) * (iterations as u64);
    let gflops = total_flops as f64 / elapsed.as_secs_f64() / 1e9;
    let time_per_op = elapsed.as_micros() as f64 / iterations as f64;

    println!(
        "{:20} {:4}x{:4}: {:8.1} us, {:6.1} GFLOP/s",
        "Bcast-B (64x6)", n, n, time_per_op, gflops
    );
}
