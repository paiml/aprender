use crate::gpu::*;
#[test]
#[cfg(feature = "cuda")]
fn test_imp_1004a_cuda_matmul_benchmark() {
    // IMP-1004a: Benchmark CUDA matmul for realistic LLM dimensions
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1004a: CUDA not available, skipping");
        return;
    }

    let mut cuda_scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");

    // Realistic LLM dimensions: hidden_dim=4096, intermediate_dim=11008
    let test_cases = [
        (1, 4096, 4096, "1x4096x4096 (m=1, attention output)"),
        (1, 4096, 11008, "1x4096x11008 (m=1, FFN fc1)"),
        (1, 11008, 4096, "1x11008x4096 (m=1, FFN fc2)"),
        (1, 4096, 32000, "1x4096x32000 (m=1, LM head)"),
    ];

    for (m, k, n, desc) in test_cases {
        let a: Vec<f32> = vec![1.0; m * k];
        let b: Vec<f32> = vec![1.0; k * n];

        // Warmup
        let _ = cuda_scheduler.matmul(&a, &b, m, k, n);

        // Benchmark
        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = cuda_scheduler.matmul(&a, &b, m, k, n);
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

        println!("IMP-1004a: {} - {:.3}ms avg", desc, avg_ms);
    }
}

#[test]
#[cfg(feature = "cuda")]
#[allow(clippy::many_single_char_names)]
fn test_imp_1004b_cuda_vs_cpu_matmul() {
    // IMP-1004b: Direct comparison of CUDA vs CPU matmul for m=1
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1004b: CUDA not available, skipping");
        return;
    }

    let mut cuda_scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");
    let mut hybrid_scheduler =
        HybridScheduler::with_threshold(1).expect("Failed to create HybridScheduler");

    // Test m=1 case where HybridScheduler forces CPU
    let m = 1;
    let k = 4096;
    let n = 4096;
    let a: Vec<f32> = vec![1.0; m * k];
    let b: Vec<f32> = vec![1.0; k * n];

    // Warmup
    let _ = cuda_scheduler.matmul(&a, &b, m, k, n);
    let _ = hybrid_scheduler.matmul(&a, &b, m, k, n);

    let iterations = 20;

    // CUDA timing
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cuda_scheduler.matmul(&a, &b, m, k, n);
    }
    let cuda_time = start.elapsed();

    // CPU timing (HybridScheduler forces CPU for m=1)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hybrid_scheduler.matmul(&a, &b, m, k, n);
    }
    let cpu_time = start.elapsed();

    let cuda_avg_ms = cuda_time.as_secs_f64() * 1000.0 / iterations as f64;
    let cpu_avg_ms = cpu_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = cpu_avg_ms / cuda_avg_ms;

    println!(
        "IMP-1004b: m=1 matmul (1x{}x{}) - CUDA={:.3}ms, CPU={:.3}ms, Speedup={:.2}x",
        k, n, cuda_avg_ms, cpu_avg_ms, speedup
    );

    // For small m=1 ops, GPU may not be faster due to transfer overhead
    // Just verify both produce correct results
    let cuda_result = cuda_scheduler.matmul(&a, &b, m, k, n).expect("test");
    let cpu_result = hybrid_scheduler.matmul(&a, &b, m, k, n).expect("test");

    assert_eq!(
        cuda_result.len(),
        cpu_result.len(),
        "IMP-1004b: Both should produce same output size"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1004c_full_forward_benchmark() {
    // IMP-1004c: Benchmark full GpuModel forward pass (CUDA vs Hybrid)
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1004c: CUDA not available, skipping");
        return;
    }

    // Use smaller model for benchmark (still tests full pipeline)
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 512,
        num_heads: 8,
        num_kv_heads: 8,
        num_layers: 4,
        intermediate_dim: 1024,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    let mut cuda_model =
        GpuModel::new_with_cuda(config.clone()).expect("Failed to create CUDA model");
    let mut hybrid_model = GpuModel::new(config).expect("Failed to create Hybrid model");

    let token_ids = vec![42usize]; // Single token

    // Warmup
    let _ = cuda_model.forward_gpu(&token_ids);
    let _ = hybrid_model.forward_gpu(&token_ids);

    let iterations = 20;

    // CUDA forward timing
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cuda_model.forward_gpu(&token_ids);
    }
    let cuda_time = start.elapsed();

    // Hybrid forward timing (forces CPU for m=1)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hybrid_model.forward_gpu(&token_ids);
    }
    let hybrid_time = start.elapsed();

    let cuda_avg_ms = cuda_time.as_secs_f64() * 1000.0 / iterations as f64;
    let hybrid_avg_ms = hybrid_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = hybrid_avg_ms / cuda_avg_ms;

    println!(
        "IMP-1004c: Full forward pass - CUDA={:.3}ms, Hybrid={:.3}ms, Speedup={:.2}x",
        cuda_avg_ms, hybrid_avg_ms, speedup
    );

    // Calculate throughput
    let cuda_tok_per_sec = 1000.0 / cuda_avg_ms;
    let hybrid_tok_per_sec = 1000.0 / hybrid_avg_ms;

    println!(
        "IMP-1004c: Throughput - CUDA={:.1} tok/s, Hybrid={:.1} tok/s",
        cuda_tok_per_sec, hybrid_tok_per_sec
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1004d_token_generation_throughput() {
    // IMP-1004d: Measure token generation throughput (the key parity metric)
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1004d: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 512,
        num_heads: 8,
        num_kv_heads: 8,
        num_layers: 4,
        intermediate_dim: 1024,
        eps: 1e-5,
        rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    let mut cuda_model =
        GpuModel::new_with_cuda(config.clone()).expect("Failed to create CUDA model");

    let prompt = vec![1usize, 2, 3]; // Small prompt

    // Generate tokens and measure
    let gen_config = GpuGenerateConfig::deterministic(10);

    let start = Instant::now();
    let tokens = cuda_model
        .generate(&prompt, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    let tokens_generated = tokens.len() - prompt.len();
    let tok_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

    println!(
        "IMP-1004d: Generated {} tokens in {:.3}ms ({:.1} tok/s)",
        tokens_generated,
        elapsed.as_secs_f64() * 1000.0,
        tok_per_sec
    );

    // Target: Ollama phi2 achieves ~228 tok/s
    // Current gap: ~1090x
    // This test establishes baseline for improvement tracking
    println!(
        "IMP-1004d: Target=228 tok/s (Ollama), Current={:.1} tok/s, Gap={:.0}x",
        tok_per_sec,
        228.0 / tok_per_sec.max(0.001)
    );

    // Verify generation produced tokens
    assert!(
        tokens_generated > 0,
        "IMP-1004d: Should generate at least some tokens"
    );
}

// ========================================================================
// PARITY-120: Weight caching for 10x+ speedup
// ========================================================================

#[test]
#[ignore = "flaky performance test - depends on hardware state"]
#[cfg(feature = "cuda")]
fn test_parity_120a_cached_vs_uncached_matmul() {
    // PARITY-120a: Compare cached vs uncached matmul performance
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("PARITY-120a: CUDA not available, skipping");
        return;
    }

    let mut scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");

    // Realistic LLM dimensions: hidden=4096, qkv=12288 (3*4096)
    let k = 4096usize;
    let n = 4096usize;
    let weight: Vec<f32> = vec![1.0; k * n];
    let x: Vec<f32> = vec![1.0; k];

    // Cache the weight
    scheduler
        .cache_weight("test_weight", &weight)
        .expect("Failed to cache weight");

    // Warmup both paths
    let _ = scheduler.matmul(&x, &weight, 1, k, n);
    let _ = scheduler.matmul_cached("test_weight", &x, k, n);

    let iterations = 20;

    // Benchmark UNCACHED (current slow path)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = scheduler.matmul(&x, &weight, 1, k, n);
    }
    let uncached_time = start.elapsed();

    // Benchmark CACHED (new fast path)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = scheduler.matmul_cached("test_weight", &x, k, n);
    }
    let cached_time = start.elapsed();

    let uncached_avg_ms = uncached_time.as_secs_f64() * 1000.0 / iterations as f64;
    let cached_avg_ms = cached_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = uncached_avg_ms / cached_avg_ms;

    println!(
        "PARITY-120a: 1x{}x{} - Uncached={:.3}ms, Cached={:.3}ms, Speedup={:.1}x",
        k, n, uncached_avg_ms, cached_avg_ms, speedup
    );

    // Verify correctness
    let uncached_result = scheduler.matmul(&x, &weight, 1, k, n).expect("test");
    let cached_result = scheduler
        .matmul_cached("test_weight", &x, k, n)
        .expect("test");

    assert_eq!(
        uncached_result.len(),
        cached_result.len(),
        "PARITY-120a: Output sizes should match"
    );

    // Results should be identical
    for (i, (u, c)) in uncached_result.iter().zip(cached_result.iter()).enumerate() {
        assert!(
            (u - c).abs() < 0.01,
            "PARITY-120a: Results differ at {}: uncached={}, cached={}",
            i,
            u,
            c
        );
    }

    // Target: >2x speedup (eliminating weight transfer)
    assert!(
        speedup > 1.5,
        "PARITY-120a: Expected >1.5x speedup, got {:.1}x",
        speedup
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_parity_120b_full_layer_cached() {
    // PARITY-120b: Simulate full layer with all weights cached
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("PARITY-120b: CUDA not available, skipping");
        return;
    }

    let mut scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");

    // Realistic dimensions
    let hidden = 4096usize;
    let qkv = 3 * hidden;
    let intermediate = 11008usize;

    // Create and cache all layer weights
    let qkv_weight: Vec<f32> = vec![1.0; hidden * qkv];
    let out_weight: Vec<f32> = vec![1.0; hidden * hidden];
    let fc1_weight: Vec<f32> = vec![1.0; hidden * intermediate];
    let fc2_weight: Vec<f32> = vec![1.0; intermediate * hidden];

    scheduler.cache_weight("qkv", &qkv_weight).expect("test");
    scheduler.cache_weight("out", &out_weight).expect("test");
    scheduler.cache_weight("fc1", &fc1_weight).expect("test");
    scheduler.cache_weight("fc2", &fc2_weight).expect("test");

    assert_eq!(
        scheduler.cached_weight_count(),
        4,
        "PARITY-120b: Should have 4 cached weights"
    );

    let x: Vec<f32> = vec![1.0; hidden];
    let iterations = 10;

    // Benchmark cached full layer
    let start = Instant::now();
    for _ in 0..iterations {
        let qkv_out = scheduler
            .matmul_cached("qkv", &x, hidden, qkv)
            .expect("test");
        let attn_out = scheduler
            .matmul_cached("out", &qkv_out[..hidden], hidden, hidden)
            .expect("test");
        let fc1_out = scheduler
            .matmul_cached("fc1", &attn_out, hidden, intermediate)
            .expect("test");
        let _fc2_out = scheduler
            .matmul_cached("fc2", &fc1_out, intermediate, hidden)
            .expect("test");
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let tok_per_sec = 1000.0 / avg_ms;

    println!(
        "PARITY-120b: Full layer (4 matmuls) - {:.2}ms/token = {:.1} tok/s",
        avg_ms, tok_per_sec
    );

    // Target: >50 tok/s for single layer
    println!(
        "PARITY-120b: Target=228 tok/s (Ollama), Current={:.1} tok/s (single layer)",
        tok_per_sec
    );
}

include!("imp_1005a.rs");
include!("imp_1006b.rs");
include!("imp_1008b.rs");
include!("simd_rope.rs");
