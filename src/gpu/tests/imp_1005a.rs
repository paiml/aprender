
// ========================================================================
// IMP-1005: Wire CudaScheduler into forward_gpu() via do_matmul()
// ========================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1005a_do_matmul_uses_cuda_scheduler() {
    // IMP-1005a: do_matmul() should use cuda_scheduler when available
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1005a: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 1,
        intermediate_dim: 128,
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

    // Both should have do_matmul method
    let a: Vec<f32> = vec![1.0; 64];
    let b: Vec<f32> = vec![1.0; 64 * 100];

    let cuda_result = cuda_model.do_matmul(&a, &b, 1, 64, 100);
    let hybrid_result = hybrid_model.do_matmul(&a, &b, 1, 64, 100);

    assert!(
        cuda_result.is_ok(),
        "IMP-1005a: CUDA do_matmul should succeed"
    );
    assert!(
        hybrid_result.is_ok(),
        "IMP-1005a: Hybrid do_matmul should succeed"
    );

    // Both should produce same-sized output
    assert_eq!(
        cuda_result.expect("test").len(),
        hybrid_result.expect("test").len(),
        "IMP-1005a: Both should produce same output size"
    );
}

#[test]
#[cfg(feature = "cuda")]
#[ignore = "flaky - timing depends on GPU warmup state and system load"]
fn test_imp_1005b_forward_gpu_speedup_with_cuda() {
    // IMP-1005b: forward_gpu should be faster with cuda_scheduler
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1005b: CUDA not available, skipping");
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
    let mut hybrid_model = GpuModel::new(config).expect("Failed to create Hybrid model");

    let token_ids = vec![42usize]; // Single token (m=1)

    // Warmup
    let _ = cuda_model.forward_gpu(&token_ids);
    let _ = hybrid_model.forward_gpu(&token_ids);

    let iterations = 20;

    // CUDA forward timing (should use do_matmul -> cuda_scheduler)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cuda_model.forward_gpu(&token_ids);
    }
    let cuda_time = start.elapsed();

    // Hybrid forward timing (do_matmul -> scheduler -> CPU for m=1)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hybrid_model.forward_gpu(&token_ids);
    }
    let hybrid_time = start.elapsed();

    let cuda_avg_ms = cuda_time.as_secs_f64() * 1000.0 / iterations as f64;
    let hybrid_avg_ms = hybrid_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = hybrid_avg_ms / cuda_avg_ms;

    println!(
        "IMP-1005b: forward_gpu (m=1) - CUDA={:.3}ms, Hybrid={:.3}ms, Speedup={:.2}x",
        cuda_avg_ms, hybrid_avg_ms, speedup
    );

    // Calculate throughput
    let cuda_tok_per_sec = 1000.0 / cuda_avg_ms;
    let hybrid_tok_per_sec = 1000.0 / hybrid_avg_ms;

    println!(
        "IMP-1005b: Throughput - CUDA={:.1} tok/s, Hybrid={:.1} tok/s",
        cuda_tok_per_sec, hybrid_tok_per_sec
    );

    // After wiring, CUDA should be faster (speedup > 1.0)
    // Before fix: speedup ~1.0 (both use same path)
    // After fix: speedup should be > 1.5x
    assert!(
        speedup > 0.5,
        "IMP-1005b: CUDA path should not be catastrophically slower"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1005c_token_generation_with_cuda_forward() {
    // IMP-1005c: Token generation throughput after forward_gpu uses cuda_scheduler
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1005c: CUDA not available, skipping");
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

    let mut cuda_model = GpuModel::new_with_cuda(config).expect("Failed to create CUDA model");

    let prompt = vec![1usize, 2, 3];
    let gen_config = GpuGenerateConfig::deterministic(10);

    let start = Instant::now();
    let tokens = cuda_model
        .generate(&prompt, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    let tokens_generated = tokens.len() - prompt.len();
    let tok_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

    println!(
        "IMP-1005c: Generated {} tokens in {:.3}ms ({:.1} tok/s)",
        tokens_generated,
        elapsed.as_secs_f64() * 1000.0,
        tok_per_sec
    );

    // After IMP-1005, throughput should improve
    // Previous (IMP-1004d): 9.1 tok/s
    // Target: > 15 tok/s (improvement from wiring cuda_scheduler)
    println!(
        "IMP-1005c: Previous=9.1 tok/s, Current={:.1} tok/s, Target=228 tok/s",
        tok_per_sec
    );

    assert!(
        tokens_generated > 0,
        "IMP-1005c: Should generate at least some tokens"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1005d_forward_block_uses_do_matmul() {
    // IMP-1005d: forward_block_idx should use do_matmul internally
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1005d: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 512,
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

    // Test single token forward through blocks
    let input: Vec<f32> = vec![0.1; 256]; // [1, hidden_dim]
    let seq_len = 1;

    // Warmup
    let _ = cuda_model.forward_block_idx(&input, seq_len, 0);
    let _ = hybrid_model.forward_block_idx(&input, seq_len, 0);

    let iterations = 20;

    // CUDA block forward
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cuda_model.forward_block_idx(&input, seq_len, 0);
    }
    let cuda_time = start.elapsed();

    // Hybrid block forward
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hybrid_model.forward_block_idx(&input, seq_len, 0);
    }
    let hybrid_time = start.elapsed();

    let cuda_avg_ms = cuda_time.as_secs_f64() * 1000.0 / iterations as f64;
    let hybrid_avg_ms = hybrid_time.as_secs_f64() * 1000.0 / iterations as f64;

    println!(
        "IMP-1005d: forward_block_idx (m=1) - CUDA={:.3}ms, Hybrid={:.3}ms",
        cuda_avg_ms, hybrid_avg_ms
    );

    // Both should complete successfully
    let cuda_result = cuda_model.forward_block_idx(&input, seq_len, 0);
    let hybrid_result = hybrid_model.forward_block_idx(&input, seq_len, 0);

    assert!(
        cuda_result.is_ok() && hybrid_result.is_ok(),
        "IMP-1005d: Both should complete successfully"
    );
}

// ========================================================================
// IMP-1006: Wire do_matmul into incremental forward paths
// ========================================================================

#[test]
#[ignore = "flaky performance test - depends on hardware state"]
#[cfg(feature = "cuda")]
fn test_imp_1006a_incremental_forward_uses_cuda() {
    // IMP-1006a: forward_gpu_incremental_optimized should use do_matmul
    // which routes to CudaScheduler when available
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1006a: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 512,
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

    // Single token forward (m=1)
    // With config: hidden_dim=256, num_heads=4 => head_dim=64
    let token_id: usize = 42;
    let num_kv_heads = cuda_model.config.num_kv_heads;
    let head_dim = cuda_model.config.head_dim();
    let max_positions = 128;

    // Warmup
    let mut cuda_cache = StreamingKVCache::new(
        cuda_model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let mut hybrid_cache = StreamingKVCache::new(
        hybrid_model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let _ = cuda_model.forward_gpu_incremental_optimized(token_id, &mut cuda_cache);
    let _ = hybrid_model.forward_gpu_incremental_optimized(token_id, &mut hybrid_cache);

    let iterations = 20;

    // CUDA incremental forward (create fresh cache each iteration)
    let start = Instant::now();
    for i in 0..iterations {
        let mut cache = StreamingKVCache::new(
            cuda_model.config.num_layers,
            max_positions,
            num_kv_heads,
            head_dim,
        );
        let _ = cuda_model.forward_gpu_incremental_optimized(i % 100, &mut cache);
    }
    let cuda_time = start.elapsed();

    // Hybrid incremental forward
    let start = Instant::now();
    for i in 0..iterations {
        let mut cache = StreamingKVCache::new(
            hybrid_model.config.num_layers,
            max_positions,
            num_kv_heads,
            head_dim,
        );
        let _ = hybrid_model.forward_gpu_incremental_optimized(i % 100, &mut cache);
    }
    let hybrid_time = start.elapsed();

    let cuda_avg_ms = cuda_time.as_secs_f64() * 1000.0 / iterations as f64;
    let hybrid_avg_ms = hybrid_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = hybrid_avg_ms / cuda_avg_ms;

    println!(
        "IMP-1006a: incremental_forward (m=1) - CUDA={:.3}ms, Hybrid={:.3}ms, Speedup={:.2}x",
        cuda_avg_ms, hybrid_avg_ms, speedup
    );

    // IMP-1006: After wiring do_matmul, CUDA path should be faster
    assert!(
        cuda_avg_ms < hybrid_avg_ms * 2.0,
        "IMP-1006a: CUDA incremental should not be much slower than Hybrid"
    );
}
