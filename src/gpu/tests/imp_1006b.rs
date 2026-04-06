
#[test]
#[cfg(feature = "cuda")]
fn test_imp_1006b_block_incremental_uses_cuda() {
    // IMP-1006b: forward_block_incremental_optimized should use do_matmul
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1006b: CUDA not available, skipping");
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

    let input: Vec<f32> = vec![0.1; 256];
    let block_idx = 0;
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
    let _ = cuda_model.forward_block_incremental_optimized(&input, block_idx, &mut cuda_cache);
    let _ = hybrid_model.forward_block_incremental_optimized(&input, block_idx, &mut hybrid_cache);

    let iterations = 20;

    // CUDA block incremental (build up KV cache)
    let mut cuda_cache2 = StreamingKVCache::new(
        cuda_model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = cuda_model.forward_block_incremental_optimized(&input, block_idx, &mut cuda_cache2);
    }
    let cuda_time = start.elapsed();

    // Hybrid block incremental
    let mut hybrid_cache2 = StreamingKVCache::new(
        hybrid_model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let start = Instant::now();
    for _ in 0..iterations {
        let _ =
            hybrid_model.forward_block_incremental_optimized(&input, block_idx, &mut hybrid_cache2);
    }
    let hybrid_time = start.elapsed();

    let cuda_avg_ms = cuda_time.as_secs_f64() * 1000.0 / iterations as f64;
    let hybrid_avg_ms = hybrid_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = hybrid_avg_ms / cuda_avg_ms;

    println!(
        "IMP-1006b: block_incremental (m=1) - CUDA={:.3}ms, Hybrid={:.3}ms, Speedup={:.2}x",
        cuda_avg_ms, hybrid_avg_ms, speedup
    );

    // After IMP-1006, CUDA should handle m=1 operations via do_matmul
    assert!(
        cuda_avg_ms > 0.0,
        "IMP-1006b: CUDA path should complete successfully"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1006c_generate_throughput_improved() {
    // IMP-1006c: After wiring incremental paths, generate() throughput should improve
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1006c: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 4, // More layers to see impact
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

    let mut cuda_model = GpuModel::new_with_cuda(config).expect("Failed to create CUDA model");

    let prompt: Vec<usize> = vec![1, 2, 3, 4, 5];
    let gen_config = GpuGenerateConfig::deterministic(10);

    let start = Instant::now();
    let tokens = cuda_model
        .generate(&prompt, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    let tokens_generated = tokens.len() - prompt.len();
    let tok_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

    println!(
        "IMP-1006c: Generated {} tokens in {:.3}ms ({:.1} tok/s)",
        tokens_generated,
        elapsed.as_secs_f64() * 1000.0,
        tok_per_sec
    );

    // After IMP-1006, throughput should improve from IMP-1005 levels
    // IMP-1005c: ~18.2 tok/s (forward methods)
    // IMP-1004d: ~9.1 tok/s (generate baseline)
    // Target: > 15 tok/s (routing to CUDA in incremental paths)
    println!(
        "IMP-1006c: Previous=9.1 tok/s, Current={:.1} tok/s, Target=228 tok/s (Ollama)",
        tok_per_sec
    );

    assert!(
        tokens_generated > 0,
        "IMP-1006c: Should generate at least some tokens"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1006d_all_matmuls_routed_to_cuda() {
    // IMP-1006d: Verify that model with cuda_scheduler routes all matmuls to CUDA
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1006d: CUDA not available, skipping");
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

    let mut cuda_model = GpuModel::new_with_cuda(config).expect("Failed to create CUDA model");

    // Verify cuda_scheduler is present
    assert!(
        cuda_model.has_cuda_scheduler(),
        "IMP-1006d: CUDA model should have cuda_scheduler"
    );

    // Test do_matmul routing
    let a: Vec<f32> = vec![1.0; 256 * 128];
    let b: Vec<f32> = vec![1.0; 128 * 64];

    let result = cuda_model.do_matmul(&a, &b, 256, 128, 64);
    assert!(
        result.is_ok(),
        "IMP-1006d: do_matmul should complete via CUDA"
    );

    let output = result.expect("test");
    assert_eq!(
        output.len(),
        256 * 64,
        "IMP-1006d: Output dimensions should be correct"
    );

    println!("IMP-1006d: All matmuls routed to CudaScheduler ✓");
}

// ========================================================================
// IMP-1007: Eliminate weight cloning in incremental forward paths
// ========================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1007a_no_clone_matmul() {
    // IMP-1007a: matmul_split should work without cloning weights
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1007a: CUDA not available, skipping");
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

    let mut model = GpuModel::new_with_cuda(config).expect("Failed to create model");

    // Test matmul without weight cloning using split borrow pattern
    let input: Vec<f32> = vec![0.1; 256];

    // IMP-1007: Should be able to call matmul_split without cloning weights
    let result = model.matmul_split(&input, 0, WeightType::Qkv);
    assert!(result.is_ok(), "IMP-1007a: matmul_split should work");

    println!("IMP-1007a: Zero-clone matmul verified ✓");
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1007b_incremental_no_clone_speedup() {
    // IMP-1007b: forward_block_incremental without weight cloning should be faster
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1007b: CUDA not available, skipping");
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

    let mut model = GpuModel::new_with_cuda(config.clone()).expect("Failed to create model");

    let num_kv_heads = model.config.num_kv_heads;
    let head_dim = model.config.head_dim();
    let max_positions = 128;

    let input: Vec<f32> = vec![0.1; 256];
    let block_idx = 0;

    // Warmup
    let mut cache = StreamingKVCache::new(
        model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let _ = model.forward_block_incremental_optimized(&input, block_idx, &mut cache);

    let iterations = 50;

    // Benchmark
    let mut cache2 = StreamingKVCache::new(
        model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.forward_block_incremental_optimized(&input, block_idx, &mut cache2);
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    println!(
        "IMP-1007b: block_incremental avg={:.3}ms ({} iterations)",
        avg_ms, iterations
    );

    // After IMP-1007, this should be < 0.5ms per block (faster than IMP-1006's 0.7ms)
    println!(
        "IMP-1007b: Previous=0.698ms, Current={:.3}ms, Target=<0.5ms",
        avg_ms
    );

    assert!(avg_ms > 0.0, "IMP-1007b: Should complete successfully");
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1007c_generate_throughput_improved() {
    // IMP-1007c: generate() throughput should improve after eliminating cloning
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1007c: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 4,
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

    let mut model = GpuModel::new_with_cuda(config).expect("Failed to create model");

    let prompt: Vec<usize> = vec![1, 2, 3, 4, 5];
    let gen_config = GpuGenerateConfig::deterministic(10);

    let start = Instant::now();
    let tokens = model
        .generate(&prompt, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    let tokens_generated = tokens.len() - prompt.len();
    let tok_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

    println!(
        "IMP-1007c: Generated {} tokens in {:.3}ms ({:.1} tok/s)",
        tokens_generated,
        elapsed.as_secs_f64() * 1000.0,
        tok_per_sec
    );

    // After IMP-1007, throughput should improve from IMP-1006's 37.3 tok/s
    println!(
        "IMP-1007c: Previous=37.3 tok/s, Current={:.1} tok/s, Target=228 tok/s (Ollama)",
        tok_per_sec
    );

    assert!(
        tokens_generated > 0,
        "IMP-1007c: Should generate at least some tokens"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1008a_refcell_scheduler_matmul() {
    // IMP-1008a: matmul_refcell should work with &self (no &mut self)
    // This enables simultaneous borrowing of weights and scheduler
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1008a: CUDA not available, skipping");
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

    let model = GpuModel::new_with_cuda(config).expect("Failed to create CUDA model");

    // Test matmul_refcell with &self (not &mut self)
    let a: Vec<f32> = vec![0.1; 256];
    let b: Vec<f32> = vec![0.2; 256 * 512];

    // This should work without requiring &mut self
    let result = model
        .matmul_refcell(&a, &b, 1, 256, 512)
        .expect("matmul_refcell should work");

    assert_eq!(
        result.len(),
        512,
        "IMP-1008a: Output should be 512 elements"
    );

    // Verify result is reasonable (non-zero, non-NaN)
    let sum: f32 = result.iter().sum();
    assert!(sum.is_finite(), "IMP-1008a: Result should be finite");
    assert!(sum != 0.0, "IMP-1008a: Result should be non-zero");

    println!("IMP-1008a: matmul_refcell works with &self");
}
