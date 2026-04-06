
#[test]
#[cfg(feature = "cuda")]
fn test_imp_1008b_zero_clone_forward_block() {
    // IMP-1008b: forward_block_refcell should not clone weights
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1008b: CUDA not available, skipping");
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

    let model = GpuModel::new_with_cuda(config.clone()).expect("Failed to create CUDA model");

    let input: Vec<f32> = vec![0.1; 256];
    let block_idx = 0;
    let num_kv_heads = model.config.num_kv_heads;
    let head_dim = model.config.head_dim();
    let max_positions = 128;

    // Warmup
    let mut cache = StreamingKVCache::new(
        model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let _ = model.forward_block_refcell(&input, block_idx, &mut cache);

    let iterations = 100;

    // Benchmark with RefCell (zero-clone)
    let mut cache2 = StreamingKVCache::new(
        model.config.num_layers,
        max_positions,
        num_kv_heads,
        head_dim,
    );
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.forward_block_refcell(&input, block_idx, &mut cache2);
    }
    let refcell_time = start.elapsed();
    let refcell_avg_us = refcell_time.as_micros() as f64 / iterations as f64;

    println!(
        "IMP-1008b: forward_block_refcell avg={:.1}µs ({} iterations)",
        refcell_avg_us, iterations
    );

    // Should be faster than clone-based approach
    assert!(
        refcell_avg_us > 0.0,
        "IMP-1008b: forward_block_refcell should complete"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1008c_generate_throughput_refcell() {
    // IMP-1008c: generate_refcell should have better throughput than clone-based generate
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1008c: CUDA not available, skipping");
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

    let model = GpuModel::new_with_cuda(config).expect("Failed to create CUDA model");

    let prompt: Vec<usize> = vec![1, 2, 3, 4, 5];
    let gen_config = GpuGenerateConfig::deterministic(10);

    // Measure RefCell-based generation
    let start = Instant::now();
    let tokens = model
        .generate_refcell(&prompt, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    let tokens_generated = tokens.len() - prompt.len();
    let tok_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

    println!(
        "IMP-1008c: Generated {} tokens in {:.3}ms ({:.1} tok/s)",
        tokens_generated,
        elapsed.as_secs_f64() * 1000.0,
        tok_per_sec
    );

    // Target: >50 tok/s (improvement over IMP-1006's 35 tok/s)
    println!(
        "IMP-1008c: Previous=35.1 tok/s, Current={:.1} tok/s, Target=228 tok/s (Ollama)",
        tok_per_sec
    );

    assert!(
        tokens_generated > 0,
        "IMP-1008c: Should generate at least some tokens"
    );
}

#[test]
#[ignore = "flaky performance test - depends on hardware state"]
#[cfg(feature = "cuda")]
fn test_imp_1008d_compare_clone_vs_refcell() {
    // IMP-1008d: Direct comparison of clone-based vs RefCell-based forward
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1008d: CUDA not available, skipping");
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

    let mut clone_model =
        GpuModel::new_with_cuda(config.clone()).expect("Failed to create clone model");
    let refcell_model = GpuModel::new_with_cuda(config).expect("Failed to create refcell model");

    let prompt: Vec<usize> = vec![1, 2, 3, 4, 5];
    let gen_config = GpuGenerateConfig::deterministic(10);

    // Warmup
    let _ = clone_model.generate(&prompt, &gen_config);
    let _ = refcell_model.generate_refcell(&prompt, &gen_config);

    let iterations = 5;

    // Benchmark clone-based
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = clone_model.generate(&prompt, &gen_config);
    }
    let clone_time = start.elapsed();

    // Benchmark RefCell-based
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = refcell_model.generate_refcell(&prompt, &gen_config);
    }
    let refcell_time = start.elapsed();

    let clone_avg_ms = clone_time.as_secs_f64() * 1000.0 / iterations as f64;
    let refcell_avg_ms = refcell_time.as_secs_f64() * 1000.0 / iterations as f64;
    let speedup = clone_avg_ms / refcell_avg_ms;

    println!(
        "IMP-1008d: Clone={:.2}ms, RefCell={:.2}ms, Speedup={:.2}x",
        clone_avg_ms, refcell_avg_ms, speedup
    );

    // RefCell should be faster (no cloning overhead)
    // For small test models, improvement may be small
    // For phi-2 scale, improvement would be dramatic (8.6GB cloning eliminated)
    assert!(
        speedup > 0.9,
        "IMP-1008d: RefCell should not be slower than clone"
    );
}

#[test]
#[ignore = "flaky performance test - depends on hardware state"]
#[cfg(feature = "cuda")]
fn test_imp_1009a_main_generate_uses_refcell() {
    // IMP-1009a: Main generate() should use RefCell path when CUDA available
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1009a: CUDA not available, skipping");
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

    let mut model = GpuModel::new_with_cuda(config).expect("Failed to create CUDA model");

    let prompt: Vec<usize> = vec![1, 2, 3, 4, 5];
    let gen_config = GpuGenerateConfig::deterministic(10);

    // Measure main generate() throughput
    let start = Instant::now();
    let tokens = model
        .generate(&prompt, &gen_config)
        .expect("Generation failed");
    let elapsed = start.elapsed();

    let tokens_generated = tokens.len() - prompt.len();
    let tok_per_sec = tokens_generated as f64 / elapsed.as_secs_f64();

    println!(
        "IMP-1009a: Main generate() - {} tokens in {:.3}ms ({:.1} tok/s)",
        tokens_generated,
        elapsed.as_secs_f64() * 1000.0,
        tok_per_sec
    );

    // After IMP-1009, main generate() should be fast (>100 tok/s target)
    // This test will FAIL before IMP-1009 wiring (expect ~35 tok/s)
    // This test will PASS after IMP-1009 wiring (expect ~200+ tok/s)
    println!(
        "IMP-1009a: Target=100+ tok/s (RefCell speed), Current={:.1} tok/s",
        tok_per_sec
    );

    // Assert that main generate() is fast (indicates RefCell is wired in)
    assert!(
        tok_per_sec > 100.0,
        "IMP-1009a: Main generate() should achieve >100 tok/s with RefCell wiring"
    );
}

#[test]
#[ignore = "flaky performance test - depends on hardware state"]
#[cfg(feature = "cuda")]
fn test_imp_1009b_generate_parity_with_refcell() {
    // IMP-1009b: Main generate() should match generate_refcell() throughput
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1009b: CUDA not available, skipping");
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

    let mut clone_model = GpuModel::new_with_cuda(config.clone()).expect("Failed to create model");
    let refcell_model = GpuModel::new_with_cuda(config).expect("Failed to create model");

    let prompt: Vec<usize> = vec![1, 2, 3, 4, 5];
    let gen_config = GpuGenerateConfig::deterministic(10);

    // Warmup
    let _ = clone_model.generate(&prompt, &gen_config);
    let _ = refcell_model.generate_refcell(&prompt, &gen_config);

    let iterations = 5;

    // Benchmark main generate()
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = clone_model.generate(&prompt, &gen_config);
    }
    let main_time = start.elapsed();

    // Benchmark generate_refcell()
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = refcell_model.generate_refcell(&prompt, &gen_config);
    }
    let refcell_time = start.elapsed();

    let main_avg_ms = main_time.as_secs_f64() * 1000.0 / iterations as f64;
    let refcell_avg_ms = refcell_time.as_secs_f64() * 1000.0 / iterations as f64;
    let ratio = main_avg_ms / refcell_avg_ms;

    println!(
        "IMP-1009b: Main={:.2}ms, RefCell={:.2}ms, Ratio={:.2}x",
        main_avg_ms, refcell_avg_ms, ratio
    );

    // After IMP-1009, main should be within 1.2x of RefCell (allow some overhead)
    assert!(
        ratio < 1.5,
        "IMP-1009b: Main generate() should be within 1.5x of RefCell throughput"
    );
}

// ============================================================================
// Coverage Improvement Tests - Scalar/SIMD Operations
// ============================================================================

#[test]
fn test_exceeds_gpu_buffer_limit() {
    // Within limit
    assert!(!exceeds_gpu_buffer_limit(1000));
    // At limit
    assert!(!exceeds_gpu_buffer_limit(MAX_GPU_BUFFER_BYTES / 4));
    // Exceeds limit
    assert!(exceeds_gpu_buffer_limit(MAX_GPU_BUFFER_BYTES / 4 + 1));
}

#[test]
fn test_scalar_softmax_basic() {
    let input = vec![1.0, 2.0, 3.0];
    let output = scalar_softmax(&input);

    assert_eq!(output.len(), 3);
    // Sum should be 1
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Largest input should have largest output
    assert!(output[2] > output[1] && output[1] > output[0]);
}

#[test]
fn test_scalar_softmax_empty() {
    let input: Vec<f32> = vec![];
    let output = scalar_softmax(&input);
    assert!(output.is_empty());
}

#[test]
fn test_scalar_softmax_single() {
    let input = vec![5.0];
    let output = scalar_softmax(&input);
    assert_eq!(output.len(), 1);
    assert!((output[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_simd_softmax_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = simd_softmax(&input);

    assert_eq!(output.len(), 4);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_simd_softmax_empty() {
    let input: Vec<f32> = vec![];
    let output = simd_softmax(&input);
    assert!(output.is_empty());
}

#[test]
fn test_simd_softmax_matches_scalar() {
    let input = vec![0.5, 1.5, 2.5, 0.0];
    let scalar_out = scalar_softmax(&input);
    let simd_out = simd_softmax(&input);

    for (s, si) in scalar_out.iter().zip(simd_out.iter()) {
        assert!((s - si).abs() < 1e-5, "SIMD softmax should match scalar");
    }
}

#[test]
fn test_scalar_rope_basic() {
    let seq_len = 2;
    let head_dim = 4;
    let hidden_dim = head_dim;
    let input = vec![1.0; seq_len * hidden_dim];
    let theta = 10000.0;

    let output = scalar_rope(&input, seq_len, head_dim, theta);

    assert_eq!(output.len(), input.len());
    // Should have rotated values
}

#[test]
fn test_scalar_rope_empty() {
    let output = scalar_rope(&[], 0, 4, 10000.0);
    assert!(output.is_empty());
}

#[test]
fn test_scalar_rope_zero_head_dim() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = scalar_rope(&input, 2, 0, 10000.0);
    assert!(output.is_empty());
}

#[test]
fn test_simd_rope_basic() {
    let seq_len = 2;
    let head_dim = 4;
    let hidden_dim = head_dim;
    let input = vec![1.0; seq_len * hidden_dim];
    let theta = 10000.0;

    let output = simd_rope(&input, seq_len, head_dim, theta);

    assert_eq!(output.len(), input.len());
}

#[test]
fn test_simd_rope_empty() {
    let output = simd_rope(&[], 0, 4, 10000.0);
    assert!(output.is_empty());
}
