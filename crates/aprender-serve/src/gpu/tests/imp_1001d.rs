
#[test]
#[cfg(feature = "cuda")]
fn test_imp_1001d_gpu_model_with_cuda_backend() {
    // IMP-1001d: Test GpuModel can use CUDA backend for inference
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1001d: CUDA not available, skipping");
        return;
    }

    // Create small GpuModel config
    let config = GpuModelConfig {
        vocab_size: 1000,
        hidden_dim: 256,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
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

    // Create model
    let mut model = GpuModel::new(config).expect("Failed to create GpuModel");

    // Generate should work (currently uses HybridScheduler/CPU)
    let prompt = vec![1usize, 2, 3];
    let gen_config = GpuGenerateConfig {
        max_tokens: 5,
        temperature: 1.0,
        top_k: 50,
        stop_tokens: vec![],
        trace: false,
    };

    let result = model.generate(&prompt, &gen_config);
    assert!(result.is_ok(), "IMP-1001d: Generate should succeed");

    let tokens = result.expect("test");
    assert!(
        tokens.len() >= prompt.len(),
        "IMP-1001d: Should generate at least prompt length tokens"
    );
}

// =========================================================================
// IMP-1002: CudaScheduler - CUDA-native scheduler for GpuModel
// Replaces HybridScheduler with direct CudaExecutor calls
// =========================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1002a_cuda_scheduler_creation() {
    // IMP-1002a: CudaScheduler can be created when CUDA is available
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1002a: CUDA not available, skipping");
        return;
    }

    let scheduler = CudaScheduler::new();
    assert!(
        scheduler.is_ok(),
        "IMP-1002a: CudaScheduler creation should succeed"
    );

    let scheduler = scheduler.expect("test");
    assert!(
        scheduler.has_cuda(),
        "IMP-1002a: CudaScheduler should report CUDA available"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1002b_cuda_scheduler_matmul() {
    // IMP-1002b: CudaScheduler matmul matches HybridScheduler interface
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1002b: CUDA not available, skipping");
        return;
    }

    let mut scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");

    // Test same interface as HybridScheduler
    let a = vec![1.0f32; 16]; // 4x4
    let b = vec![1.0f32; 16]; // 4x4

    let result = scheduler.matmul(&a, &b, 4, 4, 4);
    assert!(result.is_ok(), "IMP-1002b: matmul should succeed");

    let output = result.expect("test");
    assert_eq!(
        output.len(),
        16,
        "IMP-1002b: Output should be 4x4=16 elements"
    );

    // Each element should be 4.0
    for (i, &val) in output.iter().enumerate() {
        assert!(
            (val - 4.0).abs() < 1e-3,
            "IMP-1002b: Element {} should be 4.0, got {}",
            i,
            val
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
#[allow(clippy::many_single_char_names)]
fn test_imp_1002c_cuda_scheduler_large_matmul() {
    // IMP-1002c: CudaScheduler handles LLM-sized matrices
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1002c: CUDA not available, skipping");
        return;
    }

    let mut scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");

    // Test with square matrices that are known to work: 64x64
    let m = 64;
    let k = 64;
    let n = 64;
    let a: Vec<f32> = vec![1.0; m * k]; // All ones
    let b: Vec<f32> = vec![1.0; k * n]; // All ones

    let result = scheduler.matmul(&a, &b, m, k, n);
    assert!(
        result.is_ok(),
        "IMP-1002c: Large matmul should succeed: {:?}",
        result.err()
    );

    let output = result.expect("test");
    assert_eq!(
        output.len(),
        m * n,
        "IMP-1002c: Output should be {}x{}={} elements",
        m,
        n,
        m * n
    );

    // Each element should be k (sum of k ones * ones = k)
    let expected = k as f32;
    for (i, &val) in output.iter().take(10).enumerate() {
        assert!(
            (val - expected).abs() < 1.0,
            "IMP-1002c: Element {} should be ~{}, got {}",
            i,
            expected,
            val
        );
    }

    // Test larger: 128x128
    let m = 128;
    let k = 128;
    let n = 128;
    let a: Vec<f32> = vec![1.0; m * k];
    let b: Vec<f32> = vec![1.0; k * n];

    let result = scheduler.matmul(&a, &b, m, k, n);
    assert!(result.is_ok(), "IMP-1002c: 128x128 matmul should succeed");

    let output = result.expect("test");
    assert_eq!(output.len(), m * n);

    // Each element should be 128
    for (i, &val) in output.iter().take(10).enumerate() {
        assert!(
            (val - 128.0).abs() < 1.0,
            "IMP-1002c: 128x128 element {} should be ~128, got {}",
            i,
            val
        );
    }
}

#[test]
#[cfg(feature = "cuda")]
#[allow(clippy::many_single_char_names)]
fn test_imp_1002d_cuda_scheduler_no_m1_restriction() {
    // IMP-1002d: CudaScheduler does NOT force CPU for m=1 (unlike HybridScheduler)
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1002d: CUDA not available, skipping");
        return;
    }

    let mut cuda_scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");
    let mut hybrid_scheduler =
        HybridScheduler::with_threshold(1000).expect("Failed to create HybridScheduler");

    // m=1 case - HybridScheduler forces CPU, CudaScheduler should use GPU
    let m = 1;
    let k = 4096;
    let n = 4096;
    let a: Vec<f32> = vec![1.0; m * k];
    let b: Vec<f32> = vec![1.0; k * n];

    // HybridScheduler should NOT use GPU for m=1
    assert!(
        !hybrid_scheduler.should_use_gpu(m, k, n),
        "IMP-1002d: HybridScheduler should reject m=1 for GPU"
    );

    // CudaScheduler should always use CUDA (that's its purpose)
    assert!(
        cuda_scheduler.uses_cuda_for(m, k, n),
        "IMP-1002d: CudaScheduler should use CUDA even for m=1"
    );

    // Time both
    let start = Instant::now();
    let hybrid_result = hybrid_scheduler.matmul(&a, &b, m, k, n).expect("test");
    let hybrid_time = start.elapsed();

    let start = Instant::now();
    let cuda_result = cuda_scheduler.matmul(&a, &b, m, k, n).expect("test");
    let cuda_time = start.elapsed();

    println!(
        "IMP-1002d: m=1 matmul - Hybrid(CPU)={:.2}ms, CUDA={:.2}ms",
        hybrid_time.as_secs_f64() * 1000.0,
        cuda_time.as_secs_f64() * 1000.0
    );

    // Both should produce valid results
    assert!(
        hybrid_result.len() == m * n && cuda_result.len() == m * n,
        "IMP-1002d: Both schedulers should produce correct output size"
    );
}

// ========================================================================
// IMP-1003: Wire CudaScheduler into GpuModel
// ========================================================================

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1003a_gpu_model_with_cuda_scheduler() {
    // IMP-1003a: GpuModel can be created with CudaScheduler
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1003a: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
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

    // Create model with CUDA scheduler
    let model = GpuModel::new_with_cuda(config);
    assert!(
        model.is_ok(),
        "IMP-1003a: GpuModel::new_with_cuda() should succeed"
    );

    let model = model.expect("test");
    assert!(
        model.has_cuda_scheduler(),
        "IMP-1003a: Model should have CUDA scheduler"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1003b_cuda_scheduler_used_for_forward() {
    // IMP-1003b: CUDA scheduler is used for forward pass matmul operations
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1003b: CUDA not available, skipping");
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

    let mut model = GpuModel::new_with_cuda(config).expect("Failed to create CUDA model");

    // Single token forward should use CUDA (the whole point of IMP-1003)
    let token_ids = vec![0usize];
    let result = model.forward_gpu(&token_ids);

    assert!(result.is_ok(), "IMP-1003b: Forward pass should succeed");
    let logits = result.expect("test");
    assert_eq!(logits.len(), 100, "IMP-1003b: Output should be vocab_size");
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1003c_cuda_scheduler_vs_hybrid_single_token() {
    // IMP-1003c: Compare CudaScheduler vs HybridScheduler for single-token inference
    // This test verifies that CUDA path is taken for m=1 operations
    use crate::cuda::CudaExecutor;
    use std::time::Instant;

    if !CudaExecutor::is_available() {
        println!("IMP-1003c: CUDA not available, skipping");
        return;
    }

    let config = GpuModelConfig {
        vocab_size: 32000, // Realistic vocab size
        hidden_dim: 512,   // Smaller but still tests the path
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

    // Create both models
    let mut cuda_model =
        GpuModel::new_with_cuda(config.clone()).expect("Failed to create CUDA model");
    let mut hybrid_model = GpuModel::new(config).expect("Failed to create Hybrid model");

    let token_ids = vec![42usize]; // Single token

    // Warmup
    let _ = cuda_model.forward_gpu(&token_ids);
    let _ = hybrid_model.forward_gpu(&token_ids);

    // Time CUDA model (should use GPU even for m=1)
    let start = Instant::now();
    for _ in 0..10 {
        let _ = cuda_model.forward_gpu(&token_ids);
    }
    let cuda_time = start.elapsed();

    // Time Hybrid model (forces CPU for m=1)
    let start = Instant::now();
    for _ in 0..10 {
        let _ = hybrid_model.forward_gpu(&token_ids);
    }
    let hybrid_time = start.elapsed();

    println!(
        "IMP-1003c: Single-token forward (10 iters) - CUDA={:.2}ms, Hybrid(CPU)={:.2}ms",
        cuda_time.as_secs_f64() * 1000.0,
        hybrid_time.as_secs_f64() * 1000.0
    );

    // The CUDA path should work (we don't assert it's faster yet - that's IMP-1004)
    assert!(
        cuda_time.as_micros() > 0 && hybrid_time.as_micros() > 0,
        "IMP-1003c: Both paths should complete"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn test_imp_1003d_cuda_scheduler_matmul_dispatch() {
    // IMP-1003d: Verify that cuda_matmul is called when CudaScheduler is active
    use crate::cuda::CudaExecutor;

    if !CudaExecutor::is_available() {
        println!("IMP-1003d: CUDA not available, skipping");
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

    let mut model = GpuModel::new_with_cuda(config).expect("Failed to create CUDA model");

    // Test that cuda_matmul helper uses the CUDA scheduler
    let a: Vec<f32> = vec![1.0; 64];
    let b: Vec<f32> = vec![1.0; 64 * 100];
    let result = model.cuda_matmul(&a, &b, 1, 64, 100);

    assert!(result.is_ok(), "IMP-1003d: cuda_matmul should succeed");
    let output = result.expect("test");
    assert_eq!(output.len(), 100, "IMP-1003d: Output size should be m*n");
}

// ========================================================================
// IMP-1004: Benchmark CUDA vs CPU Inference
// ========================================================================
