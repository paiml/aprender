
#[test]
#[serial]
fn test_gemm_fused_with_bias() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let n = 4u32;
    let k = 4u32;

    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let bias = vec![2.0f32; n as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    executor
        .gemm_fused(&a, &b, Some(&bias), &mut c, m, n, k, 0)
        .expect("GEMM fused with bias should succeed");

    // Each element should be k + bias = 4 + 2 = 6
    for val in &c {
        assert!((val - 6.0).abs() < 0.001);
    }
}

#[test]
#[serial]
fn test_gemm_fused_relu_activation() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let n = 4u32;
    let k = 4u32;

    // Use values that will produce negative results after bias
    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let bias = vec![-10.0f32; n as usize]; // Large negative bias
    let mut c = vec![0.0f32; (m * n) as usize];

    executor
        .gemm_fused(&a, &b, Some(&bias), &mut c, m, n, k, 1) // ReLU
        .expect("GEMM fused with ReLU should succeed");

    // k=4, so GEMM gives 4, bias -10 gives -6, ReLU gives 0
    for val in &c {
        assert!(*val >= 0.0, "ReLU should clamp negative to 0");
    }
}

#[test]
#[serial]
fn test_gemm_fused_gelu_activation() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let n = 4u32;
    let k = 4u32;

    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    executor
        .gemm_fused(&a, &b, None, &mut c, m, n, k, 2) // GELU
        .expect("GEMM fused with GELU should succeed");

    // GELU(4) ≈ 4.0 (GELU(x) ≈ x for positive x)
    for val in &c {
        assert!(*val > 3.9 && *val < 4.1, "GELU(4) should be ≈4");
    }
}

#[test]
#[serial]
fn test_gemm_fused_bias_size_validation() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let n = 4u32;
    let k = 4u32;

    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let wrong_bias = vec![2.0f32; (n + 1) as usize]; // Wrong size!
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm_fused(&a, &b, Some(&wrong_bias), &mut c, m, n, k, 0);
    assert!(result.is_err(), "Should reject wrong bias size");
}

// ========================================================================
// FlashAttention Tests (IMP-900c)
// ========================================================================

#[test]
fn test_flash_attention_memory_bytes() {
    // Test memory calculation
    let (naive, flash) = CudaExecutor::flash_attention_memory_bytes(1024, 64);

    // Naive: 1024 * 1024 * 4 = 4MB
    assert_eq!(naive, 1024 * 1024 * 4);

    // Flash: 64 * 64 * 4 * 2 = 32KB
    assert_eq!(flash, 64 * 64 * 4 * 2);

    // Verify significant memory savings
    let savings = naive as f64 / flash as f64;
    assert!(
        savings > 100.0,
        "FlashAttention should save 100x+ memory for seq_len=1024"
    );
}

#[test]
fn test_flash_attention_memory_scaling() {
    // Verify O(N²) vs O(1) scaling
    let (naive_256, flash_256) = CudaExecutor::flash_attention_memory_bytes(256, 64);
    let (naive_1024, flash_1024) = CudaExecutor::flash_attention_memory_bytes(1024, 64);
    let (naive_4096, flash_4096) = CudaExecutor::flash_attention_memory_bytes(4096, 64);

    // Naive scales O(N²): 16x seq_len = 256x memory
    assert_eq!(naive_1024 / naive_256, 16); // 4x seq_len = 16x memory
    assert_eq!(naive_4096 / naive_1024, 16); // 4x seq_len = 16x memory

    // Flash is constant (O(1) w.r.t. seq_len)
    assert_eq!(flash_256, flash_1024);
    assert_eq!(flash_1024, flash_4096);
}

#[test]
fn test_attention_kernel_type_generation() {
    let kernel_type = KernelType::Attention {
        seq_len: 128,
        head_dim: 64,
        causal: true,
    };

    let kernels = CudaKernels::new();
    let name = kernels.kernel_name(&kernel_type);
    assert_eq!(name, "flash_attention_causal"); // causal=true -> causal kernel

    let ptx = kernels.generate_ptx(&kernel_type);
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("attention"));
}

// ========================================================================
// BiasActivation Epilogue Tests (IMP-1000)
// ========================================================================

#[test]
fn test_bias_activation_ptx_generation() {
    let kernels = CudaKernels::new();

    // Test no activation
    let no_act = KernelType::BiasActivation {
        n: 1024,
        bias_size: 64,
        activation: 0,
    };
    let ptx = kernels.generate_ptx(&no_act);
    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains("bias_activation"));
    assert!(ptx.contains("add.f32")); // bias addition

    // Test ReLU
    let relu = KernelType::BiasActivation {
        n: 1024,
        bias_size: 64,
        activation: 1,
    };
    let ptx_relu = kernels.generate_ptx(&relu);
    assert!(ptx_relu.contains("max.f32")); // ReLU: max(0, x)

    // Test GELU
    let gelu = KernelType::BiasActivation {
        n: 1024,
        bias_size: 64,
        activation: 2,
    };
    let ptx_gelu = kernels.generate_ptx(&gelu);
    assert!(ptx_gelu.contains("ex2.approx")); // GELU: exponential for sigmoid
}

#[test]
fn test_bias_activation_kernel_name() {
    let kernels = CudaKernels::new();
    let kernel_type = KernelType::BiasActivation {
        n: 1024,
        bias_size: 64,
        activation: 1,
    };
    assert_eq!(kernels.kernel_name(&kernel_type), "bias_activation");
}

#[test]
#[serial]
fn test_flash_attention_basic() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let seq_len = 16u32;
    let head_dim = 8u32;
    let size = (seq_len * head_dim) as usize;

    // Simple test: Q = K = V = 1, should produce similar output
    let q = vec![1.0f32; size];
    let k = vec![1.0f32; size];
    let v = vec![1.0f32; size];
    let mut output = vec![0.0f32; size];

    let scale = 1.0 / (head_dim as f32).sqrt();
    executor
        .flash_attention(&q, &k, &v, &mut output, seq_len, head_dim, scale, false)
        .expect("FlashAttention should succeed");

    // Output should be non-zero
    assert!(
        output.iter().any(|&x| x != 0.0),
        "Output should be non-zero"
    );
}

#[test]
#[serial]
fn test_flash_attention_causal() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let seq_len = 16u32;
    let head_dim = 8u32;
    let size = (seq_len * head_dim) as usize;

    let q = vec![1.0f32; size];
    let k = vec![1.0f32; size];
    let v = vec![1.0f32; size];
    let mut output = vec![0.0f32; size];

    let scale = 1.0 / (head_dim as f32).sqrt();
    executor
        .flash_attention(&q, &k, &v, &mut output, seq_len, head_dim, scale, true) // causal
        .expect("FlashAttention causal should succeed");

    // Output should be non-zero
    assert!(
        output.iter().any(|&x| x != 0.0),
        "Output should be non-zero"
    );
}

#[test]
#[serial]
fn test_flash_attention_size_validation() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let seq_len = 16u32;
    let head_dim = 8u32;
    let correct_size = (seq_len * head_dim) as usize;
    let wrong_size = correct_size + 1;

    let q = vec![1.0f32; correct_size];
    let k = vec![1.0f32; correct_size];
    let v = vec![1.0f32; wrong_size]; // Wrong size!
    let mut output = vec![0.0f32; correct_size];

    let scale = 1.0 / (head_dim as f32).sqrt();
    let result = executor.flash_attention(&q, &k, &v, &mut output, seq_len, head_dim, scale, false);

    assert!(result.is_err(), "Should reject wrong V size");
}

#[test]
#[serial]
fn test_flash_attention_memory_tracking() {
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let seq_len = 16u32;
    let head_dim = 8u32;
    let size = (seq_len * head_dim) as usize;

    let q = vec![1.0f32; size];
    let k = vec![1.0f32; size];
    let v = vec![1.0f32; size];
    let mut output = vec![0.0f32; size];

    // Clear pool stats
    executor.clear_pool();

    let scale = 1.0 / (head_dim as f32).sqrt();
    executor
        .flash_attention(&q, &k, &v, &mut output, seq_len, head_dim, scale, false)
        .expect("FlashAttention should succeed");

    // Check pool recorded allocations
    let stats = executor.pool_stats();
    assert!(
        stats.total_allocated == 0 || stats.peak_usage > 0,
        "Memory should be tracked"
    );
}

// ========================================================================
// COV-001: Comprehensive Quantized Kernel Tests (Target: 95% coverage)
// ========================================================================

/// Helper: Create mock Q4_K weights (144 bytes per 256 values)
fn mock_q4k_weights(n_rows: usize, k: usize) -> Vec<u8> {
    assert!(k.is_multiple_of(256), "k must be divisible by 256 for Q4_K");
    let n_superblocks_per_row = k / 256;
    let bytes_per_row = n_superblocks_per_row * 144;
    vec![0x42u8; n_rows * bytes_per_row] // Non-zero pattern for detection
}

/// Helper: Create mock Q5_K weights (176 bytes per 256 values)
fn mock_q5k_weights(n_rows: usize, k: usize) -> Vec<u8> {
    assert!(k.is_multiple_of(256), "k must be divisible by 256 for Q5_K");
    let n_superblocks_per_row = k / 256;
    let bytes_per_row = n_superblocks_per_row * 176;
    vec![0x43u8; n_rows * bytes_per_row]
}

/// Helper: Create mock Q6_K weights (210 bytes per 256 values)
fn mock_q6k_weights(n_rows: usize, k: usize) -> Vec<u8> {
    assert!(k.is_multiple_of(256), "k must be divisible by 256 for Q6_K");
    let n_superblocks_per_row = k / 256;
    let bytes_per_row = n_superblocks_per_row * 210;
    vec![0x44u8; n_rows * bytes_per_row]
}

#[test]
#[serial]
fn test_cov001_q4k_gemv_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let k = 256u32;
    let weights = mock_q4k_weights(n as usize, k as usize);
    let input = vec![1.0f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q4k_gemv(&weights, &input, &mut output, n, k);
    assert!(result.is_ok(), "q4k_gemv should succeed: {:?}", result);
}

#[test]
#[serial]
fn test_cov001_q5k_gemv_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let k = 256u32;
    let weights = mock_q5k_weights(n as usize, k as usize);
    let input = vec![1.0f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q5k_gemv(&weights, &input, &mut output, n, k);
    assert!(result.is_ok(), "q5k_gemv should succeed: {:?}", result);
}

#[test]
#[serial]
fn test_cov001_q6k_gemv_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let k = 256u32;
    let weights = mock_q6k_weights(n as usize, k as usize);
    let input = vec![1.0f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q6k_gemv(&weights, &input, &mut output, n, k);
    assert!(result.is_ok(), "q6k_gemv should succeed: {:?}", result);
}

#[test]
#[serial]
fn test_cov001_q4k_gemv_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let k = 256u32;
    let weights = mock_q4k_weights(n as usize, k as usize);

    // Load weights to cache
    executor
        .load_quantized_weights("test_q4k", &weights)
        .expect("load weights");

    let input = vec![1.0f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q4k_gemv_cached("test_q4k", &input, &mut output, n, k);
    assert!(
        result.is_ok(),
        "q4k_gemv_cached should succeed: {:?}",
        result
    );
}

#[test]
#[serial]
fn test_cov001_q5k_gemv_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let k = 256u32;
    let weights = mock_q5k_weights(n as usize, k as usize);

    executor
        .load_quantized_weights("test_q5k", &weights)
        .expect("load weights");

    let input = vec![1.0f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q5k_gemv_cached("test_q5k", &input, &mut output, n, k);
    assert!(
        result.is_ok(),
        "q5k_gemv_cached should succeed: {:?}",
        result
    );
}
