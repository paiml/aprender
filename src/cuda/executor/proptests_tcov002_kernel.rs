
// =========================================================================
// T-COV-002: kernel_name() coverage - verify names are non-empty
// =========================================================================

#[test]
fn test_tcov002_kernel_names_non_empty() {
    let kernels = CudaKernels::new();

    // Test kernel types return valid non-empty names
    let test_cases: Vec<KernelType> = vec![
        KernelType::GemmNaive {
            m: 64,
            n: 64,
            k: 64,
        },
        KernelType::GemmTiled {
            m: 64,
            n: 64,
            k: 64,
            tile_size: 32,
        },
        KernelType::GemmTensorCore {
            m: 64,
            n: 64,
            k: 64,
        },
        KernelType::Gemv { k: 4096, n: 4096 },
        KernelType::Softmax { dim: 4096 },
        KernelType::LayerNorm {
            hidden_size: 4096,
            epsilon: 1e-5,
            affine: true,
        },
        KernelType::Attention {
            seq_len: 128,
            head_dim: 64,
            causal: false,
        },
        KernelType::RmsNorm {
            hidden_size: 4096,
            epsilon: 1e-6,
        },
        KernelType::ResidualAdd { n: 4096 },
    ];

    for kernel_type in test_cases {
        let name = kernels.kernel_name(&kernel_type);
        assert!(
            !name.is_empty(),
            "KernelType {:?} should have non-empty name",
            kernel_type
        );
    }
}

// =========================================================================
// T-COV-003: WeightQuantType coverage
// =========================================================================

#[test]
fn test_tcov003a_weight_quant_type_from_ggml() {
    // Test all GGML type mappings
    assert_eq!(
        WeightQuantType::from_ggml_type(2),
        Some(WeightQuantType::Q4_0)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(3),
        Some(WeightQuantType::Q4_1)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(6),
        Some(WeightQuantType::Q5_0)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(8),
        Some(WeightQuantType::Q8_0)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(12),
        Some(WeightQuantType::Q4K)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(13),
        Some(WeightQuantType::Q5K)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(14),
        Some(WeightQuantType::Q6K)
    );

    // Unknown types
    assert_eq!(WeightQuantType::from_ggml_type(255), None);
    // GH-374: GGML type 0 is F32 (used by APR LM head checkpoints)
    assert_eq!(
        WeightQuantType::from_ggml_type(0),
        Some(WeightQuantType::F32)
    );
}

#[test]
fn test_tcov003b_weight_quant_type_bytes() {
    // Q4_K: 256 values, 144 bytes per superblock
    assert_eq!(WeightQuantType::Q4K.bytes_per_superblock(), 144);

    // Q5_K: 256 values, 176 bytes per superblock
    assert_eq!(WeightQuantType::Q5K.bytes_per_superblock(), 176);

    // Q6_K: 256 values, 210 bytes per superblock
    assert_eq!(WeightQuantType::Q6K.bytes_per_superblock(), 210);

    // Q4_0: 32 values, 18 bytes per block (8 blocks = 144 bytes per 256)
    assert_eq!(WeightQuantType::Q4_0.bytes_per_superblock(), 18 * 8);

    // Q4_1: 32 values, 20 bytes per block
    assert_eq!(WeightQuantType::Q4_1.bytes_per_superblock(), 20 * 8);

    // Q5_0: 32 values, 22 bytes per block
    assert_eq!(WeightQuantType::Q5_0.bytes_per_superblock(), 22 * 8);

    // Q8_0: 32 values, 34 bytes per block
    assert_eq!(WeightQuantType::Q8_0.bytes_per_superblock(), 34 * 8);
}

#[test]
fn test_tcov003c_weight_quant_type_matches_size() {
    // Q4_K: n_rows × n_cols / 256 superblocks × 144 bytes
    let rows = 4096;
    let cols = 4096;
    let q4k_size = (rows * cols / 256) * 144;
    assert!(WeightQuantType::Q4K.matches_size(q4k_size, rows, cols));

    // Q6_K: n_rows × n_cols / 256 superblocks × 210 bytes
    let q6k_size = (rows * cols / 256) * 210;
    assert!(WeightQuantType::Q6K.matches_size(q6k_size, rows, cols));

    // Wrong size should not match
    assert!(!WeightQuantType::Q4K.matches_size(q6k_size, rows, cols));
}

#[test]
fn test_tcov003d_weight_quant_type_from_size() {
    let rows = 4096;
    let cols = 4096;

    // Q4_K detection
    let q4k_size = (rows * cols / 256) * 144;
    let detected = WeightQuantType::from_size(q4k_size, rows, cols);
    assert_eq!(detected, Some(WeightQuantType::Q4K));

    // Q6_K detection
    let q6k_size = (rows * cols / 256) * 210;
    let detected = WeightQuantType::from_size(q6k_size, rows, cols);
    assert_eq!(detected, Some(WeightQuantType::Q6K));

    // Q8_0 detection (small block format)
    let q8_0_size = (rows * cols / 32) * 34;
    let detected = WeightQuantType::from_size(q8_0_size, rows, cols);
    assert_eq!(detected, Some(WeightQuantType::Q8_0));
}

// =========================================================================
// T-COV-004: SizeClass coverage
// =========================================================================

#[test]
fn test_tcov004a_size_class_for_size() {
    // Various sizes
    let small = SizeClass::for_size(1024);
    assert!(small.is_some());

    let medium = SizeClass::for_size(64 * 1024);
    assert!(medium.is_some());

    let large = SizeClass::for_size(1024 * 1024);
    assert!(large.is_some());

    // Very large sizes may or may not be supported
    let very_large = SizeClass::for_size(200_000_000);
    // Just verify it doesn't panic
    let _ = very_large;
}

#[test]
fn test_tcov004b_size_class_bytes() {
    // Get a size class and verify bytes() returns a value
    if let Some(class) = SizeClass::for_size(1024) {
        let bytes = class.bytes();
        assert!(bytes >= 1024);
    }
}

// =========================================================================
// T-COV-005: GpuMemoryPool coverage
// =========================================================================

#[test]
fn test_tcov005a_gpu_memory_pool_basic() {
    let mut pool = GpuMemoryPool::new();

    // Record allocation/deallocation
    pool.record_allocation(1024);
    pool.record_deallocation(1024);
}

#[test]
fn test_tcov005b_gpu_memory_pool_with_max_size() {
    let max_size = 128 * 1024 * 1024;
    let pool = GpuMemoryPool::with_max_size(max_size);
    // Pool created with custom max size
    let stats = pool.stats();
    assert_eq!(stats.pool_hits, 0);
    assert_eq!(stats.pool_misses, 0);
}

#[test]
fn test_tcov005c_gpu_memory_pool_stats() {
    let mut pool = GpuMemoryPool::new();

    // Record several operations
    for i in 0..10 {
        pool.record_allocation((i + 1) * 1024);
    }

    let stats = pool.stats();
    assert!(stats.peak_usage > 0);
}

#[test]
fn test_tcov005d_gpu_memory_pool_try_get() {
    let mut pool = GpuMemoryPool::new();

    // Try to get a buffer - should fail (empty pool)
    let result = pool.try_get(1024);
    assert!(result.is_none()); // Pool starts empty

    // Verify miss was recorded
    let stats = pool.stats();
    assert_eq!(stats.pool_misses, 1);
}

// =========================================================================
// T-COV-006: StagingBufferPool extended coverage
// =========================================================================

#[test]
fn test_tcov006a_staging_pool_with_max_size() {
    let max_size = 64 * 1024 * 1024;
    let pool = StagingBufferPool::with_max_size(max_size);
    let stats = pool.stats();
    assert_eq!(stats.free_buffers, 0);
}

#[test]
fn test_tcov006b_staging_pool_size_classes() {
    let mut pool = StagingBufferPool::new();

    // Request different size classes
    let buf_tiny = pool.get(512);
    let buf_small = pool.get(8 * 1024);
    let buf_medium = pool.get(128 * 1024);

    // Return them
    pool.put(buf_tiny);
    pool.put(buf_small);
    pool.put(buf_medium);

    let stats = pool.stats();
    assert!(stats.free_buffers >= 3);
}

// =========================================================================
// T-COV-007: Presets module coverage
// =========================================================================

#[test]
fn test_tcov007_presets_coverage() {
    // Test all preset functions
    let llama_attn = presets::llama_attention(2048, 128);
    match llama_attn {
        KernelType::Attention {
            seq_len,
            head_dim,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 128);
            assert!(causal);
        },
        _ => panic!("Expected Attention kernel"),
    }

    let ffn = presets::ffn_gemm(1, 4096, 11008);
    match ffn {
        KernelType::GemmTiled { m, n, k, tile_size } => {
            assert_eq!(m, 1);
            assert_eq!(n, 11008);
            assert_eq!(k, 4096);
            assert_eq!(tile_size, 32);
        },
        _ => panic!("Expected GemmTiled kernel"),
    }

    let q4k = presets::q4k_inference(1, 4096, 4096);
    match q4k {
        KernelType::QuantizedGemm { m, n, k } => {
            assert_eq!(m, 1);
            assert_eq!(n, 4096);
            assert_eq!(k, 4096);
        },
        _ => panic!("Expected QuantizedGemm kernel"),
    }

    let rmsnorm = presets::rmsnorm(4096);
    match rmsnorm {
        KernelType::LayerNorm {
            hidden_size,
            epsilon,
            affine,
        } => {
            assert_eq!(hidden_size, 4096);
            assert!(epsilon > 0.0);
            assert!(!affine); // RMSNorm preset uses affine=false
        },
        _ => panic!("Expected LayerNorm kernel (preset::rmsnorm returns LayerNorm)"),
    }

    let mha = presets::multi_head_attention(2048, 64, 32);
    match mha {
        KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 64);
            assert_eq!(n_heads, 32);
            assert!(causal);
        },
        _ => panic!("Expected MultiHeadAttention kernel"),
    }

    let phi2_mha = presets::phi2_multi_head_attention(2048);
    match phi2_mha {
        KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 80);
            assert_eq!(n_heads, 32);
            assert!(causal);
        },
        _ => panic!("Expected MultiHeadAttention kernel"),
    }

    let tc_attn = presets::tensor_core_attention(2048, 64, 32);
    match tc_attn {
        KernelType::AttentionTensorCore {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 64);
            assert_eq!(n_heads, 32);
            assert!(causal);
        },
        _ => panic!("Expected AttentionTensorCore kernel"),
    }

    let llama_tc = presets::llama_tensor_core_attention(2048);
    match llama_tc {
        KernelType::AttentionTensorCore {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 128);
            assert_eq!(n_heads, 32);
            assert!(causal);
        },
        _ => panic!("Expected AttentionTensorCore kernel"),
    }
}

// =========================================================================
// T-COV-009: CudaKernels::cuda_likely_available() coverage
// =========================================================================

#[test]
fn test_tcov009_cuda_likely_available() {
    // This function checks environment heuristics
    let likely = CudaKernels::cuda_likely_available();
    // On RTX 4090, this should return true
    // The function itself should not panic
    println!("cuda_likely_available: {}", likely);
}
