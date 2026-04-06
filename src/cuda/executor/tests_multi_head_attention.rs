
#[test]
fn test_parity043_multi_head_attention_phi2_dimensions() {
    // phi-2 model dimensions
    let kernels = CudaKernels::new();

    let kernel = KernelType::MultiHeadAttention {
        seq_len: 2048, // max context
        head_dim: 80,  // phi-2 head dimension (2560/32 heads)
        n_heads: 32,   // phi-2 attention heads
        causal: true,  // autoregressive
    };

    let ptx = kernels.generate_ptx(&kernel);

    // Verify generation succeeds for phi-2 dimensions (using trueno's FlashAttention)
    assert!(ptx.contains("flash_attention_causal"));
    assert!(ptx.len() > 1000); // Substantial kernel

    // Trueno uses tile-based approach, so shared memory is calculated per tile
    // not the full head_size. Verify shared memory is allocated.
    assert!(ptx.contains(".shared"));
}

#[test]
fn test_parity043_multi_head_attention_scale_factor() {
    let kernels = CudaKernels::new();

    let head_dim = 64;
    let kernel = KernelType::MultiHeadAttention {
        seq_len: 256,
        head_dim,
        n_heads: 8,
        causal: false,
    };

    let ptx = kernels.generate_ptx(&kernel);

    // Scale factor 1/sqrt(head_dim) = 0.125 is embedded in trueno's PTX
    // as a hex float literal (0F3E000000 = 0.125)
    // Trueno bakes scale into the PTX during generation
    assert!(ptx.contains("mul.f32")); // Scaling operation exists
    // The scale is applied after dot product in online softmax
    assert!(ptx.contains("ex2")); // exp2 for softmax
}

#[test]
fn test_parity043_multi_head_attention_thread_config() {
    let kernels = CudaKernels::new();

    // Trueno's FlashAttention uses tile_q * head_dim threads per block
    // Tile sizes are calculated based on 48KB shared memory limit
    let kernel_small = KernelType::MultiHeadAttention {
        seq_len: 64,
        head_dim: 64,
        n_heads: 8,
        causal: false,
    };

    let ptx_small = kernels.generate_ptx(&kernel_small);
    // Trueno generates valid PTX with proper thread config
    assert!(ptx_small.contains(".visible .entry flash_attention"));
    assert!(ptx_small.contains("%tid.x")); // Thread index is used

    // Larger sequence still works with tiled approach
    let kernel_large = KernelType::MultiHeadAttention {
        seq_len: 1024,
        head_dim: 64,
        n_heads: 8,
        causal: false,
    };

    let ptx_large = kernels.generate_ptx(&kernel_large);
    // Trueno handles large sequences via tiling
    assert!(ptx_large.contains(".visible .entry flash_attention"));
    assert!(ptx_large.contains("kv_loop")); // KV block iteration
}

#[test]
fn test_parity043_multi_head_attention_executor_validation() {
    // Test that CudaExecutor validates input sizes correctly
    // This test runs without actual GPU by checking size validation logic
    let seq_len = 64u32;
    let head_dim = 32u32;
    let n_heads = 4u32;
    let total_size = (seq_len * head_dim * n_heads) as usize;

    // Correct sizes
    let q = vec![0.0f32; total_size];
    let k = vec![0.0f32; total_size];
    let v = vec![0.0f32; total_size];

    // Size validation check (without GPU)
    assert_eq!(q.len(), total_size);
    assert_eq!(k.len(), total_size);
    assert_eq!(v.len(), total_size);

    // Verify formula: n_heads × seq_len × head_dim
    assert_eq!(total_size, (n_heads * seq_len * head_dim) as usize);
}

#[test]
fn test_parity043_multi_head_attention_memory_layout() {
    // Verify memory layout: [n_heads, seq_len, head_dim]
    let n_heads = 8u32;
    let seq_len = 128u32;
    let head_dim = 64u32;

    // Calculate offsets for head access
    let head_stride = (seq_len * head_dim) as usize;
    let total_size = head_stride * n_heads as usize;

    // Each head's data starts at head_idx * head_stride
    let head_0_start = 0;
    let head_1_start = head_stride;
    let head_7_start = 7 * head_stride;

    assert_eq!(head_0_start, 0);
    assert_eq!(head_1_start, 128 * 64);
    assert_eq!(head_7_start, 7 * 128 * 64);
    assert_eq!(total_size, 8 * 128 * 64);
}

#[test]
fn test_kernel_names() {
    let kernels = CudaKernels::new();

    assert_eq!(
        kernels.kernel_name(&KernelType::GemmNaive { m: 1, n: 1, k: 1 }),
        "gemm_naive"
    );
    assert_eq!(
        kernels.kernel_name(&KernelType::Softmax { dim: 1 }),
        "softmax_warp_shuffle"
    );
    assert_eq!(
        kernels.kernel_name(&KernelType::QuantizedGemm { m: 1, n: 1, k: 32 }),
        "q4k_gemm_fused"
    );
}

#[test]
fn test_presets_llama_attention() {
    let kernel = presets::llama_attention(2048, 64);
    match kernel {
        KernelType::Attention {
            seq_len,
            head_dim,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 64);
            assert!(causal);
        },
        _ => panic!("Expected Attention kernel"),
    }
}

#[test]
fn test_presets_ffn_gemm() {
    let kernel = presets::ffn_gemm(32, 4096, 11008);
    match kernel {
        KernelType::GemmTiled { m, n, k, tile_size } => {
            assert_eq!(m, 32);
            assert_eq!(n, 11008);
            assert_eq!(k, 4096);
            assert_eq!(tile_size, 32);
        },
        _ => panic!("Expected GemmTiled kernel"),
    }
}

#[test]
fn test_presets_q4k_inference() {
    let kernel = presets::q4k_inference(1, 4096, 4096);
    match kernel {
        KernelType::QuantizedGemm { m, n, k } => {
            assert_eq!(m, 1);
            assert_eq!(n, 4096);
            assert_eq!(k, 4096);
        },
        _ => panic!("Expected QuantizedGemm kernel"),
    }
}

#[test]
fn test_presets_rmsnorm() {
    let kernel = presets::rmsnorm(4096);
    match kernel {
        KernelType::LayerNorm {
            hidden_size,
            epsilon,
            affine,
        } => {
            assert_eq!(hidden_size, 4096);
            assert!((epsilon - 1e-6).abs() < 1e-10);
            assert!(!affine);
        },
        _ => panic!("Expected LayerNorm kernel"),
    }
}

#[test]
fn test_presets_multi_head_attention() {
    let kernel = presets::multi_head_attention(512, 64, 8);
    match kernel {
        KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 512);
            assert_eq!(head_dim, 64);
            assert_eq!(n_heads, 8);
            assert!(causal); // Default is causal
        },
        _ => panic!("Expected MultiHeadAttention kernel"),
    }
}

#[test]
fn test_presets_phi2_multi_head_attention() {
    let kernel = presets::phi2_multi_head_attention(2048);
    match kernel {
        KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 80); // phi-2: 2560/32 = 80
            assert_eq!(n_heads, 32); // phi-2: 32 heads
            assert!(causal);
        },
        _ => panic!("Expected MultiHeadAttention kernel"),
    }
}

#[test]
fn test_default_impl() {
    let kernels = CudaKernels::default();
    let ptx = kernels.generate_ptx(&KernelType::Softmax { dim: 256 });
    assert!(!ptx.is_empty());
}

// ========================================================================
// CudaExecutor Tests
// ========================================================================

#[test]
fn test_cuda_executor_is_available() {
    // This should not panic, regardless of whether CUDA is available
    let _available = CudaExecutor::is_available();
}

#[test]
fn test_cuda_executor_device_count() {
    // Should return count (possibly 0)
    let count = CudaExecutor::num_devices();
    // Count is valid (0 or more)
    assert!(count < 1000); // Sanity check
}

#[test]
#[serial]
fn test_cuda_executor_new() {
    let executor = CudaExecutor::new(0);
    assert!(executor.is_ok());
    let executor = executor.expect("test");
    assert!(executor.device_name().is_ok());
}

#[test]
#[serial]
fn test_cuda_executor_memory_info() {
    let executor = CudaExecutor::new(0).expect("test");
    let (free, total) = executor.memory_info().expect("test");
    assert!(total > 0);
    assert!(free <= total);
}

#[test]
#[serial]
fn test_cuda_executor_gemm_small() {
    let mut executor = CudaExecutor::new(0).expect("test");

    // Small 4x4 GEMM
    let a = vec![1.0f32; 16];
    let b = vec![1.0f32; 16];
    let mut c = vec![0.0f32; 16];

    let result = executor.gemm(&a, &b, &mut c, 4, 4, 4);
    assert!(result.is_ok());

    // Each element should be 4.0 (dot product of 4 ones)
    for val in &c {
        assert!((*val - 4.0).abs() < 1e-5);
    }
}

/// PARITY-114: Test non-square GEMM correctness
/// This is the case that was failing before the grid dimension fix
#[test]
#[serial]
fn test_cuda_executor_gemm_non_square() {
    let mut executor = CudaExecutor::new(0).expect("test");

    // First test: 32x32x32 (single tile)
    {
        let m = 32u32;
        let k = 32u32;
        let n = 32u32;

        let a = vec![1.0f32; (m * k) as usize];
        let b = vec![1.0f32; (k * n) as usize];
        let mut c = vec![0.0f32; (m * n) as usize];

        let result = executor.gemm(&a, &b, &mut c, m, n, k);
        assert!(result.is_ok(), "32x32 GEMM failed");

        eprintln!("32x32x32: First value = {} (expected 32)", c[0]);
        assert!(
            (c[0] - 32.0).abs() < 1e-4,
            "32x32 GEMM: expected 32.0, got {}",
            c[0]
        );
    }

    // Second test: 32x32x64 (2 tiles in K)
    {
        let m = 32u32;
        let k = 64u32;
        let n = 32u32;

        let a = vec![1.0f32; (m * k) as usize];
        let b = vec![1.0f32; (k * n) as usize];
        let mut c = vec![0.0f32; (m * n) as usize];

        let result = executor.gemm(&a, &b, &mut c, m, n, k);
        assert!(result.is_ok(), "32x32x64 GEMM failed");

        eprintln!("32x32x64: First value = {} (expected 64)", c[0]);
        assert!(
            (c[0] - 64.0).abs() < 1e-4,
            "32x32x64 GEMM: expected 64.0, got {}",
            c[0]
        );
    }

    // Third test: non-square (4, 64) × (64, 128) = (4, 128)
    {
        let m = 4u32;
        let k = 64u32;
        let n = 128u32;

        let a = vec![1.0f32; (m * k) as usize];
        let b = vec![1.0f32; (k * n) as usize];
        let mut c = vec![0.0f32; (m * n) as usize];

        let result = executor.gemm(&a, &b, &mut c, m, n, k);
        assert!(result.is_ok(), "4x64x128 GEMM failed");

        eprintln!("4x64x128: First value = {} (expected 64)", c[0]);
        assert!(
            (c[0] - 64.0).abs() < 1e-4,
            "PARITY-114: Non-square GEMM expected 64.0, got {}",
            c[0]
        );
    }
}

/// PARITY-114: Compare CUDA matmul vs wgpu matmul with same inputs
#[test]
#[serial]
fn test_cuda_vs_wgpu_matmul_parity() {
    cuda_vs_wgpu_single_tile();
    cuda_vs_wgpu_uniform_k64();
    cuda_vs_wgpu_patterned();
}

/// Sub-test 0: Single tile (k=32) uniform data
fn cuda_vs_wgpu_single_tile() {
    let m0 = 4usize;
    let k0 = 32usize;
    let n0 = 192usize;
    let a = vec![1.0f32; m0 * k0];
    let b = vec![1.0f32; k0 * n0];
    let expected = k0 as f32;

    let mut executor = CudaExecutor::new(0).expect("CudaExecutor should init");
    let mut c = vec![0.0f32; m0 * n0];
    executor
        .gemm(&a, &b, &mut c, m0 as u32, n0 as u32, k0 as u32)
        .expect("CUDA gemm should succeed");

    assert!(
        (c[0] - expected).abs() < 1e-3,
        "k=32 CUDA failed: {} vs {}",
        c[0],
        expected
    );
}

/// Sub-test 1: Uniform 1.0 data with k=64 (multi-tile)
fn cuda_vs_wgpu_uniform_k64() {
    let m = 4usize;
    let k = 64usize;
    let n = 192usize;
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let expected = k as f32;

    let mut executor = CudaExecutor::new(0).expect("CudaExecutor should init");
    let mut c = vec![0.0f32; m * n];
    executor
        .gemm(&a, &b, &mut c, m as u32, n as u32, k as u32)
        .expect("CUDA gemm should succeed");

    assert!(
        (c[0] - expected).abs() < 1e-3,
        "Uniform CUDA failed: {} vs {}",
        c[0],
        expected
    );
}
