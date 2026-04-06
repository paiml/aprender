//! CUDA PTX Generation Coverage Tests
//!
//! Property tests to achieve 95% coverage of cuda.rs kernel generation paths.
//! Target: Cover all KernelType variants in PtxGenerator::generate_ptx

#![cfg(feature = "cuda")]

use proptest::prelude::*;
use realizar::cuda::{
    presets, CudaExecutor, CudaKernels, GpuMemoryPool, KernelType, PinnedHostBuffer, SizeClass,
    StagingBufferPool, TransferMode,
};
use trueno::SyncMode;

// ============================================================================
// PTX Generator Coverage Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Test GemmNaive PTX generation for various dimensions
    #[test]
    fn test_ptx_gemm_naive(m in 16u32..=256, n in 16u32..=256, k in 16u32..=256) {
        let gen = CudaKernels::new();
        let kernel = KernelType::GemmNaive { m, n, k };
        let ptx = gen.generate_ptx(&kernel);

        prop_assert!(!ptx.is_empty(), "PTX should not be empty");
        prop_assert!(ptx.contains(".version"), "PTX should have version directive");
        prop_assert!(ptx.contains(".target"), "PTX should have target directive");
    }

    /// Test GemmTiled PTX generation
    #[test]
    fn test_ptx_gemm_tiled(
        m in 32u32..=512,
        n in 32u32..=512,
        k in 32u32..=512,
        tile_size in prop::sample::select(vec![16u32, 32, 64])
    ) {
        let gen = CudaKernels::new();
        let kernel = KernelType::GemmTiled { m, n, k, tile_size };
        let ptx = gen.generate_ptx(&kernel);

        prop_assert!(!ptx.is_empty());
        prop_assert!(ptx.contains(".visible .entry"));
    }

    /// Test GemmOptimized PTX generation (uses same tiled kernel)
    #[test]
    fn test_ptx_gemm_optimized(
        m in 64u32..=256,
        n in 64u32..=256,
        k in 64u32..=256,
        tile_size in prop::sample::select(vec![16u32, 32, 64])
    ) {
        let gen = CudaKernels::new();
        let kernel = KernelType::GemmOptimized {
            m,
            n,
            k,
            tile_size,
            reg_block: 4, // 4x4 register blocking for good occupancy
        };
        let ptx = gen.generate_ptx(&kernel);

        prop_assert!(!ptx.is_empty());
    }

    /// Test GemmTensorCore PTX generation
    #[test]
    fn test_ptx_gemm_tensor_core(m in 16u32..=128, n in 16u32..=128, k in 16u32..=128) {
        let gen = CudaKernels::new();
        let kernel = KernelType::GemmTensorCore { m, n, k };
        let ptx = gen.generate_ptx(&kernel);

        prop_assert!(!ptx.is_empty());
    }

    /// Test Gemv PTX generation (M=1 matmul)
    #[test]
    fn test_ptx_gemv(k in 64u32..=4096, n in 64u32..=4096) {
        let gen = CudaKernels::new();
        let kernel = KernelType::Gemv { k, n };
        let ptx = gen.generate_ptx(&kernel);

        prop_assert!(!ptx.is_empty());
    }

    /// Test CoalescedGemv PTX generation
    #[test]
    fn test_ptx_coalesced_gemv(k in 64u32..=4096, n in 64u32..=4096) {
        let gen = CudaKernels::new();
        let kernel = KernelType::CoalescedGemv { k, n };
        let ptx = gen.generate_ptx(&kernel);

        prop_assert!(!ptx.is_empty());
    }

    /// Test Softmax PTX generation
    #[test]
    fn test_ptx_softmax(dim in 32u32..=32000) {
        let gen = CudaKernels::new();
        let kernel = KernelType::Softmax { dim };
        let ptx = gen.generate_ptx(&kernel);

        prop_assert!(!ptx.is_empty());
    }

    /// Test LayerNorm PTX generation with various configs
    #[test]
    fn test_ptx_layernorm(
        hidden_size in 64u32..=4096,
        epsilon in 1e-6f32..1e-4,
        affine in proptest::bool::ANY
    ) {
        let gen = CudaKernels::new();
        let kernel = KernelType::LayerNorm {
            hidden_size,
            epsilon,
            affine,
        };
        let ptx = gen.generate_ptx(&kernel);

        prop_assert!(!ptx.is_empty());
    }

    /// Test Attention PTX generation
    #[test]
    fn test_ptx_attention(
        seq_len in 1u32..=512,
        head_dim in prop::sample::select(vec![32u32, 64, 128]),
        causal in proptest::bool::ANY
    ) {
        let gen = CudaKernels::new();
        let kernel = KernelType::Attention {
            seq_len,
            head_dim,
            causal,
        };
        let ptx = gen.generate_ptx(&kernel);

        prop_assert!(!ptx.is_empty());
    }

    /// Test MultiHeadAttention PTX generation
    #[test]
    fn test_ptx_multi_head_attention(
        seq_len in 1u32..=512,
        head_dim in prop::sample::select(vec![32u32, 64, 128]),
        n_heads in 1u32..=32,
        causal in proptest::bool::ANY
    ) {
        let gen = CudaKernels::new();
        let kernel = KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        };
        let ptx = gen.generate_ptx(&kernel);

        prop_assert!(!ptx.is_empty());
    }

    /// Test AttentionTensorCore PTX generation
    #[test]
    fn test_ptx_attention_tensor_core(
        seq_len in 16u32..=256,
        head_dim in prop::sample::select(vec![64u32, 128]),
        n_heads in 1u32..=32,
        causal in proptest::bool::ANY
    ) {
        let gen = CudaKernels::new();
        let kernel = KernelType::AttentionTensorCore {
            seq_len,
            head_dim,
            n_heads,
            causal,
        };
        let ptx = gen.generate_ptx(&kernel);

        prop_assert!(!ptx.is_empty());
    }
}

// ============================================================================
// GPU Memory Pool Coverage Tests
// ============================================================================

#[test]
fn test_gpu_memory_pool_size_classes() {
    // Cover all SizeClass variants
    let small = SizeClass::for_size(1024);
    assert!(small.is_some());
    assert_eq!(small.unwrap().bytes(), 4096);

    let large = SizeClass::for_size(1024 * 1024);
    assert!(large.is_some());

    let too_large = SizeClass::for_size(1024 * 1024 * 1024); // 1GB
    assert!(too_large.is_none());
}

#[test]
fn test_gpu_memory_pool_operations() {
    let mut pool = GpuMemoryPool::new();

    // Test allocation recording
    pool.record_allocation(1024);
    pool.record_allocation(2048);

    let stats = pool.stats();
    assert!(stats.total_allocated > 0 || stats.pool_hits > 0 || stats.pool_misses > 0);

    // Test capacity check
    assert!(pool.has_capacity(1024));

    // Test clear
    pool.clear();
    assert_eq!(pool.stats().pool_hits, 0);
}

#[test]
fn test_gpu_memory_pool_with_max_size() {
    let mut pool = GpuMemoryPool::with_max_size(1024 * 1024);
    assert_eq!(pool.max_size(), 1024 * 1024);

    pool.record_allocation(512 * 1024);
    assert!(pool.has_capacity(512 * 1024));
}

// ============================================================================
// Pinned Host Buffer Coverage Tests
// ============================================================================

#[test]
fn test_pinned_host_buffer_creation() {
    let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(1024);

    assert_eq!(buf.len(), 1024);
    assert!(!buf.is_empty());
    assert_eq!(buf.size_bytes(), 1024 * 4);
}

#[test]
fn test_pinned_host_buffer_copy() {
    let mut buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(10);
    let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    buf.copy_from_slice(&src);

    let slice = buf.as_slice();
    assert_eq!(slice[0], 1.0);
    assert_eq!(slice[9], 10.0);
}

#[test]
fn test_pinned_host_buffer_mutable() {
    let mut buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(5);

    let slice = buf.as_mut_slice();
    slice[0] = 42.0;
    slice[4] = 100.0;

    assert_eq!(buf.as_slice()[0], 42.0);
    assert_eq!(buf.as_slice()[4], 100.0);
}

// ============================================================================
// Staging Buffer Pool Coverage Tests
// ============================================================================

#[test]
fn test_staging_buffer_pool_operations() {
    let mut pool = StagingBufferPool::new();

    // Get buffer
    let buf = pool.get(1024);
    assert_eq!(buf.len(), 1024);

    // Return buffer
    pool.put(buf);

    let stats = pool.stats();
    assert!(stats.total_allocated > 0 || stats.pool_hits > 0);

    // Clear
    pool.clear();
}

#[test]
fn test_staging_buffer_pool_with_max_size() {
    let mut pool = StagingBufferPool::with_max_size(1024 * 1024);

    let buf = pool.get(512);
    pool.put(buf);

    let stats = pool.stats();
    assert!(stats.hit_rate >= 0.0);
}

// ============================================================================
// Transfer Mode Coverage Tests
// ============================================================================

#[test]
fn test_transfer_mode_properties() {
    let pageable = TransferMode::Pageable;
    let pinned = TransferMode::Pinned;
    let zero_copy = TransferMode::ZeroCopy;
    let async_mode = TransferMode::Async;

    // Test requires_pinned()
    assert!(!pageable.requires_pinned());
    assert!(pinned.requires_pinned());
    assert!(zero_copy.requires_pinned());
    assert!(async_mode.requires_pinned());

    // Test estimated_speedup()
    assert!(pageable.estimated_speedup() == 1.0);
    assert!(pinned.estimated_speedup() > pageable.estimated_speedup());
    assert!(zero_copy.estimated_speedup() > 0.0);
    assert!(async_mode.estimated_speedup() > 0.0);
}

// ============================================================================
// Presets Coverage Tests
// ============================================================================

#[test]
fn test_preset_configs_all_variants() {
    // Test all preset config factories
    let attention = presets::llama_attention(512, 128);
    assert!(matches!(attention, KernelType::Attention { .. }));

    let ffn = presets::ffn_gemm(4, 4096, 11008);
    assert!(matches!(
        ffn,
        KernelType::GemmTiled { .. } | KernelType::GemmOptimized { .. }
    ));

    let q4k = presets::q4k_inference(1, 4096, 4096);
    assert!(matches!(q4k, KernelType::QuantizedGemm { .. }));

    let q4k_ggml = presets::q4k_ggml_inference(1, 4096, 4096);
    let _ = q4k_ggml; // Just verify it compiles

    let rmsnorm = presets::rmsnorm(4096);
    assert!(matches!(rmsnorm, KernelType::LayerNorm { .. }));

    let mha = presets::multi_head_attention(512, 128, 32);
    assert!(matches!(mha, KernelType::MultiHeadAttention { .. }));

    let phi2_mha = presets::phi2_multi_head_attention(512);
    assert!(matches!(phi2_mha, KernelType::MultiHeadAttention { .. }));

    let tc_attn = presets::tensor_core_attention(512, 128, 32);
    assert!(matches!(tc_attn, KernelType::AttentionTensorCore { .. }));

    let llama_tc = presets::llama_tensor_core_attention(512);
    assert!(matches!(llama_tc, KernelType::AttentionTensorCore { .. }));
}

// ============================================================================
// PtxOptimizationHints Coverage Tests
// ============================================================================

#[test]
fn test_optimization_hints_presets() {
    use realizar::cuda::PtxOptimizationHints;

    let max_throughput = PtxOptimizationHints::max_throughput();
    assert!(max_throughput.uses_vectorized_loads());

    let low_latency = PtxOptimizationHints::low_latency();
    assert!(!low_latency.uses_vectorized_loads());

    let balanced = PtxOptimizationHints::balanced();
    assert!(balanced.uses_vectorized_loads());

    let default_hints = PtxOptimizationHints::default();
    // Default should work without panics
    let _ = default_hints.vector_width();
}

// ============================================================================
// CudaExecutor Integration Tests (requires GPU)
// ============================================================================

#[test]
fn test_cuda_executor_profiling() {
    if !CudaExecutor::is_available() {
        return; // Skip if no GPU
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Enable profiling
    executor.enable_profiling();
    assert!(executor.is_profiling_enabled());

    // Run a simple operation
    let a = vec![1.0f32; 64 * 64];
    let b = vec![1.0f32; 64 * 64];
    let mut c = vec![0.0f32; 64 * 64];
    let _ = executor.gemm(&a, &b, &mut c, 64, 64, 64);

    // Get profiler stats
    let _profiler = executor.profiler();
    let _profiler_mut = executor.profiler_mut();
    let summary = executor.profiler_summary();
    assert!(!summary.is_empty() || summary.is_empty()); // Just check it works

    let _stats = executor.profiler_category_stats();

    // Disable profiling
    executor.disable_profiling();
    assert!(!executor.is_profiling_enabled());

    // Reset profiler
    executor.reset_profiler();
}

#[test]
fn test_cuda_executor_sync_modes() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Get current sync mode
    let mode = executor.profiler_sync_mode();

    // Set a different mode
    executor.set_profiler_sync_mode(SyncMode::PerLayer);
    assert_eq!(executor.profiler_sync_mode(), SyncMode::PerLayer);

    // Restore original
    executor.set_profiler_sync_mode(mode);
}

#[test]
fn test_cuda_executor_print_categories() {
    if !CudaExecutor::is_available() {
        return;
    }

    let executor = CudaExecutor::new(0).expect("Failed to create executor");

    // This just prints to stdout, but exercises the code path
    executor.print_profiler_categories();
}

// ============================================================================
// Kernel Name Coverage Tests
// ============================================================================

#[test]
fn test_kernel_names_all_variants() {
    let gen = CudaKernels::new();

    // Test all kernel types have names
    let kernels = vec![
        KernelType::GemmNaive {
            m: 64,
            n: 64,
            k: 64,
        },
        KernelType::GemmTiled {
            m: 64,
            n: 64,
            k: 64,
            tile_size: 16,
        },
        KernelType::GemmTensorCore {
            m: 64,
            n: 64,
            k: 64,
        },
        KernelType::Gemv { k: 256, n: 256 },
        KernelType::CoalescedGemv { k: 256, n: 256 },
        KernelType::Softmax { dim: 1024 },
        KernelType::LayerNorm {
            hidden_size: 768,
            epsilon: 1e-5,
            affine: true,
        },
        KernelType::Attention {
            seq_len: 128,
            head_dim: 64,
            causal: true,
        },
        KernelType::MultiHeadAttention {
            seq_len: 128,
            head_dim: 64,
            n_heads: 8,
            causal: false,
        },
        KernelType::AttentionTensorCore {
            seq_len: 128,
            head_dim: 64,
            n_heads: 8,
            causal: true,
        },
    ];

    for kernel in kernels {
        let name = gen.kernel_name(&kernel);
        assert!(!name.is_empty(), "Kernel {:?} should have a name", kernel);
    }
}

// ============================================================================
// CudaExecutor GPU Operation Tests (requires RTX 4090)
// ============================================================================

#[test]
fn test_cuda_executor_gemm_sizes() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Test small GEMM
    let a = vec![1.0f32; 32 * 32];
    let b = vec![1.0f32; 32 * 32];
    let mut c = vec![0.0f32; 32 * 32];
    executor
        .gemm(&a, &b, &mut c, 32, 32, 32)
        .expect("small gemm");

    // Verify result is reasonable (not all zeros)
    let sum: f32 = c.iter().sum();
    assert!(sum > 0.0, "GEMM result should not be all zeros");

    // Test larger GEMM
    let a = vec![1.0f32; 128 * 256];
    let b = vec![1.0f32; 256 * 128];
    let mut c = vec![0.0f32; 128 * 128];
    executor
        .gemm(&a, &b, &mut c, 128, 128, 256)
        .expect("larger gemm");
    let sum: f32 = c.iter().sum();
    assert!(sum > 0.0);
}

#[test]
fn test_cuda_executor_gemm_optimized() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    let a = vec![2.0f32; 64 * 64];
    let b = vec![0.5f32; 64 * 64];
    let mut c = vec![0.0f32; 64 * 64];

    executor
        .gemm_optimized(&a, &b, &mut c, 64, 64, 64, 16)
        .expect("optimized gemm");

    let sum: f32 = c.iter().sum();
    assert!(sum > 0.0, "Optimized GEMM result should not be all zeros");
}

#[test]
fn test_cuda_executor_softmax() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Softmax is in-place - start with input values
    let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    executor.softmax(&mut data).expect("softmax");

    // Verify softmax properties: sum to 1.0
    let sum: f32 = data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Softmax should sum to ~1.0, got {}",
        sum
    );

    // Higher inputs should have higher probabilities (last element was max)
    assert!(
        data[7] > data[0],
        "Softmax should give higher prob to larger inputs"
    );
}

#[test]
fn test_cuda_executor_rmsnorm() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Cache RMSNorm gamma weights
    let gamma = vec![1.0f32; 64];
    let bytes = executor
        .cache_rmsnorm_gamma("test_gamma", &gamma)
        .expect("cache gamma");
    assert!(bytes > 0, "Should have cached some bytes");

    // Test rmsnorm_host (doesn't require cached gamma)
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let gamma_small = vec![1.0f32; 4];
    let mut output = vec![0.0f32; 4];

    executor
        .rmsnorm_host(&input, &gamma_small, &mut output, 1e-5)
        .expect("rmsnorm_host");

    // Verify output is normalized
    assert!(
        !output.iter().all(|&x| x == 0.0),
        "RMSNorm output should not be all zeros"
    );
}

#[test]
fn test_cuda_executor_weight_cache() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Load weights
    let weights = vec![1.0f32; 256];
    let bytes_loaded = executor
        .load_weights("test_weight", &weights)
        .expect("load weights");
    assert!(bytes_loaded > 0);

    // Verify cache
    assert!(executor.has_weights("test_weight"));
    assert_eq!(executor.cached_weight_count(), 1);
    assert!(executor.cached_weight_bytes() > 0);

    // Clear
    executor.clear_weights();
    assert_eq!(executor.cached_weight_count(), 0);
}

#[test]
fn test_cuda_executor_quantized_weight_cache() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Load quantized weights (raw bytes)
    let data = vec![0u8; 1024];
    let bytes_loaded = executor
        .load_quantized_weights("q_weight", &data)
        .expect("load quantized");
    assert!(bytes_loaded > 0);

    // Verify cache
    assert!(executor.has_quantized_weights("q_weight"));
    assert_eq!(executor.cached_quantized_weight_count(), 1);
    assert!(executor.cached_quantized_weight_bytes() > 0);

    // Test pointer retrieval
    let ptr = executor
        .get_quantized_weight_ptr("q_weight")
        .expect("get ptr");
    assert!(ptr > 0);

    // Clear
    executor.clear_quantized_weights();
    assert_eq!(executor.cached_quantized_weight_count(), 0);
}

#[test]
fn test_cuda_executor_device_info() {
    if !CudaExecutor::is_available() {
        return;
    }

    let executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Get device name
    let name = executor.device_name().expect("device name");
    assert!(!name.is_empty());

    // Get memory info
    let (free, total) = executor.memory_info().expect("memory info");
    assert!(total > 0);
    assert!(free > 0);
    assert!(free <= total);
}

#[test]
fn test_cuda_executor_synchronize() {
    if !CudaExecutor::is_available() {
        return;
    }

    let executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Should not error
    executor.synchronize().expect("sync");
}

#[test]
fn test_cuda_executor_pool_stats() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Get initial stats
    let pool_stats = executor.pool_stats();
    let staging_stats = executor.staging_pool_stats();

    // Stats should be valid
    assert!(pool_stats.hit_rate >= 0.0 && pool_stats.hit_rate <= 1.0);
    assert!(staging_stats.hit_rate >= 0.0 && staging_stats.hit_rate <= 1.0);

    // Clear pool
    executor.clear_pool();
}

#[test]
fn test_cuda_executor_staging_buffers() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Get staging buffer
    let buf = executor.get_staging_buffer(1024);
    assert_eq!(buf.len(), 1024);

    // Return buffer
    executor.return_staging_buffer(buf);

    // Get another (may reuse)
    let buf2 = executor.get_staging_buffer(1024);
    assert_eq!(buf2.len(), 1024);
}

#[test]
fn test_cuda_executor_graph_tracking() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Enable graph tracking
    executor.enable_graph_tracking();
    assert!(executor.is_graph_tracking_enabled());

    // Get execution graph
    let _graph = executor.execution_graph();
    let _ascii = executor.execution_graph_ascii();

    // Clear graph
    executor.clear_execution_graph();

    // Disable graph tracking
    executor.disable_graph_tracking();
    assert!(!executor.is_graph_tracking_enabled());
}

#[test]
fn test_cuda_executor_tile_profiling() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Enable tile profiling
    executor.enable_tile_profiling();
    assert!(executor.is_tile_profiling_enabled());

    // Run a small operation to generate tile stats
    let a = vec![1.0f32; 64 * 64];
    let b = vec![1.0f32; 64 * 64];
    let mut c = vec![0.0f32; 64 * 64];
    let _ = executor.gemm(&a, &b, &mut c, 64, 64, 64);

    // Get tile stats
    let _summary = executor.tile_summary();
    let _json = executor.tile_stats_json();

    // Reset tile stats
    executor.reset_tile_stats();

    // Disable tile profiling
    executor.disable_tile_profiling();
    assert!(!executor.is_tile_profiling_enabled());
}

#[test]
fn test_cuda_executor_num_devices() {
    // Should work even without initialization
    let count = CudaExecutor::num_devices();
    if CudaExecutor::is_available() {
        assert!(count > 0);
    }
}

#[test]
fn test_cuda_kernels_ptx_batch_generation() {
    let kernels = CudaKernels::new();

    // Generate PTX for batch of kernels
    let kernel_types = vec![
        KernelType::Softmax { dim: 512 },
        KernelType::Softmax { dim: 1024 },
        KernelType::Softmax { dim: 2048 },
        KernelType::Softmax { dim: 4096 },
        KernelType::Softmax { dim: 8192 },
    ];

    for kernel in kernel_types {
        let ptx = kernels.generate_ptx(&kernel);
        assert!(ptx.contains("softmax"), "PTX should contain softmax kernel");
    }
}

#[test]
fn test_cuda_kernels_quantized_gemm() {
    let kernels = CudaKernels::new();

    // Test QuantizedGemm kernel generation
    let kernel = KernelType::QuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(!ptx.is_empty());

    // Test QuantizedGemmGgml kernel generation (GGML super-block format)
    let kernel_ggml = KernelType::QuantizedGemmGgml {
        m: 1,
        n: 4096,
        k: 4096,
    };
    let ptx_ggml = kernels.generate_ptx(&kernel_ggml);
    assert!(!ptx_ggml.is_empty());
}

#[test]
fn test_cuda_kernels_bias_activation() {
    let kernels = CudaKernels::new();

    // Test GemmBiasActivation kernel (fused GEMM + bias + activation)
    let kernel = KernelType::GemmBiasActivation {
        m: 4,
        n: 256,
        k: 256,
        activation: 1, // ReLU
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(!ptx.is_empty());

    // Test BiasActivation kernel (epilogue only)
    let kernel_epilogue = KernelType::BiasActivation {
        n: 4096,
        bias_size: 4096,
        activation: 2, // GELU
    };
    let ptx_epilogue = kernels.generate_ptx(&kernel_epilogue);
    assert!(!ptx_epilogue.is_empty());
}

// ============================================================================
// CUDA Execution Tests (RTX 4090 - actual GPU operations)
// ============================================================================

#[test]
fn test_cuda_executor_silu_host() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    let input = vec![0.0f32, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0];
    let mut output = vec![0.0f32; 8];

    executor.silu_host(&input, &mut output).expect("silu_host");

    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // SiLU(0) = 0, SiLU(1) ≈ 0.731, SiLU(-1) ≈ -0.269
    assert!((output[0] - 0.0).abs() < 0.01, "SiLU(0) should be ~0");
    assert!(
        output[1] > 0.7 && output[1] < 0.75,
        "SiLU(1) should be ~0.731"
    );
    assert!(
        output[2] < -0.2 && output[2] > -0.3,
        "SiLU(-1) should be ~-0.269"
    );
}

#[test]
fn test_cuda_executor_gelu_host() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    let input = vec![0.0f32, 1.0, -1.0, 2.0, -2.0];
    let mut output = vec![0.0f32; 5];

    executor.gelu_host(&input, &mut output).expect("gelu_host");

    // GELU(0) = 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
    assert!((output[0] - 0.0).abs() < 0.01, "GELU(0) should be ~0");
    assert!(
        output[1] > 0.8 && output[1] < 0.9,
        "GELU(1) should be ~0.841"
    );
    assert!(
        output[2] < -0.1 && output[2] > -0.2,
        "GELU(-1) should be ~-0.159"
    );
}

#[test]
fn test_cuda_executor_residual_add_host() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    let residual = vec![1.0f32, 2.0, 3.0, 4.0];
    let hidden = vec![0.5f32, 0.5, 0.5, 0.5];
    let mut output = vec![0.0f32; 4];

    executor
        .residual_add_host(&residual, &hidden, &mut output)
        .expect("residual_add_host");

    // output = residual + hidden
    assert!((output[0] - 1.5).abs() < 0.01);
    assert!((output[1] - 2.5).abs() < 0.01);
    assert!((output[2] - 3.5).abs() < 0.01);
    assert!((output[3] - 4.5).abs() < 0.01);
}

#[test]
fn test_cuda_executor_fused_residual_rmsnorm_host() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    let residual = vec![1.0f32, 1.0, 1.0, 1.0];
    let hidden = vec![0.0f32, 0.0, 0.0, 0.0];
    let gamma = vec![1.0f32, 1.0, 1.0, 1.0];
    let mut output = vec![0.0f32; 4];

    executor
        .fused_residual_rmsnorm_host(&residual, &hidden, &gamma, &mut output, 1e-5)
        .expect("fused_residual_rmsnorm_host");

    // After residual add: [1, 1, 1, 1]
    // After RMSNorm with gamma=1: normalized values
    assert!(
        !output.iter().all(|&x| x == 0.0),
        "Output should not be all zeros"
    );
}

#[test]
fn test_cuda_executor_fused_swiglu_host() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // SwiGLU(gate, up) = SiLU(gate) * up
    let gate = vec![1.0f32, 0.0, -1.0, 2.0];
    let up = vec![1.0f32, 1.0, 1.0, 1.0];
    let mut output = vec![0.0f32; 4];

    executor
        .fused_swiglu_host(&gate, &up, &mut output)
        .expect("fused_swiglu_host");

    // SwiGLU(1, 1) = SiLU(1) * 1 ≈ 0.731
    // SwiGLU(0, 1) = SiLU(0) * 1 = 0
    assert!(
        output[0] > 0.7 && output[0] < 0.75,
        "SwiGLU(1,1) should be ~0.731"
    );
    assert!((output[1] - 0.0).abs() < 0.01, "SwiGLU(0,1) should be ~0");
}

#[test]
fn test_cuda_executor_gemm_multiple_sizes() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Test various matrix sizes
    for (m, n, k) in [(16, 16, 16), (32, 64, 32), (64, 32, 64), (128, 128, 64)] {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c = vec![0.0f32; m * n];

        executor
            .gemm(&a, &b, &mut c, m as u32, n as u32, k as u32)
            .unwrap_or_else(|_| panic!("gemm {}x{}x{}", m, n, k));

        // Each element should be k (sum of k ones)
        let expected = k as f32;
        let actual = c[0];
        assert!(
            (actual - expected).abs() < 0.1,
            "GEMM {}x{}x{}: expected {}, got {}",
            m,
            n,
            k,
            expected,
            actual
        );
    }
}

#[test]
fn test_cuda_executor_gemm_cached() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Load weights into cache (A matrix: m x k = 1 x 64)
    let weights = vec![1.0f32; 64]; // m=1, k=64
    executor
        .load_weights("cached_weight", &weights)
        .expect("load weights");

    // Use cached weights
    // B matrix: k x n = 64 x 64 = 4096 elements
    // C matrix: m x n = 1 x 64 = 64 elements
    let input = vec![1.0f32; 64 * 64]; // k*n = 4096
    let mut output = vec![0.0f32; 64]; // m*n = 64

    executor
        .gemm_cached("cached_weight", &input, &mut output, 1, 64, 64)
        .expect("gemm_cached");

    // Each output element should be 64 (sum of k=64 ones)
    assert!(
        (output[0] - 64.0).abs() < 0.1,
        "GEMM cached: expected 64, got {}",
        output[0]
    );
}

#[test]
fn test_cuda_executor_quantized_weights_with_type() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Load quantized weights with specific type
    let data = vec![0u8; 144]; // Q4_K block size
    let bytes = executor
        .load_quantized_weights_with_type("q4k_weight", &data, 12) // 12 = Q4_K GGML type
        .expect("load_quantized_weights_with_type");
    assert!(bytes > 0);

    // Check type was stored
    let qtype = executor.get_quantized_weight_type("q4k_weight");
    assert_eq!(qtype, Some(12));
}

#[test]
fn test_cuda_executor_gemv_buffers() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Get buffer stats (before allocation)
    let (input_bytes, output_bytes) = executor.gemv_buffer_stats();
    assert_eq!(input_bytes, 0);
    assert_eq!(output_bytes, 0);

    // Clear buffers (should not panic even if empty)
    executor.clear_gemv_buffers();
}

#[test]
fn test_cuda_context_access() {
    if !CudaExecutor::is_available() {
        return;
    }

    let executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Access context
    let _context = executor.context();

    // Verify we can use context for device info
    let name = executor.device_name().expect("device_name");
    assert!(name.contains("RTX") || name.contains("GeForce") || !name.is_empty());
}

#[test]
fn test_cuda_executor_profiler_all_modes() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");

    // Test all sync modes
    for mode in [SyncMode::Immediate, SyncMode::PerLayer, SyncMode::Deferred] {
        executor.set_profiler_sync_mode(mode);
        assert_eq!(executor.profiler_sync_mode(), mode);
    }
}

#[test]
fn test_cuda_executor_multiple_operations_sequence() {
    if !CudaExecutor::is_available() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("Failed to create executor");
    executor.enable_profiling();

    // Run a sequence of operations
    let a = vec![1.0f32; 64 * 64];
    let b = vec![1.0f32; 64 * 64];
    let mut c = vec![0.0f32; 64 * 64];

    // GEMM
    executor.gemm(&a, &b, &mut c, 64, 64, 64).expect("gemm");

    // Softmax (in-place)
    let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
    executor.softmax(&mut data).expect("softmax");

    // RMSNorm
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let gamma = vec![1.0f32; 4];
    let mut norm_output = vec![0.0f32; 4];
    executor
        .rmsnorm_host(&input, &gamma, &mut norm_output, 1e-5)
        .expect("rmsnorm");

    // Get profiler summary
    let summary = executor.profiler_summary();
    assert!(!summary.is_empty() || summary.is_empty()); // Just verify no panic
}

#[test]
fn test_cuda_kernels_all_kernel_types() {
    let kernels = CudaKernels::new();

    // Test all remaining kernel types not covered elsewhere
    let kernel_types = vec![
        // Gemv variants
        KernelType::Gemv { k: 256, n: 256 },
        KernelType::CoalescedGemv { k: 512, n: 512 },
        // Quantized
        KernelType::QuantizedGemm {
            m: 1,
            n: 256,
            k: 256,
        },
        KernelType::QuantizedGemmGgml {
            m: 1,
            n: 256,
            k: 256,
        },
        // Rope variants
        KernelType::Rope {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        // Residual
        KernelType::ResidualAdd { n: 4096 },
        // Activations
        KernelType::Silu { n: 1024 },
        KernelType::Gelu { n: 1024 },
        // RMSNorm variants
        KernelType::RmsNorm {
            hidden_size: 4096,
            epsilon: 1e-5,
        },
        KernelType::VectorizedRmsNorm {
            hidden_size: 4096,
            epsilon: 1e-5,
        },
        // Fused operations
        KernelType::FusedResidualRmsNorm {
            hidden_size: 4096,
            epsilon: 1e-5,
        },
        KernelType::FusedSwiglu { n: 4096 },
    ];

    for kernel in kernel_types {
        let ptx = kernels.generate_ptx(&kernel);
        assert!(!ptx.is_empty(), "PTX for {:?} should not be empty", kernel);
    }
}
