
#[test]
#[serial]
fn test_cov001_q6k_gemv_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let k = 256u32;
    let weights = mock_q6k_weights(n as usize, k as usize);

    executor
        .load_quantized_weights("test_q6k", &weights)
        .expect("load weights");

    let input = vec![1.0f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q6k_gemv_cached("test_q6k", &input, &mut output, n, k);
    assert!(
        result.is_ok(),
        "q6k_gemv_cached should succeed: {:?}",
        result
    );
}

// ========================================================================
// COV-002: High-level CUDA function tests (slice-based API)
// ========================================================================

#[test]
#[serial]
fn test_cov002_softmax_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
    let result = executor.softmax(&mut data);
    assert!(result.is_ok(), "softmax should succeed: {:?}", result);

    // Verify softmax properties: sum to 1, all positive
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum should be 1.0");
    assert!(
        data.iter().all(|&x| x > 0.0),
        "all values should be positive"
    );
}

#[test]
#[serial]
fn test_cov002_gemm_optimized() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32;
    let n = 32u32;
    let k = 32u32;
    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let tile_size = 32u32;
    let result = executor.gemm_optimized(&a, &b, &mut c, m, n, k, tile_size);
    assert!(
        result.is_ok(),
        "gemm_optimized should succeed: {:?}",
        result
    );

    // Each element should be k (dot product of k ones)
    for val in &c {
        assert!(
            (*val - k as f32).abs() < 1e-3,
            "expected {}, got {}",
            k,
            val
        );
    }
}

#[test]
#[serial]
fn test_cov002_gemm_fused_variants() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 16u32;
    let n = 16u32;
    let k = 16u32;
    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let bias = vec![1.0f32; n as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    // Test with bias and no activation (0)
    let result = executor.gemm_fused(&a, &b, Some(&bias), &mut c, m, n, k, 0);
    assert!(
        result.is_ok(),
        "gemm_fused with no activation should succeed: {:?}",
        result
    );

    // Test with bias and ReLU activation (1)
    c.fill(0.0);
    let result = executor.gemm_fused(&a, &b, Some(&bias), &mut c, m, n, k, 1);
    assert!(
        result.is_ok(),
        "gemm_fused with ReLU should succeed: {:?}",
        result
    );

    // Test with bias and GELU activation (2)
    c.fill(0.0);
    let result = executor.gemm_fused(&a, &b, Some(&bias), &mut c, m, n, k, 2);
    assert!(
        result.is_ok(),
        "gemm_fused with GELU should succeed: {:?}",
        result
    );
}

#[test]
#[serial]
fn test_cov002_flash_attention_multi_head() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let seq_len = 8u32;
    let head_dim = 8u32;
    let n_heads = 4u32;
    let size = (seq_len * head_dim * n_heads) as usize;

    let q = vec![1.0f32; size];
    let k = vec![1.0f32; size];
    let v = vec![1.0f32; size];
    let mut output = vec![0.0f32; size];

    let result = executor.flash_attention_multi_head(
        &q,
        &k,
        &v,
        &mut output,
        seq_len,
        head_dim,
        n_heads,
        true,
    );
    assert!(
        result.is_ok(),
        "flash_attention_multi_head should succeed: {:?}",
        result
    );
}

#[test]
#[serial]
fn test_cov002_silu_gelu_host() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let size = 256usize;
    let input = vec![1.0f32; size];
    let mut silu_out = vec![0.0f32; size];
    let mut gelu_out = vec![0.0f32; size];

    let result = executor.silu_host(&input, &mut silu_out);
    assert!(result.is_ok(), "silu_host should succeed: {:?}", result);

    let result = executor.gelu_host(&input, &mut gelu_out);
    assert!(result.is_ok(), "gelu_host should succeed: {:?}", result);

    // SiLU and GELU should produce different results
    assert!(silu_out[0] != gelu_out[0], "SiLU and GELU should differ");
}

#[test]
#[serial]
fn test_cov002_elementwise_mul_host() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let size = 256usize;
    let a = vec![2.0f32; size];
    let b = vec![3.0f32; size];
    let mut output = vec![0.0f32; size];

    let result = executor.elementwise_mul_host(&a, &b, &mut output);
    assert!(
        result.is_ok(),
        "elementwise_mul_host should succeed: {:?}",
        result
    );
    assert!((output[0] - 6.0).abs() < 1e-5, "2 * 3 should be 6");
}

#[test]
#[serial]
fn test_cov002_load_and_clear_weights() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let weights = vec![1.0f32; 1024];

    // Load weights
    let result = executor.load_weights("test_weights", &weights);
    assert!(result.is_ok(), "load_weights should succeed");

    // Check cache stats
    assert!(executor.has_weights("test_weights"));
    assert_eq!(executor.cached_weight_count(), 1);
    assert!(executor.cached_weight_bytes() > 0);

    // Clear weights
    executor.clear_weights();
    assert!(!executor.has_weights("test_weights"));
    assert_eq!(executor.cached_weight_count(), 0);
}

#[test]
#[serial]
fn test_cov002_load_quantized_weights() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Mock Q4_K weights: 144 bytes per 256 values
    let weights = vec![0x42u8; 144];

    // Load quantized weights
    let result = executor.load_quantized_weights("q4k_test", &weights);
    assert!(result.is_ok(), "load_quantized_weights should succeed");

    // Check cache stats
    assert!(executor.cached_quantized_weight_count() > 0);
    assert!(executor.cached_quantized_weight_bytes() > 0);

    // Clear
    executor.clear_quantized_weights();
    assert_eq!(executor.cached_quantized_weight_count(), 0);
}

#[test]
#[serial]
fn test_cov002_profiler_operations() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Enable profiling
    executor.enable_profiling();
    assert!(executor.is_profiling_enabled());

    // Get profiler and reset
    let _profiler = executor.profiler();
    let _profiler_mut = executor.profiler_mut();
    executor.reset_profiler();

    // Get profiler summary
    let _summary = executor.profiler_summary();

    // Disable profiling
    executor.disable_profiling();
    assert!(!executor.is_profiling_enabled());
}

#[test]
#[serial]
fn test_cov002_graph_tracking() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Enable graph tracking
    executor.enable_graph_tracking();
    assert!(executor.is_graph_tracking_enabled());

    // Get execution graph
    let _graph = executor.execution_graph();
    let _ascii = executor.execution_graph_ascii();

    // Clear and disable
    executor.clear_execution_graph();
    executor.disable_graph_tracking();
    assert!(!executor.is_graph_tracking_enabled());
}

#[test]
#[serial]
fn test_cov002_tile_profiling() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Enable tile profiling
    executor.enable_tile_profiling();
    assert!(executor.is_tile_profiling_enabled());

    // Get tile stats
    let _summary = executor.tile_summary();
    let _json = executor.tile_stats_json();

    // Reset and disable
    executor.reset_tile_stats();
    executor.disable_tile_profiling();
    assert!(!executor.is_tile_profiling_enabled());
}

#[test]
#[serial]
fn test_cov002_memory_and_device_info() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get device name
    let name = executor.device_name().expect("device_name should succeed");
    assert!(name.contains("NVIDIA") || name.contains("RTX") || name.contains("GeForce"));

    // Get memory info
    let mem_info = executor.memory_info();
    assert!(mem_info.is_ok(), "memory_info should succeed");
    let (free, total) = mem_info.expect("CUDA operation failed");
    assert!(total > 0, "total memory should be > 0");
    assert!(free <= total, "free should be <= total");

    // Get context
    let _ctx = executor.context();
}

#[test]
#[serial]
fn test_cov002_staging_buffer_operations() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get staging buffer
    let buf = executor.get_staging_buffer(1024);
    assert!(buf.len() >= 1024);

    // Return staging buffer
    executor.return_staging_buffer(buf);

    // Get pool stats
    let _stats = executor.staging_pool_stats();

    // Clear pool
    executor.clear_pool();
}

#[test]
#[serial]
fn test_cov002_synchronize() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize();
    assert!(result.is_ok(), "synchronize should succeed");
}

#[test]
fn test_cov002_cuda_likely_available() {
    // This should return true on a system with CUDA (checks /dev/nvidia0 or CUDA_VISIBLE_DEVICES)
    let likely = CudaKernels::cuda_likely_available();
    // On a system with RTX 4090, this should be true
    assert!(
        likely,
        "cuda_likely_available should be true on a system with NVIDIA GPU"
    );
}

#[test]
fn test_cov002_is_available_and_num_devices() {
    let available = CudaExecutor::is_available();
    let num_devices = CudaExecutor::num_devices();

    if available {
        assert!(
            num_devices > 0,
            "If CUDA available, num_devices should be > 0"
        );
    }
}

#[test]
fn test_cov001_transfer_mode_properties() {
    let modes = [
        TransferMode::Pageable,
        TransferMode::Pinned,
        TransferMode::Async,
        TransferMode::ZeroCopy,
    ];

    for mode in modes {
        let speedup = mode.estimated_speedup();
        assert!(speedup >= 1.0, "Speedup should be >= 1.0");

        let requires_pinned = mode.requires_pinned();
        match mode {
            TransferMode::Pageable => assert!(!requires_pinned),
            _ => assert!(requires_pinned),
        }
    }
}
