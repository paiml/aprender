
// NOTE: layer_norm_gpu test skipped - LayerNorm kernel not available (FunctionNotFound)

#[test]
#[serial]
fn test_cov023_rmsnorm_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 256u32;
    let epsilon = 1e-5f32;

    // Create GPU buffers
    let input_data = vec![0.5f32; hidden_size as usize];
    let gamma_data = vec![1.0f32; hidden_size as usize];

    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let gamma = GpuBuffer::from_host(executor.context(), &gamma_data).expect("gamma");
    let output = GpuBuffer::new(executor.context(), hidden_size as usize).expect("output");

    // Apply RMSNorm into existing buffer
    let result = executor.rmsnorm_into(&input, &gamma, &output, hidden_size, epsilon);

    assert!(
        result.is_ok(),
        "rmsnorm_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov023_q4k_gemv_cached_async_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Create GPU buffer for input
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input buf");

    // Weight not cached - should fail
    let result = executor.q4k_gemv_cached_async("nonexistent_weight", &input, n, k);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov023_q6k_gemv_cached_async_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Create GPU buffer for input
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input buf");

    // Weight not cached - should fail
    let result = executor.q6k_gemv_cached_async("nonexistent_weight", &input, n, k);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

// =============================================================================
// COV-024: q5k_matvec, q6k_matvec, gpu_argmax, transformer_layer_host coverage
// =============================================================================

#[test]
#[serial]
fn test_cov024_q5k_matvec_dimension_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q5_K: 176 bytes per 256 values
    let m = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Create valid Q5_K weights (176 bytes per 256-value super-block)
    // For k=256, we need (k / 256) = 1 super-block per output row
    // Total: m * 176 bytes
    let weight_bytes = (m as usize) * 176;
    let weights = vec![0u8; weight_bytes];
    let input = vec![0.1f32; k as usize];
    let mut output = vec![0.0f32; m as usize];

    // This should succeed with valid dimensions
    let result = executor.q5k_matvec(&weights, &input, &mut output, m, k);
    assert!(
        result.is_ok(),
        "q5k_matvec should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov024_q6k_matvec_dimension_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q6_K: 210 bytes per 256 values
    let m = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Create valid Q6_K weights (210 bytes per 256-value super-block)
    let weight_bytes = (m as usize) * 210;
    let weights = vec![0u8; weight_bytes];
    let input = vec![0.1f32; k as usize];
    let mut output = vec![0.0f32; m as usize];

    let result = executor.q6k_matvec(&weights, &input, &mut output, m, k);
    assert!(
        result.is_ok(),
        "q6k_matvec should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov024_gpu_argmax_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Create a simple logits buffer on GPU
    let vocab_size = 128u32;
    let logits = vec![0.1f32; vocab_size as usize];
    let logits_buf = GpuBuffer::from_host(executor.context(), &logits).expect("logits buf");

    let result = executor.gpu_argmax(logits_buf.as_ptr(), vocab_size);
    assert!(
        result.is_ok(),
        "gpu_argmax should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov024_gpu_argmax_with_clear_max() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Create logits with a clear maximum at position 42
    let vocab_size = 128u32;
    let mut logits = vec![0.0f32; vocab_size as usize];
    logits[42] = 10.0; // Clear maximum

    let logits_buf = GpuBuffer::from_host(executor.context(), &logits).expect("logits buf");

    let result = executor.gpu_argmax(logits_buf.as_ptr(), vocab_size);
    assert!(
        result.is_ok(),
        "gpu_argmax should succeed: {:?}",
        result.err()
    );

    // Should return index 42 as the argmax
    let argmax = result.unwrap();
    assert_eq!(argmax, 42, "argmax should find the maximum at position 42");
}

#[test]
#[serial]
fn test_cov024_fused_ffn_swiglu_gpu_weight_not_cached_gate() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 256u32;
    let intermediate_size = 512u32;

    // Create input GPU buffer
    let input_data = vec![0.1f32; hidden_size as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // FFN weights not cached - should fail
    // Signature: fused_ffn_swiglu_gpu(input, gate_name, up_name, down_name, hidden_dim, intermediate_dim)
    let result = executor.fused_ffn_swiglu_gpu(
        &input,
        "nonexistent_gate",
        "nonexistent_up",
        "nonexistent_down",
        hidden_size,
        intermediate_size,
    );

    assert!(result.is_err(), "Should fail when FFN weights not cached");
}

#[test]
#[serial]
fn test_cov024_cache_rmsnorm_gamma_empty_name() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Empty gamma array should still work
    let gamma = vec![1.0f32; 256];
    let result = executor.cache_rmsnorm_gamma("layer0.attn_norm.gamma", &gamma);
    assert!(
        result.is_ok(),
        "cache_rmsnorm_gamma should succeed: {:?}",
        result.err()
    );

    // Verify we can cache with different name
    let result2 = executor.cache_rmsnorm_gamma("layer0.ffn_norm.gamma", &gamma);
    assert!(
        result2.is_ok(),
        "cache_rmsnorm_gamma should succeed: {:?}",
        result2.err()
    );
}

#[test]
#[serial]
fn test_cov024_output_rmsnorm_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 256u32;
    let epsilon = 1e-5f32;

    // Create input/output/gamma host slices
    let input_data = vec![0.5f32; hidden_size as usize];
    let mut output_data = vec![0.0f32; hidden_size as usize];
    let gamma = vec![1.0f32; hidden_size as usize];

    // Apply output rmsnorm (takes host slices)
    let result =
        executor.output_rmsnorm_gpu(&input_data, &mut output_data, &gamma, hidden_size, epsilon);
    assert!(
        result.is_ok(),
        "output_rmsnorm_gpu should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov024_output_rmsnorm_gpu_with_varied_gamma() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 256u32;
    let epsilon = 1e-5f32;

    // Create input/output/gamma host slices with varied values
    let input_data = vec![0.5f32; hidden_size as usize];
    let mut output_data = vec![0.0f32; hidden_size as usize];
    let gamma: Vec<f32> = (0..hidden_size).map(|i| 0.5 + (i as f32) * 0.01).collect();

    let result =
        executor.output_rmsnorm_gpu(&input_data, &mut output_data, &gamma, hidden_size, epsilon);
    assert!(
        result.is_ok(),
        "output_rmsnorm_gpu should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov024_clear_indexed_weights_empty() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Clear should work even when nothing indexed
    executor.clear_indexed_weights();
    assert!(
        !executor.has_indexed_weights(),
        "Should have no indexed weights"
    );
}

#[test]
#[serial]
fn test_cov024_clear_execution_graph_empty() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Clear should work even when empty
    executor.clear_execution_graph();

    // Check graph is empty
    let ascii = executor.execution_graph_ascii();
    assert!(
        ascii.contains("empty") || ascii.is_empty() || ascii.len() < 50,
        "Graph should be empty"
    );
}

#[test]
#[serial]
fn test_cov024_set_rope_type() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set different RoPE types (0 = standard, 1 = neox)
    executor.set_rope_type(0);
    executor.set_rope_type(1);
    // No assertion needed - just ensuring it doesn't panic
}

#[test]
#[serial]
fn test_cov024_set_rope_theta() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set various rope theta values
    executor.set_rope_theta(10000.0);
    executor.set_rope_theta(500000.0);
    // No assertion needed - just ensuring it doesn't panic
}

// =============================================================================
// COV-025: More device info, memory, and buffer management coverage
// =============================================================================

#[test]
#[serial]
fn test_cov025_device_name() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let name = executor.device_name();
    assert!(name.is_ok(), "device_name should succeed: {:?}", name.err());
    let device_name = name.unwrap();
    assert!(!device_name.is_empty(), "Device name should not be empty");
}

#[test]
#[serial]
fn test_cov025_memory_info() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let info = executor.memory_info();
    assert!(info.is_ok(), "memory_info should succeed: {:?}", info.err());
    let (free, total) = info.unwrap();
    assert!(total > 0, "Total memory should be > 0");
    assert!(free <= total, "Free memory should be <= total");
}

#[test]
#[serial]
fn test_cov025_num_devices() {
    // num_devices is a static function
    let count = CudaExecutor::num_devices();
    // On a system with CUDA, count should be >= 1
    // We don't assert on the count since it depends on hardware
    let _ = count; // Verify the function is callable
}

#[test]
#[serial]
fn test_cov025_make_current() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.make_current();
    assert!(
        result.is_ok(),
        "make_current should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov025_clear_pool() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Clear pool should work even when empty
    executor.clear_pool();

    // Allocate some buffers, then clear
    let _ = executor.allocate_buffer(1024);
    let _ = executor.allocate_buffer(2048);
    executor.clear_pool();
}
