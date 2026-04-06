
/// Test fused_gate_up_into basic functionality
#[test]
#[serial]
fn test_cov028_fused_gate_up_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 64u32;
    let intermediate_size = 128u32;

    let x_data = vec![0.1f32; hidden_size as usize];
    let x = GpuBuffer::from_host(executor.context(), &x_data).expect("x");

    // Weight matrices: hidden_size x intermediate_size
    let w_gate_data = vec![0.01f32; (hidden_size * intermediate_size) as usize];
    let w_up_data = vec![0.01f32; (hidden_size * intermediate_size) as usize];

    let w_gate = GpuBuffer::from_host(executor.context(), &w_gate_data).expect("w_gate");
    let w_up = GpuBuffer::from_host(executor.context(), &w_up_data).expect("w_up");

    let output = GpuBuffer::new(executor.context(), intermediate_size as usize).expect("output");

    let result =
        executor.fused_gate_up_into(&x, &w_gate, &w_up, &output, hidden_size, intermediate_size);
    assert!(
        result.is_ok(),
        "fused_gate_up_into should succeed: {:?}",
        result.err()
    );
}

/// Test rope_into basic functionality
#[test]
#[serial]
fn test_cov028_rope_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4u32;
    let head_dim = 32u32;
    let position = 5u32;
    let theta = 10000.0f32;

    let input_size = (num_heads * head_dim) as usize;
    let input_data = vec![0.5f32; input_size];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), input_size).expect("output");

    let result = executor.rope_into(&input, &output, position, num_heads, head_dim, theta);
    assert!(
        result.is_ok(),
        "rope_into should succeed: {:?}",
        result.err()
    );
}

/// Test rope_into with different positions
#[test]
#[serial]
fn test_cov028_rope_into_varying_positions() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 8u32;
    let head_dim = 64u32;
    let theta = 10000.0f32;

    let input_size = (num_heads * head_dim) as usize;
    let input_data = vec![1.0f32; input_size];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), input_size).expect("output");

    // Test multiple positions
    for position in [0, 1, 10, 100, 1000] {
        let result = executor.rope_into(&input, &output, position, num_heads, head_dim, theta);
        assert!(
            result.is_ok(),
            "rope_into at position {} should succeed: {:?}",
            position,
            result.err()
        );
    }
}

/// Test batched_q4k_gemv_into basic functionality
#[test]
#[serial]
fn test_cov028_batched_q4k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32; // batch size
    let n = 32u32;
    let k = 256u32;

    // Load Q4K weights
    let weight_bytes = (n as usize) * 144; // Q4K is 144 bytes per 256 values
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_batched_q4k", &weights)
        .expect("load weights");
    let weight_ptr = executor
        .get_quantized_weight_ptr("test_batched_q4k")
        .expect("get ptr");

    // Input: m x k elements
    let input_data = vec![0.1f32; (m * k) as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // Output: m x n elements
    let output = GpuBuffer::new(executor.context(), (m * n) as usize).expect("output");

    let result = executor.batched_q4k_gemv_into(weight_ptr, &input, &output, m, n, k);
    assert!(
        result.is_ok(),
        "batched_q4k_gemv_into should succeed: {:?}",
        result.err()
    );
}

/// Test batched_q4k_gemv_into with M=16 (multi-warp path)
#[test]
#[serial]
fn test_cov028_batched_q4k_gemv_into_m16() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 16u32; // triggers multi-warp kernel
    let n = 32u32;
    let k = 256u32;

    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_batched_q4k_m16", &weights)
        .expect("load");
    let weight_ptr = executor
        .get_quantized_weight_ptr("test_batched_q4k_m16")
        .expect("get ptr");

    let input_data = vec![0.1f32; (m * k) as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), (m * n) as usize).expect("output");

    let result = executor.batched_q4k_gemv_into(weight_ptr, &input, &output, m, n, k);
    assert!(
        result.is_ok(),
        "batched_q4k_gemv_into M=16 should succeed: {:?}",
        result.err()
    );
}

/// Test batched_q6k_gemv_into basic functionality
#[test]
#[serial]
fn test_cov028_batched_q6k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let n = 32u32;
    let k = 256u32;

    // Load Q6K weights (210 bytes per 256 values)
    let weight_bytes = (n as usize) * 210;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_batched_q6k", &weights, 14)
        .expect("load");
    let weight_ptr = executor
        .get_quantized_weight_ptr("test_batched_q6k")
        .expect("get ptr");

    let input_data = vec![0.1f32; (m * k) as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), (m * n) as usize).expect("output");

    let result = executor.batched_q6k_gemv_into(weight_ptr, &input, &output, m, n, k);
    assert!(
        result.is_ok(),
        "batched_q6k_gemv_into should succeed: {:?}",
        result.err()
    );
}

/// Test layer_norm_gpu basic functionality
#[test]
#[serial]
fn test_cov028_layer_norm_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 64u32;
    let batch_size = 1u32;
    let epsilon = 1e-5f32;

    let input_data = vec![0.5f32; hidden_size as usize];
    let gamma_data = vec![1.0f32; hidden_size as usize];
    let beta_data = vec![0.0f32; hidden_size as usize];

    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), hidden_size as usize).expect("output");
    let gamma = GpuBuffer::from_host(executor.context(), &gamma_data).expect("gamma");
    let beta = GpuBuffer::from_host(executor.context(), &beta_data).expect("beta");

    let result = executor.layer_norm_gpu(
        &input,
        &output,
        &gamma,
        &beta,
        hidden_size,
        batch_size,
        epsilon,
    );
    assert!(
        result.is_ok(),
        "layer_norm_gpu should succeed: {:?}",
        result.err()
    );
}

/// Test layer_norm_gpu with batch
#[test]
#[serial]
fn test_cov028_layer_norm_gpu_batched() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 128u32;
    let batch_size = 4u32;
    let epsilon = 1e-6f32;

    let input_data = vec![0.5f32; (hidden_size * batch_size) as usize];
    let gamma_data = vec![1.0f32; hidden_size as usize];
    let beta_data = vec![0.1f32; hidden_size as usize];

    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output =
        GpuBuffer::new(executor.context(), (hidden_size * batch_size) as usize).expect("output");
    let gamma = GpuBuffer::from_host(executor.context(), &gamma_data).expect("gamma");
    let beta = GpuBuffer::from_host(executor.context(), &beta_data).expect("beta");

    let result = executor.layer_norm_gpu(
        &input,
        &output,
        &gamma,
        &beta,
        hidden_size,
        batch_size,
        epsilon,
    );
    assert!(
        result.is_ok(),
        "layer_norm_gpu batched should succeed: {:?}",
        result.err()
    );
}

/// Test compute_stream getter
#[test]
#[serial]
fn test_cov028_compute_stream() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // compute_stream() returns a reference to CudaStream
    let stream = executor.compute_stream();
    // Just verify we can access it without panic
    assert!(
        std::ptr::from_ref(stream) as usize != 0,
        "stream should be valid"
    );
}

// ============================================================================
// COV-029: More weight and workspace tests
// ============================================================================

/// Test load_weights basic functionality
#[test]
#[serial]
fn test_cov029_load_weights_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let weights = vec![0.1f32; 256];
    let result = executor.load_weights("test_weight", &weights);
    assert!(
        result.is_ok(),
        "load_weights should succeed: {:?}",
        result.err()
    );

    let bytes = result.unwrap();
    assert_eq!(bytes, 256 * 4, "Should load 256 f32 values (1024 bytes)");
}

/// Test load_weights and has_weights
#[test]
#[serial]
fn test_cov029_load_weights_and_has() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        !executor.has_weights("my_weight"),
        "Should not have weight initially"
    );

    let weights = vec![1.0f32; 128];
    executor.load_weights("my_weight", &weights).expect("load");

    assert!(
        executor.has_weights("my_weight"),
        "Should have weight after load"
    );
    assert!(
        !executor.has_weights("other_weight"),
        "Should not have unloaded weight"
    );
}

/// Test cached_weight_count
#[test]
#[serial]
fn test_cov029_cached_weight_count() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    assert_eq!(
        executor.cached_weight_count(),
        0,
        "Initial count should be 0"
    );

    executor.load_weights("w1", &[1.0f32; 64]).expect("load w1");
    assert_eq!(executor.cached_weight_count(), 1, "Count should be 1");

    executor.load_weights("w2", &[1.0f32; 64]).expect("load w2");
    assert_eq!(executor.cached_weight_count(), 2, "Count should be 2");
}

/// Test cached_weight_bytes
#[test]
#[serial]
fn test_cov029_cached_weight_bytes() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    assert_eq!(
        executor.cached_weight_bytes(),
        0,
        "Initial bytes should be 0"
    );

    executor
        .load_weights("w1", &[1.0f32; 100])
        .expect("load w1");
    assert_eq!(
        executor.cached_weight_bytes(),
        400,
        "Should be 400 bytes (100 * 4)"
    );

    executor.load_weights("w2", &[1.0f32; 50]).expect("load w2");
    assert_eq!(
        executor.cached_weight_bytes(),
        600,
        "Should be 600 bytes total"
    );
}

/// Test has_indexed_weights
#[test]
#[serial]
fn test_cov029_has_indexed_weights() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially should not have indexed weights
    assert!(
        !executor.has_indexed_weights(),
        "Should not have indexed weights initially"
    );
}

/// Test return_staging_buffer
#[test]
#[serial]
fn test_cov029_return_staging_buffer() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get a staging buffer
    let buf = executor.get_staging_buffer(256);

    // Return it to the pool
    executor.return_staging_buffer(buf);

    // Pool should have returned buffer
    let stats = executor.staging_pool_stats();
    assert!(
        stats.free_buffers >= 1,
        "Pool should have at least 1 buffer after return"
    );
}
