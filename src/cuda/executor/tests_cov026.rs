
#[test]
#[serial]
fn test_cov026_q5_0_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q5_0: 22 bytes per 32 values (2 scale + 4 high bits + 16 low bits)
    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 32)

    // Load quantized weights
    let weight_bytes = (n as usize) * (k as usize / 32) * 22;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_q5_0_into", &weights, 6)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q5_0_into")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q5_0_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q5_0_gemv_into should succeed: {:?}",
        result.err()
    );
}

// =============================================================================
// COV-027: Tiled/Fused GEMV and async quantization coverage
// =============================================================================

#[test]
#[serial]
fn test_cov027_q4k_gemv_into_tiled_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Load quantized weights
    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_tiled", &weights)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_tiled")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q4k_gemv_into_tiled(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q4k_gemv_into_tiled should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov027_tiled_q4k_gemv_cached_async_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;
    let outputs_per_block = 4u32;

    // Create input buffer
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // Weight not cached - should fail
    let result =
        executor.tiled_q4k_gemv_cached_async("nonexistent_tiled", &input, n, k, outputs_per_block);
    assert!(result.is_err(), "Should fail when weight not cached");
}

#[test]
#[serial]
fn test_cov027_chunked_tiled_q4k_gemv_cached_async_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 512u32; // larger K for chunked version
    let outputs_per_block = 4u32;

    // Create input buffer
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // Weight not cached - should fail
    let result = executor.chunked_tiled_q4k_gemv_cached_async(
        "nonexistent_chunked",
        &input,
        n,
        k,
        outputs_per_block,
    );
    assert!(result.is_err(), "Should fail when weight not cached");
}

#[test]
#[serial]
fn test_cov027_dp4a_q4k_gemv_cached_async_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Create input buffer
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // Weight not cached - should fail
    let result = executor.dp4a_q4k_gemv_cached_async("nonexistent_dp4a", &input, n, k);
    assert!(result.is_err(), "Should fail when weight not cached");
}

#[test]
#[serial]
fn test_cov027_fused_rmsnorm_q4k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32; // output dim
    let k = 256u32; // input dim (also hidden size for rmsnorm)
    let epsilon = 1e-5f32;

    // Load quantized weights
    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_fused_rmsnorm", &weights)
        .expect("load weights");
    let weight_ptr = executor
        .get_quantized_weight_ptr("test_fused_rmsnorm")
        .expect("get ptr");

    // Cache gamma
    let gamma = vec![1.0f32; k as usize];
    executor
        .cache_rmsnorm_gamma("test_fused_gamma", &gamma)
        .expect("cache gamma");

    // Get gamma pointer (need to use internal cache)
    // For this test, we'll create gamma as a GPU buffer
    let gamma_buf = GpuBuffer::from_host(executor.context(), &gamma).expect("gamma buf");
    let gamma_ptr = gamma_buf.as_ptr();

    // Create input/output buffers
    let input_data = vec![0.5f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result =
        executor.fused_rmsnorm_q4k_gemv_into(weight_ptr, &input, gamma_ptr, &output, k, n, epsilon);
    assert!(
        result.is_ok(),
        "fused_rmsnorm_q4k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov027_fused_gate_up_q4k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let k = 256u32; // input dim (hidden_size)
    let n = 512u32; // output dim (intermediate_size)

    // Load gate and up weights
    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_gate_fused", &weights)
        .expect("load gate");
    executor
        .load_quantized_weights("test_up_fused", &weights)
        .expect("load up");

    let gate_ptr = executor
        .get_quantized_weight_ptr("test_gate_fused")
        .expect("get gate ptr");
    let up_ptr = executor
        .get_quantized_weight_ptr("test_up_fused")
        .expect("get up ptr");

    // Create input and separate output buffers for gate and up
    let input_data = vec![0.5f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let gate_output = GpuBuffer::new(executor.context(), n as usize).expect("gate output");
    let up_output = GpuBuffer::new(executor.context(), n as usize).expect("up output");

    // fused_gate_up_q4k_gemv_into(gate_ptr, up_ptr, input, gate_output, up_output, k, n)
    let result = executor.fused_gate_up_q4k_gemv_into(
        gate_ptr,
        up_ptr,
        &input,
        &gate_output,
        &up_output,
        k,
        n,
    );
    assert!(
        result.is_ok(),
        "fused_gate_up_q4k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov027_q4k_gemv_cached_tiled_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Create host slices (q4k_gemv_cached_tiled takes host slices, not GPU buffers)
    let input_data = vec![0.1f32; k as usize];
    let mut output_data = vec![0.0f32; n as usize];

    // Weight not cached - should fail
    let result = executor.q4k_gemv_cached_tiled(
        "nonexistent_cached_tiled",
        &input_data,
        &mut output_data,
        n,
        k,
    );
    assert!(result.is_err(), "Should fail when weight not cached");
}

#[test]
#[serial]
fn test_cov027_q4k_gemv_indexed_async_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Load weights and get pointer
    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_indexed", &weights)
        .expect("load");
    let weight_ptr = executor
        .get_quantized_weight_ptr("test_indexed")
        .expect("get ptr");

    // Create input buffer
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // q4k_gemv_indexed_async takes weight_ptr, not layer_idx
    let result = executor.q4k_gemv_indexed_async(weight_ptr, &input, n, k);
    assert!(
        result.is_ok(),
        "q4k_gemv_indexed_async should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov027_q6k_gemv_indexed_async_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Load Q6_K weights and get pointer
    let weight_bytes = (n as usize) * 210; // Q6_K is 210 bytes per 256 values
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_q6k_indexed", &weights, 14)
        .expect("load");
    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q6k_indexed")
        .expect("get ptr");

    // Create input buffer
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // q6k_gemv_indexed_async takes weight_ptr, not layer_idx
    let result = executor.q6k_gemv_indexed_async(weight_ptr, &input, n, k);
    assert!(
        result.is_ok(),
        "q6k_gemv_indexed_async should succeed: {:?}",
        result.err()
    );
}

// ============================================================================
// COV-028: More function coverage tests
// ============================================================================

/// Test fused_qkv_into basic functionality
#[test]
#[serial]
fn test_cov028_fused_qkv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 64u32;
    let kv_dim = 64u32;

    // Create input and weight buffers
    let x_data = vec![0.1f32; hidden_size as usize];
    let x = GpuBuffer::from_host(executor.context(), &x_data).expect("x");

    // Weight matrices: Q is hidden_size x hidden_size, K/V are hidden_size x kv_dim
    let w_q_data = vec![0.01f32; (hidden_size * hidden_size) as usize];
    let w_k_data = vec![0.01f32; (hidden_size * kv_dim) as usize];
    let w_v_data = vec![0.01f32; (hidden_size * kv_dim) as usize];

    let w_q = GpuBuffer::from_host(executor.context(), &w_q_data).expect("w_q");
    let w_k = GpuBuffer::from_host(executor.context(), &w_k_data).expect("w_k");
    let w_v = GpuBuffer::from_host(executor.context(), &w_v_data).expect("w_v");

    // Output buffers
    let out_q = GpuBuffer::new(executor.context(), hidden_size as usize).expect("out_q");
    let out_k = GpuBuffer::new(executor.context(), kv_dim as usize).expect("out_k");
    let out_v = GpuBuffer::new(executor.context(), kv_dim as usize).expect("out_v");

    let result = executor.fused_qkv_into(
        &x,
        &w_q,
        &w_k,
        &w_v,
        &out_q,
        &out_k,
        &out_v,
        hidden_size,
        kv_dim,
    );
    assert!(
        result.is_ok(),
        "fused_qkv_into should succeed: {:?}",
        result.err()
    );
}

/// Test fused_qkv_into with GQA (different kv_dim)
#[test]
#[serial]
fn test_cov028_fused_qkv_into_gqa() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 128u32;
    let kv_dim = 32u32; // GQA: fewer KV heads

    let x_data = vec![0.1f32; hidden_size as usize];
    let x = GpuBuffer::from_host(executor.context(), &x_data).expect("x");

    let w_q_data = vec![0.01f32; (hidden_size * hidden_size) as usize];
    let w_k_data = vec![0.01f32; (hidden_size * kv_dim) as usize];
    let w_v_data = vec![0.01f32; (hidden_size * kv_dim) as usize];

    let w_q = GpuBuffer::from_host(executor.context(), &w_q_data).expect("w_q");
    let w_k = GpuBuffer::from_host(executor.context(), &w_k_data).expect("w_k");
    let w_v = GpuBuffer::from_host(executor.context(), &w_v_data).expect("w_v");

    let out_q = GpuBuffer::new(executor.context(), hidden_size as usize).expect("out_q");
    let out_k = GpuBuffer::new(executor.context(), kv_dim as usize).expect("out_k");
    let out_v = GpuBuffer::new(executor.context(), kv_dim as usize).expect("out_v");

    let result = executor.fused_qkv_into(
        &x,
        &w_q,
        &w_k,
        &w_v,
        &out_q,
        &out_k,
        &out_v,
        hidden_size,
        kv_dim,
    );
    assert!(
        result.is_ok(),
        "fused_qkv_into with GQA should succeed: {:?}",
        result.err()
    );
}
