
#[test]
#[serial]
fn test_cov018_fused_residual_rmsnorm_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let residual = vec![1.0f32; n as usize];
    let input = vec![0.5f32; n as usize];
    let gamma = vec![1.0f32; n as usize];

    let residual_gpu = GpuBuffer::from_host(&executor.context, &residual).expect("residual");
    let input_gpu = GpuBuffer::from_host(&executor.context, &input).expect("input");
    let gamma_gpu = GpuBuffer::from_host(&executor.context, &gamma).expect("gamma");
    let output_gpu = GpuBuffer::<f32>::new(&executor.context, n as usize).expect("output");

    // gamma_ptr: usize, output, hidden_size, epsilon
    let result = executor.fused_residual_rmsnorm_into(
        &residual_gpu,
        &input_gpu,
        gamma_gpu.as_ptr() as usize,
        &output_gpu,
        n,
        1e-5,
    );
    assert!(
        result.is_ok(),
        "fused_residual_rmsnorm_into failed: {:?}",
        result.err()
    );
}

// =========================================================================
// COV-019: attention.rs coverage tests
// Target: 49.32% -> ~55%+ coverage
// =========================================================================

#[test]
fn test_cov019_flash_attention_memory_bytes() {
    // Pure function test - no CUDA required
    let seq_len = 512u32;
    let head_dim = 64u32;

    let (naive, flash) = CudaExecutor::flash_attention_memory_bytes(seq_len, head_dim);

    // Naive: seq_len^2 * 4 bytes
    assert_eq!(naive, 512 * 512 * 4, "Naive memory should be seq_len^2 * 4");

    // Flash: block_size^2 * 4 * 2 (block_size=64)
    assert_eq!(
        flash,
        64 * 64 * 4 * 2,
        "Flash memory should be block_size^2 * 4 * 2"
    );

    // Flash should use much less memory
    assert!(flash < naive, "Flash should use less memory than naive");
}

#[test]
fn test_cov019_flash_attention_memory_bytes_large_seq() {
    // Test with larger sequence
    let seq_len = 4096u32;
    let head_dim = 128u32;

    let (naive, flash) = CudaExecutor::flash_attention_memory_bytes(seq_len, head_dim);

    // Naive scales quadratically
    assert_eq!(naive, 4096u64 * 4096 * 4);

    // Flash stays constant
    assert_eq!(flash, 64 * 64 * 4 * 2);

    // Flash savings are huge for large sequences
    assert!(
        flash < naive / 1000,
        "Flash should save >1000x memory for large sequences"
    );
}

#[test]
#[serial]
fn test_cov019_tensor_core_attention_dimension_not_multiple_16() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // seq_len=17 is not multiple of 16
    let seq_len = 17u32;
    let head_dim = 64u32; // Valid
    let n_heads = 4u32;

    let size = (seq_len * head_dim * n_heads) as usize;
    let q = vec![0.1f32; size];
    let k = vec![0.1f32; size];
    let v = vec![0.1f32; size];
    let mut output = vec![0.0f32; size];

    let result =
        executor.tensor_core_attention(&q, &k, &v, &mut output, seq_len, head_dim, n_heads, false);

    assert!(
        result.is_err(),
        "Should fail with dimension not multiple of 16"
    );
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("multiple of 16"),
        "Error should mention multiple of 16: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov019_tensor_core_attention_head_dim_not_multiple_16() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // head_dim=63 is not multiple of 16
    let seq_len = 32u32; // Valid
    let head_dim = 63u32;
    let n_heads = 4u32;

    let size = (seq_len * head_dim * n_heads) as usize;
    let q = vec![0.1f32; size];
    let k = vec![0.1f32; size];
    let v = vec![0.1f32; size];
    let mut output = vec![0.0f32; size];

    let result =
        executor.tensor_core_attention(&q, &k, &v, &mut output, seq_len, head_dim, n_heads, true);

    assert!(
        result.is_err(),
        "Should fail with head_dim not multiple of 16"
    );
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("multiple of 16"),
        "Error should mention multiple of 16: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov019_tensor_core_attention_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let seq_len = 32u32;
    let head_dim = 64u32;
    let n_heads = 4u32;
    let expected_size = (seq_len * head_dim * n_heads) as usize;

    // Q is correct size
    let q = vec![0.1f32; expected_size];
    // K is wrong size
    let k = vec![0.1f32; expected_size - 100];
    let v = vec![0.1f32; expected_size];
    let mut output = vec![0.0f32; expected_size];

    let result =
        executor.tensor_core_attention(&q, &k, &v, &mut output, seq_len, head_dim, n_heads, false);

    assert!(result.is_err(), "Should fail with size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("size mismatch") || err_msg.contains("expected"),
        "Error should mention size mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov019_gemm_fp16_dimension_not_multiple_16() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // m=17 is not multiple of 16
    let m = 17u32;
    let n = 32u32;
    let k = 64u32;

    let a = vec![0.1f32; (m * k) as usize];
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm_fp16(&a, &b, &mut c, m, n, k);

    assert!(
        result.is_err(),
        "Should fail with dimension not multiple of 16"
    );
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("multiple of 16"),
        "Error should mention multiple of 16: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov019_gemm_fp16_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32;
    let n = 32u32;
    let k = 64u32;

    // A is wrong size
    let a = vec![0.1f32; 100]; // Should be m*k = 2048
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm_fp16(&a, &b, &mut c, m, n, k);

    assert!(result.is_err(), "Should fail with size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("mismatch") || err_msg.contains("expected"),
        "Error should mention mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov019_batched_incremental_attention_not_initialized() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Don't initialize batched KV cache
    let num_heads = 4usize;
    let head_dim = 64usize;
    let m = 2usize;
    let positions = vec![0u32, 0u32];

    let q_batched = GpuBuffer::<f32>::new(&executor.context, m * num_heads * head_dim).expect("q");
    let k_batched = GpuBuffer::<f32>::new(&executor.context, m * num_heads * head_dim).expect("k");
    let v_batched = GpuBuffer::<f32>::new(&executor.context, m * num_heads * head_dim).expect("v");
    let out_batched =
        GpuBuffer::<f32>::new(&executor.context, m * num_heads * head_dim).expect("out");

    let result = executor.batched_incremental_attention_into(
        0, // layer_idx
        &q_batched,
        &k_batched,
        &v_batched,
        &out_batched,
        m,
        &positions,
    );

    assert!(result.is_err(), "Should fail without batched KV cache init");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not initialized") || err_msg.contains("PAR-119"),
        "Error should mention not initialized: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov019_flash_decoding_not_initialized() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Don't initialize flash decoding
    let num_heads = 4usize;
    let head_dim = 64usize;
    let m = 2usize;
    let positions = vec![0u32, 0u32];

    let q_batched = GpuBuffer::<f32>::new(&executor.context, m * num_heads * head_dim).expect("q");
    let k_batched = GpuBuffer::<f32>::new(&executor.context, m * num_heads * head_dim).expect("k");
    let v_batched = GpuBuffer::<f32>::new(&executor.context, m * num_heads * head_dim).expect("v");
    let out_batched =
        GpuBuffer::<f32>::new(&executor.context, m * num_heads * head_dim).expect("out");

    let result = executor.flash_decoding_attention_into(
        0, // layer_idx
        &q_batched,
        &k_batched,
        &v_batched,
        &out_batched,
        m,
        &positions,
    );

    assert!(result.is_err(), "Should fail without flash decoding init");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not initialized") || err_msg.contains("PAR-118"),
        "Error should mention not initialized: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov019_init_flash_decoding_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 8usize;
    let head_dim = 64usize;
    let max_seq_len = 512usize;
    let batch_size = 4usize;

    let result = executor.init_flash_decoding(num_heads, head_dim, max_seq_len, batch_size);

    assert!(
        result.is_ok(),
        "init_flash_decoding should succeed: {:?}",
        result.err()
    );
    assert!(
        executor.flash_decode_enabled,
        "flash_decode_enabled should be true"
    );
    assert_eq!(executor.flash_decode_max_seq_len, max_seq_len);
}

#[test]
#[serial]
fn test_cov019_incremental_attention_async_kv_not_initialized() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set up KV cache parameters but don't init the cache buffers
    executor.kv_num_heads = 4;
    executor.kv_num_kv_heads = 4;
    executor.kv_head_dim = 64;
    executor.kv_cache_max_len = 128;

    let q_dim = executor.kv_num_heads * executor.kv_head_dim;
    let kv_dim = executor.kv_num_kv_heads * executor.kv_head_dim;

    let q_gpu = GpuBuffer::<f32>::new(&executor.context, q_dim).expect("q");
    let k_gpu = GpuBuffer::<f32>::new(&executor.context, kv_dim).expect("k");
    let v_gpu = GpuBuffer::<f32>::new(&executor.context, kv_dim).expect("v");

    let result = executor.incremental_attention_async(0, &q_gpu, &k_gpu, &v_gpu);

    assert!(result.is_err(), "Should fail without KV cache init");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not initialized") || err_msg.contains("PAR-023"),
        "Error should mention not initialized: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov019_incremental_attention_into_kv_not_initialized() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set up KV cache parameters but don't init the cache buffers
    executor.kv_num_heads = 4;
    executor.kv_num_kv_heads = 4;
    executor.kv_head_dim = 64;
    executor.kv_cache_max_len = 128;

    let q_dim = executor.kv_num_heads * executor.kv_head_dim;
    let kv_dim = executor.kv_num_kv_heads * executor.kv_head_dim;

    let q_gpu = GpuBuffer::<f32>::new(&executor.context, q_dim).expect("q");
    let k_gpu = GpuBuffer::<f32>::new(&executor.context, kv_dim).expect("k");
    let v_gpu = GpuBuffer::<f32>::new(&executor.context, kv_dim).expect("v");
    let out_gpu = GpuBuffer::<f32>::new(&executor.context, q_dim).expect("out");

    let result = executor.incremental_attention_into(0, &q_gpu, &k_gpu, &v_gpu, &out_gpu);

    assert!(result.is_err(), "Should fail without KV cache init");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not initialized") || err_msg.contains("PAR-052"),
        "Error should mention not initialized: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov019_tensor_core_attention_valid_run() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // All dimensions multiples of 16
    let seq_len = 32u32;
    let head_dim = 64u32;
    let n_heads = 4u32;

    let size = (seq_len * head_dim * n_heads) as usize;
    let q = vec![0.1f32; size];
    let k = vec![0.1f32; size];
    let v = vec![0.1f32; size];
    let mut output = vec![0.0f32; size];

    // This should succeed (dimensions are valid)
    let result =
        executor.tensor_core_attention(&q, &k, &v, &mut output, seq_len, head_dim, n_heads, false);

    assert!(
        result.is_ok(),
        "tensor_core_attention should succeed: {:?}",
        result.err()
    );

    // Output should have some non-zero values
    let has_nonzero = output.iter().any(|&x| x.abs() > 1e-10);
    assert!(has_nonzero, "Output should have non-zero values");
}
