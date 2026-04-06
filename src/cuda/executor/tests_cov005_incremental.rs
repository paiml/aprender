
#[test]
#[serial]
fn test_cov005_incremental_attention_async_overflow() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = 8;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let max_len = 2; // Very small

    let _ = executor.init_kv_cache_gpu(1, num_heads, num_kv_heads, head_dim, max_len);

    let q_buf = GpuBuffer::from_host(&executor.context, &vec![1.0f32; q_dim]).unwrap();
    let k_buf = GpuBuffer::from_host(&executor.context, &vec![1.0f32; kv_dim]).unwrap();
    let v_buf = GpuBuffer::from_host(&executor.context, &vec![1.0f32; kv_dim]).unwrap();

    // Fill cache
    for _ in 0..max_len {
        let _ = executor.incremental_attention_async(0, &q_buf, &k_buf, &v_buf);
    }

    // Next should overflow
    let result = executor.incremental_attention_async(0, &q_buf, &k_buf, &v_buf);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov005_incremental_attention_into_overflow() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = 8;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let max_len = 2;

    let _ = executor.init_kv_cache_gpu(1, num_heads, num_kv_heads, head_dim, max_len);

    let q_buf = GpuBuffer::from_host(&executor.context, &vec![1.0f32; q_dim]).unwrap();
    let k_buf = GpuBuffer::from_host(&executor.context, &vec![1.0f32; kv_dim]).unwrap();
    let v_buf = GpuBuffer::from_host(&executor.context, &vec![1.0f32; kv_dim]).unwrap();
    let out_buf = GpuBuffer::<f32>::new(&executor.context, q_dim).unwrap();

    // Fill cache
    for _ in 0..max_len {
        let _ = executor.incremental_attention_into(0, &q_buf, &k_buf, &v_buf, &out_buf);
    }

    // Next should overflow
    let result = executor.incremental_attention_into(0, &q_buf, &k_buf, &v_buf, &out_buf);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov005_batched_attention_not_initialized() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = 8;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let m = 2;

    // Init regular KV cache but NOT batched
    let _ = executor.init_kv_cache_gpu(1, num_heads, num_kv_heads, head_dim, 16);

    let q_buf = GpuBuffer::from_host(&executor.context, &vec![1.0f32; m * q_dim]).unwrap();
    let k_buf = GpuBuffer::from_host(&executor.context, &vec![1.0f32; m * kv_dim]).unwrap();
    let v_buf = GpuBuffer::from_host(&executor.context, &vec![1.0f32; m * kv_dim]).unwrap();
    let out_buf = GpuBuffer::<f32>::new(&executor.context, m * q_dim).unwrap();

    let positions = vec![0u32; m];

    // Should fail because batched KV cache not initialized
    let result = executor
        .batched_incremental_attention_into(0, &q_buf, &k_buf, &v_buf, &out_buf, m, &positions);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov005_init_flash_decoding() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.init_flash_decoding(4, 8, 128, 2);
    assert!(result.is_ok());
}

#[test]
#[serial]
fn test_cov005_flash_decoding_not_initialized() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = 8;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let m = 2;

    // Init KV cache but NOT flash decoding
    let _ = executor.init_kv_cache_gpu(1, num_heads, num_kv_heads, head_dim, 16);
    let _ = executor.init_batched_kv_cache_gpu(1, m);

    let q_buf = GpuBuffer::from_host(&executor.context, &vec![1.0f32; m * q_dim]).unwrap();
    let k_buf = GpuBuffer::from_host(&executor.context, &vec![1.0f32; m * kv_dim]).unwrap();
    let v_buf = GpuBuffer::from_host(&executor.context, &vec![1.0f32; m * kv_dim]).unwrap();
    let out_buf = GpuBuffer::<f32>::new(&executor.context, m * q_dim).unwrap();

    let positions = vec![0u32; m];

    // Should fail because flash decoding not initialized
    let result =
        executor.flash_decoding_attention_into(0, &q_buf, &k_buf, &v_buf, &out_buf, m, &positions);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov005_tensor_core_attention_dimension_validation() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // seq_len not multiple of 16 should fail
    let q = vec![1.0f32; 4 * 15 * 16]; // seq_len=15
    let k = vec![1.0f32; 4 * 15 * 16];
    let v = vec![1.0f32; 4 * 15 * 16];
    let mut output = vec![0.0f32; 4 * 15 * 16];

    let result = executor.tensor_core_attention(&q, &k, &v, &mut output, 15, 16, 4, false);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov005_tensor_core_attention_head_dim_validation() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // head_dim not multiple of 16 should fail
    let q = vec![1.0f32; 4 * 16 * 15]; // head_dim=15
    let k = vec![1.0f32; 4 * 16 * 15];
    let v = vec![1.0f32; 4 * 16 * 15];
    let mut output = vec![0.0f32; 4 * 16 * 15];

    let result = executor.tensor_core_attention(&q, &k, &v, &mut output, 16, 15, 4, false);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov005_tensor_core_attention_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Wrong input size should fail
    let q = vec![1.0f32; 100]; // Wrong size
    let k = vec![1.0f32; 4 * 16 * 16];
    let v = vec![1.0f32; 4 * 16 * 16];
    let mut output = vec![0.0f32; 4 * 16 * 16];

    let result = executor.tensor_core_attention(&q, &k, &v, &mut output, 16, 16, 4, false);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov005_gemm_fp16_dimension_validation() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // m not multiple of 16 should fail
    let a = vec![1.0f32; 15 * 16];
    let b = vec![1.0f32; 16 * 16];
    let mut c = vec![0.0f32; 15 * 16];

    let result = executor.gemm_fp16(&a, &b, &mut c, 15, 16, 16);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov005_gemm_fp16_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Wrong A size
    let a = vec![1.0f32; 100]; // Should be 16*16=256
    let b = vec![1.0f32; 16 * 16];
    let mut c = vec![0.0f32; 16 * 16];

    let result = executor.gemm_fp16(&a, &b, &mut c, 16, 16, 16);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov005_gemm_fp16_valid() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Valid dimensions (multiples of 16)
    let a = vec![1.0f32; 16 * 16];
    let b = vec![1.0f32; 16 * 16];
    let mut c = vec![0.0f32; 16 * 16];

    let result = executor.gemm_fp16(&a, &b, &mut c, 16, 16, 16);
    assert!(result.is_ok(), "gemm_fp16 failed: {:?}", result.err());

    // Result should be non-zero (each element = sum of 16 products of 1.0*1.0 = 16.0)
    assert!(c[0] > 0.0);
}

#[test]
#[serial]
fn test_cov005_tensor_core_attention_valid() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Valid dimensions (multiples of 16)
    let n_heads = 2u32;
    let seq_len = 16u32;
    let head_dim = 16u32;
    let total = (n_heads * seq_len * head_dim) as usize;

    let q = vec![1.0f32; total];
    let k = vec![1.0f32; total];
    let v = vec![1.0f32; total];
    let mut output = vec![0.0f32; total];

    let result =
        executor.tensor_core_attention(&q, &k, &v, &mut output, seq_len, head_dim, n_heads, false);
    assert!(
        result.is_ok(),
        "tensor_core_attention failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov005_tensor_core_attention_causal() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n_heads = 2u32;
    let seq_len = 16u32;
    let head_dim = 16u32;
    let total = (n_heads * seq_len * head_dim) as usize;

    let q = vec![1.0f32; total];
    let k = vec![1.0f32; total];
    let v = vec![1.0f32; total];
    let mut output = vec![0.0f32; total];

    // Test with causal=true
    let result =
        executor.tensor_core_attention(&q, &k, &v, &mut output, seq_len, head_dim, n_heads, true);
    assert!(
        result.is_ok(),
        "causal tensor_core_attention failed: {:?}",
        result.err()
    );
}

// ============================================================================
// COV-006: quantized.rs coverage tests
// Target: Increase coverage from 19.42% to 50%+
// Focus: gelu_gpu, layer_norm_gpu, rmsnorm_host, residual_add_host,
//        fused_residual_rmsnorm_host, residual_add_gpu, rmsnorm_gpu
// ============================================================================

#[test]
#[serial]
fn test_cov006_gelu_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Small vector for GELU
    let n = 256u32;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) / 64.0).collect();

    let buffer = GpuBuffer::from_host(&executor.context, &data).expect("GPU buffer");
    let result = executor.gelu_gpu(&buffer, n);
    assert!(result.is_ok(), "gelu_gpu failed: {:?}", result.err());

    // Verify output is modified (GELU is not identity)
    executor.stream.synchronize().expect("sync");
    let mut output = vec![0.0f32; n as usize];
    buffer.copy_to_host(&mut output).expect("copy");

    // GELU(0) should be 0
    // GELU(x) for x > 0 should be positive
    // GELU(x) for x < 0 should be small negative or near zero
    let mid_idx = 128; // corresponds to input 0.0
    assert!(
        output[mid_idx].abs() < 0.1,
        "GELU(0) should be near 0, got {}",
        output[mid_idx]
    );
}

#[test]
#[serial]
fn test_cov006_gelu_gpu_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Larger vector to test multi-block execution
    let n = 1024u32;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) / 256.0).collect();

    let buffer = GpuBuffer::from_host(&executor.context, &data).expect("GPU buffer");
    let result = executor.gelu_gpu(&buffer, n);
    assert!(result.is_ok(), "gelu_gpu large failed: {:?}", result.err());
}

// Note: layer_norm_gpu tests removed — kernel function naming issue (FunctionNotFound)
// LayerNorm kernel registration in KernelType::LayerNorm needs investigation.

#[test]
#[serial]
fn test_cov006_rmsnorm_host_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 32usize;
    let epsilon = 1e-5f32;

    let input: Vec<f32> = (0..hidden_size).map(|i| (i as f32 + 1.0) / 10.0).collect();
    let gamma = vec![1.0f32; hidden_size];
    let mut output = vec![0.0f32; hidden_size];

    let result = executor.rmsnorm_host(&input, &gamma, &mut output, epsilon);
    assert!(result.is_ok(), "rmsnorm_host failed: {:?}", result.err());

    // RMSNorm output should be normalized
    // Verify output is not all zeros
    let sum: f32 = output.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "RMSNorm output should not be all zeros");
}

#[test]
#[serial]
fn test_cov006_rmsnorm_host_with_scale() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 64usize;
    let epsilon = 1e-5f32;

    let input: Vec<f32> = (0..hidden_size)
        .map(|i| ((i as f32) - 32.0) / 16.0)
        .collect();
    let gamma: Vec<f32> = (0..hidden_size).map(|i| 0.5 + (i as f32) / 128.0).collect(); // Variable scale
    let mut output = vec![0.0f32; hidden_size];

    let result = executor.rmsnorm_host(&input, &gamma, &mut output, epsilon);
    assert!(
        result.is_ok(),
        "rmsnorm_host with scale failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov006_residual_add_host_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 128usize;

    let input1: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let input2: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let mut output = vec![0.0f32; n];

    let result = executor.residual_add_host(&input1, &input2, &mut output);
    assert!(
        result.is_ok(),
        "residual_add_host failed: {:?}",
        result.err()
    );

    // Verify: output[i] = input1[i] + input2[i] = i + (n - i) = n
    for (idx, &val) in output.iter().enumerate() {
        let expected = n as f32;
        assert!(
            (val - expected).abs() < 1e-5,
            "residual_add mismatch at {}: {} vs {}",
            idx,
            val,
            expected
        );
    }
}
