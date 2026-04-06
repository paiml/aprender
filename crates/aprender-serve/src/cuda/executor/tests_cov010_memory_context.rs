
#[test]
#[serial]
fn test_cov010_memory_info() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.memory_info();
    assert!(result.is_ok(), "memory_info failed: {:?}", result.err());

    let (free, total) = result.unwrap();
    assert!(total > 0, "Total memory should be > 0");
    assert!(free <= total, "Free memory should be <= total");
}

#[test]
#[serial]
fn test_cov010_context() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get context reference
    let _context = executor.context();
}

#[test]
#[serial]
fn test_cov010_synchronize() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize();
    assert!(result.is_ok(), "synchronize failed: {:?}", result.err());
}

#[test]
#[serial]
fn test_cov010_pool_stats() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let stats = executor.pool_stats();
    // Stats should return valid struct (total_allocated is usize, always >= 0)
    let _ = stats.total_allocated; // Just verify field access works
}

#[test]
#[serial]
fn test_cov010_staging_pool_stats() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let stats = executor.staging_pool_stats();
    // Stats should return valid struct (total_allocated is usize, always >= 0)
    let _ = stats.total_allocated; // Just verify field access works
}

#[test]
#[serial]
fn test_cov010_staging_buffer_roundtrip() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get a staging buffer (minimum size is 1024)
    let buf = executor.get_staging_buffer(256);
    assert!(
        buf.len() >= 256,
        "Staging buffer should be at least 256 elements"
    );

    // Return it to the pool
    executor.return_staging_buffer(buf);
}

#[test]
#[serial]
fn test_cov010_clear_pool() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Clear pool (should not panic even when empty)
    executor.clear_pool();
}

// ============================================================================
// COV-011: layer.rs additional coverage tests
// Target: Increase coverage from 15.49%
// Focus: preload functions, cache functions, workspace output, read hidden state
// ============================================================================

#[test]
#[serial]
fn test_cov011_preload_output_norm() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let gamma = vec![1.0f32; 64];
    let result = executor.preload_output_norm(&gamma);
    assert!(
        result.is_ok(),
        "preload_output_norm failed: {:?}",
        result.err()
    );

    assert!(
        executor.has_output_norm(),
        "Should have output norm after preload"
    );
}

#[test]
#[serial]
fn test_cov011_has_output_norm_false_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        !executor.has_output_norm(),
        "Should not have output norm initially"
    );
}

#[test]
#[serial]
fn test_cov011_cache_rmsnorm_gamma() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let gamma = vec![1.0f32; 128];
    let result = executor.cache_rmsnorm_gamma("test_layer_0_attn_norm", &gamma);
    assert!(
        result.is_ok(),
        "cache_rmsnorm_gamma failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov011_preload_qkv_bias() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Function expects &[Option<&[f32]>] for each bias array (per-head optional biases)
    let q_bias_data = vec![0.1f32; 64];
    let k_bias_data = vec![0.2f32; 64];
    let v_bias_data = vec![0.3f32; 64];

    // Wrap as optional slices (one head with bias)
    let q_biases: Vec<Option<&[f32]>> = vec![Some(q_bias_data.as_slice())];
    let k_biases: Vec<Option<&[f32]>> = vec![Some(k_bias_data.as_slice())];
    let v_biases: Vec<Option<&[f32]>> = vec![Some(v_bias_data.as_slice())];

    // Pass 1 as num_layers (not layer index 0)
    let result = executor.preload_qkv_bias(1, &q_biases, &k_biases, &v_biases);
    assert!(
        result.is_ok(),
        "preload_qkv_bias failed: {:?}",
        result.err()
    );

    assert!(executor.has_qkv_bias(0), "Should have QKV bias for layer 0");
}

#[test]
#[serial]
fn test_cov011_has_qkv_bias_false_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        !executor.has_qkv_bias(0),
        "Should not have QKV bias initially"
    );
    assert!(
        !executor.has_qkv_bias(5),
        "Should not have QKV bias for any layer"
    );
}

#[test]
#[serial]
fn test_cov011_preload_lm_head_bias() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Preload with Some bias
    let bias = vec![0.1f32; 1024];
    let result = executor.preload_lm_head_bias(Some(&bias));
    assert!(
        result.is_ok(),
        "preload_lm_head_bias failed: {:?}",
        result.err()
    );

    assert!(
        executor.has_lm_head_bias(),
        "Should have LM head bias after preload"
    );
}

#[test]
#[serial]
fn test_cov011_preload_lm_head_bias_none() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Preload with None (no bias)
    let result = executor.preload_lm_head_bias(None);
    assert!(
        result.is_ok(),
        "preload_lm_head_bias None failed: {:?}",
        result.err()
    );

    assert!(
        !executor.has_lm_head_bias(),
        "Should not have LM head bias when None"
    );
}

#[test]
#[serial]
fn test_cov011_has_lm_head_bias_false_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        !executor.has_lm_head_bias(),
        "Should not have LM head bias initially"
    );
}

#[test]
#[serial]
fn test_cov011_workspace_output_none_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        executor.workspace_output().is_none(),
        "Workspace output should be None initially"
    );
}

#[test]
#[serial]
fn test_cov011_workspace_output_after_init() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Init KV cache and workspace
    let _ = executor.init_kv_cache_gpu(1, 4, 4, 8, 16);
    let _ = executor.init_workspace(32, 64);

    // Workspace output may still be None until forward pass, but the method should work
    let _output = executor.workspace_output();
}

// NOTE: fused_ffn_swiglu_host requires pre-cached GPU weights looked up by name.
// Weight caching is covered by forward_gpu_resident tests. Removing direct test
// since weight setup requires full model context.

#[test]
#[serial]
fn test_cov011_gpu_argmax_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Create a simple logits buffer on GPU
    let vocab_size = 256u32;
    let logits: Vec<f32> = (0..vocab_size).map(|i| i as f32).collect();

    let logits_buf = GpuBuffer::from_host(&executor.context, &logits).expect("logits buffer");
    executor.stream.synchronize().expect("sync");

    let result = executor.gpu_argmax(logits_buf.as_ptr(), vocab_size);
    assert!(result.is_ok(), "gpu_argmax failed: {:?}", result.err());

    let argmax_idx = result.unwrap();
    // The maximum value is at index vocab_size-1 (255)
    assert_eq!(
        argmax_idx,
        vocab_size - 1,
        "Argmax should return index of max value"
    );
}

#[test]
#[serial]
fn test_cov011_gpu_argmax_middle() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Create logits with max in the middle
    let vocab_size = 128u32;
    let mut logits = vec![0.0f32; vocab_size as usize];
    logits[64] = 100.0; // Max at index 64

    let logits_buf = GpuBuffer::from_host(&executor.context, &logits).expect("logits buffer");
    executor.stream.synchronize().expect("sync");

    let result = executor.gpu_argmax(logits_buf.as_ptr(), vocab_size);
    assert!(
        result.is_ok(),
        "gpu_argmax middle failed: {:?}",
        result.err()
    );

    let argmax_idx = result.unwrap();
    assert_eq!(argmax_idx, 64, "Argmax should return 64");
}

// ==============================================================================
// COV-012: Additional quantized.rs coverage - batched operations
// ==============================================================================

#[test]
#[serial]
fn test_cov012_rmsnorm_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 64u32;
    let input: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.01).collect();
    let gamma = vec![1.0f32; hidden_size as usize]; // Unit scale

    let input_gpu = GpuBuffer::from_host(&executor.context, &input).expect("input buffer");
    let gamma_gpu = GpuBuffer::from_host(&executor.context, &gamma).expect("gamma buffer");
    let output_gpu =
        GpuBuffer::<f32>::new(&executor.context, hidden_size as usize).expect("output buffer");

    let result = executor.rmsnorm_into(&input_gpu, &gamma_gpu, &output_gpu, hidden_size, 1e-5);
    assert!(result.is_ok(), "rmsnorm_into failed: {:?}", result.err());

    executor.stream.synchronize().expect("sync");
    let mut output = vec![0.0f32; hidden_size as usize];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // RMSNorm normalizes: output should have reasonable L2 norm
    let l2: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(l2 > 0.0, "Output should have non-zero L2 norm");
}

#[test]
#[serial]
fn test_cov012_batched_rmsnorm_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 32u32;
    let batch_size = 4u32;
    let total = (hidden_size * batch_size) as usize;

    // Create packed input [M × hidden_size]
    let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let gamma = vec![1.0f32; hidden_size as usize]; // Shared gamma

    let input_gpu = GpuBuffer::from_host(&executor.context, &input).expect("input buffer");
    let gamma_gpu = GpuBuffer::from_host(&executor.context, &gamma).expect("gamma buffer");
    let output_gpu = GpuBuffer::<f32>::new(&executor.context, total).expect("output buffer");

    let result = executor.batched_rmsnorm_into(
        &input_gpu,
        &gamma_gpu,
        &output_gpu,
        hidden_size,
        batch_size,
        1e-5,
    );
    assert!(
        result.is_ok(),
        "batched_rmsnorm_into failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");
    let mut output = vec![0.0f32; total];
    output_gpu.copy_to_host(&mut output).expect("copy to host");

    // Check each sequence in batch was normalized
    for seq in 0..batch_size {
        let start = (seq * hidden_size) as usize;
        let end = start + hidden_size as usize;
        let l2: f32 = output[start..end].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(l2 > 0.0, "Sequence {} should have non-zero L2 norm", seq);
    }
}
