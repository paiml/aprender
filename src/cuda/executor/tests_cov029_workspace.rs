
/// Test workspace_batch_size
#[test]
#[serial]
fn test_cov029_workspace_batch_size() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially should be 0
    assert_eq!(
        executor.workspace_batch_size(),
        0,
        "Initial batch size should be 0"
    );

    // After init_workspace
    executor.init_workspace(512, 256).expect("init workspace");
    assert_eq!(
        executor.workspace_batch_size(),
        1,
        "Batch size should be 1 after init"
    );
}

/// Test workspace_batch_size after batched init
#[test]
#[serial]
fn test_cov029_workspace_batch_size_batched() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    executor.init_workspace(512, 256).expect("init workspace");
    executor
        .init_batched_workspace(512, 256, 4)
        .expect("init batched");

    assert_eq!(executor.workspace_batch_size(), 4, "Batch size should be 4");
}

/// Test profiler_mut
#[test]
#[serial]
fn test_cov029_profiler_mut() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Get mutable profiler access
    let profiler = executor.profiler_mut();
    // Just verify we can access it
    assert!(
        std::ptr::from_mut(profiler) as usize != 0,
        "profiler should be valid"
    );
}

/// Test execution_graph_ascii
#[test]
#[serial]
fn test_cov029_execution_graph_ascii() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let ascii = executor.execution_graph_ascii();
    // Just verify the function returns without panic - the actual format varies
    // Empty graph or a tree structure are both valid
    let _ = ascii.len();
}

/// Test tile_stats
#[test]
#[serial]
fn test_cov029_tile_stats() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Access tile stats for different levels
    let _macro_stats = executor.tile_stats(trueno::TileLevel::Macro);
    let _midi_stats = executor.tile_stats(trueno::TileLevel::Midi);
    // Just verify we can access without panic
}

// =============================================================================
// COV-030: CudaExecutor Layer API Coverage Tests
// Target: Increase layer.rs coverage from 20% to higher
// =============================================================================

/// Test workspace_output returns None before init
#[test]
#[serial]
fn test_cov030_workspace_output_before_init() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before init, workspace_output should return None
    assert!(
        executor.workspace_output().is_none(),
        "workspace_output should be None before init"
    );
}

/// Test workspace_output returns Some after init
#[test]
#[serial]
fn test_cov030_workspace_output_after_init() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initialize workspace
    executor.init_workspace(512, 256).expect("init workspace");

    // After init, workspace_output should return Some
    assert!(
        executor.workspace_output().is_some(),
        "workspace_output should be Some after init"
    );
}

/// Test has_rmsnorm_weights before and after preload
#[test]
#[serial]
fn test_cov030_has_rmsnorm_weights() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before preload
    assert!(
        !executor.has_rmsnorm_weights(0),
        "Should not have weights for layer 0 initially"
    );
    assert!(
        !executor.has_rmsnorm_weights(1),
        "Should not have weights for layer 1 initially"
    );

    // Preload for 2 layers - need &[&[f32]] type
    let attn_norm_0 = vec![1.0f32; 512];
    let attn_norm_1 = vec![1.0f32; 512];
    let ffn_norm_0 = vec![1.0f32; 512];
    let ffn_norm_1 = vec![1.0f32; 512];
    let attn_norms: &[&[f32]] = &[&attn_norm_0, &attn_norm_1];
    let ffn_norms: &[&[f32]] = &[&ffn_norm_0, &ffn_norm_1];
    executor
        .preload_rmsnorm_weights(2, attn_norms, ffn_norms)
        .expect("preload");

    // After preload
    assert!(
        executor.has_rmsnorm_weights(0),
        "Should have weights for layer 0 after preload"
    );
    assert!(
        executor.has_rmsnorm_weights(1),
        "Should have weights for layer 1 after preload"
    );
    assert!(
        !executor.has_rmsnorm_weights(2),
        "Should not have weights for layer 2 (not preloaded)"
    );
}

/// Test has_output_norm before and after preload
#[test]
#[serial]
fn test_cov030_has_output_norm() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before preload
    assert!(
        !executor.has_output_norm(),
        "Should not have output norm initially"
    );

    // Preload output norm
    let gamma = vec![1.0f32; 512];
    executor
        .preload_output_norm(&gamma)
        .expect("preload output norm");

    // After preload
    assert!(
        executor.has_output_norm(),
        "Should have output norm after preload"
    );
}

/// Test has_qkv_bias before and after preload
#[test]
#[serial]
fn test_cov030_has_qkv_bias() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before preload
    assert!(
        !executor.has_qkv_bias(0),
        "Should not have QKV bias for layer 0 initially"
    );

    // Preload for 1 layer - need &[Option<&[f32]>] type
    let q_bias_0 = vec![0.0f32; 512];
    let k_bias_0 = vec![0.0f32; 64];
    let v_bias_0 = vec![0.0f32; 64];
    let q_biases: &[Option<&[f32]>] = &[Some(&q_bias_0)];
    let k_biases: &[Option<&[f32]>] = &[Some(&k_bias_0)];
    let v_biases: &[Option<&[f32]>] = &[Some(&v_bias_0)];
    executor
        .preload_qkv_bias(1, q_biases, k_biases, v_biases)
        .expect("preload");

    // After preload
    assert!(
        executor.has_qkv_bias(0),
        "Should have QKV bias for layer 0 after preload"
    );
    assert!(
        !executor.has_qkv_bias(1),
        "Should not have QKV bias for layer 1 (not preloaded)"
    );
}

/// Test has_lm_head_bias before and after preload
#[test]
#[serial]
fn test_cov030_has_lm_head_bias() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before preload
    assert!(
        !executor.has_lm_head_bias(),
        "Should not have LM head bias initially"
    );

    // Preload LM head bias
    let bias = vec![0.0f32; 32000];
    executor
        .preload_lm_head_bias(Some(&bias))
        .expect("preload lm head bias");

    // After preload
    assert!(
        executor.has_lm_head_bias(),
        "Should have LM head bias after preload"
    );
}

/// Test output_rmsnorm_gpu basic operation
#[test]
#[serial]
fn test_cov030_output_rmsnorm_gpu() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 512u32;
    let epsilon = 1e-5f32;

    // Create input, output, and gamma buffers
    let input = vec![1.0f32; hidden_dim as usize];
    let mut output = vec![0.0f32; hidden_dim as usize];
    let gamma = vec![1.0f32; hidden_dim as usize];

    let result = executor.output_rmsnorm_gpu(&input, &mut output, &gamma, hidden_dim, epsilon);
    assert!(
        result.is_ok(),
        "output_rmsnorm_gpu should succeed: {:?}",
        result.err()
    );

    // Output should be normalized (not all zeros)
    let sum: f32 = output.iter().sum();
    assert!(sum.abs() > 0.0, "Output should be non-zero after RMSNorm");
}

/// Test gpu_argmax basic operation
#[test]
#[serial]
fn test_cov030_gpu_argmax_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Create logits buffer with a clear maximum
    let vocab_size = 1024u32;
    let mut logits = vec![0.0f32; vocab_size as usize];
    logits[42] = 100.0; // Make index 42 the maximum

    // Upload to GPU
    let logits_gpu = GpuBuffer::from_host(executor.context(), &logits).expect("upload logits");
    let logits_ptr = logits_gpu.as_ptr();

    let result = executor.gpu_argmax(logits_ptr, vocab_size);
    assert!(
        result.is_ok(),
        "gpu_argmax should succeed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap(), 42, "gpu_argmax should return index 42");
}

/// Test gpu_argmax with large vocab
#[test]
#[serial]
fn test_cov030_gpu_argmax_large_vocab() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Typical LLM vocab size
    let vocab_size = 32000u32;
    let mut logits = vec![-10.0f32; vocab_size as usize];
    logits[31999] = 50.0; // Last index is maximum

    let logits_gpu = GpuBuffer::from_host(executor.context(), &logits).expect("upload logits");
    let logits_ptr = logits_gpu.as_ptr();

    let result = executor.gpu_argmax(logits_ptr, vocab_size);
    assert!(result.is_ok(), "gpu_argmax with large vocab should succeed");
    assert_eq!(result.unwrap(), 31999, "Should find max at last index");
}

/// Test read_hidden_state_to_cpu error before workspace init
#[test]
#[serial]
fn test_cov030_read_hidden_state_error_before_init() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before workspace init, should return error
    let result = executor.read_hidden_state_to_cpu();
    assert!(
        result.is_err(),
        "read_hidden_state_to_cpu should error before workspace init"
    );
}

/// Test transformer_layer_batched error validation
#[test]
#[serial]
fn test_cov030_transformer_layer_batched_validation() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Create a dummy input buffer
    let input_data = vec![0.1f32; 512 * 4];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");

    // Create dummy IndexedLayerWeights
    let layer_weights = test_zeroed_layer_weights();

    // Without workspace init, should return error
    let result = executor.transformer_layer_batched(
        &input,
        0,
        &layer_weights,
        4,
        &[0, 1, 2, 3],
        512,
        256,
        1e-5,
    );
    assert!(
        result.is_err(),
        "transformer_layer_batched should error without workspace"
    );
}

/// Test transformer_layer_batched batch size mismatch
#[test]
#[serial]
fn test_cov030_transformer_layer_batched_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Init workspace for batch_size=2
    executor.init_workspace(512, 256).expect("init workspace");
    executor
        .init_batched_workspace(512, 256, 2)
        .expect("init batched");

    let input_data = vec![0.1f32; 512 * 4];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let layer_weights = test_zeroed_layer_weights();

    // Request batch_size=4 but workspace is for 2
    let result = executor.transformer_layer_batched(
        &input,
        0,
        &layer_weights,
        4, // mismatch!
        &[0, 1, 2, 3],
        512,
        256,
        1e-5,
    );
    assert!(result.is_err(), "Should error on batch size mismatch");
}
