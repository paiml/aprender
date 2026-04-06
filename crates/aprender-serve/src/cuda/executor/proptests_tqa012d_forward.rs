
/// T-QA-012d: Test multi-layer forward pass via forward_all_layers_gpu
///
/// Tests the complete forward pass through multiple transformer layers.
#[test]
#[serial]
fn test_tqa012d_forward_all_layers_gpu() {
    if !CudaExecutor::is_available() {
        println!("T-QA-012d: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Use 2 layers to test multi-layer handling
    let hidden_dim = 256u32;
    let intermediate_dim = 1024u32;
    let num_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 64usize;
    let num_layers = 2usize;
    let max_seq_len = 32usize;
    let epsilon = 1e-5f32;

    // Initialize KV cache for all layers
    executor
        .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
        .expect("T-QA-012d: KV cache init");

    // Load weights for both layers
    for layer_idx in 0..num_layers {
        let layer_prefix = format!("blk.{}", layer_idx);

        let qkvo_weights = create_mock_q4k_weights_for_harness(256, 256);
        executor
            .load_quantized_weights(&format!("{}.attn_q.weight", layer_prefix), &qkvo_weights)
            .expect("CUDA operation failed");
        executor
            .load_quantized_weights(&format!("{}.attn_k.weight", layer_prefix), &qkvo_weights)
            .expect("CUDA operation failed");
        executor
            .load_quantized_weights(&format!("{}.attn_v.weight", layer_prefix), &qkvo_weights)
            .expect("CUDA operation failed");
        executor
            .load_quantized_weights(
                &format!("{}.attn_output.weight", layer_prefix),
                &qkvo_weights,
            )
            .expect("CUDA operation failed");

        let gate_up_weights = create_mock_q4k_weights_for_harness(1024, 256);
        let down_weights = create_mock_q4k_weights_for_harness(256, 1024);
        executor
            .load_quantized_weights(
                &format!("{}.ffn_gate.weight", layer_prefix),
                &gate_up_weights,
            )
            .expect("CUDA operation failed");
        executor
            .load_quantized_weights(&format!("{}.ffn_up.weight", layer_prefix), &gate_up_weights)
            .expect("CUDA operation failed");
        executor
            .load_quantized_weights(&format!("{}.ffn_down.weight", layer_prefix), &down_weights)
            .expect("CUDA operation failed");

        // Cache RMSNorm gammas
        let gamma = vec![1.0f32; hidden_dim as usize];
        executor
            .cache_rmsnorm_gamma(&format!("blk.{}.attn_norm.gamma", layer_idx), &gamma)
            .expect("CUDA operation failed");
        executor
            .cache_rmsnorm_gamma(&format!("blk.{}.ffn_norm.gamma", layer_idx), &gamma)
            .expect("CUDA operation failed");
    }

    // Cache output norm using preload_output_norm
    let gamma = vec![1.0f32; hidden_dim as usize];
    executor
        .preload_output_norm(&gamma)
        .expect("CUDA operation failed");

    // Cache LM head (output.weight) for final projection
    let lm_head_weights = create_mock_q4k_weights_for_harness(1000, 256); // vocab_size=1000
    executor
        .load_quantized_weights("output.weight", &lm_head_weights)
        .expect("CUDA operation failed");

    // Build indexed weights for forward_all_layers_gpu
    // GH-279: Use default (LLaMA-like) arch for test fixtures
    let arch = crate::gguf::ArchConstraints::from_architecture("llama");
    executor
        .build_indexed_weights(num_layers, |layer_idx| format!("blk.{}", layer_idx), &arch)
        .expect("T-QA-012d: Build indexed weights");

    // Input/output slices (forward_all_layers_gpu uses slices, not GpuBuffer)
    let input_data = vec![0.1f32; hidden_dim as usize];
    let mut output_data = vec![0.0f32; hidden_dim as usize];
    let position = 0u32;

    // Execute forward_all_layers_gpu
    let result = executor.forward_all_layers_gpu(
        &input_data,
        &mut output_data,
        position,
        num_layers,
        hidden_dim,
        intermediate_dim,
        epsilon,
    );

    assert!(
        result.is_ok(),
        "T-QA-012d: forward_all_layers_gpu should execute: {:?}",
        result.err()
    );
    assert_eq!(output_data.len(), hidden_dim as usize);
    println!("T-QA-012d: forward_all_layers_gpu multi-layer execution PASSED");
}

/// T-QA-012e: Test error handling for missing weights
///
/// Verifies that transformer_layer_gpu returns appropriate error when weights are missing.
#[test]
#[serial]
fn test_tqa012e_transformer_layer_gpu_missing_weights() {
    if !CudaExecutor::is_available() {
        println!("T-QA-012e: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 256u32;
    let intermediate_dim = 1024u32;
    let num_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 64usize;
    let num_layers = 1usize;
    let max_seq_len = 32usize;
    let layer_idx = 0usize;
    let layer_prefix = "blk.0";
    let epsilon = 1e-5f32;

    // Initialize KV cache but DON'T load weights
    executor
        .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
        .expect("T-QA-012e: KV cache init");

    let gamma = vec![1.0f32; hidden_dim as usize];
    let attn_norm_gamma =
        GpuBuffer::from_host(&executor.context, &gamma).expect("CUDA operation failed");
    let ffn_norm_gamma =
        GpuBuffer::from_host(&executor.context, &gamma).expect("CUDA operation failed");

    let input_data = vec![0.1f32; hidden_dim as usize];
    let input =
        GpuBuffer::from_host(&executor.context, &input_data).expect("CUDA operation failed");

    // Attempt execution without weights - should fail
    let result = executor.transformer_layer_gpu(
        &input,
        layer_idx,
        layer_prefix,
        hidden_dim,
        intermediate_dim,
        &attn_norm_gamma,
        &ffn_norm_gamma,
        epsilon,
    );

    assert!(
        result.is_err(),
        "T-QA-012e: transformer_layer_gpu should fail without weights"
    );
    // Extract error without unwrap_err (which requires Debug on Ok type)
    let err = match result {
        Ok(_) => panic!("T-QA-012e: Expected error but got Ok"),
        Err(e) => e,
    };
    let err_msg = format!("{:?}", err);
    assert!(
        err_msg.contains("not cached") || err_msg.contains("PAR-023"),
        "T-QA-012e: Error should mention missing cached weights: {}",
        err_msg
    );
    println!("T-QA-012e: Missing weights error handling PASSED");
}

/// T-QA-012f: Test incremental attention with KV cache update
///
/// Verifies that calling transformer_layer_gpu multiple times updates KV cache correctly.
#[test]
#[serial]
fn test_tqa012f_transformer_layer_kv_cache_update() {
    if !CudaExecutor::is_available() {
        println!("T-QA-012f: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 256u32;
    let intermediate_dim = 1024u32;
    let num_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 64usize;
    let num_layers = 1usize;
    let max_seq_len = 32usize;
    let layer_idx = 0usize;
    let layer_prefix = "blk.0";
    let epsilon = 1e-5f32;

    executor
        .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
        .expect("T-QA-012f: KV cache init");

    // Load weights
    let qkvo_weights = create_mock_q4k_weights_for_harness(256, 256);
    executor
        .load_quantized_weights(&format!("{}.attn_q.weight", layer_prefix), &qkvo_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.attn_k.weight", layer_prefix), &qkvo_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.attn_v.weight", layer_prefix), &qkvo_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(
            &format!("{}.attn_output.weight", layer_prefix),
            &qkvo_weights,
        )
        .expect("CUDA operation failed");

    let gate_up_weights = create_mock_q4k_weights_for_harness(1024, 256);
    let down_weights = create_mock_q4k_weights_for_harness(256, 1024);
    executor
        .load_quantized_weights(
            &format!("{}.ffn_gate.weight", layer_prefix),
            &gate_up_weights,
        )
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.ffn_up.weight", layer_prefix), &gate_up_weights)
        .expect("CUDA operation failed");
    executor
        .load_quantized_weights(&format!("{}.ffn_down.weight", layer_prefix), &down_weights)
        .expect("CUDA operation failed");

    let gamma = vec![1.0f32; hidden_dim as usize];
    let attn_norm_gamma =
        GpuBuffer::from_host(&executor.context, &gamma).expect("CUDA operation failed");
    let ffn_norm_gamma =
        GpuBuffer::from_host(&executor.context, &gamma).expect("CUDA operation failed");

    // Execute twice to verify KV cache updates
    for token_idx in 0..2 {
        let input_data = vec![0.1f32 * (token_idx as f32 + 1.0); hidden_dim as usize];
        let input =
            GpuBuffer::from_host(&executor.context, &input_data).expect("CUDA operation failed");

        let result = executor.transformer_layer_gpu(
            &input,
            layer_idx,
            layer_prefix,
            hidden_dim,
            intermediate_dim,
            &attn_norm_gamma,
            &ffn_norm_gamma,
            epsilon,
        );

        assert!(
            result.is_ok(),
            "T-QA-012f: Token {} should process successfully: {:?}",
            token_idx,
            result.err()
        );
    }

    // Verify KV cache length increased
    let cache_len = executor
        .kv_cache_lengths
        .get(&layer_idx)
        .copied()
        .unwrap_or(0);
    assert_eq!(
        cache_len, 2,
        "T-QA-012f: KV cache should have 2 entries after 2 tokens"
    );
    println!("T-QA-012f: KV cache update across multiple tokens PASSED");
}

// ============================================================================
// T-QA-013: Synthetic Graph Tests
// ============================================================================
// These tests exercise CUDA graph capture/replay state management.

/// T-QA-013a: Test decode graph state management
///
/// Verifies has_decode_graph and clear_decode_graph work correctly.
#[test]
#[serial]
fn test_tqa013a_decode_graph_state() {
    if !CudaExecutor::is_available() {
        println!("T-QA-013a: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially no graph captured
    assert!(
        !executor.has_decode_graph(),
        "T-QA-013a: No decode graph initially"
    );

    // Clear graph (should be no-op on empty state)
    executor.clear_decode_graph();
    assert!(
        !executor.has_decode_graph(),
        "T-QA-013a: Still no graph after clear"
    );

    println!("T-QA-013a: Decode graph state management PASSED");
}

/// T-QA-013b: Test workspace and indexed weight checks
///
/// Verifies the workspace and indexed weight state checks used by graph capture.
#[test]
#[serial]
fn test_tqa013b_workspace_and_indexed_weights() {
    if !CudaExecutor::is_available() {
        println!("T-QA-013b: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially no workspace
    assert!(
        !executor.has_workspace(),
        "T-QA-013b: No workspace initially"
    );

    // Initially no indexed weights
    assert!(
        !executor.has_indexed_weights(),
        "T-QA-013b: No indexed weights initially"
    );

    // Clear indexed weights (should be no-op)
    executor.clear_indexed_weights();
    assert!(
        !executor.has_indexed_weights(),
        "T-QA-013b: Still no indexed weights after clear"
    );

    println!("T-QA-013b: Workspace and indexed weights checks PASSED");
}

/// T-QA-013c: Test CUDA graph opt-in behavior (CUDA_GRAPH_ENABLE)
///
/// CONTRACT: cuda-graph-safety-v1 FALSIFY-CGS-003/004
/// Verifies that graph capture is opt-in only (CUDA_GRAPH_ENABLE=1).
/// Default (no env var) must NOT attempt graph capture.
#[test]
#[serial]
fn test_tqa013c_graph_opt_in_only() {
    if !CudaExecutor::is_available() {
        println!("T-QA-013c: CUDA not available, skipping");
        return;
    }

    // Without CUDA_GRAPH_ENABLE, graphs should never be captured
    std::env::remove_var("CUDA_GRAPH_ENABLE");

    let executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        !executor.has_decode_graph(),
        "T-QA-013c: No graph should be captured without CUDA_GRAPH_ENABLE=1"
    );

    println!("T-QA-013c: CUDA graph opt-in enforcement PASSED (cuda-graph-safety-v1)");
}

/// T-QA-013d: Test graphed forward with incomplete state (falls back to non-graphed)
///
/// Verifies that forward_all_layers_gpu_to_logits_graphed gracefully falls back
/// when workspace/indexed weights are not available.
#[test]
#[serial]
fn test_tqa013d_graphed_forward_fallback() {
    if !CudaExecutor::is_available() {
        println!("T-QA-013d: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Model dimensions
    let hidden_dim = 256u32;
    let intermediate_dim = 1024u32;
    let num_layers = 1usize;
    let vocab_size = 1000u32;
    let epsilon = 1e-5f32;

    // Input/output (without setting up weights - will fail at forward pass)
    let input = vec![0.1f32; hidden_dim as usize];
    let mut logits = vec![0.0f32; vocab_size as usize];

    // Try graphed forward without weights - should fail with missing weights error
    let result = executor.forward_all_layers_gpu_to_logits_graphed(
        &input,
        &mut logits,
        0,
        num_layers,
        hidden_dim,
        intermediate_dim,
        vocab_size,
        epsilon,
    );

    // Expect error due to missing weights (not a graph capture error)
    assert!(result.is_err(), "T-QA-013d: Should fail without weights");
    let err_msg = format!("{:?}", result.expect_err("CUDA operation failed"));
    // Error should mention missing cached weights or norms, not graph capture failure
    assert!(
        err_msg.contains("not cached")
            || err_msg.contains("PAR-023")
            || err_msg.contains("Workspace"),
        "T-QA-013d: Error should be about missing state, not graph: {}",
        err_msg
    );

    // No graph should be captured on failure
    assert!(
        !executor.has_decode_graph(),
        "T-QA-013d: No graph captured on failure"
    );

    println!("T-QA-013d: Graphed forward fallback on incomplete state PASSED");
}
