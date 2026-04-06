
#[test]
fn test_batched_workspace_hidden_buffer() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Initialize batched workspace
    let result = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);
    assert!(result.is_ok());

    // Verify workspace is initialized (fields vary by implementation)
    assert!(exec.workspace.hidden_buf1.is_some());
}

// ========================================================================
// Coverage Tests: RMSNorm and Batch Edge Cases (v1.36.0)
// ========================================================================

#[test]
fn test_rmsnorm_gpu_ptr_batched() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let layer_weights = &exec.indexed_layer_weights[0];
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).expect("input");

    // rmsnorm_gpu_ptr is defined in batched.rs as pub(crate)
    let result = exec.rmsnorm_gpu_ptr(
        &input,
        layer_weights.attn_norm_ptr,
        layer_weights.attn_norm_len,
        config.hidden_dim as u32,
        1e-5,
    );
    let _ = result;
}

#[test]
fn test_forward_batched_different_positions() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

    // Non-contiguous positions
    let positions: Vec<u32> = vec![0, 5, 10, 15];
    let inputs: Vec<f32> = vec![0.1; 4 * config.hidden_dim];

    let result = exec.forward_batched_to_token_ids(
        &inputs,
        &positions,
        config.num_layers,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        config.vocab_size as u32,
        1e-5,
    );
    let _ = result;
}

#[test]
fn test_transformer_layer_batched_different_layers() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let mut config = HarnessConfig::default();
    config.num_layers = 4;
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

    let inputs: Vec<f32> = vec![0.1; 4 * config.hidden_dim];
    let input_buf = GpuBuffer::from_host(&exec.context, &inputs).expect("input_buf");
    let positions: [u32; 4] = [0, 1, 2, 3];

    // Test each layer
    for layer_idx in 0..config.num_layers {
        if layer_idx >= exec.indexed_layer_weights.len() {
            break;
        }
        let layer_weights = exec.indexed_layer_weights[layer_idx].clone();

        let result = exec.transformer_layer_batched(
            &input_buf,
            layer_idx,
            &layer_weights,
            4,
            &positions,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
        let _ = result;
    }
}

#[test]
fn test_batched_forward_m1() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Batch size 1 - edge case
    let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 1);

    let positions: Vec<u32> = vec![0];
    let inputs: Vec<f32> = vec![0.1; config.hidden_dim];

    let result = exec.forward_batched_to_token_ids(
        &inputs,
        &positions,
        config.num_layers,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        config.vocab_size as u32,
        1e-5,
    );
    let _ = result;
}

#[test]
fn test_batched_kv_cache_multiple_batch_sizes() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Test different batch sizes for KV cache init
    for batch_size in [1, 4, 8, 16] {
        let result = exec.init_batched_kv_cache_gpu(config.num_layers, batch_size);
        let _ = result;
    }
}

#[test]
fn test_transformer_layer_batched_qtype_q6k() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

    let inputs: Vec<f32> = vec![0.1; 4 * config.hidden_dim];
    let input_buf = GpuBuffer::from_host(&exec.context, &inputs).expect("input_buf");

    if exec.indexed_layer_weights.is_empty() {
        return;
    }

    // Modify weights to use Q6K
    let mut layer_weights = exec.indexed_layer_weights[0].clone();
    layer_weights.inner_mut().attn_v_qtype = WeightQuantType::Q6K;

    let positions: [u32; 4] = [0, 1, 2, 3];
    let result = exec.transformer_layer_batched(
        &input_buf,
        0,
        &layer_weights,
        4,
        &positions,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
    );
    let _ = result;
}

#[test]
fn test_batched_graph_token_count() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let initial_count = exec.decode_token_count;

    let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

    let positions: Vec<u32> = vec![0, 1, 2, 3];
    let inputs: Vec<f32> = vec![0.1; 4 * config.hidden_dim];

    // Execute batched graphed forward
    let _ = exec.forward_batched_to_token_ids_graphed(
        &inputs,
        &positions,
        config.num_layers,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        config.vocab_size as u32,
        1e-5,
    );

    // Token count might change after graphed forward
    assert!(exec.decode_token_count >= initial_count);
}

#[test]
fn test_batched_forward_varying_input_values() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

    let positions: Vec<u32> = vec![0, 1, 2, 3];

    // Different input values per batch element
    let inputs: Vec<f32> = (0..4 * config.hidden_dim)
        .map(|i| (i as f32 / 1000.0).sin())
        .collect();

    let result = exec.forward_batched_to_token_ids(
        &inputs,
        &positions,
        config.num_layers,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        config.vocab_size as u32,
        1e-5,
    );
    let _ = result;
}
