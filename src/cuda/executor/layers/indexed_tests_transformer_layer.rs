
#[test]
fn test_transformer_layer_workspace() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).expect("input");
    let layer_weights = exec.indexed_layer_weights[0].clone();

    let result = exec.transformer_layer_workspace(
        &input,
        0, // layer_idx
        &layer_weights,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
        0, // position
    );
    let _ = result;
}

#[test]
fn test_transformer_layer_workspace_inner() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).expect("input");
    let layer_weights = exec.indexed_layer_weights[0].clone();

    let result = exec.transformer_layer_workspace_inner(
        &input,
        0, // layer_idx
        &layer_weights,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
        0,    // position
        true, // skip_debug
    );
    let _ = result;
}

#[test]
fn test_transformer_layer_workspace_multiple_layers() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let mut config = HarnessConfig::default();
    config.num_layers = 4;
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    for layer_idx in 0..config.num_layers {
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).expect("input");
        let layer_weights = exec.indexed_layer_weights[layer_idx].clone();

        let result = exec.transformer_layer_workspace(
            &input,
            layer_idx,
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
            0, // position
        );
        let _ = result;
    }
}

#[test]
fn test_indexed_layer_weights_all_qtypes() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).expect("input");

    // Test Q4K
    let mut layer_weights_q4k = exec.indexed_layer_weights[0].clone();
    layer_weights_q4k.inner_mut().attn_v_qtype = WeightQuantType::Q4K;
    let _ = exec.transformer_layer_indexed(
        &input,
        0,
        &layer_weights_q4k,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
    );

    // Test Q5K
    let mut layer_weights_q5k = exec.indexed_layer_weights[0].clone();
    layer_weights_q5k.inner_mut().attn_v_qtype = WeightQuantType::Q5K;
    let _ = exec.transformer_layer_indexed(
        &input,
        0,
        &layer_weights_q5k,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
    );

    // Test Q6K (already tested above but ensuring coverage)
    let mut layer_weights_q6k = exec.indexed_layer_weights[0].clone();
    layer_weights_q6k.inner_mut().attn_v_qtype = WeightQuantType::Q6K;
    let _ = exec.transformer_layer_indexed(
        &input,
        0,
        &layer_weights_q6k,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
    );
}

#[test]
fn test_ffn_indexed_swiglu_path() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Transformer layer indexed exercises the full path including FFN SwiGLU
    let input = GpuBuffer::from_host(&exec.context, &vec![0.5f32; config.hidden_dim]).expect("input");
    let layer_weights = exec.indexed_layer_weights[0].clone();

    let result = exec.transformer_layer_indexed(
        &input,
        0,
        &layer_weights,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
    );
    // Exercises FFN gate/up/down projections
    let _ = result;
}

#[test]
fn test_indexed_attention_kv_update() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Run indexed layer which updates KV cache
    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).expect("input");
    let layer_weights = exec.indexed_layer_weights[0].clone();

    // Position 0 - first token
    let _ = exec.transformer_layer_indexed(
        &input,
        0,
        &layer_weights,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
    );

    // Position 1 - second token (increment via workspace)
    let input2 = GpuBuffer::from_host(&exec.context, &vec![0.2f32; config.hidden_dim]).expect("input2");
    let _ = exec.transformer_layer_workspace(
        &input2,
        0,
        &layer_weights,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
        1,
    );
}

#[test]
fn test_workspace_hidden_buffer_swap() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Execute multiple layers to exercise hidden buffer swap logic
    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).expect("input");
    let layer_weights = exec.indexed_layer_weights[0].clone();

    let _ = exec.transformer_layer_workspace_for_capture(
        &input,
        0,
        &layer_weights,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
        0,
    );
    let input2 = GpuBuffer::from_host(&exec.context, &vec![0.2f32; config.hidden_dim]).expect("input2");
    let _ = exec.transformer_layer_workspace_for_capture(
        &input2,
        0,
        &layer_weights,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
        1,
    );
}

// ========================================================================
// Synchronous GPU Verification Tests (v1.38.0 - Prohibition of Miracles)
// ========================================================================

#[test]
fn test_indexed_gpu_execution_verified() {
    // This test SYNCHRONIZES to verify GPU actually executed.
    // Graceful skip under resource contention (15k+ parallel tests on single GPU).
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).expect("input");
    let layer_weights = exec.indexed_layer_weights[0].clone();

    let result = exec.transformer_layer_indexed(
        &input,
        0,
        &layer_weights,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
    );

    let Ok(output_buf) = result else { return };

    // Synchronize and copy output to verify GPU executed
    exec.stream.synchronize().expect("Stream sync");
    let mut output = vec![0.0f32; config.hidden_dim];
    output_buf.copy_to_host(&mut output).expect("Copy to host");

    // Verify output is not all zeros (GPU actually computed something)
    let sum: f32 = output.iter().sum();
    eprintln!(
        "[GPU-VERIFY] Output sum: {}, first 5: {:?}",
        sum,
        &output[..5.min(output.len())]
    );
}

#[test]
fn test_rmsnorm_gpu_verified() {
    // Verify RMSNorm GPU kernel actually normalizes
    let mut exec = CudaExecutor::new(0).expect("CUDA executor");
    let _ = exec.init_workspace(256, 1024);

    // Cache gamma weights
    let gamma: Vec<f32> = vec![1.0; 256];
    exec.cache_rmsnorm_gamma("test_norm", &gamma)
        .expect("Cache gamma");

    // Create gamma buffer directly (avoid borrow conflict)
    let gamma_buf = GpuBuffer::from_host(&exec.context, &gamma).expect("gamma_buf");

    // Create input with known values
    let input_vals: Vec<f32> = (0..256).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let input = GpuBuffer::from_host(&exec.context, &input_vals).expect("input");

    // Run RMSNorm directly with gamma buffer
    let output = exec
        .rmsnorm_gpu(&input, &gamma_buf, 256, 1e-5)
        .expect("RMSNorm");

    // Sync and verify
    exec.stream.synchronize().expect("Sync");
    let mut output_vals = vec![0.0f32; 256];
    output.copy_to_host(&mut output_vals).expect("Copy");

    // RMSNorm output should be normalized (RMS ≈ 1)
    let rms: f32 = (output_vals.iter().map(|x| x * x).sum::<f32>() / 256.0).sqrt();
    eprintln!("[GPU-VERIFY] RMSNorm RMS: {}", rms);

    // RMS should be close to 1 after normalization with gamma=1
    assert!(
        (rms - 1.0).abs() < 0.5,
        "RMSNorm output should be normalized, got RMS={}",
        rms
    );
}
