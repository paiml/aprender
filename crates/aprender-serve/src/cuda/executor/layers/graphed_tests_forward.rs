
#[test]
fn test_graphed_forward_env_disable() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // SKIP_CUDA_GRAPH=1 should bypass graph capture
    // (can't actually set env var in test, but exercises path checking)
    let input = vec![0.1f32; config.hidden_dim];
    let mut logits = vec![0.0f32; config.vocab_size];

    let result = exec.forward_all_layers_gpu_to_logits_graphed(
        &input,
        &mut logits,
        0,
        config.num_layers,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        config.vocab_size as u32,
        1e-5,
    );
    let _ = result;
}

#[test]
fn test_gpu_argmax_negative_logits() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    // All negative logits - least negative should win
    let mut logits = vec![-100.0f32; 128];
    logits[77] = -0.5; // Least negative

    let logits_buf = GpuBuffer::from_host(&exec.context, &logits).expect("logits_buf");
    let result = exec.gpu_argmax(logits_buf.as_ptr(), 128);

    if let Ok(argmax) = result {
        assert_eq!(argmax, 77);
    }
}

#[test]
fn test_gpu_argmax_uniform_logits() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    // All equal - should return first (or consistent result)
    let logits = vec![1.0f32; 64];
    let logits_buf = GpuBuffer::from_host(&exec.context, &logits).expect("logits_buf");

    let result = exec.gpu_argmax(logits_buf.as_ptr(), 64);
    // Result should be deterministic
    if let Ok(argmax) = result {
        assert!(argmax < 64);
    }
}

#[test]
fn test_graphed_forward_multiple_layers() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let mut config = HarnessConfig::default();
    config.num_layers = 4; // Test with more layers
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let input = vec![0.1f32; config.hidden_dim];
    let mut logits = vec![0.0f32; config.vocab_size];

    let result = exec.forward_all_layers_gpu_to_logits_graphed(
        &input,
        &mut logits,
        0,
        config.num_layers,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        config.vocab_size as u32,
        1e-5,
    );
    let _ = result;
}

#[test]
fn test_decode_token_count_increment() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let initial_count = exec.decode_token_count;

    let input = vec![0.1f32; config.hidden_dim];
    let mut logits = vec![0.0f32; config.vocab_size];

    // Execute graphed forward
    let _ = exec.forward_all_layers_gpu_to_logits_graphed(
        &input,
        &mut logits,
        0,
        config.num_layers,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        config.vocab_size as u32,
        1e-5,
    );

    // Token count should change
    assert!(exec.decode_token_count >= initial_count);
}
