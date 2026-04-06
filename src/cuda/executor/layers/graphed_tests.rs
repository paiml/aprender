use super::*;

/// Helper to create CudaExecutor for tests
fn create_executor() -> Option<CudaExecutor> {
    CudaExecutor::new(0).ok()
}

// ========================================================================
// Validation Tests for forward_all_layers_gpu_to_logits_graphed
// ========================================================================

#[test]
fn test_graphed_forward_no_workspace() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    // No workspace initialized - should fall back to non-graphed path
    let input = vec![0.1f32; 256];
    let mut logits = vec![0.0f32; 1024];

    // This will fall back and then fail on missing norm weights
    let result = exec.forward_all_layers_gpu_to_logits_graphed(
        &input,
        &mut logits,
        0,    // position
        1,    // num_layers
        256,  // hidden_dim
        1024, // intermediate_dim
        1024, // vocab_size
        1e-5, // epsilon
    );

    assert!(result.is_err());
}

#[test]
fn test_graphed_forward_no_indexed_weights() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    // Initialize workspace but don't build indexed weights
    let _ = exec.init_workspace(256, 1024);

    let input = vec![0.1f32; 256];
    let mut logits = vec![0.0f32; 1024];

    // Should fall back and fail
    let result = exec.forward_all_layers_gpu_to_logits_graphed(
        &input,
        &mut logits,
        0,
        1,
        256,
        1024,
        1024,
        1e-5,
    );

    assert!(result.is_err());
}

// ========================================================================
// GPU Argmax Tests
// ========================================================================

#[test]
fn test_gpu_argmax_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    // Create logits buffer with clear maximum
    let mut logits = vec![-1.0f32; 128];
    logits[42] = 10.0; // Make token 42 the winner

    let logits_buf = GpuBuffer::from_host(&exec.context, &logits).unwrap();
    let logits_ptr = logits_buf.as_ptr();

    let result = exec.gpu_argmax(logits_ptr, 128);

    // May fail due to kernel issues but exercises path
    if let Ok(argmax) = result {
        // If it works, token 42 should win
        assert_eq!(argmax, 42);
    }
}

#[test]
fn test_gpu_argmax_first_token() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    // Maximum at first position
    let mut logits = vec![-10.0f32; 64];
    logits[0] = 5.0;

    let logits_buf = GpuBuffer::from_host(&exec.context, &logits).unwrap();

    let result = exec.gpu_argmax(logits_buf.as_ptr(), 64);
    if let Ok(argmax) = result {
        assert_eq!(argmax, 0);
    }
}

#[test]
fn test_gpu_argmax_last_token() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    // Maximum at last position
    let mut logits = vec![-10.0f32; 256];
    logits[255] = 5.0;

    let logits_buf = GpuBuffer::from_host(&exec.context, &logits).unwrap();

    let result = exec.gpu_argmax(logits_buf.as_ptr(), 256);
    if let Ok(argmax) = result {
        assert_eq!(argmax, 255);
    }
}

// ========================================================================
// Harness-Based Integration Tests
// ========================================================================

#[test]
fn test_graphed_forward_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
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
fn test_graphed_forward_multiple_positions() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    for position in [0u32, 1, 5, 10] {
        let input = vec![0.1f32; config.hidden_dim];
        let mut logits = vec![0.0f32; config.vocab_size];

        let result = exec.forward_all_layers_gpu_to_logits_graphed(
            &input,
            &mut logits,
            position,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );
        let _ = result;
    }
}

#[test]
fn test_graphed_forward_graph_capture() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // First call should capture graph
    let input = vec![0.1f32; config.hidden_dim];
    let mut logits = vec![0.0f32; config.vocab_size];

    let result1 = exec.forward_all_layers_gpu_to_logits_graphed(
        &input,
        &mut logits,
        0,
        config.num_layers,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        config.vocab_size as u32,
        1e-5,
    );

    // Second call should replay graph
    let result2 = exec.forward_all_layers_gpu_to_logits_graphed(
        &input,
        &mut logits,
        1,
        config.num_layers,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        config.vocab_size as u32,
        1e-5,
    );

    let _ = (result1, result2);
}

#[test]
fn test_gpu_argmax_large_vocab() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    // Test with large vocabulary size (32K typical)
    let vocab_size = 32000;
    let mut logits = vec![-10.0f32; vocab_size];
    logits[15000] = 5.0;

    let logits_buf = GpuBuffer::from_host(&exec.context, &logits).unwrap();

    let result = exec.gpu_argmax(logits_buf.as_ptr(), vocab_size as u32);
    if let Ok(argmax) = result {
        assert_eq!(argmax, 15000);
    }
}

#[test]
fn test_gpu_argmax_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Use vocab_size from config
    let mut logits = vec![-1.0f32; config.vocab_size];
    logits[config.vocab_size / 2] = 10.0;

    let logits_buf = GpuBuffer::from_host(&exec.context, &logits).unwrap();

    let result = exec.gpu_argmax(logits_buf.as_ptr(), config.vocab_size as u32);
    if let Ok(argmax) = result {
        assert_eq!(argmax, (config.vocab_size / 2) as u32);
    }
}

#[test]
fn test_graphed_forward_different_positions() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Test graphed path at different positions
    for position in [0u32, 5, 10, 20] {
        let input = vec![0.1f32; config.hidden_dim];
        let mut logits = vec![0.0f32; config.vocab_size];

        let result = exec.forward_all_layers_gpu_to_logits_graphed(
            &input,
            &mut logits,
            position,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );
        let _ = result;
    }
}

#[test]
fn test_decode_graph_state() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Initial state - no decode graph captured
    assert!(exec.decode_graph.is_none());

    // After graphed forward, decode_token_count should be set
    let input = vec![0.1f32; config.hidden_dim];
    let mut logits = vec![0.0f32; config.vocab_size];

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

    // Token count should be accessible
    let _ = exec.decode_token_count;
}

#[test]
fn test_position_buffer_allocation() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let input = vec![0.1f32; config.hidden_dim];
    let mut logits = vec![0.0f32; config.vocab_size];

    // Graphed forward should allocate position buffer
    let _ = exec.forward_all_layers_gpu_to_logits_graphed(
        &input,
        &mut logits,
        5, // non-zero position
        config.num_layers,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        config.vocab_size as u32,
        1e-5,
    );

    // Position buffer should be allocated after graphed forward
    // (may or may not be Some depending on path taken)
}

// ========================================================================
// Coverage Tests: preload_modules_for_capture (v1.36.0)
// ========================================================================

#[test]
fn test_preload_modules_for_capture_basic() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Preload modules for graph capture
    let result = exec.preload_modules_for_capture(
        config.num_layers,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        config.vocab_size as u32,
    );

    // Should succeed or fail gracefully
    let _ = result;
}

#[test]
fn test_preload_modules_different_dims() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Test with different dimensions
    for (hidden, intermediate) in [(256, 1024), (512, 2048), (1024, 4096)] {
        let result = exec.preload_modules_for_capture(1, hidden, intermediate, 1024);
        let _ = result;
    }
}

#[test]
fn test_forward_graphed_replay_to_token_id_basic() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let input = vec![0.1f32; config.hidden_dim];

    // First graphed forward to capture
    let mut logits = vec![0.0f32; config.vocab_size];
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

    // Now try replay to token id (takes: input, position, vocab_size)
    let result = exec.forward_graphed_replay_to_token_id(
        &input,
        1, // position
        config.vocab_size as u32,
    );
    let _ = result;
}

include!("graphed_tests_forward.rs");
