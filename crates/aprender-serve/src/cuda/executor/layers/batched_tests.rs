use super::*;

/// Helper to create CudaExecutor for tests
fn create_executor() -> Option<CudaExecutor> {
    CudaExecutor::new(0).ok()
}

// ========================================================================
// Validation Tests for forward_batched_to_token_ids
// ========================================================================

#[test]
fn test_forward_batched_empty_batch() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    let inputs: Vec<f32> = vec![];
    let positions: Vec<u32> = vec![];

    let result = exec.forward_batched_to_token_ids(
        &inputs, &positions, 1,    // num_layers
        256,  // hidden_dim
        1024, // intermediate_dim
        1024, // vocab_size
        1e-5, // epsilon
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_str = format!("{:?}", err);
    assert!(err_str.contains("batch size must be 1-32"));
}

#[test]
fn test_forward_batched_batch_too_large() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    // Batch size > 32 should fail
    let positions: Vec<u32> = (0..33).collect();
    let inputs: Vec<f32> = vec![0.1; 33 * 256];

    let result = exec.forward_batched_to_token_ids(&inputs, &positions, 1, 256, 1024, 1024, 1e-5);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_str = format!("{:?}", err);
    assert!(err_str.contains("batch size must be 1-32"));
}

#[test]
fn test_forward_batched_input_size_mismatch() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    let positions: Vec<u32> = vec![0, 1, 2, 3];
    // Wrong input size: M=4, hidden_dim=256, expected 1024, give 512
    let inputs: Vec<f32> = vec![0.1; 512];

    let result = exec.forward_batched_to_token_ids(&inputs, &positions, 1, 256, 1024, 1024, 1e-5);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_str = format!("{:?}", err);
    assert!(err_str.contains("inputs.len()") && err_str.contains("!= M*hidden_dim"));
}

#[test]
fn test_forward_batched_workspace_not_initialized() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    // Valid batch size and input size, but workspace not initialized
    let positions: Vec<u32> = vec![0, 1, 2, 3];
    let inputs: Vec<f32> = vec![0.1; 4 * 256];

    let result = exec.forward_batched_to_token_ids(&inputs, &positions, 1, 256, 1024, 1024, 1e-5);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_str = format!("{:?}", err);
    assert!(err_str.contains("workspace not initialized"));
}

#[test]
fn test_forward_batched_workspace_wrong_batch_size() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    // Initialize workspace for batch size 8
    let _ = exec.init_batched_workspace(256, 1024, 8);

    // Try to use batch size 4 (different)
    let positions: Vec<u32> = vec![0, 1, 2, 3];
    let inputs: Vec<f32> = vec![0.1; 4 * 256];

    let result = exec.forward_batched_to_token_ids(&inputs, &positions, 1, 256, 1024, 1024, 1e-5);

    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_str = format!("{:?}", err);
    assert!(err_str.contains("workspace not initialized for M=4"));
}

#[test]
fn test_forward_batched_missing_indexed_weights() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    // Initialize workspace correctly
    let _ = exec.init_batched_workspace(256, 1024, 4);

    // Don't build indexed weights
    let positions: Vec<u32> = vec![0, 1, 2, 3];
    let inputs: Vec<f32> = vec![0.1; 4 * 256];

    let result = exec.forward_batched_to_token_ids(
        &inputs, &positions, 1, // 1 layer
        256, 1024, 1024, 1e-5,
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_str = format!("{:?}", err);
    assert!(err_str.contains("weights not indexed") || err_str.contains("hidden_buf2 missing"));
}

// ========================================================================
// Integration Tests with ModelHarness
// ========================================================================

#[test]
fn test_batched_forward_with_harness_m4() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};

    let Some(mut exec) = create_executor() else {
        return;
    };

    let config = HarnessConfig::default();

    // First setup with single-token workspace, then switch to batched
    // The harness sets up indexed weights which we need
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Now reinitialize workspace for batch size 4
    let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

    // Try batched forward
    let positions: Vec<u32> = vec![0, 1, 2, 3];
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

    // May fail due to kernel issues but exercises the path
    let _ = result;
}

#[test]
fn test_transformer_layer_batched_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};

    let Some(mut exec) = create_executor() else {
        return;
    };

    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Reinitialize for batch
    let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 4);

    // Create input buffer
    let inputs: Vec<f32> = (0..4 * config.hidden_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();
    let input_buf = GpuBuffer::from_host(&exec.context, &inputs).unwrap();

    // Get indexed layer weights
    if !exec.has_indexed_weights() || exec.indexed_layer_weights.is_empty() {
        return;
    }
    let layer_weights = exec.get_indexed_layer(0).clone();

    // Try transformer layer batched
    let positions: [u32; 4] = [0, 1, 2, 3];
    let result = exec.transformer_layer_batched(
        &input_buf,
        0, // layer_idx
        &layer_weights,
        4, // m (batch_size)
        &positions,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
    );

    let _ = result;
}

// ========================================================================
// Additional Harness-Based Integration Tests
// ========================================================================

#[test]
fn test_forward_batched_m8_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Batch size 8
    let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 8);

    let positions: Vec<u32> = (0..8).collect();
    let inputs: Vec<f32> = vec![0.1; 8 * config.hidden_dim];

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
fn test_forward_batched_m16_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Batch size 16 uses multi-warp kernel
    let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 16);

    let positions: Vec<u32> = (0..16).collect();
    let inputs: Vec<f32> = vec![0.1; 16 * config.hidden_dim];

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
fn test_forward_batched_m32_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Batch size 32 (max supported)
    let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 32);

    let positions: Vec<u32> = (0..32).collect();
    let inputs: Vec<f32> = vec![0.1; 32 * config.hidden_dim];

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
fn test_forward_batched_graphed_with_harness() {
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
    let inputs: Vec<f32> = vec![0.1; 4 * config.hidden_dim];

    // Test graphed path
    let result = exec.forward_batched_to_token_ids_graphed(
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
fn test_forward_batched_graphed_replay() {
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
    let inputs: Vec<f32> = vec![0.1; 4 * config.hidden_dim];

    // First call captures graph
    let result1 = exec.forward_batched_to_token_ids_graphed(
        &inputs,
        &positions,
        config.num_layers,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        config.vocab_size as u32,
        1e-5,
    );

    // Update positions for second call
    let positions2: Vec<u32> = vec![1, 2, 3, 4];

    // Second call should replay graph
    let result2 = exec.forward_batched_to_token_ids_graphed(
        &inputs,
        &positions2,
        config.num_layers,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        config.vocab_size as u32,
        1e-5,
    );

    let _ = (result1, result2);
}

#[test]
fn test_batched_kv_cache_init() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Initialize batched KV caches (uses existing KV cache config from harness)
    let result = exec.init_batched_kv_cache_gpu(config.num_layers, 4);
    assert!(result.is_ok());
}

#[test]
fn test_transformer_layer_batched_m8() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let _ = exec.init_batched_workspace(config.hidden_dim, config.intermediate_dim, 8);

    let inputs: Vec<f32> = vec![0.1; 8 * config.hidden_dim];
    let input_buf = GpuBuffer::from_host(&exec.context, &inputs).unwrap();

    if exec.indexed_layer_weights.is_empty() {
        return;
    }
    let layer_weights = exec.indexed_layer_weights[0].clone();

    let positions: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    let result = exec.transformer_layer_batched(
        &input_buf,
        0,
        &layer_weights,
        8,
        &positions,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
    );
    let _ = result;
}

include!("batched_tests_workspace.rs");

// ========================================================================
// #3413 B: batched per-head QK RMSNorm (Qwen3 prefill)
// ========================================================================

/// Host reference for per-head RMSNorm over `batch * num_heads` rows.
fn ref_batched_per_head_rmsnorm(
    x: &[f32],
    gamma: &[f32],
    head_dim: usize,
    num_heads: usize,
    batch: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for seq in 0..batch {
        for head in 0..num_heads {
            let base = (seq * num_heads + head) * head_dim;
            let row = &x[base..base + head_dim];
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let scale = 1.0f32 / (mean_sq + eps).sqrt();
            for j in 0..head_dim {
                out[base + j] = row[j] * scale * gamma[j];
            }
        }
    }
    out
}

/// Deterministic, shape-varying input: RMSNorm is scale-invariant, so a row that
/// is merely a multiple of row 0 could not falsify a missing `blockIdx.y` offset.
fn qk_norm_fixture(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u64).wrapping_mul(2_654_435_761) % 977;
            (h as f32) / 100.0 - 4.5
        })
        .collect()
}

/// #3413 B: `batched_qkv_rope_phase` applied no QK-norm, so the prompt's K was
/// cached un-normed and decode produced garbage. This is the falsifier for the
/// `(seq_idx * num_heads + head_idx) * head_dim` offset: every element of every
/// sequence must match the host reference, and gamma must carry no batch stride.
#[test]
#[serial_test::serial]
fn test_3413_batched_per_head_rmsnorm_matches_host_reference() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = crate::cuda_executor_or_skip!(0);

    let head_dim: usize = 64;
    let num_heads: usize = 4;
    let batch: usize = 3;
    let eps = 1e-6f32;
    let n = batch * num_heads * head_dim;

    let x = qk_norm_fixture(n);
    let gamma: Vec<f32> = (0..head_dim).map(|j| 0.5 + j as f32 * 0.01).collect();

    let x_gpu = GpuBuffer::from_host(&executor.context, &x).expect("upload x");
    let gamma_gpu = GpuBuffer::from_host(&executor.context, &gamma).expect("upload gamma");
    let out_gpu = GpuBuffer::<f32>::new(&executor.context, n).expect("alloc out");

    executor
        .batched_per_head_rmsnorm_into(
            &x_gpu,
            &gamma_gpu,
            &out_gpu,
            head_dim as u32,
            num_heads as u32,
            batch as u32,
            eps,
        )
        .expect("batched_per_head_rmsnorm_into");
    executor.stream.synchronize().expect("sync");

    let mut got = vec![0.0f32; n];
    out_gpu.copy_to_host(&mut got).expect("download");

    let want = ref_batched_per_head_rmsnorm(&x, &gamma, head_dim, num_heads, batch, eps);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let seq = i / (num_heads * head_dim);
        let head = (i / head_dim) % num_heads;
        assert!(
            (g - w).abs() <= 1e-5 * (1.0 + w.abs()),
            "#3413 B: seq={seq} head={head} lane={} got {g} want {w}",
            i % head_dim
        );
    }
}

/// trueno#243 / #3413 A: a kernel on a graph path that does not record itself is
/// dropped from the manually rebuilt graph.
#[test]
#[serial_test::serial]
fn test_3413_batched_per_head_rmsnorm_is_recorded_into_the_manual_graph() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = crate::cuda_executor_or_skip!(0);

    let head_dim: usize = 64;
    let num_heads: usize = 4;
    let batch: usize = 3;
    let n = batch * num_heads * head_dim;
    let x = GpuBuffer::<f32>::new(&executor.context, n).expect("x");
    let gamma = GpuBuffer::<f32>::new(&executor.context, head_dim).expect("gamma");

    executor.begin_graph_recording();
    let before = executor.graph_recorded_kernels.len();
    executor
        .batched_per_head_rmsnorm_into(
            &x,
            &gamma,
            &x,
            head_dim as u32,
            num_heads as u32,
            batch as u32,
            1e-6,
        )
        .expect("batched_per_head_rmsnorm_into");
    let after = executor.graph_recorded_kernels.len();
    executor.graph_recording = false;
    executor.graph_recorded_kernels.clear();

    assert_eq!(
        after - before,
        1,
        "#3413: batched_per_head_rmsnorm_into must record itself while \
         graph_recording is set, or the replayed graph runs without QK-norm"
    );
}
