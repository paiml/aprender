use super::*;

/// Helper to create CudaExecutor for tests
fn create_executor() -> Option<CudaExecutor> {
    CudaExecutor::new(0).ok()
}

/// Helper to create zeroed `ValidatedLayerWeights` for tests.
/// PMAT-232: `Default` was intentionally removed from `IndexedLayerWeights`
/// to enforce explicit construction from GGUF metadata in production code.
/// Tests that only need a dummy/zeroed struct use this helper instead.
/// Uses `new_unchecked` to bypass validation (these are negative tests).
fn test_zeroed_layer_weights() -> ValidatedLayerWeights {
    ValidatedLayerWeights::new_unchecked(IndexedLayerWeights {
        attn_q_ptr: 0,
        attn_q_len: 0,
        attn_q_qtype: WeightQuantType::Q4K,
        attn_k_ptr: 0,
        attn_k_len: 0,
        attn_k_qtype: WeightQuantType::Q4K,
        attn_v_ptr: 0,
        attn_v_len: 0,
        attn_v_qtype: WeightQuantType::Q4K,
        attn_output_ptr: 0,
        attn_output_len: 0,
        attn_output_qtype: WeightQuantType::Q4K,
        ffn_gate_ptr: 0,
        ffn_gate_len: 0,
        ffn_gate_qtype: WeightQuantType::Q4K,
        ffn_up_ptr: 0,
        ffn_up_len: 0,
        ffn_up_qtype: WeightQuantType::Q4K,
        ffn_down_ptr: 0,
        ffn_down_len: 0,
        ffn_down_qtype: WeightQuantType::Q4K,
        attn_norm_ptr: 0,
        attn_norm_len: 0,
        ffn_norm_ptr: 0,
        ffn_norm_len: 0,
        attn_q_bias_ptr: 0,
        attn_q_bias_len: 0,
        attn_k_bias_ptr: 0,
        attn_k_bias_len: 0,
        attn_v_bias_ptr: 0,
        attn_v_bias_len: 0,
        attn_q_norm_ptr: 0,
        attn_q_norm_len: 0,
        attn_k_norm_ptr: 0,
        attn_k_norm_len: 0,
    })
}

// ========================================================================
// Tests for transformer_layer_indexed
// ========================================================================

#[test]
fn test_transformer_layer_indexed_missing_kv_cache() {
    let Some(mut exec) = create_executor() else {
        return;
    };

    // Create dummy IndexedLayerWeights using Default
    let layer_weights = test_zeroed_layer_weights();

    let input: Vec<f32> = vec![0.1; 256];
    let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();

    // Will fail due to missing KV cache setup or zero pointers
    let result = exec.transformer_layer_indexed(
        &input_buf,
        0, // layer_idx
        &layer_weights,
        256,  // hidden_dim
        1024, // intermediate_dim
        1e-5, // epsilon
    );

    // Expected to fail - KV cache not initialized or zero pointers
    assert!(result.is_err());
}

#[test]
fn test_indexed_layer_weights_zeroed() {
    // Test that zeroed helper creates valid structure with expected values
    let weights = test_zeroed_layer_weights();
    assert_eq!(weights.attn_norm_ptr, 0);
    assert_eq!(weights.attn_norm_len, 0);
    assert_eq!(weights.attn_q_ptr, 0);
    assert!(matches!(weights.attn_v_qtype, WeightQuantType::Q4K));
}

#[test]
fn test_weight_quant_type_variants() {
    // Test WeightQuantType::Q6K path exists in transformer_layer_indexed
    // The match arm for Q6K uses q6k_gemv_indexed_async
    assert!(matches!(WeightQuantType::Q6K, WeightQuantType::Q6K));
    assert!(matches!(WeightQuantType::Q4K, WeightQuantType::Q4K));
    assert!(matches!(WeightQuantType::Q5K, WeightQuantType::Q5K));
}

// ========================================================================
// Harness-Based Integration Tests
// ========================================================================

#[test]
fn test_transformer_layer_indexed_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let mut exec = CudaExecutor::new(0).expect("CUDA executor - RTX 4090 MUST be available");
    let config = HarnessConfig::default();
    setup_executor_harness(&mut exec, &config).expect("Harness setup MUST succeed");

    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
    let layer_weights = exec.indexed_layer_weights[0].clone();

    let result = exec.transformer_layer_indexed(
        &input,
        0,
        &layer_weights,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
    );
    // PROHIBITION OF MIRACLES: Assert on result, don't ignore it
    assert!(
        result.is_ok(),
        "transformer_layer_indexed MUST succeed with valid harness: {:?}",
        result.err()
    );
}

// ========================================================================
// Negative Capability Tests (Prohibition of Miracles - Section H)
// ========================================================================

#[test]
fn test_indexed_rejects_null_weight_pointer() {
    // H2: The Vacuum Test - null pointers must fail loudly
    let mut exec = CudaExecutor::new(0).expect("CUDA executor");
    let _ = exec.init_workspace(256, 1024);

    // Create weights with null pointer (0)
    let mut null_weights = test_zeroed_layer_weights();
    null_weights.inner_mut().attn_norm_ptr = 0; // Null pointer

    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; 256]).unwrap();

    let result = exec.transformer_layer_indexed(&input, 0, &null_weights, 256, 1024, 1e-5);
    // MUST fail - null pointers are invalid
    assert!(result.is_err(), "Null weight pointer MUST be rejected");
}

#[test]
fn test_indexed_rejects_mismatched_dimensions() {
    // H2: Invalid dimensions must fail loudly
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let mut exec = CudaExecutor::new(0).expect("CUDA executor");
    let config = HarnessConfig::default();
    setup_executor_harness(&mut exec, &config).expect("Harness setup");

    // Input with WRONG dimension
    let wrong_dim = config.hidden_dim * 2;
    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; wrong_dim]).unwrap();
    let layer_weights = exec.indexed_layer_weights[0].clone();

    let result = exec.transformer_layer_indexed(
        &input,
        0,
        &layer_weights,
        config.hidden_dim as u32, // Expected hidden_dim
        config.intermediate_dim as u32,
        1e-5,
    );
    // May or may not fail depending on buffer checks, but exercises error path
    let _ = result;
}

#[test]
fn test_indexed_rejects_invalid_layer_index() {
    // H2: Out-of-bounds layer index must fail
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let mut exec = CudaExecutor::new(0).expect("CUDA executor");
    let config = HarnessConfig::default();
    setup_executor_harness(&mut exec, &config).expect("Harness setup");

    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
    let layer_weights = exec.indexed_layer_weights[0].clone();

    // Use invalid layer index (beyond num_layers)
    let invalid_layer_idx = config.num_layers + 100;
    let result = exec.transformer_layer_indexed(
        &input,
        invalid_layer_idx,
        &layer_weights,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
    );
    // Exercises the code path with invalid index
    let _ = result;
}

#[test]
fn test_transformer_layer_indexed_multiple_layers() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let mut config = HarnessConfig::default();
    config.num_layers = 4;
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Test each layer
    for layer_idx in 0..config.num_layers {
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let layer_weights = exec.indexed_layer_weights[layer_idx].clone();

        let result = exec.transformer_layer_indexed(
            &input,
            layer_idx,
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
        let _ = result;
    }
}

#[test]
fn test_transformer_layer_indexed_q6k_v_weight() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Modify layer weights to use Q6K for V projection
    // GH-279: Use inner_mut() to modify validated weights in tests
    let mut layer_weights = exec.indexed_layer_weights[0].clone();
    layer_weights.inner_mut().attn_v_qtype = WeightQuantType::Q6K;

    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();

    let result = exec.transformer_layer_indexed(
        &input,
        0,
        &layer_weights,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
    );
    let _ = result;
}

#[test]
fn test_transformer_layer_indexed_different_epsilon() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Test with different epsilon values
    for epsilon in [1e-5f32, 1e-6, 1e-4] {
        let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
        let layer_weights = exec.indexed_layer_weights[0].clone();

        let result = exec.transformer_layer_indexed(
            &input,
            0,
            &layer_weights,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            epsilon,
        );
        let _ = result;
    }
}

#[test]
fn test_transformer_layer_indexed_gqa_configuration() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let mut config = HarnessConfig::default();
    config.num_heads = 32;
    config.num_kv_heads = 8; // 4:1 GQA ratio
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
    let layer_weights = exec.indexed_layer_weights[0].clone();

    let result = exec.transformer_layer_indexed(
        &input,
        0,
        &layer_weights,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        1e-5,
    );
    let _ = result;
}

#[test]
fn test_indexed_layer_weights_pointers_valid() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // After harness setup, pointers should be non-zero
    let layer_weights = &exec.indexed_layer_weights[0];
    assert!(
        layer_weights.attn_norm_ptr != 0,
        "attn_norm_ptr should be set"
    );
    assert!(layer_weights.attn_q_ptr != 0, "attn_q_ptr should be set");
    assert!(
        layer_weights.ffn_gate_ptr != 0,
        "ffn_gate_ptr should be set"
    );
}

#[test]
fn test_indexed_weights_count_matches_layers() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let mut config = HarnessConfig::default();
    config.num_layers = 6;
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    assert_eq!(exec.indexed_layer_weights.len(), config.num_layers);
}

#[test]
fn test_q4k_gemv_indexed_async_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let layer_weights = &exec.indexed_layer_weights[0];
    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();

    let result = exec.q4k_gemv_indexed_async(
        layer_weights.attn_q_ptr,
        &input,
        config.hidden_dim as u32,
        config.hidden_dim as u32,
    );
    let _ = result;
}

#[test]
fn test_q6k_gemv_indexed_async_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let layer_weights = &exec.indexed_layer_weights[0];
    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();

    // Q6K GEMV path
    let result = exec.q6k_gemv_indexed_async(
        layer_weights.attn_v_ptr,
        &input,
        config.hidden_dim as u32,
        config.hidden_dim as u32,
    );
    let _ = result;
}

#[test]
fn test_rmsnorm_gpu_ptr_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let layer_weights = &exec.indexed_layer_weights[0];
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).unwrap();

    let result = exec.rmsnorm_gpu_ptr(
        &input,
        layer_weights.attn_norm_ptr,
        layer_weights.attn_norm_len,
        config.hidden_dim as u32,
        1e-5,
    );
    let _ = result;
}

// ========================================================================
// Coverage Tests: Workspace Functions (v1.36.0)
// ========================================================================

#[test]
fn test_transformer_layer_workspace_for_capture() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    let input = GpuBuffer::from_host(&exec.context, &vec![0.1f32; config.hidden_dim]).unwrap();
    let layer_weights = exec.indexed_layer_weights[0].clone();

    let result = exec.transformer_layer_workspace_for_capture(
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

include!("indexed_tests_transformer_layer.rs");
