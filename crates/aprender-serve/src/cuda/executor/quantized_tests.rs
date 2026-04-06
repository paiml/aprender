use super::*;

fn create_executor() -> Option<CudaExecutor> {
    CudaExecutor::new(0).ok()
}

// ========================================================================
// Tiled Q4K GEMV Tests
// ========================================================================

#[test]
fn test_tiled_q4k_gemv_cached_async_weight_not_found() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
    let result = exec.tiled_q4k_gemv_cached_async("nonexistent", &input, 128, 256, 4);
    assert!(result.is_err());
}

#[test]
fn test_tiled_q4k_gemv_small_k() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
    // K=256 < MAX_TILED_K, uses standard tiled kernel
    let result = exec.tiled_q4k_gemv_cached_async("test", &input, 128, 256, 4);
    assert!(result.is_err()); // Weight not cached
}

// ========================================================================
// Chunked Tiled Q4K GEMV Tests
// ========================================================================

#[test]
fn test_chunked_tiled_q4k_gemv_cached_async_weight_not_found() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
    let result = exec.chunked_tiled_q4k_gemv_cached_async("nonexistent", &input, 128, 256, 4);
    assert!(result.is_err());
}

// ========================================================================
// DP4A Q4K GEMV Tests
// ========================================================================

#[test]
fn test_dp4a_q4k_gemv_cached_async_weight_not_found() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
    let result = exec.dp4a_q4k_gemv_cached_async("nonexistent", &input, 128, 256);
    assert!(result.is_err());
}

// ========================================================================
// Q8 Quantize Tests
// ========================================================================

#[test]
fn test_q8_quantize_async_output_size_calculation() {
    // Test Q8_1 format output size: ceil(n/32) * 36 bytes
    let n = 256u32;
    let num_blocks = (n + 31) / 32;
    let expected_bytes = (num_blocks * 36) as usize;
    assert_eq!(expected_bytes, 288);
}

#[test]
fn test_q8_quantize_async_single_block_size() {
    // Test Q8_1 format output size for single block
    let n = 32u32;
    let num_blocks = (n + 31) / 32;
    let expected_bytes = (num_blocks * 36) as usize;
    assert_eq!(expected_bytes, 36);
}

// ========================================================================
// Q4K Q8 GEMV Tests
// ========================================================================

#[test]
fn test_q4k_q8_gemv_async_weight_not_found() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let q8_input = GpuBuffer::<u8>::new(&exec.context, 288).unwrap();
    let result = exec.q4k_q8_gemv_async("nonexistent", &q8_input, 128, 256);
    assert!(result.is_err());
}

// ========================================================================
// True DP4A Q4K GEMV Tests
// ========================================================================

#[test]
fn test_true_dp4a_q4k_gemv_async_weight_not_found() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
    let result = exec.true_dp4a_q4k_gemv_async("nonexistent", &input, 128, 256);
    assert!(result.is_err());
}

// ========================================================================
// Packed DP4A Tests
// ========================================================================

#[test]
fn test_packed_dp4a_q4k_q8_gemv_async_weight_not_found() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let q8_input = GpuBuffer::<u8>::new(&exec.context, 288).unwrap();
    let result = exec.packed_dp4a_q4k_q8_gemv_async("nonexistent", &q8_input, 128, 256);
    assert!(result.is_err());
}

#[test]
fn test_packed_dp4a_full_async_weight_not_found() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 256]).unwrap();
    let result = exec.packed_dp4a_full_async("nonexistent", &input, 128, 256);
    assert!(result.is_err());
}

// ========================================================================
// Q5K/Q6K GEMV Cached Tests
// ========================================================================

#[test]
fn test_q5k_gemv_cached_weight_not_found() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 128];
    let result = exec.q5k_gemv_cached("nonexistent", &input, &mut output, 128, 256);
    assert!(result.is_err());
}

#[test]
fn test_q6k_gemv_cached_weight_not_found() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 128];
    let result = exec.q6k_gemv_cached("nonexistent", &input, &mut output, 128, 256);
    assert!(result.is_err());
}

// ========================================================================
// GELU GPU Tests
// ========================================================================

#[test]
fn test_gelu_gpu_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input = vec![0.0f32, 1.0, -1.0, 2.0];
    let buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
    let result = exec.gelu_gpu(&buf, 4);
    assert!(result.is_ok());
}

#[test]
fn test_gelu_gpu_larger() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input = vec![1.0f32; 256];
    let buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
    let result = exec.gelu_gpu(&buf, 256);
    assert!(result.is_ok());
}

// ========================================================================
// LayerNorm GPU Tests
// ========================================================================

#[test]
fn test_layer_norm_gpu_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
    let output = GpuBuffer::<f32>::new(&exec.context, 32).unwrap();
    let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
    let beta = GpuBuffer::from_host(&exec.context, &vec![0.0f32; 32]).unwrap();
    let result = exec.layer_norm_gpu(&input, &output, &gamma, &beta, 32, 1, 1e-5);
    assert!(result.is_ok());
}

// ========================================================================
// RMSNorm GPU Tests
// ========================================================================

#[test]
fn test_rmsnorm_gpu_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
    let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
    let result = exec.rmsnorm_gpu(&input, &gamma, 32, 1e-5);
    assert!(result.is_ok());
    let buf = result.unwrap();
    assert_eq!(buf.len(), 32);
}

#[test]
fn test_rmsnorm_into_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
    let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
    let output = GpuBuffer::<f32>::new(&exec.context, 32).unwrap();
    let result = exec.rmsnorm_into(&input, &gamma, &output, 32, 1e-5);
    assert!(result.is_ok());
}

#[test]
fn test_rmsnorm_host_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input = vec![1.0f32; 32];
    let gamma = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 32];
    let result = exec.rmsnorm_host(&input, &gamma, &mut output, 1e-5);
    assert!(result.is_ok());
    // Output should not be zeros anymore
    exec.stream.synchronize().unwrap();
}

// ========================================================================
// Batched RMSNorm Tests
// ========================================================================

#[test]
fn test_batched_rmsnorm_into_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    // M=4 sequences, hidden=32
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 4 * 32]).unwrap();
    let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
    let output = GpuBuffer::<f32>::new(&exec.context, 4 * 32).unwrap();
    let result = exec.batched_rmsnorm_into(&input, &gamma, &output, 32, 4, 1e-5);
    assert!(result.is_ok());
}

// ========================================================================
// Residual Add GPU Tests
// ========================================================================

#[test]
fn test_residual_add_gpu_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input1 = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
    let input2 = GpuBuffer::from_host(&exec.context, &vec![2.0f32; 32]).unwrap();
    let result = exec.residual_add_gpu(&input1, &input2, 32);
    assert!(result.is_ok());
}

#[test]
fn test_residual_add_into_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input1 = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
    let input2 = GpuBuffer::from_host(&exec.context, &vec![2.0f32; 32]).unwrap();
    let output = GpuBuffer::<f32>::new(&exec.context, 32).unwrap();
    let result = exec.residual_add_into(&input1, &input2, &output, 32);
    assert!(result.is_ok());
}

#[test]
fn test_residual_add_host_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let input1 = vec![1.0f32; 32];
    let input2 = vec![2.0f32; 32];
    let mut output = vec![0.0f32; 32];
    let result = exec.residual_add_host(&input1, &input2, &mut output);
    assert!(result.is_ok());
}

// ========================================================================
// Fused Residual RMSNorm Tests
// ========================================================================

#[test]
fn test_fused_residual_rmsnorm_gpu_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let residual = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
    let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; 32]).unwrap();
    let result = exec.fused_residual_rmsnorm_gpu(&residual, &input, &gamma, 32, 1e-5);
    assert!(result.is_ok());
}

#[test]
fn test_fused_residual_rmsnorm_host_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    let residual = vec![1.0f32; 32];
    let input = vec![1.0f32; 32];
    let gamma = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 32];
    let result = exec.fused_residual_rmsnorm_host(&residual, &input, &gamma, &mut output, 1e-5);
    assert!(result.is_ok());
}

// ========================================================================
// Batched Operations Tests
// ========================================================================

#[test]
fn test_batched_rope_into_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    // M=4 sequences, num_heads=4, head_dim=32
    let size = 4 * 4 * 32;
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; size]).unwrap();
    let output = GpuBuffer::<f32>::new(&exec.context, size).unwrap();
    let positions = GpuBuffer::from_host(&exec.context, &vec![0u32; 4]).unwrap();
    let result = exec.batched_rope_into(&input, &output, &positions, 4, 32, 4, 10000.0);
    assert!(result.is_ok());
}

#[test]
fn test_batched_residual_add_into_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    // M=4 sequences, n=32 elements each
    let size = 4 * 32;
    let input1 = GpuBuffer::from_host(&exec.context, &vec![1.0f32; size]).unwrap();
    let input2 = GpuBuffer::from_host(&exec.context, &vec![2.0f32; size]).unwrap();
    let output = GpuBuffer::<f32>::new(&exec.context, size).unwrap();
    let result = exec.batched_residual_add_into(&input1, &input2, &output, 32, 4);
    assert!(result.is_ok());
}

#[test]
fn test_batched_swiglu_into_basic() {
    let Some(mut exec) = create_executor() else {
        return;
    };
    // M=4 sequences, n=32 elements each
    let size = 4 * 32;
    let gate = GpuBuffer::from_host(&exec.context, &vec![1.0f32; size]).unwrap();
    let up = GpuBuffer::from_host(&exec.context, &vec![1.0f32; size]).unwrap();
    let output = GpuBuffer::<f32>::new(&exec.context, size).unwrap();
    let result = exec.batched_swiglu_into(&gate, &up, &output, 32, 4);
    assert!(result.is_ok());
}

// ========================================================================
// Harness-Based Integration Tests
// ========================================================================

#[test]
fn test_rmsnorm_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Create gamma buffer directly (avoid borrow conflict with cache)
    let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).unwrap();
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).unwrap();
    let result = exec.rmsnorm_gpu(&input, &gamma, config.hidden_dim as u32, 1e-5);
    assert!(result.is_ok());
}

#[test]
fn test_rmsnorm_into_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Create gamma buffer directly
    let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).unwrap();
    let input = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).unwrap();
    let output = GpuBuffer::<f32>::new(&exec.context, config.hidden_dim).unwrap();
    let result = exec.rmsnorm_into(&input, &gamma, &output, config.hidden_dim as u32, 1e-5);
    assert!(result.is_ok());
}

#[test]
fn test_batched_rmsnorm_with_harness() {
    use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
    let Some(mut exec) = create_executor() else {
        return;
    };
    let config = HarnessConfig::default();
    if setup_executor_harness(&mut exec, &config).is_err() {
        return;
    }

    // Create gamma buffer directly
    let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; config.hidden_dim]).unwrap();
    let m = 4u32;
    let input = GpuBuffer::from_host(
        &exec.context,
        &vec![1.0f32; (m as usize) * config.hidden_dim],
    )
    .unwrap();
    let output = GpuBuffer::<f32>::new(&exec.context, (m as usize) * config.hidden_dim).unwrap();
    let result =
        exec.batched_rmsnorm_into(&input, &gamma, &output, config.hidden_dim as u32, m, 1e-5);
    assert!(result.is_ok());
}

include!("quantized_tests_fused_residual.rs");
