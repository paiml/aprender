//! CudaExecutor tests Part 06 - COV-026 through COV-031
//!
//! Coverage tests for:
//! - COV-026: Coalesced/Vectorized/DP4A GEMV variants
//! - COV-027: Tiled/Fused GEMV and async quantization
//! - COV-028: More function coverage (fused_qkv, rope, batched gemv, layer_norm)
//! - COV-029: Weight and workspace tests
//! - COV-030: CudaExecutor Layer API coverage
//! - COV-031: Additional Activation & Attention coverage

use super::*;
use crate::cuda::types::{IndexedLayerWeights, ValidatedLayerWeights, WeightQuantType};
use serial_test::serial;

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

// =============================================================================
// COV-026: Coalesced/Vectorized/DP4A GEMV variants coverage
// =============================================================================

#[test]
#[serial]
fn test_cov026_coalesced_q4k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q4_K: 144 bytes per 256 values (super-block)
    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Load quantized weights to get a GPU pointer
    let weight_bytes = (n as usize) * 144; // n rows of Q4_K data
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_coalesced", &weights)
        .expect("load weights");

    // Get weight pointer
    let weight_ptr = executor
        .get_quantized_weight_ptr("test_coalesced")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.coalesced_q4k_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "coalesced_q4k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_vectorized_q4k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Load quantized weights
    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_vectorized", &weights)
        .expect("load weights");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_vectorized")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.vectorized_q4k_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "vectorized_q4k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_dp4a_q4k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Load quantized weights
    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_dp4a", &weights)
        .expect("load weights");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_dp4a")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.dp4a_q4k_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "dp4a_q4k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_coalesced_q6k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q6_K: 210 bytes per 256 values
    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Load quantized weights
    let weight_bytes = (n as usize) * 210;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_coalesced_q6k", &weights, 14)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_coalesced_q6k")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.coalesced_q6k_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "coalesced_q6k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_q4k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Load quantized weights
    let weight_bytes = (n as usize) * 144;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights("test_q4k_into", &weights)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q4k_into")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q4k_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q4k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_q6k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Load quantized weights
    let weight_bytes = (n as usize) * 210;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_q6k_into", &weights, 14)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q6k_into")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q6k_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q6k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_q5k_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q5_K: 176 bytes per 256 values
    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 256)

    // Load quantized weights
    let weight_bytes = (n as usize) * 176;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_q5k_into", &weights, 13)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q5k_into")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q5k_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q5k_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_q8_0_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q8_0: 34 bytes per 32 values (2 bytes scale + 32 int8)
    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 32)

    // Load quantized weights
    // k=256 means 8 blocks of 32 values = 8 * 34 = 272 bytes per row
    let weight_bytes = (n as usize) * (k as usize / 32) * 34;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_q8_0_into", &weights, 8)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q8_0_into")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q8_0_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q8_0_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
#[ignore = "PTX compilation issue CUDA_ERROR_INVALID_PTX - needs kernel fix"]
fn test_cov026_q4_0_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q4_0: 18 bytes per 32 values (2 bytes scale + 16 bytes of 4-bit)
    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 32)

    // Load quantized weights
    let weight_bytes = (n as usize) * (k as usize / 32) * 18;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_q4_0_into", &weights, 2)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q4_0_into")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q4_0_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q4_0_gemv_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov026_q4_1_gemv_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Q4_1: 20 bytes per 32 values (2 scale + 2 min + 16 data)
    let n = 32u32; // output dim
    let k = 256u32; // input dim (must be divisible by 32)

    // Load quantized weights
    let weight_bytes = (n as usize) * (k as usize / 32) * 20;
    let weights = vec![0u8; weight_bytes];
    executor
        .load_quantized_weights_with_type("test_q4_1_into", &weights, 3)
        .expect("load");

    let weight_ptr = executor
        .get_quantized_weight_ptr("test_q4_1_into")
        .expect("get ptr");

    // Create input/output buffers
    let input_data = vec![0.1f32; k as usize];
    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let output = GpuBuffer::new(executor.context(), n as usize).expect("output");

    let result = executor.q4_1_gemv_into(weight_ptr, &input, &output, n, k);
    assert!(
        result.is_ok(),
        "q4_1_gemv_into should succeed: {:?}",
        result.err()
    );
}

include!("tests_cov026.rs");
include!("tests_cov028_fused.rs");
include!("tests_cov029_workspace.rs");
include!("tests_cov030_transformer.rs");
