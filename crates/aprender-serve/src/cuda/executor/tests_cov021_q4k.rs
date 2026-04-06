//! CudaExecutor tests Part 05 - COV-021 through COV-025
//!
//! Coverage tests for:
//! - COV-021: activations.rs untested functions (Refs PMAT-802)
//! - COV-022: layer.rs utility functions (Refs PMAT-802)
//! - COV-023: quantized.rs functions (Refs PMAT-802)
//! - COV-024: q5k_matvec, q6k_matvec, gpu_argmax, transformer_layer_host
//! - COV-025: device info, memory, and buffer management

use super::*;
use serial_test::serial;

// =============================================================================
// COV-021: activations.rs untested functions (Refs PMAT-802)
// =============================================================================

#[test]
#[serial]
fn test_cov021_q4k_gemv_gpu_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Create GPU buffers
    let input = vec![0.1f32; k as usize];
    let input_buf = GpuBuffer::from_host(executor.context(), &input).expect("input buf");
    let output_buf = GpuBuffer::new(executor.context(), n as usize).expect("output buf");

    // Weight not cached - should fail
    let result = executor.q4k_gemv_gpu("nonexistent_weight", &input_buf, &output_buf, n, k);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_tensor_core_q4k_gemm_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 16u32;
    let k = 256u32;
    let n = 32u32;

    // Create GPU buffers
    let input = vec![0.1f32; (m * k) as usize];
    let input_buf = GpuBuffer::from_host(executor.context(), &input).expect("input buf");
    let output_buf = GpuBuffer::new(executor.context(), (m * n) as usize).expect("output buf");

    // Weight not cached - should fail
    let result =
        executor.tensor_core_q4k_gemm("nonexistent_weight", &input_buf, &output_buf, m, k, n);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_tensor_core_q4k_gemm_cached_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let k = 256u32;
    let n = 32u32;

    let input = vec![0.1f32; (m * k) as usize];
    let mut output = vec![0.0f32; (m * n) as usize];

    // Weight not cached - should fail
    let result =
        executor.tensor_core_q4k_gemm_cached("nonexistent_weight", &input, &mut output, m, k, n);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_tensor_core_q4k_gemm_cached_input_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let k = 256u32;
    let n = 32u32;

    // Wrong input size - should fail validation before checking weight
    let input = vec![0.1f32; 100]; // Should be m*k = 1024
    let mut output = vec![0.0f32; (m * n) as usize];

    let result = executor.tensor_core_q4k_gemm_cached("any_weight", &input, &mut output, m, k, n);

    assert!(result.is_err(), "Should fail with input size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("Input size"),
        "Error should mention input size: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_tensor_core_q4k_gemm_cached_output_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let k = 256u32;
    let n = 32u32;

    // Correct input size but wrong output size
    let input = vec![0.1f32; (m * k) as usize];
    let mut output = vec![0.0f32; 50]; // Should be m*n = 128

    let result = executor.tensor_core_q4k_gemm_cached("any_weight", &input, &mut output, m, k, n);

    assert!(result.is_err(), "Should fail with output size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("Output size"),
        "Error should mention output size: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_batched_q4k_gemv_cached_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let k = 256u32;
    let n = 32u32;

    let input = vec![0.1f32; (m * k) as usize];
    let mut output = vec![0.0f32; (m * n) as usize];

    // Weight not cached - should fail
    let result =
        executor.batched_q4k_gemv_cached("nonexistent_weight", &input, &mut output, m, k, n);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_batched_q4k_gemv_cached_input_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let k = 256u32;
    let n = 32u32;

    // Wrong input size - should fail validation
    let input = vec![0.1f32; 100]; // Should be m*k = 1024
    let mut output = vec![0.0f32; (m * n) as usize];

    let result = executor.batched_q4k_gemv_cached("any_weight", &input, &mut output, m, k, n);

    assert!(result.is_err(), "Should fail with input size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("Input size"),
        "Error should mention input size: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_batched_q4k_gemv_cached_output_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 4u32;
    let k = 256u32;
    let n = 32u32;

    // Correct input size but wrong output size
    let input = vec![0.1f32; (m * k) as usize];
    let mut output = vec![0.0f32; 50]; // Should be m*n = 128

    let result = executor.batched_q4k_gemv_cached("any_weight", &input, &mut output, m, k, n);

    assert!(result.is_err(), "Should fail with output size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("Output size"),
        "Error should mention output size: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_fused_ffn_q4k_up_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 256u32;
    let intermediate_dim = 512u32;

    let input = vec![0.1f32; hidden_dim as usize];
    let mut output = vec![0.0f32; hidden_dim as usize];

    // FFN up weight not cached - should fail
    let result = executor.fused_ffn_q4k(
        &input,
        &mut output,
        "nonexistent_up",
        "any_down",
        hidden_dim,
        intermediate_dim,
    );

    assert!(result.is_err(), "Should fail when up weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("FFN up weight") || err_msg.contains("not cached"),
        "Error should mention FFN up weight not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_fused_ffn_q4k_down_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 256u32;
    let intermediate_dim = 512u32;

    // Cache the up weight first (Q4K format: 144 bytes per 256 values)
    let num_superblocks = (intermediate_dim as usize * hidden_dim as usize + 255) / 256;
    let up_weight_bytes = num_superblocks * 144;
    let up_weights = vec![0u8; up_weight_bytes];
    executor
        .load_quantized_weights("test_up_weight", &up_weights)
        .expect("load up weight");

    let input = vec![0.1f32; hidden_dim as usize];
    let mut output = vec![0.0f32; hidden_dim as usize];

    // FFN down weight not cached - should fail
    let result = executor.fused_ffn_q4k(
        &input,
        &mut output,
        "test_up_weight",
        "nonexistent_down",
        hidden_dim,
        intermediate_dim,
    );

    assert!(result.is_err(), "Should fail when down weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("FFN down weight") || err_msg.contains("not cached"),
        "Error should mention FFN down weight not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov021_rope_neox_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4u32;
    let head_dim = 64u32;
    let total_size = (num_heads * head_dim) as usize;

    // Create GPU buffers for input and output
    let input_data = vec![0.5f32; total_size];
    let input_buf = GpuBuffer::from_host(executor.context(), &input_data).expect("input buf");
    let output_buf = GpuBuffer::new(executor.context(), total_size).expect("output buf");

    // Position starts at 0
    let position = 0u32;
    let rope_theta = 10000.0f32;

    // RoPE NeoX variant (split halves pairing)
    let result = executor.rope_neox_into(
        &input_buf,
        &output_buf,
        position,
        num_heads,
        head_dim,
        rope_theta,
    );

    assert!(
        result.is_ok(),
        "rope_neox_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov021_rope_indirect_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4u32;
    let head_dim = 64u32;
    let total_size = (num_heads * head_dim) as usize;

    // Create GPU buffers for input and output
    let input_data = vec![0.5f32; total_size];
    let input_buf = GpuBuffer::from_host(executor.context(), &input_data).expect("input buf");
    let output_buf = GpuBuffer::new(executor.context(), total_size).expect("output buf");

    // Position from device memory (single position value)
    let positions: Vec<u32> = vec![0];
    let positions_buf =
        GpuBuffer::from_host(executor.context(), &positions).expect("positions buf");

    let rope_theta = 10000.0f32;

    // RoPE with indirect position lookup
    let result = executor.rope_indirect_into(
        &input_buf,
        &output_buf,
        &positions_buf,
        num_heads,
        head_dim,
        rope_theta,
    );

    assert!(
        result.is_ok(),
        "rope_indirect_into should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov021_rope_neox_indirect_into_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4u32;
    let head_dim = 64u32;
    let total_size = (num_heads * head_dim) as usize;

    // Create GPU buffers for input and output
    let input_data = vec![0.5f32; total_size];
    let input_buf = GpuBuffer::from_host(executor.context(), &input_data).expect("input buf");
    let output_buf = GpuBuffer::new(executor.context(), total_size).expect("output buf");

    // Position from device memory (single position value)
    let positions: Vec<u32> = vec![0];
    let positions_buf =
        GpuBuffer::from_host(executor.context(), &positions).expect("positions buf");

    let rope_theta = 10000.0f32;

    // RoPE NeoX with indirect position lookup
    let result = executor.rope_neox_indirect_into(
        &input_buf,
        &output_buf,
        &positions_buf,
        num_heads,
        head_dim,
        rope_theta,
    );

    assert!(
        result.is_ok(),
        "rope_neox_indirect_into should succeed: {:?}",
        result.err()
    );
}

include!("tests_cov022_has.rs");
include!("tests_cov023_rmsnorm.rs");
include!("tests_cov025_get.rs");
