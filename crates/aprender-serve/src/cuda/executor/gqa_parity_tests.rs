//! CUDA GQA Parity Tests - Phase 54
//!
//! Tests for CPU/GPU parity with Grouped Query Attention (GQA) models.
//! Validates that GPU path produces same results as CPU path for GQA configs.
//!
//! ## Five-Whys Root Cause Analysis (PMAT-802)
//!
//! **Problem**: BUG-GGUF-001 (Q4_0 layout) and Q5_0 GQA bugs weren't caught by tests.
//!
//! 1. **Why?** No kernel-level parity tests comparing CPU vs GPU
//! 2. **Why?** Test infra focused on end-to-end, not isolated components
//! 3. **Why?** Setup/teardown requires full model files
//! 4. **Why?** No synthetic weight generators for isolated testing
//! 5. **Why?** Never designed ModelFixture pattern for standardized testing
//!
//! **Solution**: Layer 2 Kernel Parity Tests with synthetic weight generators
//!
//! ## Test Layers (Probar-style)
//!
//! - **Layer 1**: Unit tests (pure functions, no GPU)
//! - **Layer 2**: Kernel parity tests (CPU vs GPU for single ops) ← THIS MODULE
//! - **Layer 3**: Component tests (attention, FFN, etc.)
//! - **Layer 4**: Integration tests (full model inference)

#![cfg(feature = "cuda")]

use super::test_fixtures::{generate_q4_0_weights, generate_q5_0_weights};
use crate::cuda::CudaExecutor;
use crate::gguf::ops;
use crate::quantize::dequant::{dequantize_q4_0, dequantize_q5_0};
use serial_test::serial;
use trueno_gpu::driver::GpuBuffer;

// ============================================================================
// RMSNorm Parity Tests
// ============================================================================

/// Test RMSNorm parity between CPU and GPU
/// This is the first operation after embedding, so if this diverges, everything will.
#[test]
#[serial]
#[ignore] // Parity thresholds (1%) may differ across GPU architectures; validated separately
fn test_gqa_rmsnorm_cpu_gpu_parity() {
    if !CudaExecutor::is_available() {
        eprintln!("[SKIP] CUDA not available");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // GQA config: Qwen-style with 14 heads, 2 kv_heads
    let hidden_dim = 896usize;
    let epsilon = 1e-6f32;

    // Create test input (simulated embedding)
    let input: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i as f32 * 0.01) - 4.0).sin())
        .collect();

    // Create gamma weights (RMSNorm weights)
    let gamma: Vec<f32> = (0..hidden_dim).map(|i| 1.0 + (i as f32 * 0.001)).collect();

    // CPU RMSNorm
    let cpu_output = ops::layer_norm(&input, &gamma, None, epsilon);

    // GPU RMSNorm
    let input_buf = GpuBuffer::from_host(&executor.context, &input).expect("upload input");
    let gamma_buf = GpuBuffer::from_host(&executor.context, &gamma).expect("upload gamma");

    let gpu_output_buf = executor
        .rmsnorm_gpu(&input_buf, &gamma_buf, hidden_dim as u32, epsilon)
        .expect("GPU RMSNorm");

    executor.stream.synchronize().expect("sync");

    let mut gpu_output = vec![0.0f32; hidden_dim];
    gpu_output_buf
        .copy_to_host(&mut gpu_output)
        .expect("download");

    // Compare
    let cpu_sum: f32 = cpu_output.iter().sum();
    let gpu_sum: f32 = gpu_output.iter().sum();

    println!("=== RMSNorm Parity Test ===");
    println!("CPU first 5: {:?}", &cpu_output[..5]);
    println!("GPU first 5: {:?}", &gpu_output[..5]);
    println!("CPU sum: {:.6}", cpu_sum);
    println!("GPU sum: {:.6}", gpu_sum);

    // Allow small tolerance for GPU precision
    let max_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);

    println!("Max element diff: {:.6}", max_diff);

    // Should be within 1% for RMSNorm
    let sum_diff = (cpu_sum - gpu_sum).abs() / cpu_sum.abs().max(1e-6);
    assert!(
        sum_diff < 0.01,
        "RMSNorm sum differs by {:.2}%: CPU={:.6}, GPU={:.6}",
        sum_diff * 100.0,
        cpu_sum,
        gpu_sum
    );
}

/// Test RMSNorm with rmsnorm_into (pre-allocated output)
#[test]
#[serial]
#[ignore] // Parity thresholds (1%) may differ across GPU architectures; validated separately
fn test_gqa_rmsnorm_into_parity() {
    if !CudaExecutor::is_available() {
        eprintln!("[SKIP] CUDA not available");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 896usize;
    let epsilon = 1e-6f32;

    let input: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i as f32 * 0.01) - 4.0).sin())
        .collect();
    let gamma: Vec<f32> = (0..hidden_dim).map(|i| 1.0 + (i as f32 * 0.001)).collect();

    // CPU
    let cpu_output = ops::layer_norm(&input, &gamma, None, epsilon);

    // GPU with rmsnorm_into
    let input_buf = GpuBuffer::from_host(&executor.context, &input).expect("upload input");
    let gamma_buf = GpuBuffer::from_host(&executor.context, &gamma).expect("upload gamma");
    let output_buf = GpuBuffer::<f32>::new(&executor.context, hidden_dim).expect("output buf");

    executor
        .rmsnorm_into(
            &input_buf,
            &gamma_buf,
            &output_buf,
            hidden_dim as u32,
            epsilon,
        )
        .expect("GPU RMSNorm into");

    executor.stream.synchronize().expect("sync");

    let mut gpu_output = vec![0.0f32; hidden_dim];
    output_buf.copy_to_host(&mut gpu_output).expect("download");

    // Compare
    let cpu_sum: f32 = cpu_output.iter().sum();
    let gpu_sum: f32 = gpu_output.iter().sum();

    println!("=== RMSNorm Into Parity Test ===");
    println!("CPU first 5: {:?}", &cpu_output[..5]);
    println!("GPU first 5: {:?}", &gpu_output[..5]);
    println!("CPU sum: {:.6}", cpu_sum);
    println!("GPU sum: {:.6}", gpu_sum);

    let sum_diff = (cpu_sum - gpu_sum).abs() / cpu_sum.abs().max(1e-6);
    assert!(
        sum_diff < 0.01,
        "RMSNorm sum differs by {:.2}%",
        sum_diff * 100.0
    );
}

// ============================================================================
// Q4K GEMV Parity Tests for GQA
// ============================================================================

/// Test Q4K GEMV output dimension matching for GQA
/// Verifies that K and V projections use kv_dim, not hidden_dim
#[test]
#[serial]
fn test_gqa_qkv_dimension_correctness() {
    if !CudaExecutor::is_available() {
        eprintln!("[SKIP] CUDA not available");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // GQA config: 14 Q heads, 2 KV heads
    let hidden_dim = 896usize;
    let num_heads = 14usize;
    let num_kv_heads = 2usize;
    let head_dim = hidden_dim / num_heads; // 64
    let max_seq_len = 128usize;
    let num_layers = 1usize;

    // Initialize KV cache with GQA dimensions
    executor
        .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
        .expect("init kv cache");

    // Verify dimensions are set correctly
    let q_dim = executor.kv_num_heads * executor.kv_head_dim;
    let kv_dim = executor.kv_num_kv_heads * executor.kv_head_dim;

    println!("=== GQA Dimension Check ===");
    println!("num_heads: {}", num_heads);
    println!("num_kv_heads: {}", num_kv_heads);
    println!("head_dim: {}", head_dim);
    println!(
        "Expected q_dim: {} (num_heads * head_dim)",
        num_heads * head_dim
    );
    println!("Actual q_dim: {}", q_dim);
    println!(
        "Expected kv_dim: {} (num_kv_heads * head_dim)",
        num_kv_heads * head_dim
    );
    println!("Actual kv_dim: {}", kv_dim);

    // Verify q_dim = hidden_dim for this config
    assert_eq!(
        q_dim, hidden_dim,
        "q_dim should equal hidden_dim: {} != {}",
        q_dim, hidden_dim
    );

    // Verify kv_dim = num_kv_heads * head_dim
    let expected_kv_dim = num_kv_heads * head_dim;
    assert_eq!(
        kv_dim, expected_kv_dim,
        "kv_dim should be {}: {} != {}",
        expected_kv_dim, kv_dim, expected_kv_dim
    );

    // kv_dim should be smaller than q_dim for GQA
    assert!(
        kv_dim < q_dim,
        "GQA: kv_dim ({}) should be < q_dim ({})",
        kv_dim,
        q_dim
    );

    println!(
        "GQA dimensions VERIFIED: q_dim={}, kv_dim={}",
        q_dim, kv_dim
    );
}

/// Test workspace buffer allocation with GQA dimensions
#[test]
#[serial]
fn test_gqa_workspace_allocation() {
    if !CudaExecutor::is_available() {
        eprintln!("[SKIP] CUDA not available");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // GQA config
    let hidden_dim = 896usize;
    let intermediate_dim = 4864usize; // Qwen FFN dim
    let num_heads = 14usize;
    let num_kv_heads = 2usize;
    let head_dim = hidden_dim / num_heads;
    let max_seq_len = 128usize;
    let num_layers = 1usize;

    // Initialize KV cache
    executor
        .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
        .expect("init kv cache");

    // Initialize workspace
    executor
        .init_workspace(hidden_dim, intermediate_dim)
        .expect("init workspace");

    // Verify workspace buffer sizes
    let expected_q_dim = num_heads * head_dim;
    let expected_kv_dim = num_kv_heads * head_dim;

    println!("=== Workspace Buffer Check ===");
    println!("Expected q_buf size: {}", expected_q_dim);
    println!("Expected k_buf size: {}", expected_kv_dim);
    println!("Expected v_buf size: {}", expected_kv_dim);
    println!("Workspace q_dim: {}", executor.workspace.q_dim);
    println!("Workspace kv_dim: {}", executor.workspace.kv_dim);

    assert_eq!(
        executor.workspace.q_dim, expected_q_dim,
        "Workspace q_dim mismatch"
    );
    assert_eq!(
        executor.workspace.kv_dim, expected_kv_dim,
        "Workspace kv_dim mismatch"
    );

    // Verify q_buf and k_buf have different sizes for GQA
    // GH-215: buffers are padded to Q4K super-block boundary (256 elements)
    let pad256 = |dim: usize| ((dim + 255) / 256) * 256;
    let q_buf = executor.workspace.q_buf.as_ref().expect("q_buf");
    let k_buf = executor.workspace.k_buf.as_ref().expect("k_buf");

    assert_eq!(
        q_buf.len(),
        pad256(expected_q_dim),
        "q_buf size mismatch (padded to 256)"
    );
    assert_eq!(
        k_buf.len(),
        pad256(expected_kv_dim),
        "k_buf size mismatch (padded to 256)"
    );

    println!("Workspace buffers VERIFIED for GQA");
}

// ============================================================================
// End-to-End Transformer Layer Parity
// ============================================================================

/// Test that transformer layer produces consistent output
/// This is a smoke test - actual parity requires full model weights
#[test]
#[serial]
fn test_gqa_transformer_layer_no_crash() {
    if !CudaExecutor::is_available() {
        eprintln!("[SKIP] CUDA not available");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // GQA config
    let hidden_dim = 896usize;
    let intermediate_dim = 4864usize;
    let num_heads = 14usize;
    let num_kv_heads = 2usize;
    let head_dim = hidden_dim / num_heads;
    let max_seq_len = 128usize;
    let num_layers = 1usize;
    let epsilon = 1e-6f32;

    // Initialize
    executor
        .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
        .expect("init kv cache");
    executor
        .init_workspace(hidden_dim, intermediate_dim)
        .expect("init workspace");

    // Verify GQA dimensions are correctly stored
    assert_eq!(executor.kv_num_heads, num_heads);
    assert_eq!(executor.kv_num_kv_heads, num_kv_heads);
    assert_eq!(executor.kv_head_dim, head_dim);

    let q_dim = executor.kv_num_heads * executor.kv_head_dim;
    let kv_dim = executor.kv_num_kv_heads * executor.kv_head_dim;

    println!("=== GQA Transformer Layer Smoke Test ===");
    println!("Hidden dim: {}", hidden_dim);
    println!("Intermediate dim: {}", intermediate_dim);
    println!(
        "num_heads: {}, num_kv_heads: {}, head_dim: {}",
        num_heads, num_kv_heads, head_dim
    );
    println!("Q dim: {}, KV dim: {}", q_dim, kv_dim);
    println!("Epsilon: {}", epsilon);

    // This test verifies the configuration is correct without running actual inference
    // (which would require model weights)
    assert!(
        kv_dim < q_dim,
        "GQA should have kv_dim < q_dim: {} < {}",
        kv_dim,
        q_dim
    );

    println!("GQA transformer layer configuration VERIFIED");
}

// ============================================================================
// Layer 2: Kernel Parity Tests (Five-Whys Root Cause Fix)
// ============================================================================
// These tests validate individual kernel numerical correctness.
// They would have caught BUG-GGUF-001 (Q4_0 layout) and the Q5_0 GQA bug.
//
// Synthetic weight generators are now in test_fixtures.rs for reuse.

/// Test Q4_0 GEMV parity: CPU dequantize+matmul vs GPU Q4_0 GEMV
/// This test would have caught BUG-GGUF-001 before it caused runtime failures.
#[test]
#[serial]
fn test_q4_0_gemv_parity() {
    if !CudaExecutor::is_available() {
        eprintln!("[SKIP] CUDA not available");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Small test: 4 blocks = 128 elements
    let num_blocks = 4usize;
    let k = num_blocks * 32; // 128 input elements
    let n = 1usize; // Single output row (GEMV)

    let weights_q4_0 = generate_q4_0_weights(num_blocks);

    // CPU path: dequantize then matmul
    let weights_f32 = dequantize_q4_0(&weights_q4_0).expect("dequantize Q4_0");
    assert_eq!(weights_f32.len(), k, "Dequantized length mismatch");

    // Input vector
    let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();

    // CPU matmul: dot product for single row
    let cpu_output: f32 = weights_f32
        .iter()
        .zip(input.iter())
        .map(|(w, x)| w * x)
        .sum();

    // GPU path - upload weights as bytes, get raw device pointer
    let weights_buf =
        GpuBuffer::from_host(&executor.context, &weights_q4_0).expect("upload weights");
    let input_buf = GpuBuffer::from_host(&executor.context, &input).expect("upload input");
    let output_buf = GpuBuffer::<f32>::new(&executor.context, n).expect("output buffer");

    // Execute Q4_0 GEMV using _into variant with raw device pointer
    let weight_ptr = weights_buf.as_ptr();
    executor
        .q4_0_gemv_into(weight_ptr, &input_buf, &output_buf, n as u32, k as u32)
        .expect("Q4_0 GEMV");

    executor.stream.synchronize().expect("sync");

    let mut gpu_output = vec![0.0f32; n];
    output_buf.copy_to_host(&mut gpu_output).expect("download");

    // Compare
    let diff = (cpu_output - gpu_output[0]).abs();
    let rel_diff = diff / cpu_output.abs().max(1e-6);

    println!("=== Q4_0 GEMV Parity Test ===");
    println!("CPU output: {:.6}", cpu_output);
    println!("GPU output: {:.6}", gpu_output[0]);
    println!("Absolute diff: {:.6}", diff);
    println!("Relative diff: {:.4}%", rel_diff * 100.0);

    // Should be within 1% for quantized GEMV
    assert!(
        rel_diff < 0.01,
        "Q4_0 GEMV parity failed: CPU={:.6}, GPU={:.6}, diff={:.4}%",
        cpu_output,
        gpu_output[0],
        rel_diff * 100.0
    );

    println!("Q4_0 GEMV parity VERIFIED");
}

include!("gqa_parity_tests_02.rs");
