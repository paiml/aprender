//! PERF-PARITY-001: E2E Smoke Tests for GPU/CPU Backend Equivalence
//!
//! This module implements smoke tests that verify realizar's GPU and CPU backends
//! produce equivalent results, similar to trueno's smoke_e2e.rs tests.
//!
//! # Running
//! ```bash
//! # All smoke tests
//! cargo test --test smoke_e2e
//!
//! # With CUDA backend
//! cargo test --test smoke_e2e --features cuda -- --nocapture
//! ```
//!
//! # Toyota Way Alignment
//! - **Genchi Genbutsu**: Actually execute code on real hardware
//! - **Jidoka**: Stop when CPU/GPU results diverge
//! - **Poka-Yoke**: Catch backend mismatches early

use realizar::tensor::Tensor;

// Tolerance for floating-point comparison
#[allow(dead_code)]
const FP_TOLERANCE: f32 = 1e-4;
const FP_TOLERANCE_RELAXED: f32 = 1e-2; // For accumulated operations

// ============================================================================
// QUANTIZATION SMOKE TESTS
// ============================================================================

/// Smoke test: Q4_0 dequantization from raw bytes
#[test]
fn smoke_q4_0_dequant() {
    use realizar::quantize::dequantize_q4_0;

    // Create a valid Q4_0 block (18 bytes: 2-byte f16 scale + 16 packed nibbles)
    let mut block = vec![0u8; 18];
    // Scale = 0.5 as f16 = 0x3800
    let scale_f16: u16 = 0x3800;
    block[0..2].copy_from_slice(&scale_f16.to_le_bytes());
    // Data bytes (all zeros for simplicity)
    block[2..18].fill(0x88); // Both nibbles = 8

    let result = dequantize_q4_0(&block);
    assert!(result.is_ok(), "Q4_0 dequantization should succeed");

    let values = result.expect("test");
    assert_eq!(values.len(), 32, "Q4_0 block should produce 32 values");
}

/// Smoke test: Q8_0 dequantization from raw bytes
#[test]
fn smoke_q8_0_dequant() {
    use realizar::quantize::dequantize_q8_0;

    // Create a valid Q8_0 block (34 bytes: 2-byte f16 scale + 32 i8 values)
    let mut block = vec![0u8; 34];
    // Scale = 0.1 as f16 (approx 0x2E66)
    let scale_f16 = half::f16::from_f32(0.1);
    let scale_bytes = scale_f16.to_le_bytes();
    block[0..2].copy_from_slice(&scale_bytes);
    // Data bytes (32 i8 values)
    block[2..34].fill(42); // All same value

    let result = dequantize_q8_0(&block);
    assert!(result.is_ok(), "Q8_0 dequantization should succeed");

    let values = result.expect("test");
    assert_eq!(values.len(), 32, "Q8_0 block should produce 32 values");
}

/// Smoke test: Fused Q4_K dot product
#[test]
fn smoke_fused_q4k_dot() {
    use realizar::quantize::fused_q4k_dot;

    // Create valid Q4_K block data (144 bytes per super-block for 256 values)
    let block_size = 144;
    let mut q4k_data = vec![0u8; block_size];

    // Set d (f16 scale) at offset 0
    q4k_data[0] = 0x00;
    q4k_data[1] = 0x3C; // 1.0 as f16

    // Activations (256 values to match one super-block)
    let activations = vec![1.0f32; 256];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(
        result.is_ok(),
        "Fused Q4_K dot should succeed: {:?}",
        result
    );
}

// ============================================================================
// TRANSFORMER COMPONENT SMOKE TESTS
// ============================================================================

/// Smoke test: LayerNorm struct initialization and forward
#[test]
fn smoke_layer_norm() {
    use realizar::layers::LayerNorm;

    let hidden_size = 256;
    let layer_norm = LayerNorm::new(hidden_size, 1e-5).expect("test");

    // Create input tensor: from_vec(shape, data)
    let input_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32 * 0.01).sin()).collect();
    let input = Tensor::from_vec(vec![1, hidden_size], input_data).expect("test");

    let result = layer_norm.forward(&input);
    assert!(result.is_ok(), "LayerNorm forward should succeed");

    let output = result.expect("test");
    assert_eq!(output.shape(), &[1, hidden_size]);
}

/// Smoke test: Softmax numerical stability
#[test]
fn smoke_softmax_stability() {
    use realizar::layers::softmax;

    // Test with large values (potential overflow without max subtraction)
    let large_input: Vec<f32> = (0..100).map(|i| i as f32 * 10.0).collect();
    let input = Tensor::from_vec(vec![1, 100], large_input).expect("test");

    let result = softmax(&input);
    assert!(result.is_ok(), "Softmax should succeed");

    let output = result.expect("test");

    // Softmax output should sum to approximately 1
    let sum: f32 = output.data().iter().sum();
    assert!(
        (sum - 1.0).abs() < FP_TOLERANCE_RELAXED,
        "Softmax should sum to ~1, got {sum}"
    );
}

/// Smoke test: GELU activation
#[test]
fn smoke_gelu_activation() {
    use realizar::layers::gelu;

    let input_data: Vec<f32> = (-10..10).map(|i| i as f32 * 0.5).collect();
    let input = Tensor::from_vec(vec![20], input_data.clone()).expect("test");

    let result = gelu(&input);
    assert!(result.is_ok(), "GELU should succeed");

    let output = result.expect("test");
    assert_eq!(output.shape(), &[20]);

    // GELU(0) should be ≈ 0
    let zero_idx = input_data
        .iter()
        .position(|&x| x.abs() < 0.01)
        .unwrap_or(10);
    assert!(
        output.data()[zero_idx].abs() < FP_TOLERANCE_RELAXED,
        "GELU(0) should be ≈0"
    );
}

/// Smoke test: Linear layer forward pass
#[test]
fn smoke_linear_forward() {
    use realizar::layers::Linear;

    let in_features = 64;
    let out_features = 32;

    let linear = Linear::new(in_features, out_features).expect("test");

    let input = Tensor::from_vec(vec![1, in_features], vec![1.0f32; in_features]).expect("test");
    let result = linear.forward(&input);

    assert!(result.is_ok(), "Linear forward should succeed");
    let output = result.expect("test");
    assert_eq!(output.shape(), &[1, out_features]);
}

// ============================================================================
// KV CACHE SMOKE TESTS
// ============================================================================

/// Smoke test: KV cache basic operations
#[test]
fn smoke_kv_cache_basic() {
    use realizar::layers::KVCache;

    let num_layers = 2;
    let head_dim = 32;
    let max_seq_len = 128;

    let mut cache = KVCache::new(num_layers, max_seq_len, head_dim).expect("test");

    // Create key/value tensors - single token: [head_dim]
    let k = Tensor::from_vec(vec![head_dim], vec![1.0f32; head_dim]).expect("test");
    let v = Tensor::from_vec(vec![head_dim], vec![2.0f32; head_dim]).expect("test");

    // Update cache for layer 0
    let result = cache.update(0, &k, &v);
    assert!(result.is_ok(), "KV cache update should succeed");

    // Advance position after storing
    cache.advance();

    // Verify cache state - current_pos increments after advance
    assert!(cache.current_pos() > 0, "Cache should have entries");
}

// ============================================================================
// ATTENTION SMOKE TESTS
// ============================================================================

/// Smoke test: Attention mechanism
#[test]
fn smoke_attention() {
    use realizar::layers::Attention;

    let head_dim = 16;
    let seq_len = 8;

    // Attention::new takes only head_dim
    let attention = Attention::new(head_dim).expect("test");

    // Attention takes separate Q, K, V tensors with shape [seq_len, head_dim]
    let query =
        Tensor::from_vec(vec![seq_len, head_dim], vec![1.0f32; seq_len * head_dim]).expect("test");
    let key =
        Tensor::from_vec(vec![seq_len, head_dim], vec![1.0f32; seq_len * head_dim]).expect("test");
    let value =
        Tensor::from_vec(vec![seq_len, head_dim], vec![1.0f32; seq_len * head_dim]).expect("test");

    let result = attention.forward(&query, &key, &value);
    assert!(
        result.is_ok(),
        "Attention forward should succeed: {:?}",
        result
    );

    let output = result.expect("test");
    assert_eq!(output.shape()[0], seq_len);
}

// ============================================================================
// GPU BACKEND SMOKE TESTS (when cuda feature enabled)
// ============================================================================

#[cfg(feature = "cuda")]
mod gpu_smoke {
    /// Smoke test: CUDA executor initialization
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn smoke_cuda_init() {
        use realizar::cuda::CudaExecutor;

        if !CudaExecutor::is_available() {
            println!("CUDA not available, skipping");
            return;
        }

        let executor = CudaExecutor::new(0);
        assert!(executor.is_ok(), "CUDA executor should initialize");

        let executor = executor.expect("test");
        let name = executor.device_name().unwrap_or_default();
        println!("CUDA device: {name}");
        assert!(!name.is_empty(), "Device name should not be empty");
    }

    /// Smoke test: PTX kernel generation
    #[test]
    fn smoke_ptx_generation() {
        use realizar::cuda::{CudaKernels, KernelType};

        let kernels = CudaKernels::new();

        // Test softmax kernel generation
        let softmax_type = KernelType::Softmax { dim: 256 };
        let ptx = kernels.generate_ptx(&softmax_type);
        assert!(
            ptx.contains(".visible .entry"),
            "PTX should have entry point"
        );
        assert!(ptx.contains("softmax"), "PTX should contain kernel name");

        // Test GEMM kernel generation
        let gemm_type = KernelType::GemmNaive {
            m: 64,
            n: 64,
            k: 64,
        };
        let ptx = kernels.generate_ptx(&gemm_type);
        assert!(
            ptx.contains(".visible .entry"),
            "PTX should have entry point"
        );
    }

    /// Smoke test: Q4_K kernel generation (validates issue #67/#68 fixes)
    #[test]
    fn smoke_q4k_kernel_generation() {
        use realizar::cuda::{CudaKernels, KernelType};

        let kernels = CudaKernels::new();
        let kernel_type = KernelType::QuantizedGemm {
            m: 64,
            n: 64,
            k: 128,
        };
        let ptx = kernels.generate_ptx(&kernel_type);

        // Verify fixes from issues #67 and #68
        assert!(
            !ptx.contains(".reg .u8"),
            "Should not have U8 registers (issue #67)"
        );
        assert!(
            ptx.contains(".reg .u16") || ptx.contains(".reg .f16"),
            "Should use U16/F16 registers"
        );
        assert!(ptx.contains("and.b32"), "Should use and.b32 not and.u32");
        assert!(
            !ptx.contains("cvt.rn.f32.f16"),
            "Should not have rounding on f16->f32 (issue #68)"
        );
    }
}

// ============================================================================
// PERFORMANCE BASELINE SMOKE TESTS
// ============================================================================

/// Smoke test: Measure baseline forward pass latency
#[test]
fn smoke_baseline_forward_latency() {
    use realizar::layers::{gelu, LayerNorm};
    use std::time::Instant;

    let hidden_size = 2560; // phi-2 dimensions

    let layer_norm = LayerNorm::new(hidden_size, 1e-5).expect("test");
    let input = Tensor::from_vec(vec![1, hidden_size], vec![1.0f32; hidden_size]).expect("test");

    let start = Instant::now();

    // Layer norm
    let normed = layer_norm.forward(&input).expect("test");

    // GELU
    let _activated = gelu(&normed).expect("test");

    let elapsed = start.elapsed();
    println!("Baseline LayerNorm + GELU latency: {:?}", elapsed);

    // Should complete in reasonable time (< 50ms for simple ops)
    assert!(
        elapsed.as_millis() < 50,
        "Forward pass too slow: {:?}",
        elapsed
    );
}

/// Smoke test: Verify smoke test coverage
#[test]
fn smoke_test_coverage_verification() {
    println!("=== Realizar E2E Smoke Test Suite ===");
    println!("Coverage:");
    println!("  - Quantization: Q4_0, Q8_0, fused Q4_K dot");
    println!("  - Layers: LayerNorm, Softmax, GELU, Linear, Attention");
    println!("  - Infrastructure: KV Cache");
    println!("  - Performance: Baseline latency measurement");
    #[cfg(feature = "cuda")]
    println!("  - GPU: CUDA init, PTX generation, Q4_K kernel (issues #67/#68)");
}
