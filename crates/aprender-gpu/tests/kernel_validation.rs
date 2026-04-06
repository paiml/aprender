//! Kernel Validation Tests (No CUDA Required)
//!
//! These tests validate PTX structure and scalar implementations without
//! requiring CUDA hardware. Run with:
//!
//! ```bash
//! cargo test -p trueno-gpu --test kernel_validation --features cuda
//! ```

#![cfg(feature = "cuda")]

use trueno_gpu::kernels::{Activation, AttentionKernel, BiasActivationKernel, GemvKernel, Kernel};

// ============================================================================
// SCALAR BASELINE IMPLEMENTATIONS
// ============================================================================

/// Scalar bias + activation implementation for validation
fn scalar_bias_activation(x: &[f32], bias: &[f32], activation: Activation) -> Vec<f32> {
    x.iter()
        .enumerate()
        .map(|(i, &val)| {
            let biased = val + bias[i % bias.len()];
            match activation {
                Activation::None => biased,
                Activation::ReLU => biased.max(0.0),
                Activation::GELU => {
                    // GELU approximation: x * sigmoid(1.702 * x)
                    let scaled = 1.702 * biased;
                    let sigmoid = 1.0 / (1.0 + (-scaled).exp());
                    biased * sigmoid
                }
            }
        })
        .collect()
}

// ============================================================================
// BIAS ACTIVATION KERNEL TESTS
// ============================================================================

#[test]
fn bias_activation_scalar_known_values() {
    // Test with known values to verify scalar implementation
    let x = vec![1.0, -1.0, 0.5, -0.5];
    let bias = vec![0.1, 0.2];

    // None activation: just add bias
    let none_result = scalar_bias_activation(&x, &bias, Activation::None);
    assert!((none_result[0] - 1.1).abs() < 1e-6); // 1.0 + 0.1
    assert!((none_result[1] - (-0.8)).abs() < 1e-6); // -1.0 + 0.2
    assert!((none_result[2] - 0.6).abs() < 1e-6); // 0.5 + 0.1
    assert!((none_result[3] - (-0.3)).abs() < 1e-6); // -0.5 + 0.2

    // ReLU activation: max(0, x + bias)
    let relu_result = scalar_bias_activation(&x, &bias, Activation::ReLU);
    assert!((relu_result[0] - 1.1).abs() < 1e-6); // max(0, 1.1) = 1.1
    assert!((relu_result[1] - 0.0).abs() < 1e-6); // max(0, -0.8) = 0.0
    assert!((relu_result[2] - 0.6).abs() < 1e-6); // max(0, 0.6) = 0.6
    assert!((relu_result[3] - 0.0).abs() < 1e-6); // max(0, -0.3) = 0.0

    // GELU activation: x * sigmoid(1.702 * x)
    let gelu_result = scalar_bias_activation(&x, &bias, Activation::GELU);
    // GELU(1.1) ≈ 1.1 * sigmoid(1.8722) ≈ 1.1 * 0.866 ≈ 0.953
    assert!(gelu_result[0] > 0.9 && gelu_result[0] < 1.0);
    // GELU(-0.8) ≈ -0.8 * sigmoid(-1.3616) ≈ -0.8 * 0.204 ≈ -0.163
    assert!(gelu_result[1] < 0.0 && gelu_result[1] > -0.2);
}

#[test]
fn bias_activation_ptx_structure_none() {
    let kernel = BiasActivationKernel::new(1024, 64);
    let ptx = kernel.emit_ptx();

    // Must have proper PTX structure
    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains(".target sm_"));
    assert!(ptx.contains(".visible .entry bias_activation"));

    // Must have all required parameters
    assert!(ptx.contains(".param .u64 output"));
    assert!(ptx.contains(".param .u64 bias"));
    assert!(ptx.contains(".param .u32 n"));

    // Must have bounds check
    assert!(ptx.contains("setp.ge.u32"));

    // Must have bias modulo indexing
    assert!(ptx.contains("rem.u32"));

    // Must have bias addition
    assert!(ptx.contains("add.f32"));

    // None activation should NOT have max or ex2
    assert!(!ptx.contains("max.f32"), "None activation should not have max");
}

#[test]
fn bias_activation_ptx_structure_relu() {
    let kernel = BiasActivationKernel::new(1024, 64).with_relu();
    let ptx = kernel.emit_ptx();

    // Must have proper PTX structure
    assert!(ptx.contains(".visible .entry bias_activation"));

    // ReLU must use max.f32
    assert!(ptx.contains("max.f32"), "ReLU requires max.f32 instruction");

    // ReLU should NOT have ex2 (GELU instruction)
    assert!(!ptx.contains("ex2.approx"), "ReLU should not have GELU ex2 instruction");
}

#[test]
fn bias_activation_ptx_structure_gelu() {
    let kernel = BiasActivationKernel::new(1024, 64).with_gelu();
    let ptx = kernel.emit_ptx();

    // Must have proper PTX structure
    assert!(ptx.contains(".visible .entry bias_activation"));

    // GELU must use ex2 for exp approximation
    assert!(
        ptx.contains("ex2.approx") || ptx.contains("ex2.f32"),
        "GELU requires ex2 instruction for exp"
    );

    // GELU must use div for sigmoid
    assert!(
        ptx.contains("div.rn.f32") || ptx.contains("div.f32"),
        "GELU requires div instruction for sigmoid"
    );

    // GELU uses 1.702 coefficient (encoded as hex float)
    assert!(
        ptx.contains("0F3FD9DB23") || ptx.contains("1.702"),
        "GELU should have 1.702 coefficient"
    );
}

#[test]
fn bias_activation_all_variants_valid_ptx() {
    for activation in [Activation::None, Activation::ReLU, Activation::GELU] {
        let kernel = BiasActivationKernel::new(4096, 256).with_activation(activation);
        let ptx = kernel.emit_ptx();

        // All variants must have valid PTX
        assert!(ptx.contains(".version"), "{:?} missing PTX version", activation);
        assert!(ptx.contains(".entry"), "{:?} missing entry point", activation);
        assert!(ptx.contains("ret;"), "{:?} missing return statement", activation);
    }
}

#[test]
fn bias_activation_various_sizes() {
    // Test various n and bias_size combinations
    let test_cases = [
        (64, 16),    // Small
        (256, 64),   // Medium
        (1024, 128), // Large
        (4096, 256), // XL
        (100, 17),   // Non-aligned
        (1000, 33),  // Prime bias size
    ];

    for (n, bias_size) in test_cases {
        let kernel = BiasActivationKernel::new(n, bias_size).with_gelu();
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry"), "Failed for n={}, bias_size={}", n, bias_size);
        assert!(ptx.contains("rem.u32"), "Missing modulo for n={}, bias_size={}", n, bias_size);
    }
}

// ============================================================================
// GEMV KERNEL TESTS
// ============================================================================

#[test]
fn gemv_ptx_structure() {
    let kernel = GemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Must have proper PTX structure
    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains(".visible .entry gemv_warp_reduce"));

    // Must have all required parameters
    assert!(ptx.contains(".param .u64 y_ptr"));
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 x_ptr"));
    assert!(ptx.contains(".param .u32 k_dim"));
    assert!(ptx.contains(".param .u32 n_dim"));

    // Must use warp shuffle for reduction
    assert!(
        ptx.contains("shfl.sync.down") || ptx.contains("shfl.down"),
        "GEMV should use warp shuffle"
    );

    // Must use FMA for dot product
    assert!(ptx.contains("fma.rn.f32") || ptx.contains("mad.f32"), "GEMV should use FMA");
}

#[test]
fn gemv_various_dimensions() {
    let test_cases = [
        (4096, 32000), // LLM vocab projection
        (4096, 4096),  // Square
        (2048, 8192),  // Wide
        (8192, 2048),  // Tall
        (128, 128),    // Small
    ];

    for (k, n) in test_cases {
        let kernel = GemvKernel::new(k, n);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry"), "Failed for k={}, n={}", k, n);
        assert!(ptx.contains("shfl"), "Missing warp shuffle for k={}, n={}", k, n);
    }
}

// ============================================================================
// KERNEL NAME TESTS
// ============================================================================

#[test]
fn kernel_names_correct() {
    assert_eq!(BiasActivationKernel::new(1024, 64).name(), "bias_activation");
    assert_eq!(BiasActivationKernel::new(1024, 64).with_relu().name(), "bias_activation");
    assert_eq!(BiasActivationKernel::new(1024, 64).with_gelu().name(), "bias_activation");
    assert_eq!(GemvKernel::new(4096, 4096).name(), "gemv_warp_reduce");
}

// ============================================================================
// TENSOR CORE ATTENTION TESTS
// ============================================================================

#[test]
fn tensor_core_attention_ptx_structure() {
    let kernel = AttentionKernel::tensor_core(512, 64);
    let ptx = kernel.emit_ptx();

    // Must have proper PTX structure
    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains(".target sm_"));
    assert!(ptx.contains(".visible .entry flash_attention_tensor_core"));

    // Must have required parameters
    assert!(ptx.contains(".param .u64 q_ptr"));
    assert!(ptx.contains(".param .u64 k_ptr"));
    assert!(ptx.contains(".param .u64 v_ptr"));
    assert!(ptx.contains(".param .u64 o_ptr"));

    // Must have shared memory
    assert!(ptx.contains(".shared"));

    // Must have WMMA instructions
    assert!(ptx.contains("wmma.load.a"), "Tensor Core kernel should have wmma.load.a");
    assert!(ptx.contains("wmma.load.b"), "Tensor Core kernel should have wmma.load.b");
    assert!(ptx.contains("wmma.mma"), "Tensor Core kernel should have wmma.mma");
    assert!(ptx.contains("wmma.store"), "Tensor Core kernel should have wmma.store");

    // Must have cvta.shared for generic address conversion (required for WMMA)
    // Note: cvta.shared (without .to) converts shared→generic
    assert!(
        ptx.contains("cvta.shared.u64"),
        "Tensor Core kernel must use cvta.shared.u64 for WMMA address conversion"
    );
}

#[test]
fn tensor_core_attention_ptx_validate_with_ptxas() {
    // Skip if ptxas is not available
    if std::process::Command::new("ptxas").arg("--version").output().is_err() {
        println!("Skipping ptxas validation - ptxas not found");
        return;
    }

    let kernel = AttentionKernel::tensor_core(512, 64);
    let ptx = kernel.emit_ptx();

    // Write to temp file
    let temp_dir = std::env::temp_dir();
    let ptx_path = temp_dir.join("test_tensor_core_attention.ptx");
    std::fs::write(&ptx_path, &ptx).expect("Failed to write PTX");

    // Validate with ptxas
    let output = std::process::Command::new("ptxas")
        .arg("--gpu-name")
        .arg("sm_89")
        .arg(&ptx_path)
        .arg("-o")
        .arg("/dev/null")
        .output()
        .expect("Failed to run ptxas");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("ptxas validation failed:\n{}", stderr);
        eprintln!("\nPTX content:\n{}", ptx);
        panic!("PTX validation failed: {}", stderr);
    }

    println!("✓ Tensor Core attention PTX validated with ptxas");

    // Cleanup
    let _ = std::fs::remove_file(&ptx_path);
}
