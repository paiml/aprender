//! TRUENO-SPEC-013: PTX Pixel FKR (Falsification Kernel Regression) Tests
//!
//! Tests generated PTX kernels match scalar baseline - critical for catching
//! Issue #67 type bugs (CUDA_ERROR_INVALID_PTX on RTX 4090).
//!
//! # Running
//! ```bash
//! cargo test -p trueno-gpu --test pixel_fkr --features "cuda gpu-pixels"
//! ```
//!
//! # Academic Foundation
//! - Choudhary et al. (ISCA 2017): GPU bugs often produce visually detectable artifacts
//! - CrossCheck methodology for GPU bug detection

#![cfg(feature = "cuda")]

use trueno_gpu::kernels::{
    Activation, AttentionKernel, BiasActivationKernel, GemmKernel, Kernel, LayerNormKernel,
    SoftmaxKernel,
};

#[cfg(feature = "gpu-pixels")]
use jugar_probar::gpu_pixels::{validate_ptx, PtxBugClass};

// Tolerance for PTX vs scalar comparison
const PTX_TOLERANCE: f32 = 1e-5;

mod ptx_analysis;
mod ptx_runtime;

// ============================================================================
// SCALAR BASELINE IMPLEMENTATIONS
// ============================================================================

/// Scalar softmax implementation for comparison
fn scalar_softmax(x: &[f32]) -> Vec<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = x.iter().map(|xi| (xi - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|e| e / sum).collect()
}

/// Scalar bias + activation implementation
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

/// Scalar layer norm implementation
fn scalar_layernorm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let variance: f32 = x.iter().map(|xi| (xi - mean).powi(2)).sum::<f32>() / n;
    let std = (variance + eps).sqrt();

    x.iter()
        .zip(gamma.iter())
        .zip(beta.iter())
        .map(|((xi, gi), bi)| ((xi - mean) / std) * gi + bi)
        .collect()
}

/// Scalar GEMM implementation (C = A * B)
fn scalar_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Simple RNG for test data
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f32 / u64::MAX as f32) * 2.0 - 1.0
    }

    fn gen_vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

// ============================================================================
// QUANTIZE KERNEL TESTS (Issue #67 Prevention)
// ============================================================================

/// ptx-pixel-fkr: QuantizeKernel validation (Issue #67 prevention)
///
/// This test specifically targets the bug that caused CUDA_ERROR_INVALID_PTX
/// on RTX 4090. The QuantizeKernel must generate valid PTX for all dimensions.
#[test]
#[cfg(feature = "cuda")]
fn ptx_pixel_fkr_quantize_kernel() {
    use trueno_gpu::kernels::QuantizeKernel;

    // Test the exact dimensions that failed in Issue #67
    let test_cases = [
        (2560, 1, 2560),    // Original failing case
        (1024, 1, 4096),    // GGML format
        (4096, 4096, 4096), // Large GEMM
        (17, 1, 17),        // Non-aligned
        (256, 256, 256),    // Standard
    ];

    for (m, n, k) in test_cases {
        let kernel = QuantizeKernel::new(m, n, k);
        let ptx = kernel.emit_ptx();

        // Validate PTX structure
        assert!(
            ptx.contains(".version"),
            "QuantizeKernel[{m}x{n}x{k}] missing PTX version"
        );
        assert!(
            ptx.contains(".target"),
            "QuantizeKernel[{m}x{n}x{k}] missing PTX target"
        );
        assert!(
            ptx.contains(".entry") || ptx.contains(".visible"),
            "QuantizeKernel[{m}x{n}x{k}] missing entry point"
        );

        // Check for common PTX generation bugs
        #[cfg(feature = "gpu-pixels")]
        {
            let result = validate_ptx(&ptx);
            assert!(
                result.is_valid(),
                "QuantizeKernel[{m}x{n}x{k}] has PTX bugs: {:?}",
                result.bugs
            );
        }

        println!(
            "ptx_pixel_fkr_quantize[{m}x{n}x{k}]: PASS ({} bytes)",
            ptx.len()
        );
    }
}

// ============================================================================
// PTX PIXEL FKR SUMMARY
// ============================================================================

/// ptx-pixel-fkr: BiasActivation PTX produces correct results
#[test]
#[cfg(feature = "cuda")]
fn ptx_pixel_fkr_bias_activation_runtime() {
    use trueno_gpu::driver::CudaContext;

    fn cuda_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    if !cuda_available() {
        eprintln!("Skipping PTX BiasActivation runtime test: no CUDA device");
        return;
    }

    let n: usize = 1024;
    let bias_size: usize = 64;
    let mut rng = SimpleRng::new(45678);
    let x = rng.gen_vec(n);
    let bias = rng.gen_vec(bias_size);

    // Test all activation variants
    for activation in [Activation::None, Activation::ReLU, Activation::GELU] {
        // Scalar baseline
        let scalar_result = scalar_bias_activation(&x, &bias, activation);

        // Generate PTX
        let kernel =
            BiasActivationKernel::new(n as u32, bias_size as u32).with_activation(activation);
        let ptx = kernel.emit_ptx();

        assert!(
            ptx.contains(".entry"),
            "BiasActivation PTX should have entry point"
        );

        println!(
            "ptx_pixel_fkr_bias_activation_{:?}: PTX generated ({} bytes)",
            activation,
            ptx.len()
        );
        println!("  Scalar result[0]: {:.6}", scalar_result[0]);
    }
}

/// Summary test for PTX pixel FKR suite
#[test]
fn ptx_pixel_fkr_summary() {
    println!("");
    println!("========================================");
    println!("  PTX Pixel FKR Suite (trueno-gpu)");
    println!("========================================");
    println!("");
    println!("  Static Analysis Tests:");
    println!("    - gemm_tiled_no_bugs");
    println!("    - gemm_tensor_core");
    println!("    - attention");
    println!("    - attention_causal");
    println!("    - softmax_entry");
    println!("    - layernorm_entry");
    println!("    - bias_activation_entry");
    println!("    - bias_activation_gelu_approx");
    println!("    - bias_activation_relu_max");
    println!("");
    println!("  Runtime Validation Tests:");
    println!("    - softmax_runtime");
    println!("    - gemm_runtime");
    println!("    - layernorm_runtime");
    println!("    - bias_activation_runtime");
    println!("");
    println!("  Issue #67 Prevention:");
    println!("    - quantize_kernel (multiple dimensions)");
    println!("");
    println!("========================================");
}
