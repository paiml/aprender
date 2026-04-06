//! Golden tests for kernel parameter validation, dimension boundaries, and name correctness.

use trueno_gpu::kernels::{
    GemmKernel, Kernel, LayerNormKernel, RmsNormKernel, SoftmaxKernel, VectorizedRmsNormKernel,
};

// ============================================================================
// KERNEL PARAMETER VALIDATION - Golden Tests
// ============================================================================

#[test]
fn golden_gemm_params_present() {
    let kernel = GemmKernel::naive(64, 64, 64);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".param"), "GOLDEN FAIL: Missing parameters in GEMM\nPTX:\n{}", ptx);
}

#[test]
fn golden_layernorm_params_present() {
    let kernel = LayerNormKernel::new(768);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".param .u64 input_ptr"),
        "GOLDEN FAIL: Missing input_ptr in LayerNorm\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".param .u64 output_ptr"),
        "GOLDEN FAIL: Missing output_ptr in LayerNorm\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".param .u64 gamma_ptr"),
        "GOLDEN FAIL: Missing gamma_ptr in LayerNorm\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_softmax_params_present() {
    let kernel = SoftmaxKernel::new(1024);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".param"), "GOLDEN FAIL: Missing parameters in Softmax\nPTX:\n{}", ptx);
}

// ============================================================================
// VARIOUS DIMENSIONS - Boundary Tests
// ============================================================================

#[test]
fn golden_gemm_small_dimensions() {
    let kernel = GemmKernel::naive(16, 16, 16);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry"), "Small GEMM should still generate valid kernel");
}

#[test]
fn golden_gemm_large_dimensions() {
    let kernel = GemmKernel::naive(2048, 2048, 2048);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry"), "Large GEMM should still generate valid kernel");
}

#[test]
fn golden_layernorm_various_hidden_sizes() {
    for hidden_size in [256, 512, 768, 1024, 1536, 2048, 4096] {
        let kernel = LayerNormKernel::new(hidden_size);
        let ptx = kernel.emit_ptx();
        assert!(
            ptx.contains(".entry"),
            "LayerNorm hidden_size={} should generate valid kernel",
            hidden_size
        );
    }
}

#[test]
fn golden_softmax_various_sizes() {
    for size in [256, 512, 1024, 2048, 4096] {
        let kernel = SoftmaxKernel::new(size);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry"), "Softmax size={} should generate valid kernel", size);
    }
}

// ============================================================================
// KERNEL NAME CORRECTNESS - Golden Tests
// ============================================================================

#[test]
fn golden_kernel_names_consistent() {
    assert_eq!(LayerNormKernel::new(768).name(), "layernorm_warp_shuffle");
    assert_eq!(LayerNormKernel::new(768).without_warp_shuffle().name(), "layernorm_shared");
    assert_eq!(RmsNormKernel::new(2048).name(), "rmsnorm");
    assert_eq!(VectorizedRmsNormKernel::new(2048).name(), "rmsnorm_vectorized");
}
