//! Golden tests for softmax, layernorm, rmsnorm kernel generators.

use trueno_gpu::kernels::{
    BatchedSoftmaxKernel, Kernel, LayerNormKernel, RmsNormKernel, SoftmaxKernel,
    VectorizedRmsNormKernel,
};

// ============================================================================
// SOFTMAX KERNEL - Golden Tests
// ============================================================================

#[test]
fn golden_softmax_kernel_structure() {
    let kernel = SoftmaxKernel::new(1024);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry"), "GOLDEN FAIL: Missing .entry in softmax\nPTX:\n{}", ptx);
    assert!(
        ptx.contains("max") || ptx.contains("shfl"),
        "GOLDEN FAIL: Missing max/reduction in softmax\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("ex2") || ptx.contains("exp"),
        "GOLDEN FAIL: Missing exp in softmax\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("div") || ptx.contains("rcp"),
        "GOLDEN FAIL: Missing division in softmax\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_softmax_kernel_structure() {
    let kernel = BatchedSoftmaxKernel::new(1024, 8);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry"),
        "GOLDEN FAIL: Missing .entry in batched softmax\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("%ctaid") || ptx.contains("%tid"),
        "GOLDEN FAIL: Missing thread/block indexing in batched softmax\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// LAYERNORM KERNEL - Golden Tests
// ============================================================================

#[test]
fn golden_layernorm_warp_shuffle_kernel() {
    let kernel = LayerNormKernel::new(768);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry layernorm_warp_shuffle"),
        "GOLDEN FAIL: Missing layernorm_warp_shuffle entry\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("shfl"),
        "GOLDEN FAIL: Missing warp shuffle in LayerNorm warp_shuffle\nPTX:\n{}",
        ptx
    );
    assert!(ptx.contains("rsqrt"), "GOLDEN FAIL: Missing rsqrt in LayerNorm\nPTX:\n{}", ptx);
    assert!(ptx.contains("div"), "GOLDEN FAIL: Missing division in LayerNorm\nPTX:\n{}", ptx);
}

#[test]
fn golden_layernorm_shared_memory_kernel() {
    let kernel = LayerNormKernel::new(768).without_warp_shuffle();
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry layernorm_shared"),
        "GOLDEN FAIL: Missing layernorm_shared entry\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".shared") || ptx.contains("ld.shared") || ptx.contains("st.shared"),
        "GOLDEN FAIL: Missing shared memory in LayerNorm shared\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("bar.sync"),
        "GOLDEN FAIL: Missing barrier in LayerNorm shared\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_layernorm_epsilon_customization() {
    let kernel1 = LayerNormKernel::new(768).with_epsilon(1e-5);
    let kernel2 = LayerNormKernel::new(768).with_epsilon(1e-6);

    let ptx1 = kernel1.emit_ptx();
    let ptx2 = kernel2.emit_ptx();

    assert!(ptx1.contains(".entry"), "Kernel 1 should have entry");
    assert!(ptx2.contains(".entry"), "Kernel 2 should have entry");
}

#[test]
fn golden_layernorm_without_affine() {
    let kernel = LayerNormKernel::new(768).without_affine();
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry"),
        "GOLDEN FAIL: Missing entry in LayerNorm without affine\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// RMSNORM KERNEL - Golden Tests
// ============================================================================

#[test]
fn golden_rmsnorm_kernel_structure() {
    let kernel = RmsNormKernel::new(2048);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry rmsnorm"), "GOLDEN FAIL: Missing rmsnorm entry\nPTX:\n{}", ptx);
    assert!(ptx.contains("shfl"), "GOLDEN FAIL: Missing warp shuffle in RMSNorm\nPTX:\n{}", ptx);
    assert!(ptx.contains("rsqrt"), "GOLDEN FAIL: Missing rsqrt in RMSNorm\nPTX:\n{}", ptx);
    assert!(ptx.contains("mul"), "GOLDEN FAIL: Missing multiplication in RMSNorm\nPTX:\n{}", ptx);
}

#[test]
fn golden_vectorized_rmsnorm_kernel_structure() {
    let kernel = VectorizedRmsNormKernel::new(2048);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry rmsnorm_vectorized"),
        "GOLDEN FAIL: Missing rmsnorm_vectorized entry\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".shared") || ptx.contains("ld.shared") || ptx.contains("st.shared"),
        "GOLDEN FAIL: Missing shared memory in vectorized RMSNorm\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("bar.sync"),
        "GOLDEN FAIL: Missing barrier in vectorized RMSNorm\nPTX:\n{}",
        ptx
    );
}
