//! Golden Tests for Elementwise Kernels
//!
//! ⚠️ IMMUTABLE GUARDIAN - DO NOT MODIFY WITHOUT FALSIFICATION EVIDENCE
//!
//! These tests are LOCKED as immutable guardians of elementwise kernel correctness.
//! To modify: First demonstrate a falsifying test case (black swan).
//!
//! Verifies that each elementwise kernel emits valid PTX with expected structure.
//!
//! Requires `cuda` feature: `cargo test -p trueno-gpu --test golden_elementwise_kernels --features cuda`

#![cfg(feature = "cuda")]

use trueno_gpu::kernels::{
    BatchedResidualAddKernel, BatchedScaleKernel, BatchedSwigluKernel, BatchedTransposeKernel,
    ElementwiseMulKernel, FusedResidualRmsNormKernel, FusedSwigluKernel, GeluKernel, Kernel,
    ResidualAddKernel, RopeKernel, ScaleKernel, SiluKernel, TransposeKernel,
};

// ============================================================================
// RESIDUAL ADD KERNELS - Golden Tests
// ============================================================================

#[test]
fn golden_residual_add_kernel() {
    let kernel = ResidualAddKernel::new(1024);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry residual_add"),
        "GOLDEN FAIL: Missing residual_add entry\nPTX:\n{}",
        ptx
    );
    assert!(ptx.contains("add.f32"), "GOLDEN FAIL: Missing f32 add in residual_add\nPTX:\n{}", ptx);
    assert!(
        ptx.contains("ld.global"),
        "GOLDEN FAIL: Missing global load in residual_add\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("st.global"),
        "GOLDEN FAIL: Missing global store in residual_add\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_residual_add_kernel() {
    let kernel = BatchedResidualAddKernel::new(768, 8);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry batched_residual_add"),
        "GOLDEN FAIL: Missing batched_residual_add entry\nPTX:\n{}",
        ptx
    );
    // Uses ctaid.y for batch index
    assert!(
        ptx.contains("%ctaid.y"),
        "GOLDEN FAIL: Missing batch index (ctaid.y) in batched_residual_add\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// ACTIVATION KERNELS - Golden Tests
// ============================================================================

#[test]
fn golden_silu_kernel() {
    let kernel = SiluKernel::new(1024);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry silu"), "GOLDEN FAIL: Missing silu entry\nPTX:\n{}", ptx);
    // SiLU needs exp (via ex2) and division for sigmoid
    assert!(ptx.contains("ex2"), "GOLDEN FAIL: Missing exp (ex2) in silu\nPTX:\n{}", ptx);
    assert!(ptx.contains("div"), "GOLDEN FAIL: Missing division in silu\nPTX:\n{}", ptx);
}

#[test]
fn golden_gelu_kernel() {
    let kernel = GeluKernel::new(1024);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry gelu"), "GOLDEN FAIL: Missing gelu entry\nPTX:\n{}", ptx);
    // GELU needs multiple operations for tanh approximation
    assert!(ptx.contains("mul.f32"), "GOLDEN FAIL: Missing mul.f32 in gelu\nPTX:\n{}", ptx);
    assert!(ptx.contains("ex2"), "GOLDEN FAIL: Missing exp (ex2) in gelu\nPTX:\n{}", ptx);
}

// ============================================================================
// ELEMENTWISE OPS - Golden Tests
// ============================================================================

#[test]
fn golden_elementwise_mul_kernel() {
    let kernel = ElementwiseMulKernel::new(1024);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry elementwise_mul"),
        "GOLDEN FAIL: Missing elementwise_mul entry\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("mul.f32"),
        "GOLDEN FAIL: Missing mul.f32 in elementwise_mul\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_scale_kernel() {
    let kernel = ScaleKernel::new(1024);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry scale"), "GOLDEN FAIL: Missing scale entry\nPTX:\n{}", ptx);
    assert!(ptx.contains("mul.f32"), "GOLDEN FAIL: Missing mul.f32 in scale\nPTX:\n{}", ptx);
}

// ============================================================================
// FUSED KERNELS - Golden Tests
// ============================================================================

#[test]
fn golden_fused_residual_rmsnorm_kernel() {
    let kernel = FusedResidualRmsNormKernel::new(768);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry fused_residual_rmsnorm"),
        "GOLDEN FAIL: Missing fused_residual_rmsnorm entry\nPTX:\n{}",
        ptx
    );
    // RMSNorm needs rsqrt
    assert!(
        ptx.contains("rsqrt"),
        "GOLDEN FAIL: Missing rsqrt in fused_residual_rmsnorm\nPTX:\n{}",
        ptx
    );
    // Uses warp shuffle for reduction
    assert!(
        ptx.contains("shfl"),
        "GOLDEN FAIL: Missing warp shuffle in fused_residual_rmsnorm\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_fused_residual_rmsnorm_custom_epsilon() {
    let kernel = FusedResidualRmsNormKernel::new(768).with_epsilon(1e-6);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry fused_residual_rmsnorm"),
        "Custom epsilon should still produce valid kernel"
    );
}

#[test]
fn golden_fused_swiglu_kernel() {
    let kernel = FusedSwigluKernel::new(1024);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry fused_swiglu"),
        "GOLDEN FAIL: Missing fused_swiglu entry\nPTX:\n{}",
        ptx
    );
    // SwiGLU = SiLU(gate) * up, needs exp and mul
    assert!(ptx.contains("ex2"), "GOLDEN FAIL: Missing exp (ex2) in fused_swiglu\nPTX:\n{}", ptx);
    assert!(ptx.contains("mul.f32"), "GOLDEN FAIL: Missing mul.f32 in fused_swiglu\nPTX:\n{}", ptx);
}

#[test]
fn golden_batched_swiglu_kernel() {
    let kernel = BatchedSwigluKernel::new(2048, 8);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry batched_swiglu"),
        "GOLDEN FAIL: Missing batched_swiglu entry\nPTX:\n{}",
        ptx
    );
    // Should use batch index
    assert!(
        ptx.contains("%ctaid"),
        "GOLDEN FAIL: Missing block index in batched_swiglu\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// ROPE (Rotary Position Embedding) KERNELS - Golden Tests
// ============================================================================

#[test]
fn golden_rope_kernel() {
    let kernel = RopeKernel::new(8, 64, 10000.0); // num_heads=8, head_dim=64, theta=10000
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry rope"), "GOLDEN FAIL: Missing rope entry\nPTX:\n{}", ptx);
    // RoPE needs sin and cos (or their approximations)
    assert!(
        ptx.contains("sin") || ptx.contains("ex2"),
        "GOLDEN FAIL: Missing trig functions in rope\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// TRANSPOSE KERNELS - Golden Tests
// ============================================================================

#[test]
fn golden_transpose_kernel() {
    let kernel = TransposeKernel::new(64, 64);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry transpose"),
        "GOLDEN FAIL: Missing transpose entry\nPTX:\n{}",
        ptx
    );
    // Simple transpose uses global memory loads/stores
    assert!(
        ptx.contains("ld.global") && ptx.contains("st.global"),
        "GOLDEN FAIL: Missing global memory ops in transpose\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_transpose_kernel() {
    let kernel = BatchedTransposeKernel::new(8, 64, 64); // batch=8, rows=64, cols=64
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry batched_transpose"),
        "GOLDEN FAIL: Missing batched_transpose entry\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// SCALE KERNELS - Golden Tests
// ============================================================================

#[test]
fn golden_batched_scale_kernel() {
    let kernel = BatchedScaleKernel::new(1024);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry batched_scale"),
        "GOLDEN FAIL: Missing batched_scale entry\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains("mul.f32"),
        "GOLDEN FAIL: Missing mul.f32 in batched_scale\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// DIMENSION VARIATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_silu_various_sizes() {
    for n in [256, 512, 1024, 2048, 4096] {
        let kernel = SiluKernel::new(n);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry silu"), "SiLU n={} should generate valid kernel", n);
    }
}

#[test]
fn golden_gelu_various_sizes() {
    for n in [256, 512, 1024, 2048] {
        let kernel = GeluKernel::new(n);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gelu"), "GELU n={} should generate valid kernel", n);
    }
}

#[test]
fn golden_fused_residual_rmsnorm_various_hidden() {
    for hidden in [256, 512, 768, 1024, 2048, 4096] {
        let kernel = FusedResidualRmsNormKernel::new(hidden);
        let ptx = kernel.emit_ptx();
        assert!(
            ptx.contains(".entry fused_residual_rmsnorm"),
            "FusedResidualRmsNorm hidden={} should generate valid kernel",
            hidden
        );
    }
}

#[test]
fn golden_rope_various_head_dims() {
    for head_dim in [32, 64, 128] {
        let kernel = RopeKernel::new(8, head_dim, 10000.0); // num_heads=8, head_dim, theta=10000
        let ptx = kernel.emit_ptx();
        assert!(
            ptx.contains(".entry rope"),
            "RoPE head_dim={} should generate valid kernel",
            head_dim
        );
    }
}

// ============================================================================
// KERNEL NAME VERIFICATION - Golden Tests
// ============================================================================

#[test]
fn golden_elementwise_kernel_names() {
    assert_eq!(ResidualAddKernel::new(1024).name(), "residual_add");
    assert_eq!(BatchedResidualAddKernel::new(768, 8).name(), "batched_residual_add");
    assert_eq!(SiluKernel::new(1024).name(), "silu");
    assert_eq!(GeluKernel::new(1024).name(), "gelu");
    assert_eq!(ElementwiseMulKernel::new(1024).name(), "elementwise_mul");
    assert_eq!(ScaleKernel::new(1024).name(), "scale");
    assert_eq!(FusedResidualRmsNormKernel::new(768).name(), "fused_residual_rmsnorm");
    assert_eq!(FusedSwigluKernel::new(1024).name(), "fused_swiglu");
    assert_eq!(BatchedSwigluKernel::new(1024, 8).name(), "batched_swiglu");
    assert_eq!(RopeKernel::new(8, 64, 10000.0).name(), "rope");
    assert_eq!(TransposeKernel::new(64, 64).name(), "transpose");
}
