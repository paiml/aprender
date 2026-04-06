//! PARITY-114 Barrier Safety Tests - All Kernels

use super::*;

/// PARITY-114: GEMM naive kernel is barrier-safe
#[test]
fn test_barrier_safety_gemm_naive() {
    let kernel = GemmKernel::naive(64, 64, 64);
    let result = kernel.analyze_barrier_safety();
    assert!(result.is_safe, "GEMM naive should be barrier-safe: {:?}", result.violations);
}

/// PARITY-114: GEMM tiled kernel is barrier-safe
#[test]
fn test_barrier_safety_gemm_tiled() {
    let kernel = GemmKernel::tiled(64, 64, 64, 16);
    let result = kernel.analyze_barrier_safety();
    assert!(result.is_safe, "GEMM tiled should be barrier-safe: {:?}", result.violations);
}

/// PARITY-114: GEMM tensor core kernel is barrier-safe
#[test]
fn test_barrier_safety_gemm_tensor_core() {
    let kernel = GemmKernel::tensor_core(64, 64, 64);
    let result = kernel.analyze_barrier_safety();
    assert!(result.is_safe, "GEMM tensor core should be barrier-safe: {:?}", result.violations);
}

/// PARITY-114: GEMM WMMA FP16 kernel is barrier-safe
#[test]
fn test_barrier_safety_gemm_wmma() {
    let kernel = GemmKernel::wmma_fp16(64, 64, 64);
    let result = kernel.analyze_barrier_safety();
    assert!(result.is_safe, "GEMM WMMA should be barrier-safe: {:?}", result.violations);
}

/// PARITY-114: Attention kernel is barrier-safe
#[test]
fn test_barrier_safety_attention() {
    let kernel = AttentionKernel::new(64, 32);
    let result = kernel.analyze_barrier_safety();
    assert!(result.is_safe, "Attention should be barrier-safe: {:?}", result.violations);
}

/// PARITY-114: Tensor Core attention kernel is barrier-safe
#[test]
fn test_barrier_safety_attention_tensor_core() {
    let kernel = AttentionKernel::tensor_core(64, 32);
    let result = kernel.analyze_barrier_safety();
    assert!(result.is_safe, "TC Attention should be barrier-safe: {:?}", result.violations);
}

/// PARITY-114: Softmax kernel is barrier-safe
#[test]
fn test_barrier_safety_softmax() {
    let kernel = SoftmaxKernel::new(1024);
    let result = kernel.analyze_barrier_safety();
    assert!(result.is_safe, "Softmax should be barrier-safe: {:?}", result.violations);
}

/// PARITY-114: LayerNorm kernel is barrier-safe
#[test]
fn test_barrier_safety_layernorm() {
    let kernel = LayerNormKernel::new(512);
    let result = kernel.analyze_barrier_safety();
    assert!(result.is_safe, "LayerNorm should be barrier-safe: {:?}", result.violations);
}

/// PARITY-114: validate_barrier_safety returns Ok for safe kernels
#[test]
fn test_validate_barrier_safety_ok() {
    let kernel = GemmKernel::naive(32, 32, 32);
    assert!(kernel.validate_barrier_safety().is_ok(), "Safe kernel should pass validation");
}

/// PARITY-114: emit_ptx_validated works for safe kernels
#[test]
fn test_emit_ptx_validated_works() {
    let kernel = GemmKernel::naive(32, 32, 32);
    let ptx = kernel.emit_ptx_validated(); // Should not panic
    assert!(ptx.contains(".entry"));
}

/// PARITY-114: MWV Q6K kernel is barrier-safe (GH-118)
#[test]
fn test_barrier_safety_mwv_q6k() {
    let kernel = MultiWarpQ6KGemvKernel::new(1536, 1536);
    let result = kernel.analyze_barrier_safety();
    assert!(result.is_safe, "MWV Q6K should be barrier-safe: {:?}", result.violations);
}

/// PARITY-114: MWV Q6K warp variants are all barrier-safe
#[test]
fn test_barrier_safety_mwv_q6k_warp_variants() {
    for warps in [1, 2, 3, 4, 6, 8] {
        let kernel = MultiWarpQ6KGemvKernel::with_warps(1536, 1536, warps);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "MWV Q6K ({warps} warps) should be barrier-safe: {:?}",
            result.violations
        );
    }
}

/// PARITY-114: Boundary condition - non-divisible dimensions are barrier-safe
#[test]
fn test_barrier_safety_boundary_conditions() {
    // Test dimensions not divisible by tile size
    let test_cases = [
        GemmKernel::tensor_core(17, 17, 17),
        GemmKernel::tensor_core(33, 33, 33),
        GemmKernel::tensor_core(100, 100, 100),
    ];

    for kernel in test_cases {
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "Boundary case {} should be barrier-safe: {:?}",
            kernel.name(),
            result.violations
        );
    }
}
