//! Tests for Vectorized Q4K, Fused Gate Up Q4K, and FP16 Q4K kernels.

use super::super::super::*;

// =========================================================================
// VECTORIZED Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_vectorized_q4k_gemv_kernel_name() {
    let kernel = VectorizedQ4KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.name(), "vectorized_q4k_gemv");
}

#[test]
fn test_vectorized_q4k_gemv_config() {
    let kernel = VectorizedQ4KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 32000);
}

#[test]
fn test_vectorized_q4k_gemv_ptx_generation() {
    let kernel = VectorizedQ4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry vectorized_q4k_gemv"));
    assert!(ptx.contains("ld.global"));
}

#[test]
fn test_vectorized_q4k_gemv_has_warp_shuffle() {
    let kernel = VectorizedQ4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("shfl"));
}

// =========================================================================
// FUSED GATE UP Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_fused_gate_up_q4k_gemv_kernel_name() {
    let kernel = FusedGateUpQ4KGemvKernel::new(4096, 11008);
    assert_eq!(kernel.name(), "fused_gate_up_q4k_gemv");
}

#[test]
fn test_fused_gate_up_q4k_gemv_config() {
    let kernel = FusedGateUpQ4KGemvKernel::new(4096, 11008);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 11008);
}

#[test]
fn test_fused_gate_up_q4k_gemv_ptx_generation() {
    let kernel = FusedGateUpQ4KGemvKernel::new(4096, 11008);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry fused_gate_up_q4k_gemv"));
    assert!(ptx.contains("ld.global"));
}

#[test]
fn test_fused_gate_up_q4k_gemv_has_arithmetic() {
    let kernel = FusedGateUpQ4KGemvKernel::new(4096, 11008);
    let ptx = kernel.emit_ptx();

    // Fused gate up uses FMA for efficient multiply-accumulate
    assert!(ptx.contains("fma") || ptx.contains("mul") || ptx.contains("add"));
}

#[test]
fn test_fused_gate_up_q4k_gemv_has_shared_memory() {
    let kernel = FusedGateUpQ4KGemvKernel::new(4096, 11008);
    let ptx_kernel = kernel.build_ptx();

    // Uses shared memory for input caching
    assert!(ptx_kernel.shared_memory_bytes() > 0);
}

// =========================================================================
// FP16 Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_fp16_q4k_gemv_config() {
    let kernel = Fp16Q4KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 32000);
}

#[test]
fn test_fp16_q4k_gemv_ptx_generation() {
    let kernel = Fp16Q4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry fp16_q4k_gemv"));
}

#[test]
fn test_fp16_q4k_gemv_uses_f16() {
    let kernel = Fp16Q4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Should use f16 operations
    assert!(ptx.contains("f16"));
}
