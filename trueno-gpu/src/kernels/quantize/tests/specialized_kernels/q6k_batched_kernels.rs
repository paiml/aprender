//! Tests for Coalesced Q6K GEMV and Batched Q6K GEMV kernels (PAR-130).

use super::super::super::*;
use proptest::prelude::*;

// =========================================================================
// COALESCED Q6K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_coalesced_q6k_gemv_kernel_name() {
    let kernel = CoalescedQ6KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.name(), "coalesced_q6k_gemv");
}

#[test]
fn test_coalesced_q6k_gemv_config() {
    let kernel = CoalescedQ6KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 32000);
}

#[test]
fn test_coalesced_q6k_gemv_ptx_generation() {
    let kernel = CoalescedQ6KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry coalesced_q6k_gemv"));
    assert!(ptx.contains("ld.global"));
}

#[test]
fn test_coalesced_q6k_gemv_num_super_blocks() {
    let kernel = CoalescedQ6KGemvKernel::new(4096, 4096);
    assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256 = 16

    let kernel2 = CoalescedQ6KGemvKernel::new(512, 4096);
    assert_eq!(kernel2.num_super_blocks_per_row(), 2); // 512 / 256 = 2
}

#[test]
fn test_coalesced_q6k_gemv_has_warp_shuffle() {
    // PAR-066: Coalesced kernel uses warp shuffle for scale broadcast
    let kernel = CoalescedQ6KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Should use shfl.sync.idx for broadcasting scales from lane 0-15
    assert!(ptx.contains("shfl.sync"), "Should use warp shuffle for scale broadcast");
}

#[test]
fn test_coalesced_q6k_gemv_has_loops() {
    let kernel = CoalescedQ6KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("sb_loop"), "Should have super-block loop");
}

#[test]
fn test_coalesced_q6k_gemv_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = CoalescedQ6KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(
        result.is_safe,
        "Coalesced Q6K GEMV should pass barrier safety: {:?}",
        result.violations
    );
}

#[test]
fn test_coalesced_q6k_gemv_vs_basic_different() {
    // Coalesced and basic Q6K GEMV should produce different PTX
    let coalesced = CoalescedQ6KGemvKernel::new(4096, 4096);
    let basic = Q6KGemvKernel::new(4096, 4096);

    let ptx_coalesced = coalesced.emit_ptx();
    let ptx_basic = basic.emit_ptx();

    assert_ne!(
        ptx_coalesced, ptx_basic,
        "Coalesced and basic kernels should produce different PTX"
    );
    assert!(
        ptx_coalesced.contains("coalesced"),
        "Coalesced PTX should contain 'coalesced' in entry name"
    );
}

proptest! {
    #[test]
    fn prop_coalesced_q6k_generates_valid_ptx(
        k_factor in 1u32..8,
        n in 32u32..4096
    ) {
        let k = k_factor * 256;
        let kernel = CoalescedQ6KGemvKernel::new(k, n);
        let ptx = kernel.emit_ptx();

        prop_assert!(!ptx.is_empty());
        prop_assert!(ptx.contains(".visible .entry coalesced_q6k_gemv"));
    }

    #[test]
    fn prop_coalesced_q6k_super_blocks_correct(k_factor in 1u32..16) {
        let k = k_factor * 256;
        let kernel = CoalescedQ6KGemvKernel::new(k, 64);
        prop_assert_eq!(kernel.num_super_blocks_per_row(), k_factor);
    }
}

// =========================================================================
// BATCHED Q6K GEMV KERNEL TESTS (PAR-130)
// =========================================================================

#[test]
fn test_batched_q6k_gemv_kernel_name() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 32000, 8);
    assert_eq!(kernel.name(), "batched_q6k_gemv_warp_reduce");
}

#[test]
fn test_batched_q6k_gemv_config() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 32000, 8);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 32000);
    assert_eq!(kernel.m, 8);
}

#[test]
fn test_batched_q6k_gemv_num_super_blocks() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256 = 16

    let kernel2 = BatchedQ6KGemvKernel::new(512, 4096, 4);
    assert_eq!(kernel2.num_super_blocks_per_row(), 2); // 512 / 256 = 2
}

#[test]
fn test_batched_q6k_gemv_ptx_generation() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".visible .entry batched_q6k_gemv_warp_reduce"),
        "PTX should contain entry point"
    );
    assert!(ptx.contains("ld.global"), "Should have global memory loads");
}

#[test]
fn test_batched_q6k_gemv_has_m_param() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    let ptx = kernel.emit_ptx();

    // Batched kernel should have m_dim parameter
    assert!(ptx.contains(".param .u32 m_dim"), "Should have m_dim parameter for batch size");
}

#[test]
fn test_batched_q6k_gemv_has_warp_shuffle() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    let ptx = kernel.emit_ptx();

    // Uses warp shuffle for reduction
    assert!(ptx.contains("shfl.sync.down"), "Should use warp shuffle down for reduction");
}

#[test]
fn test_batched_q6k_gemv_has_loops() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    let ptx = kernel.emit_ptx();

    // Should have super-block loop (values are compile-time unrolled, no val_loop)
    assert!(ptx.contains("sb_loop"), "Should have super-block loop");
}

#[test]
fn test_batched_q6k_gemv_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(result.is_safe, "Batched Q6K GEMV should pass barrier safety: {:?}", result.violations);
}

#[test]
fn test_batched_q6k_gemv_different_batch_sizes() {
    // Test various batch sizes produce valid PTX
    for m in [1, 2, 4, 8, 16, 32] {
        let kernel = BatchedQ6KGemvKernel::new(4096, 4096, m);
        let ptx = kernel.emit_ptx();
        assert!(!ptx.is_empty(), "PTX should not be empty for m={m}");
        assert!(
            ptx.contains(".visible .entry batched_q6k_gemv_warp_reduce"),
            "PTX should have entry point for m={m}"
        );
    }
}

#[test]
fn test_batched_q6k_gemv_outputs_multiple_results() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    let ptx = kernel.emit_ptx();

    // Should have multiple st.global for writing M outputs
    let store_count = ptx.matches("st.global.f32").count();
    // With M=4 batch size, kernel should write 4 results per block
    assert!(
        store_count >= 4,
        "Should have at least 4 st.global.f32 for batch size 4, found {store_count}"
    );
}

#[test]
fn test_batched_q6k_gemv_vs_non_batched_different() {
    // Batched and non-batched kernels should produce different PTX
    let batched = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    let non_batched = Q6KGemvKernel::new(4096, 4096);

    let ptx_batched = batched.emit_ptx();
    let ptx_non_batched = non_batched.emit_ptx();

    assert_ne!(
        ptx_batched, ptx_non_batched,
        "Batched and non-batched kernels should produce different PTX"
    );
    assert!(ptx_batched.contains("batched"), "Batched PTX should contain 'batched' in entry name");
}

proptest! {
    #[test]
    fn prop_batched_q6k_generates_valid_ptx(
        k_factor in 1u32..8,
        n in 32u32..4096,
        m in 1u32..16
    ) {
        let k = k_factor * 256;
        let kernel = BatchedQ6KGemvKernel::new(k, n, m);
        let ptx = kernel.emit_ptx();

        prop_assert!(!ptx.is_empty());
        prop_assert!(ptx.contains(".visible .entry batched_q6k_gemv_warp_reduce"));
        prop_assert!(ptx.contains(".param .u32 m_dim"));
    }

    #[test]
    fn prop_batched_q6k_super_blocks_correct(k_factor in 1u32..16) {
        let k = k_factor * 256;
        let kernel = BatchedQ6KGemvKernel::new(k, 64, 4);
        prop_assert_eq!(kernel.num_super_blocks_per_row(), k_factor);
    }
}
