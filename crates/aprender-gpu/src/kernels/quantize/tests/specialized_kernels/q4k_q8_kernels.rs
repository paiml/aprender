//! Tests for Coalesced Q4K, DP4A Q4K, True DP4A Q4K, Multi-Warp Vectorized Q4K,
//! MWV DP4A Q4K, Wide Q4K, Q8 Quantize, Q4K Q8 Dot, Packed DP4A Q4K Q8, and
//! Batched Q4K kernels.

use super::super::super::*;

// =========================================================================
// COALESCED Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_coalesced_q4k_gemv_kernel_name() {
    let kernel = CoalescedQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "coalesced_q4k_gemv");
}

#[test]
fn test_coalesced_q4k_gemv_config() {
    let kernel = CoalescedQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_coalesced_q4k_gemv_ptx_generation() {
    let kernel = CoalescedQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry coalesced_q4k_gemv"));
    assert!(ptx.contains("ld.global"));
}

// =========================================================================
// DP4A Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_dp4a_q4k_gemv_kernel_name() {
    let kernel = Dp4aQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "dp4a_q4k_gemv");
}

#[test]
fn test_dp4a_q4k_gemv_config() {
    let kernel = Dp4aQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_dp4a_q4k_gemv_ptx_generation() {
    let kernel = Dp4aQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry dp4a_q4k_gemv"));
    // Should have dp4a instructions for int8 dot product
    assert!(ptx.contains("dp4a"));
}

// =========================================================================
// TRUE DP4A Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_true_dp4a_q4k_gemv_kernel_name() {
    let kernel = TrueDp4aQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "true_dp4a_q4k_gemv");
}

#[test]
fn test_true_dp4a_q4k_gemv_config() {
    let kernel = TrueDp4aQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_true_dp4a_q4k_gemv_ptx_generation() {
    let kernel = TrueDp4aQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry true_dp4a_q4k_gemv"));
    assert!(ptx.contains("dp4a"));
}

// =========================================================================
// MULTI-WARP VECTORIZED Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_mwv_q4k_gemv_kernel_name() {
    let kernel = MultiWarpVectorizedQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "mwv_q4k_gemv");
}

#[test]
fn test_mwv_q4k_gemv_config() {
    let kernel = MultiWarpVectorizedQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
    assert_eq!(kernel.num_warps, 4);
}

#[test]
fn test_mwv_q4k_gemv_ptx_generation() {
    let kernel = MultiWarpVectorizedQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry mwv_q4k_gemv"));
    assert!(ptx.contains("ld.global"));
    assert!(ptx.contains("shfl"));
}

#[test]
fn test_mwv_q4k_gemv_shared_memory() {
    let kernel = MultiWarpVectorizedQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // 4 warps → needs shared memory for cross-warp reduction
    assert!(ptx.contains(".shared"));
}

// =========================================================================
// MWV DP4A Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_mwv_dp4a_q4k_gemv_kernel_name() {
    let kernel = MwvDp4aQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "mwv_dp4a_q4k_gemv");
}

#[test]
fn test_mwv_dp4a_q4k_gemv_config() {
    let kernel = MwvDp4aQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
    assert_eq!(kernel.num_warps, 3);
}

#[test]
fn test_mwv_dp4a_q4k_gemv_ptx_generation() {
    let kernel = MwvDp4aQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry mwv_dp4a_q4k_gemv"));
    // Uses DP4A integer dot products for Q8_1 activations
    assert!(ptx.contains("dp4a"));
    assert!(ptx.contains("ld.global"));
    // GH-176: Low-occupancy/high-ILP strategy — maximize registers per thread
    assert!(ptx.contains(".maxnreg 255"), "Must emit .maxnreg 255 for max ILP");
}

#[test]
fn test_mwv_dp4a_q4k_gemv_shared_memory() {
    let kernel = MwvDp4aQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // 3 warps → needs shared memory for cross-warp reduction
    assert!(ptx.contains(".shared"));
}

// =========================================================================
// WIDE Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_wide_q4k_gemv_kernel_name() {
    let kernel = WideQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "wide_q4k_gemv");
}

#[test]
fn test_wide_q4k_gemv_config() {
    let kernel = WideQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
    assert_eq!(kernel.num_warps, 8);
}

#[test]
fn test_wide_q4k_gemv_ptx_generation() {
    let kernel = WideQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry wide_q4k_gemv"));
    assert!(ptx.contains("ld.global"));
    assert!(ptx.contains("shfl"));
}

#[test]
fn test_wide_q4k_gemv_shared_memory() {
    let kernel = WideQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // 8 warps (256 threads) → needs shared memory for cross-warp reduction
    assert!(ptx.contains(".shared"));
}

// =========================================================================
// Q8 QUANTIZE KERNEL TESTS
// =========================================================================

#[test]
fn test_q8_quantize_kernel_name() {
    let kernel = Q8QuantizeKernel::new(3584);
    assert_eq!(kernel.name(), "q8_quantize");
}

#[test]
fn test_q8_quantize_config() {
    let kernel = Q8QuantizeKernel::new(3584);
    assert_eq!(kernel.n, 3584);
}

#[test]
fn test_q8_quantize_ptx_generation() {
    let kernel = Q8QuantizeKernel::new(3584);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry q8_quantize"));
    assert!(ptx.contains("ld.global"));
    assert!(ptx.contains("st.global"));
}

/// Regression test: Q8QuantizeKernel uses both U32 and S32 types.
/// Before the fix, both used %r prefix -> `.reg .u32 %r<11>; .reg .s32 %r<5>;` -> CUDA_ERROR_INVALID_PTX.
/// Fix: S32 now uses %ri prefix, so declarations don't conflict.
#[test]
fn test_q8_quantize_separate_signed_unsigned_registers() {
    let kernel = Q8QuantizeKernel::new(3584);
    let ptx = kernel.emit_ptx();

    // U32 registers use %r prefix
    assert!(ptx.contains(".reg .u32  %r<"), "Missing .u32 register declaration in PTX");
    // S32 registers use %ri prefix (not %r -- that would conflict!)
    assert!(ptx.contains(".reg .s32  %ri<"), "Missing .s32 register declaration in PTX");

    // Verify no duplicate %r declarations (only one .reg line should contain "%r<")
    let r_decl_count = ptx
        .lines()
        .filter(|l| {
            let trimmed = l.trim();
            trimmed.starts_with(".reg") && trimmed.contains(" %r<")
        })
        .count();
    assert_eq!(
        r_decl_count, 1,
        "Expected exactly 1 %r register declaration, got {}. PTX:\n{}",
        r_decl_count, ptx
    );
}

// =========================================================================
// Q4K Q8 DOT KERNEL TESTS
// =========================================================================

#[test]
fn test_q4k_q8_dot_kernel_name() {
    let kernel = Q4KQ8DotKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "q4k_q8_dot");
}

#[test]
fn test_q4k_q8_dot_config() {
    let kernel = Q4KQ8DotKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_q4k_q8_dot_ptx_generation() {
    let kernel = Q4KQ8DotKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry q4k_q8_dot"));
    assert!(ptx.contains("ld.global")); // Loads data
    assert!(ptx.contains("shfl")); // Warp shuffle for reduction
}

// =========================================================================
// PACKED DP4A Q4K Q8 KERNEL TESTS
// =========================================================================

#[test]
fn test_packed_dp4a_q4k_q8_kernel_name() {
    let kernel = PackedDp4aQ4KQ8Kernel::new(3584, 4096);
    assert_eq!(kernel.name(), "packed_dp4a_q4k_q8");
}

#[test]
fn test_packed_dp4a_q4k_q8_config() {
    let kernel = PackedDp4aQ4KQ8Kernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_packed_dp4a_q4k_q8_ptx_generation() {
    let kernel = PackedDp4aQ4KQ8Kernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry packed_dp4a_q4k_q8"));
    assert!(ptx.contains("dp4a"));
}

#[test]
fn test_packed_dp4a_q4k_q8_has_warp_shuffle() {
    let kernel = PackedDp4aQ4KQ8Kernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();
    // This kernel uses warp shuffle for reduction
    assert!(ptx.contains("shfl"));
}

// =========================================================================
// BATCHED Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_batched_q4k_gemv_kernel_name() {
    let kernel = BatchedQ4KGemvKernel::new(4096, 32000, 4);
    assert_eq!(kernel.name(), "batched_q4k_gemv_warp_reduce");
}

#[test]
fn test_batched_q4k_gemv_config() {
    let kernel = BatchedQ4KGemvKernel::new(4096, 32000, 4);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 32000);
    assert_eq!(kernel.m, 4);
}

#[test]
fn test_batched_q4k_gemv_ptx_generation() {
    let kernel = BatchedQ4KGemvKernel::new(4096, 4096, 4);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry batched_q4k_gemv"));
    assert!(ptx.contains("ld.global"));
}

#[test]
fn test_batched_q4k_gemv_has_warp_shuffle() {
    let kernel = BatchedQ4KGemvKernel::new(4096, 4096, 2);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("shfl"));
}

#[test]
fn test_batched_q4k_gemv_num_super_blocks() {
    let kernel = BatchedQ4KGemvKernel::new(4096, 4096, 4);
    assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256
}
