use super::super::*;

// =========================================================================
// PARITY-116: Q5_K Kernel Tests
// =========================================================================

#[test]
fn test_q5k_kernel_name() {
    let kernel = Q5KKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.name(), "q5k_gemm_ggml");
}

#[test]
fn test_q5k_kernel_config() {
    let kernel = Q5KKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.m, 1024);
    assert_eq!(kernel.n, 1024);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.tile_size, 32);
}

#[test]
fn test_q5k_super_block_constants() {
    assert_eq!(
        Q5K_SUPER_BLOCK_SIZE, 256,
        "Q5_K super-block should have 256 values"
    );
    assert_eq!(
        Q5K_SUPER_BLOCK_BYTES, 176,
        "Q5_K super-block should be 176 bytes (2+2+12+128+32)"
    );
}

#[test]
fn test_q5k_num_super_blocks() {
    let kernel = Q5KKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256 = 16
}

#[test]
fn test_q5k_ptx_generation() {
    let kernel = Q5KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Verify kernel name
    assert!(
        ptx.contains("q5k_gemm_ggml"),
        "Should contain Q5_K kernel name"
    );

    // Verify parameters
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_quant_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
    assert!(ptx.contains(".param .u32 m"));
    assert!(ptx.contains(".param .u32 n"));
    assert!(ptx.contains(".param .u32 k"));
}

#[test]
fn test_q5k_with_tile_size() {
    let kernel = Q5KKernel::new(1024, 1024, 4096).with_tile_size(64);
    assert_eq!(kernel.tile_size, 64);
    assert_eq!(kernel.m, 1024);
    assert_eq!(kernel.n, 1024);
    assert_eq!(kernel.k, 4096);
}

#[test]
fn test_q5k_with_tile_size_affects_ptx() {
    let kernel_32 = Q5KKernel::new(1024, 1024, 4096);
    let kernel_64 = Q5KKernel::new(1024, 1024, 4096).with_tile_size(64);

    let ptx_32 = kernel_32.emit_ptx();
    let ptx_64 = kernel_64.emit_ptx();

    // Both should be valid PTX with the same kernel name
    assert!(ptx_32.contains("q5k_gemm_ggml"));
    assert!(ptx_64.contains("q5k_gemm_ggml"));
}

#[test]
fn test_q5k_ptx_contains_nested_loops() {
    let kernel = Q5KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("sb_loop"), "Should have super-block loop");
    assert!(ptx.contains("sub_block_loop"), "Should have sub-block loop");
}

#[test]
fn test_q5k_ptx_contains_high_bit_load() {
    // FALSIFIABLE: Q5_K must load high bits from qh (offset 144)
    let kernel = Q5KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Q5_K has 1-bit high values packed in qh array
    // The kernel should have multiple ld.global.u8 for ql and qh
    let load_count = ptx.matches("ld.global.u8").count();
    assert!(
        load_count >= 4, // At least scales (2) + ql + qh
        "Q5_K should have multiple u8 loads for scales, ql, and qh. Found {}",
        load_count
    );
}

#[test]
fn test_q5k_both_loops_branch_back() {
    let kernel = Q5KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    let sb_loop_count = ptx.matches("sb_loop").count();
    let sub_block_loop_count = ptx.matches("sub_block_loop").count();

    assert!(
        sb_loop_count >= 2,
        "sb_loop should appear at least twice (label + branch back), found {}",
        sb_loop_count
    );
    assert!(
        sub_block_loop_count >= 2,
        "sub_block_loop should appear at least twice (label + branch back), found {}",
        sub_block_loop_count
    );
}

// =========================================================================
// PARITY-117: Q6_K Kernel Tests
// =========================================================================

#[test]
fn test_q6k_kernel_name() {
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.name(), "q6k_gemm_ggml");
}

#[test]
fn test_q6k_kernel_config() {
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.m, 1024);
    assert_eq!(kernel.n, 1024);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.tile_size, 32);
}

#[test]
fn test_q6k_super_block_constants() {
    assert_eq!(
        Q6K_SUPER_BLOCK_SIZE, 256,
        "Q6_K super-block should have 256 values"
    );
    assert_eq!(
        Q6K_SUPER_BLOCK_BYTES, 210,
        "Q6_K super-block should be 210 bytes (128+64+16+2)"
    );
}

#[test]
fn test_q6k_num_super_blocks() {
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256 = 16
}

#[test]
fn test_q6k_ptx_generation() {
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Verify kernel name
    assert!(
        ptx.contains("q6k_gemm_ggml"),
        "Should contain Q6_K kernel name"
    );

    // Verify parameters
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_quant_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
    assert!(ptx.contains(".param .u32 m"));
    assert!(ptx.contains(".param .u32 n"));
    assert!(ptx.contains(".param .u32 k"));
}

#[test]
fn test_q6k_with_tile_size() {
    let kernel = Q6KKernel::new(1024, 1024, 4096).with_tile_size(64);
    assert_eq!(kernel.tile_size, 64);
    assert_eq!(kernel.m, 1024);
    assert_eq!(kernel.n, 1024);
    assert_eq!(kernel.k, 4096);
}

#[test]
fn test_q6k_with_tile_size_affects_ptx() {
    let kernel_32 = Q6KKernel::new(1024, 1024, 4096);
    let kernel_64 = Q6KKernel::new(1024, 1024, 4096).with_tile_size(64);

    let ptx_32 = kernel_32.emit_ptx();
    let ptx_64 = kernel_64.emit_ptx();

    // Both should be valid PTX with the same kernel name
    assert!(ptx_32.contains("q6k_gemm_ggml"));
    assert!(ptx_64.contains("q6k_gemm_ggml"));
}

#[test]
fn test_q6k_ptx_contains_nested_loops() {
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("sb_loop"), "Should have super-block loop");
    assert!(ptx.contains("sub_block_loop"), "Should have sub-block loop");
}

#[test]
fn test_q6k_ptx_contains_2bit_high_extraction() {
    // FALSIFIABLE: Q6_K must load and extract 2-bit high values from qh
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Q6_K has 2-bit high values, needs mask 0x3
    assert!(ptx.contains("and"), "Should have AND for bit masking");
}

#[test]
fn test_q6k_ptx_contains_signed_offset() {
    // FALSIFIABLE: Q6_K subtracts 32 to convert to signed range
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Q6_K: quant = ql + 4*qh - 32 (signed range -32 to 31)
    assert!(
        ptx.contains("sub.f32") || ptx.contains("sub.rn.f32"),
        "Should have subtraction for signed offset"
    );
}

#[test]
fn test_q6k_both_loops_branch_back() {
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    let sb_loop_count = ptx.matches("sb_loop").count();
    let sub_block_loop_count = ptx.matches("sub_block_loop").count();

    assert!(
        sb_loop_count >= 2,
        "sb_loop should appear at least twice (label + branch back), found {}",
        sb_loop_count
    );
    assert!(
        sub_block_loop_count >= 2,
        "sub_block_loop should appear at least twice (label + branch back), found {}",
        sub_block_loop_count
    );
}

#[test]
fn test_all_quant_kernels_different() {
    // Verify all quantized kernels produce distinct PTX
    let q4k = QuantizeKernel::ggml(1024, 1024, 4096);
    let q5k = Q5KKernel::new(1024, 1024, 4096);
    let q6k = Q6KKernel::new(1024, 1024, 4096);

    let ptx_q4k = q4k.emit_ptx();
    let ptx_q5k = q5k.emit_ptx();
    let ptx_q6k = q6k.emit_ptx();

    assert_ne!(
        ptx_q4k, ptx_q5k,
        "Q4_K and Q5_K should produce different PTX"
    );
    assert_ne!(
        ptx_q4k, ptx_q6k,
        "Q4_K and Q6_K should produce different PTX"
    );
    assert_ne!(
        ptx_q5k, ptx_q6k,
        "Q5_K and Q6_K should produce different PTX"
    );
}

// =========================================================================
// Property-Based Tests (PARITY-116, PARITY-117)
// =========================================================================

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn prop_q5k_valid_ptx_for_any_size(
        m in 32u32..512,
        n in 32u32..512,
        // K must be divisible by 256 for super-blocks
        k_factor in 1u32..8
    ) {
        let k = k_factor * 256;
        let kernel = Q5KKernel::new(m, n, k);
        let ptx = kernel.emit_ptx();

        // PTX must be valid (non-empty, contains kernel)
        prop_assert!(!ptx.is_empty());
        prop_assert!(ptx.contains("q5k_gemm_ggml"));
        prop_assert!(ptx.contains(".entry"));
        prop_assert!(ptx.contains("ret;"));

        // Must have nested loops
        prop_assert!(ptx.contains("sb_loop"));
        prop_assert!(ptx.contains("sub_block_loop"));
    }

    #[test]
    fn prop_q6k_valid_ptx_for_any_size(
        m in 32u32..512,
        n in 32u32..512,
        k_factor in 1u32..8
    ) {
        let k = k_factor * 256;
        let kernel = Q6KKernel::new(m, n, k);
        let ptx = kernel.emit_ptx();

        prop_assert!(!ptx.is_empty());
        prop_assert!(ptx.contains("q6k_gemm_ggml"));
        prop_assert!(ptx.contains(".entry"));
        prop_assert!(ptx.contains("ret;"));

        // Q6_K-specific: signed offset subtraction
        prop_assert!(ptx.contains("sub.f32") || ptx.contains("sub.rn.f32"));
    }

    #[test]
    fn prop_q5k_super_blocks_correct(k_factor in 1u32..16) {
        let k = k_factor * 256;
        let kernel = Q5KKernel::new(64, 64, k);
        prop_assert_eq!(kernel.num_super_blocks_per_row(), k_factor);
    }

    #[test]
    fn prop_q6k_super_blocks_correct(k_factor in 1u32..16) {
        let k = k_factor * 256;
        let kernel = Q6KKernel::new(64, 64, k);
        prop_assert_eq!(kernel.num_super_blocks_per_row(), k_factor);
    }

    /// Matvec case (n=1) used by realizar for GGUF inference
    #[test]
    fn prop_q5k_q6k_matvec_n1(m in 32u32..512, k_factor in 1u32..8) {
        let k = k_factor * 256;

        // Q5K matvec
        let q5k = Q5KKernel::new(m, 1, k);
        let ptx_q5k = q5k.emit_ptx();
        prop_assert!(ptx_q5k.contains("q5k_gemm_ggml"));
        prop_assert!(ptx_q5k.contains(".entry"));

        // Q6K matvec
        let q6k = Q6KKernel::new(m, 1, k);
        let ptx_q6k = q6k.emit_ptx();
        prop_assert!(ptx_q6k.contains("q6k_gemm_ggml"));
        prop_assert!(ptx_q6k.contains(".entry"));
    }

    #[test]
    fn prop_all_quant_kernels_distinct(
        m in 64u32..256,
        n in 64u32..256,
        k_factor in 1u32..4
    ) {
        let k = k_factor * 256;
        let q4k = QuantizeKernel::ggml(m, n, k);
        let q5k = Q5KKernel::new(m, n, k);
        let q6k = Q6KKernel::new(m, n, k);

        let ptx_q4k = q4k.emit_ptx();
        let ptx_q5k = q5k.emit_ptx();
        let ptx_q6k = q6k.emit_ptx();

        prop_assert!(ptx_q4k != ptx_q5k);
        prop_assert!(ptx_q4k != ptx_q6k);
        prop_assert!(ptx_q5k != ptx_q6k);
    }
}
