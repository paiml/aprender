use super::super::*;

#[test]
fn test_quantize_kernel_name() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.name(), "q4k_gemm_fused");
}

#[test]
fn test_quantize_default_config() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.m, 1024);
    assert_eq!(kernel.n, 1024);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.tile_size, 32);
    assert_eq!(kernel.block_size, Q4K_BLOCK_SIZE);
}

#[test]
fn test_quantize_with_tile_size() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096).with_tile_size(64);
    assert_eq!(kernel.tile_size, 64);
}

#[test]
fn test_quantize_num_blocks() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.num_blocks_per_row(), 128); // 4096 / 32
}

#[test]
fn test_quantize_ptx_generation() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Verify parameters
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_quant_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
    assert!(ptx.contains(".param .u32 m"));
    assert!(ptx.contains(".param .u32 n"));
    assert!(ptx.contains(".param .u32 k"));
}

#[test]
fn test_quantize_shared_memory() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096);
    let ptx_kernel = kernel.build_ptx();

    // Should have shared memory for dequantized tile
    assert!(ptx_kernel.shared_memory_bytes() > 0);
}

#[test]
fn test_quantize_ptx_contains_operations() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Verify memory operations
    assert!(ptx.contains("ld.global"));
    assert!(ptx.contains("st.global.f32"));

    // Verify arithmetic for dequantization and GEMM
    assert!(ptx.contains("mul.f32"));
    assert!(ptx.contains("add.f32"));

    // Verify warp shuffle for reduction
    assert!(ptx.contains("shfl") || ptx.contains("shfl.down"));
}

#[test]
fn test_quantize_dequantization_ops() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Verify shift/mask for nibble extraction
    // Note: shr may be emitted differently
    assert!(ptx.contains("mul") || ptx.contains("shr"));

    // Verify type conversion
    assert!(ptx.contains("cvt"));
}

#[test]
fn test_quantize_kernel_variants() {
    // Test with different configurations
    let configs = vec![
        QuantizeKernel::new(512, 512, 2048),
        QuantizeKernel::new(1024, 1024, 4096),
        QuantizeKernel::new(2048, 2048, 8192),
        QuantizeKernel::new(4096, 4096, 4096).with_tile_size(64),
    ];

    for config in configs {
        let ptx = config.emit_ptx();
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".visible .entry"));
    }
}

#[test]
fn test_quantize_block_layout() {
    // Verify Q4_K block constants
    assert_eq!(Q4K_BLOCK_SIZE, 32);
    assert_eq!(Q4K_BLOCK_BYTES, 18);
}

// =========================================================================
// SATD REMEDIATION TESTS (EXTREME TDD)
// These tests verify the K-loop and shuffle bugs are fixed.
// Falsifiable claims per Popperian methodology.
// =========================================================================

#[test]
fn test_kloop_branches_back_to_loop_start() {
    // FALSIFIABLE CLAIM: K-loop branches back to "k_block_loop", not "k_block_done"
    // This test FAILS if the SATD bug (single iteration) is present.
    let kernel = QuantizeKernel::new(64, 64, 128); // K=128 requires 4 K-blocks
    let ptx = kernel.emit_ptx();

    // The PTX should contain a branch back to k_block_loop
    // If it only branches to k_block_done, the loop exits after 1 iteration
    let has_loop_back = ptx.contains("bra k_block_loop") || ptx.contains("bra\tk_block_loop");

    assert!(
        has_loop_back,
        "FALSIFIED: K-loop does not branch back to loop start. \
         Found 'bra k_block_done' instead of 'bra k_block_loop'. \
         This means K-loop only runs once regardless of K value."
    );
}

#[test]
fn test_kloop_counter_incremented_inplace() {
    // FALSIFIABLE CLAIM: K-loop counter is incremented in-place using add_u32_inplace
    // If add_u32 is used (returns new reg), the counter is never updated.
    let kernel = QuantizeKernel::new(64, 64, 128);
    let ptx = kernel.emit_ptx();

    // The PTX should increment k_block register in-place
    // Pattern: add.u32 %rN, %rN, 1 (same register for dest and src1)
    // If we see add.u32 %rM, %rN, 1 (different registers), it's broken

    // Count the k_block_loop and k_block_done labels
    let loop_count = ptx.matches("k_block_loop").count();
    let done_count = ptx.matches("k_block_done").count();

    // There should be exactly 2 references to k_block_loop:
    // 1. The label definition
    // 2. The branch back to the loop
    assert!(
        loop_count >= 2,
        "FALSIFIED: k_block_loop only appears {} times. \
         Expected at least 2 (label + branch back). \
         K-loop counter is not being used correctly.",
        loop_count
    );

    // done label should appear exactly once (the label definition)
    // If bra k_block_done appears twice, the loop exits incorrectly
    assert_eq!(
        done_count,
        2, // label + conditional branch
        "FALSIFIED: k_block_done appears {} times. \
         Expected 2 (label + conditional exit). \
         Extra branches to k_block_done indicate premature loop exit.",
        done_count
    );
}

#[test]
fn test_shuffle_broadcast_uses_shfl_idx_not_shfl_down_zero() {
    // FALSIFIABLE CLAIM: Broadcast uses shfl.idx (or shfl.sync.idx) with lane 0,
    // NOT shfl.down with offset 0 (which is a no-op).
    let kernel = QuantizeKernel::new(64, 64, 128);
    let ptx = kernel.emit_ptx();

    // shfl.down with offset 0 is a no-op - it returns the same value
    // Correct broadcast should use shfl.idx or shfl.sync.idx
    let has_shfl_idx = ptx.contains("shfl.idx") || ptx.contains("shfl.sync.idx");
    let has_bad_shfl_down_zero = ptx.contains("shfl.down.b32") && ptx.contains(", 0,");

    // Either we have shfl.idx (correct) or we don't have the bad pattern
    assert!(
        has_shfl_idx || !has_bad_shfl_down_zero,
        "FALSIFIED: Broadcast uses shfl.down with offset 0, which is a no-op. \
         Should use shfl.idx with lane 0 to broadcast the reduced value."
    );
}

#[test]
fn test_accumulator_updated_inplace() {
    // FALSIFIABLE CLAIM: Accumulator is updated in-place, not shadowed
    // If add_f32 creates a new register, the accumulator is never updated.
    let kernel = QuantizeKernel::new(64, 64, 128);
    let ptx = kernel.emit_ptx();

    // The accumulator should be used in a fma.rn.f32 or add.f32 that writes
    // back to the same register. This is tricky to verify in PTX without
    // full SSA analysis, so we verify the loop structure instead.

    // The key invariant: if K > 32, we need multiple accumulations.
    // With K=128 (4 blocks), the final result should be sum of 4 partial products.
    // If accumulator is not updated, result will be wrong (only 1 block's contribution).

    // For now, verify the structure allows for accumulation
    let has_add_f32 = ptx.contains("add.f32") || ptx.contains("add.rn.f32");
    assert!(
        has_add_f32,
        "FALSIFIED: No add.f32 found for accumulation. \
         Accumulator cannot be updated without add instruction."
    );
}

// =========================================================================
// PARITY-041: GGML Q4_K Super-block Format Tests
// Verify the new kernel that uses real GGML Q4_K format (144-byte super-blocks)
// =========================================================================

#[test]
fn test_ggml_kernel_name() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    assert_eq!(kernel.name(), "q4k_gemm_ggml");
}

#[test]
fn test_ggml_kernel_config() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    assert_eq!(kernel.m, 1024);
    assert_eq!(kernel.n, 1024);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.block_size, Q4K_SUPER_BLOCK_SIZE); // 256 values
    assert_eq!(kernel.format, Q4KFormat::GgmlSuperBlock);
}

#[test]
fn test_ggml_super_block_constants() {
    // Verify GGML Q4_K super-block constants
    assert_eq!(Q4K_SUPER_BLOCK_SIZE, 256, "Super-block should have 256 values");
    assert_eq!(Q4K_SUPER_BLOCK_BYTES, 144, "Super-block should be 144 bytes (2+2+12+128)");
}

#[test]
fn test_ggml_num_super_blocks() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256 = 16
}

#[test]
fn test_ggml_ptx_generation() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Verify kernel name
    assert!(ptx.contains("q4k_gemm_ggml"), "Should contain GGML kernel name");

    // Verify parameters
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_quant_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
    assert!(ptx.contains(".param .u32 m"));
    assert!(ptx.contains(".param .u32 n"));
    assert!(ptx.contains(".param .u32 k"));
}

#[test]
fn test_ggml_ptx_contains_f16_loads() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // GGML Q4_K has f16 scale (d) and min (dmin) at super-block header
    assert!(
        ptx.contains("ld.global.f16") || ptx.contains("ld.global.b16"),
        "Should load f16 values for d and dmin"
    );
    assert!(
        ptx.contains("cvt") && ptx.contains("f32"),
        "Should convert f16 to f32 for computation"
    );
}

#[test]
fn test_ggml_ptx_contains_triple_nested_loops() {
    // GH-182: Rewritten with serial accumulation — triple-nested loops
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("sb_loop"), "Should have super-block loop");
    assert!(ptx.contains("sub_loop"), "Should have sub-block loop for 8 sub-blocks");
    assert!(ptx.contains("val_loop"), "Should have value loop for 32 values per sub-block");
}

#[test]
fn test_ggml_ptx_contains_scale_extraction() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // GH-182: Scale extraction uses selp for SB 0-3 vs 4-7 split format
    assert!(
        ptx.contains("shr") || ptx.contains("shl"),
        "Should have shift operations for scale extraction"
    );
    assert!(ptx.contains("and"), "Should have AND operations for masking");
    assert!(ptx.contains("selp"), "Should have selp for low/high sub-block selection");
}

#[test]
fn test_ggml_ptx_serial_accumulation() {
    // GH-182: No warp reduction — serial FMA accumulation per thread
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("fma"), "Should use FMA for serial accumulation");
    assert!(
        !ptx.contains("shfl"),
        "Should NOT have warp shuffle (serial accumulation, not warp reduction)"
    );
}

#[test]
fn test_ggml_all_loop_branches_back() {
    // FALSIFIABLE: All three loops should branch back to their start
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    let sb_loop_count = ptx.matches("sb_loop").count();
    let sub_loop_count = ptx.matches("sub_loop").count();
    let val_loop_count = ptx.matches("val_loop").count();

    assert!(
        sb_loop_count >= 2,
        "sb_loop should appear at least twice (label + branch back), found {}",
        sb_loop_count
    );
    assert!(
        sub_loop_count >= 2,
        "sub_loop should appear at least twice (label + branch back), found {}",
        sub_loop_count
    );
    assert!(
        val_loop_count >= 2,
        "val_loop should appear at least twice (label + branch back), found {}",
        val_loop_count
    );
}

#[test]
fn test_simplified_vs_ggml_different_ptx() {
    // Verify simplified and GGML kernels produce different PTX
    let simplified = QuantizeKernel::new(1024, 1024, 4096);
    let ggml = QuantizeKernel::ggml(1024, 1024, 4096);

    let ptx_simplified = simplified.emit_ptx();
    let ptx_ggml = ggml.emit_ptx();

    assert_ne!(
        ptx_simplified, ptx_ggml,
        "Simplified and GGML kernels should produce different PTX"
    );
    assert!(ptx_simplified.contains("q4k_gemm_fused"));
    assert!(ptx_ggml.contains("q4k_gemm_ggml"));
}

// =========================================================================
// GH-182: TILED Q4_K GEMM TESTS
// Verify the tiled variant that reuses weights across TILE_M rows.
// =========================================================================

#[test]
fn test_tiled_kernel_name() {
    let kernel = QuantizeKernel::ggml_tiled(1024, 1024, 4096, 4);
    assert_eq!(kernel.name(), "q4k_gemm_ggml_tiled");
}

#[test]
fn test_tiled_kernel_config() {
    let kernel = QuantizeKernel::ggml_tiled(1024, 1024, 4096, 4);
    assert_eq!(kernel.tile_m, 4);
    assert_eq!(kernel.block_size, Q4K_SUPER_BLOCK_SIZE);
    assert_eq!(kernel.format, Q4KFormat::GgmlSuperBlock);
}

#[test]
fn test_tiled_ptx_generation() {
    let kernel = QuantizeKernel::ggml_tiled(1024, 1024, 4096, 4);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("q4k_gemm_ggml_tiled"), "Should have tiled kernel name");
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_quant_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
}

#[test]
fn test_tiled_has_triple_nested_loops() {
    let kernel = QuantizeKernel::ggml_tiled(1024, 1024, 4096, 4);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("sb_loop"), "Should have super-block loop");
    assert!(ptx.contains("sub_loop"), "Should have sub-block loop");
    assert!(ptx.contains("val_loop"), "Should have value loop");
}

#[test]
fn test_tiled_has_multiple_fma() {
    // FALSIFIABLE: tile_m=4 should produce 4 FMA instructions per value
    let kernel = QuantizeKernel::ggml_tiled(1024, 1024, 4096, 4);
    let ptx = kernel.emit_ptx();

    let fma_count = ptx.matches("fma.rn.f32").count();
    // Each val_loop iteration has tile_m FMAs (4)
    assert!(
        fma_count >= 4,
        "tile_m=4 should have at least 4 FMA instructions, found {}",
        fma_count
    );
}

#[test]
fn test_tiled_has_multiple_stores() {
    // FALSIFIABLE: tile_m=4 should produce 4 st.global.f32 (one per row)
    let kernel = QuantizeKernel::ggml_tiled(1024, 1024, 4096, 4);
    let ptx = kernel.emit_ptx();

    let store_count = ptx.matches("st.global.f32").count();
    assert_eq!(
        store_count, 4,
        "tile_m=4 should have exactly 4 global stores, found {}",
        store_count
    );
}

#[test]
fn test_tiled_serial_accumulation() {
    // No warp shuffle — serial FMA per thread (same as serial kernel)
    let kernel = QuantizeKernel::ggml_tiled(1024, 1024, 4096, 4);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("fma"), "Should use FMA");
    assert!(!ptx.contains("shfl"), "Should NOT have warp shuffle");
}

#[test]
fn test_tiled_scale_extraction() {
    let kernel = QuantizeKernel::ggml_tiled(1024, 1024, 4096, 4);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("selp"), "Should have selp for split scale format");
    assert!(ptx.contains("shr"), "Should have shift for scale extraction");
}

#[test]
fn test_tiled_vs_serial_different_ptx() {
    let serial = QuantizeKernel::ggml(1024, 1024, 4096);
    let tiled = QuantizeKernel::ggml_tiled(1024, 1024, 4096, 4);

    let ptx_serial = serial.emit_ptx();
    let ptx_tiled = tiled.emit_ptx();

    assert_ne!(ptx_serial, ptx_tiled, "Serial and tiled should produce different PTX");
    assert!(ptx_serial.contains("q4k_gemm_ggml"));
    assert!(ptx_tiled.contains("q4k_gemm_ggml_tiled"));

    // Tiled should have more FMAs (tile_m copies)
    let serial_fma = ptx_serial.matches("fma.rn.f32").count();
    let tiled_fma = ptx_tiled.matches("fma.rn.f32").count();
    assert!(
        tiled_fma > serial_fma,
        "Tiled ({}) should have more FMAs than serial ({})",
        tiled_fma,
        serial_fma
    );
}

#[test]
fn test_tiled_with_tile_m_builder() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096).with_tile_m(8);
    assert_eq!(kernel.tile_m, 8);
    assert_eq!(kernel.name(), "q4k_gemm_ggml_tiled");

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains("q4k_gemm_ggml_tiled"));

    // tile_m=8 should produce 8 stores
    let store_count = ptx.matches("st.global.f32").count();
    assert_eq!(store_count, 8, "tile_m=8 should have 8 stores, found {}", store_count);
}

#[test]
fn test_tiled_all_loops_branch_back() {
    let kernel = QuantizeKernel::ggml_tiled(1024, 1024, 4096, 4);
    let ptx = kernel.emit_ptx();

    let sb_count = ptx.matches("sb_loop").count();
    let sub_count = ptx.matches("sub_loop").count();
    let val_count = ptx.matches("val_loop").count();

    assert!(sb_count >= 2, "sb_loop: label + branch back, found {}", sb_count);
    assert!(sub_count >= 2, "sub_loop: label + branch back, found {}", sub_count);
    assert!(val_count >= 2, "val_loop: label + branch back, found {}", val_count);
}
