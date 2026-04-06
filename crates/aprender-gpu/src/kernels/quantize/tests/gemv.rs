use super::super::*;

// =========================================================================
// PAR-003: Q4_K/Q5_K/Q6_K GEMV Kernel Tests
// GEMV kernels for M=1 decode throughput (token generation critical path)
// =========================================================================

#[test]
fn test_q4k_gemv_kernel_name() {
    let kernel = Q4KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.name(), "q4k_gemv_warp_reduce");
}

#[test]
fn test_q4k_gemv_kernel_config() {
    let kernel = Q4KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 32000);
    assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256
}

#[test]
fn test_q4k_gemv_ptx_generation() {
    let kernel = Q4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Verify kernel name
    assert!(ptx.contains("q4k_gemv_warp_reduce"), "Should contain GEMV kernel name");

    // Verify parameters (different from GEMM)
    assert!(ptx.contains(".param .u64 y_ptr"), "Missing y_ptr param");
    assert!(ptx.contains(".param .u64 w_ptr"), "Missing w_ptr param");
    assert!(ptx.contains(".param .u64 x_ptr"), "Missing x_ptr param");
    assert!(ptx.contains(".param .u32 k_dim"), "Missing k_dim param");
    assert!(ptx.contains(".param .u32 n_dim"), "Missing n_dim param");
}

#[test]
fn test_q4k_gemv_has_warp_shuffle() {
    let kernel = Q4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // GEMV uses warp shuffle for reduction (like GemvKernel)
    assert!(
        ptx.contains("shfl.sync.down") || ptx.contains("shfl.down"),
        "Q4K GEMV should use warp shuffle for reduction"
    );
}

#[test]
fn test_q4k_gemv_no_shared_memory() {
    let kernel = Q4KGemvKernel::new(4096, 4096);
    let ptx_kernel = kernel.build_ptx();

    // GEMV kernels don't need shared memory - each warp works independently
    assert_eq!(ptx_kernel.shared_memory_bytes(), 0, "Q4K GEMV should not use shared memory");
}

#[test]
fn test_q4k_gemv_has_fma() {
    let kernel = Q4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Should use FMA for accumulation
    assert!(
        ptx.contains("fma.rn.f32") || ptx.contains("mad.f32"),
        "Q4K GEMV should use FMA for accumulation"
    );
}

#[test]
fn test_q5k_gemv_kernel_name() {
    let kernel = Q5KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.name(), "q5k_gemv_warp_reduce");
}

#[test]
fn test_q5k_gemv_ptx_generation() {
    let kernel = Q5KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("q5k_gemv_warp_reduce"), "Should contain GEMV kernel name");
    assert!(ptx.contains(".param .u64 y_ptr"), "Missing y_ptr param");
    assert!(ptx.contains(".param .u64 w_ptr"), "Missing w_ptr param");
    assert!(ptx.contains(".param .u64 x_ptr"), "Missing x_ptr param");
}

#[test]
fn test_q6k_gemv_kernel_name() {
    let kernel = Q6KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.name(), "q6k_gemv_warp_reduce");
}

#[test]
fn test_q6k_gemv_ptx_generation() {
    let kernel = Q6KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("q6k_gemv_warp_reduce"), "Should contain GEMV kernel name");
    assert!(ptx.contains(".param .u64 y_ptr"), "Missing y_ptr param");
    assert!(ptx.contains(".param .u64 w_ptr"), "Missing w_ptr param");
    assert!(ptx.contains(".param .u64 x_ptr"), "Missing x_ptr param");
}

#[test]
fn test_all_gemv_kernels_different() {
    let q4k = Q4KGemvKernel::new(4096, 4096);
    let q5k = Q5KGemvKernel::new(4096, 4096);
    let q6k = Q6KGemvKernel::new(4096, 4096);

    let ptx_q4k = q4k.emit_ptx();
    let ptx_q5k = q5k.emit_ptx();
    let ptx_q6k = q6k.emit_ptx();

    assert_ne!(ptx_q4k, ptx_q5k, "Q4K and Q5K GEMV should produce different PTX");
    assert_ne!(ptx_q5k, ptx_q6k, "Q5K and Q6K GEMV should produce different PTX");
    assert_ne!(ptx_q4k, ptx_q6k, "Q4K and Q6K GEMV should produce different PTX");
}

#[test]
fn test_q4k_gemv_vs_gemm_different() {
    let gemv = Q4KGemvKernel::new(4096, 4096);
    let gemm = QuantizeKernel::ggml(1, 4096, 4096);

    let ptx_gemv = gemv.emit_ptx();
    let ptx_gemm = gemm.emit_ptx();

    // GEMV and GEMM should have different kernel names and structures
    assert!(ptx_gemv.contains("gemv"), "GEMV kernel should have 'gemv' in name");
    assert!(ptx_gemm.contains("gemm"), "GEMM kernel should have 'gemm' in name");
    assert_ne!(ptx_gemv, ptx_gemm, "GEMV and GEMM should produce different PTX");
}

#[test]
fn test_q4k_gemv_loop_branches_back() {
    // FALSIFIABLE: Super-block loop should branch back to start
    let kernel = Q4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    let sb_loop_count = ptx.matches("sb_loop").count();
    assert!(
        sb_loop_count >= 2,
        "sb_loop should appear at least twice (label + branch back), found {}",
        sb_loop_count
    );
}

#[test]
fn test_q4k_gemv_barrier_safety() {
    // GEMV kernels don't use barriers, so they should be trivially barrier-safe
    use crate::ptx::optimize::barrier_safety;
    let kernel = Q4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(result.is_safe, "Q4K GEMV should be barrier-safe: {:?}", result.violations);
}

#[test]
fn test_q5k_gemv_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = Q5KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(result.is_safe, "Q5K GEMV should be barrier-safe: {:?}", result.violations);
}

#[test]
fn test_q6k_gemv_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = Q6KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(result.is_safe, "Q6K GEMV should be barrier-safe: {:?}", result.violations);
}

// =========================================================================
// PAR-030: Fused RMSNorm + Q4K GEMV Kernel Tests
// =========================================================================

#[test]
fn test_fused_rmsnorm_q4k_gemv_kernel_name() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "fused_rmsnorm_q4k_gemv");
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_config() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
    assert!((kernel.epsilon - 1e-5).abs() < 1e-10);
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_with_epsilon() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096).with_epsilon(1e-6);
    assert!((kernel.epsilon - 1e-6).abs() < 1e-10);
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_ptx_generation() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // Verify parameters
    assert!(ptx.contains(".param .u64 y_ptr"));
    assert!(ptx.contains(".param .u64 w_ptr"));
    assert!(ptx.contains(".param .u64 x_ptr"));
    assert!(ptx.contains(".param .u64 gamma_ptr"));
    assert!(ptx.contains(".param .u32 k_dim"));
    assert!(ptx.contains(".param .u32 n_dim"));

    // Verify kernel name
    assert!(ptx.contains("fused_rmsnorm_q4k_gemv"));
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_shared_memory() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096);
    let ptx_kernel = kernel.build_ptx();

    // Should have shared memory for normalized input + warp partials:
    // 3584 * 4 + 32 = 14368 bytes
    assert!(ptx_kernel.shared_memory_bytes() > 0);
    assert_eq!(ptx_kernel.shared_memory_bytes(), 3584 * 4 + 32);
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_operations() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // Verify RMSNorm operations
    assert!(ptx.contains("rsqrt"), "Should have rsqrt for RMSNorm");
    assert!(ptx.contains("div.rn.f32"), "Should have division for mean");

    // Verify warp shuffle for reductions
    assert!(ptx.contains("shfl"), "Should have warp shuffle");

    // Verify shared memory operations
    assert!(ptx.contains("ld.shared.f32"), "Should load from shared memory");
    assert!(ptx.contains("st.shared.f32"), "Should store to shared memory");

    // Verify barrier synchronization
    assert!(ptx.contains("bar.sync"), "Should have barrier sync");

    // Verify Q4K dequantization (d, dmin loads)
    assert!(ptx.contains("cvt.f32.f16"), "Should convert F16 to F32 for d/dmin");
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_loop_structure() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // Verify load loop exists
    let load_loop_count = ptx.matches("load_loop").count();
    assert!(
        load_loop_count >= 2,
        "load_loop should appear at least twice (label + branch), found {}",
        load_loop_count
    );

    // Verify norm loop exists
    let norm_loop_count = ptx.matches("norm_loop").count();
    assert!(
        norm_loop_count >= 2,
        "norm_loop should appear at least twice (label + branch), found {}",
        norm_loop_count
    );

    // Verify super-block loop exists
    let sb_loop_count = ptx.matches("sb_loop").count();
    assert!(
        sb_loop_count >= 2,
        "sb_loop should appear at least twice (label + branch), found {}",
        sb_loop_count
    );
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_barrier_safety() {
    // This kernel uses barriers, need to verify barrier safety
    use crate::ptx::optimize::barrier_safety;
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(
        result.is_safe,
        "Fused RMSNorm+Q4K GEMV should be barrier-safe: {:?}",
        result.violations
    );
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_qwen3b_config() {
    // Qwen 3B typical dimensions
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 18944); // hidden -> intermediate
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".visible .entry"));
}

// =========================================================================
// PAR-031: Tiled Q4K GEMV Kernel Tests
// =========================================================================

#[test]
fn test_tiled_q4k_gemv_kernel_name() {
    let kernel = TiledQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "tiled_q4k_gemv");
}

#[test]
fn test_tiled_q4k_gemv_config() {
    let kernel = TiledQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
    assert_eq!(kernel.outputs_per_block, 4);
}

#[test]
fn test_tiled_q4k_gemv_with_outputs_per_block() {
    let kernel = TiledQ4KGemvKernel::new(3584, 4096).with_outputs_per_block(8);
    assert_eq!(kernel.outputs_per_block, 8);
}

#[test]
fn test_tiled_q4k_gemv_ptx_generation() {
    let kernel = TiledQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // Verify parameters
    assert!(ptx.contains(".param .u64 y_ptr"));
    assert!(ptx.contains(".param .u64 w_ptr"));
    assert!(ptx.contains(".param .u64 x_ptr"));
    assert!(ptx.contains(".param .u32 k_dim"));
    assert!(ptx.contains(".param .u32 n_dim"));

    // Verify kernel name
    assert!(ptx.contains("tiled_q4k_gemv"));
}

#[test]
fn test_tiled_q4k_gemv_shared_memory() {
    let kernel = TiledQ4KGemvKernel::new(3584, 4096);
    let ptx_kernel = kernel.build_ptx();

    // Should have shared memory for input vector: 3584 * 4 = 14336 bytes
    assert!(ptx_kernel.shared_memory_bytes() > 0);
    assert_eq!(ptx_kernel.shared_memory_bytes(), 3584 * 4);
}

#[test]
fn test_tiled_q4k_gemv_operations() {
    let kernel = TiledQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // GH-37 FIX: Verify direct .shared addressing (not generic cvta.shared)
    // The kernel uses ld.shared.f32 / st.shared.f32 with u32 offsets
    assert!(
        !ptx.contains("cvta.shared"),
        "GH-37: Should NOT use cvta.shared (generic addressing removed)"
    );
    assert!(
        ptx.contains("ld.shared.f32"),
        "GH-37: Should use direct ld.shared.f32 for shared memory loads"
    );
    assert!(
        ptx.contains("st.shared.f32"),
        "GH-37: Should use direct st.shared.f32 for shared memory stores"
    );

    // Verify barrier synchronization
    assert!(ptx.contains("bar.sync"), "Should have barrier sync");

    // Verify warp shuffle for reductions
    assert!(ptx.contains("shfl"), "Should have warp shuffle");

    // Verify Q4K dequantization (d, dmin loads)
    assert!(ptx.contains("cvt.f32.f16"), "Should convert F16 to F32 for d/dmin");
}

#[test]
fn test_tiled_q4k_gemv_loop_structure() {
    let kernel = TiledQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // Verify load loop exists
    let load_loop_count = ptx.matches("load_loop").count();
    assert!(
        load_loop_count >= 2,
        "load_loop should appear at least twice (label + branch), found {}",
        load_loop_count
    );

    // Verify super-block loop exists
    let sb_loop_count = ptx.matches("sb_loop").count();
    assert!(
        sb_loop_count >= 2,
        "sb_loop should appear at least twice (label + branch), found {}",
        sb_loop_count
    );
}

#[test]
fn test_tiled_q4k_gemv_barrier_safety() {
    // This kernel uses barriers for shared memory synchronization
    use crate::ptx::optimize::barrier_safety;
    let kernel = TiledQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(result.is_safe, "Tiled Q4K GEMV should be barrier-safe: {:?}", result.violations);
}

#[test]
fn test_tiled_q4k_gemv_qwen3b_config() {
    // Qwen 3B dimensions
    let kernel = TiledQ4KGemvKernel::new(3584, 18944).with_outputs_per_block(8);
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".visible .entry"));
}
