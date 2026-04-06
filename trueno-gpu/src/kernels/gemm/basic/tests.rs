//! Tests for basic GEMM kernels

use super::*;
use crate::kernels::Kernel;

// =========================================================================
// GemmConfig Tests
// =========================================================================

#[test]
fn test_gemm_config_default() {
    let config = GemmConfig::default();
    assert_eq!(config.m, 1024);
    assert_eq!(config.n, 1024);
    assert_eq!(config.k, 1024);
    assert_eq!(config.tile_size, 32);
    assert!(!config.use_tensor_cores);
}

#[test]
fn test_gemm_config_clone() {
    let config = GemmConfig { m: 512, n: 256, k: 128, tile_size: 16, use_tensor_cores: true };
    let cloned = config.clone();
    assert_eq!(cloned.m, 512);
    assert_eq!(cloned.n, 256);
    assert_eq!(cloned.k, 128);
    assert_eq!(cloned.tile_size, 16);
    assert!(cloned.use_tensor_cores);
}

#[test]
fn test_gemm_config_debug() {
    let config = GemmConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("GemmConfig"));
    assert!(debug_str.contains("1024"));
}

// =========================================================================
// GemmKernel Constructor Tests
// =========================================================================

#[test]
fn test_gemm_kernel_naive_constructor() {
    let kernel = GemmKernel::naive(128, 256, 64);
    assert_eq!(kernel.config.m, 128);
    assert_eq!(kernel.config.n, 256);
    assert_eq!(kernel.config.k, 64);
    assert_eq!(kernel.config.tile_size, 32); // default
    assert!(!kernel.config.use_tensor_cores);
}

#[test]
fn test_gemm_kernel_tiled_constructor() {
    let kernel = GemmKernel::tiled(256, 512, 128, 16);
    assert_eq!(kernel.config.m, 256);
    assert_eq!(kernel.config.n, 512);
    assert_eq!(kernel.config.k, 128);
    assert_eq!(kernel.config.tile_size, 16);
    assert!(!kernel.config.use_tensor_cores);
}

#[test]
fn test_gemm_kernel_tiled_unrolled_constructor() {
    let kernel = GemmKernel::tiled_unrolled(128, 256, 64, 32);
    assert_eq!(kernel.config.m, 128);
    assert_eq!(kernel.config.n, 256);
    assert_eq!(kernel.config.k, 64);
    assert_eq!(kernel.config.tile_size, 32);
    assert!(!kernel.config.use_tensor_cores);
}

#[test]
fn test_gemm_kernel_tensor_core_constructor() {
    let kernel = GemmKernel::tensor_core(512, 512, 256);
    assert_eq!(kernel.config.m, 512);
    assert_eq!(kernel.config.n, 512);
    assert_eq!(kernel.config.k, 256);
    assert!(kernel.config.use_tensor_cores);
}

#[test]
fn test_gemm_kernel_wmma_fp16_constructor() {
    let kernel = GemmKernel::wmma_fp16(256, 256, 128);
    assert_eq!(kernel.config.m, 256);
    assert_eq!(kernel.config.n, 256);
    assert_eq!(kernel.config.k, 128);
    assert_eq!(kernel.config.tile_size, 16); // WMMA uses 16x16x16 tiles
    assert!(kernel.config.use_tensor_cores);
}

#[test]
fn test_gemm_kernel_clone() {
    let kernel = GemmKernel::naive(64, 64, 64);
    let cloned = kernel.clone();
    assert_eq!(cloned.config.m, 64);
    assert_eq!(cloned.name(), kernel.name());
}

#[test]
fn test_gemm_kernel_debug() {
    let kernel = GemmKernel::naive(32, 32, 32);
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("GemmKernel"));
    assert!(debug_str.contains("config"));
}

// =========================================================================
// Kernel Trait Tests - name()
// =========================================================================

#[test]
fn test_kernel_name_naive() {
    let kernel = GemmKernel::naive(64, 64, 64);
    assert_eq!(kernel.name(), "gemm_naive");
}

#[test]
fn test_kernel_name_tiled() {
    let kernel = GemmKernel::tiled(64, 64, 64, 16);
    assert_eq!(kernel.name(), "gemm_tiled");
}

#[test]
fn test_kernel_name_tiled_unrolled() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 64, 16);
    assert_eq!(kernel.name(), "gemm_tiled_unrolled");
}

#[test]
fn test_kernel_name_tensor_core() {
    let kernel = GemmKernel::tensor_core(64, 64, 64);
    assert_eq!(kernel.name(), "gemm_tensor_core");
}

#[test]
fn test_kernel_name_wmma_fp16() {
    let kernel = GemmKernel::wmma_fp16(64, 64, 64);
    assert_eq!(kernel.name(), "gemm_wmma_fp16");
}

// =========================================================================
// Kernel Trait Tests - build_ptx()
// =========================================================================

#[test]
fn test_build_ptx_naive_returns_valid_kernel() {
    let kernel = GemmKernel::naive(64, 64, 64);
    let _ptx_kernel = kernel.build_ptx();
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry gemm_naive"));
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
}

#[test]
fn test_build_ptx_tiled_returns_valid_kernel() {
    let kernel = GemmKernel::tiled(64, 64, 64, 16);
    let ptx_kernel = kernel.build_ptx();
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry gemm_tiled"));
    assert!(ptx.contains("bar.sync")); // barriers for shared memory
    assert!(ptx_kernel.shared_memory_bytes() > 0);
}

#[test]
fn test_build_ptx_tiled_unrolled_returns_valid_kernel() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 64, 16);
    let ptx_kernel = kernel.build_ptx();
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry gemm_tiled_unrolled"));
    assert!(ptx.contains("bar.sync")); // barriers for shared memory
    assert!(ptx_kernel.shared_memory_bytes() > 0);
}

#[test]
fn test_build_ptx_tensor_core_returns_valid_kernel() {
    let kernel = GemmKernel::tensor_core(64, 64, 64);
    let ptx_kernel = kernel.build_ptx();
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry gemm_tensor_core"));
    assert!(ptx.contains("bar.sync"));
    assert!(ptx_kernel.shared_memory_bytes() > 0);
}

#[test]
fn test_build_ptx_wmma_fp16_returns_valid_kernel() {
    let kernel = GemmKernel::wmma_fp16(64, 64, 64);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry gemm_wmma_fp16"));
    assert!(ptx.contains("wmma")); // WMMA instructions
}

// =========================================================================
// TiledUnrolled Variant Tests (Main Coverage Gap)
// =========================================================================

#[test]
fn test_tiled_unrolled_emits_valid_ptx() {
    let kernel = GemmKernel::tiled_unrolled(128, 128, 128, 32);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains(".target sm_70"));
    assert!(ptx.contains(".address_size 64"));
    assert!(ptx.contains(".visible .entry gemm_tiled_unrolled"));
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
    assert!(ptx.contains(".param .u32 m"));
    assert!(ptx.contains(".param .u32 n"));
    assert!(ptx.contains(".param .u32 k"));
    assert!(ptx.contains("bar.sync"));
    assert!(ptx.contains("ret;"));
}

#[test]
fn test_tiled_unrolled_uses_shared_memory() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 64, 16);
    let ptx_kernel = kernel.build_ptx();

    assert!(ptx_kernel.shared_memory_bytes() > 0);
    assert_eq!(ptx_kernel.shared_memory_bytes(), 16 * 16 * 4 * 2);
}

#[test]
fn test_tiled_unrolled_various_tile_sizes() {
    for tile_size in [8, 16, 32] {
        let kernel = GemmKernel::tiled_unrolled(128, 128, 128, tile_size);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry gemm_tiled_unrolled"), "Failed for tile_size={tile_size}");
        assert!(ptx.contains("bar.sync"), "Missing barriers for tile_size={tile_size}");
    }
}

#[test]
fn test_tiled_unrolled_various_dimensions() {
    let test_cases = [
        (32, 32, 32, 8),
        (64, 128, 64, 16),
        (256, 256, 256, 32),
        (100, 100, 100, 8), // non-power-of-two
    ];

    for (m, n, k, tile) in test_cases {
        let kernel = GemmKernel::tiled_unrolled(m, n, k, tile);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry gemm_tiled_unrolled"), "Failed for ({m}, {n}, {k}, {tile})");
        assert!(ptx.contains("ret;"), "Missing ret for ({m}, {n}, {k}, {tile})");
    }
}

#[test]
fn test_tiled_unrolled_shared_memory_calculation() {
    let kernel = GemmKernel::tiled_unrolled(128, 128, 128, 16);
    let ptx_kernel = kernel.build_ptx();

    let expected_smem = 16 * 16 * 4 * 2; // 2048 bytes
    assert_eq!(ptx_kernel.shared_memory_bytes(), expected_smem);

    let kernel2 = GemmKernel::tiled_unrolled(256, 256, 256, 32);
    let ptx_kernel2 = kernel2.build_ptx();

    let expected_smem2 = 32 * 32 * 4 * 2; // 8192 bytes
    assert_eq!(ptx_kernel2.shared_memory_bytes(), expected_smem2);
}

#[test]
fn test_tiled_unrolled_ptx_contains_fma() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("fma.rn.f32"));
}

#[test]
fn test_tiled_unrolled_n_tiles_calculation() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry gemm_tiled_unrolled"));
}

// =========================================================================
// Barrier Safety Tests
// =========================================================================

#[test]
fn test_tiled_unrolled_barrier_safety() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 64, 16);
    let result = kernel.analyze_barrier_safety();

    assert!(result.is_safe, "TiledUnrolled GEMM should be barrier-safe: {:?}", result.violations);
}

#[test]
fn test_tiled_unrolled_validate_barrier_safety() {
    let kernel = GemmKernel::tiled_unrolled(128, 128, 128, 32);
    let result = kernel.validate_barrier_safety();

    assert!(result.is_ok(), "TiledUnrolled should pass barrier validation: {:?}", result);
}

#[test]
fn test_tiled_unrolled_emit_ptx_validated() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 64, 16);
    let ptx = kernel.emit_ptx_validated();
    assert!(!ptx.is_empty());
}

// =========================================================================
// Edge Cases
// =========================================================================

#[test]
fn test_tiled_unrolled_minimum_dimensions() {
    let kernel = GemmKernel::tiled_unrolled(8, 8, 8, 8);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry gemm_tiled_unrolled"));
}

#[test]
fn test_tiled_unrolled_large_dimensions() {
    let kernel = GemmKernel::tiled_unrolled(2048, 2048, 2048, 32);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry gemm_tiled_unrolled"));
}

#[test]
fn test_tiled_unrolled_non_square() {
    let kernel = GemmKernel::tiled_unrolled(64, 128, 256, 16);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry gemm_tiled_unrolled"));
}

#[test]
fn test_tiled_unrolled_k_not_divisible_by_tile() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 100, 16);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry gemm_tiled_unrolled"));
}

// =========================================================================
// PTX Instruction Verification
// =========================================================================

#[test]
fn test_naive_ptx_contains_expected_instructions() {
    let kernel = GemmKernel::naive(64, 64, 64);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("ld.global"));
    assert!(ptx.contains("st.global"));
    assert!(ptx.contains("fma.rn.f32") || ptx.contains("mad"));
    assert!(ptx.contains("add"));
    assert!(ptx.contains("mul"));
    assert!(ptx.contains("bra"));
    assert!(ptx.contains("setp"));
}

#[test]
fn test_tiled_ptx_contains_expected_instructions() {
    let kernel = GemmKernel::tiled(64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("ld.shared"));
    assert!(ptx.contains("st.shared"));
    assert!(ptx.contains("ld.global"));
    assert!(ptx.contains("st.global"));
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_tiled_unrolled_ptx_contains_expected_instructions() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("ld.shared"));
    assert!(ptx.contains("st.shared"));
    assert!(ptx.contains("ld.global"));
    assert!(ptx.contains("st.global"));
    assert!(ptx.contains("bar.sync"));
    assert!(ptx.contains("fma.rn.f32"));
}

// =========================================================================
// Consistency Tests
// =========================================================================

#[test]
fn test_kernel_names_are_consistent() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 64, 16);
    let name1 = kernel.name();
    let name2 = kernel.name();
    assert_eq!(name1, name2);
    assert_eq!(name1, "gemm_tiled_unrolled");
}

#[test]
fn test_ptx_emission_structural_consistency() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 64, 16);
    let ptx1 = kernel.emit_ptx();
    let ptx2 = kernel.emit_ptx();

    assert!(ptx1.contains(".entry gemm_tiled_unrolled"));
    assert!(ptx2.contains(".entry gemm_tiled_unrolled"));
    assert!(ptx1.contains(".param .u64 a_ptr"));
    assert!(ptx2.contains(".param .u64 a_ptr"));
    assert_eq!(ptx1.matches("bar.sync").count(), ptx2.matches("bar.sync").count());
    assert_eq!(ptx1.matches("fma.rn.f32").count(), ptx2.matches("fma.rn.f32").count());
    assert_eq!(ptx1.len(), ptx2.len());
}

#[test]
fn test_as_module_for_tiled_unrolled() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 64, 16);
    let module = kernel.as_module();
    let ptx = module.emit();

    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains(".target sm_70"));
    assert!(ptx.contains(".address_size 64"));
    assert!(ptx.contains(".entry gemm_tiled_unrolled"));
}

#[test]
fn test_tiled_gemm_barriers_use_correct_ids() {
    let kernel = GemmKernel::tiled(64, 64, 64, 16);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains("bar.sync 0;"), "tiled GEMM must contain bar.sync 0 (before inner loop)");
    assert!(ptx.contains("bar.sync 1;"), "tiled GEMM must contain bar.sync 1 (after inner loop)");
}
