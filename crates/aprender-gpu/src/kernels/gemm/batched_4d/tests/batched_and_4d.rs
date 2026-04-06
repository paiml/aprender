//! Batched GEMM (Issue #71) and 4D batched GEMM tests:
//! naive, tiled, WMMA, z-dimension, config, barrier safety,
//! boundary conditions, traits, PTX content, shared memory, loops.

use super::*;

// =========================================================================
// Batched GEMM Tests (Issue #71)
// =========================================================================

#[test]
fn test_batched_gemm_naive() {
    let kernel = BatchedGemmKernel::naive(4, 64, 64, 64);
    assert_eq!(kernel.name(), "batched_gemm_naive");
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry batched_gemm_naive"));
    assert!(ptx.contains(".param .u32 batch"));
}

#[test]
fn test_batched_gemm_tiled() {
    let kernel = BatchedGemmKernel::tiled(4, 64, 64, 64, 16);
    assert_eq!(kernel.name(), "batched_gemm_tiled");
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry batched_gemm_tiled"));
    assert!(ptx.contains("bar.sync"));
}

/// WAPR-PERF-011: Test batched WMMA kernel for multi-head attention
#[test]
fn test_batched_gemm_wmma_fp16() {
    let kernel = BatchedGemmKernel::wmma_fp16(6, 94, 64, 64);
    assert_eq!(kernel.name(), "batched_gemm_wmma_fp16");

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry batched_gemm_wmma_fp16"));
    assert!(ptx.contains(".param .u32 batch"));
    assert!(ptx.contains("bar.sync"));
    assert!(ptx.contains("cvta.shared.u64"));
    assert!(ptx.contains("wmma") || ptx.contains("mma"));
}

#[test]
fn test_batched_gemm_uses_z_dimension() {
    let kernel = BatchedGemmKernel::naive(8, 32, 32, 32);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains("%ctaid.z"));
}

#[test]
fn test_batched_gemm_config_default() {
    let config = BatchedGemmConfig::default();
    assert_eq!(config.batch, 1);
    assert_eq!(config.m, 1024);
    assert_eq!(config.n, 1024);
    assert_eq!(config.k, 1024);
    assert_eq!(config.tile_size, 16);
}

#[test]
fn test_batched_4d_gemm() {
    let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
    assert_eq!(kernel.name(), "batched_4d_gemm");
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry batched_4d_gemm"));
    assert!(ptx.contains(".param .u32 batch"));
    assert!(ptx.contains(".param .u32 heads"));
}

#[test]
fn test_batched_4d_gemm_with_tile_size() {
    let kernel = Batched4DGemmKernel::with_tile_size(2, 8, 64, 64, 32, 32);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry batched_4d_gemm"));
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_batched_4d_gemm_config_default() {
    let config = Batched4DGemmConfig::default();
    assert_eq!(config.batch, 1);
    assert_eq!(config.heads, 8);
    assert_eq!(config.m, 512);
    assert_eq!(config.n, 512);
    assert_eq!(config.k, 64);
    assert_eq!(config.tile_size, 16);
}

#[test]
fn test_batched_4d_gemm_uses_batch_head_indexing() {
    let kernel = Batched4DGemmKernel::new(4, 12, 128, 128, 64);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains("%ctaid.z"));
    assert!(ptx.contains("div.") || ptx.contains("rem."));
}

/// PARITY-114: Verify batched GEMM tiled is barrier-safe
#[test]
fn test_barrier_safety_batched_gemm_tiled() {
    let kernel = BatchedGemmKernel::tiled(4, 64, 64, 64, 16);
    let result = kernel.analyze_barrier_safety();
    assert!(result.is_safe);
}

/// PARITY-114: Verify batched 4D GEMM is barrier-safe
#[test]
fn test_barrier_safety_batched_4d_gemm() {
    let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
    let result = kernel.analyze_barrier_safety();
    assert!(result.is_safe);
}

/// Test batched GEMM boundary conditions
#[test]
fn test_batched_gemm_boundary_conditions() {
    let boundary_cases = [
        (1, 17, 17, 17, 16),    // Single batch, non-power-of-2
        (8, 100, 100, 100, 16), // Multiple batches
        (16, 1, 64, 64, 16),    // Single row
    ];

    for (batch, m, n, k, tile) in boundary_cases {
        let kernel = BatchedGemmKernel::tiled(batch, m, n, k, tile);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry"));
        assert!(ptx.contains("bar.sync"));
    }
}

/// Test 4D GEMM boundary conditions
#[test]
fn test_batched_4d_gemm_boundary_conditions() {
    let boundary_cases = [(1, 1, 64, 64, 32), (2, 12, 17, 17, 17), (4, 8, 128, 64, 32)];

    for (batch, heads, m, n, k) in boundary_cases {
        let kernel = Batched4DGemmKernel::new(batch, heads, m, n, k);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry"));
        assert!(ptx.contains("bar.sync"));
    }
}

// =========================================================================
// Additional tests for Batched4DGemmKernel coverage (95%+ target)
// =========================================================================

/// Test Debug trait for Batched4DGemmConfig
#[test]
fn test_batched_4d_gemm_config_debug() {
    let config = Batched4DGemmConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("Batched4DGemmConfig"));
    assert!(debug_str.contains("batch"));
    assert!(debug_str.contains("heads"));
    assert!(debug_str.contains("tile_size"));
}

/// Test Clone trait for Batched4DGemmConfig
#[test]
fn test_batched_4d_gemm_config_clone() {
    let config = Batched4DGemmConfig {
        batch: 4,
        heads: 12,
        m: 256,
        n: 256,
        k: 64,
        tile_size: 32,
    };
    let cloned = config.clone();
    assert_eq!(config.batch, cloned.batch);
    assert_eq!(config.heads, cloned.heads);
    assert_eq!(config.m, cloned.m);
    assert_eq!(config.n, cloned.n);
    assert_eq!(config.k, cloned.k);
    assert_eq!(config.tile_size, cloned.tile_size);
}

/// Test Debug trait for Batched4DGemmKernel
#[test]
fn test_batched_4d_gemm_kernel_debug() {
    let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("Batched4DGemmKernel"));
    assert!(debug_str.contains("config"));
}

/// Test Clone trait for Batched4DGemmKernel
#[test]
fn test_batched_4d_gemm_kernel_clone() {
    let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
    let cloned = kernel.clone();
    assert_eq!(kernel.name(), cloned.name());
    assert_eq!(kernel.config.batch, cloned.config.batch);
    assert_eq!(kernel.config.heads, cloned.config.heads);
    assert_eq!(kernel.config.m, cloned.config.m);
    assert_eq!(kernel.config.n, cloned.config.n);
    assert_eq!(kernel.config.k, cloned.config.k);
}

/// Test as_module() method for Batched4DGemmKernel
#[test]
fn test_batched_4d_gemm_as_module() {
    let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
    let module = kernel.as_module();
    let ptx = module.emit();

    // Verify module structure
    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains(".target sm_70"));
    assert!(ptx.contains(".address_size 64"));
    assert!(ptx.contains(".entry batched_4d_gemm"));
}

/// Test PTX content for 4D batched GEMM
#[test]
fn test_batched_4d_gemm_ptx_content() {
    let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
    let ptx = kernel.emit_ptx();

    // Verify all parameters are present
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
    assert!(ptx.contains(".param .u32 batch"));
    assert!(ptx.contains(".param .u32 heads"));
    assert!(ptx.contains(".param .u32 m"));
    assert!(ptx.contains(".param .u32 n"));
    assert!(ptx.contains(".param .u32 k"));

    // Verify batch/head indexing
    assert!(ptx.contains("%ctaid.z"));
    assert!(ptx.contains("div.u32") || ptx.contains("rem.u32"));
}

/// Test shared memory allocation for 4D batched GEMM
#[test]
fn test_batched_4d_gemm_shared_memory() {
    let kernel = Batched4DGemmKernel::with_tile_size(2, 8, 64, 64, 32, 16);
    let ptx_kernel = kernel.build_ptx();

    // Shared memory = tile_size * tile_size * 4 * 2 (for A and B tiles)
    // 16 * 16 * 4 * 2 = 2048 bytes
    assert_eq!(ptx_kernel.shared_memory_bytes(), 2048);
}

/// Test 4D GEMM with large tile size
#[test]
fn test_batched_4d_gemm_large_tile() {
    let kernel = Batched4DGemmKernel::with_tile_size(1, 4, 128, 128, 64, 32);
    let ptx = kernel.emit_ptx();

    // Verify kernel generates valid PTX
    assert!(ptx.contains(".entry batched_4d_gemm"));
    assert!(ptx.contains("bar.sync"));

    // Check shared memory is larger
    let ptx_kernel = kernel.build_ptx();
    assert_eq!(ptx_kernel.shared_memory_bytes(), 32 * 32 * 4 * 2);
}

/// Test 4D GEMM with minimum dimensions
#[test]
fn test_batched_4d_gemm_minimum_dims() {
    let kernel = Batched4DGemmKernel::new(1, 1, 1, 1, 1);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry batched_4d_gemm"));
    assert!(ptx.contains("bar.sync"));
}

/// Test 4D GEMM loop structure
#[test]
fn test_batched_4d_gemm_loop_structure() {
    let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
    let ptx = kernel.emit_ptx();

    // Verify tile loop
    assert!(ptx.contains("tile_loop:"));
    assert!(ptx.contains("tile_loop_end:"));

    // Verify inner k loop
    assert!(ptx.contains("inner_k_loop:"));
    assert!(ptx.contains("inner_k_end:"));

    // Verify exit label
    assert!(ptx.contains("exit:"));
}

/// Test 4D GEMM FMA operations
#[test]
fn test_batched_4d_gemm_fma_operations() {
    let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
    let ptx = kernel.emit_ptx();

    // Verify FMA is used for accumulation
    assert!(ptx.contains("fma.rn.f32"));

    // Verify shared memory operations
    assert!(ptx.contains("ld.shared.f32"));
    assert!(ptx.contains("st.shared.f32"));

    // Verify global memory operations
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("st.global.f32"));
}

/// Test 4D GEMM skip labels for bounds checking
#[test]
fn test_batched_4d_gemm_skip_labels() {
    let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
    let ptx = kernel.emit_ptx();

    // Verify skip labels for A and B loading
    assert!(ptx.contains("skip_a_load:"));
    assert!(ptx.contains("skip_b_load:"));
}

/// Test 4D GEMM with varying head counts
#[test]
fn test_batched_4d_gemm_varying_heads() {
    let head_counts = [1, 2, 4, 8, 12, 16, 32];

    for heads in head_counts {
        let kernel = Batched4DGemmKernel::new(2, heads, 64, 64, 32);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry batched_4d_gemm"));
        assert!(ptx.contains("bar.sync"));

        // Config should match
        assert_eq!(kernel.config.heads, heads);
    }
}

/// Test 4D GEMM with varying batch sizes
#[test]
fn test_batched_4d_gemm_varying_batches() {
    let batch_sizes = [1, 2, 4, 8, 16, 32];

    for batch in batch_sizes {
        let kernel = Batched4DGemmKernel::new(batch, 8, 64, 64, 32);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry batched_4d_gemm"));
        assert_eq!(kernel.config.batch, batch);
    }
}

/// Test 4D GEMM barrier safety passes
#[test]
fn test_batched_4d_gemm_barrier_safety_result() {
    let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
    let result = kernel.analyze_barrier_safety();

    assert!(result.is_safe);
    assert!(result.violations.is_empty());
    assert!(result.barrier_count > 0);
}

/// Test 4D GEMM with non-power-of-2 dimensions
#[test]
fn test_batched_4d_gemm_non_power_of_2() {
    let cases = [
        (3, 7, 33, 33, 17),
        (5, 11, 100, 100, 50),
        (2, 6, 94, 64, 64), // Typical attention pattern
    ];

    for (batch, heads, m, n, k) in cases {
        let kernel = Batched4DGemmKernel::new(batch, heads, m, n, k);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry batched_4d_gemm"));
        assert!(ptx.contains("bar.sync"));
    }
}

/// Test name() method returns correct value
#[test]
fn test_batched_4d_gemm_name() {
    let kernel = Batched4DGemmKernel::new(1, 1, 64, 64, 64);
    assert_eq!(kernel.name(), "batched_4d_gemm");
}

/// Test config default values match documentation
#[test]
fn test_batched_4d_gemm_config_default_values() {
    let config = Batched4DGemmConfig::default();

    // Verify documented defaults
    assert_eq!(config.batch, 1, "Default batch should be 1");
    assert_eq!(config.heads, 8, "Default heads should be 8");
    assert_eq!(config.m, 512, "Default m should be 512");
    assert_eq!(config.n, 512, "Default n should be 512");
    assert_eq!(config.k, 64, "Default k should be 64");
    assert_eq!(config.tile_size, 16, "Default tile_size should be 16");
}
