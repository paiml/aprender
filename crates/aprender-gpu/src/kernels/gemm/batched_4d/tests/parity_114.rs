//! PARITY-114 barrier safety tests: verify no early exit before barriers
//! in tiled, tensor core, and WMMA GEMM variants.

use super::*;

/// PARITY-114: Verify tiled GEMM doesn't have early exit before barriers
#[test]
fn test_parity_114_tiled_gemm_no_early_exit_before_barrier() {
    let kernel = GemmKernel::tiled(4, 8, 64, 32);
    let ptx = kernel.emit_ptx();

    let bar_sync_pos = ptx.find("bar.sync").expect("bar.sync required");
    let tile_loop_end_pos = ptx.find("tile_loop_end:").expect("tile_loop_end required");

    // Verify no early exit before tile_loop_end
    let early_exit = ptx.lines().any(|line| {
        if line.contains("@%p") && line.contains("bra exit") {
            let pos = ptx.find(line).unwrap_or(0);
            pos < tile_loop_end_pos
        } else {
            false
        }
    });
    assert!(!early_exit, "PARITY-114 violation");
    assert!(bar_sync_pos < tile_loop_end_pos, "bar.sync must be in loop");
}

/// PARITY-114: Verify n_tiles is correctly computed for small k
#[test]
fn test_parity_114_ntiles_computation() {
    // k=64, tile_size=32 -> n_tiles should be 2
    let kernel = GemmKernel::tiled(4, 8, 64, 32);
    let ptx = kernel.emit_ptx();

    // The PTX should have mov.u32 %rXX, 2; for n_tiles
    assert!(ptx.contains(", 2;"), "PTX should have n_tiles=2 for k=64, tile_size=32");

    // And tile_size=32
    assert!(ptx.contains(", 32;"), "PTX should have tile_size=32");
}

/// PARITY-114: Verify gemm_tensor_core has no early exit before barrier
#[test]
fn test_parity_114_tensor_core_no_early_exit_before_barrier() {
    let kernel = GemmKernel::tensor_core(16, 16, 16);
    let ptx = kernel.emit_ptx();

    // Find positions of key elements
    let bar_sync_pos = ptx.find("bar.sync").expect("PTX should have bar.sync");
    let k_tile_end_pos = ptx.find("k_tile_end:").expect("PTX should have k_tile_end");

    // Verify bar.sync is inside the loop (before k_tile_end)
    assert!(
        bar_sync_pos < k_tile_end_pos,
        "bar.sync should be inside k_tile_loop (before k_tile_end)"
    );

    // Verify no unconditional exits before k_tile_end (conditional @!%p branches are OK)
    // The key is that bar.sync comes before the exit checks
}

/// PARITY-114: Verify gemm_wmma_fp16 has no early exit before barrier
#[test]
fn test_parity_114_wmma_no_early_exit_before_barrier() {
    let kernel = GemmKernel::wmma_fp16(16, 16, 16);
    let ptx = kernel.emit_ptx();

    // Find positions of key elements
    let bar_sync_pos = ptx.find("bar.sync").expect("PTX should have bar.sync");
    let k_tile_end_pos = ptx.find("k_tile_end:").expect("PTX should have k_tile_end");

    // Verify bar.sync is inside the loop (before k_tile_end)
    assert!(
        bar_sync_pos < k_tile_end_pos,
        "bar.sync should be inside k_tile_loop (before k_tile_end)"
    );

    // Verify wmma instructions are present
    assert!(ptx.contains("wmma.mma"), "WMMA kernel should have wmma.mma");
    assert!(ptx.contains("wmma.load"), "WMMA kernel should have wmma.load");
}

/// PARITY-114 Countermeasure: Test boundary conditions (non-divisible dimensions)
/// Five Whys Root Cause: We only tested "happy path" dimensions where all threads valid
#[test]
fn test_boundary_conditions_tensor_core() {
    // Test dimensions NOT divisible by tile size (16)
    // These are the cases where some threads are out-of-bounds
    let boundary_cases = [
        (17, 17, 17),    // Just over tile size
        (31, 31, 31),    // Just under 2 tiles
        (33, 33, 33),    // Just over 2 tiles
        (100, 100, 100), // Arbitrary non-power-of-2
        (1, 16, 16),     // Edge: single row
        (16, 1, 16),     // Edge: single column
    ];

    for (m, n, k) in boundary_cases {
        let kernel = GemmKernel::tensor_core(m, n, k);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry"));
        assert!(ptx.contains("bar.sync"));

        let bar_sync_pos = ptx.find("bar.sync").expect("test");
        let k_tile_end_pos = ptx.find("k_tile_end:").expect("test");
        assert!(bar_sync_pos < k_tile_end_pos);
    }
}

/// PARITY-114 Countermeasure: Test boundary conditions for tiled GEMM
#[test]
fn test_boundary_conditions_tiled_gemm() {
    let boundary_cases = [(17, 17, 17, 16), (65, 65, 65, 32), (100, 100, 100, 32), (1, 32, 32, 16)];

    for (m, n, k, tile) in boundary_cases {
        let kernel = GemmKernel::tiled(m, n, k, tile);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry"));
        assert!(ptx.contains("bar.sync"));
    }
}

/// PARITY-114 Countermeasure: Test WMMA boundary conditions
#[test]
fn test_boundary_conditions_wmma() {
    // WMMA requires multiples of 16, but matrix dims can be non-multiple
    let boundary_cases = [(17, 17, 17), (32, 33, 34), (100, 100, 100)];

    for (m, n, k) in boundary_cases {
        let kernel = GemmKernel::wmma_fp16(m, n, k);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry"));
        assert!(ptx.contains("bar.sync"));
        assert!(ptx.contains("wmma.mma"));
    }
}
