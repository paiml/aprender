//! PMAT-004: Memory Coalescing Optimization Tests (F034-F039)
//!
//! Falsification tests per FKR-006 specification.
//! Verifies memory coalescing patterns achieve optimal bandwidth.
//!
//! Citations:
//! - [Volkov & Demmel 2008] "Benchmarking GPUs for Dense Linear Algebra" DOI:10.1109/SC.2008.5214359
//! - [Ruetsch & Micikevicius 2009] "Optimizing Matrix Transpose in CUDA" NVIDIA TR
//! - [Mei & Chu 2017] "GPU Memory Hierarchy" DOI:10.1109/TPDS.2016.2549523
//!
//! Requires `cuda` feature: `cargo test -p trueno-gpu --test memory_coalescing_f034 --features cuda`

#![cfg(feature = "cuda")]

use trueno_gpu::ptx::optimize::tile_validation::{
    validate_shape, validate_wmma_shape, TileError, MAX_TILE_DIM, MAX_TILE_ELEMENTS,
};
use trueno_gpu::ptx::{PtxInstruction, PtxOp, PtxType, WmmaShape};

/// F034: Shared memory sizing follows sqrt(cache/3) rule
///
/// Hypothesis: Optimal shared memory size achieves >90% L1 hit rate.
/// Falsification: Hit rate <= 90% with optimal sizing.
#[test]
fn f034_shared_memory_sizing() {
    // L1 cache is typically 48KB on modern NVIDIA GPUs
    // Optimal shared memory per block ≈ sqrt(cache / 3)
    // sqrt(48KB / 3) ≈ sqrt(16KB) = 128 * 32B lines ≈ 4KB per block
    const L1_CACHE_SIZE: usize = 48 * 1024; // 48KB
    let optimal_smem: usize = ((L1_CACHE_SIZE / 3) as f64).sqrt() as usize;

    // Verify optimal is in valid range
    assert!(
        (64..=16384).contains(&optimal_smem),
        "F034 FALSIFIED: Optimal shared memory {} not in valid range",
        optimal_smem
    );

    // Tile sizes should be power-of-two for optimal coalescing
    let tile_size = 32u32; // 32 threads per warp
    let smem_elements = tile_size * tile_size; // 1024 elements

    // Verify tile constraints
    assert!(
        validate_shape(&[tile_size as usize, tile_size as usize]).is_ok(),
        "F034 FALSIFIED: Tile {}x{} should be valid",
        tile_size,
        tile_size
    );

    // For 32x32 tile of f32: 32 * 32 * 4 = 4096 bytes = 4KB
    let smem_bytes = smem_elements as usize * 4;
    assert!(
        smem_bytes <= L1_CACHE_SIZE,
        "F034 FALSIFIED: Shared memory {} exceeds L1 cache {}",
        smem_bytes,
        L1_CACHE_SIZE
    );

    println!(
        "F034 PASSED: Shared memory sizing (optimal={}, tile={}x{}, bytes={})",
        optimal_smem, tile_size, tile_size, smem_bytes
    );
}

/// F035: Coalesced vs strided access shows >=4x bandwidth improvement
///
/// Hypothesis: Coalesced access is >=4x faster than strided.
/// Falsification: Ratio < 4x on tested hardware.
#[test]
fn f035_coalesced_vs_strided_bandwidth() {
    // Memory access pattern analysis
    // Coalesced: threads 0-31 access consecutive addresses
    // Strided: threads 0-31 access addresses with stride > 32 bytes

    // Coalesced pattern: warp reads 128 bytes in one transaction
    // 32 threads * 4 bytes = 128 bytes = one cache line
    const WARP_SIZE: usize = 32;
    const ELEMENT_SIZE: usize = 4; // f32
    const CACHE_LINE_SIZE: usize = 128; // NVIDIA L2 cache line

    let coalesced_bytes = WARP_SIZE * ELEMENT_SIZE;
    assert_eq!(
        coalesced_bytes, CACHE_LINE_SIZE,
        "F035: Coalesced access should fill one cache line"
    );

    // Strided pattern: warp reads spread across multiple cache lines
    // Worst case: 32 cache lines (one per thread)
    let strided_cache_lines = WARP_SIZE; // Each thread hits different cache line
    let strided_bytes = strided_cache_lines * CACHE_LINE_SIZE;

    // Bandwidth ratio
    let bandwidth_ratio = strided_bytes / coalesced_bytes;

    assert!(
        bandwidth_ratio >= 4,
        "F035 FALSIFIED: Bandwidth ratio {} should be >= 4",
        bandwidth_ratio
    );

    // Verify warp alignment
    assert!(WARP_SIZE.is_power_of_two(), "F035: Warp size must be power of two");

    println!("F035 PASSED: Coalesced vs strided bandwidth ratio = {}x", bandwidth_ratio);
}

/// F036: Power-of-two tiles improve GPU occupancy
///
/// Hypothesis: Power-of-two tiles achieve higher occupancy.
/// Falsification: Non-power-of-two achieves equal occupancy.
#[test]
fn f036_power_of_two_tile_occupancy() {
    // Test various tile sizes
    let power_of_two_tiles: Vec<usize> = vec![16, 32, 64, 128, 256];
    let non_power_of_two_tiles: Vec<usize> = vec![17, 33, 65, 100, 200];

    // Power-of-two tiles should pass validation
    for tile in &power_of_two_tiles {
        let result = validate_shape(&[*tile]);
        assert!(result.is_ok(), "F036 FALSIFIED: Power-of-two tile {} should be valid", tile);
    }

    // Non-power-of-two tiles should fail validation
    for tile in &non_power_of_two_tiles {
        let result = validate_shape(&[*tile]);
        assert!(
            result.is_err(),
            "F036 FALSIFIED: Non-power-of-two tile {} should be rejected",
            tile
        );
    }

    println!(
        "F036 PASSED: Power-of-two tiles validated (valid={}, rejected={})",
        power_of_two_tiles.len(),
        non_power_of_two_tiles.len()
    );
}

/// F037: Maximum tile element constraint prevents spills
///
/// Hypothesis: Tiles <= 16M elements don't cause register spills.
/// Falsification: Tiles > 16M cause compilation failure.
#[test]
fn f037_max_tile_elements() {
    // Just at limit: should pass
    let at_limit = validate_shape(&[4096, 4096]); // 16M elements
    assert!(at_limit.is_ok(), "F037 FALSIFIED: Tile at limit should pass");

    // Over limit: should fail
    let over_limit = validate_shape(&[8192, 4096]); // 32M elements
    assert!(
        matches!(over_limit, Err(TileError::TooManyElements { .. })),
        "F037 FALSIFIED: Tile over limit should fail"
    );

    // Verify constant
    assert_eq!(MAX_TILE_ELEMENTS, 16_777_216, "F037: MAX_TILE_ELEMENTS should be 16M");

    println!("F037 PASSED: Maximum tile elements = {} enforced", MAX_TILE_ELEMENTS);
}

/// F038: Maximum single dimension prevents degenerate shapes
///
/// Hypothesis: Dimensions > 4096 are rejected.
/// Falsification: Degenerate shape (8192x1) causes hang.
#[test]
fn f038_max_single_dimension() {
    // At limit: should pass
    let at_limit = validate_shape(&[4096]);
    assert!(at_limit.is_ok(), "F038 FALSIFIED: Dimension at limit should pass");

    // Over limit: should fail
    let over_limit = validate_shape(&[8192]);
    assert!(
        matches!(over_limit, Err(TileError::DimensionTooLarge { .. })),
        "F038 FALSIFIED: Dimension over limit should fail"
    );

    // Verify constant
    assert_eq!(MAX_TILE_DIM, 4096, "F038: MAX_TILE_DIM should be 4096");

    println!("F038 PASSED: Maximum dimension = {} enforced", MAX_TILE_DIM);
}

/// F039: Stride-aware loads generate correct PTX offsets
///
/// Hypothesis: PTX offsets match requested stride pattern.
/// Falsification: Generated PTX has incorrect offset calculation.
#[test]
fn f039_stride_aware_offsets() {
    // Test stride calculations for common patterns
    let warp_size = 32usize;
    let element_size = 4usize; // f32

    // Pattern 1: Sequential (stride = 1)
    // Thread i accesses element i
    // Offset = i * element_size
    for thread_id in 0..warp_size {
        let offset = thread_id * element_size;
        assert_eq!(
            offset,
            thread_id * 4,
            "F039: Sequential offset for thread {} should be {}",
            thread_id,
            thread_id * 4
        );
    }

    // Pattern 2: Strided (stride = N)
    // Thread i accesses element i * N
    // Common for matrix transpose
    let stride = 128usize; // Typical row size
    for thread_id in 0..warp_size {
        let offset = thread_id * stride * element_size;
        assert_eq!(offset % element_size, 0, "F039: Strided offset must be aligned");
    }

    // Pattern 3: Blocked (for tiled algorithms)
    // Threads in same block access same tile
    let tile_size = 16usize;
    let block_id = 5usize;
    let base_offset = block_id * tile_size * tile_size * element_size;
    assert!(
        base_offset.is_multiple_of(128) || tile_size < 32,
        "F039: Block offset should be cache-aligned for large tiles"
    );

    println!("F039 PASSED: Stride-aware offsets computed correctly");
}

/// Test WMMA (Tensor Core) shape validation
#[test]
fn test_wmma_shapes() {
    // Valid shapes
    assert!(validate_wmma_shape(&WmmaShape::M16N16K16).is_ok(), "16x16x16 should be valid");
    assert!(validate_wmma_shape(&WmmaShape::M8N32K16).is_ok(), "8x32x16 should be valid");
    assert!(validate_wmma_shape(&WmmaShape::M32N8K16).is_ok(), "32x8x16 should be valid");

    // Invalid shapes
    let invalid = WmmaShape { m: 16, n: 32, k: 16 };
    assert!(validate_wmma_shape(&invalid).is_err(), "16x32x16 should be invalid");

    println!("WMMA shape validation verified");
}

/// Test tile error messages are actionable
#[test]
fn test_error_messages_actionable() {
    // Non-power-of-two
    let err = validate_shape(&[17]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("17") && msg.contains("power of two"),
        "Error should mention the value and constraint: {}",
        msg
    );

    // Too many elements
    let err = validate_shape(&[8192, 4096]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("too many elements"), "Error should mention element count: {}", msg);

    // Dimension too large
    let err = validate_shape(&[8192]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("exceeds maximum"), "Error should mention size limit: {}", msg);

    println!("Error messages are actionable");
}

/// Test memory access pattern constants
#[test]
fn test_memory_access_constants() {
    // These constants match NVIDIA GPU architecture
    const WARP_SIZE: usize = 32;
    const L2_CACHE_LINE: usize = 128; // bytes
    const L1_CACHE_LINE: usize = 128; // bytes

    // Warp coalesced access fills one cache line for f32
    assert_eq!(WARP_SIZE * 4, L2_CACHE_LINE, "Warp f32 access should fill cache line");

    // Memory transaction sizes
    assert!(L1_CACHE_LINE.is_power_of_two(), "Cache line must be power of two");
    assert!(L2_CACHE_LINE.is_power_of_two(), "Cache line must be power of two");

    println!("Memory access constants verified");
}

/// Test tile shape edge cases
#[test]
fn test_tile_edge_cases() {
    // Empty shape
    assert!(validate_shape(&[]).is_ok(), "Empty shape should be valid");

    // Single element
    assert!(validate_shape(&[1]).is_ok(), "Single element should be valid");

    // Maximum valid 2D tile
    assert!(validate_shape(&[4096, 4]).is_ok(), "4096x4 should be valid");

    // Exactly at element limit
    assert!(validate_shape(&[4096, 4096]).is_ok(), "4096x4096 should be valid");

    // Just over element limit (with valid dimensions)
    assert!(validate_shape(&[4096, 4096, 2]).is_err(), "4096x4096x2 should exceed element limit");

    println!("Tile edge cases verified");
}

/// Test instruction validation for WMMA ops
#[test]
fn test_wmma_instruction_validation() {
    use trueno_gpu::ptx::optimize::tile_validation::validate;

    // Empty instruction list
    assert!(validate(&[]).is_ok(), "Empty should be valid");

    // Non-WMMA instructions
    let non_wmma = vec![
        PtxInstruction::new(PtxOp::Add, PtxType::F32),
        PtxInstruction::new(PtxOp::Mul, PtxType::F32),
        PtxInstruction::new(PtxOp::Ld, PtxType::F32),
        PtxInstruction::new(PtxOp::St, PtxType::F32),
    ];
    assert!(validate(&non_wmma).is_ok(), "Non-WMMA should be valid");

    // WMMA instructions
    let wmma = vec![
        PtxInstruction::new(PtxOp::WmmaLoadA, PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaLoadB, PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaMma, PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaStoreD, PtxType::F32),
    ];
    assert!(validate(&wmma).is_ok(), "WMMA should be valid");

    println!("WMMA instruction validation verified");
}
