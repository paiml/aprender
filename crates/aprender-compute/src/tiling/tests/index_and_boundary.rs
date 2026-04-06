use super::super::*;

// ========================================================================
// Index calculator tests
// ========================================================================

#[test]
fn test_index_calculator_macro_offset() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config.clone(), 1024, 1024, 1024);

    let (row, col) = calc.macro_tile_offset(0);
    assert_eq!((row, col), (0, 0));

    let (_row, col) = calc.macro_tile_offset(1);
    assert_eq!(col, config.macro_tile.n);
}

#[test]
fn test_index_calculator_boundary() {
    let config = TilingConfig::cpu_avx2_matmul();

    // With 512×512 problem and 256×256 tiles, first tile is NOT a boundary
    let calc_large = TcbIndexCalculator::new(config.clone(), 512, 512, 256);
    assert!(!calc_large.is_boundary_tile(0));

    // With 100×100 problem and 256×256 tiles, first (only) tile IS a boundary
    let calc_small = TcbIndexCalculator::new(config, 100, 100, 256);
    assert!(calc_small.is_boundary_tile(0));

    // Actual dimensions should be clamped to problem size
    let (actual_m, actual_n) = calc_small.actual_tile_dims(0);
    assert_eq!(actual_m, 100);
    assert_eq!(actual_n, 100);
}

#[test]
fn test_pack_a_index() {
    // mr=4, kc=256, panel 0
    let idx = pack_a_index(0, 0, 4, 256, 64);
    assert_eq!(idx, 0);

    // Second element in first panel
    let idx = pack_a_index(1, 0, 4, 256, 64);
    assert_eq!(idx, 1);

    // First element, second k
    let idx = pack_a_index(0, 1, 4, 256, 64);
    assert_eq!(idx, 4);
}

#[test]
fn test_swizzle_index() {
    // XOR swizzling should avoid bank conflicts
    let idx0 = swizzle_index(0);
    let idx32 = swizzle_index(32);
    // These would conflict without swizzling (both bank 0)
    // With swizzling: 0 ^ 0 = 0, 32 ^ 1 = 33
    assert_ne!(idx0 % 32, idx32 % 32);
}

#[test]
fn test_optimal_prefetch_distance() {
    let geom = TcbGeometry::new(4, 8, 64);
    let dist = optimal_prefetch_distance(&geom, TcbLevel::Midi);
    assert!(dist >= 1);
}

// ========================================================================
// Boundary handling tests (F321-F340)
// ========================================================================

// F321: Odd-Sized Matrix Handling
#[test]
fn test_odd_sized_matrices() {
    let config = TilingConfig::cpu_avx2_matmul();

    // Test various odd sizes
    for (m, n, k) in [(127, 255, 513), (1, 1, 1), (7, 13, 31)] {
        let calc = TcbIndexCalculator::new(config.clone(), m, n, k);
        let num_tiles = calc.num_k_blocks();
        assert!(num_tiles >= 1);
    }
}

// F322: Zero-Padding Efficiency
#[test]
fn test_tile_count_calculation() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config.clone(), 1024, 1024, 1024);

    let num_macro = calc.config.num_macro_tiles(1024, 1024);
    let num_midi = calc.config.midi_tiles_per_macro();
    let num_micro = calc.config.micro_tiles_per_midi();

    assert!(num_macro > 0);
    assert!(num_midi > 0);
    assert!(num_micro > 0);
}

// F323: Single-element matrices
#[test]
fn test_single_element_matrix() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config, 1, 1, 256);

    assert!(calc.is_boundary_tile(0));
    let (actual_m, actual_n) = calc.actual_tile_dims(0);
    assert_eq!(actual_m, 1);
    assert_eq!(actual_n, 1);
}

// F324: Prime-sized matrices (no clean tiling)
#[test]
fn test_prime_sized_matrices() {
    let config = TilingConfig::cpu_avx2_matmul();

    // Prime sizes: 127, 251, 509 (all < macro_tile.m which is 256)
    for size in [127, 251] {
        let calc = TcbIndexCalculator::new(config.clone(), size, size, 256);
        let num_tiles = config.num_macro_tiles(size, size);
        assert!(num_tiles >= 1);

        // Tiles smaller than macro size are boundary tiles
        assert!(calc.is_boundary_tile(0));
    }

    // 509 > 256, so first tile is NOT a boundary, but second tile IS
    let calc = TcbIndexCalculator::new(config.clone(), 509, 509, 256);
    // First tile (0,0 to 255,255) is not boundary for 509×509
    assert!(!calc.is_boundary_tile(0));
    // Second tile (0,256 to 255,508) IS boundary (509-256=253 < 256)
    assert!(calc.is_boundary_tile(1));
}

// F328: Tile offset at boundaries
#[test]
fn test_tile_offset_boundaries() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config.clone(), 1000, 1000, 256);

    // Last tile index
    let num_tiles = config.num_macro_tiles(1000, 1000);
    let last_idx = num_tiles - 1;

    let (row, col) = calc.macro_tile_offset(last_idx);
    // Should be within bounds
    assert!(row < 1000 + config.macro_tile.m);
    assert!(col < 1000 + config.macro_tile.n);
}

// F329: Index calculator consistency
#[test]
fn test_index_calculator_consistency() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config.clone(), 512, 512, 256);

    // Macro offset for tile 0 should be (0, 0)
    let (r0, c0) = calc.macro_tile_offset(0);
    assert_eq!((r0, c0), (0, 0));

    // Linear offset should match
    let linear = calc.block_to_linear_offset(0, 512);
    assert_eq!(linear, 0);

    // A and B offsets at k_block=0 should also be 0
    let a_off = calc.a_offset(0, 0);
    let b_off = calc.b_offset(0, 0);
    assert_eq!(a_off, 0);
    assert_eq!(b_off, 0);
}

// F339: TcbIndexCalculator midi/micro offsets
#[test]
fn test_index_calculator_midi_offset() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config, 1024, 1024, 1024);

    let (row, col) = calc.midi_tile_offset(0);
    assert_eq!((row, col), (0, 0));

    let (row1, col1) = calc.midi_tile_offset(1);
    // Second midi tile should be one midi_tile.n to the right
    assert_eq!(row1, 0);
    assert!(col1 > 0);
}

#[test]
fn test_index_calculator_micro_offset() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config, 1024, 1024, 1024);

    let (row, col) = calc.micro_tile_offset(0);
    assert_eq!((row, col), (0, 0));

    let (row1, col1) = calc.micro_tile_offset(1);
    assert_eq!(row1, 0);
    assert!(col1 > 0);
}

// F340: pack_b_index
#[test]
fn test_pack_b_index() {
    // nr=8, kc=64, panel 0
    let idx = pack_b_index(0, 0, 8, 64, 64);
    assert_eq!(idx, 0);

    // Second element in first panel (col 1)
    let idx = pack_b_index(0, 1, 8, 64, 64);
    assert_eq!(idx, 1);

    // First element, second row
    let idx = pack_b_index(1, 0, 8, 64, 64);
    assert_eq!(idx, 8);

    // Second panel (col 8)
    let idx = pack_b_index(0, 8, 8, 64, 64);
    // panel 1 * 64 * 8 + 0 * 8 + 0 = 512
    assert_eq!(idx, 512);
}

// F351: Index calculator k_blocks
#[test]
fn test_index_calculator_k_blocks() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config.clone(), 512, 512, 1024);

    // 1024 / 256 = 4 K blocks
    assert_eq!(calc.num_k_blocks(), 4);

    // Non-divisible case
    let calc2 = TcbIndexCalculator::new(config, 512, 512, 300);
    // ceil(300 / 256) = 2
    assert_eq!(calc2.num_k_blocks(), 2);
}

// F352: A and B offset calculations
#[test]
fn test_ab_offset_calculations() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config.clone(), 512, 512, 512);

    // A offset: row * problem_k + col
    let a_off = calc.a_offset(1, 0); // macro_row=1, k_block=0
    assert_eq!(a_off, (config.macro_tile.m * 512) as usize);

    // B offset: row * problem_n + col
    let b_off = calc.b_offset(0, 1); // k_block=0, macro_col=1
    assert_eq!(b_off, config.macro_tile.n as usize);
}

// F354: Prefetch with different levels
#[test]
fn test_prefetch_all_levels() {
    let geom = TcbGeometry::new(4, 8, 64);

    let dist_micro = optimal_prefetch_distance(&geom, TcbLevel::Micro);
    let dist_midi = optimal_prefetch_distance(&geom, TcbLevel::Midi);
    let dist_macro = optimal_prefetch_distance(&geom, TcbLevel::Macro);

    // Macro should have larger distance (higher latency)
    assert!(dist_macro >= dist_midi);
    assert!(dist_midi >= dist_micro);
    // All should be at least 1
    assert!(dist_micro >= 1);
}

// F355: PrefetchLocality Debug
#[test]
fn test_prefetch_locality_debug() {
    let loc = PrefetchLocality::T0;
    let debug = format!("{:?}", loc);
    assert!(debug.contains("T0"));

    let loc2 = PrefetchLocality::NonTemporal;
    let debug2 = format!("{:?}", loc2);
    assert!(debug2.contains("NonTemporal"));
}
