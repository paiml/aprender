#![allow(clippy::disallowed_methods)]
//! Tiling Compute Blocks (TCB) Demo
//!
//! Demonstrates hierarchical cache-blocked tiling for high-performance GEMM
//! and Q4_K quantized inference, based on TILING-SPEC-001.
//!
//! Key concepts:
//! - Three-level tiling: Macro (L3), Midi (L2), Micro (Registers)
//! - Q4_K superblock alignment (256 elements = 144 bytes)
//! - Goto-style panel packing for cache efficiency
//!
//! Run: cargo run --example tiling_demo
//!
//! References:
//! - Lam et al. (1991): "The Cache Performance and Optimizations of Blocked Algorithms"
//! - Goto & van de Geijn (2008): "Anatomy of High-Performance Matrix Multiplication"
//! - Volkov (2010): "Better Performance at Lower Occupancy"

use trueno::tiling::{
    optimal_prefetch_distance, pack_a_index, pack_b_index, swizzle_index, TcbGeometry,
    TcbIndexCalculator, TcbLevel, TiledQ4KMatvec, TilingConfig, Q4K_SUPERBLOCK_BYTES,
    Q4K_SUPERBLOCK_SIZE,
};

fn main() {
    println!("=== Trueno Tiling Compute Blocks (TCB) Demo ===\n");

    // =========================================================================
    // 1. TcbGeometry: The fundamental tile unit
    // =========================================================================
    println!("1. TcbGeometry - Tile dimensions and arithmetic intensity");
    println!("   ─────────────────────────────────────────────────────────");

    let micro_tile = TcbGeometry::new(4, 8, 256);
    println!("   Micro-tile: {} (M×N×K with alignment)", micro_tile);
    println!("   Total elements: {}", micro_tile.total_elements());
    println!("   Total FLOPs: {}", micro_tile.total_flops());
    println!("   Arithmetic intensity: {:.2} FLOP/byte", micro_tile.arithmetic_intensity());
    println!("   Q4_K aligned: {}", micro_tile.is_q4k_aligned());

    // AVX-512 aligned geometry
    let avx512_tile = TcbGeometry::with_alignment(4, 16, 128, 64);
    println!("\n   AVX-512 tile: {}", avx512_tile);
    println!("   A tile bytes: {} KB", avx512_tile.a_tile_bytes() / 1024);
    println!("   B tile bytes: {} KB", avx512_tile.b_tile_bytes() / 1024);
    println!("   Fits in L1 (32KB): {}", avx512_tile.fits_in_cache(32 * 1024));

    // =========================================================================
    // 2. TilingConfig: Complete hierarchical configuration
    // =========================================================================
    println!("\n2. TilingConfig - Hierarchical cache-blocked tiling");
    println!("   ──────────────────────────────────────────────────");

    // GPU Q4_K MatVec configuration
    let gpu_config = TilingConfig::gpu_q4k_matvec();
    print_config(&gpu_config);

    // CPU AVX-512 MatMul configuration
    let avx512_config = TilingConfig::cpu_avx512_matmul();
    print_config(&avx512_config);

    // CPU AVX2 Q4K MatVec configuration
    let avx2_q4k = TilingConfig::cpu_avx2_q4k_matvec();
    print_config(&avx2_q4k);

    // =========================================================================
    // 3. TcbIndexCalculator: Hierarchical index mapping
    // =========================================================================
    println!("\n3. TcbIndexCalculator - Tile-to-memory mapping");
    println!("   ─────────────────────────────────────────────");

    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config.clone(), 1024, 1024, 1024);

    println!("   Problem: 1024×1024×1024 GEMM");
    println!("   Macro tiles: {}", config.num_macro_tiles(1024, 1024));
    println!("   Midi tiles per macro: {}", config.midi_tiles_per_macro());
    println!("   Micro tiles per midi: {}", config.micro_tiles_per_midi());
    println!("   K blocks: {}", calc.num_k_blocks());

    // Show tile offsets
    println!("\n   Tile offset mapping:");
    for tile_idx in 0..4 {
        let (row, col) = calc.macro_tile_offset(tile_idx);
        let is_boundary = calc.is_boundary_tile(tile_idx);
        println!("     Tile {}: offset ({}, {}), boundary: {}", tile_idx, row, col, is_boundary);
    }

    // Boundary handling
    let small_calc = TcbIndexCalculator::new(config.clone(), 100, 100, 256);
    let (actual_m, actual_n) = small_calc.actual_tile_dims(0);
    println!("\n   Boundary handling (100×100 problem):");
    println!("     First tile is boundary: {}", small_calc.is_boundary_tile(0));
    println!("     Actual dimensions: {}×{}", actual_m, actual_n);

    // =========================================================================
    // 4. Memory layout helpers
    // =========================================================================
    println!("\n4. Memory Layout - Panel packing and swizzling");
    println!("   ───────────────────────────────────────────────");

    // Goto algorithm panel packing
    println!("   Panel-major packing (Goto algorithm):");
    println!("     pack_a_index(row=0, col=0, mr=4, kc=256) = {}", pack_a_index(0, 0, 4, 256, 64));
    println!("     pack_a_index(row=1, col=0, mr=4, kc=256) = {}", pack_a_index(1, 0, 4, 256, 64));
    println!("     pack_a_index(row=0, col=1, mr=4, kc=256) = {}", pack_a_index(0, 1, 4, 256, 64));
    println!(
        "     pack_a_index(row=4, col=0, mr=4, kc=256) = {} (next panel)",
        pack_a_index(4, 0, 4, 256, 64)
    );

    println!("\n   Panel B packing:");
    println!("     pack_b_index(row=0, col=0, nr=8, kc=64) = {}", pack_b_index(0, 0, 8, 64, 64));
    println!("     pack_b_index(row=1, col=0, nr=8, kc=64) = {}", pack_b_index(1, 0, 8, 64, 64));
    println!(
        "     pack_b_index(row=0, col=8, nr=8, kc=64) = {} (next panel)",
        pack_b_index(0, 8, 8, 64, 64)
    );

    // Shared memory swizzling
    println!("\n   XOR swizzling for bank conflict avoidance:");
    for idx in [0, 32, 64, 96] {
        let swizzled = swizzle_index(idx);
        println!("     idx {} → swizzled {} (bank {})", idx, swizzled, swizzled % 32);
    }

    // =========================================================================
    // 5. Q4_K Tiled MatVec
    // =========================================================================
    println!("\n5. TiledQ4KMatvec - Quantized inference tiling");
    println!("   ───────────────────────────────────────────────");

    println!("   Q4_K format constants:");
    println!("     Superblock size: {} elements", Q4K_SUPERBLOCK_SIZE);
    println!("     Superblock bytes: {} bytes", Q4K_SUPERBLOCK_BYTES);
    println!("     Compression ratio: {:.2}x vs f32", (256.0 * 4.0) / Q4K_SUPERBLOCK_BYTES as f32);

    let matvec = TiledQ4KMatvec::new(4096, 4096);
    println!("\n   4096×4096 Q4_K MatVec:");
    println!("     Superblocks per row: {}", matvec.superblocks_per_row());
    println!("     Total superblocks: {}", matvec.total_superblocks());
    println!("     Weight offset for row 100: {} bytes", matvec.weight_row_offset(100));
    println!("     Optimal parallel rows (256KB L2): {}", matvec.optimal_parallel_rows(256 * 1024));

    let stats = matvec.stats();
    println!("\n   Statistics:");
    println!("     Weight bytes: {:.2} MB", stats.total_weight_bytes as f64 / (1024.0 * 1024.0));
    println!("     Input bytes: {} KB", stats.input_bytes / 1024);
    println!("     Output bytes: {} KB", stats.output_bytes / 1024);
    println!("     Arithmetic ops: {:.0}M", stats.arithmetic_ops as f64 / 1_000_000.0);
    println!("     Arithmetic intensity: {:.2} FLOP/byte", stats.arithmetic_intensity);

    // =========================================================================
    // 6. Prefetch optimization
    // =========================================================================
    println!("\n6. Prefetch Optimization - Cache-aware data loading");
    println!("   ─────────────────────────────────────────────────────");

    let geom = TcbGeometry::new(4, 8, 64);
    println!("   For micro-tile {}:", geom);
    println!(
        "     L1 prefetch distance: {} tiles",
        optimal_prefetch_distance(&geom, TcbLevel::Micro)
    );
    println!(
        "     L2 prefetch distance: {} tiles",
        optimal_prefetch_distance(&geom, TcbLevel::Midi)
    );
    println!(
        "     L3 prefetch distance: {} tiles",
        optimal_prefetch_distance(&geom, TcbLevel::Macro)
    );

    println!("\n   Cache level typical sizes:");
    println!("     L1 (Micro): {} KB", TcbLevel::Micro.typical_cache_bytes() / 1024);
    println!("     L2 (Midi): {} KB", TcbLevel::Midi.typical_cache_bytes() / 1024);
    println!("     L3 (Macro): {} MB", TcbLevel::Macro.typical_cache_bytes() / (1024 * 1024));

    // =========================================================================
    // 7. Configuration comparison
    // =========================================================================
    println!("\n7. Backend Configuration Comparison");
    println!("   ────────────────────────────────────");

    compare_backends();

    println!("\n=== Demo Complete ===");
}

fn print_config(config: &TilingConfig) {
    println!("\n   {} ({:?}):", config.name, config.backend);
    println!(
        "     Macro: {}×{}×{} (L3/Global)",
        config.macro_tile.m, config.macro_tile.n, config.macro_tile.k
    );
    println!(
        "     Midi:  {}×{}×{} (L2/Shared)",
        config.midi_tile.m, config.midi_tile.n, config.midi_tile.k
    );
    println!(
        "     Micro: {}×{}×{} (Registers)",
        config.micro_tile.m, config.micro_tile.n, config.micro_tile.k
    );
    if let Ok(()) = config.validate() {
        println!("     Validation: PASS");
    } else {
        println!("     Validation: FAIL");
    }
}

fn compare_backends() {
    let configs = [
        TilingConfig::cpu_avx2_matmul(),
        TilingConfig::cpu_avx512_matmul(),
        TilingConfig::gpu_q4k_matmul(),
    ];

    println!(
        "\n   {:20} {:>10} {:>10} {:>10} {:>12}",
        "Backend", "Micro M", "Micro N", "Alignment", "AI (FLOP/B)"
    );
    println!("   {:─<20} {:─>10} {:─>10} {:─>10} {:─>12}", "", "", "", "", "");

    for config in &configs {
        println!(
            "   {:20} {:>10} {:>10} {:>10} {:>12.2}",
            config.name,
            config.micro_tile.m,
            config.micro_tile.n,
            config.micro_tile.alignment,
            config.micro_tile.arithmetic_intensity()
        );
    }
}
