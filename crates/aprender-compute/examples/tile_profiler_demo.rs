#![allow(clippy::disallowed_methods)]
//! Tile Profiler Demo
//!
//! Demonstrates BrickProfiler tile-level profiling for hierarchical
//! cache-blocked tiling (TILING-SPEC-001).
//!
//! Key features:
//! - Per-tile timing (Macro/Midi/Micro)
//! - GFLOP/s and throughput tracking
//! - Arithmetic intensity analysis
//! - JSON export for pmat integration
//!
//! Run: cargo run --example tile_profiler_demo

use trueno::brick::{BrickProfiler, TileLevel};
use trueno::tiling::{TiledQ4KMatvec, TilingConfig, Q4K_SUPERBLOCK_BYTES};

fn main() {
    println!("=== Trueno Tile Profiler Demo (TILING-SPEC-001) ===\n");

    // =========================================================================
    // 1. Basic tile profiling
    // =========================================================================
    println!("1. Basic Tile Profiling");
    println!("   ─────────────────────");

    let mut profiler = BrickProfiler::new();
    profiler.enable_tile_profiling();

    // Simulate macro tile execution
    println!("\n   Simulating hierarchical tile execution...");

    // Macro tile: 512x512x512 = 256M FLOPs (2*M*N*K)
    let macro_elements: u64 = 512 * 512;
    let macro_flops: u64 = 2 * 512 * 512 * 512;

    for i in 0..4 {
        let timer = profiler.start_tile(TileLevel::Macro, i, 0);
        // Simulate some work
        std::thread::sleep(std::time::Duration::from_millis(10));
        profiler.stop_tile(timer, macro_elements, macro_flops);
    }

    // Midi tile: 64x64x64 = 512K FLOPs
    let midi_elements: u64 = 64 * 64;
    let midi_flops: u64 = 2 * 64 * 64 * 64;

    for i in 0..64 {
        let timer = profiler.start_tile(TileLevel::Midi, i % 8, i / 8);
        std::thread::sleep(std::time::Duration::from_micros(100));
        profiler.stop_tile(timer, midi_elements, midi_flops);
    }

    // Micro tile: 4x8 = 32 elements, 2K FLOPs
    let micro_elements: u64 = 4 * 8;
    let micro_flops: u64 = 2 * 4 * 8 * 256;

    for i in 0..512 {
        let timer = profiler.start_tile(TileLevel::Micro, i % 16, i / 16);
        // Micro tiles are very fast, no sleep
        std::hint::black_box(i * 2);
        profiler.stop_tile(timer, micro_elements, micro_flops);
    }

    // Print summary
    println!("{}", profiler.tile_summary());

    // =========================================================================
    // 2. Statistics analysis
    // =========================================================================
    println!("2. Detailed Statistics Analysis");
    println!("   ─────────────────────────────");

    let all_stats = profiler.all_tile_stats();
    for stats in all_stats {
        if stats.count > 0 {
            println!("\n   {} tiles:", stats.level.name().to_uppercase());
            println!("     Samples: {}", stats.count);
            println!("     Total time: {:.2} ms", stats.total_ns as f64 / 1_000_000.0);
            println!("     Avg time: {:.2} µs", stats.avg_us());
            println!("     Min time: {:.2} µs", stats.min_ns as f64 / 1000.0);
            println!("     Max time: {:.2} µs", stats.max_ns as f64 / 1000.0);
            println!("     Throughput: {:.2} Melem/s", stats.throughput() / 1_000_000.0);
            println!("     GFLOP/s: {:.2}", stats.gflops());
            println!("     Arithmetic intensity: {:.2} FLOP/byte", stats.arithmetic_intensity());
        }
    }

    // =========================================================================
    // 3. Q4K MatVec tile profiling
    // =========================================================================
    println!("\n3. Q4K MatVec Tile Profiling");
    println!("   ──────────────────────────");

    let mut q4k_profiler = BrickProfiler::new();
    q4k_profiler.enable_tile_profiling();

    let matvec = TiledQ4KMatvec::new(1024, 1024);
    let weights = vec![0u8; matvec.total_superblocks() * Q4K_SUPERBLOCK_BYTES];
    let input = vec![1.0f32; 1024];
    let mut output = vec![0.0f32; 1024];

    println!("   Executing 1024x1024 Q4K MatVec with tile profiling...\n");

    // Profile multiple executions with tile tracking
    for batch in 0..10 {
        let timer = q4k_profiler.start_tile(TileLevel::Macro, batch, 0);

        // Execute the actual Q4K MatVec
        matvec.execute_scalar(&weights, &input, &mut output);

        // Calculate FLOPs: 2 ops per element (multiply + add)
        let flops = (1024 * 1024 * 2) as u64;
        q4k_profiler.stop_tile(timer, (1024 * 1024) as u64, flops);
    }

    let macro_stats = q4k_profiler.tile_stats(TileLevel::Macro);
    println!("   Q4K MatVec Results:");
    println!("     Batches: {}", macro_stats.count);
    println!("     Avg time: {:.2} ms", macro_stats.avg_us() / 1000.0);
    println!("     Throughput: {:.2} Melem/s", macro_stats.throughput() / 1_000_000.0);
    println!("     GFLOP/s: {:.2}", macro_stats.gflops());

    // =========================================================================
    // 4. Cache efficiency analysis
    // =========================================================================
    println!("\n4. Cache Efficiency Analysis");
    println!("   ───────────────────────────");

    // Theoretical peaks (example values for modern CPUs)
    let avx2_peak_gflops = 100.0; // ~100 GFLOP/s for AVX2 FMADD
    let avx512_peak_gflops = 200.0; // ~200 GFLOP/s for AVX-512

    println!("   Reference peaks:");
    println!("     AVX2: {:.0} GFLOP/s", avx2_peak_gflops);
    println!("     AVX-512: {:.0} GFLOP/s", avx512_peak_gflops);

    let macro_stats = profiler.tile_stats(TileLevel::Macro);
    if macro_stats.count > 0 {
        println!("\n   Macro tile efficiency:");
        println!("     vs AVX2: {:.1}%", macro_stats.cache_efficiency(avx2_peak_gflops) * 100.0);
        println!(
            "     vs AVX-512: {:.1}%",
            macro_stats.cache_efficiency(avx512_peak_gflops) * 100.0
        );
    }

    // =========================================================================
    // 5. JSON export
    // =========================================================================
    println!("\n5. JSON Export (pmat integration)");
    println!("   ─────────────────────────────────");

    let json = profiler.tile_stats_to_json();
    println!("   {}", json);

    // =========================================================================
    // 6. Tile geometry analysis
    // =========================================================================
    println!("\n6. Tile Geometry Analysis");
    println!("   ─────────────────────────");

    let configs = [
        TilingConfig::cpu_avx2_matmul(),
        TilingConfig::cpu_avx512_matmul(),
        TilingConfig::gpu_q4k_matvec(),
    ];

    println!("\n   {:20} {:>10} {:>12} {:>12}", "Config", "AI", "Macro AI", "Micro AI");
    println!("   {:─<20} {:─>10} {:─>12} {:─>12}", "", "", "", "");

    for config in &configs {
        println!(
            "   {:20} {:>10.2} {:>12.2} {:>12.2}",
            config.name,
            config.micro_tile.arithmetic_intensity(),
            config.macro_tile.arithmetic_intensity(),
            config.micro_tile.arithmetic_intensity()
        );
    }

    // =========================================================================
    // 7. Reset and reuse
    // =========================================================================
    println!("\n7. Profiler Reset Demo");
    println!("   ──────────────────────");

    println!("   Before reset: {} macro samples", profiler.tile_stats(TileLevel::Macro).count);
    profiler.reset_tile_stats();
    println!("   After reset: {} macro samples", profiler.tile_stats(TileLevel::Macro).count);

    println!("\n=== Demo Complete ===");
}
