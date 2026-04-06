#![allow(clippy::disallowed_methods)]
//! SIMD Softmax and Quantization Demo (SIMD-EXP, QUANT-Q5K)
//!
//! This example demonstrates:
//! 1. SIMD-accelerated softmax using polynomial exp approximation
//! 2. Q5_K and Q6_K quantization formats (llama.cpp compatible)
//! 3. BrickProfiler benchmarks comparing SIMD vs scalar performance
//!
//! Run: cargo run --example simd_softmax_quant --release

use std::time::Instant;
use trueno::{
    BlockQ5K, BlockQ6K, BrickProfiler, ComputeBackend as Backend, ComputeOp, DotQ5KOp, DotQ6KOp,
    SoftmaxOp, TileLevel,
};

fn main() {
    println!("=== Trueno SIMD Softmax & Quantization Demo ===\n");

    // Initialize profiler
    let mut profiler = BrickProfiler::new();
    profiler.enable_tile_profiling();

    // Demo 1: SIMD Softmax Performance
    demo_simd_softmax(&mut profiler);

    // Demo 2: Q5_K Quantization
    demo_q5k_quantization(&mut profiler);

    // Demo 3: Q6_K Quantization
    demo_q6k_quantization(&mut profiler);

    // Print profiler summary
    println!("\n=== Tile Profiling Summary ===");
    println!("{}", profiler.tile_summary());
}

fn demo_simd_softmax(profiler: &mut BrickProfiler) {
    println!("--- SIMD Softmax Performance ---\n");

    // Large vector for meaningful benchmark
    let size = 4096;
    let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 20.0).collect();
    let op = SoftmaxOp::new(size);

    // Benchmark Scalar
    let timer = profiler.start_tile(TileLevel::Macro, 0, 0);
    let start = Instant::now();
    let _scalar_result = op.execute(input.clone(), Backend::Scalar).unwrap();
    let scalar_time = start.elapsed();
    profiler.stop_tile(timer, size as u64, (size * 3) as u64); // 3 ops: sub, exp, div

    // Benchmark SIMD (Auto selects best available)
    let timer = profiler.start_tile(TileLevel::Macro, 0, 1);
    let start = Instant::now();
    let simd_result = op.execute(input.clone(), Backend::Auto).unwrap();
    let simd_time = start.elapsed();
    profiler.stop_tile(timer, size as u64, (size * 3) as u64);

    // Verify correctness
    let sum: f32 = simd_result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax sum should be 1.0, got {}", sum);

    let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
    println!("  Vector size: {}", size);
    println!("  Scalar time: {:?}", scalar_time);
    println!("  SIMD time:   {:?}", simd_time);
    println!("  Speedup:     {:.2}x", speedup);
    println!("  Sum check:   {:.6} (should be 1.0)", sum);
    println!();
}

fn demo_q5k_quantization(profiler: &mut BrickProfiler) {
    println!("--- Q5_K Quantization (5-bit with k-quant scales) ---\n");

    // Create a test block with some quantized data
    let block = BlockQ5K {
        d: 0.1,
        dmin: -0.05,
        scales: [40, 42, 44, 46, 48, 50, 52, 54, 0, 0, 0, 0], // Non-zero scales
        qh: [0b0101_0101; 32],                                // Alternating 5th bits
        qs: [0x55; 128],                                      // Pattern: 0101 0101
    };

    // Dequantize
    let timer = profiler.start_tile(TileLevel::Midi, 1, 0);
    let start = Instant::now();
    let mut dequant = [0.0f32; 256];
    block.dequantize(&mut dequant);
    let dequant_time = start.elapsed();
    profiler.stop_tile(timer, 256, 256 * 3); // 3 ops per element

    println!("  Block size:      {} elements", BlockQ5K::BLOCK_SIZE);
    println!("  Dequant time:    {:?}", dequant_time);
    println!("  Sample values:   [{:.4}, {:.4}, {:.4}, ...]", dequant[0], dequant[1], dequant[2]);

    // Dot product with Q5_K
    let op = DotQ5KOp::new(256);
    let x = vec![1.0f32; 256];

    let timer = profiler.start_tile(TileLevel::Midi, 1, 1);
    let start = Instant::now();
    let dot = op.execute((vec![block], x), Backend::Auto).unwrap();
    let dot_time = start.elapsed();
    profiler.stop_tile(timer, 256, 256 * 2); // 2 ops: mul, add

    println!("  Dot product:     {:.4}", dot);
    println!("  Dot time:        {:?}", dot_time);
    println!();
}

fn demo_q6k_quantization(profiler: &mut BrickProfiler) {
    println!("--- Q6_K Quantization (6-bit with k-quant scales) ---\n");

    // Create a test block with some quantized data
    let block = BlockQ6K {
        ql: [0x55; 128], // Pattern: 0101 0101 (low 4 bits)
        qh: [0x55; 64],  // Pattern: 0101 0101 (high 2 bits)
        scales: [10, 12, 14, 16, 18, 20, 22, 24, 10, 12, 14, 16, 18, 20, 22, 24],
        d: 0.1,
    };

    // Dequantize
    let timer = profiler.start_tile(TileLevel::Midi, 2, 0);
    let start = Instant::now();
    let mut dequant = [0.0f32; 256];
    block.dequantize(&mut dequant);
    let dequant_time = start.elapsed();
    profiler.stop_tile(timer, 256, 256 * 3); // 3 ops per element

    println!("  Block size:      {} elements", BlockQ6K::BLOCK_SIZE);
    println!("  Dequant time:    {:?}", dequant_time);
    println!("  Sample values:   [{:.4}, {:.4}, {:.4}, ...]", dequant[0], dequant[1], dequant[2]);

    // Dot product with Q6_K
    let op = DotQ6KOp::new(256);
    let x = vec![1.0f32; 256];

    let timer = profiler.start_tile(TileLevel::Midi, 2, 1);
    let start = Instant::now();
    let dot = op.execute((vec![block], x), Backend::Auto).unwrap();
    let dot_time = start.elapsed();
    profiler.stop_tile(timer, 256, 256 * 2); // 2 ops: mul, add

    println!("  Dot product:     {:.4}", dot);
    println!("  Dot time:        {:?}", dot_time);
    println!();
}
