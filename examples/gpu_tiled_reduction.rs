#![allow(clippy::disallowed_methods)]
//! GPU Tiled Reduction Demo
//!
//! Demonstrates the tiled reduction shader for efficient GPU-based
//! sum, max, and min operations on large matrices.
//!
//! **Key Concepts**:
//! - 16x16 tile-based workgroup reduction
//! - Two-phase reduction: intra-tile then cross-tile
//! - Optimal for data already resident on GPU
//!
//! **Performance Characteristics** (Metal, AMD Radeon Pro W5700X):
//! - Consistent ~150 Melem/s throughput
//! - ~8ms baseline transfer overhead
//! - GPU reduction optimal when part of larger pipeline

use trueno::backends::gpu::GpuBackend;

fn main() -> Result<(), String> {
    println!("=== Trueno GPU Tiled Reduction Demo ===\n");

    // Check GPU availability
    if !GpuBackend::is_available() {
        println!("⚠️  GPU not available (wgpu backend not found)");
        println!("   This example requires a GPU with Vulkan/Metal/DX12 support");
        return Ok(());
    }

    println!("✅ GPU available - initializing backend...\n");

    let mut gpu = GpuBackend::new();

    // Demo 1: Small matrix sum
    println!("📊 Demo 1: Small Matrix Sum (16x16 = 256 elements)");
    demo_small_sum(&mut gpu)?;

    println!();

    // Demo 2: Large matrix sum
    println!("📊 Demo 2: Large Matrix Sum (1000x1000 = 1M elements)");
    demo_large_sum(&mut gpu)?;

    println!();

    // Demo 3: Tiled max reduction
    println!("📊 Demo 3: Tiled Max Reduction");
    demo_tiled_max(&mut gpu)?;

    println!();

    // Demo 4: Tiled min reduction
    println!("📊 Demo 4: Tiled Min Reduction");
    demo_tiled_min(&mut gpu)?;

    println!();

    // Performance guidance
    println!("📈 Performance Guidance");
    println!("   ┌──────────────────────────────────────────────────────────┐");
    println!("   │ GPU tiled reduction is optimal when:                     │");
    println!("   │ • Data is already on GPU (no transfer cost)              │");
    println!("   │ • Reduction is part of larger GPU pipeline               │");
    println!("   │ • Latency hiding in async workloads                      │");
    println!("   │                                                          │");
    println!("   │ For standalone CPU→GPU→CPU reductions, prefer SIMD:      │");
    println!("   │ • trueno::vector_sum() uses AVX/AVX-512/NEON             │");
    println!("   │ • 7-37x faster due to no transfer overhead               │");
    println!("   └──────────────────────────────────────────────────────────┘");

    Ok(())
}

/// Demo 1: Small matrix that fits in a single 16x16 tile
fn demo_small_sum(gpu: &mut GpuBackend) -> Result<(), String> {
    // Create 16x16 matrix with known values
    let width = 16;
    let height = 16;
    let data: Vec<f32> = (0..width * height).map(|i| i as f32).collect();

    // Expected sum: 0 + 1 + 2 + ... + 255 = 255 * 256 / 2 = 32640
    let expected: f32 = (0..256).map(|i| i as f32).sum();

    println!("   Matrix: {}x{} = {} elements", width, height, width * height);
    println!("   Values: 0, 1, 2, ..., 255");
    println!("   Expected sum: {}", expected);

    let result = gpu.tiled_sum_2d_gpu(&data, width, height)?;

    println!("   GPU result: {}", result);

    if (result - expected).abs() < 1e-3 {
        println!("   ✅ Passed");
    } else {
        println!("   ❌ Failed: expected {}, got {}", expected, result);
    }

    Ok(())
}

/// Demo 2: Large matrix spanning multiple tiles
fn demo_large_sum(gpu: &mut GpuBackend) -> Result<(), String> {
    let width = 1000;
    let height = 1000;

    // Create matrix with value 1.0 (sum should equal element count)
    let data: Vec<f32> = vec![1.0; width * height];
    let expected = (width * height) as f32;

    println!("   Matrix: {}x{} = {} elements", width, height, width * height);
    println!("   All values: 1.0");
    println!("   Expected sum: {}", expected);

    let start = std::time::Instant::now();
    let result = gpu.tiled_sum_2d_gpu(&data, width, height)?;
    let elapsed = start.elapsed();

    println!("   GPU result: {}", result);
    println!("   Time: {:?}", elapsed);
    println!(
        "   Throughput: {:.2} Melem/s",
        (width * height) as f64 / elapsed.as_secs_f64() / 1_000_000.0
    );

    if (result - expected).abs() < 1e-1 {
        println!("   ✅ Passed");
    } else {
        println!("   ❌ Failed: expected {}, got {}", expected, result);
    }

    Ok(())
}

/// Demo 3: Tiled max reduction
fn demo_tiled_max(gpu: &mut GpuBackend) -> Result<(), String> {
    let width = 100;
    let height = 100;

    // Create matrix with a single maximum value
    let mut data: Vec<f32> = vec![1.0; width * height];
    data[5050] = 999.0; // Hidden max in the middle

    println!("   Matrix: {}x{} = {} elements", width, height, width * height);
    println!("   Values: all 1.0 except one 999.0 at index 5050");

    let result = gpu.tiled_max_2d_gpu(&data, width, height)?;

    println!("   GPU max: {}", result);

    if (result - 999.0).abs() < 1e-3 {
        println!("   ✅ Passed");
    } else {
        println!("   ❌ Failed: expected 999.0, got {}", result);
    }

    Ok(())
}

/// Demo 4: Tiled min reduction
fn demo_tiled_min(gpu: &mut GpuBackend) -> Result<(), String> {
    let width = 100;
    let height = 100;

    // Create matrix with a single minimum value
    let mut data: Vec<f32> = vec![100.0; width * height];
    data[7777] = -42.0; // Hidden min

    println!("   Matrix: {}x{} = {} elements", width, height, width * height);
    println!("   Values: all 100.0 except one -42.0 at index 7777");

    let result = gpu.tiled_min_2d_gpu(&data, width, height)?;

    println!("   GPU min: {}", result);

    if (result - (-42.0)).abs() < 1e-3 {
        println!("   ✅ Passed");
    } else {
        println!("   ❌ Failed: expected -42.0, got {}", result);
    }

    Ok(())
}
