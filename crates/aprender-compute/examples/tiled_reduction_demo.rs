#![allow(clippy::disallowed_methods)]
//! Tiled Reduction Demo
//!
//! Demonstrates the CUDA-tile-behavior inspired memory hierarchy abstractions:
//! - TensorView: Structured view into memory with shape/stride metadata
//! - PartitionView: Tiling strategy for efficient GPU work distribution
//! - Tiled reduction: 16x16 tile-based parallel reduction algorithms
//!
//! Run: cargo run --example tiled_reduction_demo

use trueno::backends::gpu::{
    tiled_max_2d, tiled_min_2d, tiled_sum_2d, MemoryLayout, PartitionView, TensorView, TILE_SIZE,
};

fn main() {
    println!("=== Trueno Tiled Reduction Demo ===\n");

    // -------------------------------------------------------------------------
    // 1. TensorView: Memory hierarchy abstraction
    // -------------------------------------------------------------------------
    println!("1. TensorView - Structured memory views");
    println!("   ─────────────────────────────────────");

    // Create a 4D tensor view (batch=2, channels=3, height=32, width=32)
    let view: TensorView<f32> = TensorView::new([2, 3, 32, 32]);
    println!("   Shape: {:?}", view.shape());
    println!("   Strides: {:?}", view.strides());
    println!("   Total elements: {}", view.numel());
    println!("   Layout: {:?}", view.layout());

    // Create with explicit strides (e.g., for transposed memory)
    let transposed = TensorView::<f32>::with_strides([32, 32, 3, 2], [1, 32, 1024, 3072]);
    println!("\n   Transposed view:");
    println!("   Shape: {:?}", transposed.shape());
    println!("   Is contiguous: {}", transposed.is_contiguous());

    // -------------------------------------------------------------------------
    // 2. PartitionView: Tiling strategy for GPU workgroups
    // -------------------------------------------------------------------------
    println!("\n2. PartitionView - GPU work distribution");
    println!("   ───────────────────────────────────────");

    // Partition a 64x64 tensor into 16x16 tiles (standard GPU workgroup size)
    let tensor: TensorView<f32> = TensorView::new([64, 64, 1, 1]);
    let partition: PartitionView<f32> = PartitionView::new(tensor, [16, 16, 1, 1]);

    println!("   Tensor shape: [64, 64]");
    println!("   Tile shape: [16, 16]");
    println!("   Tile count: {:?}", partition.tile_count());
    println!("   Total tiles: {}", partition.total_tiles());

    // Handle non-aligned dimensions (100x100 with 16x16 tiles)
    let non_aligned: TensorView<f32> = TensorView::new([100, 100, 1, 1]);
    let partition2: PartitionView<f32> = PartitionView::new(non_aligned, [16, 16, 1, 1]);
    println!("\n   Non-aligned tensor: [100, 100]");
    println!("   Tile count: {:?} (ceil division)", partition2.tile_count());
    println!("   Total tiles: {}", partition2.total_tiles());

    // Get tile info for edge tiles
    if let Some(tile_info) = partition2.get_tile([6, 6, 0, 0]) {
        println!("   Edge tile [6,6] size: {:?}", tile_info.size);
        println!("   Edge tile is edge: {}", tile_info.is_edge);
    }

    // -------------------------------------------------------------------------
    // 3. Tiled Reduction: 16x16 tile-based parallel reduction
    // -------------------------------------------------------------------------
    println!("\n3. Tiled Reduction - Parallel sum/max/min");
    println!("   ───────────────────────────────────────");
    println!("   TILE_SIZE = {} (matches GPU workgroup)", TILE_SIZE);

    // Create test data: 32x32 = 1024 elements
    let width = 32;
    let height = 32;
    let data: Vec<f32> = (1..=1024).map(|x| x as f32).collect();

    // Tiled sum reduction
    let sum = tiled_sum_2d(&data, width, height);
    let expected_sum: f32 = (1..=1024).sum::<i32>() as f32;
    println!("\n   Tiled Sum (32x32 matrix):");
    println!("   Result: {}", sum);
    println!("   Expected: {}", expected_sum);
    println!("   Match: {}", (sum - expected_sum).abs() < 1e-3);

    // Tiled max reduction
    let max_data: Vec<f32> =
        vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 8.0, 4.0, 6.0, 10.0, 15.0, 12.0, 11.0, 14.0, 13.0, 16.0];
    let max = tiled_max_2d(&max_data, 4, 4);
    println!("\n   Tiled Max (4x4 matrix with max=16):");
    println!("   Result: {}", max);
    println!("   Expected: 16.0");

    // Tiled min reduction
    let min_data: Vec<f32> = vec![5.0, 3.0, 7.0, -1.0, 9.0, 2.0, 8.0, 4.0, 6.0];
    let min = tiled_min_2d(&min_data, 3, 3);
    println!("\n   Tiled Min (3x3 matrix with min=-1):");
    println!("   Result: {}", min);
    println!("   Expected: -1.0");

    // -------------------------------------------------------------------------
    // 4. Performance characteristics
    // -------------------------------------------------------------------------
    println!("\n4. Performance Characteristics");
    println!("   ─────────────────────────────");
    println!("   - 16x16 tiles match GPU workgroup size");
    println!("   - Tree reduction: 16 -> 8 -> 4 -> 2 -> 1");
    println!("   - Row reduction first, then column reduction");
    println!("   - Identity elements: sum=0, max=-inf, min=inf");
    println!("   - CPU fallback for validation/testing");
    println!("   - GPU shaders use same algorithm structure");

    // -------------------------------------------------------------------------
    // 5. Memory layout support
    // -------------------------------------------------------------------------
    println!("\n5. Memory Layout Support");
    println!("   ───────────────────────");

    let row_major: TensorView<f32> = TensorView::new([4, 4, 1, 1]);
    let col_major: TensorView<f32> =
        TensorView::new([4, 4, 1, 1]).with_layout(MemoryLayout::ColumnMajor);

    println!("   Row-major strides: {:?}", row_major.strides());
    println!("   Col-major strides: {:?}", col_major.strides());
    println!("   Row-major is contiguous: {}", row_major.is_contiguous());
    println!("   Col-major is contiguous: {}", col_major.is_contiguous());

    println!("\n=== Demo Complete ===");
}
