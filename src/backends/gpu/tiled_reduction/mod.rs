//! CPU fallback implementation of tiled reduction algorithms
//!
//! This module provides CPU implementations that mirror the GPU tiled reduction
//! algorithms. These are useful for:
//! - Testing and validation (compare GPU results against CPU reference)
//! - Fallback when GPU is unavailable
//! - Understanding the algorithm without GPU complexity
//!
//! The algorithms use the same 16×16 tile structure as the GPU shaders.

use super::partition_view::PartitionView;
use super::tensor_view::TensorView;

/// Default tile size for 2D reductions (matches GPU workgroup size)
pub const TILE_SIZE: usize = 16;

/// Reduction operation trait for generic tile reduction
pub trait ReduceOp {
    /// Identity element for the reduction (0 for sum, -inf for max, inf for min)
    fn identity() -> f32;
    /// Combine two values
    fn combine(a: f32, b: f32) -> f32;
}

/// Sum reduction operation
pub struct SumOp;

impl ReduceOp for SumOp {
    #[inline]
    fn identity() -> f32 {
        0.0
    }

    #[inline]
    fn combine(a: f32, b: f32) -> f32 {
        a + b
    }
}

/// Max reduction operation
pub struct MaxOp;

impl ReduceOp for MaxOp {
    #[inline]
    fn identity() -> f32 {
        f32::NEG_INFINITY
    }

    #[inline]
    fn combine(a: f32, b: f32) -> f32 {
        a.max(b)
    }
}

/// Min reduction operation
pub struct MinOp;

impl ReduceOp for MinOp {
    #[inline]
    fn identity() -> f32 {
        f32::INFINITY
    }

    #[inline]
    fn combine(a: f32, b: f32) -> f32 {
        a.min(b)
    }
}

/// Perform tiled reduction on 2D data (CPU fallback)
///
/// This simulates the GPU algorithm:
/// 1. Partition input into 16×16 tiles
/// 2. Reduce each tile to a single value
/// 3. Combine partial results
pub fn tiled_reduce_2d<Op: ReduceOp>(data: &[f32], width: usize, height: usize) -> f32 {
    let partial = collect_tile_results::<Op>(data, width, height);
    partial.iter().copied().fold(Op::identity(), Op::combine)
}

/// Convenience function for tiled sum reduction
#[inline]
pub fn tiled_sum_2d(data: &[f32], width: usize, height: usize) -> f32 {
    tiled_reduce_2d::<SumOp>(data, width, height)
}

/// Convenience function for tiled max reduction
#[inline]
pub fn tiled_max_2d(data: &[f32], width: usize, height: usize) -> f32 {
    tiled_reduce_2d::<MaxOp>(data, width, height)
}

/// Convenience function for tiled min reduction
#[inline]
pub fn tiled_min_2d(data: &[f32], width: usize, height: usize) -> f32 {
    tiled_reduce_2d::<MinOp>(data, width, height)
}

/// Compute partial tile results for verification
///
/// Returns the partial reduction result for each tile, which can be
/// compared against GPU partial results buffer for validation.
pub fn tiled_reduce_partial<Op: ReduceOp>(data: &[f32], width: usize, height: usize) -> Vec<f32> {
    collect_tile_results::<Op>(data, width, height)
}

/// Shared implementation: reduce each tile and return partial results.
fn collect_tile_results<Op: ReduceOp>(data: &[f32], width: usize, height: usize) -> Vec<f32> {
    if data.is_empty() || width == 0 || height == 0 {
        return vec![Op::identity()];
    }

    let view: TensorView<f32> = TensorView::new([height, width, 1, 1]);
    let partition: PartitionView<f32> = PartitionView::new(view, [TILE_SIZE, TILE_SIZE, 1, 1]);

    let tiles_y = partition.tile_count()[0];
    let tiles_x = partition.tile_count()[1];

    let mut results = Vec::with_capacity(tiles_y * tiles_x);
    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            results.push(reduce_tile::<Op>(data, width, height, tile_x, tile_y));
        }
    }
    results
}

/// Load data into a tile with bounds checking.
fn load_tile(
    tile: &mut [[f32; TILE_SIZE]; TILE_SIZE],
    data: &[f32],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
) {
    #[allow(clippy::needless_range_loop)]
    for ly in 0..TILE_SIZE {
        let gy = start_y + ly;
        if gy >= height {
            break;
        }
        #[allow(clippy::needless_range_loop)]
        for lx in 0..TILE_SIZE {
            let gx = start_x + lx;
            if gx >= width {
                break;
            }
            tile[ly][lx] = data[gy * width + gx];
        }
    }
}

/// Tree reduction along rows: halve stride each step until 1.
fn reduce_rows<Op: ReduceOp>(tile: &mut [[f32; TILE_SIZE]; TILE_SIZE]) {
    #[allow(clippy::needless_range_loop)]
    for ly in 0..TILE_SIZE {
        let mut stride = TILE_SIZE / 2;
        while stride > 0 {
            for lx in 0..stride {
                tile[ly][lx] = Op::combine(tile[ly][lx], tile[ly][lx + stride]);
            }
            stride /= 2;
        }
    }
}

/// Tree reduction along columns: halve stride each step on column 0.
fn reduce_columns<Op: ReduceOp>(tile: &mut [[f32; TILE_SIZE]; TILE_SIZE]) {
    let mut stride = TILE_SIZE / 2;
    while stride > 0 {
        for ly in 0..stride {
            tile[ly][0] = Op::combine(tile[ly][0], tile[ly + stride][0]);
        }
        stride /= 2;
    }
}

/// Reduce a single 16×16 tile using tree reduction pattern
fn reduce_tile<Op: ReduceOp>(
    data: &[f32],
    width: usize,
    height: usize,
    tile_x: usize,
    tile_y: usize,
) -> f32 {
    let mut tile = [[Op::identity(); TILE_SIZE]; TILE_SIZE];
    load_tile(&mut tile, data, width, height, tile_x * TILE_SIZE, tile_y * TILE_SIZE);
    reduce_rows::<Op>(&mut tile);
    reduce_columns::<Op>(&mut tile);
    tile[0][0]
}

#[cfg(test)]
mod tests;
