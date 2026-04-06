//! PartitionView - Tiling Strategy for GPU Compute
//!
//! Divides a TensorView into tiles for efficient GPU processing.
//! Enables automatic work distribution across thread blocks.
//!
//! # cuda-tile-behavior.md References
//!
//! - Section 3.2: Two-Level Memory Hierarchy
//! - Falsification tests #36-45: PartitionView correctness
//!
//! # Academic Foundation
//!
//! Based on Volkov & Demmel (2008): Power-of-two tiles achieve 95%+ peak throughput.

use super::tensor_view::TensorView;
use std::marker::PhantomData;

/// A tiling strategy over a TensorView.
///
/// PartitionView divides a tensor into tiles of a specified shape,
/// enabling efficient GPU processing with shared memory optimization.
///
/// # Type Parameters
///
/// * `T` - Element type of the underlying tensor
///
/// # cuda-tile-behavior.md References
///
/// - Falsification test #36: Tile count calculation is correct
/// - Falsification test #37: Tile iteration covers all elements
/// - Falsification test #38: Edge tiles are handled correctly
#[derive(Debug)]
pub struct PartitionView<T> {
    /// The underlying tensor being partitioned
    tensor: TensorView<T>,
    /// Shape of each tile
    tile_shape: [usize; 4],
    /// Phantom data for type safety
    _marker: PhantomData<T>,
}

/// Information about a single tile within a partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileInfo {
    /// Tile index in each dimension
    pub tile_idx: [usize; 4],
    /// Starting element index in each dimension
    pub start: [usize; 4],
    /// Size of this tile in each dimension (may be smaller at edges)
    pub size: [usize; 4],
    /// Whether this is an edge tile (smaller than full tile size)
    pub is_edge: bool,
}

impl<T> PartitionView<T> {
    /// Create a new PartitionView with the given tile shape.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to partition
    /// * `tile_shape` - Shape of each tile
    ///
    /// # Panics
    ///
    /// Panics if any tile dimension is zero.
    ///
    /// # cuda-tile-behavior.md References
    ///
    /// - Falsification test #36: Tile count calculation is correct
    pub fn new(tensor: TensorView<T>, tile_shape: [usize; 4]) -> Self {
        assert!(tile_shape.iter().all(|&d| d > 0), "Tile dimensions must be non-zero");
        Self { tensor, tile_shape, _marker: PhantomData }
    }

    /// Create a PartitionView with power-of-two tile sizes.
    ///
    /// This is recommended for GPU compute as it enables efficient
    /// memory coalescing and avoids bank conflicts.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to partition
    /// * `tile_log2` - Log2 of tile size for each dimension
    ///
    /// # cuda-tile-behavior.md References
    ///
    /// - Falsification test #1: Power-of-two tiles improve GPU occupancy
    pub fn new_power_of_two(tensor: TensorView<T>, tile_log2: [usize; 4]) -> Self {
        let tile_shape =
            [1 << tile_log2[0], 1 << tile_log2[1], 1 << tile_log2[2], 1 << tile_log2[3]];
        Self::new(tensor, tile_shape)
    }

    /// Create a PartitionView with 2D tiles (for matrix operations).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to partition
    /// * `tile_rows` - Number of rows per tile
    /// * `tile_cols` - Number of columns per tile
    pub fn new_2d(tensor: TensorView<T>, tile_rows: usize, tile_cols: usize) -> Self {
        Self::new(tensor, [tile_rows, tile_cols, 1, 1])
    }

    /// Get the underlying tensor.
    pub fn tensor(&self) -> &TensorView<T> {
        &self.tensor
    }

    /// Get the tile shape.
    pub fn tile_shape(&self) -> &[usize; 4] {
        &self.tile_shape
    }

    /// Get the number of tiles in each dimension.
    ///
    /// # cuda-tile-behavior.md References
    ///
    /// - Falsification test #36: Tile count calculation is correct
    pub fn tile_count(&self) -> [usize; 4] {
        let tensor_shape = self.tensor.shape();
        [
            tensor_shape[0].div_ceil(self.tile_shape[0]),
            tensor_shape[1].div_ceil(self.tile_shape[1]),
            tensor_shape[2].div_ceil(self.tile_shape[2]),
            tensor_shape[3].div_ceil(self.tile_shape[3]),
        ]
    }

    /// Get the total number of tiles.
    pub fn total_tiles(&self) -> usize {
        let count = self.tile_count();
        count.iter().product()
    }

    /// Get information about a specific tile.
    ///
    /// # Arguments
    ///
    /// * `tile_idx` - Index of the tile in each dimension
    ///
    /// # Returns
    ///
    /// TileInfo containing the tile's position and size.
    ///
    /// # cuda-tile-behavior.md References
    ///
    /// - Falsification test #38: Edge tiles are handled correctly
    pub fn get_tile(&self, tile_idx: [usize; 4]) -> Option<TileInfo> {
        let tile_count = self.tile_count();

        // Validate tile index
        for i in 0..4 {
            if tile_idx[i] >= tile_count[i] {
                return None;
            }
        }

        let tensor_shape = self.tensor.shape();
        let mut start = [0usize; 4];
        let mut size = [0usize; 4];
        let mut is_edge = false;

        for i in 0..4 {
            start[i] = tile_idx[i] * self.tile_shape[i];
            let remaining = tensor_shape[i] - start[i];
            size[i] = remaining.min(self.tile_shape[i]);

            // Check if this is an edge tile
            if size[i] < self.tile_shape[i] {
                is_edge = true;
            }
        }

        Some(TileInfo { tile_idx, start, size, is_edge })
    }

    /// Get a TensorView for a specific tile.
    ///
    /// # Arguments
    ///
    /// * `tile_idx` - Index of the tile in each dimension
    ///
    /// # Returns
    ///
    /// A TensorView representing the tile, or None if index is invalid.
    pub fn get_tile_view(&self, tile_idx: [usize; 4]) -> Option<TensorView<T>> {
        let info = self.get_tile(tile_idx)?;

        // Create a sliced view for this tile
        let mut view = self.tensor.clone();

        for i in 0..4 {
            if self.tensor.shape()[i] > 1 {
                view = view.slice_dim(i, info.start[i]..info.start[i] + info.size[i]);
            }
        }

        Some(view)
    }

    /// Iterate over all tiles.
    ///
    /// # cuda-tile-behavior.md References
    ///
    /// - Falsification test #37: Tile iteration covers all elements
    pub fn iter_tiles(&self) -> TileIterator<'_, T> {
        TileIterator { partition: self, current: [0, 0, 0, 0], done: false }
    }

    /// Check if tiles are power-of-two sized.
    ///
    /// Power-of-two tiles are preferred for GPU compute.
    pub fn is_power_of_two_tiles(&self) -> bool {
        self.tile_shape.iter().all(|&d| d.is_power_of_two())
    }

    /// Get the number of elements per tile (maximum).
    pub fn elements_per_tile(&self) -> usize {
        self.tile_shape.iter().product()
    }

    /// Get recommended workgroup size for GPU dispatch.
    ///
    /// Returns (x, y, z) workgroup dimensions based on tile shape.
    pub fn recommended_workgroup_size(&self) -> (u32, u32, u32) {
        // Common GPU workgroup limits
        const MAX_WORKGROUP_SIZE: usize = 256;
        const MAX_DIM: usize = 16;

        let tile_2d = [self.tile_shape[0], self.tile_shape[1]];

        // For 2D tiles, use 2D workgroups
        if tile_2d[0] > 1 && tile_2d[1] > 1 {
            let x = tile_2d[1].min(MAX_DIM) as u32;
            let y = tile_2d[0].min(MAX_DIM) as u32;
            let z = 1;
            (x, y, z)
        } else {
            // 1D workgroup
            let size = self.elements_per_tile().min(MAX_WORKGROUP_SIZE);
            (size as u32, 1, 1)
        }
    }
}

impl<T> Clone for PartitionView<T> {
    fn clone(&self) -> Self {
        Self { tensor: self.tensor.clone(), tile_shape: self.tile_shape, _marker: PhantomData }
    }
}

/// Iterator over tiles in a PartitionView.
pub struct TileIterator<'a, T> {
    partition: &'a PartitionView<T>,
    current: [usize; 4],
    done: bool,
}

impl<T> Iterator for TileIterator<'_, T> {
    type Item = TileInfo;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let tile = self.partition.get_tile(self.current)?;
        let tile_count = self.partition.tile_count();

        // Advance to next tile (row-major order)
        self.current[3] += 1;
        for i in (0..4).rev() {
            if self.current[i] >= tile_count[i] {
                self.current[i] = 0;
                if i > 0 {
                    self.current[i - 1] += 1;
                } else {
                    self.done = true;
                }
            } else {
                break;
            }
        }

        Some(tile)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let total = self.partition.total_tiles();
        (total, Some(total))
    }
}

impl<T> ExactSizeIterator for TileIterator<'_, T> {}

#[cfg(test)]
mod tests;
