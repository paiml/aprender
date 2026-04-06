//! GPU reduction operations
//!
//! Parallel max/sum reductions and 2D tiled reductions (sum/max/min).
//!
//! # Submodules
//!
//! - `reduce_1d` - 1D parallel reductions (max, sum) used by activation functions
//! - `tiled_2d` - Generic 2D tiled reduction infrastructure

mod reduce_1d;
mod tiled_2d;

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
use super::super::runtime;
use super::super::shaders;
use super::GpuDevice;

impl GpuDevice {
    /// 2D Tiled Sum Reduction on GPU (sync, native only)
    ///
    /// Uses 16x16 workgroups for efficient parallel reduction with
    /// optimal memory coalescing. GPU version of `tiled_sum_2d`.
    ///
    /// # Arguments
    ///
    /// * `data` - Input 2D data in row-major order
    /// * `width` - Number of columns
    /// * `height` - Number of rows
    ///
    /// # Returns
    ///
    /// Sum of all elements
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn tiled_sum_2d(&self, data: &[f32], width: usize, height: usize) -> Result<f32, String> {
        runtime::block_on(self.tiled_sum_2d_async(data, width, height))
    }

    /// 2D Tiled Sum Reduction on GPU (async, works on all platforms)
    pub async fn tiled_sum_2d_async(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Result<f32, String> {
        self.tiled_reduce_2d_async(
            data,
            width,
            height,
            shaders::TILED_SUM_REDUCTION_SHADER,
            "TiledSum",
            0.0, // identity for sum
            |partials| partials.iter().sum(),
        )
        .await
    }

    /// 2D Tiled Max Reduction on GPU (sync, native only)
    ///
    /// Uses 16x16 workgroups for efficient parallel max reduction.
    /// GPU version of `tiled_max_2d`.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn tiled_max_2d(&self, data: &[f32], width: usize, height: usize) -> Result<f32, String> {
        runtime::block_on(self.tiled_max_2d_async(data, width, height))
    }

    /// 2D Tiled Max Reduction on GPU (async, works on all platforms)
    pub async fn tiled_max_2d_async(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Result<f32, String> {
        self.tiled_reduce_2d_async(
            data,
            width,
            height,
            shaders::TILED_MAX_REDUCTION_SHADER,
            "TiledMax",
            f32::NEG_INFINITY, // identity for max
            |partials| partials.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        )
        .await
    }

    /// 2D Tiled Min Reduction on GPU (sync, native only)
    ///
    /// Uses 16x16 workgroups for efficient parallel min reduction.
    /// GPU version of `tiled_min_2d`.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn tiled_min_2d(&self, data: &[f32], width: usize, height: usize) -> Result<f32, String> {
        runtime::block_on(self.tiled_min_2d_async(data, width, height))
    }

    /// 2D Tiled Min Reduction on GPU (async, works on all platforms)
    pub async fn tiled_min_2d_async(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Result<f32, String> {
        self.tiled_reduce_2d_async(
            data,
            width,
            height,
            shaders::TILED_MIN_REDUCTION_SHADER,
            "TiledMin",
            f32::INFINITY, // identity for min
            |partials| partials.iter().copied().fold(f32::INFINITY, f32::min),
        )
        .await
    }
}
