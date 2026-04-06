//! GPU memory management and VRAM detection
//!
//! Detects available GPU memory and manages memory limits for paging.
//! Based on Funke et al. (2018) "Pipelined Query Processing in Coprocessor Environments"

use super::GpuDevice;
use anyhow::Result;

/// Default morsel size (128MB chunks as per DuckDB/Umbra)
pub const DEFAULT_MORSEL_SIZE: usize = 128 * 1024 * 1024; // 128 MB

/// GPU memory limits and statistics
#[derive(Debug, Clone)]
pub struct GpuMemoryLimits {
    /// Total VRAM available (bytes)
    pub total_vram: u64,

    /// Maximum memory to use for graph data (70% of total to leave headroom)
    pub usable_vram: u64,

    /// Morsel size for chunking (default 128MB)
    pub morsel_size: usize,

    /// Maximum number of morsels that fit in VRAM
    pub max_morsels: usize,
}

impl GpuMemoryLimits {
    /// Detect GPU memory limits
    ///
    /// # Errors
    ///
    /// Returns error if GPU device is not available
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    pub fn detect(device: &GpuDevice) -> Result<Self> {
        // Get adapter limits from wgpu
        let limits = device.device().limits();

        // Estimate VRAM from buffer size limit (conservative estimate)
        // Most GPUs have max_buffer_size = VRAM size or similar
        let total_vram = limits.max_buffer_size;

        // Use 70% of VRAM for graph data (30% for shaders, temporaries, OS)
        let usable_vram = (total_vram as f64 * 0.7) as u64;

        let morsel_size = DEFAULT_MORSEL_SIZE;
        let max_morsels = (usable_vram as usize / morsel_size).max(1);

        Ok(Self { total_vram, usable_vram, morsel_size, max_morsels })
    }

    /// Check if graph fits entirely in VRAM
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn fits_in_vram(&self, graph_size_bytes: usize) -> bool {
        graph_size_bytes <= self.usable_vram as usize
    }

    /// Calculate number of morsels needed for given size
    #[must_use]
    pub fn morsels_needed(&self, size_bytes: usize) -> usize {
        size_bytes.div_ceil(self.morsel_size)
    }

    /// Get recommended tile size in nodes for chunking
    #[must_use]
    pub fn recommended_tile_size(&self, node_size_bytes: usize) -> usize {
        if node_size_bytes == 0 {
            return 1000; // Default fallback
        }
        (self.morsel_size / node_size_bytes).max(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_limits_detection() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_memory_limits_detection: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();
        let limits = GpuMemoryLimits::detect(&device).unwrap();

        println!("Total VRAM: {} bytes", limits.total_vram);
        println!("Usable VRAM: {} bytes", limits.usable_vram);
        println!("Max morsels: {}", limits.max_morsels);

        assert!(limits.total_vram > 0);
        assert!(limits.usable_vram > 0);
        assert!(limits.usable_vram <= limits.total_vram);
        assert!(limits.max_morsels > 0);
    }

    #[test]
    fn test_fits_in_vram() {
        let limits = GpuMemoryLimits {
            total_vram: 8 * 1024 * 1024 * 1024,  // 8GB
            usable_vram: 5 * 1024 * 1024 * 1024, // 5GB usable
            morsel_size: DEFAULT_MORSEL_SIZE,
            max_morsels: 40,
        };

        assert!(limits.fits_in_vram(100 * 1024 * 1024)); // 100MB fits
        assert!(limits.fits_in_vram(4 * 1024 * 1024 * 1024)); // 4GB fits
        assert!(!limits.fits_in_vram(6 * 1024 * 1024 * 1024)); // 6GB doesn't fit
    }

    #[test]
    fn test_morsels_needed() {
        let limits = GpuMemoryLimits {
            total_vram: 8 * 1024 * 1024 * 1024,
            usable_vram: 5 * 1024 * 1024 * 1024,
            morsel_size: DEFAULT_MORSEL_SIZE,
            max_morsels: 40,
        };

        assert_eq!(limits.morsels_needed(128 * 1024 * 1024), 1);
        assert_eq!(limits.morsels_needed(256 * 1024 * 1024), 2);
        assert_eq!(limits.morsels_needed(200 * 1024 * 1024), 2);
    }
}
