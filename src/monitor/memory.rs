//! GPU memory metrics (TRUENO-SPEC-010 Section 4.1.2)

/// GPU memory metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuMemoryMetrics {
    /// Total VRAM in bytes
    pub total: u64,
    /// Used VRAM in bytes
    pub used: u64,
    /// Free VRAM in bytes
    pub free: u64,
    /// Number of active allocations (if tracked)
    pub allocations: u64,
}

impl GpuMemoryMetrics {
    /// Create new memory metrics
    #[must_use]
    pub const fn new(total: u64, used: u64, free: u64) -> Self {
        Self { total, used, free, allocations: 0 }
    }

    /// Calculate usage percentage (0.0 - 100.0)
    #[must_use]
    pub fn usage_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.used as f64 / self.total as f64) * 100.0
        }
    }

    /// Get used VRAM in megabytes
    #[must_use]
    pub fn used_mb(&self) -> u64 {
        self.used / (1024 * 1024)
    }

    /// Get free VRAM in megabytes
    #[must_use]
    pub fn free_mb(&self) -> u64 {
        self.free / (1024 * 1024)
    }
}
