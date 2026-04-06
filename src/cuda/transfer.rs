
/// Statistics for staging buffer pool
#[derive(Debug, Clone)]
pub struct StagingPoolStats {
    /// Total bytes allocated
    pub total_allocated: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Number of buffer reuses
    pub pool_hits: usize,
    /// Number of new allocations
    pub pool_misses: usize,
    /// Number of free buffers
    pub free_buffers: usize,
    /// Hit rate (0.0 - 1.0)
    pub hit_rate: f64,
}

/// Zero-copy transfer configuration (PARITY-042)
///
/// Controls how data is transferred between host and device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransferMode {
    /// Standard pageable memory transfer
    /// - Simplest, works everywhere
    /// - Involves CPU copy to staging area
    #[default]
    Pageable,
    /// Pinned memory transfer (faster DMA)
    /// - 1.5-2x faster than pageable
    /// - Requires page-locked memory
    Pinned,
    /// Zero-copy mapped memory (no transfer)
    /// - GPU directly accesses host memory
    /// - Best for infrequent access patterns
    /// - Requires unified memory support
    ZeroCopy,
    /// Async transfer with stream overlap
    /// - Transfer while previous kernel runs
    /// - Best for pipelined workloads
    Async,
}

impl TransferMode {
    /// Check if this mode requires pinned memory
    #[must_use]
    pub fn requires_pinned(&self) -> bool {
        matches!(self, Self::Pinned | Self::ZeroCopy | Self::Async)
    }

    /// Estimated speedup vs pageable transfer
    #[must_use]
    pub fn estimated_speedup(&self) -> f64 {
        match self {
            Self::Pageable => 1.0,
            Self::Pinned => 1.7,   // ~70% faster DMA
            Self::ZeroCopy => 2.0, // No transfer overhead
            Self::Async => 1.5,    // Overlap hides latency
        }
    }
}
