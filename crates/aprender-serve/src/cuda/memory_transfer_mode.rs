
// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_mode_estimated_speedup() {
        assert!((TransferMode::Pageable.estimated_speedup() - 1.0).abs() < f64::EPSILON);
        assert!((TransferMode::Pinned.estimated_speedup() - 1.7).abs() < f64::EPSILON);
        assert!((TransferMode::ZeroCopy.estimated_speedup() - 2.0).abs() < f64::EPSILON);
        assert!((TransferMode::Async.estimated_speedup() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_transfer_mode_eq() {
        assert_eq!(TransferMode::Pageable, TransferMode::Pageable);
        assert_ne!(TransferMode::Pageable, TransferMode::Pinned);
    }

    #[test]
    fn test_transfer_mode_clone_copy() {
        let mode = TransferMode::Pinned;
        let cloned = mode;
        assert_eq!(cloned, TransferMode::Pinned);
    }

    #[test]
    fn test_transfer_mode_debug() {
        let debug_str = format!("{:?}", TransferMode::ZeroCopy);
        assert!(debug_str.contains("ZeroCopy"));
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_pool_hit_rate_calculation() {
        let mut pool = GpuMemoryPool::new();

        // Add a buffer
        let handle = GpuBufferHandle {
            size: 4096,
            in_use: false,
        };
        pool.return_buffer(handle);

        // First get should hit (from pool)
        let _ = pool.try_get(4096);

        // Second get should miss (pool empty)
        let _ = pool.try_get(4096);

        let stats = pool.stats();
        assert_eq!(stats.pool_hits, 1);
        assert_eq!(stats.pool_misses, 1);
        assert!((stats.hit_rate - 0.5).abs() < 0.001);
    }
include!("memory_size_class.rs");
}
