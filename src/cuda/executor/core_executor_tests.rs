
#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Constructor Tests
    // ========================================================================

    #[test]
    fn test_new_device_0() {
        let result = CudaExecutor::new(0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_new_invalid_device() {
        // Device 999 almost certainly doesn't exist
        let result = CudaExecutor::new(999);
        assert!(result.is_err());
    }

    // ========================================================================
    // Availability Tests
    // ========================================================================

    #[test]
    fn test_is_available() {
        // On a CUDA-enabled system, this should return true
        let available = CudaExecutor::is_available();
        // We can't assert true unconditionally, but we know this test runs with CUDA
        let _ = available; // Just verify it can be queried
    }

    #[test]
    fn test_num_devices() {
        let count = CudaExecutor::num_devices();
        // Should be at least 1 on CUDA system
        assert!(count >= 1);
    }

    // ========================================================================
    // Context and Thread Safety Tests
    // ========================================================================

    #[test]
    fn test_make_current() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(exec.make_current().is_ok());
    }

    #[test]
    fn test_context_accessor() {
        let Some(exec) = create_executor() else {
            return;
        };
        let _ctx = exec.context();
        // Just verify it returns a reference without panicking
    }

    #[test]
    fn test_compute_stream_accessor() {
        let Some(exec) = create_executor() else {
            return;
        };
        let _stream = exec.compute_stream();
    }

    // ========================================================================
    // Profiler Tests
    // ========================================================================

    #[test]
    fn test_enable_disable_profiling() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        assert!(!exec.is_profiling_enabled());
        exec.enable_profiling();
        assert!(exec.is_profiling_enabled());
        exec.disable_profiling();
        assert!(!exec.is_profiling_enabled());
    }

    #[test]
    fn test_profiler_accessor() {
        let Some(exec) = create_executor() else {
            return;
        };
        let _profiler = exec.profiler();
    }

    #[test]
    fn test_profiler_mut_accessor() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let _profiler = exec.profiler_mut();
    }

    #[test]
    fn test_reset_profiler() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.reset_profiler();
    }

    #[test]
    fn test_profiler_summary() {
        let Some(exec) = create_executor() else {
            return;
        };
        let summary = exec.profiler_summary();
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_profiler_sync_mode() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let initial_mode = exec.profiler_sync_mode();
        exec.set_profiler_sync_mode(trueno::SyncMode::Deferred);
        assert_eq!(exec.profiler_sync_mode(), trueno::SyncMode::Deferred);
        exec.set_profiler_sync_mode(initial_mode);
    }

    #[test]
    fn test_profiler_category_stats() {
        let Some(exec) = create_executor() else {
            return;
        };
        let stats = exec.profiler_category_stats();
        assert_eq!(stats.len(), trueno::BrickCategory::COUNT);
    }

    // ========================================================================
    // Execution Graph Tests
    // ========================================================================

    #[test]
    fn test_enable_disable_graph_tracking() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        assert!(!exec.is_graph_tracking_enabled());
        exec.enable_graph_tracking();
        assert!(exec.is_graph_tracking_enabled());
        exec.disable_graph_tracking();
        assert!(!exec.is_graph_tracking_enabled());
    }

    #[test]
    fn test_execution_graph_accessor() {
        let Some(exec) = create_executor() else {
            return;
        };
        let _graph = exec.execution_graph();
    }

    #[test]
    fn test_execution_graph_ascii() {
        let Some(exec) = create_executor() else {
            return;
        };
        let ascii = exec.execution_graph_ascii();
        // Empty graph, but should still return something
        let _ = ascii;
    }

    #[test]
    fn test_clear_execution_graph() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.clear_execution_graph();
    }

    // ========================================================================
    // Tile Profiling Tests
    // ========================================================================

    #[test]
    fn test_enable_disable_tile_profiling() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        assert!(!exec.is_tile_profiling_enabled());
        exec.enable_tile_profiling();
        assert!(exec.is_tile_profiling_enabled());
        exec.disable_tile_profiling();
        assert!(!exec.is_tile_profiling_enabled());
    }

    #[test]
    fn test_tile_stats() {
        let Some(exec) = create_executor() else {
            return;
        };
        let _stats = exec.tile_stats(trueno::TileLevel::Macro);
    }

    #[test]
    fn test_tile_summary() {
        let Some(exec) = create_executor() else {
            return;
        };
        let summary = exec.tile_summary();
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_tile_stats_json() {
        let Some(exec) = create_executor() else {
            return;
        };
        let json = exec.tile_stats_json();
        assert!(json.contains('{'));
    }

    #[test]
    fn test_reset_tile_stats() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.reset_tile_stats();
    }

    // ========================================================================
    // Device Info Tests
    // ========================================================================

    #[test]
    fn test_device_name() {
        let Some(exec) = create_executor() else {
            return;
        };
        let result = exec.device_name();
        assert!(result.is_ok());
        let name = result.unwrap();
        assert!(!name.is_empty());
    }

    #[test]
    fn test_memory_info() {
        let Some(exec) = create_executor() else {
            return;
        };
        let result = exec.memory_info();
        assert!(result.is_ok());
        let (free, total) = result.unwrap();
        assert!(total > 0);
        assert!(free <= total);
    }

    // ========================================================================
    // QWEN-010: Optimal Tile Size Tests
    // ========================================================================

    #[test]
    fn test_optimal_tile_size() {
        let Some(exec) = create_executor() else {
            return;
        };
        let tile_size = exec.optimal_tile_size();
        // Should be either 32 or 64 depending on GPU
        assert!(tile_size == 32 || tile_size == 64);

        // For RTX 4090 (our development GPU), should be 64
        if let Ok(name) = exec.device_name() {
            if name.contains("4090") {
                assert_eq!(tile_size, 64, "RTX 4090 should use 64x64 tiles");
            }
        }
    }

    // ========================================================================
    // Synchronization Tests
    // ========================================================================

    #[test]
    fn test_synchronize() {
        let Some(exec) = create_executor() else {
            return;
        };
        assert!(exec.synchronize().is_ok());
    }

    // ========================================================================
    // Memory Pool Tests
    // ========================================================================

    #[test]
    fn test_pool_stats() {
        let Some(exec) = create_executor() else {
            return;
        };
        let stats = exec.pool_stats();
        let _ = stats.pool_hits; // Verify stats are accessible
    }

    #[test]
    fn test_staging_pool_stats() {
        let Some(exec) = create_executor() else {
            return;
        };
        let stats = exec.staging_pool_stats();
        let _ = stats.pool_hits; // Verify stats are accessible
    }

    #[test]
    fn test_get_return_staging_buffer() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        let buf = exec.get_staging_buffer(1024);
        exec.return_staging_buffer(buf);
    }

    #[test]
    fn test_clear_pool() {
        let Some(mut exec) = create_executor() else {
            return;
        };
        exec.clear_pool();
    }
}
