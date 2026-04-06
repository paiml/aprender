//! Sprint 39: GPU Memory Transfer Tracking Tests (Phase 4)
//!
//! Integration tests for GPU memory transfer tracing (CPU↔GPU).
//! Follows Sprint 37 (Phase 1: kernel tracing) test patterns.
//!
//! Test coverage:
//! - Transfer direction tracking (CPU→GPU, GPU→CPU)
//! - Bandwidth calculation
//! - Span attributes (bytes, duration, bandwidth)
//! - Adaptive sampling (fast transfers skipped, slow transfers traced)
//! - Unified tracing (transfers + kernels in one trace)

#[cfg(all(feature = "gpu-tracing", feature = "otlp"))]
mod gpu_transfer_tests {
    use renacer::otlp_exporter::{GpuMemoryTransfer, OtlpConfig, OtlpExporter, TransferDirection};

    /// Test that CPU→GPU transfers are tracked with correct attributes
    #[test]
    fn test_cpu_to_gpu_transfer_traced() {
        // Setup: OTLP exporter with test endpoint
        let config = OtlpConfig::new(
            "http://localhost:4317".to_string(),
            "test-transfer-upload".to_string(),
        );
        let exporter = OtlpExporter::new(config, None).expect("Failed to create OTLP exporter");

        // Create a transfer record (10MB upload taking 25ms)
        let transfer = GpuMemoryTransfer::new(
            "mesh_data_upload".to_string(),
            TransferDirection::CpuToGpu,
            10485760, // 10MB
            25000,    // 25ms
            Some("VERTEX".to_string()),
            100, // threshold
        );

        // Verify bandwidth calculation
        // Expected: (10485760 bytes * 1,000,000) / (25000 μs * 1,048,576) = 400 MB/s
        assert!(
            (transfer.bandwidth_mbps - 400.0).abs() < 1.0,
            "Expected ~400 MB/s, got {}",
            transfer.bandwidth_mbps
        );
        assert!(transfer.is_slow); // 25ms > 100μs threshold

        // Record the transfer (should create a span)
        exporter.record_gpu_transfer(transfer);

        // NOTE: In full integration test with Jaeger, we would:
        // 1. Query Jaeger API for traces
        // 2. Verify span with name "gpu_transfer: mesh_data_upload" exists
        // 3. Verify attributes: direction="cpu_to_gpu", bytes=10485760, bandwidth_mbps=400
    }

    /// Test that GPU→CPU transfers are tracked
    #[test]
    fn test_gpu_to_cpu_transfer_traced() {
        // Setup: OTLP exporter
        let config = OtlpConfig::new(
            "http://localhost:4317".to_string(),
            "test-transfer-download".to_string(),
        );
        let exporter = OtlpExporter::new(config, None).expect("Failed to create OTLP exporter");

        // Create a transfer record (8MB download taking 1ms)
        let transfer = GpuMemoryTransfer::new(
            "framebuffer_readback".to_string(),
            TransferDirection::GpuToCpu,
            8388608, // 8MB
            1000,    // 1ms
            None,
            100, // threshold
        );

        // Verify bandwidth calculation
        // Expected: (8388608 * 1,000,000) / (1000 * 1,048,576) = 8000 MB/s
        assert!(
            (transfer.bandwidth_mbps - 8000.0).abs() < 10.0,
            "Expected ~8000 MB/s, got {}",
            transfer.bandwidth_mbps
        );
        assert!(transfer.is_slow); // 1ms > 100μs threshold

        // Record the transfer
        exporter.record_gpu_transfer(transfer);

        // NOTE: Full integration test would verify span attributes
    }

    /// Test that fast transfers are NOT traced (adaptive sampling)
    #[test]
    fn test_fast_transfer_not_slow() {
        // Create a fast transfer (100 bytes taking 50μs)
        let transfer = GpuMemoryTransfer::new(
            "small_upload".to_string(),
            TransferDirection::CpuToGpu,
            100, // 100 bytes
            50,  // 50μs
            None,
            100, // threshold
        );

        // Verify it's NOT marked as slow
        assert!(!transfer.is_slow); // 50μs < 100μs threshold

        // In real usage, the caller would NOT call record_gpu_transfer()
        // for fast transfers due to adaptive sampling
    }

    /// Test that large slow transfers are traced
    #[test]
    fn test_large_slow_transfer_traced() {
        let config =
            OtlpConfig::new("http://localhost:4317".to_string(), "test-large-transfer".to_string());
        let exporter = OtlpExporter::new(config, None).expect("Failed to create OTLP exporter");

        // Create a large slow transfer (100MB taking 200ms)
        let transfer = GpuMemoryTransfer::new(
            "large_data_upload".to_string(),
            TransferDirection::CpuToGpu,
            104857600, // 100MB
            200000,    // 200ms
            Some("STORAGE".to_string()),
            100, // threshold
        );

        // Verify bandwidth: (104857600 * 1,000,000) / (200000 * 1,048,576) = 500 MB/s
        assert!(
            (transfer.bandwidth_mbps - 500.0).abs() < 1.0,
            "Expected ~500 MB/s, got {}",
            transfer.bandwidth_mbps
        );
        assert!(transfer.is_slow);

        exporter.record_gpu_transfer(transfer);
    }

    /// Test bandwidth calculation edge case (zero duration)
    #[test]
    fn test_zero_duration_bandwidth() {
        let transfer = GpuMemoryTransfer::new(
            "instant_transfer".to_string(),
            TransferDirection::CpuToGpu,
            1000,
            0, // 0μs duration (edge case)
            None,
            100,
        );

        // Should not panic, bandwidth should be 0
        assert_eq!(transfer.bandwidth_mbps, 0.0);
    }

    /// Test TransferDirection string representation
    #[test]
    fn test_transfer_direction_as_str() {
        assert_eq!(TransferDirection::CpuToGpu.as_str(), "cpu_to_gpu");
        assert_eq!(TransferDirection::GpuToCpu.as_str(), "gpu_to_cpu");
    }

    /// Test that transfers and kernels can coexist in unified trace
    #[test]
    fn test_transfers_and_kernels_unified() {
        use renacer::otlp_exporter::{ComputeBlock, GpuKernel};

        let config =
            OtlpConfig::new("http://localhost:4317".to_string(), "test-unified-trace".to_string());
        let mut exporter = OtlpExporter::new(config, None).expect("Failed to create OTLP exporter");

        // Start root span for the process
        exporter.start_root_span("test-unified-app", 12345);

        // Record a SIMD compute block (Sprint 32)
        let simd_block = ComputeBlock {
            operation: "calculate_statistics",
            duration_us: 227,
            elements: 10000,
            is_slow: true,
        };
        exporter.record_compute_block(simd_block);

        // Record a memory transfer (Sprint 39 Phase 4)
        let transfer = GpuMemoryTransfer::new(
            "mesh_upload".to_string(),
            TransferDirection::CpuToGpu,
            10485760,
            25000,
            None,
            100,
        );
        exporter.record_gpu_transfer(transfer);

        // Record a GPU kernel (Sprint 37 Phase 1)
        let gpu_kernel = GpuKernel {
            kernel: "vertex_shader".to_string(),
            duration_us: 3000,
            backend: "wgpu",
            workgroup_size: Some("[256,1,1]".to_string()),
            elements: Some(100000),
            is_slow: true,
        };
        exporter.record_gpu_kernel(gpu_kernel);

        // Record another memory transfer (GPU→CPU)
        let readback = GpuMemoryTransfer::new(
            "framebuffer_readback".to_string(),
            TransferDirection::GpuToCpu,
            8388608,
            1000,
            None,
            100,
        );
        exporter.record_gpu_transfer(readback);

        // End root span
        exporter.end_root_span(0);

        // NOTE: Full integration test would verify:
        // - Single trace with root span "process: test-unified-app"
        // - Child span "compute_block: calculate_statistics" (227μs)
        // - Child span "gpu_transfer: mesh_upload" (25ms, cpu_to_gpu)
        // - Child span "gpu_kernel: vertex_shader" (3ms)
        // - Child span "gpu_transfer: framebuffer_readback" (1ms, gpu_to_cpu)
        // - All spans share same trace_id
        // - Timeline shows: SIMD (227μs) << transfer (25ms) > kernel (3ms)
    }

    /// Test GpuMemoryTransfer struct creation
    #[test]
    fn test_gpu_memory_transfer_struct() {
        let transfer = GpuMemoryTransfer::new(
            "test_transfer".to_string(),
            TransferDirection::CpuToGpu,
            1048576, // 1MB
            10000,   // 10ms
            Some("UNIFORM".to_string()),
            100,
        );

        assert_eq!(transfer.label, "test_transfer");
        assert_eq!(transfer.direction, TransferDirection::CpuToGpu);
        assert_eq!(transfer.bytes, 1048576);
        assert_eq!(transfer.duration_us, 10000);
        assert_eq!(transfer.buffer_usage, Some("UNIFORM".to_string()));
        assert!(transfer.is_slow);
        // Bandwidth: (1048576 * 1,000,000) / (10000 * 1,048,576) = 100 MB/s
        assert!((transfer.bandwidth_mbps - 100.0).abs() < 0.1);
    }
}

/// Test that code compiles without gpu-tracing feature
#[cfg(not(feature = "gpu-tracing"))]
#[test]
fn test_gpu_transfer_feature_disabled() {
    // When gpu-tracing feature is disabled, GpuMemoryTransfer should not exist
    // This test just verifies the code compiles without the feature

    // Attempting to use GpuMemoryTransfer would result in compile error:
    // "cannot find type `GpuMemoryTransfer` in module `otlp_exporter`"

    // This is expected behavior - gpu-tracing is opt-in
}

/// Test TransferDirection enum without gpu-tracing feature
#[cfg(feature = "otlp")]
#[test]
fn test_transfer_direction_enum() {
    use renacer::otlp_exporter::TransferDirection;

    let cpu_to_gpu = TransferDirection::CpuToGpu;
    let gpu_to_cpu = TransferDirection::GpuToCpu;

    assert_eq!(cpu_to_gpu.as_str(), "cpu_to_gpu");
    assert_eq!(gpu_to_cpu.as_str(), "gpu_to_cpu");
    assert_ne!(cpu_to_gpu, gpu_to_cpu);
}
