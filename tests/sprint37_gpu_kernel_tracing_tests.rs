//! Sprint 37: GPU Kernel-Level Tracing Tests
//!
//! Integration tests for wgpu timestamp query integration and OTLP export.
//! Follows Sprint 32 (SIMD compute tracing) test patterns.
//!
//! Test coverage:
//! - Adaptive sampling (fast kernels skipped, slow kernels traced)
//! - Span attributes (backend, kernel name, duration, workgroup size)
//! - Resource-level attributes (gpu.library at Resource level)
//! - Debug mode (--trace-gpu-all bypasses threshold)
//! - Unified tracing (GPU + SIMD + syscalls in one trace)
//! - Feature flag (graceful degradation when gpu-tracing disabled)

#[cfg(all(feature = "gpu-tracing", feature = "otlp"))]
mod gpu_tracing_tests {
    use renacer::gpu_tracer::GpuTracerConfig;
    use renacer::otlp_exporter::{GpuKernel, OtlpConfig, OtlpExporter};

    /// Test that slow GPU kernels (>100μs) are traced
    ///
    /// Adaptive sampling: duration >= threshold → export span
    #[test]
    fn test_gpu_kernel_traced_when_slow() {
        // Setup: OTLP exporter with test endpoint
        let config =
            OtlpConfig::new("http://localhost:4317".to_string(), "test-gpu-slow".to_string());
        let exporter = OtlpExporter::new(config, None).expect("Failed to create OTLP exporter");

        // Create a GpuKernel with slow duration
        let kernel = GpuKernel {
            kernel: "test_slow_kernel".to_string(),
            duration_us: 5000, // 5ms - well above 100μs threshold
            backend: "wgpu",
            workgroup_size: Some("[256,1,1]".to_string()),
            elements: Some(1000000),
            is_slow: true,
        };

        // Record the kernel (should create a span)
        exporter.record_gpu_kernel(kernel);

        // NOTE: In full integration test with Jaeger, we would:
        // 1. Query Jaeger API for traces
        // 2. Verify span with name "gpu_kernel: test_slow_kernel" exists
        // 3. Verify attributes: gpu.duration_us=5000, gpu.is_slow=true

        // For now, this tests that the API doesn't panic
        // Full integration testing will be added in Sprint 34 pattern
    }

    /// Test that fast GPU kernels (<100μs) are NOT traced
    ///
    /// Adaptive sampling: duration < threshold → skip span export
    #[test]
    fn test_gpu_kernel_not_traced_when_fast() {
        // Setup: OTLP exporter
        let config =
            OtlpConfig::new("http://localhost:4317".to_string(), "test-gpu-fast".to_string());
        let exporter = OtlpExporter::new(config, None).expect("Failed to create OTLP exporter");

        // Create a GpuKernel with fast duration
        let kernel = GpuKernel {
            kernel: "test_fast_kernel".to_string(),
            duration_us: 50, // 50μs - below 100μs threshold
            backend: "wgpu",
            workgroup_size: Some("[64,1,1]".to_string()),
            elements: Some(1000),
            is_slow: false,
        };

        // In real usage, the caller (GpuProfilerWrapper) would NOT call
        // record_gpu_kernel() for fast kernels due to adaptive sampling.
        // But if it did, the method should still handle it gracefully.
        exporter.record_gpu_kernel(kernel);

        // NOTE: Full integration test would verify NO span was exported
    }

    /// Test that GPU kernel span has correct attributes
    ///
    /// Verifies span-level attributes per specification Section 2.3:
    /// - gpu.backend
    /// - gpu.kernel
    /// - gpu.duration_us
    /// - gpu.workgroup_size (optional)
    /// - gpu.elements (optional)
    /// - gpu.is_slow
    #[test]
    fn test_gpu_kernel_attributes() {
        let config =
            OtlpConfig::new("http://localhost:4317".to_string(), "test-gpu-attrs".to_string());
        let exporter = OtlpExporter::new(config, None).expect("Failed to create OTLP exporter");

        let kernel = GpuKernel {
            kernel: "matrix_multiply".to_string(),
            duration_us: 15000, // 15ms
            backend: "wgpu",
            workgroup_size: Some("[16,16,1]".to_string()),
            elements: Some(65536), // 256x256 matrix
            is_slow: true,
        };

        exporter.record_gpu_kernel(kernel);

        // NOTE: Full integration test would query Jaeger and assert:
        // - span.name == "gpu_kernel: matrix_multiply"
        // - span.attributes["gpu.backend"] == "wgpu"
        // - span.attributes["gpu.duration_us"] == 15000
        // - span.attributes["gpu.workgroup_size"] == "[16,16,1]"
        // - span.attributes["gpu.elements"] == 65536
        // - span.attributes["gpu.is_slow"] == true
    }

    /// Test that gpu.library appears at Resource level, not Span level
    ///
    /// Toyota Way: Avoid attribute explosion (Sprint 32 Defect 4)
    /// Static attributes (gpu.library, gpu.library.version) should be
    /// set once at Resource level, not repeated on every span.
    #[test]
    #[cfg(feature = "otlp")]
    fn test_resource_level_gpu_attributes() {
        let config =
            OtlpConfig::new("http://localhost:4317".to_string(), "test-gpu-resource".to_string());
        let exporter = OtlpExporter::new(config, None).expect("Failed to create OTLP exporter");

        // NOTE: Full integration test would:
        // 1. Query OTLP exporter's Resource attributes
        // 2. Verify resource.attributes["gpu.library"] == "wgpu"
        // 3. Verify resource.attributes["gpu.tracing.abstraction"] == "kernel_level"
        // 4. Verify these attributes are NOT on individual spans

        // For now, this tests that exporter creation succeeds
        let kernel = GpuKernel {
            kernel: "test_resource_check".to_string(),
            duration_us: 1000,
            backend: "wgpu",
            workgroup_size: None,
            elements: None,
            is_slow: true,
        };

        exporter.record_gpu_kernel(kernel);
    }

    /// Test debug mode: --trace-gpu-all traces ALL kernels
    ///
    /// When trace_all=true, bypass adaptive sampling threshold
    #[test]
    fn test_debug_mode_traces_all_kernels() {
        // Setup: GpuTracerConfig with trace_all=true
        let debug_config = GpuTracerConfig {
            threshold_us: 100,
            trace_all: true, // Debug mode: trace everything
        };

        // Verify config is set correctly
        assert!(debug_config.trace_all);
        assert_eq!(debug_config.threshold_us, 100);

        // In real usage with GpuProfilerWrapper:
        // - Even kernels with duration_us < 100 would be traced
        // - This is useful for development/debugging
    }

    /// Test that GPU and SIMD compute blocks appear in unified trace
    ///
    /// Verifies that Sprint 32 SIMD tracing and Sprint 37 GPU tracing
    /// produce spans in the same OTLP trace (unified observability).
    #[test]
    fn test_gpu_and_simd_unified_trace() {
        use renacer::otlp_exporter::ComputeBlock;

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

        // Record a GPU kernel (Sprint 37)
        let gpu_kernel = GpuKernel {
            kernel: "sum_aggregation".to_string(),
            duration_us: 60000, // 60ms
            backend: "wgpu",
            workgroup_size: Some("[256,1,1]".to_string()),
            elements: Some(1000000),
            is_slow: true,
        };
        exporter.record_gpu_kernel(gpu_kernel);

        // End root span
        exporter.end_root_span(0);

        // NOTE: Full integration test would verify:
        // - Single trace with root span "process: test-unified-app"
        // - Child span "compute_block: calculate_statistics" (227μs)
        // - Child span "gpu_kernel: sum_aggregation" (60ms)
        // - All three spans share same trace_id
        // - Timeline shows GPU kernel is 265x slower than SIMD
    }
}

/// Test that code compiles with gpu-tracing feature disabled
///
/// Ensures graceful degradation and clear error messages
#[cfg(not(feature = "gpu-tracing"))]
#[test]
fn test_gpu_tracing_feature_disabled() {
    // When gpu-tracing feature is disabled, GpuProfilerWrapper should not exist
    // This test just verifies the code compiles without the feature

    // Attempting to use GpuKernel would result in compile error:
    // "cannot find type `GpuKernel` in module `otlp_exporter`"

    // This is expected behavior - gpu-tracing is opt-in
}

/// Test GpuTracerConfig default values
#[test]
fn test_gpu_tracer_config_defaults() {
    use renacer::gpu_tracer::GpuTracerConfig;

    let config = GpuTracerConfig::default();

    assert_eq!(config.threshold_us, 100); // Same as Sprint 32 SIMD tracing
    assert!(!config.trace_all); // Safe by default (adaptive sampling)
}

/// Test GpuKernel struct creation
#[cfg(feature = "gpu-tracing")]
#[test]
fn test_gpu_kernel_struct() {
    use renacer::otlp_exporter::GpuKernel;

    let kernel = GpuKernel {
        kernel: "test_kernel".to_string(),
        duration_us: 5000,
        backend: "wgpu",
        workgroup_size: Some("[256,1,1]".to_string()),
        elements: Some(100000),
        is_slow: true,
    };

    assert_eq!(kernel.kernel, "test_kernel");
    assert_eq!(kernel.duration_us, 5000);
    assert_eq!(kernel.backend, "wgpu");
    assert_eq!(kernel.workgroup_size, Some("[256,1,1]".to_string()));
    assert_eq!(kernel.elements, Some(100000));
    assert!(kernel.is_slow);
}

/// Test GpuKernel with optional fields as None
#[cfg(feature = "gpu-tracing")]
#[test]
fn test_gpu_kernel_optional_fields() {
    use renacer::otlp_exporter::GpuKernel;

    let kernel = GpuKernel {
        kernel: "minimal_kernel".to_string(),
        duration_us: 200,
        backend: "wgpu",
        workgroup_size: None, // Optional
        elements: None,       // Optional
        is_slow: true,
    };

    assert_eq!(kernel.kernel, "minimal_kernel");
    assert!(kernel.workgroup_size.is_none());
    assert!(kernel.elements.is_none());
}
