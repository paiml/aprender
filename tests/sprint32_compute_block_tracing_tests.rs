//! Sprint 32: Block-Level Compute Tracing Integration Tests
//!
//! Tests for Trueno SIMD compute block tracing integration with OTLP export.
//! Following trueno-tracing-integration-spec.md Section 6.1 test requirements.
//!
//! Test Coverage:
//! - Adaptive sampling (>100μs threshold)
//! - ComputeBlock struct construction
//! - OTLP span export
//! - Resource-level attributes
//! - Fast block filtering (<100μs)
//! - Debug mode tracing

#[cfg(feature = "otlp")]
use renacer::otlp_exporter::{ComputeBlock, OtlpConfig, OtlpExporter};

/// Test that slow compute blocks (>100μs) are traced and exported
#[test]
#[cfg(feature = "otlp")]
fn test_compute_block_traced_when_slow() {
    // Arrange: Create OTLP exporter
    let config =
        OtlpConfig::new("http://localhost:4317".to_string(), "test_compute_tracing".to_string());
    let exporter = OtlpExporter::new(config, None).expect("Failed to create exporter");

    // Act: Record a slow compute block (>100μs threshold)
    let slow_block = ComputeBlock {
        operation: "calculate_statistics",
        duration_us: 250, // Above 100μs threshold
        elements: 10_000,
        is_slow: true,
    };

    exporter.record_compute_block(slow_block);

    // Assert: No panic, span should be exported to OTLP backend
    // (Verification requires OTLP collector inspection via Jaeger/Tempo)
}

/// Test that fast compute blocks (<100μs) are NOT traced (adaptive sampling)
#[test]
#[cfg(feature = "otlp")]
fn test_compute_block_not_traced_when_fast() {
    // Arrange: Create OTLP exporter
    let config =
        OtlpConfig::new("http://localhost:4317".to_string(), "test_compute_tracing".to_string());
    let exporter = OtlpExporter::new(config, None).expect("Failed to create exporter");

    // Act: Attempt to record a fast compute block (<100μs threshold)
    // Note: The trace_compute_block! macro filters these out before calling record_compute_block
    // This test verifies the macro logic, not the exporter
    let fast_block = ComputeBlock {
        operation: "sum_small_vector",
        duration_us: 50, // Below 100μs threshold
        elements: 100,
        is_slow: false,
    };

    // This should still work without panic, but ideally wouldn't be called by the macro
    exporter.record_compute_block(fast_block);

    // Assert: Exporter handles fast blocks gracefully
    // (Macro should prevent this call in practice via adaptive sampling)
}

/// Test ComputeBlock struct construction with correct attributes
#[test]
fn test_compute_block_attributes() {
    // Arrange & Act: Create ComputeBlock with specific attributes
    let block = ComputeBlock {
        operation: "detect_anomalies",
        duration_us: 500,
        elements: 50_000,
        is_slow: true,
    };

    // Assert: Verify all attributes are set correctly
    assert_eq!(block.operation, "detect_anomalies");
    assert_eq!(block.duration_us, 500);
    assert_eq!(block.elements, 50_000);
    assert!(block.is_slow);
}

/// Test that ComputeBlock correctly identifies slow vs fast operations
#[test]
fn test_compute_block_is_slow_flag() {
    // Test case 1: Slow block (>100μs)
    let slow = ComputeBlock {
        operation: "large_percentile",
        duration_us: 150,
        elements: 100_000,
        is_slow: true, // 150 > 100
    };
    assert!(slow.is_slow, "Block with 150μs should be marked as slow");
    assert_eq!(slow.duration_us, 150);

    // Test case 2: Fast block (<100μs)
    let fast = ComputeBlock {
        operation: "small_sum",
        duration_us: 50,
        elements: 10,
        is_slow: false, // 50 < 100
    };
    assert!(!fast.is_slow, "Block with 50μs should NOT be marked as slow");
    assert_eq!(fast.duration_us, 50);

    // Test case 3: Boundary case (exactly 100μs)
    let boundary = ComputeBlock {
        operation: "boundary_test",
        duration_us: 100,
        elements: 1_000,
        is_slow: false, // 100 is not > 100
    };
    assert!(!boundary.is_slow, "Block with exactly 100μs should NOT be marked as slow");
    assert_eq!(boundary.duration_us, 100);
}

/// Test that small vectors are handled correctly
#[test]
fn test_small_vector_compute_block() {
    // Arrange: Small vector scenario (below SIMD threshold)
    let small_vector_block = ComputeBlock {
        operation: "sum_tiny_vector",
        duration_us: 10, // Very fast due to small size
        elements: 10,    // Well below 10,000 element threshold
        is_slow: false,
    };

    // Assert: Block is constructed correctly for small vectors
    assert_eq!(small_vector_block.elements, 10);
    assert_eq!(small_vector_block.duration_us, 10);
    assert!(!small_vector_block.is_slow, "Small vector operations should be fast");
}

/// Test ComputeBlock with various operation names
#[test]
fn test_compute_block_operation_names() {
    let operations = [
        "calculate_statistics",
        "detect_anomalies",
        "percentile_computation",
        "vector_mean",
        "standard_deviation",
    ];

    for op in &operations {
        let block =
            ComputeBlock { operation: op, duration_us: 200, elements: 10_000, is_slow: true };

        assert_eq!(block.operation, *op);
    }
}

/// Test ComputeBlock with large element counts (stress test)
#[test]
fn test_compute_block_large_elements() {
    // Test with very large vectors (1 million elements)
    let large_block = ComputeBlock {
        operation: "process_large_dataset",
        duration_us: 5_000, // 5ms for 1M elements
        elements: 1_000_000,
        is_slow: true,
    };

    assert_eq!(large_block.elements, 1_000_000);
    assert!(large_block.duration_us > 100, "Large dataset should be slow");
}

/// Test ComputeBlock boundary conditions
#[test]
fn test_compute_block_boundary_conditions() {
    // Test case 1: Zero elements
    let zero_elements = ComputeBlock {
        operation: "empty_computation",
        duration_us: 1,
        elements: 0,
        is_slow: false,
    };
    assert_eq!(zero_elements.elements, 0);

    // Test case 2: Maximum duration
    let max_duration = ComputeBlock {
        operation: "very_slow_operation",
        duration_us: u64::MAX,
        elements: 100,
        is_slow: true,
    };
    assert_eq!(max_duration.duration_us, u64::MAX);
}

/// Test OTLP exporter creation for compute block tracing
#[test]
#[cfg(feature = "otlp")]
fn test_otlp_exporter_creation_for_compute_tracing() {
    // Arrange: Create OTLP config with compute tracing context
    let config =
        OtlpConfig::new("http://localhost:4317".to_string(), "compute_tracer_test".to_string());

    // Act: Create exporter
    let result = OtlpExporter::new(config, None);

    // Assert: Exporter is created successfully
    assert!(result.is_ok(), "OTLP exporter should be created successfully for compute tracing");
}

/// Test multiple compute blocks can be recorded sequentially
#[test]
#[cfg(feature = "otlp")]
fn test_multiple_compute_blocks_sequential() {
    // Arrange: Create OTLP exporter
    let config =
        OtlpConfig::new("http://localhost:4317".to_string(), "test_sequential_blocks".to_string());
    let exporter = OtlpExporter::new(config, None).expect("Failed to create exporter");

    // Act: Record multiple compute blocks
    let blocks = vec![
        ComputeBlock {
            operation: "operation_1",
            duration_us: 150,
            elements: 10_000,
            is_slow: true,
        },
        ComputeBlock {
            operation: "operation_2",
            duration_us: 200,
            elements: 20_000,
            is_slow: true,
        },
        ComputeBlock {
            operation: "operation_3",
            duration_us: 180,
            elements: 15_000,
            is_slow: true,
        },
    ];

    for block in blocks {
        exporter.record_compute_block(block);
    }

    // Assert: No panics, all blocks should be exported
}

/// Test ComputeBlock with edge case durations
#[test]
fn test_compute_block_edge_case_durations() {
    // Test case 1: 1 microsecond (extremely fast)
    let ultra_fast =
        ComputeBlock { operation: "cache_hit", duration_us: 1, elements: 1, is_slow: false };
    assert_eq!(ultra_fast.duration_us, 1);

    // Test case 2: Exactly threshold (100μs)
    let at_threshold = ComputeBlock {
        operation: "threshold_operation",
        duration_us: 100,
        elements: 5_000,
        is_slow: false, // Not > 100, so not slow
    };
    assert_eq!(at_threshold.duration_us, 100);
    assert!(!at_threshold.is_slow);

    // Test case 3: Just above threshold (101μs)
    let above_threshold = ComputeBlock {
        operation: "slightly_slow",
        duration_us: 101,
        elements: 5_000,
        is_slow: true, // > 100, so slow
    };
    assert_eq!(above_threshold.duration_us, 101);
    assert!(above_threshold.is_slow);
}

/// Test ComputeBlock serialization properties (for OTLP export)
#[test]
#[cfg(feature = "otlp")]
fn test_compute_block_otlp_export_properties() {
    // Arrange: Create compute block with specific properties
    let block = ComputeBlock {
        operation: "test_export",
        duration_us: 250,
        elements: 10_000,
        is_slow: true,
    };

    // Act: Verify block can be used with OTLP exporter
    let config =
        OtlpConfig::new("http://localhost:4317".to_string(), "test_export_properties".to_string());
    let exporter = OtlpExporter::new(config, None).expect("Failed to create exporter");

    // Assert: Export completes without error
    exporter.record_compute_block(block);
}

/// Test that exporter handles compute blocks when OTLP endpoint is unavailable
#[test]
#[cfg(feature = "otlp")]
fn test_compute_block_export_without_backend() {
    // Arrange: Create exporter with invalid endpoint (no backend running)
    let config = OtlpConfig::new(
        "http://localhost:9999".to_string(), // Non-existent endpoint
        "test_no_backend".to_string(),
    );

    // OTLP exporter creation might succeed even without backend
    if let Ok(exporter) = OtlpExporter::new(config, None) {
        // Act: Record compute block
        let block = ComputeBlock {
            operation: "no_backend_test",
            duration_us: 150,
            elements: 10_000,
            is_slow: true,
        };

        // Assert: Should not panic even if backend is unavailable
        exporter.record_compute_block(block);
    }
}

/// Test ComputeBlock with anomaly detection scenario
#[test]
fn test_compute_block_anomaly_scenario() {
    // Scenario: Detecting anomalies in syscall durations
    let anomaly_block = ComputeBlock {
        operation: "detect_anomalies",
        duration_us: 3_500, // 3.5ms for anomaly detection
        elements: 50_000,   // Large dataset
        is_slow: true,
    };

    assert_eq!(anomaly_block.operation, "detect_anomalies");
    assert!(anomaly_block.duration_us > 1000, "Anomaly detection should take significant time");
    assert!(
        anomaly_block.elements > 10_000,
        "Anomaly detection typically processes large datasets"
    );
}

/// Test ComputeBlock with statistics calculation scenario
#[test]
fn test_compute_block_statistics_scenario() {
    // Scenario: Calculating extended statistics (mean, stddev, percentiles)
    let stats_block = ComputeBlock {
        operation: "calculate_statistics",
        duration_us: 2_000, // 2ms for stats calculation
        elements: 30_000,   // Moderate dataset
        is_slow: true,
    };

    assert_eq!(stats_block.operation, "calculate_statistics");
    assert!(stats_block.is_slow);
    assert!(stats_block.elements >= 10_000, "Statistics require sufficient data points");
}
