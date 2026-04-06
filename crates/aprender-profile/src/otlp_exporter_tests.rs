use super::*;

#[test]
#[cfg(feature = "otlp")]
fn test_otlp_config_creation() {
    let config = OtlpConfig::new("http://localhost:4317".to_string(), "test-service".to_string());

    assert_eq!(config.endpoint, "http://localhost:4317");
    assert_eq!(config.service_name, "test-service");
    assert_eq!(config.batch_size, 512); // Sprint 36: default batch size
    assert_eq!(config.batch_delay_ms, 1000); // Sprint 36: default delay
    assert_eq!(config.queue_size, 2048); // Sprint 36: default queue size
}

#[test]
#[cfg(feature = "otlp")]
fn test_otlp_config_presets() {
    // Test balanced preset
    let balanced = OtlpConfig::balanced("http://localhost:4317".to_string(), "test".to_string());
    assert_eq!(balanced.batch_size, 512);
    assert_eq!(balanced.batch_delay_ms, 1000);

    // Test aggressive preset
    let aggressive =
        OtlpConfig::aggressive("http://localhost:4317".to_string(), "test".to_string());
    assert_eq!(aggressive.batch_size, 2048);
    assert_eq!(aggressive.batch_delay_ms, 5000);

    // Test low-latency preset
    let low_latency =
        OtlpConfig::low_latency("http://localhost:4317".to_string(), "test".to_string());
    assert_eq!(low_latency.batch_size, 128);
    assert_eq!(low_latency.batch_delay_ms, 100);
}

#[test]
#[cfg(feature = "otlp")]
fn test_otlp_config_builder() {
    let config = OtlpConfig::new("http://localhost:4317".to_string(), "test".to_string())
        .with_batch_size(1024)
        .with_batch_delay_ms(2000)
        .with_queue_size(4096);

    assert_eq!(config.batch_size, 1024);
    assert_eq!(config.batch_delay_ms, 2000);
    assert_eq!(config.queue_size, 4096);
}

#[test]
#[cfg(not(feature = "otlp"))]
fn test_otlp_disabled_returns_error() {
    let config = OtlpConfig::new("http://localhost:4317".to_string(), "test".to_string());

    let result = OtlpExporter::new(config, None);
    assert!(result.is_err());
}

// Sprint 40: UnifiedTrace export tests (Section 7.1)

#[test]
#[cfg(feature = "otlp")]
fn test_export_unified_trace_basic() {
    use crate::unified_trace::UnifiedTrace;

    // Create basic unified trace
    let trace = UnifiedTrace::new(1234, "test_program".to_string());

    // Create OTLP exporter
    let config = OtlpConfig::new("http://localhost:4317".to_string(), "test".to_string());
    let mut exporter = OtlpExporter::new(config, None).expect("Failed to create exporter");

    // Export should succeed
    let result = exporter.export_unified_trace(&trace);
    assert!(result.is_ok(), "Export should succeed for basic trace");
}

#[test]
#[cfg(feature = "otlp")]
fn test_export_unified_trace_with_syscalls() {
    use crate::unified_trace::{SyscallSpan, UnifiedTrace};
    use std::borrow::Cow;

    // Create trace with syscalls
    let mut trace = UnifiedTrace::new(1234, "test_program".to_string());

    // Add syscall spans
    let syscall1 = SyscallSpan::new(
        trace.process_span.span_id,
        Cow::Borrowed("open"),
        vec![],
        0,
        1000,
        500,
        None,
        &trace.clock,
    );
    trace.syscall_spans.push(syscall1);

    let syscall2 = SyscallSpan::new(
        trace.process_span.span_id,
        Cow::Borrowed("read"),
        vec![],
        512,
        1500,
        200,
        None,
        &trace.clock,
    );
    trace.syscall_spans.push(syscall2);

    // Create OTLP exporter
    let config = OtlpConfig::new("http://localhost:4317".to_string(), "test".to_string());
    let mut exporter = OtlpExporter::new(config, None).expect("Failed to create exporter");

    // Export should succeed
    let result = exporter.export_unified_trace(&trace);
    assert!(result.is_ok(), "Export should succeed with syscalls");
}

#[test]
#[cfg(feature = "otlp")]
fn test_export_unified_trace_multi_layer() {
    use crate::decision_trace::DecisionTrace;
    use crate::unified_trace::{SyscallSpan, UnifiedTrace};
    use std::borrow::Cow;

    // Create trace with multiple layers
    let mut trace = UnifiedTrace::new(1234, "test_program".to_string());

    // Add syscall
    let syscall = SyscallSpan::new(
        trace.process_span.span_id,
        Cow::Borrowed("write"),
        vec![],
        1024,
        2000,
        300,
        None,
        &trace.clock,
    );
    trace.syscall_spans.push(syscall);

    // Add GPU kernel
    let gpu_kernel = GpuKernel {
        kernel: "matrix_multiply".to_string(),
        duration_us: 150,
        backend: "wgpu",
        workgroup_size: Some("`[256,1,1]`".to_string()),
        elements: Some(1000000),
        is_slow: true,
    };
    trace.gpu_spans.push(gpu_kernel);

    // Add SIMD block
    let simd_block = ComputeBlock {
        operation: "calculate_statistics",
        duration_us: 75,
        elements: 50000,
        is_slow: false,
    };
    trace.simd_spans.push(simd_block);

    // Add transpiler decision
    let decision = DecisionTrace {
        category: "optimization".to_string(),
        name: "vectorize_loop".to_string(),
        input: serde_json::json!({}),
        result: Some(serde_json::json!("success")),
        timestamp_us: 1000,
        source_location: None,
        decision_id: None,
    };
    trace.transpiler_spans.push(decision);

    // Create OTLP exporter
    let config = OtlpConfig::new("http://localhost:4317".to_string(), "test".to_string());
    let mut exporter = OtlpExporter::new(config, None).expect("Failed to create exporter");

    // Export should succeed
    let result = exporter.export_unified_trace(&trace);
    assert!(result.is_ok(), "Export should succeed with multi-layer trace");
}

#[test]
#[cfg(feature = "otlp")]
fn test_export_unified_trace_with_gpu_transfers() {
    use crate::unified_trace::UnifiedTrace;

    // Create trace with GPU memory transfers
    let mut trace = UnifiedTrace::new(1234, "test_program".to_string());

    // Add GPU memory transfer
    let transfer = GpuMemoryTransfer::new(
        "mesh_upload".to_string(),
        TransferDirection::CpuToGpu,
        1_048_576, // 1 MB
        5000,      // 5ms
        Some("VERTEX".to_string()),
        100,
    );
    trace.gpu_memory_transfers.push(transfer);

    // Create OTLP exporter
    let config = OtlpConfig::new("http://localhost:4317".to_string(), "test".to_string());
    let mut exporter = OtlpExporter::new(config, None).expect("Failed to create exporter");

    // Export should succeed
    let result = exporter.export_unified_trace(&trace);
    assert!(result.is_ok(), "Export should succeed with GPU memory transfers");
}

#[test]
#[cfg(feature = "otlp")]
fn test_export_unified_trace_preserves_happens_before() {
    use crate::unified_trace::{SyscallSpan, UnifiedTrace};
    use std::borrow::Cow;

    // Create trace with causal relationships
    let mut trace = UnifiedTrace::new(1234, "test_program".to_string());

    // Add parent syscall (open)
    let parent = SyscallSpan::new(
        trace.process_span.span_id,
        Cow::Borrowed("open"),
        vec![],
        3, // fd
        1000,
        500,
        None,
        &trace.clock,
    );
    let parent_span_id = parent.span_id;
    trace.syscall_spans.push(parent);

    // Add child syscall (read from fd 3)
    let child = SyscallSpan::new(
        parent_span_id, // Parent is the open syscall
        Cow::Borrowed("read"),
        vec![],
        512,
        1500,
        200,
        None,
        &trace.clock,
    );
    let child_span_id = child.span_id;
    trace.syscall_spans.push(child);

    // Verify happens-before relationship exists
    assert!(
        trace.happens_before(parent_span_id, child_span_id),
        "Parent should happen before child"
    );

    // Create OTLP exporter
    let config = OtlpConfig::new("http://localhost:4317".to_string(), "test".to_string());
    let mut exporter = OtlpExporter::new(config, None).expect("Failed to create exporter");

    // Export should succeed and preserve causal ordering
    let result = exporter.export_unified_trace(&trace);
    assert!(result.is_ok(), "Export should succeed with happens-before relationships");
}
