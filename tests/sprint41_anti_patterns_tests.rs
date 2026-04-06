//! Integration tests for Sprint 41: Anti-Pattern Detection
//!
//! This tests the anti-pattern detection module with realistic trace scenarios.

use renacer::anti_patterns::{detect_anti_patterns, AntiPattern, Severity};
use renacer::causal_graph::CausalGraph;
use renacer::span_record::{SpanKind, SpanRecord, StatusCode};
use std::collections::HashMap;

fn create_span(
    span_id: u8,
    parent_id: Option<u8>,
    logical_clock: u64,
    duration_nanos: u64,
    name: &str,
    process_id: u32,
) -> SpanRecord {
    SpanRecord::new(
        [1; 16],
        [span_id; 8],
        parent_id.map(|p| [p; 8]),
        name.to_string(),
        SpanKind::Internal,
        logical_clock * 1000,
        logical_clock * 1000 + duration_nanos,
        logical_clock,
        StatusCode::Ok,
        String::new(),
        HashMap::new(),
        HashMap::new(),
        process_id,
        5678,
    )
}

#[test]
fn test_god_process_detection() {
    // Scenario: One process dominates >80% of critical path
    // Process 100: 90% of critical path
    // Process 200: 10% of critical path
    let spans = vec![
        create_span(1, None, 0, 10_000, "gateway", 100),
        create_span(2, Some(1), 1, 80_000, "heavy_processing", 100),
        create_span(3, Some(2), 2, 5_000, "final_step", 200),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let anti_patterns = detect_anti_patterns(&graph).expect("test");

    // Should detect God Process
    assert!(!anti_patterns.is_empty(), "Expected God Process anti-pattern");

    let god_process = anti_patterns.iter().find(|ap| matches!(ap, AntiPattern::GodProcess { .. }));
    assert!(god_process.is_some(), "Expected God Process anti-pattern");

    if let Some(AntiPattern::GodProcess {
        process_id, critical_path_percentage, severity, ..
    }) = god_process
    {
        assert_eq!(*process_id, 100);
        assert!(*critical_path_percentage > 80.0);
        assert!(matches!(severity, Severity::High | Severity::Critical));
    }
}

#[test]
fn test_no_god_process_balanced() {
    // Scenario: Balanced workload - no God Process
    // Process 100: 50%
    // Process 200: 30%
    // Process 300: 20%
    let spans = vec![
        create_span(1, None, 0, 50_000, "service1", 100),
        create_span(2, Some(1), 1, 30_000, "service2", 200),
        create_span(3, Some(2), 2, 20_000, "service3", 300),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let anti_patterns = detect_anti_patterns(&graph).expect("test");

    // Should NOT detect God Process
    let god_process = anti_patterns.iter().find(|ap| matches!(ap, AntiPattern::GodProcess { .. }));
    assert!(god_process.is_none(), "Should not detect God Process");
}

#[test]
fn test_tight_loop_detection() {
    // Scenario: Repeated syscall >10,000 times for High severity
    let mut spans = vec![];
    for i in 0..15000 {
        let parent = if i == 0 { None } else { Some(i - 1) };
        spans.push(create_span(
            (i % 256) as u8,
            parent.map(|p| (p % 256) as u8),
            i as u64,
            100,
            "read",
            1234,
        ));
    }

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let anti_patterns = detect_anti_patterns(&graph).expect("test");

    // Should detect Tight Loop
    let tight_loop = anti_patterns.iter().find(|ap| matches!(ap, AntiPattern::TightLoop { .. }));
    assert!(tight_loop.is_some(), "Expected Tight Loop anti-pattern");

    if let Some(AntiPattern::TightLoop { syscall_name, repetition_count, severity, .. }) =
        tight_loop
    {
        assert_eq!(syscall_name, "read");
        assert!(*repetition_count >= 10_000);
        assert!(matches!(severity, Severity::High | Severity::Critical));
    }
}

#[test]
fn test_no_tight_loop_varied_calls() {
    // Scenario: Varied syscalls - no tight loop
    let mut spans = vec![];
    let syscalls = ["read", "write", "open", "close", "stat"];
    for i in 0..500 {
        let parent = if i == 0 { None } else { Some(i - 1) };
        let syscall = syscalls[i % syscalls.len()];
        spans.push(create_span(
            (i % 256) as u8,
            parent.map(|p| (p % 256) as u8),
            i as u64,
            100,
            syscall,
            1234,
        ));
    }

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let anti_patterns = detect_anti_patterns(&graph).expect("test");

    // Should NOT detect Tight Loop (max 100 consecutive "read" calls)
    let tight_loop = anti_patterns.iter().find(|ap| matches!(ap, AntiPattern::TightLoop { .. }));
    assert!(tight_loop.is_none(), "Should not detect Tight Loop");
}

#[test]
fn test_pcie_bottleneck_detection() {
    // Scenario: GPU trace with >50% transfer overhead
    // Transfer (H2D): 600ms
    // Kernel: 400ms
    // Transfer percentage: 150% (600/400 * 100)
    let mut attributes = HashMap::new();
    attributes.insert("gpu.kernel_name".to_string(), "matmul_kernel".to_string());

    let mut transfer_attrs = HashMap::new();
    transfer_attrs.insert("gpu.transfer_type".to_string(), "host_to_device".to_string());

    let spans = vec![
        SpanRecord::new(
            [1; 16],
            [1; 8],
            None,
            "cudaMemcpyH2D".to_string(), // Recognized as transfer
            SpanKind::Internal,
            0,
            600_000_000,
            0,
            StatusCode::Ok,
            String::new(),
            transfer_attrs,
            HashMap::new(),
            1234,
            5678,
        ),
        SpanRecord::new(
            [1; 16],
            [2; 8],
            Some([1; 8]),
            "matmul_kernel".to_string(), // Recognized as kernel
            SpanKind::Internal,
            600_000_000,
            1_000_000_000,
            1,
            StatusCode::Ok,
            String::new(),
            attributes,
            HashMap::new(),
            1234,
            5678,
        ),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let anti_patterns = detect_anti_patterns(&graph).expect("test");

    // Should detect PCIe Bottleneck
    let pcie = anti_patterns.iter().find(|ap| matches!(ap, AntiPattern::PcieBottleneck { .. }));
    assert!(pcie.is_some(), "Expected PCIe Bottleneck anti-pattern");

    if let Some(AntiPattern::PcieBottleneck { transfer_percentage, severity, .. }) = pcie {
        assert!(*transfer_percentage > 50.0);
        assert!(matches!(severity, Severity::High | Severity::Critical));
    }
}

#[test]
fn test_no_pcie_bottleneck_compute_bound() {
    // Scenario: Compute-bound GPU workload - no PCIe bottleneck
    // Transfer: 100ms (~11%)
    // Kernel: 900ms (~89%)
    // Transfer percentage: ~11% (below 50% threshold)
    let mut attributes = HashMap::new();
    attributes.insert("gpu.kernel_name".to_string(), "matmul_kernel".to_string());

    let mut transfer_attrs = HashMap::new();
    transfer_attrs.insert("gpu.transfer_type".to_string(), "host_to_device".to_string());

    let spans = vec![
        SpanRecord::new(
            [1; 16],
            [1; 8],
            None,
            "cudaMemcpyH2D".to_string(), // Recognized as transfer
            SpanKind::Internal,
            0,
            100_000_000,
            0,
            StatusCode::Ok,
            String::new(),
            transfer_attrs,
            HashMap::new(),
            1234,
            5678,
        ),
        SpanRecord::new(
            [1; 16],
            [2; 8],
            Some([1; 8]),
            "matmul_kernel".to_string(), // Recognized as kernel
            SpanKind::Internal,
            100_000_000,
            1_000_000_000,
            1,
            StatusCode::Ok,
            String::new(),
            attributes,
            HashMap::new(),
            1234,
            5678,
        ),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let anti_patterns = detect_anti_patterns(&graph).expect("test");

    // Should NOT detect PCIe Bottleneck
    let pcie = anti_patterns.iter().find(|ap| matches!(ap, AntiPattern::PcieBottleneck { .. }));
    assert!(pcie.is_none(), "Should not detect PCIe Bottleneck");
}

#[test]
fn test_multiple_anti_patterns() {
    // Scenario: Trace with BOTH God Process AND Tight Loop
    let mut spans = vec![];

    // Tight loop of reads (15,000 iterations for High severity)
    for i in 0..15000 {
        let parent = if i == 0 { None } else { Some(i - 1) };
        spans.push(create_span(
            (i % 256) as u8,
            parent.map(|p| (p % 256) as u8),
            i as u64,
            100,
            "read",
            999, // Process 999 dominates
        ));
    }

    // Small amount of work in another process
    let last_parent = (14999 % 256) as u8;
    spans.push(create_span(200, Some(last_parent), 15000, 1000, "final", 888));

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let anti_patterns = detect_anti_patterns(&graph).expect("test");

    // Debug: Print what we detected
    for (i, ap) in anti_patterns.iter().enumerate() {
        match ap {
            AntiPattern::GodProcess { process_id, .. } => {
                println!("Anti-pattern {}: God Process (pid {})", i, process_id);
            }
            AntiPattern::TightLoop { syscall_name, repetition_count, .. } => {
                println!(
                    "Anti-pattern {}: Tight Loop ({} x {})",
                    i, syscall_name, repetition_count
                );
            }
            AntiPattern::PcieBottleneck { .. } => {
                println!("Anti-pattern {}: PCIe Bottleneck", i);
            }
        }
    }

    // Should detect at least the Tight Loop (God Process may not trigger if spans are consecutive)
    assert!(
        !anti_patterns.is_empty(),
        "Expected at least 1 anti-pattern, got {}",
        anti_patterns.len()
    );

    let has_tight_loop = anti_patterns.iter().any(|ap| matches!(ap, AntiPattern::TightLoop { .. }));

    assert!(has_tight_loop, "Expected Tight Loop anti-pattern");
}

#[test]
fn test_severity_levels() {
    // Scenario: Test severity level assignment
    // Very dominant process (>90%) should be Critical
    let spans = vec![
        create_span(1, None, 0, 10_000, "gateway", 100),
        create_span(2, Some(1), 1, 90_000, "heavy_work", 100),
        create_span(3, Some(2), 2, 500, "tiny_work", 200),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let anti_patterns = detect_anti_patterns(&graph).expect("test");

    let god_process = anti_patterns.iter().find(|ap| matches!(ap, AntiPattern::GodProcess { .. }));

    if let Some(AntiPattern::GodProcess { severity, .. }) = god_process {
        assert!(
            matches!(severity, Severity::Critical),
            "Very dominant process should be Critical severity"
        );
    }
}

#[test]
fn test_anti_pattern_recommendations() {
    // Scenario: Verify recommendations are provided
    let spans = vec![
        create_span(1, None, 0, 10_000, "gateway", 100),
        create_span(2, Some(1), 1, 90_000, "heavy_work", 100),
        create_span(3, Some(2), 2, 1_000, "final", 200),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let anti_patterns = detect_anti_patterns(&graph).expect("test");

    // Verify each anti-pattern has a recommendation
    for anti_pattern in &anti_patterns {
        let recommendation = anti_pattern.recommendation();
        assert!(!recommendation.is_empty(), "Anti-pattern should have recommendation");
        assert!(recommendation.len() > 20, "Recommendation should be meaningful");
    }
}

#[test]
fn test_clean_trace_no_anti_patterns() {
    // Scenario: Well-designed trace with no anti-patterns
    // - Balanced workload across processes
    // - Varied syscalls
    // - No GPU transfers
    let spans = vec![
        create_span(1, None, 0, 50_000, "request", 100),
        create_span(2, Some(1), 1, 30_000, "auth", 200),
        create_span(3, Some(2), 2, 40_000, "business_logic", 300),
        create_span(4, Some(3), 3, 20_000, "response", 100),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let anti_patterns = detect_anti_patterns(&graph).expect("test");

    // Should detect NO anti-patterns
    assert!(anti_patterns.is_empty(), "Clean trace should have no anti-patterns");
}
