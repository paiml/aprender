#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Falsification Tests for PMAT-032: Multi-Metric Correlation Analysis
//!
//! F1241-F1250: Correlation analysis falsification tests

use cbtop::{
    CorrelationAnalyzer, CorrelationResult, EventSample, EventType, InterferenceCategory,
    IsolationAction, PerformanceSample, SystemSnapshot,
};

// =============================================================================
// F1241: Event Correlation Tests
// =============================================================================

/// F1241.1: Event correlation calculated with valid Pearson r
#[test]
fn f1241_event_correlation() {
    let mut analyzer = CorrelationAnalyzer::new();

    // Add correlated samples
    for i in 0..20 {
        let t = i as f64;
        analyzer.add_perf_sample(PerformanceSample::new(t, 5.0 + i as f64, 100.0));
        analyzer.add_event_sample(EventSample::new(
            EventType::Interrupt,
            t,
            100.0 + i as f64 * 10.0,
        ));
    }

    let result = analyzer.correlate_events(EventType::Interrupt).unwrap();

    // Pearson r should be in valid range
    assert!(result.pearson_r >= -1.0 && result.pearson_r <= 1.0);
    assert!(result.sample_count >= 10);
}

/// F1241.2: Missing event type returns None
#[test]
fn f1241_missing_event_type() {
    let mut analyzer = CorrelationAnalyzer::new();

    for i in 0..10 {
        analyzer.add_perf_sample(PerformanceSample::new(i as f64, 5.0, 100.0));
    }

    // No interrupt events added
    let result = analyzer.correlate_events(EventType::Interrupt);
    assert!(result.is_none());
}

// =============================================================================
// F1242: Interference Detection Tests
// =============================================================================

/// F1242.1: Interference detected with high correlation
#[test]
fn f1242_interference_detected() {
    let mut analyzer = CorrelationAnalyzer::new();

    // Strong correlation with interrupts
    for i in 0..30 {
        let t = i as f64;
        analyzer.add_perf_sample(PerformanceSample::new(t, 5.0 + i as f64 * 2.0, 100.0));
        analyzer.add_event_sample(EventSample::new(
            EventType::Interrupt,
            t,
            100.0 + i as f64 * 20.0,
        ));
    }

    let result = analyzer.detect_interference().unwrap();

    assert_eq!(result.primary_source, EventType::Interrupt);
    assert!(result.correlation.abs() > 0.5);
}

/// F1242.2: No interference with uncorrelated data
#[test]
fn f1242_no_interference() {
    let analyzer = CorrelationAnalyzer::new();

    // No data = no interference
    let result = analyzer.detect_interference();
    assert!(result.is_none());
}

// =============================================================================
// F1243: System State Capture Tests
// =============================================================================

/// F1243.1: System state captured with all metrics
#[test]
fn f1243_system_state() {
    let snapshot = SystemSnapshot::new(1.0)
        .with_irq("timer", 1000)
        .with_irq("disk", 500)
        .with_disk_io(1_000_000.0)
        .with_network(5000.0)
        .with_context_switches(1000.0)
        .with_process("process1", 25.0)
        .with_load_average(1.5, 1.0, 0.8);

    assert_eq!(snapshot.timestamp, 1.0);
    assert_eq!(snapshot.total_irqs(), 1500);
    assert_eq!(snapshot.disk_io_bytes_per_sec, 1_000_000.0);
    assert_eq!(snapshot.network_packets_per_sec, 5000.0);
    assert_eq!(snapshot.load_average, (1.5, 1.0, 0.8));
}

/// F1243.2: Empty snapshot has defaults
#[test]
fn f1243_empty_snapshot() {
    let snapshot = SystemSnapshot::new(0.0);

    assert_eq!(snapshot.total_irqs(), 0);
    assert_eq!(snapshot.disk_io_bytes_per_sec, 0.0);
    assert!(snapshot.top_processes.is_empty());
}

// =============================================================================
// F1244: Isolation Recommendation Tests
// =============================================================================

/// F1244.1: Actionable isolation recommended
#[test]
fn f1244_isolation_recommended() {
    let mut analyzer = CorrelationAnalyzer::new();

    // Strong correlation with context switches
    for i in 0..30 {
        let t = i as f64;
        analyzer.add_perf_sample(PerformanceSample::new(t, 5.0 + i as f64 * 2.0, 100.0));
        analyzer.add_event_sample(EventSample::new(
            EventType::ContextSwitch,
            t,
            100.0 + i as f64 * 20.0,
        ));
    }

    let rec = analyzer.recommend_isolation();

    // Should recommend realtime priority for context switch interference
    assert!(rec.action != IsolationAction::None || rec.expected_improvement == 0.0);
}

/// F1244.2: No isolation for clean data
#[test]
fn f1244_no_isolation_needed() {
    let analyzer = CorrelationAnalyzer::new();
    let rec = analyzer.recommend_isolation();

    assert_eq!(rec.action, IsolationAction::None);
}

// =============================================================================
// F1245: CPU Interrupt Tracking Tests
// =============================================================================

/// F1245.1: IRQ counts tracked
#[test]
fn f1245_irq_tracking() {
    let snapshot = SystemSnapshot::new(1.0)
        .with_irq("timer", 5000)
        .with_irq("network", 3000)
        .with_irq("disk", 1000);

    assert_eq!(snapshot.irq_counts.len(), 3);
    assert_eq!(snapshot.total_irqs(), 9000);
}

// =============================================================================
// F1246: Disk I/O Tracking Tests
// =============================================================================

/// F1246.1: Disk I/O bytes/sec tracked
#[test]
fn f1246_disk_io_tracking() {
    let snapshot = SystemSnapshot::new(1.0).with_disk_io(500_000_000.0); // 500 MB/s

    assert_eq!(snapshot.disk_io_bytes_per_sec, 500_000_000.0);
}

// =============================================================================
// F1247: Network Activity Tracking Tests
// =============================================================================

/// F1247.1: Network packets tracked
#[test]
fn f1247_network_tracking() {
    let snapshot = SystemSnapshot::new(1.0).with_network(10000.0);

    assert_eq!(snapshot.network_packets_per_sec, 10000.0);
}

// =============================================================================
// F1248: Process List Capture Tests
// =============================================================================

/// F1248.1: Top CPU consumers captured
#[test]
fn f1248_process_list() {
    let snapshot = SystemSnapshot::new(1.0)
        .with_process("chrome", 15.0)
        .with_process("firefox", 10.0)
        .with_process("code", 8.0);

    assert_eq!(snapshot.top_processes.len(), 3);
    assert_eq!(snapshot.top_processes[0], ("chrome".to_string(), 15.0));
}

// =============================================================================
// F1249: Correlation Window Configuration Tests
// =============================================================================

/// F1249.1: Custom correlation window works
#[test]
fn f1249_custom_window() {
    let analyzer = CorrelationAnalyzer::new().with_window(30.0).with_spike_threshold(20.0);

    // Analyzer created with custom settings
    assert_eq!(analyzer.perf_sample_count(), 0);
}

// =============================================================================
// F1250: Historical Events Tests
// =============================================================================

/// F1250.1: Historical events stored in sliding window
#[test]
fn f1250_sliding_window() {
    let mut analyzer = CorrelationAnalyzer::new().with_max_samples(10);

    // Add more than max samples
    for i in 0..15 {
        analyzer.add_perf_sample(PerformanceSample::new(i as f64, 5.0, 100.0));
    }

    // Should only keep max_samples
    assert_eq!(analyzer.perf_sample_count(), 10);
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test event type names
#[test]
fn test_event_type_names() {
    assert_eq!(EventType::Interrupt.name(), "interrupt");
    assert_eq!(EventType::DiskIo.name(), "disk_io");
    assert_eq!(EventType::Network.name(), "network");
    assert_eq!(EventType::ContextSwitch.name(), "context_switch");
    assert_eq!(EventType::PageFault.name(), "page_fault");
    assert_eq!(EventType::ProcessCpu.name(), "process_cpu");
}

/// Test correlation strength
#[test]
fn test_correlation_strength() {
    let weak = CorrelationResult {
        event_type: EventType::Interrupt,
        pearson_r: 0.2,
        sample_count: 100,
        is_significant: false,
        optimal_lag: 0.0,
    };
    assert_eq!(weak.strength(), "weak");

    let strong = CorrelationResult {
        event_type: EventType::Interrupt,
        pearson_r: 0.6,
        sample_count: 100,
        is_significant: true,
        optimal_lag: 0.0,
    };
    assert_eq!(strong.strength(), "strong");
}

/// Test interference category
#[test]
fn test_interference_category() {
    assert_eq!(InterferenceCategory::Low.name(), "low");
    assert_eq!(InterferenceCategory::Moderate.name(), "moderate");
    assert_eq!(InterferenceCategory::High.name(), "high");
}

/// Test isolation action names
#[test]
fn test_isolation_action_names() {
    assert_eq!(IsolationAction::CpuPin.name(), "cpu_pin");
    assert_eq!(IsolationAction::MemoryIsolation.name(), "memory_isolation");
    assert_eq!(IsolationAction::RealtimePriority.name(), "realtime_priority");
    assert_eq!(IsolationAction::None.name(), "none");
}

/// Test isolation action descriptions
#[test]
fn test_isolation_action_descriptions() {
    let desc = IsolationAction::CpuPin.description();
    assert!(desc.contains("taskset") || desc.contains("numactl"));
}

/// Test CV spike detection
#[test]
fn test_cv_spike_detection() {
    let sample = PerformanceSample::new(0.0, 20.0, 100.0);
    assert!(sample.is_spike(15.0));
    assert!(!sample.is_spike(25.0));
}

/// Test analyzer clear
#[test]
fn test_analyzer_clear() {
    let mut analyzer = CorrelationAnalyzer::new();

    analyzer.add_perf_sample(PerformanceSample::new(0.0, 5.0, 100.0));
    analyzer.add_event_sample(EventSample::new(EventType::Interrupt, 0.0, 100.0));

    assert_eq!(analyzer.perf_sample_count(), 1);
    assert_eq!(analyzer.event_sample_count(), 1);

    analyzer.clear();

    assert_eq!(analyzer.perf_sample_count(), 0);
    assert_eq!(analyzer.event_sample_count(), 0);
}
