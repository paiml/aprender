//! Comprehensive tests for gpu/diagnostics.rs
//!
//! Coverage target: Fill gaps in the ~10% uncovered code.

use super::diagnostics::*;
use std::collections::HashMap;

// ============================================================================
// LogLevel Tests
// ============================================================================

#[test]
fn test_log_level_as_str_all_variants() {
    // Test internal as_str() through to_json() which uses it
    let entry_trace = LogEntry::new(LogLevel::Trace, "trace msg");
    assert!(entry_trace.to_json().contains("TRACE"));

    let entry_debug = LogEntry::new(LogLevel::Debug, "debug msg");
    assert!(entry_debug.to_json().contains("DEBUG"));

    let entry_info = LogEntry::new(LogLevel::Info, "info msg");
    assert!(entry_info.to_json().contains("INFO"));

    let entry_warn = LogEntry::new(LogLevel::Warn, "warn msg");
    assert!(entry_warn.to_json().contains("WARN"));

    let entry_error = LogEntry::new(LogLevel::Error, "error msg");
    assert!(entry_error.to_json().contains("ERROR"));
}

#[test]
fn test_log_level_ordering() {
    assert!(LogLevel::Trace < LogLevel::Debug);
    assert!(LogLevel::Debug < LogLevel::Info);
    assert!(LogLevel::Info < LogLevel::Warn);
    assert!(LogLevel::Warn < LogLevel::Error);

    // PartialOrd/Ord
    assert!(LogLevel::Trace <= LogLevel::Trace);
    assert!(LogLevel::Error >= LogLevel::Warn);
}

// ============================================================================
// LogEntry Tests
// ============================================================================

#[test]
fn test_log_entry_basic_construction() {
    let entry = LogEntry::new(LogLevel::Info, "test message");
    assert_eq!(entry.level(), LogLevel::Info);
    assert!(entry.timestamp() > 0);
    assert!(entry.correlation_id().is_none());
}

#[test]
fn test_log_entry_with_correlation_id() {
    let entry = LogEntry::new(LogLevel::Debug, "msg").with_correlation_id("req-123");
    assert_eq!(entry.correlation_id(), Some("req-123"));
}

#[test]
fn test_log_entry_with_field() {
    let entry = LogEntry::new(LogLevel::Warn, "warning")
        .with_field("key1", "value1")
        .with_field("key2", "value2");

    let json = entry.to_json();
    assert!(json.contains("\"key1\":\"value1\""));
    assert!(json.contains("\"key2\":\"value2\""));
}

#[test]
fn test_log_entry_to_json_full() {
    let entry = LogEntry::new(LogLevel::Error, "error occurred")
        .with_correlation_id("corr-456")
        .with_field("module", "gpu")
        .with_field("line", "42");

    let json = entry.to_json();
    assert!(json.contains("\"level\":\"ERROR\""));
    assert!(json.contains("\"message\":\"error occurred\""));
    assert!(json.contains("\"correlation_id\":\"corr-456\""));
    assert!(json.contains("\"module\":\"gpu\""));
    assert!(json.contains("\"line\":\"42\""));
    assert!(json.contains("\"timestamp\":"));
}

#[test]
fn test_log_entry_clone() {
    let entry = LogEntry::new(LogLevel::Info, "clone test")
        .with_correlation_id("c-1")
        .with_field("f", "v");
    let cloned = entry.clone();
    assert_eq!(cloned.level(), entry.level());
    assert_eq!(cloned.correlation_id(), entry.correlation_id());
}

// ============================================================================
// LogConfig Tests
// ============================================================================

#[test]
fn test_log_config_defaults() {
    let config = LogConfig::new();
    // Default level is Info
    let logger = Logger::new(config);
    assert!(logger.is_enabled(LogLevel::Info, "any"));
    assert!(logger.is_enabled(LogLevel::Warn, "any"));
    assert!(!logger.is_enabled(LogLevel::Debug, "any"));
}

#[test]
fn test_log_config_with_level() {
    let config = LogConfig::new().with_level(LogLevel::Debug);
    let logger = Logger::new(config);
    assert!(logger.is_enabled(LogLevel::Debug, "any"));
    assert!(logger.is_enabled(LogLevel::Trace, "any") == false);
}

#[test]
fn test_log_config_with_json_format() {
    let config = LogConfig::new().with_json_format(true);
    // JSON format is stored but doesn't affect is_enabled
    let logger = Logger::new(config);
    assert!(logger.is_enabled(LogLevel::Info, "test"));
}

#[test]
fn test_log_config_with_module_level() {
    let config = LogConfig::new()
        .with_level(LogLevel::Warn)
        .with_module_level("gpu", LogLevel::Debug)
        .with_module_level("api", LogLevel::Error);

    let logger = Logger::new(config);

    // Default level is Warn
    assert!(!logger.is_enabled(LogLevel::Info, "other_module"));
    assert!(logger.is_enabled(LogLevel::Warn, "other_module"));

    // gpu module has Debug level
    assert!(logger.is_enabled(LogLevel::Debug, "gpu"));
    assert!(logger.is_enabled(LogLevel::Info, "gpu"));

    // api module has Error level
    assert!(!logger.is_enabled(LogLevel::Warn, "api"));
    assert!(logger.is_enabled(LogLevel::Error, "api"));
}

#[test]
fn test_log_config_default_trait() {
    let config = LogConfig::default();
    let logger = Logger::new(config);
    assert!(logger.is_enabled(LogLevel::Info, "test"));
}

// ============================================================================
// Logger Tests
// ============================================================================

#[test]
fn test_logger_is_enabled_boundary() {
    let config = LogConfig::new().with_level(LogLevel::Warn);
    let logger = Logger::new(config);

    assert!(!logger.is_enabled(LogLevel::Trace, "mod"));
    assert!(!logger.is_enabled(LogLevel::Debug, "mod"));
    assert!(!logger.is_enabled(LogLevel::Info, "mod"));
    assert!(logger.is_enabled(LogLevel::Warn, "mod"));
    assert!(logger.is_enabled(LogLevel::Error, "mod"));
}

// ============================================================================
// PhaseTimer Tests
// ============================================================================

#[test]
fn test_phase_timer_basic() {
    let timer = PhaseTimer::new();
    timer.start_phase("phase1");
    std::thread::sleep(std::time::Duration::from_micros(100));
    timer.end_phase("phase1");

    let breakdown = timer.breakdown();
    assert!(breakdown.contains_key("phase1"));
    assert!(breakdown["phase1"] > 0);
}

#[test]
fn test_phase_timer_multiple_phases() {
    let timer = PhaseTimer::new();

    timer.start_phase("embed");
    timer.end_phase("embed");

    timer.start_phase("attention");
    timer.end_phase("attention");

    timer.start_phase("ffn");
    timer.end_phase("ffn");

    let breakdown = timer.breakdown();
    assert_eq!(breakdown.len(), 3);
    assert!(breakdown.contains_key("embed"));
    assert!(breakdown.contains_key("attention"));
    assert!(breakdown.contains_key("ffn"));
}

#[test]
fn test_phase_timer_end_without_start() {
    let timer = PhaseTimer::new();
    // End a phase that was never started - should be a no-op
    timer.end_phase("nonexistent");
    let breakdown = timer.breakdown();
    assert!(!breakdown.contains_key("nonexistent"));
}

#[test]
fn test_phase_timer_overwrite_phase() {
    let timer = PhaseTimer::new();
    timer.start_phase("p");
    timer.end_phase("p");
    let b1 = timer.breakdown()["p"];

    // Restart same phase
    timer.start_phase("p");
    std::thread::sleep(std::time::Duration::from_micros(50));
    timer.end_phase("p");
    let b2 = timer.breakdown()["p"];

    // Second timing should overwrite first
    assert!(b2 > 0);
}

#[test]
fn test_phase_timer_default_trait() {
    let timer = PhaseTimer::default();
    timer.start_phase("test");
    timer.end_phase("test");
    assert!(!timer.breakdown().is_empty());
}

// ============================================================================
// MemoryTracker Tests
// ============================================================================

#[test]
fn test_memory_tracker_allocation_deallocation() {
    let tracker = MemoryTracker::new();

    tracker.record_allocation("buf1", 1000);
    let report = tracker.report();
    assert_eq!(report.current_bytes, 1000);
    assert_eq!(report.peak_bytes, 1000);
    assert_eq!(report.allocation_count, 1);

    tracker.record_allocation("buf2", 500);
    let report = tracker.report();
    assert_eq!(report.current_bytes, 1500);
    assert_eq!(report.peak_bytes, 1500);
    assert_eq!(report.allocation_count, 2);

    tracker.record_deallocation("buf1", 1000);
    let report = tracker.report();
    assert_eq!(report.current_bytes, 500);
    assert_eq!(report.peak_bytes, 1500); // Peak unchanged
}

#[test]
fn test_memory_tracker_peak_tracking() {
    let tracker = MemoryTracker::new();

    tracker.record_allocation("a", 1000);
    tracker.record_allocation("b", 2000);
    // Peak = 3000

    tracker.record_deallocation("a", 1000);
    tracker.record_deallocation("b", 2000);
    // Current = 0, but peak should still be 3000

    let report = tracker.report();
    assert_eq!(report.current_bytes, 0);
    assert_eq!(report.peak_bytes, 3000);
}

#[test]
fn test_memory_tracker_default_trait() {
    let tracker = MemoryTracker::default();
    let report = tracker.report();
    assert_eq!(report.current_bytes, 0);
    assert_eq!(report.peak_bytes, 0);
    assert_eq!(report.allocation_count, 0);
}

// ============================================================================
// DiagnosticsCollector Tests
// ============================================================================

#[test]
fn test_diagnostics_collector_record_timing() {
    let collector = DiagnosticsCollector::new();

    let mut timing = HashMap::new();
    timing.insert("phase1".to_string(), 100u64);
    timing.insert("phase2".to_string(), 200u64);

    collector.record_request_timing("req-1", timing);

    let summary = collector.summary();
    assert_eq!(summary.request_count, 1);
}

#[test]
fn test_diagnostics_collector_multiple_requests() {
    let collector = DiagnosticsCollector::new();

    for i in 0..5 {
        let mut timing = HashMap::new();
        timing.insert("total".to_string(), (i * 10) as u64);
        collector.record_request_timing(&format!("req-{}", i), timing);
    }

    let summary = collector.summary();
    assert_eq!(summary.request_count, 5);
}

#[test]
fn test_diagnostics_collector_memory_snapshot() {
    let collector = DiagnosticsCollector::new();

    let report = MemoryReport {
        peak_bytes: 5000,
        current_bytes: 3000,
        allocation_count: 10,
    };

    collector.record_memory_snapshot(report);
    // Verify it doesn't panic and state is maintained
    let summary = collector.summary();
    assert_eq!(summary.request_count, 0); // Snapshots don't increment request count
}

#[test]
fn test_diagnostics_collector_default_trait() {
    let collector = DiagnosticsCollector::default();
    let summary = collector.summary();
    assert_eq!(summary.request_count, 0);
}

// ============================================================================
// DebugMode Tests
// ============================================================================

#[test]
fn test_debug_mode_enable_disable() {
    let mode = DebugMode::new();
    assert!(!mode.is_enabled());

    mode.enable();
    assert!(mode.is_enabled());

    mode.disable();
    assert!(!mode.is_enabled());
}

#[test]
fn test_debug_mode_default_trait() {
    let mode = DebugMode::default();
    assert!(!mode.is_enabled());
}

// ============================================================================
// RequestCapture Tests
// ============================================================================

#[test]
fn test_request_capture_basic() {
    let capture = RequestCapture::new();
    assert_eq!(capture.input(), "");
    assert!(capture.params().is_empty());
}

#[test]
fn test_request_capture_with_input() {
    let capture = RequestCapture::new().with_input("Hello, world!");
    assert_eq!(capture.input(), "Hello, world!");
}

#[test]
fn test_request_capture_with_params() {
    let capture = RequestCapture::new()
        .with_params("temperature", "0.7")
        .with_params("max_tokens", "100");

    let params = capture.params();
    assert_eq!(params.get("temperature"), Some(&"0.7".to_string()));
    assert_eq!(params.get("max_tokens"), Some(&"100".to_string()));
}

#[test]
fn test_request_capture_to_json() {
    let capture = RequestCapture::new()
        .with_input("test input")
        .with_params("key", "value");

    let json = capture.to_json();
    assert!(json.contains("\"input\":\"test input\""));
    assert!(json.contains("\"params\":{"));
    assert!(json.contains("\"key\":\"value\""));
}

#[test]
fn test_request_capture_from_json_success() {
    let json = r#"{"input":"parsed input","params":{}}"#;
    let capture = RequestCapture::from_json(json).expect("should parse");
    assert_eq!(capture.input(), "parsed input");
}

#[test]
fn test_request_capture_from_json_missing_input() {
    let json = r#"{"params":{}}"#;
    let result = RequestCapture::from_json(json);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Missing input");
}

#[test]
fn test_request_capture_from_json_invalid_input() {
    let json = r#"{"input":"#; // Malformed - no closing quote
    let result = RequestCapture::from_json(json);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Invalid input");
}

#[test]
fn test_request_capture_default_trait() {
    let capture = RequestCapture::default();
    assert_eq!(capture.input(), "");
}

#[test]
fn test_request_capture_clone() {
    let capture = RequestCapture::new()
        .with_input("clone test")
        .with_params("p", "v");
    let cloned = capture.clone();
    assert_eq!(cloned.input(), capture.input());
    assert_eq!(cloned.params(), capture.params());
}

include!("diagnostics_tests_state_dump.rs");
