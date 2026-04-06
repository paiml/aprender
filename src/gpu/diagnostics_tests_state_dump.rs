
// ============================================================================
// StateDump Tests
// ============================================================================

#[test]
fn test_state_dump_basic() {
    let dump = StateDump::new();
    assert_eq!(dump.error(), "");
    assert_eq!(dump.stack_trace(), "");
    assert!(dump.state().is_empty());
}

#[test]
fn test_state_dump_with_error() {
    let dump = StateDump::new().with_error("OutOfMemory");
    assert_eq!(dump.error(), "OutOfMemory");
}

#[test]
fn test_state_dump_with_stack_trace() {
    let dump = StateDump::new().with_stack_trace("at foo.rs:42\nat bar.rs:10");
    assert_eq!(dump.stack_trace(), "at foo.rs:42\nat bar.rs:10");
}

#[test]
fn test_state_dump_with_state() {
    let dump = StateDump::new()
        .with_state("batch_size", "32")
        .with_state("seq_len", "128");

    let state = dump.state();
    assert_eq!(state.get("batch_size"), Some(&"32".to_string()));
    assert_eq!(state.get("seq_len"), Some(&"128".to_string()));
}

#[test]
fn test_state_dump_to_json() {
    let dump = StateDump::new()
        .with_error("TestError")
        .with_stack_trace("line1\nline2")
        .with_state("k", "v");

    let json = dump.to_json();
    assert!(json.contains("\"error\":\"TestError\""));
    assert!(json.contains("\"stack_trace\":\"line1\\nline2\"")); // Newlines escaped
    assert!(json.contains("\"state\":{"));
    assert!(json.contains("\"k\":\"v\""));
}

#[test]
fn test_state_dump_to_json_empty() {
    let dump = StateDump::new();
    let json = dump.to_json();
    assert!(json.contains("\"error\":\"\""));
    assert!(json.contains("\"stack_trace\":\"\""));
    assert!(json.contains("\"state\":{}"));
}

#[test]
fn test_state_dump_default_trait() {
    let dump = StateDump::default();
    assert_eq!(dump.error(), "");
}

#[test]
fn test_state_dump_clone() {
    let dump = StateDump::new()
        .with_error("err")
        .with_stack_trace("trace")
        .with_state("s", "v");
    let cloned = dump.clone();
    assert_eq!(cloned.error(), dump.error());
    assert_eq!(cloned.stack_trace(), dump.stack_trace());
    assert_eq!(cloned.state(), dump.state());
}

// ============================================================================
// MemoryReport Tests
// ============================================================================

#[test]
fn test_memory_report_clone() {
    let report = MemoryReport {
        peak_bytes: 1000,
        current_bytes: 500,
        allocation_count: 5,
    };
    let cloned = report.clone();
    assert_eq!(cloned.peak_bytes, report.peak_bytes);
    assert_eq!(cloned.current_bytes, report.current_bytes);
    assert_eq!(cloned.allocation_count, report.allocation_count);
}

#[test]
fn test_memory_report_debug() {
    let report = MemoryReport {
        peak_bytes: 2048,
        current_bytes: 1024,
        allocation_count: 3,
    };
    let debug = format!("{:?}", report);
    assert!(debug.contains("MemoryReport"));
    assert!(debug.contains("2048"));
    assert!(debug.contains("1024"));
    assert!(debug.contains("3"));
}

// ============================================================================
// DiagnosticsSummary Tests
// ============================================================================

#[test]
fn test_diagnostics_summary_clone() {
    let summary = DiagnosticsSummary { request_count: 42 };
    let cloned = summary.clone();
    assert_eq!(cloned.request_count, 42);
}

#[test]
fn test_diagnostics_summary_debug() {
    let summary = DiagnosticsSummary { request_count: 100 };
    let debug = format!("{:?}", summary);
    assert!(debug.contains("DiagnosticsSummary"));
    assert!(debug.contains("100"));
}
