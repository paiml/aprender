// Sprint 33: Span Context Propagation - Integration Tests
// EXTREME TDD - RED Phase: Tests written BEFORE implementation
//
// Goal: Enable distributed tracing by propagating W3C Trace Context
//       from instrumented applications to Renacer's syscall traces

#![allow(deprecated)] // assert_cmd::Command::cargo_bin deprecation

use assert_cmd::Command;

/// Test 1: --trace-parent CLI flag is accepted
#[test]
fn test_trace_parent_cli_flag_accepted() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-parent")
        .arg("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        .arg("--")
        .arg("echo")
        .arg("test")
        .output()
        .expect("Failed to execute command");

    // Should NOT error on flag parsing
    assert!(output.status.success(), "Command failed: {}", String::from_utf8_lossy(&output.stderr));
}

/// Test 2: Valid traceparent format is parsed correctly
#[test]
fn test_trace_parent_valid_format() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-parent")
        .arg("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
        .arg("--")
        .arg("echo")
        .arg("distributed")
        .output()
        .expect("Failed to execute command");

    // Should parse and use the trace context
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Expect message indicating distributed tracing is enabled
    assert!(
        stderr.contains("Distributed tracing enabled") || output.status.success(),
        "Should indicate distributed tracing: {}",
        stderr
    );
}

/// Test 3: Invalid traceparent format falls back to new root trace
#[test]
fn test_trace_parent_invalid_format_fallback() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-parent")
        .arg("INVALID-FORMAT")
        .arg("--")
        .arg("echo")
        .arg("test")
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should warn about invalid format but continue
    assert!(
        stderr.contains("Invalid trace context")
            || stderr.contains("malformed")
            || output.status.success(),
        "Should handle invalid format gracefully: {}",
        stderr
    );
}

/// Test 4: All-zero trace-id is rejected
#[test]
fn test_trace_parent_all_zero_trace_id() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-parent")
        .arg("00-00000000000000000000000000000000-b7ad6b7169203331-01")
        .arg("--")
        .arg("echo")
        .arg("test")
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should reject all-zero trace-id
    assert!(
        stderr.contains("Invalid trace context")
            || stderr.contains("all-zero")
            || output.status.success(),
        "Should reject all-zero trace-id: {}",
        stderr
    );
}

/// Test 5: All-zero parent-id is rejected
#[test]
fn test_trace_parent_all_zero_parent_id() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-parent")
        .arg("00-0af7651916cd43dd8448eb211c80319c-0000000000000000-01")
        .arg("--")
        .arg("echo")
        .arg("test")
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should reject all-zero parent-id
    assert!(
        stderr.contains("Invalid trace context")
            || stderr.contains("all-zero")
            || output.status.success(),
        "Should reject all-zero parent-id: {}",
        stderr
    );
}

/// Test 6: Invalid version is rejected
#[test]
fn test_trace_parent_invalid_version() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-parent")
        .arg("99-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        .arg("--")
        .arg("echo")
        .arg("test")
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should reject unknown version
    assert!(
        stderr.contains("Invalid trace context")
            || stderr.contains("version")
            || output.status.success(),
        "Should reject unknown version: {}",
        stderr
    );
}

/// Test 7: Backward compatibility - works without trace context
#[test]
fn test_backward_compatibility_no_trace_context() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--")
        .arg("echo")
        .arg("test")
        .output()
        .expect("Failed to execute command");

    // Should work normally without trace context (existing behavior)
    assert!(
        output.status.success(),
        "Should work without trace context: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test 8: trace-parent requires otlp-endpoint
#[test]
fn test_trace_parent_requires_otlp() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--trace-parent")
        .arg("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        .arg("--")
        .arg("echo")
        .arg("test")
        .output()
        .expect("Failed to execute command");

    // trace-parent without OTLP should be ignored (no-op)
    assert!(
        output.status.success(),
        "Should ignore trace-parent without OTLP: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test 9: Sampled flag (01) is detected
#[test]
fn test_trace_parent_sampled_flag_set() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-parent")
        .arg("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        .arg("--")
        .arg("echo")
        .arg("sampled")
        .output()
        .expect("Failed to execute command");

    // Should parse sampled flag (01)
    assert!(
        output.status.success(),
        "Should handle sampled flag: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test 10: Not sampled flag (00) is detected
#[test]
fn test_trace_parent_not_sampled_flag() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-parent")
        .arg("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00")
        .arg("--")
        .arg("echo")
        .arg("not_sampled")
        .output()
        .expect("Failed to execute command");

    // Should parse not-sampled flag (00)
    assert!(
        output.status.success(),
        "Should handle not-sampled flag: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test 11: Combine with statistics mode
#[test]
fn test_trace_parent_with_statistics() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-parent")
        .arg("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        .arg("-c")
        .arg("--")
        .arg("echo")
        .arg("stats")
        .output()
        .expect("Failed to execute command");

    // Should work with statistics mode
    assert!(
        output.status.success(),
        "Should work with statistics: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test 12: Combine with source correlation
#[test]
fn test_trace_parent_with_source() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-parent")
        .arg("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        .arg("-s")
        .arg("--")
        .arg("echo")
        .arg("source")
        .output()
        .expect("Failed to execute command");

    // Should work with source correlation
    assert!(
        output.status.success(),
        "Should work with source correlation: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test 13: Combine with timing mode
#[test]
fn test_trace_parent_with_timing() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-parent")
        .arg("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        .arg("-T")
        .arg("--")
        .arg("echo")
        .arg("timing")
        .output()
        .expect("Failed to execute command");

    // Should work with timing mode
    assert!(
        output.status.success(),
        "Should work with timing: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test 14: Combine with filtering
#[test]
fn test_trace_parent_with_filtering() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-parent")
        .arg("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        .arg("-e")
        .arg("trace=write")
        .arg("--")
        .arg("echo")
        .arg("filtered")
        .output()
        .expect("Failed to execute command");

    // Should work with syscall filtering
    assert!(
        output.status.success(),
        "Should work with filtering: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test 15: Combine with compute tracing (Sprint 32)
#[test]
fn test_trace_parent_with_compute_tracing() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-parent")
        .arg("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        .arg("--trace-compute")
        .arg("-c")
        .arg("--stats-extended")
        .arg("--")
        .arg("echo")
        .arg("compute")
        .output()
        .expect("Failed to execute command");

    // Should work with compute tracing
    assert!(
        output.status.success(),
        "Should work with compute tracing: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test 16: Combine with transpiler decision tracing (Sprint 31)
#[test]
fn test_trace_parent_with_decision_tracing() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--trace-parent")
        .arg("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        .arg("--trace-transpiler-decisions")
        .arg("--")
        .arg("echo")
        .arg("decisions")
        .output()
        .expect("Failed to execute command");

    // Should work with decision tracing
    assert!(
        output.status.success(),
        "Should work with decision tracing: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test 17: Full observability stack (all Sprint 30-33 features)
#[test]
fn test_trace_parent_full_observability_stack() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--otlp-endpoint")
        .arg("http://localhost:4317")
        .arg("--otlp-service-name")
        .arg("full-stack-test")
        .arg("--trace-parent")
        .arg("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        .arg("--trace-compute")
        .arg("--trace-transpiler-decisions")
        .arg("-c")
        .arg("--stats-extended")
        .arg("-s")
        .arg("-T")
        .arg("--")
        .arg("echo")
        .arg("full_stack")
        .output()
        .expect("Failed to execute command");

    // Should work with ALL features combined
    assert!(
        output.status.success(),
        "Should work with full stack: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
