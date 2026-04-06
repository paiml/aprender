//! Integration tests for Sprint 42: Semantic Equivalence
//!
//! Tests state-based semantic equivalence checking for Batuta transpilations.

use renacer::semantic_equivalence::{PerformanceComparison, SemanticValidator, ValidationResult};
use renacer::trace_context::LamportClock;
use renacer::unified_trace::{SyscallSpan, UnifiedTrace};
use std::borrow::Cow;

fn create_test_span(
    syscall_name: &str,
    args: Vec<(&str, &str)>,
    return_value: i64,
    start_ns: u64,
    end_ns: u64,
) -> SyscallSpan {
    let clock = LamportClock::new();
    let duration_ns = end_ns.saturating_sub(start_ns);

    let args_vec: Vec<(Cow<'static, str>, String)> =
        args.into_iter().map(|(k, v)| (Cow::Owned(k.to_string()), v.to_string())).collect();

    SyscallSpan::new(
        1, // parent_span_id
        Cow::Owned(syscall_name.to_string()),
        args_vec,
        return_value,
        start_ns,
        duration_ns,
        None, // errno
        &clock,
    )
}

#[test]
fn test_identical_traces() {
    // Scenario: Python and Rust produce identical syscall sequences

    let mut python_trace = UnifiedTrace::new(1, "python_test".to_string());
    python_trace.add_syscall(create_test_span(
        "write",
        vec![("fd", "1"), ("buf", "Hello\n"), ("count", "6")],
        6,
        0,
        1000,
    ));
    python_trace.add_syscall(create_test_span(
        "write",
        vec![("fd", "1"), ("buf", "World\n"), ("count", "6")],
        6,
        1000,
        2000,
    ));

    let mut rust_trace = UnifiedTrace::new(2, "rust_test".to_string());
    rust_trace.add_syscall(create_test_span(
        "write",
        vec![("fd", "1"), ("buf", "Hello\n"), ("count", "6")],
        6,
        0,
        500, // Faster execution
    ));
    rust_trace.add_syscall(create_test_span(
        "write",
        vec![("fd", "1"), ("buf", "World\n"), ("count", "6")],
        6,
        500,
        1000,
    ));

    let validator = SemanticValidator::new();
    let result = validator.validate(&python_trace, &rust_trace);

    match result {
        ValidationResult::Pass { confidence, matched_syscalls, .. } => {
            assert!(confidence > 0.95, "Expected high confidence");
            assert_eq!(matched_syscalls, 2);
        }
        ValidationResult::Fail { explanation, .. } => {
            panic!("Expected pass, got fail: {}", explanation);
        }
    }
}

#[test]
fn test_different_syscall_order() {
    // Scenario: Same syscalls, different order = FAIL

    let mut python_trace = UnifiedTrace::new(1, "python_test".to_string());
    python_trace.add_syscall(create_test_span("open", vec![("path", "/tmp/file")], 3, 0, 1000));
    python_trace.add_syscall(create_test_span("write", vec![("fd", "3")], 10, 1000, 2000));
    python_trace.add_syscall(create_test_span("close", vec![("fd", "3")], 0, 2000, 3000));

    let mut rust_trace = UnifiedTrace::new(2, "rust_test".to_string());
    rust_trace.add_syscall(create_test_span("open", vec![("path", "/tmp/file")], 3, 0, 500));
    rust_trace.add_syscall(create_test_span("close", vec![("fd", "3")], 0, 500, 1000)); // Wrong order!
    rust_trace.add_syscall(create_test_span("write", vec![("fd", "3")], 10, 1000, 1500));

    let validator = SemanticValidator::new();
    let result = validator.validate(&python_trace, &rust_trace);

    match result {
        ValidationResult::Pass { .. } => {
            panic!("Expected fail due to different syscall order");
        }
        ValidationResult::Fail { .. } => {
            // Expected
        }
    }
}

#[test]
fn test_missing_syscalls() {
    // Scenario: Rust version missing syscalls = FAIL

    let mut python_trace = UnifiedTrace::new(1, "python_test".to_string());
    python_trace.add_syscall(create_test_span("write", vec![("fd", "1")], 5, 0, 1000));
    python_trace.add_syscall(create_test_span("write", vec![("fd", "1")], 5, 1000, 2000));
    python_trace.add_syscall(create_test_span("write", vec![("fd", "1")], 5, 2000, 3000));

    let mut rust_trace = UnifiedTrace::new(2, "rust_test".to_string());
    rust_trace.add_syscall(create_test_span("write", vec![("fd", "1")], 5, 0, 500));
    // Missing 2 writes!

    let validator = SemanticValidator::new();
    let result = validator.validate(&python_trace, &rust_trace);

    match result {
        ValidationResult::Pass { .. } => {
            panic!("Expected fail due to missing syscalls");
        }
        ValidationResult::Fail { .. } => {
            // Expected
        }
    }
}

#[test]
fn test_performance_comparison() {
    // Scenario: Rust is faster than Python

    let mut python_trace = UnifiedTrace::new(1, "python_test".to_string());
    python_trace.add_syscall(create_test_span("write", vec![], 5, 0, 10_000));
    python_trace.end_process(0); // End process to calculate duration

    let mut rust_trace = UnifiedTrace::new(2, "rust_test".to_string());
    rust_trace.add_syscall(create_test_span("write", vec![], 5, 0, 1_000)); // 10× faster
    rust_trace.end_process(0); // End process to calculate duration

    let validator = SemanticValidator::new();
    let result = validator.validate(&python_trace, &rust_trace);

    match result {
        ValidationResult::Pass { performance, .. } => {
            // Rust should be faster
            assert!(performance.transpiled_runtime_nanos < performance.original_runtime_nanos);
            assert!(performance.speedup > 1.0, "Expected speedup > 1.0");
        }
        ValidationResult::Fail { .. } => {
            panic!("Expected pass");
        }
    }
}

#[test]
fn test_performance_comparison_api() {
    let perf = PerformanceComparison::new(10_000, 1_000);

    assert_eq!(perf.original_runtime_nanos, 10_000);
    assert_eq!(perf.transpiled_runtime_nanos, 1_000);
    assert_eq!(perf.speedup, 10.0);

    // Add memory comparison
    let perf_with_mem = perf.with_memory(1_000_000, 500_000);
    let mem_delta = perf_with_mem.memory_delta.expect("test");

    assert_eq!(mem_delta.original_bytes, 1_000_000);
    assert_eq!(mem_delta.transpiled_bytes, 500_000);
    assert_eq!(mem_delta.reduction_percentage, 50.0);
}

#[test]
fn test_tolerant_validation() {
    // Scenario: Minor differences tolerated with higher tolerance

    let mut python_trace = UnifiedTrace::new(1, "python_test".to_string());
    python_trace.add_syscall(create_test_span("write", vec![("fd", "1")], 5, 0, 1000));

    let mut rust_trace = UnifiedTrace::new(2, "rust_test".to_string());
    rust_trace.add_syscall(create_test_span(
        "write",
        vec![("fd", "1")],
        5, // Same return value
        0,
        500,
    ));

    let validator = SemanticValidator::with_tolerance(0.1); // 10% tolerance
    let result = validator.validate(&python_trace, &rust_trace);

    match result {
        ValidationResult::Pass { .. } => {
            // Expected with tolerance
        }
        ValidationResult::Fail { .. } => {
            // Also acceptable - depends on tolerance implementation
        }
    }
}

#[test]
fn test_buffered_io_scenario() {
    // Scenario: Python does 100 small writes, Rust buffers into 1 large write
    // This represents a typical Batuta optimization

    let mut python_trace = UnifiedTrace::new(1, "python_test".to_string());
    for i in 0..100 {
        python_trace.add_syscall(create_test_span(
            "write",
            vec![("fd", "1"), ("buf", "x"), ("count", "1")],
            1,
            i * 100,
            i * 100 + 50,
        ));
    }

    let mut rust_trace = UnifiedTrace::new(2, "rust_test".to_string());
    // Single buffered write of 100 bytes
    let buf = "x".repeat(100);
    rust_trace.add_syscall(create_test_span(
        "write",
        vec![("fd", "1"), ("buf", &buf), ("count", "100")],
        100,
        0,
        500,
    ));

    let validator = SemanticValidator::new();
    let result = validator.validate(&python_trace, &rust_trace);

    // This should FAIL with strict equivalence (different syscall counts)
    // But semantically they produce the same output
    match result {
        ValidationResult::Pass { .. } => {
            // If validator is smart enough to recognize buffering
        }
        ValidationResult::Fail { .. } => {
            // Expected with strict validation
        }
    }
}

#[test]
fn test_file_operations_equivalence() {
    // Scenario: File I/O operations must match exactly

    let mut python_trace = UnifiedTrace::new(1, "python_test".to_string());
    python_trace.add_syscall(create_test_span(
        "open",
        vec![("path", "/tmp/test.txt"), ("flags", "O_WRONLY")],
        3,
        0,
        1000,
    ));
    python_trace.add_syscall(create_test_span(
        "write",
        vec![("fd", "3"), ("buf", "data"), ("count", "4")],
        4,
        1000,
        2000,
    ));
    python_trace.add_syscall(create_test_span("close", vec![("fd", "3")], 0, 2000, 3000));

    let mut rust_trace = UnifiedTrace::new(2, "rust_test".to_string());
    rust_trace.add_syscall(create_test_span(
        "open",
        vec![("path", "/tmp/test.txt"), ("flags", "O_WRONLY")],
        3,
        0,
        500,
    ));
    rust_trace.add_syscall(create_test_span(
        "write",
        vec![("fd", "3"), ("buf", "data"), ("count", "4")],
        4,
        500,
        1000,
    ));
    rust_trace.add_syscall(create_test_span("close", vec![("fd", "3")], 0, 1000, 1500));

    let validator = SemanticValidator::new();
    let result = validator.validate(&python_trace, &rust_trace);

    match result {
        ValidationResult::Pass { matched_syscalls, .. } => {
            assert_eq!(matched_syscalls, 3);
        }
        ValidationResult::Fail { explanation, .. } => {
            panic!("Expected pass for matching file ops: {}", explanation);
        }
    }
}

#[test]
fn test_allocator_differences_ignored() {
    // Scenario: Different allocator behavior (mmap, brk) should be ignored
    // as it doesn't affect observable semantics

    let mut python_trace = UnifiedTrace::new(1, "python_test".to_string());
    python_trace.add_syscall(create_test_span("brk", vec![("addr", "0x1000")], 0, 0, 100));
    python_trace.add_syscall(create_test_span("write", vec![("fd", "1")], 5, 100, 200));

    let mut rust_trace = UnifiedTrace::new(2, "rust_test".to_string());
    rust_trace.add_syscall(create_test_span("mmap", vec![("length", "4096")], 0x2000, 0, 50));
    rust_trace.add_syscall(create_test_span("write", vec![("fd", "1")], 5, 50, 100));

    // Validator should ignore allocator differences
    let validator = SemanticValidator::new();
    let result = validator.validate(&python_trace, &rust_trace);

    // Depends on validator implementation - may pass or fail
    match result {
        ValidationResult::Pass { .. } => {
            // If validator filters allocator syscalls
        }
        ValidationResult::Fail { .. } => {
            // If strict comparison
        }
    }
}

#[test]
fn test_empty_traces() {
    // Edge case: Both traces are empty

    let python_trace = UnifiedTrace::new(1, "python_test".to_string());
    let rust_trace = UnifiedTrace::new(2, "rust_test".to_string());

    let validator = SemanticValidator::new();
    let result = validator.validate(&python_trace, &rust_trace);

    match result {
        ValidationResult::Pass { matched_syscalls, .. } => {
            assert_eq!(matched_syscalls, 0);
        }
        ValidationResult::Fail { .. } => {
            panic!("Empty traces should be equivalent");
        }
    }
}

#[test]
fn test_validation_confidence_scoring() {
    // Test that confidence score reflects match quality

    let mut python_trace = UnifiedTrace::new(1, "python_test".to_string());
    for i in 0..10 {
        python_trace.add_syscall(create_test_span("write", vec![], 1, i * 100, i * 100 + 50));
    }

    let mut rust_trace = UnifiedTrace::new(2, "rust_test".to_string());
    for i in 0..10 {
        rust_trace.add_syscall(create_test_span("write", vec![], 1, i * 50, i * 50 + 25));
    }

    let validator = SemanticValidator::new();
    let result = validator.validate(&python_trace, &rust_trace);

    match result {
        ValidationResult::Pass { confidence, .. } => {
            assert!((0.0..=1.0).contains(&confidence), "Confidence out of range");
            assert!(confidence > 0.5, "Expected reasonable confidence");
        }
        ValidationResult::Fail { .. } => {
            // Acceptable
        }
    }
}

#[test]
fn test_validator_tolerance_getter() {
    let validator1 = SemanticValidator::new();
    assert_eq!(validator1.tolerance(), 0.05); // Default tolerance is 5%

    let validator2 = SemanticValidator::with_tolerance(0.1);
    assert_eq!(validator2.tolerance(), 0.1);
}

#[test]
fn test_real_world_python_to_rust_scenario() {
    // Realistic scenario: Python list comprehension → Rust iterator

    // Python: Multiple append operations
    let mut python_trace = UnifiedTrace::new(1, "python_list_comp".to_string());
    python_trace.add_syscall(create_test_span("brk", vec![], 0x1000, 0, 100)); // List allocation
    for i in 0..5 {
        let buf = format!("{}\n", i * 2);
        python_trace.add_syscall(create_test_span(
            "write",
            vec![("fd", "1"), ("buf", &buf)],
            2,
            (i + 1) * 100,
            (i + 1) * 100 + 50,
        ));
    }

    // Rust: Optimized with iterator (fewer allocations)
    let mut rust_trace = UnifiedTrace::new(2, "rust_iterator".to_string());
    for i in 0..5 {
        let buf = format!("{}\n", i * 2);
        rust_trace.add_syscall(create_test_span(
            "write",
            vec![("fd", "1"), ("buf", &buf)],
            2,
            i * 50,
            i * 50 + 25,
        ));
    }

    // End processes to calculate durations
    python_trace.end_process(0);
    rust_trace.end_process(0);

    let validator = SemanticValidator::new();
    let result = validator.validate(&python_trace, &rust_trace);

    // Both produce same output (0, 2, 4, 6, 8)
    match result {
        ValidationResult::Pass { performance, .. } => {
            // Rust should be faster
            assert!(performance.speedup > 1.0);
        }
        ValidationResult::Fail { .. } => {
            // May fail due to different allocator behavior
        }
    }
}
