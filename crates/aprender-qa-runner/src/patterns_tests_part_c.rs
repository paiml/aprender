// ── Performance/Parity/Integrity edge case tests ───────────────────────────

/// Verify check_memory_leak returns zero growth when initial RSS is zero
#[test]
fn test_perf_memory_leak_zero_initial_rss() {
    let result = PerformanceValidator::check_memory_leak(0.0, 50.0, 10.0);
    // Growth = 0.0 when initial_rss_mb == 0 (avoid div by zero)
    assert!(result.passed);
    assert!((result.measured - 0.0).abs() < f64::EPSILON);
}

/// Verify check_quantization_impact returns zero degradation when f16 perplexity is zero
#[test]
fn test_parity_quantization_zero_f16_perplexity() {
    let result = ParityChecker::check_quantization_impact(0.0, 5.0, 10.0);
    // Degradation = 0.0 when f16_perplexity == 0.0 (avoid div by zero)
    assert!(result.passed);
    assert!((result.max_diff - 0.0).abs() < f64::EPSILON);
}

/// Verify check_process_termination detects zombie process (no exit code)
#[test]
fn test_int_process_termination_zombie() {
    let result = IntegrityChecker::check_process_termination(None, false, true);
    assert!(!result.passed);
    assert!(result.description.contains("Zombie"));
}

/// Verify check_process_termination detects unclean exit without error output
#[test]
fn test_int_process_termination_unclean_exit_no_output() {
    let result = IntegrityChecker::check_process_termination(Some(1), false, false);
    assert!(!result.passed);
    assert!(result.description.contains("Unclean exit"));
}

/// Verify check_process_termination passes on error exit with output
#[test]
fn test_int_process_termination_error_exit_with_output() {
    let result = IntegrityChecker::check_process_termination(Some(1), false, true);
    assert!(result.passed);
}

/// Verify check_memory_safety detects SIGBUS signal (exit code 135)
#[test]
fn test_int_memory_safety_sigbus() {
    let result = IntegrityChecker::check_memory_safety(Some(135), "");
    assert!(!result.passed);
    assert!(result.description.contains("Bus error"));
}

/// Verify check_memory_safety detects SIGABRT signal (exit code 134)
#[test]
fn test_int_memory_safety_sigabrt() {
    let result = IntegrityChecker::check_memory_safety(Some(134), "");
    assert!(!result.passed);
    assert!(result.description.contains("Abort"));
}

/// Verify check_memory_safety detects buffer overflow in stderr
#[test]
fn test_int_memory_safety_stderr_buffer_overflow() {
    let result = IntegrityChecker::check_memory_safety(Some(0), "buffer overflow detected");
    assert!(!result.passed);
    assert!(result.description.contains("Memory safety"));
}

/// Verify check_memory_safety detects stack smashing in stderr
#[test]
fn test_int_memory_safety_stderr_stack_smashing() {
    let result = IntegrityChecker::check_memory_safety(Some(0), "stack smashing detected");
    assert!(!result.passed);
}

/// Verify check_tensor_validity detects Inf values
#[test]
fn test_int_tensor_validity_inf_values() {
    let result = IntegrityChecker::check_tensor_validity(&[1.0, f32::INFINITY, 3.0]);
    assert!(!result.passed);
    assert!(result.description.contains("Inf"));
}

/// Verify check_format_fidelity detects altered weights
#[test]
fn test_int_format_fidelity_mismatch() {
    let result = IntegrityChecker::check_format_fidelity(
        "abc123def456",
        "xyz789ghi012",
    );
    assert!(!result.passed);
    assert!(result.description.contains("altered"));
    assert!(result.evidence.is_some());
}

/// Verify check_format_fidelity passes on identical hashes
#[test]
fn test_int_format_fidelity_match() {
    let result = IntegrityChecker::check_format_fidelity(
        "abc123def456",
        "abc123def456",
    );
    assert!(result.passed);
    assert!(result.description.contains("identical"));
}

/// Verify check_determinism detects non-deterministic output
#[test]
fn test_int_determinism_different_output() {
    let result = IntegrityChecker::check_determinism("hello world", "hello earth", 42);
    assert!(!result.passed);
    assert!(result.description.contains("Non-deterministic"));
    assert!(result.evidence.is_some());
    let evidence = result.evidence.unwrap();
    assert!(evidence.contains("position"));
}

/// Verify check_determinism passes on identical output
#[test]
fn test_int_determinism_same_output() {
    let result = IntegrityChecker::check_determinism("hello world", "hello world", 42);
    assert!(result.passed);
    assert!(result.description.contains("Deterministic"));
}

/// Verify PerformanceCheckResult new() stores all fields correctly
#[test]
fn test_performance_check_result_description_content() {
    let result = PerformanceValidator::check_tps(15.0, 10.0);
    assert!(result.description.contains("TPS"));
    assert!(result.description.contains("15.0"));
    assert!(result.description.contains("10.0"));
}

/// Verify failed performance check description contains threshold exceeded suffix
#[test]
fn test_performance_check_result_failure_suffix() {
    let result = PerformanceValidator::check_tps(5.0, 10.0);
    assert!(result.description.contains("threshold exceeded"));
}

/// Verify passing lower-is-better check description does not contain threshold exceeded
#[test]
fn test_performance_check_lower_is_better_pass_no_suffix() {
    let result = PerformanceValidator::check_ttft(100, 500);
    assert!(!result.description.contains("threshold exceeded"));
}

/// Verify ParityCheckResult description for identical format parity
#[test]
fn test_parity_format_pass_description() {
    let result = ParityChecker::check_format_parity(&[1, 2, 3], &[1, 2, 3]);
    assert!(result.description.contains("identical"));
}

/// Verify ParityCheckResult description reports token difference count
#[test]
fn test_parity_format_fail_description_count() {
    let result = ParityChecker::check_format_parity(&[1, 2, 3], &[1, 9, 9]);
    assert!(result.description.contains("2 token"));
}
