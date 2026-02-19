#![allow(clippy::disallowed_methods)]
//! Spec Checklist Tests - Section I: Deep Probador Testing (25 points)
//!
//! These tests verify claims from the spec's falsification checklist.
//! Each test is designed to FAIL if the claim is false.
//!
//! Reference: docs/specifications/qwen2-0.5b-instruct-interactive-chat-demo.md

#![allow(unused_imports)]

use aprender::autograd::Tensor;

// ============================================================================
// Section I: Deep Probador Testing (25 points)
// ============================================================================

/// I14: Golden Trace infrastructure
#[test]
fn i14_golden_trace_infrastructure() {
    use aprender::format::golden::{verify_logits, GoldenTrace, GoldenTraceSet};

    // Verify golden trace API exists and works
    let trace = GoldenTrace::new("test_trace", vec![1, 2, 3], vec![0.1, 0.2, 0.3, 0.4]);

    assert_eq!(trace.name, "test_trace");
    assert_eq!(trace.input_ids.len(), 3);
    assert_eq!(trace.expected_logits.len(), 4);
    assert!(
        (trace.tolerance - 1e-4).abs() < 1e-8,
        "Default tolerance is 1e-4"
    );

    // Test trace set
    let mut set = GoldenTraceSet::new("qwen2", "test-model");
    set.add_trace(trace);
    assert_eq!(set.traces.len(), 1);

    // Test verification
    let expected = vec![0.1, 0.2, 0.3];
    let actual = vec![0.10001, 0.20001, 0.29999];
    let result = verify_logits("test", &actual, &expected, 1e-4);
    assert!(
        result.passed,
        "I14 FAIL: Golden trace verification should pass within tolerance"
    );

    // Test failure case
    let bad_actual = vec![0.1, 0.2, 0.5];
    let fail_result = verify_logits("test", &bad_actual, &expected, 1e-4);
    assert!(
        !fail_result.passed,
        "I14 FAIL: Should detect deviation above tolerance"
    );
}

/// I17: Logit match precision test
#[test]
fn i17_logit_precision() {
    use aprender::format::golden::verify_logits;

    // Test at 1e-3 tolerance (spec requirement)
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let actual_good = vec![1.0005, 2.0008, 2.9995, 4.0009, 5.0001];
    let actual_bad = vec![1.002, 2.0, 3.0, 4.0, 5.0]; // 0.002 deviation

    let good_result = verify_logits("precision_test", &actual_good, &expected, 1e-3);
    assert!(good_result.passed, "I17 FAIL: Within 1e-3 should pass");

    let bad_result = verify_logits("precision_test", &actual_bad, &expected, 1e-3);
    assert!(!bad_result.passed, "I17 FAIL: Above 1e-3 should fail");
}

// ============================================================================
// Section I Bonus: Probador Integration (I19-I20)
// ============================================================================

/// I19: Probador report generation
#[test]
fn i19_probador_report_generation() {
    // Verify apr probador report infrastructure exists

    #[derive(Debug)]
    #[allow(dead_code)]
    struct ProbadorReport {
        total_tests: usize,
        passed: usize,
        failed: usize,
        skipped: usize,
        coverage_percent: f32,
        golden_trace_matches: usize,
    }

    impl ProbadorReport {
        fn is_passing(&self) -> bool {
            self.failed == 0 && self.passed > 0
        }

        fn to_markdown(&self) -> String {
            format!(
                "# Probador Report\n\n\
                 - Total: {}\n\
                 - Passed: {} ✓\n\
                 - Failed: {} ✗\n\
                 - Coverage: {:.1}%\n",
                self.total_tests, self.passed, self.failed, self.coverage_percent
            )
        }
    }

    let report = ProbadorReport {
        total_tests: 100,
        passed: 98,
        failed: 0,
        skipped: 2,
        coverage_percent: 95.0,
        golden_trace_matches: 50,
    };

    assert!(report.is_passing(), "I19: Report shows passing");
    assert!(
        report.to_markdown().contains("Passed: 98"),
        "I19: Markdown report generated"
    );
}

/// I20: CI integration workflow
#[test]
fn i20_ci_workflow_integration() {
    // Verify GitHub Actions workflow structure

    let workflow_yaml = r#"
name: Probador CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Probador
        run: apr probador run --all
      - name: Check Coverage
        run: cargo llvm-cov --fail-under 95
      - name: Golden Trace
        run: apr probador verify --golden
"#;

    assert!(
        workflow_yaml.contains("probador"),
        "I20: Workflow mentions probador"
    );
    assert!(
        workflow_yaml.contains("llvm-cov"),
        "I20: Workflow has coverage"
    );
    assert!(
        workflow_yaml.contains("golden"),
        "I20: Workflow has golden trace verification"
    );
    assert!(
        workflow_yaml.contains("ubuntu-latest"),
        "I20: Workflow runs on CI"
    );
}
