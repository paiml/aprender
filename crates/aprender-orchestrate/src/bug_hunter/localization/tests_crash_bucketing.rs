//! Tests for crash bucketing (CrashBucketer, RootCausePattern).

use std::path::PathBuf;

use crate::bug_hunter::types::{CrashBucketingMode, FindingSeverity};

use super::*;

#[test]
fn test_crash_pattern_detection() {
    assert_eq!(
        CrashBucketer::detect_pattern("index out of bounds: 5 >= 3"),
        RootCausePattern::IndexOutOfBounds
    );
    assert_eq!(
        CrashBucketer::detect_pattern("called `Option::unwrap()` on a `None` value"),
        RootCausePattern::UnwrapOnNone
    );
    assert_eq!(
        CrashBucketer::detect_pattern("integer overflow"),
        RootCausePattern::IntegerOverflow
    );
}

#[test]
fn test_semantic_bucketing_dedup() {
    let mut bucketer = CrashBucketer::new(CrashBucketingMode::Semantic);

    // Add 3 similar crashes
    for i in 0..3 {
        bucketer.add_crash(CrashInfo {
            id: format!("crash-{}", i),
            file: PathBuf::from("src/lib.rs"),
            line: 42,
            message: "index out of bounds: the len is 5 but the index is 10".to_string(),
            stack_trace: vec![],
        });
    }

    let (total, buckets) = bucketer.stats();
    assert_eq!(total, 3);
    assert_eq!(buckets, 1); // All 3 in same bucket

    let findings = bucketer.to_findings();
    assert_eq!(findings.len(), 1);
}

#[test]
fn test_crash_pattern_assertion() {
    assert_eq!(
        CrashBucketer::detect_pattern("assertion failed: x > 0"),
        RootCausePattern::AssertionFailed
    );
}

#[test]
fn test_crash_pattern_divide_by_zero() {
    assert_eq!(
        CrashBucketer::detect_pattern("attempt to divide by zero"),
        RootCausePattern::DivisionByZero
    );
}

#[test]
fn test_crash_pattern_stack_overflow() {
    assert_eq!(
        CrashBucketer::detect_pattern("thread 'main' has overflowed its stack"),
        RootCausePattern::StackOverflow
    );
}

#[test]
fn test_crash_pattern_null_pointer() {
    assert_eq!(
        CrashBucketer::detect_pattern("null pointer dereference"),
        RootCausePattern::NullPointerDeref
    );
}

#[test]
fn test_crash_pattern_unknown() {
    assert_eq!(
        CrashBucketer::detect_pattern("some random error message"),
        RootCausePattern::Unknown
    );
}

#[test]
fn test_none_bucketing() {
    let mut bucketer = CrashBucketer::new(CrashBucketingMode::None);

    bucketer.add_crash(CrashInfo {
        id: "crash-1".to_string(),
        file: PathBuf::from("src/lib.rs"),
        line: 42,
        message: "error 1".to_string(),
        stack_trace: vec![],
    });

    bucketer.add_crash(CrashInfo {
        id: "crash-2".to_string(),
        file: PathBuf::from("src/lib.rs"),
        line: 42,
        message: "error 2".to_string(),
        stack_trace: vec![],
    });

    let (total, buckets) = bucketer.stats();
    assert_eq!(total, 2);
    assert_eq!(buckets, 2); // Each crash gets its own bucket in None mode
}

// =========================================================================
// Coverage gap: StackTrace bucketing mode
// =========================================================================

#[test]
fn test_stack_trace_bucketing() {
    let mut bucketer = CrashBucketer::new(CrashBucketingMode::StackTrace);

    // Two crashes with same top-3 frames
    for i in 0..2 {
        bucketer.add_crash(CrashInfo {
            id: format!("crash-{}", i),
            file: PathBuf::from("src/lib.rs"),
            line: 42,
            message: "error".to_string(),
            stack_trace: vec![
                StackFrame { function: "fn_a".to_string(), file: None, line: None },
                StackFrame {
                    function: "fn_b".to_string(),
                    file: Some(PathBuf::from("src/lib.rs")),
                    line: Some(10),
                },
                StackFrame { function: "fn_c".to_string(), file: None, line: None },
            ],
        });
    }

    let (total, buckets) = bucketer.stats();
    assert_eq!(total, 2);
    assert_eq!(buckets, 1, "Same top-3 frames should be 1 bucket");
}

#[test]
fn test_stack_trace_bucketing_different_frames() {
    let mut bucketer = CrashBucketer::new(CrashBucketingMode::StackTrace);

    bucketer.add_crash(CrashInfo {
        id: "c1".to_string(),
        file: PathBuf::from("a.rs"),
        line: 1,
        message: "err".to_string(),
        stack_trace: vec![StackFrame { function: "fn_x".to_string(), file: None, line: None }],
    });

    bucketer.add_crash(CrashInfo {
        id: "c2".to_string(),
        file: PathBuf::from("b.rs"),
        line: 2,
        message: "err".to_string(),
        stack_trace: vec![StackFrame { function: "fn_y".to_string(), file: None, line: None }],
    });

    let (total, buckets) = bucketer.stats();
    assert_eq!(total, 2);
    assert_eq!(buckets, 2, "Different frames should give different buckets");
}

#[test]
fn test_stack_trace_bucketing_empty_frames() {
    let mut bucketer = CrashBucketer::new(CrashBucketingMode::StackTrace);

    bucketer.add_crash(CrashInfo {
        id: "c1".to_string(),
        file: PathBuf::from("a.rs"),
        line: 1,
        message: "err".to_string(),
        stack_trace: vec![],
    });

    let (total, buckets) = bucketer.stats();
    assert_eq!(total, 1);
    assert_eq!(buckets, 1);
    // Empty frames join to empty string
}

// =========================================================================
// Coverage gap: RootCausePattern Display
// =========================================================================

#[test]
fn test_root_cause_pattern_display() {
    assert_eq!(RootCausePattern::IndexOutOfBounds.to_string(), "index_out_of_bounds");
    assert_eq!(RootCausePattern::NullPointerDeref.to_string(), "null_pointer_deref");
    assert_eq!(RootCausePattern::IntegerOverflow.to_string(), "integer_overflow");
    assert_eq!(RootCausePattern::DivisionByZero.to_string(), "division_by_zero");
    assert_eq!(RootCausePattern::StackOverflow.to_string(), "stack_overflow");
    assert_eq!(RootCausePattern::HeapOverflow.to_string(), "heap_overflow");
    assert_eq!(RootCausePattern::UseAfterFree.to_string(), "use_after_free");
    assert_eq!(RootCausePattern::DoubleFree.to_string(), "double_free");
    assert_eq!(RootCausePattern::UnwrapOnNone.to_string(), "unwrap_on_none");
    assert_eq!(RootCausePattern::AssertionFailed.to_string(), "assertion_failed");
    assert_eq!(RootCausePattern::Unknown.to_string(), "unknown");
}

// =========================================================================
// Coverage gap: detect_pattern edge cases
// =========================================================================

#[test]
fn test_detect_pattern_use_after_free() {
    assert_eq!(
        CrashBucketer::detect_pattern("use after free in allocator"),
        RootCausePattern::UseAfterFree
    );
}

#[test]
fn test_detect_pattern_double_free() {
    assert_eq!(CrashBucketer::detect_pattern("double free detected"), RootCausePattern::DoubleFree);
}

#[test]
fn test_detect_pattern_heap_overflow() {
    assert_eq!(
        CrashBucketer::detect_pattern("heap buffer overflow"),
        RootCausePattern::HeapOverflow
    );
}

#[test]
fn test_detect_pattern_indexoutofbounds_single_word() {
    assert_eq!(
        CrashBucketer::detect_pattern("IndexOutOfBounds exception"),
        RootCausePattern::IndexOutOfBounds
    );
}

#[test]
fn test_detect_pattern_nullptr() {
    assert_eq!(
        CrashBucketer::detect_pattern("nullptr dereference"),
        RootCausePattern::NullPointerDeref
    );
}

#[test]
fn test_detect_pattern_unwrap_none_variant() {
    assert_eq!(
        CrashBucketer::detect_pattern("unwrap called on None value"),
        RootCausePattern::UnwrapOnNone
    );
}

#[test]
fn test_detect_pattern_division_by_zero_variant() {
    assert_eq!(
        CrashBucketer::detect_pattern("division by zero error"),
        RootCausePattern::DivisionByZero
    );
}

#[test]
fn test_detect_pattern_assert_keyword() {
    assert_eq!(
        CrashBucketer::detect_pattern("assert_eq failed: 1 != 2"),
        RootCausePattern::AssertionFailed
    );
}

// =========================================================================
// Coverage gap: to_findings() content verification
// =========================================================================

#[test]
fn test_to_findings_content() {
    let mut bucketer = CrashBucketer::new(CrashBucketingMode::Semantic);

    bucketer.add_crash(CrashInfo {
        id: "crash-1".to_string(),
        file: PathBuf::from("src/parser.rs"),
        line: 100,
        message: "index out of bounds: len is 3 but index is 5".to_string(),
        stack_trace: vec![],
    });
    bucketer.add_crash(CrashInfo {
        id: "crash-2".to_string(),
        file: PathBuf::from("src/parser.rs"),
        line: 100,
        message: "index out of bounds: len is 10 but index is 20".to_string(),
        stack_trace: vec![],
    });

    let findings = bucketer.to_findings();
    assert_eq!(findings.len(), 1);

    let f = &findings[0];
    assert!(f.id.starts_with("BH-CRASH-"));
    assert_eq!(f.file, PathBuf::from("src/parser.rs"));
    assert_eq!(f.line, 100);
    assert_eq!(f.severity, FindingSeverity::High);
    assert!((f.suspiciousness - 0.8).abs() < f64::EPSILON);
    assert!(f.description.contains("2 occurrence(s)"));
}

#[test]
fn test_to_findings_empty_bucketer() {
    let bucketer = CrashBucketer::new(CrashBucketingMode::Semantic);
    let findings = bucketer.to_findings();
    assert!(findings.is_empty());
}

// =========================================================================
// Coverage gap: Semantic bucketing with different files
// =========================================================================

#[test]
fn test_semantic_bucketing_different_files() {
    let mut bucketer = CrashBucketer::new(CrashBucketingMode::Semantic);

    bucketer.add_crash(CrashInfo {
        id: "c1".to_string(),
        file: PathBuf::from("src/a.rs"),
        line: 10,
        message: "index out of bounds".to_string(),
        stack_trace: vec![],
    });
    bucketer.add_crash(CrashInfo {
        id: "c2".to_string(),
        file: PathBuf::from("src/b.rs"),
        line: 20,
        message: "index out of bounds".to_string(),
        stack_trace: vec![],
    });

    let (total, buckets) = bucketer.stats();
    assert_eq!(total, 2);
    // Same pattern but different files => different bucket keys
    assert_eq!(buckets, 2);
}
