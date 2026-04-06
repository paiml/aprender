// Comprehensive tests for N-gram sequence mining
//
// Toyota Way Principle: Genchi Genbutsu (Go and See)
// - Test with real-world syscall patterns from transpilers
// - Validate against examples from specification
// - Ensure false positive reduction

use super::*;
use crate::cluster::Severity;
use crate::sequence::ngram::{ngram_coverage, top_ngrams};

/// Test real-world example from specification: decy futex anomaly
#[test]
fn test_decy_futex_anomaly() {
    // Baseline: decy without async runtime (single-threaded)
    let baseline_syscalls =
        vec!["mmap".to_string(), "read".to_string(), "write".to_string(), "close".to_string()];

    // Current: decy with accidental async runtime initialization
    let current_syscalls = vec![
        "mmap".to_string(),
        "read".to_string(),
        "futex".to_string(), // NEW: synchronization
        "write".to_string(),
        "close".to_string(),
    ];

    let baseline_ngrams = extract_ngrams(&baseline_syscalls, 3);
    let current_ngrams = extract_ngrams(&current_syscalls, 3);

    let anomalies = detect_sequence_anomalies(&baseline_ngrams, &current_ngrams, 0.30);

    // Should detect new sequences involving futex
    let futex_anomalies: Vec<_> =
        anomalies.iter().filter(|a| a.ngram.iter().any(|s| s == "futex")).collect();

    assert!(!futex_anomalies.is_empty());
    assert!(futex_anomalies.iter().any(|a| a.severity == Severity::High));
}

/// Test real-world example: depyler networking telemetry leak
#[test]
fn test_depyler_telemetry_leak() {
    // Baseline: depyler without telemetry
    let baseline_syscalls = vec!["mmap".to_string(), "read".to_string(), "write".to_string()];

    // Current: depyler with sentry-rs telemetry
    let current_syscalls = vec![
        "mmap".to_string(),
        "socket".to_string(),  // NEW: telemetry
        "connect".to_string(), // NEW: telemetry
        "send".to_string(),    // NEW: telemetry
        "read".to_string(),
        "write".to_string(),
    ];

    let baseline_ngrams = extract_ngrams(&baseline_syscalls, 3);
    let current_ngrams = extract_ngrams(&current_syscalls, 3);

    let anomalies = detect_sequence_anomalies(&baseline_ngrams, &current_ngrams, 0.30);

    // Should detect networking sequence as CRITICAL
    let networking_anomalies: Vec<_> = anomalies
        .iter()
        .filter(|a| a.ngram.iter().any(|s| s.contains("socket") || s.contains("connect")))
        .collect();

    assert!(!networking_anomalies.is_empty());
    assert!(networking_anomalies.iter().any(|a| a.severity == Severity::Critical));
}

/// Test benign changes don't trigger false positives
#[test]
fn test_no_false_positive_on_order_preserving_change() {
    // Baseline: 10 iterations of pattern
    let mut baseline_syscalls = Vec::new();
    for _ in 0..10 {
        baseline_syscalls.push("mmap".to_string());
        baseline_syscalls.push("read".to_string());
        baseline_syscalls.push("write".to_string());
    }

    // Current: 15 iterations of same pattern (frequency increase, but pattern preserved)
    let mut current_syscalls = Vec::new();
    for _ in 0..15 {
        current_syscalls.push("mmap".to_string());
        current_syscalls.push("read".to_string());
        current_syscalls.push("write".to_string());
    }

    let baseline_ngrams = extract_ngrams(&baseline_syscalls, 3);
    let current_ngrams = extract_ngrams(&current_syscalls, 3);

    let anomalies = detect_sequence_anomalies(&baseline_ngrams, &current_ngrams, 0.30);

    // Should detect frequency change, but not new/missing sequences
    assert!(anomalies.iter().all(|a| a.anomaly_type == AnomalyType::FrequencyChange));

    // Should NOT be Critical severity (no networking/sync)
    assert!(anomalies.iter().all(|a| a.severity != Severity::Critical));
}

/// Test grammar violation detection (sequence reordering)
#[test]
fn test_grammar_violation_reordering() {
    // Baseline: A → B → C
    let baseline_syscalls = vec!["open".to_string(), "read".to_string(), "close".to_string()];

    // Current: A → C → B (reordered!)
    let current_syscalls = vec![
        "open".to_string(),
        "close".to_string(), // Swapped
        "read".to_string(),  // Swapped
    ];

    let baseline_ngrams = extract_ngrams(&baseline_syscalls, 2);
    let current_ngrams = extract_ngrams(&current_syscalls, 2);

    let anomalies = detect_sequence_anomalies(&baseline_ngrams, &current_ngrams, 0.30);

    // Should detect:
    // - New sequence: ["open", "close"]
    // - New sequence: ["close", "read"]
    // - Missing sequence: ["open", "read"]
    // - Missing sequence: ["read", "close"]
    assert!(anomalies.len() >= 2);
}

/// Test tight loop detection via N-gram coverage
#[test]
fn test_tight_loop_detection() {
    // Tight loop: same pattern repeated 1000 times
    let mut tight_loop_syscalls = Vec::new();
    for _ in 0..1000 {
        tight_loop_syscalls.push("futex".to_string());
        tight_loop_syscalls.push("futex".to_string());
    }

    let ngrams = extract_ngrams(&tight_loop_syscalls, 2);
    let coverage = ngram_coverage(&ngrams);

    // Should have very low coverage (only 1 unique N-gram)
    assert!(coverage < 0.01); // <1% coverage indicates tight loop
}

/// Test diverse syscall pattern (normal transpiler execution)
#[test]
fn test_diverse_pattern_normal() {
    let diverse_syscalls = vec![
        "mmap".to_string(),
        "read".to_string(),
        "write".to_string(),
        "mmap".to_string(),
        "brk".to_string(),
        "write".to_string(),
        "fsync".to_string(),
        "close".to_string(),
    ];

    let ngrams = extract_ngrams(&diverse_syscalls, 3);
    let coverage = ngram_coverage(&ngrams);

    // Should have high coverage (many unique N-grams)
    assert!(coverage > 0.5); // >50% coverage indicates diverse patterns
}

/// Test top N-grams identification for profiling
#[test]
fn test_top_ngrams_profiling() {
    let mut syscalls = vec!["mmap".to_string(), "read".to_string(), "write".to_string()];

    // Repeat hot path 10 times
    for _ in 0..10 {
        syscalls.push("mmap".to_string());
        syscalls.push("read".to_string());
    }

    let ngrams = extract_ngrams(&syscalls, 2);
    let top = top_ngrams(&ngrams, 1);

    // Most frequent N-gram should be the hot path
    assert_eq!(top[0].0, vec!["mmap".to_string(), "read".to_string()]);
    assert_eq!(top[0].1, 11); // 1 + 10 repetitions
}

/// Test empty trace handling
#[test]
fn test_empty_trace() {
    let empty_syscalls: Vec<String> = Vec::new();
    let ngrams = extract_ngrams(&empty_syscalls, 3);

    assert!(ngrams.is_empty());
    assert_eq!(ngram_coverage(&ngrams), 0.0);
}

/// Test anomaly report formatting
#[test]
fn test_anomaly_report_format() {
    let anomaly = SequenceAnomaly {
        ngram: vec!["socket".to_string(), "connect".to_string(), "send".to_string()],
        baseline_freq: 0,
        current_freq: 5,
        anomaly_type: AnomalyType::NewSequence,
        severity: Severity::Critical,
    };

    let report = anomaly.to_report_string();

    // Should contain key information
    assert!(report.contains("NEW SEQUENCE"));
    assert!(report.contains("socket"));
    assert!(report.contains("connect"));
    assert!(report.contains("send"));
    assert!(report.contains("CRITICAL"));
    assert!(report.contains("5 occurrences"));
}

/// Test configuration-driven frequency threshold
#[test]
fn test_configurable_frequency_threshold() {
    let mut baseline_ngrams = NGramMap::new();
    baseline_ngrams.insert(vec!["a".to_string(), "b".to_string()], 100);

    let mut current_ngrams = NGramMap::new();
    current_ngrams.insert(vec!["a".to_string(), "b".to_string()], 120);

    // Strict threshold (10%): should NOT trigger
    let strict_anomalies = detect_sequence_anomalies(&baseline_ngrams, &current_ngrams, 0.30);
    assert!(strict_anomalies.iter().all(|a| a.anomaly_type != AnomalyType::FrequencyChange));

    // Loose threshold (5%): should trigger
    let loose_anomalies = detect_sequence_anomalies(&baseline_ngrams, &current_ngrams, 0.05);
    assert!(loose_anomalies.iter().any(|a| a.anomaly_type == AnomalyType::FrequencyChange));
}
