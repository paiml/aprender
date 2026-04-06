//! Comparison logic for golden trace validation
//!
//! Per specification Section 2.2: Syscall Sequence Comparison

use super::error::{Result, ValidateError};
use super::golden_trace::{GoldenBaseline, SyscallTimingStats, TraceSyscallEntry};

/// Result of comparing two syscall sequences
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Whether the comparison passed
    pub passed: bool,
    /// Syscall mismatches found
    pub syscall_mismatches: Vec<SyscallMismatch>,
    /// Timing regressions found
    pub timing_regressions: Vec<TimingRegression>,
    /// Summary statistics
    pub summary: ComparisonSummary,
}

impl ComparisonResult {
    /// Create a passing result
    pub fn passed() -> Self {
        Self {
            passed: true,
            syscall_mismatches: Vec::new(),
            timing_regressions: Vec::new(),
            summary: ComparisonSummary::default(),
        }
    }

    /// Create a failed result with mismatches
    pub fn failed(mismatches: Vec<SyscallMismatch>, regressions: Vec<TimingRegression>) -> Self {
        Self {
            passed: false,
            syscall_mismatches: mismatches,
            timing_regressions: regressions,
            summary: ComparisonSummary::default(),
        }
    }
}

/// A mismatch between expected and actual syscalls
#[derive(Debug, Clone)]
pub struct SyscallMismatch {
    /// Index in the sequence
    pub index: usize,
    /// Expected syscall name/number
    pub expected: String,
    /// Actual syscall name/number
    pub found: String,
    /// Type of mismatch
    pub mismatch_type: MismatchType,
}

/// Type of syscall mismatch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MismatchType {
    /// Different syscall at same position
    Different,
    /// Extra syscall in actual trace
    Extra,
    /// Missing syscall from baseline
    Missing,
}

/// A timing regression detected
#[derive(Debug, Clone)]
pub struct TimingRegression {
    /// Syscall name
    pub syscall: String,
    /// Baseline timing in milliseconds
    pub baseline_ms: f64,
    /// Actual timing in milliseconds
    pub actual_ms: f64,
    /// Delta percentage (positive = regression)
    pub delta_percent: f64,
}

/// Summary of comparison results
#[derive(Debug, Clone, Default)]
pub struct ComparisonSummary {
    /// Total syscalls compared
    pub total_compared: usize,
    /// Number of matches
    pub matches: usize,
    /// Number of mismatches
    pub mismatches: usize,
    /// Number of timing regressions
    pub timing_regressions: usize,
}

/// Compare two syscall sequences
///
/// Returns Ok with comparison result, or Err for structural issues
pub fn compare_syscalls(
    baseline: &[TraceSyscallEntry],
    actual: &[TraceSyscallEntry],
    strict: bool,
) -> Result<ComparisonResult> {
    let mut mismatches = Vec::new();

    // Check count mismatch
    if baseline.len() != actual.len() && strict {
        return Err(ValidateError::SyscallCountMismatch {
            baseline: baseline.len(),
            current: actual.len(),
        });
        // In non-strict mode (when this block is not entered), report as mismatch but continue
    }

    // Compare sequences
    let compare_len = baseline.len().min(actual.len());
    for i in 0..compare_len {
        if baseline[i].syscall_nr != actual[i].syscall_nr {
            mismatches.push(SyscallMismatch {
                index: i,
                expected: format!("syscall_{}", baseline[i].syscall_nr),
                found: format!("syscall_{}", actual[i].syscall_nr),
                mismatch_type: MismatchType::Different,
            });
        }
    }

    // Report extra syscalls in actual
    for (i, entry) in actual.iter().enumerate().skip(compare_len) {
        mismatches.push(SyscallMismatch {
            index: i,
            expected: "(none)".to_string(),
            found: format!("syscall_{}", entry.syscall_nr),
            mismatch_type: MismatchType::Extra,
        });
    }

    // Report missing syscalls from baseline
    for (i, entry) in baseline.iter().enumerate().skip(compare_len) {
        mismatches.push(SyscallMismatch {
            index: i,
            expected: format!("syscall_{}", entry.syscall_nr),
            found: "(none)".to_string(),
            mismatch_type: MismatchType::Missing,
        });
    }

    let summary = ComparisonSummary {
        total_compared: compare_len,
        matches: compare_len
            - mismatches.iter().filter(|m| m.mismatch_type == MismatchType::Different).count(),
        mismatches: mismatches.len(),
        timing_regressions: 0,
    };

    if mismatches.is_empty() {
        Ok(ComparisonResult {
            passed: true,
            syscall_mismatches: Vec::new(),
            timing_regressions: Vec::new(),
            summary,
        })
    } else {
        Ok(ComparisonResult {
            passed: false,
            syscall_mismatches: mismatches,
            timing_regressions: Vec::new(),
            summary,
        })
    }
}

/// Compare timing with tolerance
///
/// Returns list of timing regressions that exceed tolerance
pub fn compare_timing(
    baseline_stats: &std::collections::HashMap<String, SyscallTimingStats>,
    actual_stats: &std::collections::HashMap<String, SyscallTimingStats>,
    tolerance_percent: f32,
) -> Vec<TimingRegression> {
    let mut regressions = Vec::new();

    for (syscall, baseline) in baseline_stats {
        if let Some(actual) = actual_stats.get(syscall) {
            if let Some(regression) =
                is_timing_regression(syscall, baseline.mean_ns, actual.mean_ns, tolerance_percent)
            {
                regressions.push(regression);
            }
        }
    }

    regressions
}

/// Check if timing represents a regression
///
/// Returns Some(TimingRegression) if the actual timing exceeds baseline by more than tolerance
pub fn is_timing_regression(
    syscall: &str,
    baseline_ns: u64,
    actual_ns: u64,
    tolerance_percent: f32,
) -> Option<TimingRegression> {
    if baseline_ns == 0 {
        return None;
    }

    let baseline_ms = baseline_ns as f64 / 1_000_000.0;
    let actual_ms = actual_ns as f64 / 1_000_000.0;
    let delta_percent = ((actual_ns as f64 - baseline_ns as f64) / baseline_ns as f64) * 100.0;

    if delta_percent > f64::from(tolerance_percent) {
        Some(TimingRegression {
            syscall: syscall.to_string(),
            baseline_ms,
            actual_ms,
            delta_percent,
        })
    } else {
        None
    }
}

/// Perform full validation comparison
pub fn validate_against_baseline(
    baseline: &GoldenBaseline,
    actual_syscalls: &[TraceSyscallEntry],
    actual_timing: &std::collections::HashMap<String, SyscallTimingStats>,
    tolerance_percent: f32,
    strict: bool,
    ignore_timing: bool,
) -> Result<ComparisonResult> {
    // Compare syscall sequences
    let mut result = compare_syscalls(&baseline.syscalls, actual_syscalls, strict)?;

    // Compare timing unless ignored
    if !ignore_timing {
        let timing_regressions =
            compare_timing(&baseline.timing.by_syscall, actual_timing, tolerance_percent);

        if !timing_regressions.is_empty() {
            result.passed = false;
            result.timing_regressions = timing_regressions;
        }
    }

    result.summary.timing_regressions = result.timing_regressions.len();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_compare_identical_sequences() {
        let baseline = vec![
            TraceSyscallEntry::simple(1, "read", 100),
            TraceSyscallEntry::simple(2, "write", 200),
        ];
        let actual = vec![
            TraceSyscallEntry::simple(1, "read", 100),
            TraceSyscallEntry::simple(2, "write", 200),
        ];

        let result =
            compare_syscalls(&baseline, &actual, false).expect("comparison should succeed");
        assert!(result.passed);
        assert!(result.syscall_mismatches.is_empty());
    }

    #[test]
    fn test_compare_different_sequences() {
        let baseline = vec![TraceSyscallEntry::simple(1, "read", 100)];
        let actual = vec![TraceSyscallEntry::simple(2, "write", 100)];

        let result =
            compare_syscalls(&baseline, &actual, false).expect("comparison should succeed");
        assert!(!result.passed);
        assert_eq!(result.syscall_mismatches.len(), 1);
        assert_eq!(result.syscall_mismatches[0].mismatch_type, MismatchType::Different);
    }

    #[test]
    fn test_compare_strict_mode_count_mismatch() {
        let baseline = vec![
            TraceSyscallEntry::simple(1, "read", 100),
            TraceSyscallEntry::simple(2, "write", 200),
        ];
        let actual = vec![TraceSyscallEntry::simple(1, "read", 100)];

        // Strict mode should return error on count mismatch
        let result = compare_syscalls(&baseline, &actual, true);
        assert!(result.is_err());

        // Non-strict mode should report mismatches but continue
        let result =
            compare_syscalls(&baseline, &actual, false).expect("comparison should succeed");
        assert!(!result.passed);
        assert_eq!(result.syscall_mismatches.len(), 1); // 1 missing
    }

    #[test]
    fn test_compare_extra_syscalls() {
        let baseline = vec![TraceSyscallEntry::simple(1, "read", 100)];
        let actual = vec![
            TraceSyscallEntry::simple(1, "read", 100),
            TraceSyscallEntry::simple(2, "write", 200),
            TraceSyscallEntry::simple(3, "close", 50),
        ];

        let result =
            compare_syscalls(&baseline, &actual, false).expect("comparison should succeed");
        assert!(!result.passed);
        assert_eq!(result.syscall_mismatches.len(), 2); // 2 extra
        assert!(result.syscall_mismatches.iter().all(|m| m.mismatch_type == MismatchType::Extra));
    }

    #[test]
    fn test_compare_missing_syscalls() {
        let baseline = vec![
            TraceSyscallEntry::simple(1, "read", 100),
            TraceSyscallEntry::simple(2, "write", 200),
            TraceSyscallEntry::simple(3, "close", 50),
        ];
        let actual = vec![TraceSyscallEntry::simple(1, "read", 100)];

        let result =
            compare_syscalls(&baseline, &actual, false).expect("comparison should succeed");
        assert!(!result.passed);
        assert_eq!(result.syscall_mismatches.len(), 2); // 2 missing
        assert!(result.syscall_mismatches.iter().all(|m| m.mismatch_type == MismatchType::Missing));
    }

    #[test]
    fn test_compare_empty_sequences() {
        let baseline: Vec<TraceSyscallEntry> = vec![];
        let actual: Vec<TraceSyscallEntry> = vec![];

        let result =
            compare_syscalls(&baseline, &actual, false).expect("comparison should succeed");
        assert!(result.passed);
        assert!(result.syscall_mismatches.is_empty());
    }

    #[test]
    fn test_timing_within_tolerance() {
        let result = is_timing_regression("read", 100_000, 105_000, 10.0);
        assert!(result.is_none()); // 5% is within 10% tolerance
    }

    #[test]
    fn test_timing_exceeds_tolerance() {
        let result = is_timing_regression("read", 100_000, 120_000, 10.0);
        assert!(result.is_some()); // 20% exceeds 10% tolerance
        let regression = result.expect("regression should be detected");
        assert!((regression.delta_percent - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_timing_zero_baseline() {
        let result = is_timing_regression("read", 0, 100_000, 10.0);
        assert!(result.is_none()); // Should handle zero baseline gracefully
    }

    #[test]
    fn test_timing_at_exactly_tolerance() {
        let result = is_timing_regression("read", 100_000, 110_000, 10.0);
        assert!(result.is_none()); // 10% is exactly at tolerance (not exceeding)
    }

    #[test]
    fn test_timing_just_over_tolerance() {
        let result = is_timing_regression("read", 100_000, 110_001, 10.0);
        assert!(result.is_some()); // Just over 10%
    }

    #[test]
    fn test_compare_timing_map() {
        let mut baseline_stats = HashMap::new();
        baseline_stats.insert(
            "read".to_string(),
            SyscallTimingStats {
                count: 10,
                total_ns: 1_000_000,
                mean_ns: 100_000,
                std_ns: 10_000,
                min_ns: 80_000,
                max_ns: 120_000,
                p50_ns: 100_000,
                p95_ns: 115_000,
                p99_ns: 118_000,
            },
        );

        let mut actual_stats = HashMap::new();
        actual_stats.insert(
            "read".to_string(),
            SyscallTimingStats {
                count: 10,
                total_ns: 1_200_000,
                mean_ns: 120_000, // 20% slower
                std_ns: 10_000,
                min_ns: 100_000,
                max_ns: 140_000,
                p50_ns: 120_000,
                p95_ns: 135_000,
                p99_ns: 138_000,
            },
        );

        let regressions = compare_timing(&baseline_stats, &actual_stats, 10.0);
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].syscall, "read");
    }

    #[test]
    fn test_compare_timing_no_regression() {
        let mut baseline_stats = HashMap::new();
        baseline_stats.insert(
            "read".to_string(),
            SyscallTimingStats {
                count: 10,
                total_ns: 1_000_000,
                mean_ns: 100_000,
                std_ns: 10_000,
                min_ns: 80_000,
                max_ns: 120_000,
                p50_ns: 100_000,
                p95_ns: 115_000,
                p99_ns: 118_000,
            },
        );

        let mut actual_stats = HashMap::new();
        actual_stats.insert(
            "read".to_string(),
            SyscallTimingStats {
                count: 10,
                total_ns: 1_050_000,
                mean_ns: 105_000, // 5% slower (within 10% tolerance)
                std_ns: 10_000,
                min_ns: 85_000,
                max_ns: 125_000,
                p50_ns: 105_000,
                p95_ns: 120_000,
                p99_ns: 123_000,
            },
        );

        let regressions = compare_timing(&baseline_stats, &actual_stats, 10.0);
        assert!(regressions.is_empty());
    }

    #[test]
    fn test_compare_timing_missing_syscall() {
        let mut baseline_stats = HashMap::new();
        baseline_stats.insert(
            "read".to_string(),
            SyscallTimingStats {
                count: 10,
                total_ns: 1_000_000,
                mean_ns: 100_000,
                std_ns: 10_000,
                min_ns: 80_000,
                max_ns: 120_000,
                p50_ns: 100_000,
                p95_ns: 115_000,
                p99_ns: 118_000,
            },
        );

        let actual_stats = HashMap::new(); // empty

        let regressions = compare_timing(&baseline_stats, &actual_stats, 10.0);
        assert!(regressions.is_empty()); // Missing syscall shouldn't be a regression
    }

    #[test]
    fn test_validate_against_baseline_with_timing() {
        use super::super::golden_trace::{
            GoldenBaseline, PlatformInfo, TimingStats, ToleranceConfig, TraceManifest,
            TraceStatistics,
        };

        let baseline = GoldenBaseline {
            manifest: TraceManifest {
                version: "1.0.0".to_string(),
                renacer_version: "0.1.0".to_string(),
                created_at: "2025-01-01T00:00:00Z".to_string(),
                platform: PlatformInfo {
                    os: "linux".to_string(),
                    arch: "x86_64".to_string(),
                    kernel: "6.0.0".to_string(),
                },
                command: vec!["echo".to_string(), "test".to_string()],
                statistics: TraceStatistics {
                    total_syscalls: 2,
                    total_duration_ms: 0.3,
                    unique_syscalls: 2,
                },
                tolerance: ToleranceConfig::default(),
            },
            syscalls: vec![
                TraceSyscallEntry::simple(1, "read", 100_000),
                TraceSyscallEntry::simple(2, "write", 200_000),
            ],
            timing: TimingStats {
                total_duration_ns: 300_000,
                syscall_count: 2,
                by_syscall: {
                    let mut map = HashMap::new();
                    map.insert(
                        "read".to_string(),
                        SyscallTimingStats {
                            count: 1,
                            total_ns: 100_000,
                            mean_ns: 100_000,
                            std_ns: 0,
                            min_ns: 100_000,
                            max_ns: 100_000,
                            p50_ns: 100_000,
                            p95_ns: 100_000,
                            p99_ns: 100_000,
                        },
                    );
                    map
                },
            },
        };

        let actual_syscalls = vec![
            TraceSyscallEntry::simple(1, "read", 100_000),
            TraceSyscallEntry::simple(2, "write", 200_000),
        ];

        let mut actual_timing = HashMap::new();
        actual_timing.insert(
            "read".to_string(),
            SyscallTimingStats {
                count: 1,
                total_ns: 150_000,
                mean_ns: 150_000, // 50% slower
                std_ns: 0,
                min_ns: 150_000,
                max_ns: 150_000,
                p50_ns: 150_000,
                p95_ns: 150_000,
                p99_ns: 150_000,
            },
        );

        let result = validate_against_baseline(
            &baseline,
            &actual_syscalls,
            &actual_timing,
            10.0,
            false,
            false,
        )
        .expect("validation should succeed");
        assert!(!result.passed);
        assert_eq!(result.timing_regressions.len(), 1);
    }

    #[test]
    fn test_validate_against_baseline_ignore_timing() {
        use super::super::golden_trace::{
            GoldenBaseline, PlatformInfo, TimingStats, ToleranceConfig, TraceManifest,
            TraceStatistics,
        };

        let baseline = GoldenBaseline {
            manifest: TraceManifest {
                version: "1.0.0".to_string(),
                renacer_version: "0.1.0".to_string(),
                created_at: "2025-01-01T00:00:00Z".to_string(),
                platform: PlatformInfo {
                    os: "linux".to_string(),
                    arch: "x86_64".to_string(),
                    kernel: "6.0.0".to_string(),
                },
                command: vec!["echo".to_string(), "test".to_string()],
                statistics: TraceStatistics {
                    total_syscalls: 1,
                    total_duration_ms: 0.1,
                    unique_syscalls: 1,
                },
                tolerance: ToleranceConfig::default(),
            },
            syscalls: vec![TraceSyscallEntry::simple(1, "read", 100_000)],
            timing: TimingStats {
                total_duration_ns: 100_000,
                syscall_count: 1,
                by_syscall: {
                    let mut map = HashMap::new();
                    map.insert(
                        "read".to_string(),
                        SyscallTimingStats {
                            count: 1,
                            total_ns: 100_000,
                            mean_ns: 100_000,
                            std_ns: 0,
                            min_ns: 100_000,
                            max_ns: 100_000,
                            p50_ns: 100_000,
                            p95_ns: 100_000,
                            p99_ns: 100_000,
                        },
                    );
                    map
                },
            },
        };

        let actual_syscalls = vec![TraceSyscallEntry::simple(1, "read", 100_000)];

        let mut actual_timing = HashMap::new();
        actual_timing.insert(
            "read".to_string(),
            SyscallTimingStats {
                count: 1,
                total_ns: 1_000_000,
                mean_ns: 1_000_000, // 10x slower - would fail normally
                std_ns: 0,
                min_ns: 1_000_000,
                max_ns: 1_000_000,
                p50_ns: 1_000_000,
                p95_ns: 1_000_000,
                p99_ns: 1_000_000,
            },
        );

        // With ignore_timing=true, timing regression should be ignored
        let result = validate_against_baseline(
            &baseline,
            &actual_syscalls,
            &actual_timing,
            10.0,
            false,
            true,
        )
        .expect("validation should succeed");
        assert!(result.passed); // Should pass because timing is ignored
        assert!(result.timing_regressions.is_empty());
    }

    #[test]
    fn test_comparison_result_constructors() {
        let passed = ComparisonResult::passed();
        assert!(passed.passed);
        assert!(passed.syscall_mismatches.is_empty());

        let failed = ComparisonResult::failed(
            vec![SyscallMismatch {
                index: 0,
                expected: "read".to_string(),
                found: "write".to_string(),
                mismatch_type: MismatchType::Different,
            }],
            vec![],
        );
        assert!(!failed.passed);
        assert_eq!(failed.syscall_mismatches.len(), 1);
    }

    #[test]
    fn test_mismatch_type_equality() {
        assert_eq!(MismatchType::Different, MismatchType::Different);
        assert_eq!(MismatchType::Extra, MismatchType::Extra);
        assert_eq!(MismatchType::Missing, MismatchType::Missing);
        assert_ne!(MismatchType::Different, MismatchType::Extra);
    }

    #[test]
    fn test_comparison_summary_default() {
        let summary = ComparisonSummary::default();
        assert_eq!(summary.total_compared, 0);
        assert_eq!(summary.matches, 0);
        assert_eq!(summary.mismatches, 0);
        assert_eq!(summary.timing_regressions, 0);
    }

    #[test]
    fn test_summary_counts() {
        let baseline = vec![
            TraceSyscallEntry::simple(1, "read", 100),
            TraceSyscallEntry::simple(2, "write", 200),
            TraceSyscallEntry::simple(3, "close", 50),
        ];
        let actual = vec![
            TraceSyscallEntry::simple(1, "read", 100),
            TraceSyscallEntry::simple(99, "other", 200), // Different
            TraceSyscallEntry::simple(3, "close", 50),
        ];

        let result =
            compare_syscalls(&baseline, &actual, false).expect("comparison should succeed");
        assert_eq!(result.summary.total_compared, 3);
        assert_eq!(result.summary.matches, 2); // 2 match (index 0 and 2)
        assert_eq!(result.summary.mismatches, 1); // 1 different
    }
}
