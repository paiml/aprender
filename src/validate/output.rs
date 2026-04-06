//! Output formatters for validation results
//!
//! Per specification Section 3.3: Output Formats

use super::comparison::{ComparisonResult, MismatchType};
use super::config::ValidationOutputFormat;
use serde::Serialize;

/// Format validation result according to specified format
pub fn format_result(result: &ComparisonResult, format: ValidationOutputFormat) -> String {
    match format {
        ValidationOutputFormat::Text => format_text_report(result),
        ValidationOutputFormat::Json => format_json_report(result),
        ValidationOutputFormat::JUnit => format_junit_report(result),
    }
}

/// JSON output structure per specification Section 3.3
#[derive(Debug, Serialize)]
pub struct JsonValidationResult {
    /// Overall validation status
    pub status: String,
    /// Total syscalls compared
    pub total_compared: usize,
    /// Number of matches
    pub matches: usize,
    /// Number of mismatches
    pub mismatches: usize,
    /// Number of timing regressions
    pub timing_regressions: usize,
    /// Detailed syscall mismatches
    pub syscall_details: Vec<JsonSyscallMismatch>,
    /// Detailed timing regressions
    pub timing_details: Vec<JsonTimingRegression>,
}

#[derive(Debug, Serialize)]
pub struct JsonSyscallMismatch {
    pub index: usize,
    pub expected: String,
    pub found: String,
    pub mismatch_type: String,
}

#[derive(Debug, Serialize)]
pub struct JsonTimingRegression {
    pub syscall: String,
    pub baseline_ms: f64,
    pub actual_ms: f64,
    pub delta_percent: f64,
}

/// Format as human-readable text
pub fn format_text_report(result: &ComparisonResult) -> String {
    let mut output = String::new();

    // Header
    output.push_str("=== Validation Report ===\n\n");

    // Status
    let status = if result.passed { "PASSED" } else { "FAILED" };
    output.push_str(&format!("Status: {status}\n\n"));

    // Summary
    output.push_str("Summary:\n");
    output.push_str(&format!("  Total syscalls compared: {}\n", result.summary.total_compared));
    output.push_str(&format!("  Matches: {}\n", result.summary.matches));
    output.push_str(&format!("  Mismatches: {}\n", result.summary.mismatches));
    output.push_str(&format!("  Timing regressions: {}\n", result.summary.timing_regressions));
    output.push('\n');

    // Syscall mismatches
    if !result.syscall_mismatches.is_empty() {
        output.push_str("Syscall Mismatches:\n");
        for mismatch in &result.syscall_mismatches {
            let mismatch_type = match mismatch.mismatch_type {
                MismatchType::Different => "DIFFERENT",
                MismatchType::Extra => "EXTRA",
                MismatchType::Missing => "MISSING",
            };
            output.push_str(&format!(
                "  [{}] Index {}: expected '{}', found '{}'\n",
                mismatch_type, mismatch.index, mismatch.expected, mismatch.found
            ));
        }
        output.push('\n');
    }

    // Timing regressions
    if !result.timing_regressions.is_empty() {
        output.push_str("Timing Regressions:\n");
        for regression in &result.timing_regressions {
            output.push_str(&format!(
                "  {}: {:.2}ms -> {:.2}ms ({:+.1}%)\n",
                regression.syscall,
                regression.baseline_ms,
                regression.actual_ms,
                regression.delta_percent
            ));
        }
        output.push('\n');
    }

    output
}

/// Format as JSON
pub fn format_json_report(result: &ComparisonResult) -> String {
    let json_result = JsonValidationResult {
        status: if result.passed { "passed".to_string() } else { "failed".to_string() },
        total_compared: result.summary.total_compared,
        matches: result.summary.matches,
        mismatches: result.summary.mismatches,
        timing_regressions: result.summary.timing_regressions,
        syscall_details: result
            .syscall_mismatches
            .iter()
            .map(|m| JsonSyscallMismatch {
                index: m.index,
                expected: m.expected.clone(),
                found: m.found.clone(),
                mismatch_type: match m.mismatch_type {
                    MismatchType::Different => "different".to_string(),
                    MismatchType::Extra => "extra".to_string(),
                    MismatchType::Missing => "missing".to_string(),
                },
            })
            .collect(),
        timing_details: result
            .timing_regressions
            .iter()
            .map(|t| JsonTimingRegression {
                syscall: t.syscall.clone(),
                baseline_ms: t.baseline_ms,
                actual_ms: t.actual_ms,
                delta_percent: t.delta_percent,
            })
            .collect(),
    };

    serde_json::to_string_pretty(&json_result).unwrap_or_else(|_| "{}".to_string())
}

/// Format as `JUnit` XML for CI systems
pub fn format_junit_report(result: &ComparisonResult) -> String {
    let mut output = String::new();

    output.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");

    let test_count = 1 + result.syscall_mismatches.len() + result.timing_regressions.len();
    let failure_count = result.syscall_mismatches.len() + result.timing_regressions.len();

    output.push_str(&format!(
        "<testsuite name=\"renacer-validate\" tests=\"{test_count}\" failures=\"{failure_count}\" errors=\"0\">\n"
    ));

    // Overall test case
    output.push_str("  <testcase name=\"validation-overall\" classname=\"renacer.validate\"");
    if result.passed {
        output.push_str("/>\n");
    } else {
        output.push_str(">\n");
        output.push_str(&format!(
            "    <failure message=\"Validation failed: {} mismatches, {} timing regressions\"/>\n",
            result.syscall_mismatches.len(),
            result.timing_regressions.len()
        ));
        output.push_str("  </testcase>\n");
    }

    // Individual syscall mismatch test cases
    for mismatch in &result.syscall_mismatches {
        output.push_str(&format!(
            "  <testcase name=\"syscall-{index}\" classname=\"renacer.validate.syscalls\">\n",
            index = mismatch.index
        ));
        output.push_str(&format!(
            "    <failure message=\"Expected '{}', found '{}' ({})\"/>\n",
            xml_escape(&mismatch.expected),
            xml_escape(&mismatch.found),
            match mismatch.mismatch_type {
                MismatchType::Different => "different",
                MismatchType::Extra => "extra",
                MismatchType::Missing => "missing",
            }
        ));
        output.push_str("  </testcase>\n");
    }

    // Individual timing regression test cases
    for regression in &result.timing_regressions {
        output.push_str(&format!(
            "  <testcase name=\"timing-{}\" classname=\"renacer.validate.timing\">\n",
            xml_escape(&regression.syscall)
        ));
        output.push_str(&format!(
            "    <failure message=\"Timing regression: {:.2}ms -> {:.2}ms ({:+.1}%)\"/>\n",
            regression.baseline_ms, regression.actual_ms, regression.delta_percent
        ));
        output.push_str("  </testcase>\n");
    }

    output.push_str("</testsuite>\n");

    output
}

/// Escape XML special characters
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::comparison::{ComparisonSummary, SyscallMismatch, TimingRegression};

    #[test]
    fn test_text_format_passed() {
        let result = ComparisonResult::passed();
        let text = format_text_report(&result);
        assert!(text.contains("Status: PASSED"));
    }

    #[test]
    fn test_text_format_failed_with_mismatches() {
        let result = ComparisonResult {
            passed: false,
            syscall_mismatches: vec![
                SyscallMismatch {
                    index: 0,
                    expected: "read".to_string(),
                    found: "write".to_string(),
                    mismatch_type: MismatchType::Different,
                },
                SyscallMismatch {
                    index: 1,
                    expected: "(none)".to_string(),
                    found: "close".to_string(),
                    mismatch_type: MismatchType::Extra,
                },
                SyscallMismatch {
                    index: 2,
                    expected: "open".to_string(),
                    found: "(none)".to_string(),
                    mismatch_type: MismatchType::Missing,
                },
            ],
            timing_regressions: vec![TimingRegression {
                syscall: "read".to_string(),
                baseline_ms: 0.1,
                actual_ms: 0.15,
                delta_percent: 50.0,
            }],
            summary: ComparisonSummary {
                total_compared: 3,
                matches: 0,
                mismatches: 3,
                timing_regressions: 1,
            },
        };
        let text = format_text_report(&result);
        assert!(text.contains("Status: FAILED"));
        assert!(text.contains("[DIFFERENT]"));
        assert!(text.contains("[EXTRA]"));
        assert!(text.contains("[MISSING]"));
        assert!(text.contains("Timing Regressions:"));
        assert!(text.contains("read: 0.10ms -> 0.15ms (+50.0%)"));
    }

    #[test]
    fn test_json_format_structure() {
        let result = ComparisonResult::passed();
        let json = format_json_report(&result);
        assert!(json.contains("\"status\": \"passed\""));
    }

    #[test]
    fn test_json_format_with_details() {
        let result = ComparisonResult {
            passed: false,
            syscall_mismatches: vec![SyscallMismatch {
                index: 5,
                expected: "openat".to_string(),
                found: "close".to_string(),
                mismatch_type: MismatchType::Different,
            }],
            timing_regressions: vec![TimingRegression {
                syscall: "write".to_string(),
                baseline_ms: 0.5,
                actual_ms: 1.0,
                delta_percent: 100.0,
            }],
            summary: ComparisonSummary {
                total_compared: 10,
                matches: 9,
                mismatches: 1,
                timing_regressions: 1,
            },
        };
        let json = format_json_report(&result);
        assert!(json.contains("\"status\": \"failed\""));
        assert!(json.contains("\"mismatch_type\": \"different\""));
        assert!(json.contains("\"syscall\": \"write\""));
        assert!(json.contains("\"delta_percent\": 100.0"));
    }

    #[test]
    fn test_json_format_extra_and_missing_types() {
        let result = ComparisonResult {
            passed: false,
            syscall_mismatches: vec![
                SyscallMismatch {
                    index: 0,
                    expected: "(none)".to_string(),
                    found: "mmap".to_string(),
                    mismatch_type: MismatchType::Extra,
                },
                SyscallMismatch {
                    index: 1,
                    expected: "munmap".to_string(),
                    found: "(none)".to_string(),
                    mismatch_type: MismatchType::Missing,
                },
            ],
            timing_regressions: vec![],
            summary: ComparisonSummary::default(),
        };
        let json = format_json_report(&result);
        assert!(json.contains("\"mismatch_type\": \"extra\""));
        assert!(json.contains("\"mismatch_type\": \"missing\""));
    }

    #[test]
    fn test_junit_format_valid_xml() {
        let result = ComparisonResult::passed();
        let xml = format_junit_report(&result);
        assert!(xml.starts_with("<?xml version=\"1.0\""));
        assert!(xml.contains("<testsuite"));
        assert!(xml.contains("</testsuite>"));
    }

    #[test]
    fn test_junit_format_with_failures() {
        let result = ComparisonResult {
            passed: false,
            syscall_mismatches: vec![
                SyscallMismatch {
                    index: 0,
                    expected: "read".to_string(),
                    found: "write".to_string(),
                    mismatch_type: MismatchType::Different,
                },
                SyscallMismatch {
                    index: 1,
                    expected: "(none)".to_string(),
                    found: "close".to_string(),
                    mismatch_type: MismatchType::Extra,
                },
                SyscallMismatch {
                    index: 2,
                    expected: "open".to_string(),
                    found: "(none)".to_string(),
                    mismatch_type: MismatchType::Missing,
                },
            ],
            timing_regressions: vec![TimingRegression {
                syscall: "read".to_string(),
                baseline_ms: 0.1,
                actual_ms: 0.2,
                delta_percent: 100.0,
            }],
            summary: ComparisonSummary {
                total_compared: 3,
                matches: 0,
                mismatches: 3,
                timing_regressions: 1,
            },
        };
        let xml = format_junit_report(&result);
        assert!(xml.contains("failures=\"4\"")); // 3 syscall + 1 timing
        assert!(xml.contains("<failure message=\"Validation failed"));
        assert!(xml.contains("syscall-0"));
        assert!(xml.contains("timing-read"));
        assert!(xml.contains("(different)"));
        assert!(xml.contains("(extra)"));
        assert!(xml.contains("(missing)"));
    }

    #[test]
    fn test_xml_escape() {
        // Test XML special character escaping
        assert_eq!(xml_escape("a & b"), "a &amp; b");
        assert_eq!(xml_escape("<tag>"), "&lt;tag&gt;");
        assert_eq!(xml_escape("\"quoted\""), "&quot;quoted&quot;");
        assert_eq!(xml_escape("it's"), "it&apos;s");
    }

    #[test]
    fn test_format_result_dispatch() {
        let result = ComparisonResult::passed();

        let text = format_result(&result, ValidationOutputFormat::Text);
        assert!(text.contains("Status: PASSED"));

        let json = format_result(&result, ValidationOutputFormat::Json);
        assert!(json.contains("\"status\": \"passed\""));

        let junit = format_result(&result, ValidationOutputFormat::JUnit);
        assert!(junit.contains("<testsuite"));
    }
}
