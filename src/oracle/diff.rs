//! I/O diffing utilities for verification
//!
//! Provides flexible output comparison with various normalization options.

use std::fmt::Write;

use super::ExecutionResult;

/// Options for I/O comparison
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct DiffOptions {
    /// Normalize whitespace (collapse multiple spaces, trim lines)
    pub normalize_whitespace: bool,
    /// Ignore trailing whitespace on lines
    pub ignore_trailing_whitespace: bool,
    /// Ignore case when comparing
    pub ignore_case: bool,
    /// Tolerance for floating point comparisons
    pub float_tolerance: Option<f64>,
    /// Ignore stderr differences
    pub ignore_stderr: bool,
    /// Ignore exit code differences (only compare stdout)
    pub ignore_exit_code: bool,
}

impl Default for DiffOptions {
    fn default() -> Self {
        Self {
            normalize_whitespace: false,
            ignore_trailing_whitespace: true,
            ignore_case: false,
            float_tolerance: None,
            ignore_stderr: true,
            ignore_exit_code: false,
        }
    }
}

impl DiffOptions {
    /// Create strict comparison options
    #[must_use]
    pub fn strict() -> Self {
        Self {
            normalize_whitespace: false,
            ignore_trailing_whitespace: false,
            ignore_case: false,
            float_tolerance: None,
            ignore_stderr: false,
            ignore_exit_code: false,
        }
    }

    /// Create lenient comparison options
    #[must_use]
    pub fn lenient() -> Self {
        Self {
            normalize_whitespace: true,
            ignore_trailing_whitespace: true,
            ignore_case: false,
            float_tolerance: Some(1e-9),
            ignore_stderr: true,
            ignore_exit_code: true,
        }
    }

    /// Set float tolerance
    #[must_use]
    pub fn with_float_tolerance(mut self, tolerance: f64) -> Self {
        self.float_tolerance = Some(tolerance);
        self
    }
}

/// Result of a diff operation
#[derive(Debug, Clone)]
pub struct DiffResult {
    /// Whether the outputs match
    pub matches: bool,
    /// Differences found (if any)
    pub differences: Vec<Difference>,
}

/// A single difference between expected and actual output
#[derive(Debug, Clone)]
pub struct Difference {
    /// Line number (1-indexed)
    pub line: usize,
    /// Expected content
    pub expected: String,
    /// Actual content
    pub actual: String,
    /// Type of difference
    pub kind: DifferenceKind,
}

/// Type of difference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferenceKind {
    /// Line content differs
    ContentMismatch,
    /// Line missing in actual
    MissingLine,
    /// Extra line in actual
    ExtraLine,
    /// Exit code differs
    ExitCodeMismatch,
    /// Stderr differs
    StderrMismatch,
}

/// Compare two execution results
#[must_use]
pub fn diff_results(
    expected: &ExecutionResult,
    actual: &ExecutionResult,
    options: &DiffOptions,
) -> DiffResult {
    let mut differences = Vec::new();

    // Compare exit codes
    if !options.ignore_exit_code && expected.exit_code != actual.exit_code {
        differences.push(Difference {
            line: 0,
            expected: expected.exit_code.to_string(),
            actual: actual.exit_code.to_string(),
            kind: DifferenceKind::ExitCodeMismatch,
        });
    }

    // Compare stdout
    let stdout_diffs = diff_strings(&expected.stdout, &actual.stdout, options);
    differences.extend(stdout_diffs);

    // Compare stderr
    if !options.ignore_stderr {
        let stderr_diffs = diff_strings(&expected.stderr, &actual.stderr, options);
        for mut diff in stderr_diffs {
            diff.kind = DifferenceKind::StderrMismatch;
            differences.push(diff);
        }
    }

    DiffResult {
        matches: differences.is_empty(),
        differences,
    }
}

/// Compare two strings line by line
fn diff_strings(expected: &str, actual: &str, options: &DiffOptions) -> Vec<Difference> {
    let expected_lines: Vec<&str> = expected.lines().collect();
    let actual_lines: Vec<&str> = actual.lines().collect();

    let mut differences = Vec::new();

    let max_lines = expected_lines.len().max(actual_lines.len());

    for i in 0..max_lines {
        let exp_line = expected_lines.get(i);
        let act_line = actual_lines.get(i);

        match (exp_line, act_line) {
            (Some(exp), Some(act)) => {
                if !lines_equal(exp, act, options) {
                    differences.push(Difference {
                        line: i + 1,
                        expected: (*exp).to_string(),
                        actual: (*act).to_string(),
                        kind: DifferenceKind::ContentMismatch,
                    });
                }
            }
            (Some(exp), None) => {
                differences.push(Difference {
                    line: i + 1,
                    expected: (*exp).to_string(),
                    actual: String::new(),
                    kind: DifferenceKind::MissingLine,
                });
            }
            (None, Some(act)) => {
                differences.push(Difference {
                    line: i + 1,
                    expected: String::new(),
                    actual: (*act).to_string(),
                    kind: DifferenceKind::ExtraLine,
                });
            }
            (None, None) => {
                // This case cannot occur because i < max_lines ensures
                // at least one of the lines exists
            }
        }
    }

    differences
}

/// Check if two lines are equal according to options
fn lines_equal(expected: &str, actual: &str, options: &DiffOptions) -> bool {
    let mut exp = expected.to_string();
    let mut act = actual.to_string();

    // Apply normalizations
    if options.ignore_trailing_whitespace {
        exp = exp.trim_end().to_string();
        act = act.trim_end().to_string();
    }

    if options.normalize_whitespace {
        exp = normalize_whitespace(&exp);
        act = normalize_whitespace(&act);
    }

    if options.ignore_case {
        exp = exp.to_lowercase();
        act = act.to_lowercase();
    }

    // Try direct comparison first
    if exp == act {
        return true;
    }

    // Try float comparison if enabled
    if let Some(tolerance) = options.float_tolerance {
        if floats_equal(&exp, &act, tolerance) {
            return true;
        }
    }

    false
}

/// Normalize whitespace in a string
fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Compare two strings as floats with tolerance
fn floats_equal(a: &str, b: &str, tolerance: f64) -> bool {
    // Try to parse both as floats
    match (a.trim().parse::<f64>(), b.trim().parse::<f64>()) {
        (Ok(fa), Ok(fb)) => (fa - fb).abs() < tolerance,
        _ => false,
    }
}

/// Format a diff result for display
#[must_use]
pub fn format_diff(result: &DiffResult) -> String {
    if result.matches {
        return "Outputs match".to_string();
    }

    let mut output = String::new();
    let _ = writeln!(output, "Found {} difference(s):", result.differences.len());

    for diff in &result.differences {
        match diff.kind {
            DifferenceKind::ContentMismatch => {
                let _ = writeln!(
                    output,
                    "Line {}: expected '{}', got '{}'",
                    diff.line, diff.expected, diff.actual
                );
            }
            DifferenceKind::MissingLine => {
                let _ = writeln!(output, "Line {}: missing '{}'", diff.line, diff.expected);
            }
            DifferenceKind::ExtraLine => {
                let _ = writeln!(output, "Line {}: unexpected '{}'", diff.line, diff.actual);
            }
            DifferenceKind::ExitCodeMismatch => {
                let _ = writeln!(
                    output,
                    "Exit code: expected {}, got {}",
                    diff.expected, diff.actual
                );
            }
            DifferenceKind::StderrMismatch => {
                let _ = writeln!(
                    output,
                    "Stderr line {}: expected '{}', got '{}'",
                    diff.line, diff.expected, diff.actual
                );
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(stdout: &str, exit_code: i32) -> ExecutionResult {
        ExecutionResult {
            stdout: stdout.to_string(),
            stderr: String::new(),
            exit_code,
            duration_ms: 0,
        }
    }

    #[test]
    fn test_identical_outputs() {
        let expected = make_result("hello\nworld", 0);
        let actual = make_result("hello\nworld", 0);
        let options = DiffOptions::default();

        let result = diff_results(&expected, &actual, &options);
        assert!(result.matches);
        assert!(result.differences.is_empty());
    }

    #[test]
    fn test_different_outputs() {
        let expected = make_result("hello", 0);
        let actual = make_result("world", 0);
        let options = DiffOptions::default();

        let result = diff_results(&expected, &actual, &options);
        assert!(!result.matches);
        assert_eq!(result.differences.len(), 1);
        assert_eq!(result.differences[0].kind, DifferenceKind::ContentMismatch);
    }

    #[test]
    fn test_trailing_whitespace_ignored() {
        let expected = make_result("hello  ", 0);
        let actual = make_result("hello", 0);
        let options = DiffOptions::default(); // ignore_trailing_whitespace = true

        let result = diff_results(&expected, &actual, &options);
        assert!(result.matches);
    }

    #[test]
    fn test_trailing_whitespace_strict() {
        let expected = make_result("hello  ", 0);
        let actual = make_result("hello", 0);
        let options = DiffOptions::strict();

        let result = diff_results(&expected, &actual, &options);
        assert!(!result.matches);
    }

    #[test]
    fn test_float_tolerance() {
        let expected = make_result("3.14159265", 0);
        let actual = make_result("3.14159266", 0);
        let options = DiffOptions::default().with_float_tolerance(1e-6);

        let result = diff_results(&expected, &actual, &options);
        assert!(result.matches);
    }

    #[test]
    fn test_float_no_tolerance() {
        let expected = make_result("3.14159265", 0);
        let actual = make_result("3.14159266", 0);
        let options = DiffOptions::default(); // no float_tolerance

        let result = diff_results(&expected, &actual, &options);
        assert!(!result.matches);
    }

    #[test]
    fn test_missing_line() {
        let expected = make_result("line1\nline2", 0);
        let actual = make_result("line1", 0);
        let options = DiffOptions::default();

        let result = diff_results(&expected, &actual, &options);
        assert!(!result.matches);
        assert!(result
            .differences
            .iter()
            .any(|d| d.kind == DifferenceKind::MissingLine));
    }

    #[test]
    fn test_extra_line() {
        let expected = make_result("line1", 0);
        let actual = make_result("line1\nline2", 0);
        let options = DiffOptions::default();

        let result = diff_results(&expected, &actual, &options);
        assert!(!result.matches);
        assert!(result
            .differences
            .iter()
            .any(|d| d.kind == DifferenceKind::ExtraLine));
    }

    #[test]
    fn test_exit_code_mismatch() {
        let expected = make_result("output", 0);
        let actual = make_result("output", 1);
        let options = DiffOptions::default();

        let result = diff_results(&expected, &actual, &options);
        assert!(!result.matches);
        assert!(result
            .differences
            .iter()
            .any(|d| d.kind == DifferenceKind::ExitCodeMismatch));
    }

    #[test]
    fn test_exit_code_ignored() {
        let expected = make_result("output", 0);
        let actual = make_result("output", 1);
        let options = DiffOptions::lenient();

        let result = diff_results(&expected, &actual, &options);
        assert!(result.matches);
    }

    #[test]
    fn test_normalize_whitespace() {
        let expected = make_result("hello   world", 0);
        let actual = make_result("hello world", 0);
        let mut options = DiffOptions::default();
        options.normalize_whitespace = true;

        let result = diff_results(&expected, &actual, &options);
        assert!(result.matches);
    }

    #[test]
    fn test_format_diff() {
        let expected = make_result("hello", 0);
        let actual = make_result("world", 0);
        let options = DiffOptions::default();

        let result = diff_results(&expected, &actual, &options);
        let formatted = format_diff(&result);

        assert!(formatted.contains("difference"));
        assert!(formatted.contains("hello"));
        assert!(formatted.contains("world"));
    }

    #[test]
    fn test_format_diff_match() {
        let expected = make_result("hello", 0);
        let actual = make_result("hello", 0);
        let options = DiffOptions::default();

        let result = diff_results(&expected, &actual, &options);
        let formatted = format_diff(&result);

        assert!(formatted.contains("match"));
    }

    #[test]
    fn test_format_diff_missing_line() {
        let expected = make_result("line1\nline2", 0);
        let actual = make_result("line1", 0);
        let options = DiffOptions::default();

        let result = diff_results(&expected, &actual, &options);
        let formatted = format_diff(&result);

        assert!(formatted.contains("missing"));
    }

    #[test]
    fn test_format_diff_extra_line() {
        let expected = make_result("line1", 0);
        let actual = make_result("line1\nextra", 0);
        let options = DiffOptions::default();

        let result = diff_results(&expected, &actual, &options);
        let formatted = format_diff(&result);

        assert!(formatted.contains("unexpected"));
    }

    #[test]
    fn test_format_diff_exit_code() {
        let expected = make_result("output", 0);
        let actual = make_result("output", 1);
        let options = DiffOptions::strict();

        let result = diff_results(&expected, &actual, &options);
        let formatted = format_diff(&result);

        assert!(formatted.contains("Exit code"));
    }

    #[test]
    fn test_stderr_mismatch() {
        let expected = ExecutionResult {
            stdout: "out".to_string(),
            stderr: "err1".to_string(),
            exit_code: 0,
            duration_ms: 0,
        };
        let actual = ExecutionResult {
            stdout: "out".to_string(),
            stderr: "err2".to_string(),
            exit_code: 0,
            duration_ms: 0,
        };
        let options = DiffOptions::strict();

        let result = diff_results(&expected, &actual, &options);
        assert!(!result.matches);
        assert!(result
            .differences
            .iter()
            .any(|d| d.kind == DifferenceKind::StderrMismatch));
    }

    #[test]
    fn test_format_diff_stderr() {
        let expected = ExecutionResult {
            stdout: "out".to_string(),
            stderr: "err1".to_string(),
            exit_code: 0,
            duration_ms: 0,
        };
        let actual = ExecutionResult {
            stdout: "out".to_string(),
            stderr: "err2".to_string(),
            exit_code: 0,
            duration_ms: 0,
        };
        let options = DiffOptions::strict();

        let result = diff_results(&expected, &actual, &options);
        let formatted = format_diff(&result);

        assert!(formatted.contains("Stderr"));
    }

    #[test]
    fn test_ignore_case() {
        let expected = make_result("HELLO", 0);
        let actual = make_result("hello", 0);
        let mut options = DiffOptions::default();
        options.ignore_case = true;

        let result = diff_results(&expected, &actual, &options);
        assert!(result.matches);
    }

    #[test]
    fn test_ignore_case_false() {
        let expected = make_result("HELLO", 0);
        let actual = make_result("hello", 0);
        let options = DiffOptions::default(); // ignore_case = false

        let result = diff_results(&expected, &actual, &options);
        assert!(!result.matches);
    }

    #[test]
    fn test_float_tolerance_non_float() {
        let expected = make_result("not a float", 0);
        let actual = make_result("also not", 0);
        let options = DiffOptions::default().with_float_tolerance(1e-6);

        let result = diff_results(&expected, &actual, &options);
        assert!(!result.matches); // Can't parse as floats, so mismatch
    }

    #[test]
    fn test_diff_options_debug() {
        let options = DiffOptions::default();
        let debug = format!("{:?}", options);
        assert!(debug.contains("DiffOptions"));
    }

    #[test]
    fn test_diff_options_clone() {
        let options = DiffOptions::lenient();
        let cloned = options.clone();
        assert_eq!(cloned.normalize_whitespace, options.normalize_whitespace);
    }

    #[test]
    fn test_diff_result_debug() {
        let result = DiffResult {
            matches: true,
            differences: vec![],
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("DiffResult"));
    }

    #[test]
    fn test_diff_result_clone() {
        let result = DiffResult {
            matches: false,
            differences: vec![Difference {
                line: 1,
                expected: "a".to_string(),
                actual: "b".to_string(),
                kind: DifferenceKind::ContentMismatch,
            }],
        };
        let cloned = result.clone();
        assert_eq!(cloned.matches, result.matches);
    }

    #[test]
    fn test_difference_debug() {
        let diff = Difference {
            line: 1,
            expected: "a".to_string(),
            actual: "b".to_string(),
            kind: DifferenceKind::ContentMismatch,
        };
        let debug = format!("{:?}", diff);
        assert!(debug.contains("Difference"));
    }

    #[test]
    fn test_difference_clone() {
        let diff = Difference {
            line: 1,
            expected: "a".to_string(),
            actual: "b".to_string(),
            kind: DifferenceKind::ContentMismatch,
        };
        let cloned = diff.clone();
        assert_eq!(cloned.line, diff.line);
    }

    #[test]
    fn test_difference_kind_debug() {
        let kinds = [
            DifferenceKind::ContentMismatch,
            DifferenceKind::MissingLine,
            DifferenceKind::ExtraLine,
            DifferenceKind::ExitCodeMismatch,
            DifferenceKind::StderrMismatch,
        ];
        for kind in &kinds {
            let debug = format!("{:?}", kind);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_difference_kind_copy() {
        let kind = DifferenceKind::ContentMismatch;
        let copied = kind;
        assert_eq!(copied, DifferenceKind::ContentMismatch);
    }
}
