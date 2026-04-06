//! Tarantula Spectrum-Based Fault Localization (PROBAR-WASM-003)
//!
//! Implements the Tarantula algorithm for identifying suspicious lines of code
//! based on test coverage data.
//!
//! ## Algorithm
//!
//! The suspiciousness score for a line is calculated as:
//!
//! ```text
//! suspiciousness = (failed(line) / total_failed) /
//!                  ((failed(line) / total_failed) + (passed(line) / total_passed))
//! ```
//!
//! Lines with higher suspiciousness scores are more likely to contain bugs.
//!
//! ## Integration
//!
//! This module consumes `.lcov` or `profraw` coverage artifacts to calculate
//! suspiciousness scores when proptest or other property-based tests fail.

use std::collections::HashMap;
use std::path::Path;

/// Coverage data for a single line
#[derive(Debug, Default, Clone)]
pub struct LineCoverage {
    /// Number of times this line was executed in passing tests
    pub passed_executions: usize,
    /// Number of times this line was executed in failing tests
    pub failed_executions: usize,
}

/// Tarantula suspiciousness report for a file
#[derive(Debug, Default)]
pub struct TarantulaReport {
    /// File path
    pub file: String,
    /// Line suspiciousness scores (line number -> score)
    pub line_scores: HashMap<usize, f64>,
    /// Total passing tests
    pub total_passed: usize,
    /// Total failing tests
    pub total_failed: usize,
}

impl TarantulaReport {
    /// Get the top N most suspicious lines
    #[must_use]
    pub fn top_suspicious(&self, n: usize) -> Vec<(usize, f64)> {
        let mut scores: Vec<_> = self.line_scores.iter().map(|(&l, &s)| (l, s)).collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(n);
        scores
    }

    /// Format as a hotspot report
    #[must_use]
    pub fn format_hotspot_report(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("🎯 Tarantula Hotspot Report: {}\n", self.file));
        output.push_str(&format!(
            "   Tests: {} passed, {} failed\n\n",
            self.total_passed, self.total_failed
        ));

        output.push_str("   Line  | Suspiciousness | Status\n");
        output.push_str("   ------|----------------|--------\n");

        for (line, score) in self.top_suspicious(10) {
            let status = if score > 0.8 {
                "🔴 HIGH"
            } else if score > 0.5 {
                "🟡 MEDIUM"
            } else {
                "🟢 LOW"
            };
            output.push_str(&format!("   {:5} | {:14.3} | {}\n", line, score, status));
        }

        output
    }
}

/// Tarantula fault localization engine
#[derive(Debug, Default)]
pub struct TarantulaEngine {
    /// Coverage data per file per line
    coverage: HashMap<String, HashMap<usize, LineCoverage>>,
    /// Total passing tests recorded
    total_passed: usize,
    /// Total failing tests recorded
    total_failed: usize,
}

impl TarantulaEngine {
    /// Create a new Tarantula engine
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a test execution
    ///
    /// # Arguments
    /// * `file` - Source file path
    /// * `line` - Line number
    /// * `passed` - Whether the test passed
    pub fn record_execution(&mut self, file: &str, line: usize, passed: bool) {
        let file_coverage = self.coverage.entry(file.to_string()).or_default();
        let line_coverage = file_coverage.entry(line).or_default();

        if passed {
            line_coverage.passed_executions += 1;
        } else {
            line_coverage.failed_executions += 1;
        }
    }

    /// Record a complete test run
    pub fn record_test_run(&mut self, passed: bool) {
        if passed {
            self.total_passed += 1;
        } else {
            self.total_failed += 1;
        }
    }

    /// Calculate suspiciousness score for a line
    ///
    /// Returns a value between 0.0 (not suspicious) and 1.0 (highly suspicious).
    fn calculate_suspiciousness(&self, line: &LineCoverage) -> f64 {
        if self.total_failed == 0 || self.total_passed == 0 {
            return 0.0;
        }

        let failed_ratio = line.failed_executions as f64 / self.total_failed as f64;
        let passed_ratio = line.passed_executions as f64 / self.total_passed as f64;

        if failed_ratio + passed_ratio == 0.0 {
            return 0.0;
        }

        failed_ratio / (failed_ratio + passed_ratio)
    }

    /// Generate report for a specific file
    #[must_use]
    pub fn report_for_file(&self, file: &str) -> Option<TarantulaReport> {
        let file_coverage = self.coverage.get(file)?;

        let mut line_scores = HashMap::new();
        for (&line, coverage) in file_coverage {
            let score = self.calculate_suspiciousness(coverage);
            if score > 0.0 {
                line_scores.insert(line, score);
            }
        }

        Some(TarantulaReport {
            file: file.to_string(),
            line_scores,
            total_passed: self.total_passed,
            total_failed: self.total_failed,
        })
    }

    /// Generate reports for all files with suspicious lines
    #[must_use]
    pub fn generate_all_reports(&self) -> Vec<TarantulaReport> {
        self.coverage
            .keys()
            .filter_map(|file| self.report_for_file(file))
            .filter(|r| !r.line_scores.is_empty())
            .collect()
    }

    /// Parse LCOV coverage file
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed
    pub fn parse_lcov(&mut self, path: &Path, passed: bool) -> Result<(), String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read LCOV: {e}"))?;

        let mut current_file: Option<String> = None;

        for line in content.lines() {
            if let Some(file) = line.strip_prefix("SF:") {
                current_file = Some(file.to_string());
            } else if let Some(da) = line.strip_prefix("DA:") {
                if let Some(ref file) = current_file {
                    // Format: DA:line,execution_count
                    let parts: Vec<_> = da.split(',').collect();
                    if parts.len() >= 2 {
                        if let Ok(line_num) = parts[0].parse::<usize>() {
                            if let Ok(exec_count) = parts[1].parse::<usize>() {
                                if exec_count > 0 {
                                    self.record_execution(file, line_num, passed);
                                }
                            }
                        }
                    }
                }
            } else if line == "end_of_record" {
                current_file = None;
            }
        }

        self.record_test_run(passed);
        Ok(())
    }

    /// Filter lines involving Rc or RefCell state updates
    ///
    /// Returns only lines that contain Rc/RefCell patterns, which are
    /// relevant for WASM state sync debugging.
    pub fn filter_state_related<'a>(&self, source: &'a str) -> HashMap<usize, &'a str> {
        let patterns = ["Rc::", "RefCell", "borrow", "borrow_mut", "state"];
        let mut result = HashMap::new();

        for (idx, line) in source.lines().enumerate() {
            let line_num = idx + 1;
            if patterns.iter().any(|&p| line.contains(p)) {
                result.insert(line_num, line);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suspiciousness_calculation() {
        let mut engine = TarantulaEngine::new();

        // Line executed in both passing and failing tests
        engine.record_execution("test.rs", 10, true);
        engine.record_execution("test.rs", 10, true);
        engine.record_execution("test.rs", 10, false);

        // Line executed only in failing tests
        engine.record_execution("test.rs", 20, false);

        // Record test runs
        engine.record_test_run(true);
        engine.record_test_run(true);
        engine.record_test_run(false);

        let report = engine.report_for_file("test.rs").unwrap();

        // Line 20 (only failed) should be more suspicious than line 10 (mixed)
        let score_10 = report.line_scores.get(&10).copied().unwrap_or(0.0);
        let score_20 = report.line_scores.get(&20).copied().unwrap_or(0.0);

        assert!(
            score_20 > score_10,
            "Line only in fails should be more suspicious"
        );
        assert!(
            score_20 > 0.5,
            "Line only in fails should be highly suspicious"
        );
    }

    #[test]
    fn test_top_suspicious() {
        let mut report = TarantulaReport {
            file: "test.rs".to_string(),
            line_scores: HashMap::new(),
            total_passed: 10,
            total_failed: 5,
        };

        report.line_scores.insert(10, 0.3);
        report.line_scores.insert(20, 0.9);
        report.line_scores.insert(30, 0.6);

        let top = report.top_suspicious(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 20); // Most suspicious first
        assert_eq!(top[1].0, 30);
    }

    #[test]
    fn test_filter_state_related() {
        let engine = TarantulaEngine::new();

        let source = r#"
let x = 5;
let state = Rc::new(RefCell::new(0));
*state.borrow_mut() = 10;
println!("hello");
"#;

        let filtered = engine.filter_state_related(source);

        assert!(filtered.contains_key(&3)); // Rc::new line
        assert!(filtered.contains_key(&4)); // borrow_mut line
        assert!(!filtered.contains_key(&2)); // let x = 5
        assert!(!filtered.contains_key(&5)); // println
    }

    #[test]
    fn test_format_hotspot_report_high_suspiciousness() {
        let mut report = TarantulaReport {
            file: "suspicious.rs".to_string(),
            line_scores: HashMap::new(),
            total_passed: 5,
            total_failed: 3,
        };

        // Add HIGH suspiciousness lines (> 0.8)
        report.line_scores.insert(10, 0.95);
        report.line_scores.insert(20, 0.85);

        let output = report.format_hotspot_report();

        assert!(output.contains("🎯 Tarantula Hotspot Report: suspicious.rs"));
        assert!(output.contains("5 passed, 3 failed"));
        assert!(output.contains("🔴 HIGH"));
    }

    #[test]
    fn test_format_hotspot_report_medium_suspiciousness() {
        let mut report = TarantulaReport {
            file: "medium.rs".to_string(),
            line_scores: HashMap::new(),
            total_passed: 10,
            total_failed: 2,
        };

        // Add MEDIUM suspiciousness lines (> 0.5 but <= 0.8)
        report.line_scores.insert(15, 0.65);
        report.line_scores.insert(25, 0.55);

        let output = report.format_hotspot_report();

        assert!(output.contains("🟡 MEDIUM"));
    }

    #[test]
    fn test_format_hotspot_report_low_suspiciousness() {
        let mut report = TarantulaReport {
            file: "low.rs".to_string(),
            line_scores: HashMap::new(),
            total_passed: 20,
            total_failed: 1,
        };

        // Add LOW suspiciousness lines (<= 0.5)
        report.line_scores.insert(30, 0.3);
        report.line_scores.insert(40, 0.1);

        let output = report.format_hotspot_report();

        assert!(output.contains("🟢 LOW"));
    }

    #[test]
    fn test_format_hotspot_report_all_levels() {
        let mut report = TarantulaReport {
            file: "mixed.rs".to_string(),
            line_scores: HashMap::new(),
            total_passed: 8,
            total_failed: 4,
        };

        // Mix of all suspiciousness levels
        report.line_scores.insert(100, 0.95); // HIGH
        report.line_scores.insert(200, 0.65); // MEDIUM
        report.line_scores.insert(300, 0.25); // LOW

        let output = report.format_hotspot_report();

        assert!(output.contains("🔴 HIGH"));
        assert!(output.contains("🟡 MEDIUM"));
        assert!(output.contains("🟢 LOW"));
        assert!(output.contains("Line  | Suspiciousness | Status"));
    }

    #[test]
    fn test_calculate_suspiciousness_no_failed_tests() {
        let mut engine = TarantulaEngine::new();

        // Only passing tests
        engine.record_execution("test.rs", 10, true);
        engine.record_test_run(true);
        engine.record_test_run(true);

        // With no failed tests, suspiciousness should be 0
        let report = engine.report_for_file("test.rs");
        assert!(report.is_none() || report.unwrap().line_scores.is_empty());
    }

    #[test]
    fn test_calculate_suspiciousness_no_passed_tests() {
        let mut engine = TarantulaEngine::new();

        // Only failing tests
        engine.record_execution("test.rs", 10, false);
        engine.record_test_run(false);
        engine.record_test_run(false);

        // With no passed tests, suspiciousness should be 0
        let report = engine.report_for_file("test.rs");
        assert!(report.is_none() || report.unwrap().line_scores.is_empty());
    }

    #[test]
    fn test_calculate_suspiciousness_zero_ratio_sum() {
        let mut engine = TarantulaEngine::new();

        // Record tests but no executions for specific lines
        engine.record_test_run(true);
        engine.record_test_run(false);

        // Line with no executions should have 0 suspiciousness
        let line = LineCoverage {
            passed_executions: 0,
            failed_executions: 0,
        };
        let score = engine.calculate_suspiciousness(&line);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_parse_lcov_valid_content() {
        use std::io::Write;

        let mut engine = TarantulaEngine::new();

        // Create temp LCOV file
        let lcov_content = r#"SF:src/main.rs
DA:1,5
DA:2,10
DA:3,0
end_of_record
SF:src/lib.rs
DA:10,3
DA:20,7
end_of_record
"#;

        let temp_dir = std::env::temp_dir();
        let lcov_path = temp_dir.join("test_tarantula.lcov");
        let mut file = std::fs::File::create(&lcov_path).unwrap();
        file.write_all(lcov_content.as_bytes()).unwrap();

        // Parse as passing test
        let result = engine.parse_lcov(&lcov_path, true);
        assert!(result.is_ok());

        // Parse again as failing test
        let result = engine.parse_lcov(&lcov_path, false);
        assert!(result.is_ok());

        // Check that files were recorded
        let report_main = engine.report_for_file("src/main.rs");
        let report_lib = engine.report_for_file("src/lib.rs");

        assert!(report_main.is_some());
        assert!(report_lib.is_some());

        // Clean up
        let _ = std::fs::remove_file(&lcov_path);
    }

    #[test]
    fn test_parse_lcov_nonexistent_file() {
        let mut engine = TarantulaEngine::new();

        let result = engine.parse_lcov(Path::new("/nonexistent/path.lcov"), true);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read LCOV"));
    }

    #[test]
    fn test_parse_lcov_malformed_da_lines() {
        use std::io::Write;

        let mut engine = TarantulaEngine::new();

        // LCOV with malformed DA lines
        let lcov_content = r#"SF:src/test.rs
DA:invalid,5
DA:1,invalid
DA:,
DA:1
end_of_record
"#;

        let temp_dir = std::env::temp_dir();
        let lcov_path = temp_dir.join("test_malformed.lcov");
        let mut file = std::fs::File::create(&lcov_path).unwrap();
        file.write_all(lcov_content.as_bytes()).unwrap();

        // Should not panic, just skip malformed lines
        let result = engine.parse_lcov(&lcov_path, true);
        assert!(result.is_ok());

        let _ = std::fs::remove_file(&lcov_path);
    }

    #[test]
    fn test_generate_all_reports() {
        let mut engine = TarantulaEngine::new();

        // Add coverage for multiple files
        engine.record_execution("file1.rs", 10, true);
        engine.record_execution("file1.rs", 10, false);
        engine.record_execution("file2.rs", 20, false);
        engine.record_execution("file3.rs", 30, true); // Only passing

        engine.record_test_run(true);
        engine.record_test_run(false);

        let reports = engine.generate_all_reports();

        // Should have reports for files with suspicious lines
        // file3.rs has only passing executions so score would be 0
        assert!(!reports.is_empty());

        // Verify file names are in reports
        let file_names: Vec<_> = reports.iter().map(|r| r.file.as_str()).collect();
        assert!(file_names.contains(&"file1.rs"));
        assert!(file_names.contains(&"file2.rs"));
    }

    #[test]
    fn test_line_coverage_default() {
        let coverage = LineCoverage::default();
        assert_eq!(coverage.passed_executions, 0);
        assert_eq!(coverage.failed_executions, 0);
    }

    #[test]
    fn test_tarantula_report_default() {
        let report = TarantulaReport::default();
        assert!(report.file.is_empty());
        assert!(report.line_scores.is_empty());
        assert_eq!(report.total_passed, 0);
        assert_eq!(report.total_failed, 0);
    }

    #[test]
    fn test_top_suspicious_empty_scores() {
        let report = TarantulaReport::default();
        let top = report.top_suspicious(5);
        assert!(top.is_empty());
    }

    #[test]
    fn test_top_suspicious_fewer_than_n() {
        let mut report = TarantulaReport {
            file: "test.rs".to_string(),
            line_scores: HashMap::new(),
            total_passed: 5,
            total_failed: 2,
        };

        report.line_scores.insert(1, 0.5);
        report.line_scores.insert(2, 0.8);

        // Request more than available
        let top = report.top_suspicious(10);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_report_for_nonexistent_file() {
        let engine = TarantulaEngine::new();
        assert!(engine.report_for_file("nonexistent.rs").is_none());
    }

    #[test]
    fn test_format_hotspot_report_line_formatting() {
        let mut report = TarantulaReport {
            file: "format_test.rs".to_string(),
            line_scores: HashMap::new(),
            total_passed: 3,
            total_failed: 2,
        };

        // Line number 12345 to test formatting width
        report.line_scores.insert(12345, 0.567);

        let output = report.format_hotspot_report();

        // Verify header line
        assert!(output.contains("Line  | Suspiciousness | Status"));
        assert!(output.contains("------|----------------|--------"));
        // Verify score is formatted with 3 decimal places
        assert!(output.contains("0.567"));
    }
}
