//! Reporter - Test Reporting with Andon Cord Support
//!
//! Per spec Section 3.5: Test reporting with fail-fast mode (Andon Cord pattern).
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────┐
//! │  Reporter with Andon Cord                                              │
//! │  ─────────────────────────────                                         │
//! │                                                                        │
//! │  Toyota Principle: ANDON CORD                                          │
//! │  "In Toyota factories, any worker can pull the cord to stop           │
//! │   production when a defect is detected"                               │
//! │                                                                        │
//! │  ┌────────────────────┐     ┌──────────────────────┐                  │
//! │  │  FailureMode::     │     │  FailureMode::       │                  │
//! │  │  AndonCord         │     │  CollectAll          │                  │
//! │  │                    │     │                      │                  │
//! │  │  STOP on first     │     │  Collect ALL         │                  │
//! │  │  failure (default) │     │  failures for        │                  │
//! │  │                    │     │  exploratory testing │                  │
//! │  └────────────────────┘     └──────────────────────┘                  │
//! └────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Toyota Principles Applied
//!
//! - **Andon Cord**: Stop immediately on critical failure
//! - **Jidoka**: Build quality in by failing fast

use crate::bridge::VisualDiff;
use crate::driver::Screenshot;
use crate::result::{ProbarError, ProbarResult};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, SystemTime};

/// Failure mode for test execution
///
/// Andon Cord: Stop the line on first failure
/// CollectAll: Gather all failures (for exploratory testing)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FailureMode {
    /// Stop on first failure (Toyota Andon Cord)
    #[default]
    AndonCord,
    /// Collect all failures (exploratory mode)
    CollectAll,
}

/// Test result status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestStatus {
    /// Test passed
    Passed,
    /// Test failed
    Failed,
    /// Test was skipped
    Skipped,
    /// Test is pending
    Pending,
}

impl TestStatus {
    /// Check if status is passing
    #[must_use]
    pub const fn is_passed(&self) -> bool {
        matches!(self, Self::Passed)
    }

    /// Check if status is failing
    #[must_use]
    pub const fn is_failed(&self) -> bool {
        matches!(self, Self::Failed)
    }
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResultEntry {
    /// Test name
    pub name: String,
    /// Test status
    pub status: TestStatus,
    /// Duration of test execution
    pub duration: Duration,
    /// Error message if failed
    pub error: Option<String>,
    /// Screenshot on failure
    #[serde(skip)]
    pub failure_screenshot: Option<Screenshot>,
    /// Stack trace if available
    pub stack_trace: Option<String>,
    /// Timestamp when test completed
    pub timestamp: SystemTime,
}

impl TestResultEntry {
    /// Create a passing test result
    #[must_use]
    pub fn passed(name: impl Into<String>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            status: TestStatus::Passed,
            duration,
            error: None,
            failure_screenshot: None,
            stack_trace: None,
            timestamp: SystemTime::now(),
        }
    }

    /// Create a failing test result
    #[must_use]
    pub fn failed(name: impl Into<String>, duration: Duration, error: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: TestStatus::Failed,
            duration,
            error: Some(error.into()),
            failure_screenshot: None,
            stack_trace: None,
            timestamp: SystemTime::now(),
        }
    }

    /// Create a skipped test result
    #[must_use]
    pub fn skipped(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: TestStatus::Skipped,
            duration: Duration::ZERO,
            error: None,
            failure_screenshot: None,
            stack_trace: None,
            timestamp: SystemTime::now(),
        }
    }

    /// Add a screenshot to the result
    #[must_use]
    pub fn with_screenshot(mut self, screenshot: Screenshot) -> Self {
        self.failure_screenshot = Some(screenshot);
        self
    }

    /// Add a stack trace to the result
    #[must_use]
    pub fn with_stack_trace(mut self, trace: impl Into<String>) -> Self {
        self.stack_trace = Some(trace.into());
        self
    }
}

/// Trace data for performance analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceData {
    /// Total test duration
    pub total_duration: Duration,
    /// Individual step timings
    pub step_timings: Vec<(String, Duration)>,
    /// Memory usage samples
    pub memory_samples: Vec<(Duration, u64)>,
    /// Frame rate samples
    pub fps_samples: Vec<(Duration, f64)>,
}

impl TraceData {
    /// Create new trace data
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a step timing
    pub fn add_step(&mut self, name: impl Into<String>, duration: Duration) {
        self.step_timings.push((name.into(), duration));
    }

    /// Add a memory sample
    pub fn add_memory_sample(&mut self, elapsed: Duration, bytes: u64) {
        self.memory_samples.push((elapsed, bytes));
    }

    /// Add a FPS sample
    pub fn add_fps_sample(&mut self, elapsed: Duration, fps: f64) {
        self.fps_samples.push((elapsed, fps));
    }

    /// Get average FPS
    #[must_use]
    pub fn average_fps(&self) -> f64 {
        if self.fps_samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.fps_samples.iter().map(|(_, fps)| fps).sum();
        sum / self.fps_samples.len() as f64
    }

    /// Get peak memory usage
    #[must_use]
    pub fn peak_memory(&self) -> u64 {
        self.memory_samples
            .iter()
            .map(|(_, mem)| *mem)
            .max()
            .unwrap_or(0)
    }
}

/// Andon Cord pulled error
///
/// This error is returned when fail-fast mode stops execution
#[derive(Debug)]
pub struct AndonCordPulled {
    /// Name of the failing test
    pub test_name: String,
    /// Failure message
    pub failure: String,
    /// Screenshot at time of failure
    pub screenshot: Option<Screenshot>,
}

/// Test reporter with Andon Cord support
///
/// The reporter collects test results and can generate various output formats.
/// In AndonCord mode, it stops on the first failure.
///
/// # Example
///
/// ```ignore
/// let mut reporter = Reporter::andon(); // Fail-fast mode
///
/// reporter.record(TestResultEntry::passed("test_1", Duration::from_millis(100)))?;
/// reporter.record(TestResultEntry::failed("test_2", Duration::from_millis(50), "assertion failed"))?;
/// // ^ This will return Err(AndonCordPulled)
/// ```
#[derive(Debug, Default)]
pub struct Reporter {
    /// Test results
    results: Vec<TestResultEntry>,
    /// Screenshots taken during testing
    screenshots: Vec<(String, Screenshot)>,
    /// Visual diffs
    visual_diffs: Vec<(String, VisualDiff)>,
    /// Trace data
    traces: Vec<TraceData>,
    /// Failure mode
    failure_mode: FailureMode,
    /// Suite name
    suite_name: String,
    /// Start time
    start_time: Option<SystemTime>,
}

impl Reporter {
    /// Create new reporter with default settings (CollectAll mode)
    #[must_use]
    pub fn new() -> Self {
        Self {
            suite_name: "Test Suite".to_string(),
            ..Default::default()
        }
    }

    /// Create reporter with Andon Cord mode (fail-fast)
    #[must_use]
    pub fn andon() -> Self {
        Self {
            failure_mode: FailureMode::AndonCord,
            suite_name: "Test Suite".to_string(),
            ..Default::default()
        }
    }

    /// Create reporter with CollectAll mode
    #[must_use]
    pub fn collect_all() -> Self {
        Self {
            failure_mode: FailureMode::CollectAll,
            suite_name: "Test Suite".to_string(),
            ..Default::default()
        }
    }

    /// Set suite name
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.suite_name = name.into();
        self
    }

    /// Start the test suite
    pub fn start(&mut self) {
        self.start_time = Some(SystemTime::now());
    }

    /// Record a test result
    ///
    /// # Errors
    ///
    /// In AndonCord mode, returns error if test failed
    pub fn record(&mut self, result: TestResultEntry) -> ProbarResult<()> {
        let failed = result.status.is_failed();
        let failure_info = if failed {
            Some((
                result.name.clone(),
                result.error.clone().unwrap_or_default(),
            ))
        } else {
            None
        };

        self.results.push(result);

        if self.failure_mode == FailureMode::AndonCord {
            if let Some((test_name, failure)) = failure_info {
                // ANDON CORD PULLED: Stop immediately
                return Err(ProbarError::AssertionFailed {
                    message: format!("ANDON CORD PULLED: Test '{test_name}' failed: {failure}"),
                });
            }
        }

        Ok(())
    }

    /// Add a screenshot
    pub fn add_screenshot(&mut self, name: impl Into<String>, screenshot: Screenshot) {
        self.screenshots.push((name.into(), screenshot));
    }

    /// Add a visual diff
    pub fn add_visual_diff(&mut self, name: impl Into<String>, diff: VisualDiff) {
        self.visual_diffs.push((name.into(), diff));
    }

    /// Add trace data
    pub fn add_trace(&mut self, trace: TraceData) {
        self.traces.push(trace);
    }

    /// Get number of passed tests
    #[must_use]
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.status.is_passed()).count()
    }

    /// Get number of failed tests
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.results.iter().filter(|r| r.status.is_failed()).count()
    }

    /// Get total test count
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.results.len()
    }

    /// Get pass rate (0.0 to 1.0)
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 1.0;
        }
        self.passed_count() as f64 / self.results.len() as f64
    }

    /// Check if all tests passed
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.failed_count() == 0
    }

    /// Get total duration
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.results.iter().map(|r| r.duration).sum()
    }

    /// Get test results
    #[must_use]
    pub fn results(&self) -> &[TestResultEntry] {
        &self.results
    }

    /// Get failing tests
    #[must_use]
    pub fn failures(&self) -> Vec<&TestResultEntry> {
        self.results
            .iter()
            .filter(|r| r.status.is_failed())
            .collect()
    }

    /// Generate summary string
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{}: {}/{} passed ({:.1}%)",
            self.suite_name,
            self.passed_count(),
            self.total_count(),
            self.pass_rate() * 100.0
        )
    }

    /// Generate HTML report
    ///
    /// # Errors
    ///
    /// Returns error if file writing fails
    pub fn generate_html(&self, output_path: &Path) -> ProbarResult<()> {
        let html = self.render_html();
        std::fs::write(output_path, html)?;
        Ok(())
    }

    /// Render HTML report content
    #[must_use]
    pub fn render_html(&self) -> String {
        let mut html = String::new();

        // Header
        html.push_str(r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Probar Test Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }
        .summary { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .progress-bar { background: #ddd; height: 20px; border-radius: 10px; overflow: hidden; }
        .passed { background: #4caf50; height: 100%; }
        .test { padding: 10px; margin: 5px 0; border-radius: 4px; }
        .test.pass { background: #e8f5e9; border-left: 4px solid #4caf50; }
        .test.fail { background: #ffebee; border-left: 4px solid #f44336; }
        .test.skip { background: #fff3e0; border-left: 4px solid #ff9800; }
        .error { color: #d32f2f; font-family: monospace; white-space: pre-wrap; }
        .visual-diff { display: flex; gap: 10px; margin: 10px 0; }
        .visual-diff img { max-width: 300px; border: 1px solid #ddd; }
    </style>
</head>
<body>
"#);

        // Summary
        html.push_str(&format!(
            r#"<div class="summary">
    <h1>{}</h1>
    <h2>Results: {}/{} passed ({:.1}%)</h2>
    <div class="progress-bar">
        <div class="passed" style="width: {:.1}%"></div>
    </div>
    <p>Duration: {:.2}s</p>
</div>
"#,
            self.suite_name,
            self.passed_count(),
            self.total_count(),
            self.pass_rate() * 100.0,
            self.pass_rate() * 100.0,
            self.total_duration().as_secs_f64()
        ));

        // Test results
        html.push_str("<h2>Test Results</h2>\n");
        for result in &self.results {
            let class = match result.status {
                TestStatus::Passed => "pass",
                TestStatus::Failed => "fail",
                TestStatus::Skipped | TestStatus::Pending => "skip",
            };

            html.push_str(&format!(
                r#"<div class="test {}">
    <strong>{}</strong> - {:?} ({:.2}ms)
"#,
                class,
                result.name,
                result.status,
                result.duration.as_secs_f64() * 1000.0
            ));

            if let Some(error) = &result.error {
                html.push_str(&format!(r#"    <div class="error">{error}</div>"#));
            }

            html.push_str("</div>\n");
        }

        // Visual diffs
        if !self.visual_diffs.is_empty() {
            html.push_str("<h2>Visual Differences</h2>\n");
            for (name, diff) in &self.visual_diffs {
                html.push_str(&format!(
                    r#"<div>
    <h3>{}</h3>
    <p>Similarity: {:.1}%</p>
    <div class="visual-diff">
        <div><strong>Expected</strong><br><img alt="Expected"></div>
        <div><strong>Actual</strong><br><img alt="Actual"></div>
        <div><strong>Diff</strong><br><img alt="Diff"></div>
    </div>
</div>
"#,
                    name,
                    diff.perceptual_similarity * 100.0
                ));
            }
        }

        // Footer
        html.push_str(
            r#"
<footer>
    <p>Generated by Probar - WASM Game Testing Framework</p>
</footer>
</body>
</html>
"#,
        );

        html
    }

    /// Generate JUnit XML for CI integration
    ///
    /// # Errors
    ///
    /// Returns error if file writing fails
    pub fn generate_junit(&self, output_path: &Path) -> ProbarResult<()> {
        let xml = self.render_junit();
        std::fs::write(output_path, xml)?;
        Ok(())
    }

    /// Render JUnit XML content
    #[must_use]
    pub fn render_junit(&self) -> String {
        let mut xml = String::new();

        xml.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        xml.push('\n');
        xml.push_str(&format!(
            r#"<testsuite name="{}" tests="{}" failures="{}" time="{:.3}">"#,
            self.suite_name,
            self.total_count(),
            self.failed_count(),
            self.total_duration().as_secs_f64()
        ));
        xml.push('\n');

        for result in &self.results {
            xml.push_str(&format!(
                r#"  <testcase name="{}" time="{:.3}">"#,
                result.name,
                result.duration.as_secs_f64()
            ));
            xml.push('\n');

            if let Some(error) = &result.error {
                xml.push_str(&format!(
                    r#"    <failure message="{}">{}</failure>"#,
                    escape_xml(error),
                    escape_xml(error)
                ));
                xml.push('\n');
            }

            xml.push_str("  </testcase>\n");
        }

        xml.push_str("</testsuite>\n");
        xml
    }
}

/// Escape XML special characters
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// ============================================================================
// EXTREME TDD: Tests written FIRST per spec Section 6.1
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod failure_mode_tests {
        use super::*;

        #[test]
        fn test_default_failure_mode() {
            let mode = FailureMode::default();
            assert_eq!(mode, FailureMode::AndonCord);
        }
    }

    mod test_status_tests {
        use super::*;

        #[test]
        fn test_status_is_passed() {
            assert!(TestStatus::Passed.is_passed());
            assert!(!TestStatus::Failed.is_passed());
            assert!(!TestStatus::Skipped.is_passed());
        }

        #[test]
        fn test_status_is_failed() {
            assert!(!TestStatus::Passed.is_failed());
            assert!(TestStatus::Failed.is_failed());
            assert!(!TestStatus::Skipped.is_failed());
        }
    }

    mod test_result_entry_tests {
        use super::*;

        #[test]
        fn test_passed_result() {
            let result = TestResultEntry::passed("test_1", Duration::from_millis(100));
            assert_eq!(result.name, "test_1");
            assert_eq!(result.status, TestStatus::Passed);
            assert!(result.error.is_none());
        }

        #[test]
        fn test_failed_result() {
            let result =
                TestResultEntry::failed("test_2", Duration::from_millis(50), "assertion failed");
            assert_eq!(result.name, "test_2");
            assert_eq!(result.status, TestStatus::Failed);
            assert_eq!(result.error, Some("assertion failed".to_string()));
        }

        #[test]
        fn test_skipped_result() {
            let result = TestResultEntry::skipped("test_3");
            assert_eq!(result.status, TestStatus::Skipped);
            assert_eq!(result.duration, Duration::ZERO);
        }

        #[test]
        fn test_with_stack_trace() {
            let result = TestResultEntry::failed("test", Duration::ZERO, "error")
                .with_stack_trace("at line 42");
            assert_eq!(result.stack_trace, Some("at line 42".to_string()));
        }
    }

    mod trace_data_tests {
        use super::*;

        #[test]
        fn test_new_trace() {
            let trace = TraceData::new();
            assert!(trace.step_timings.is_empty());
            assert!(trace.memory_samples.is_empty());
        }

        #[test]
        fn test_add_step() {
            let mut trace = TraceData::new();
            trace.add_step("setup", Duration::from_millis(10));
            assert_eq!(trace.step_timings.len(), 1);
        }

        #[test]
        fn test_average_fps() {
            let mut trace = TraceData::new();
            trace.add_fps_sample(Duration::ZERO, 60.0);
            trace.add_fps_sample(Duration::from_secs(1), 50.0);
            trace.add_fps_sample(Duration::from_secs(2), 55.0);
            assert!((trace.average_fps() - 55.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_average_fps_empty() {
            let trace = TraceData::new();
            assert!((trace.average_fps() - 0.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_peak_memory() {
            let mut trace = TraceData::new();
            trace.add_memory_sample(Duration::ZERO, 1000);
            trace.add_memory_sample(Duration::from_secs(1), 5000);
            trace.add_memory_sample(Duration::from_secs(2), 3000);
            assert_eq!(trace.peak_memory(), 5000);
        }
    }

    mod reporter_tests {
        use super::*;

        #[test]
        fn test_new_reporter() {
            let reporter = Reporter::new();
            assert_eq!(reporter.total_count(), 0);
            assert!(reporter.all_passed());
        }

        #[test]
        fn test_andon_reporter() {
            let reporter = Reporter::andon();
            assert_eq!(reporter.failure_mode, FailureMode::AndonCord);
        }

        #[test]
        fn test_collect_all_reporter() {
            let reporter = Reporter::collect_all();
            assert_eq!(reporter.failure_mode, FailureMode::CollectAll);
        }

        #[test]
        fn test_with_name() {
            let reporter = Reporter::new().with_name("My Tests");
            assert_eq!(reporter.suite_name, "My Tests");
        }

        #[test]
        fn test_record_passing() {
            let mut reporter = Reporter::andon();
            let result = reporter.record(TestResultEntry::passed("test", Duration::ZERO));
            assert!(result.is_ok());
            assert_eq!(reporter.passed_count(), 1);
        }

        #[test]
        fn test_andon_cord_pulled() {
            let mut reporter = Reporter::andon();
            let result = reporter.record(TestResultEntry::failed("test", Duration::ZERO, "error"));
            assert!(result.is_err());
        }

        #[test]
        fn test_collect_all_continues() {
            let mut reporter = Reporter::collect_all();
            let result1 =
                reporter.record(TestResultEntry::failed("test1", Duration::ZERO, "error"));
            let result2 = reporter.record(TestResultEntry::passed("test2", Duration::ZERO));
            assert!(result1.is_ok()); // CollectAll doesn't stop
            assert!(result2.is_ok());
            assert_eq!(reporter.failed_count(), 1);
            assert_eq!(reporter.passed_count(), 1);
        }

        #[test]
        fn test_pass_rate() {
            let mut reporter = Reporter::collect_all();
            reporter
                .record(TestResultEntry::passed("t1", Duration::ZERO))
                .unwrap();
            reporter
                .record(TestResultEntry::passed("t2", Duration::ZERO))
                .unwrap();
            reporter
                .record(TestResultEntry::failed("t3", Duration::ZERO, "err"))
                .unwrap();
            reporter
                .record(TestResultEntry::passed("t4", Duration::ZERO))
                .unwrap();

            assert!((reporter.pass_rate() - 0.75).abs() < f64::EPSILON);
        }

        #[test]
        fn test_pass_rate_empty() {
            let reporter = Reporter::new();
            assert!((reporter.pass_rate() - 1.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_total_duration() {
            let mut reporter = Reporter::collect_all();
            reporter
                .record(TestResultEntry::passed("t1", Duration::from_millis(100)))
                .unwrap();
            reporter
                .record(TestResultEntry::passed("t2", Duration::from_millis(200)))
                .unwrap();
            assert_eq!(reporter.total_duration(), Duration::from_millis(300));
        }

        #[test]
        fn test_failures() {
            let mut reporter = Reporter::collect_all();
            reporter
                .record(TestResultEntry::passed("t1", Duration::ZERO))
                .unwrap();
            reporter
                .record(TestResultEntry::failed("t2", Duration::ZERO, "err"))
                .unwrap();
            reporter
                .record(TestResultEntry::passed("t3", Duration::ZERO))
                .unwrap();

            let failures = reporter.failures();
            assert_eq!(failures.len(), 1);
            assert_eq!(failures[0].name, "t2");
        }

        #[test]
        fn test_summary() {
            let mut reporter = Reporter::collect_all().with_name("Game Tests");
            reporter
                .record(TestResultEntry::passed("t1", Duration::ZERO))
                .unwrap();
            reporter
                .record(TestResultEntry::passed("t2", Duration::ZERO))
                .unwrap();

            let summary = reporter.summary();
            assert!(summary.contains("Game Tests"));
            assert!(summary.contains("2/2"));
            assert!(summary.contains("100.0%"));
        }

        #[test]
        fn test_render_html() {
            let mut reporter = Reporter::collect_all().with_name("HTML Test");
            reporter
                .record(TestResultEntry::passed("t1", Duration::from_millis(50)))
                .unwrap();
            reporter
                .record(TestResultEntry::failed(
                    "t2",
                    Duration::from_millis(10),
                    "assertion failed",
                ))
                .unwrap();

            let html = reporter.render_html();
            assert!(html.contains("HTML Test"));
            assert!(html.contains("t1"));
            assert!(html.contains("t2"));
            assert!(html.contains("assertion failed"));
        }

        #[test]
        fn test_render_junit() {
            let mut reporter = Reporter::collect_all().with_name("JUnit Test");
            reporter
                .record(TestResultEntry::passed(
                    "passing_test",
                    Duration::from_millis(100),
                ))
                .unwrap();
            reporter
                .record(TestResultEntry::failed(
                    "failing_test",
                    Duration::from_millis(50),
                    "error msg",
                ))
                .unwrap();

            let xml = reporter.render_junit();
            assert!(xml.contains("JUnit Test"));
            assert!(xml.contains("passing_test"));
            assert!(xml.contains("failing_test"));
            assert!(xml.contains("error msg"));
        }
    }

    mod escape_xml_tests {
        use super::*;

        #[test]
        fn test_escape_special_chars() {
            assert_eq!(escape_xml("a & b"), "a &amp; b");
            assert_eq!(escape_xml("<tag>"), "&lt;tag&gt;");
            assert_eq!(escape_xml("\"quoted\""), "&quot;quoted&quot;");
            assert_eq!(escape_xml("it's"), "it&apos;s");
        }

        #[test]
        fn test_no_escape_needed() {
            assert_eq!(escape_xml("plain text"), "plain text");
        }
    }

    mod additional_coverage_tests {
        use super::*;
        use tempfile::tempdir;

        #[test]
        fn test_with_screenshot() {
            let screenshot = Screenshot::new(vec![1, 2, 3], 100, 100);
            let result = TestResultEntry::failed("test", Duration::ZERO, "error")
                .with_screenshot(screenshot);
            assert!(result.failure_screenshot.is_some());
        }

        #[test]
        fn test_reporter_start() {
            let mut reporter = Reporter::new();
            assert!(reporter.start_time.is_none());
            reporter.start();
            assert!(reporter.start_time.is_some());
        }

        #[test]
        fn test_reporter_add_screenshot() {
            let mut reporter = Reporter::new();
            let screenshot = Screenshot::new(vec![1, 2, 3], 100, 100);
            reporter.add_screenshot("test_shot", screenshot);
            assert_eq!(reporter.screenshots.len(), 1);
        }

        #[test]
        fn test_reporter_add_visual_diff() {
            let mut reporter = Reporter::new();
            let diff = VisualDiff::new(0.95, vec![1, 2, 3]);
            reporter.add_visual_diff("test_diff", diff);
            assert_eq!(reporter.visual_diffs.len(), 1);
        }

        #[test]
        fn test_reporter_add_trace() {
            let mut reporter = Reporter::new();
            let mut trace = TraceData::new();
            trace.add_step("step1", Duration::from_millis(100));
            reporter.add_trace(trace);
            assert_eq!(reporter.traces.len(), 1);
        }

        #[test]
        fn test_reporter_results_accessor() {
            let mut reporter = Reporter::collect_all();
            reporter
                .record(TestResultEntry::passed("t1", Duration::ZERO))
                .unwrap();
            reporter
                .record(TestResultEntry::passed("t2", Duration::ZERO))
                .unwrap();

            let results = reporter.results();
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].name, "t1");
            assert_eq!(results[1].name, "t2");
        }

        #[test]
        fn test_generate_html_to_file() {
            let mut reporter = Reporter::collect_all().with_name("File Test");
            reporter
                .record(TestResultEntry::passed("t1", Duration::ZERO))
                .unwrap();

            let dir = tempdir().unwrap();
            let path = dir.path().join("report.html");

            let result = reporter.generate_html(&path);
            assert!(result.is_ok());
            assert!(path.exists());

            let content = std::fs::read_to_string(&path).unwrap();
            assert!(content.contains("File Test"));
        }

        #[test]
        fn test_generate_junit_to_file() {
            let mut reporter = Reporter::collect_all().with_name("JUnit File Test");
            reporter
                .record(TestResultEntry::passed("t1", Duration::ZERO))
                .unwrap();

            let dir = tempdir().unwrap();
            let path = dir.path().join("report.xml");

            let result = reporter.generate_junit(&path);
            assert!(result.is_ok());
            assert!(path.exists());

            let content = std::fs::read_to_string(&path).unwrap();
            assert!(content.contains("JUnit File Test"));
        }

        #[test]
        fn test_render_html_with_visual_diffs() {
            let mut reporter = Reporter::collect_all().with_name("Visual Test");
            reporter
                .record(TestResultEntry::passed("t1", Duration::ZERO))
                .unwrap();

            let diff = VisualDiff::new(0.85, vec![1, 2, 3]);
            reporter.add_visual_diff("homepage", diff);

            let html = reporter.render_html();
            assert!(html.contains("Visual Differences"));
            assert!(html.contains("homepage"));
            assert!(html.contains("85.0%")); // 0.85 * 100
        }
    }
}
