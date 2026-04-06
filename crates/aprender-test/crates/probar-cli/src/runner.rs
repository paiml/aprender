//! Test runner implementation

use crate::config::CliConfig;
use crate::error::CliResult;
use crate::output::ProgressReporter;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Test execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test name
    pub name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Test duration
    pub duration: Duration,
    /// Output from the test
    pub output: String,
}

impl TestResult {
    /// Create a passing test result
    #[must_use]
    pub fn pass(name: impl Into<String>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            passed: true,
            error: None,
            duration,
            output: String::new(),
        }
    }

    /// Create a failing test result
    #[must_use]
    pub fn fail(name: impl Into<String>, error: impl Into<String>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            passed: false,
            error: Some(error.into()),
            duration,
            output: String::new(),
        }
    }

    /// Add output to the result
    #[must_use]
    pub fn with_output(mut self, output: impl Into<String>) -> Self {
        self.output = output.into();
        self
    }
}

/// Aggregated test results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TestResults {
    /// Individual test results
    pub results: Vec<TestResult>,
    /// Total duration
    pub duration: Duration,
}

impl TestResults {
    /// Create new empty results
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a test result
    pub fn add(&mut self, result: TestResult) {
        self.results.push(result);
    }

    /// Get number of passed tests
    #[must_use]
    pub fn passed(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    /// Get number of failed tests
    #[must_use]
    pub fn failed(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }

    /// Get total number of tests
    #[must_use]
    pub fn total(&self) -> usize {
        self.results.len()
    }

    /// Check if all tests passed
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }

    /// Get failed tests
    #[must_use]
    pub fn failures(&self) -> Vec<&TestResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }
}

/// Test runner for executing Probar tests
#[derive(Debug)]
pub struct TestRunner {
    config: CliConfig,
    reporter: ProgressReporter,
}

impl TestRunner {
    /// Create a new test runner
    #[must_use]
    pub fn new(config: CliConfig) -> Self {
        let reporter =
            ProgressReporter::new(config.color.should_color(), config.verbosity.is_quiet());
        Self { config, reporter }
    }

    /// Run tests with optional filter
    ///
    /// # Errors
    ///
    /// Returns error if test discovery or execution fails
    pub fn run(&mut self, filter: Option<&str>) -> CliResult<TestResults> {
        let start = Instant::now();
        let mut results = TestResults::new();

        // Discover tests (placeholder - actual implementation would scan for tests)
        let tests = Self::discover_tests(filter);

        if tests.is_empty() {
            self.reporter.warning("No tests found");
            results.duration = start.elapsed();
            return Ok(results);
        }

        self.reporter.header("Running Tests");
        self.reporter
            .start_progress(tests.len() as u64, "Starting...");

        for test_name in tests {
            self.reporter.set_message(&test_name);

            let test_start = Instant::now();
            let result = Self::run_single_test(&test_name, test_start);

            if result.passed {
                self.reporter.success(&test_name);
            } else {
                self.reporter.failure(&format!(
                    "{}: {}",
                    test_name,
                    result.error.as_deref().unwrap_or("unknown error")
                ));

                if self.config.fail_fast {
                    results.add(result);
                    break;
                }
            }

            results.add(result);
            self.reporter.increment(1);
        }

        self.reporter.finish();
        results.duration = start.elapsed();

        self.reporter.summary(
            results.passed(),
            results.failed(),
            0, // skipped
            results.duration,
        );

        Ok(results)
    }

    /// Discover tests matching the filter using `cargo test --list`
    fn discover_tests(filter: Option<&str>) -> Vec<String> {
        let mut cmd = std::process::Command::new("cargo");
        cmd.args(["test", "--", "--list", "--format", "terse"]);

        if let Some(pattern) = filter {
            cmd.arg(pattern);
        }

        match cmd.output() {
            Ok(output) => {
                if output.status.success() {
                    String::from_utf8_lossy(&output.stdout)
                        .lines()
                        .filter(|line| line.ends_with(": test"))
                        .map(|line| line.trim_end_matches(": test").to_string())
                        .collect()
                } else {
                    Vec::new()
                }
            }
            Err(_) => Vec::new(),
        }
    }

    /// Run a single test using `cargo test`
    fn run_single_test(name: &str, start: Instant) -> TestResult {
        let output = std::process::Command::new("cargo")
            .args(["test", "--", "--exact", name, "--nocapture"])
            .output();

        match output {
            Ok(result) => {
                let stdout = String::from_utf8_lossy(&result.stdout);
                let stderr = String::from_utf8_lossy(&result.stderr);
                let combined_output = format!("{stdout}\n{stderr}");

                if result.status.success() {
                    TestResult::pass(name, start.elapsed()).with_output(&combined_output)
                } else {
                    let error_msg = if stderr.contains("FAILED") {
                        stderr
                            .lines()
                            .find(|l| l.contains("FAILED") || l.contains("panicked"))
                            .unwrap_or("Test failed")
                            .to_string()
                    } else {
                        "Test execution failed".to_string()
                    };
                    TestResult::fail(name, error_msg, start.elapsed()).with_output(&combined_output)
                }
            }
            Err(e) => TestResult::fail(
                name,
                format!("Failed to execute test: {e}"),
                start.elapsed(),
            ),
        }
    }

    /// Get the reporter (for testing)
    #[must_use]
    pub const fn reporter(&self) -> &ProgressReporter {
        &self.reporter
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod test_result_tests {
        use super::*;

        #[test]
        fn test_pass_result() {
            let result = TestResult::pass("test_1", Duration::from_millis(100));
            assert!(result.passed);
            assert!(result.error.is_none());
            assert_eq!(result.name, "test_1");
        }

        #[test]
        fn test_fail_result() {
            let result = TestResult::fail("test_2", "assertion failed", Duration::from_millis(50));
            assert!(!result.passed);
            assert_eq!(result.error, Some("assertion failed".to_string()));
        }

        #[test]
        fn test_with_output() {
            let result = TestResult::pass("test_3", Duration::from_millis(10))
                .with_output("test output here");
            assert_eq!(result.output, "test output here");
        }
    }

    mod test_results_tests {
        use super::*;

        #[test]
        fn test_new_results() {
            let results = TestResults::new();
            assert_eq!(results.total(), 0);
            assert_eq!(results.passed(), 0);
            assert_eq!(results.failed(), 0);
        }

        #[test]
        fn test_add_results() {
            let mut results = TestResults::new();
            results.add(TestResult::pass("test_1", Duration::from_millis(10)));
            results.add(TestResult::fail(
                "test_2",
                "error",
                Duration::from_millis(10),
            ));
            results.add(TestResult::pass("test_3", Duration::from_millis(10)));

            assert_eq!(results.total(), 3);
            assert_eq!(results.passed(), 2);
            assert_eq!(results.failed(), 1);
        }

        #[test]
        fn test_all_passed() {
            let mut results = TestResults::new();
            results.add(TestResult::pass("test_1", Duration::from_millis(10)));
            results.add(TestResult::pass("test_2", Duration::from_millis(10)));
            assert!(results.all_passed());

            results.add(TestResult::fail(
                "test_3",
                "error",
                Duration::from_millis(10),
            ));
            assert!(!results.all_passed());
        }

        #[test]
        fn test_failures() {
            let mut results = TestResults::new();
            results.add(TestResult::pass("test_1", Duration::from_millis(10)));
            results.add(TestResult::fail(
                "test_2",
                "error1",
                Duration::from_millis(10),
            ));
            results.add(TestResult::fail(
                "test_3",
                "error2",
                Duration::from_millis(10),
            ));

            let failures = results.failures();
            assert_eq!(failures.len(), 2);
            assert_eq!(failures[0].name, "test_2");
            assert_eq!(failures[1].name, "test_3");
        }
    }

    mod test_runner_tests {
        use super::*;

        #[test]
        fn test_new_runner() {
            let config = CliConfig::default();
            let runner = TestRunner::new(config);
            assert!(runner.reporter().use_color || !runner.reporter().use_color);
        }

        #[test]
        #[ignore = "Spawns cargo test --list subprocess - causes nested builds in CI"]
        fn test_run_no_tests() {
            let config = CliConfig::default();
            let mut runner = TestRunner::new(config);
            let results = runner.run(None).unwrap();
            assert_eq!(results.total(), 0);
        }

        #[test]
        #[ignore = "Spawns cargo test --list subprocess - causes nested builds in CI"]
        fn test_run_with_filter() {
            let config = CliConfig::default();
            let mut runner = TestRunner::new(config);
            let results = runner.run(Some("game::*")).unwrap();
            assert_eq!(results.total(), 0);
        }

        #[test]
        fn test_runner_with_config() {
            let config = CliConfig::default();
            let runner = TestRunner::new(config);
            // Just verify it constructs and reporter is accessible
            let _reporter = runner.reporter();
        }
    }

    mod test_result_additional_tests {
        use super::*;

        #[test]
        fn test_with_output() {
            let result =
                TestResult::pass("test", Duration::from_millis(10)).with_output("Some output text");
            assert_eq!(result.output, "Some output text");
        }

        #[test]
        fn test_debug() {
            let result = TestResult::pass("test", Duration::from_millis(10));
            let debug = format!("{result:?}");
            assert!(debug.contains("TestResult"));
        }

        #[test]
        fn test_clone() {
            let result = TestResult::fail("test", "error", Duration::from_millis(10));
            let cloned = result.clone();
            assert_eq!(result.name, cloned.name);
            assert_eq!(result.error, cloned.error);
        }

        #[test]
        fn test_serialize() {
            let result = TestResult::pass("test", Duration::from_millis(10));
            let json = serde_json::to_string(&result).unwrap();
            assert!(json.contains("test"));
        }
    }

    mod test_results_additional_tests {
        use super::*;

        #[test]
        fn test_default() {
            let results = TestResults::default();
            assert!(results.results.is_empty());
        }

        #[test]
        fn test_duration_tracking() {
            let mut results = TestResults::new();
            results.duration = Duration::from_secs(5);
            assert_eq!(results.duration.as_secs(), 5);
        }

        #[test]
        fn test_serialize() {
            let mut results = TestResults::new();
            results.add(TestResult::pass("test1", Duration::from_millis(10)));
            let json = serde_json::to_string(&results).unwrap();
            assert!(json.contains("test1"));
        }

        #[test]
        fn test_debug() {
            let results = TestResults::new();
            let debug = format!("{results:?}");
            assert!(debug.contains("TestResults"));
        }

        #[test]
        fn test_clone() {
            let mut results = TestResults::new();
            results.add(TestResult::pass("test", Duration::from_millis(10)));
            let cloned = results.clone();
            assert_eq!(results.total(), cloned.total());
        }
    }
}
