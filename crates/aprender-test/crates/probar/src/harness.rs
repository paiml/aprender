//! Test harness for running test suites.

use std::time::{Duration, Instant};

/// A test suite containing multiple tests
#[derive(Debug, Clone)]
pub struct TestSuite {
    /// Suite name
    pub name: String,
    /// Tests in this suite
    pub tests: Vec<TestCase>,
}

impl TestSuite {
    /// Create a new test suite
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tests: Vec::new(),
        }
    }

    /// Add a test case
    pub fn add_test(&mut self, test: TestCase) {
        self.tests.push(test);
    }

    /// Get the number of tests
    #[must_use]
    pub fn test_count(&self) -> usize {
        contract_pre_test_result_reporting!();
        self.tests.len()
    }
}

/// A single test case
#[derive(Debug, Clone)]
pub struct TestCase {
    /// Test name
    pub name: String,
    /// Test timeout in milliseconds
    pub timeout_ms: u64,
}

impl TestCase {
    /// Create a new test case
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            timeout_ms: 30000, // 30 second default
        }
    }

    /// Set timeout
    #[must_use]
    pub const fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }
}

/// Result of running a single test
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test name
    pub name: String,
    /// Whether test passed
    pub passed: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Test duration
    pub duration: Duration,
}

impl TestResult {
    /// Create a passing test result
    #[must_use]
    pub fn pass(name: impl Into<String>) -> Self {
        contract_pre_test_result_reporting!();
        Self {
            name: name.into(),
            passed: true,
            error: None,
            duration: Duration::ZERO,
        }
    }

    /// Create a failing test result
    #[must_use]
    pub fn fail(name: impl Into<String>, error: impl Into<String>) -> Self {
        contract_pre_test_result_reporting!();
        Self {
            name: name.into(),
            passed: false,
            error: Some(error.into()),
            duration: Duration::ZERO,
        }
    }

    /// Set duration
    #[must_use]
    pub const fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }
}

/// Results from running a test suite
#[derive(Debug, Clone)]
pub struct SuiteResults {
    /// Suite name
    pub suite_name: String,
    /// Individual test results
    pub results: Vec<TestResult>,
    /// Total duration
    pub duration: Duration,
}

impl SuiteResults {
    /// Check if all tests passed
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }

    /// Count passed tests
    #[must_use]
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    /// Count failed tests
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }

    /// Get total test count
    #[must_use]
    pub fn total(&self) -> usize {
        self.results.len()
    }

    /// Get failed tests
    #[must_use]
    pub fn failures(&self) -> Vec<&TestResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }
}

/// Test harness for running suites
#[derive(Debug, Default)]
pub struct TestHarness {
    /// Whether to stop on first failure
    pub fail_fast: bool,
    /// Whether to run tests in parallel
    pub parallel: bool,
}

impl TestHarness {
    /// Create a new test harness
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable fail-fast mode
    #[must_use]
    pub const fn with_fail_fast(mut self) -> Self {
        self.fail_fast = true;
        self
    }

    /// Enable parallel execution
    #[must_use]
    pub const fn with_parallel(mut self) -> Self {
        self.parallel = true;
        self
    }

    /// Run a test suite
    #[must_use]
    pub fn run(&self, suite: &TestSuite) -> SuiteResults {
        let start = Instant::now();
        let results = Vec::new();

        // In a full implementation, this would actually run the tests
        // For now, return empty results for an empty suite

        SuiteResults {
            suite_name: suite.name.clone(),
            results,
            duration: start.elapsed(),
        }
    }
}
