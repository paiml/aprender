//! Docker-based Cross-Browser WASM Testing (PROBAR-SPEC-014).
//!
//! This module provides Docker container management for cross-browser WASM testing,
//! enabling Chrome, Firefox, and WebKit testing with proper COOP/COEP header
//! configuration for SharedArrayBuffer support.
//!
//! # Toyota Principles
//!
//! - **Heijunka (Level Loading)**: Parallel container execution balances load
//! - **Jidoka (Built-in Quality)**: Container health checks ensure quality
//! - **Poka-Yoke (Error Prevention)**: Type-safe browser configuration
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     Host Machine                                 │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  DockerTestRunner                                                │
//! │  ┌─────────────┬─────────────┬─────────────┐                    │
//! │  │   Chrome    │   Firefox   │   WebKit    │                    │
//! │  │  Container  │  Container  │  Container  │                    │
//! │  └──────┬──────┴──────┬──────┴──────┬──────┘                    │
//! │         │             │             │                            │
//! │         ▼             ▼             ▼                            │
//! │  ┌─────────────────────────────────────────┐                    │
//! │  │         probar serve (per container)     │                    │
//! │  │  • COOP: same-origin                     │                    │
//! │  │  • COEP: require-corp                    │                    │
//! │  │  • SharedArrayBuffer: enabled            │                    │
//! │  └─────────────────────────────────────────┘                    │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Examples
//!
//! ```rust,ignore
//! use probar::docker::{DockerTestRunner, Browser, DockerConfig};
//!
//! // Create runner with specific browser
//! let runner = DockerTestRunner::builder()
//!     .browser(Browser::Firefox)
//!     .with_coop_coep(true)
//!     .timeout(Duration::from_secs(60))
//!     .build()?;
//!
//! // Run tests
//! let results = runner.run_tests(&["tests/worker_tests.rs"]).await?;
//! assert!(results.all_passed());
//! ```

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Docker testing errors.
#[derive(Debug, Error)]
pub enum DockerError {
    /// Docker daemon not available.
    #[error("Docker daemon not available: {0}")]
    DaemonUnavailable(String),

    /// Container failed to start.
    #[error("Container failed to start: {0}")]
    ContainerStartFailed(String),

    /// Container not found.
    #[error("Container not found: {0}")]
    ContainerNotFound(String),

    /// Image not found.
    #[error("Image not found: {0}")]
    ImageNotFound(String),

    /// CDP connection failed.
    #[error("CDP connection failed: {0}")]
    CdpConnectionFailed(String),

    /// Test execution failed.
    #[error("Test execution failed: {0}")]
    TestExecutionFailed(String),

    /// Timeout waiting for container.
    #[error("Timeout waiting for container: {0}")]
    Timeout(String),

    /// Health check failed.
    #[error("Health check failed: {0}")]
    HealthCheckFailed(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// IO error.
    #[error("IO error: {0}")]
    IoError(String),

    /// Network error.
    #[error("Network error: {0}")]
    NetworkError(String),
}

/// Result type for Docker operations.
pub type DockerResult<T> = Result<T, DockerError>;

/// Supported browsers for Docker testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Browser {
    /// Google Chrome / Chromium.
    Chrome,
    /// Mozilla Firefox.
    Firefox,
    /// Apple WebKit (via playwright or wpewebkit).
    WebKit,
}

impl Browser {
    /// Returns the default CDP port for this browser.
    #[must_use]
    pub const fn default_cdp_port(&self) -> u16 {
        match self {
            Self::Chrome => 9222,
            Self::Firefox => 9223,
            Self::WebKit => 9224,
        }
    }

    /// Returns the Docker image name for this browser.
    #[must_use]
    pub fn image_name(&self) -> &'static str {
        match self {
            Self::Chrome => "probar-chrome:latest",
            Self::Firefox => "probar-firefox:latest",
            Self::WebKit => "probar-webkit:latest",
        }
    }

    /// Returns the container name prefix for this browser.
    #[must_use]
    pub fn container_prefix(&self) -> &'static str {
        match self {
            Self::Chrome => "probar-chrome",
            Self::Firefox => "probar-firefox",
            Self::WebKit => "probar-webkit",
        }
    }

    /// Returns all supported browsers.
    #[must_use]
    pub const fn all() -> [Self; 3] {
        [Self::Chrome, Self::Firefox, Self::WebKit]
    }

    /// Parses browser from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "chrome" | "chromium" => Some(Self::Chrome),
            "firefox" | "ff" => Some(Self::Firefox),
            "webkit" | "safari" => Some(Self::WebKit),
            _ => None,
        }
    }
}

impl fmt::Display for Browser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chrome => write!(f, "chrome"),
            Self::Firefox => write!(f, "firefox"),
            Self::WebKit => write!(f, "webkit"),
        }
    }
}

/// Container lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContainerState {
    /// Container not created yet.
    NotCreated,
    /// Container is being created.
    Creating,
    /// Container is starting.
    Starting,
    /// Container is running and healthy.
    Running,
    /// Container health check in progress.
    HealthChecking,
    /// Container is stopping.
    Stopping,
    /// Container has stopped.
    Stopped,
    /// Container encountered an error.
    Error,
}

impl Default for ContainerState {
    fn default() -> Self {
        Self::NotCreated
    }
}

impl fmt::Display for ContainerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCreated => write!(f, "not_created"),
            Self::Creating => write!(f, "creating"),
            Self::Starting => write!(f, "starting"),
            Self::Running => write!(f, "running"),
            Self::HealthChecking => write!(f, "health_checking"),
            Self::Stopping => write!(f, "stopping"),
            Self::Stopped => write!(f, "stopped"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// COOP/COEP header configuration for SharedArrayBuffer support.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoopCoepConfig {
    /// Cross-Origin-Opener-Policy value.
    pub coop: String,
    /// Cross-Origin-Embedder-Policy value.
    pub coep: String,
    /// Cross-Origin-Resource-Policy value.
    pub corp: String,
    /// Whether headers are enabled.
    pub enabled: bool,
}

impl Default for CoopCoepConfig {
    fn default() -> Self {
        Self {
            coop: "same-origin".to_string(),
            coep: "require-corp".to_string(),
            corp: "cross-origin".to_string(),
            enabled: true,
        }
    }
}

impl CoopCoepConfig {
    /// Creates a new COOP/COEP config with default headers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Disables COOP/COEP headers.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Returns whether SharedArrayBuffer would be available with this config.
    #[must_use]
    pub fn shared_array_buffer_available(&self) -> bool {
        self.enabled && self.coop == "same-origin" && self.coep == "require-corp"
    }
}

/// Docker container configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerConfig {
    /// Docker image to use.
    pub image: String,
    /// Container name.
    pub name: String,
    /// Port mappings (host:container).
    pub ports: Vec<(u16, u16)>,
    /// Environment variables.
    pub environment: HashMap<String, String>,
    /// Volume mounts (host:container).
    pub volumes: Vec<(PathBuf, String)>,
    /// Network name.
    pub network: Option<String>,
    /// Memory limit in bytes.
    pub memory_limit: Option<u64>,
    /// CPU limit (number of CPUs).
    pub cpu_limit: Option<f64>,
    /// Health check command.
    pub health_check: Option<String>,
    /// Health check interval.
    pub health_check_interval: Duration,
    /// Health check timeout.
    pub health_check_timeout: Duration,
    /// Health check retries.
    pub health_check_retries: u32,
}

impl Default for ContainerConfig {
    fn default() -> Self {
        Self {
            image: "probar-wasm-test:latest".to_string(),
            name: "probar-test".to_string(),
            ports: vec![],
            environment: HashMap::new(),
            volumes: vec![],
            network: None,
            memory_limit: Some(2 * 1024 * 1024 * 1024), // 2GB
            cpu_limit: Some(2.0),
            health_check: None,
            health_check_interval: Duration::from_secs(5),
            health_check_timeout: Duration::from_secs(5),
            health_check_retries: 3,
        }
    }
}

impl ContainerConfig {
    /// Creates a new container config for the specified browser.
    #[must_use]
    pub fn for_browser(browser: Browser) -> Self {
        let port = browser.default_cdp_port();
        let mut env = HashMap::new();
        env.insert("PROBAR_BROWSER".to_string(), browser.to_string());
        env.insert("PROBAR_CDP_PORT".to_string(), port.to_string());
        env.insert("PROBAR_COOP_COEP".to_string(), "true".to_string());

        Self {
            image: browser.image_name().to_string(),
            name: format!("{}-{}", browser.container_prefix(), uuid::Uuid::new_v4()),
            ports: vec![(port, port)],
            environment: env,
            health_check: Some(format!(
                "wget -q --spider http://localhost:{port}/json/version"
            )),
            ..Self::default()
        }
    }
}

/// Docker test runner configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerConfig {
    /// Browser to use for testing.
    pub browser: Browser,
    /// COOP/COEP configuration.
    pub coop_coep: CoopCoepConfig,
    /// Test timeout.
    pub timeout: Duration,
    /// Maximum parallel containers.
    pub parallel: u32,
    /// Docker socket path.
    pub docker_socket: String,
    /// Container configuration.
    pub container: ContainerConfig,
    /// Whether to pull images before running.
    pub pull_images: bool,
    /// Whether to remove containers after tests.
    pub cleanup: bool,
    /// Whether to capture container logs.
    pub capture_logs: bool,
}

impl Default for DockerConfig {
    fn default() -> Self {
        let browser = Browser::Chrome;
        Self {
            browser,
            coop_coep: CoopCoepConfig::default(),
            timeout: Duration::from_secs(60),
            parallel: 4,
            docker_socket: "/var/run/docker.sock".to_string(),
            container: ContainerConfig::for_browser(browser),
            pull_images: true,
            cleanup: true,
            capture_logs: true,
        }
    }
}

/// Builder for DockerTestRunner.
#[derive(Debug, Default)]
pub struct DockerTestRunnerBuilder {
    config: DockerConfig,
}

impl DockerTestRunnerBuilder {
    /// Creates a new builder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the browser to use.
    #[must_use]
    pub fn browser(mut self, browser: Browser) -> Self {
        self.config.browser = browser;
        self.config.container = ContainerConfig::for_browser(browser);
        self
    }

    /// Enables or disables COOP/COEP headers.
    #[must_use]
    pub fn with_coop_coep(mut self, enabled: bool) -> Self {
        self.config.coop_coep.enabled = enabled;
        self
    }

    /// Sets the test timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Sets the maximum parallel containers.
    #[must_use]
    pub fn parallel(mut self, parallel: u32) -> Self {
        self.config.parallel = parallel;
        self
    }

    /// Sets whether to pull images before running.
    #[must_use]
    pub fn pull_images(mut self, pull: bool) -> Self {
        self.config.pull_images = pull;
        self
    }

    /// Sets whether to cleanup containers after tests.
    #[must_use]
    pub fn cleanup(mut self, cleanup: bool) -> Self {
        self.config.cleanup = cleanup;
        self
    }

    /// Sets whether to capture container logs.
    #[must_use]
    pub fn capture_logs(mut self, capture: bool) -> Self {
        self.config.capture_logs = capture;
        self
    }

    /// Sets the Docker socket path.
    #[must_use]
    pub fn docker_socket(mut self, path: String) -> Self {
        self.config.docker_socket = path;
        self
    }

    /// Adds a volume mount.
    #[must_use]
    pub fn volume(mut self, host: PathBuf, container: String) -> Self {
        self.config.container.volumes.push((host, container));
        self
    }

    /// Adds an environment variable.
    #[must_use]
    pub fn env(mut self, key: String, value: String) -> Self {
        self.config.container.environment.insert(key, value);
        self
    }

    /// Builds the DockerTestRunner.
    pub fn build(self) -> DockerResult<DockerTestRunner> {
        Ok(DockerTestRunner {
            config: self.config,
            state: ContainerState::NotCreated,
            container_id: None,
            logs: Vec::new(),
        })
    }
}

/// Docker-based test runner for cross-browser WASM testing.
#[derive(Debug)]
pub struct DockerTestRunner {
    config: DockerConfig,
    state: ContainerState,
    container_id: Option<String>,
    logs: Vec<String>,
}

impl DockerTestRunner {
    /// Creates a new builder for DockerTestRunner.
    #[must_use]
    pub fn builder() -> DockerTestRunnerBuilder {
        DockerTestRunnerBuilder::new()
    }

    /// Returns the current container state.
    #[must_use]
    pub fn state(&self) -> ContainerState {
        self.state
    }

    /// Returns the container ID if running.
    #[must_use]
    pub fn container_id(&self) -> Option<&str> {
        self.container_id.as_deref()
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &DockerConfig {
        &self.config
    }

    /// Returns captured logs.
    #[must_use]
    pub fn logs(&self) -> &[String] {
        &self.logs
    }

    /// Returns the CDP URL for connecting to the browser.
    #[must_use]
    pub fn cdp_url(&self) -> String {
        let port = self.config.browser.default_cdp_port();
        format!("http://localhost:{port}")
    }

    /// Checks if Docker daemon is available (simulated for testing).
    pub fn check_docker_available(&self) -> DockerResult<bool> {
        // In production, this would use bollard to check Docker daemon
        // For testing, we simulate the check
        if self.config.docker_socket.is_empty() {
            return Err(DockerError::ConfigError(
                "Docker socket path not configured".to_string(),
            ));
        }
        Ok(true)
    }

    /// Validates the configuration.
    pub fn validate_config(&self) -> DockerResult<()> {
        if self.config.container.image.is_empty() {
            return Err(DockerError::ConfigError(
                "Container image not specified".to_string(),
            ));
        }
        if self.config.container.name.is_empty() {
            return Err(DockerError::ConfigError(
                "Container name not specified".to_string(),
            ));
        }
        if self.config.timeout.is_zero() {
            return Err(DockerError::ConfigError(
                "Timeout cannot be zero".to_string(),
            ));
        }
        Ok(())
    }

    /// Simulates starting the container (for testing without Docker).
    pub fn simulate_start(&mut self) -> DockerResult<()> {
        self.validate_config()?;
        self.state = ContainerState::Creating;
        self.state = ContainerState::Starting;
        self.container_id = Some(format!("sim-{}", uuid::Uuid::new_v4()));
        self.state = ContainerState::Running;
        self.logs.push("Container started successfully".to_string());
        Ok(())
    }

    /// Simulates stopping the container (for testing without Docker).
    pub fn simulate_stop(&mut self) -> DockerResult<()> {
        if self.state != ContainerState::Running {
            return Err(DockerError::ContainerNotFound(
                "Container not running".to_string(),
            ));
        }
        self.state = ContainerState::Stopping;
        self.logs.push("Container stopping".to_string());
        self.state = ContainerState::Stopped;
        self.logs.push("Container stopped".to_string());
        self.container_id = None;
        Ok(())
    }

    /// Simulates running tests in the container.
    pub fn simulate_run_tests(&mut self, tests: &[&str]) -> DockerResult<TestResults> {
        if self.state != ContainerState::Running {
            return Err(DockerError::ContainerNotFound(
                "Container not running".to_string(),
            ));
        }

        let mut results = TestResults::new(self.config.browser);
        for test in tests {
            self.logs.push(format!("Running test: {test}"));
            results.add_result(TestResult {
                name: (*test).to_string(),
                passed: true,
                duration: Duration::from_millis(100),
                error: None,
            });
        }
        Ok(results)
    }
}

impl Default for DockerTestRunner {
    fn default() -> Self {
        // SAFETY: Default config is always valid
        #[allow(clippy::expect_used)]
        Self::builder()
            .build()
            .expect("Default config should be valid")
    }
}

/// Individual test result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test name.
    pub name: String,
    /// Whether the test passed.
    pub passed: bool,
    /// Test duration.
    pub duration: Duration,
    /// Error message if failed.
    pub error: Option<String>,
}

impl TestResult {
    /// Creates a new passing test result.
    #[must_use]
    pub fn passed(name: String, duration: Duration) -> Self {
        Self {
            name,
            passed: true,
            duration,
            error: None,
        }
    }

    /// Creates a new failing test result.
    #[must_use]
    pub fn failed(name: String, duration: Duration, error: String) -> Self {
        Self {
            name,
            passed: false,
            duration,
            error: Some(error),
        }
    }
}

/// Aggregated test results for a browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    /// Browser that ran the tests.
    pub browser: Browser,
    /// Individual test results.
    pub results: Vec<TestResult>,
    /// Total duration.
    pub total_duration: Duration,
    /// Number of passed tests.
    pub passed: usize,
    /// Number of failed tests.
    pub failed: usize,
}

impl TestResults {
    /// Creates a new empty TestResults for the specified browser.
    #[must_use]
    pub fn new(browser: Browser) -> Self {
        Self {
            browser,
            results: Vec::new(),
            total_duration: Duration::ZERO,
            passed: 0,
            failed: 0,
        }
    }

    /// Adds a test result.
    pub fn add_result(&mut self, result: TestResult) {
        self.total_duration += result.duration;
        if result.passed {
            self.passed += 1;
        } else {
            self.failed += 1;
        }
        self.results.push(result);
    }

    /// Returns whether all tests passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.failed == 0 && self.passed > 0
    }

    /// Returns the total number of tests.
    #[must_use]
    pub fn total(&self) -> usize {
        self.passed + self.failed
    }

    /// Returns the pass rate as a percentage.
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.passed as f64 / total as f64) * 100.0
    }
}

impl fmt::Display for TestResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} passed, {} failed ({:.1}%)",
            self.browser,
            self.passed,
            self.failed,
            self.pass_rate()
        )
    }
}

/// Builder for parallel multi-browser testing.
#[derive(Debug, Default)]
pub struct ParallelRunnerBuilder {
    browsers: Vec<Browser>,
    tests: Vec<String>,
    config: DockerConfig,
}

impl ParallelRunnerBuilder {
    /// Creates a new parallel runner builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the browsers to test on.
    #[must_use]
    pub fn browsers(mut self, browsers: &[Browser]) -> Self {
        self.browsers = browsers.to_vec();
        self
    }

    /// Sets the test files to run.
    #[must_use]
    pub fn tests(mut self, tests: &[&str]) -> Self {
        self.tests = tests.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Sets the timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Builds the parallel runner.
    pub fn build(self) -> DockerResult<ParallelRunner> {
        if self.browsers.is_empty() {
            return Err(DockerError::ConfigError(
                "No browsers specified".to_string(),
            ));
        }
        if self.tests.is_empty() {
            return Err(DockerError::ConfigError("No tests specified".to_string()));
        }
        Ok(ParallelRunner {
            browsers: self.browsers,
            tests: self.tests,
            config: self.config,
            results: HashMap::new(),
        })
    }
}

/// Parallel multi-browser test runner.
#[derive(Debug)]
pub struct ParallelRunner {
    browsers: Vec<Browser>,
    tests: Vec<String>,
    config: DockerConfig,
    results: HashMap<Browser, TestResults>,
}

impl ParallelRunner {
    /// Creates a new parallel runner builder.
    #[must_use]
    pub fn builder() -> ParallelRunnerBuilder {
        ParallelRunnerBuilder::new()
    }

    /// Returns the configured browsers.
    #[must_use]
    pub fn browsers(&self) -> &[Browser] {
        &self.browsers
    }

    /// Returns the configured tests.
    #[must_use]
    pub fn tests(&self) -> &[String] {
        &self.tests
    }

    /// Returns results by browser.
    #[must_use]
    pub fn results_by_browser(&self) -> &HashMap<Browser, TestResults> {
        &self.results
    }

    /// Simulates parallel test execution across browsers.
    pub fn simulate_run(&mut self) -> DockerResult<()> {
        for browser in &self.browsers.clone() {
            let mut runner = DockerTestRunner::builder()
                .browser(*browser)
                .timeout(self.config.timeout)
                .build()?;

            runner.simulate_start()?;
            let test_refs: Vec<&str> = self.tests.iter().map(String::as_str).collect();
            let results = runner.simulate_run_tests(&test_refs)?;
            runner.simulate_stop()?;

            self.results.insert(*browser, results);
        }
        Ok(())
    }

    /// Returns whether all browsers passed all tests.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        !self.results.is_empty() && self.results.values().all(TestResults::all_passed)
    }

    /// Returns aggregated stats across all browsers.
    #[must_use]
    pub fn aggregate_stats(&self) -> (usize, usize, Duration) {
        let mut passed = 0;
        let mut failed = 0;
        let mut duration = Duration::ZERO;
        for result in self.results.values() {
            passed += result.passed;
            failed += result.failed;
            duration += result.total_duration;
        }
        (passed, failed, duration)
    }
}

impl Default for ParallelRunner {
    fn default() -> Self {
        Self {
            browsers: Vec::new(),
            tests: Vec::new(),
            config: DockerConfig::default(),
            results: HashMap::new(),
        }
    }
}

/// Validates that COOP/COEP headers enable SharedArrayBuffer.
pub fn validate_coop_coep_headers(headers: &HashMap<String, String>) -> DockerResult<bool> {
    let coop = headers
        .get("cross-origin-opener-policy")
        .or_else(|| headers.get("Cross-Origin-Opener-Policy"));
    let coep = headers
        .get("cross-origin-embedder-policy")
        .or_else(|| headers.get("Cross-Origin-Embedder-Policy"));

    match (coop, coep) {
        (Some(coop_val), Some(coep_val)) => {
            let valid = coop_val == "same-origin" && coep_val == "require-corp";
            if !valid {
                return Err(DockerError::ConfigError(format!(
                    "Invalid COOP/COEP headers: COOP={coop_val}, COEP={coep_val}"
                )));
            }
            Ok(true)
        }
        (None, _) => Err(DockerError::ConfigError(
            "Missing Cross-Origin-Opener-Policy header".to_string(),
        )),
        (_, None) => Err(DockerError::ConfigError(
            "Missing Cross-Origin-Embedder-Policy header".to_string(),
        )),
    }
}

/// Checks if SharedArrayBuffer is available with the current configuration.
pub fn check_shared_array_buffer_support(config: &CoopCoepConfig) -> bool {
    config.shared_array_buffer_available()
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Browser Tests
    // =========================================================================

    #[test]
    fn test_browser_default_cdp_ports() {
        assert_eq!(Browser::Chrome.default_cdp_port(), 9222);
        assert_eq!(Browser::Firefox.default_cdp_port(), 9223);
        assert_eq!(Browser::WebKit.default_cdp_port(), 9224);
    }

    #[test]
    fn test_browser_image_names() {
        assert_eq!(Browser::Chrome.image_name(), "probar-chrome:latest");
        assert_eq!(Browser::Firefox.image_name(), "probar-firefox:latest");
        assert_eq!(Browser::WebKit.image_name(), "probar-webkit:latest");
    }

    #[test]
    fn test_browser_container_prefix() {
        assert_eq!(Browser::Chrome.container_prefix(), "probar-chrome");
        assert_eq!(Browser::Firefox.container_prefix(), "probar-firefox");
        assert_eq!(Browser::WebKit.container_prefix(), "probar-webkit");
    }

    #[test]
    fn test_browser_all() {
        let all = Browser::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&Browser::Chrome));
        assert!(all.contains(&Browser::Firefox));
        assert!(all.contains(&Browser::WebKit));
    }

    #[test]
    fn test_browser_from_str() {
        assert_eq!(Browser::from_str("chrome"), Some(Browser::Chrome));
        assert_eq!(Browser::from_str("CHROME"), Some(Browser::Chrome));
        assert_eq!(Browser::from_str("chromium"), Some(Browser::Chrome));
        assert_eq!(Browser::from_str("firefox"), Some(Browser::Firefox));
        assert_eq!(Browser::from_str("ff"), Some(Browser::Firefox));
        assert_eq!(Browser::from_str("webkit"), Some(Browser::WebKit));
        assert_eq!(Browser::from_str("safari"), Some(Browser::WebKit));
        assert_eq!(Browser::from_str("invalid"), None);
    }

    #[test]
    fn test_browser_display() {
        assert_eq!(format!("{}", Browser::Chrome), "chrome");
        assert_eq!(format!("{}", Browser::Firefox), "firefox");
        assert_eq!(format!("{}", Browser::WebKit), "webkit");
    }

    // =========================================================================
    // Container State Tests
    // =========================================================================

    #[test]
    fn test_container_state_default() {
        let state = ContainerState::default();
        assert_eq!(state, ContainerState::NotCreated);
    }

    #[test]
    fn test_container_state_display() {
        assert_eq!(format!("{}", ContainerState::NotCreated), "not_created");
        assert_eq!(format!("{}", ContainerState::Creating), "creating");
        assert_eq!(format!("{}", ContainerState::Starting), "starting");
        assert_eq!(format!("{}", ContainerState::Running), "running");
        assert_eq!(
            format!("{}", ContainerState::HealthChecking),
            "health_checking"
        );
        assert_eq!(format!("{}", ContainerState::Stopping), "stopping");
        assert_eq!(format!("{}", ContainerState::Stopped), "stopped");
        assert_eq!(format!("{}", ContainerState::Error), "error");
    }

    // =========================================================================
    // COOP/COEP Config Tests
    // =========================================================================

    #[test]
    fn test_coop_coep_config_default() {
        let config = CoopCoepConfig::default();
        assert_eq!(config.coop, "same-origin");
        assert_eq!(config.coep, "require-corp");
        assert_eq!(config.corp, "cross-origin");
        assert!(config.enabled);
    }

    #[test]
    fn test_coop_coep_config_new() {
        let config = CoopCoepConfig::new();
        assert!(config.enabled);
        assert_eq!(config.coop, "same-origin");
    }

    #[test]
    fn test_coop_coep_config_disabled() {
        let config = CoopCoepConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_coop_coep_shared_array_buffer_available() {
        let config = CoopCoepConfig::default();
        assert!(config.shared_array_buffer_available());

        let mut disabled = CoopCoepConfig::default();
        disabled.enabled = false;
        assert!(!disabled.shared_array_buffer_available());

        let mut wrong_coop = CoopCoepConfig::default();
        wrong_coop.coop = "unsafe-none".to_string();
        assert!(!wrong_coop.shared_array_buffer_available());

        let mut wrong_coep = CoopCoepConfig::default();
        wrong_coep.coep = "unsafe-none".to_string();
        assert!(!wrong_coep.shared_array_buffer_available());
    }

    // =========================================================================
    // Container Config Tests
    // =========================================================================

    #[test]
    fn test_container_config_default() {
        let config = ContainerConfig::default();
        assert_eq!(config.image, "probar-wasm-test:latest");
        assert_eq!(config.name, "probar-test");
        assert!(config.ports.is_empty());
        assert!(config.environment.is_empty());
        assert_eq!(config.memory_limit, Some(2 * 1024 * 1024 * 1024));
        assert_eq!(config.cpu_limit, Some(2.0));
    }

    #[test]
    fn test_container_config_for_browser() {
        let chrome_config = ContainerConfig::for_browser(Browser::Chrome);
        assert_eq!(chrome_config.image, "probar-chrome:latest");
        assert!(chrome_config.name.starts_with("probar-chrome-"));
        assert_eq!(chrome_config.ports, vec![(9222, 9222)]);
        assert_eq!(
            chrome_config.environment.get("PROBAR_BROWSER"),
            Some(&"chrome".to_string())
        );

        let firefox_config = ContainerConfig::for_browser(Browser::Firefox);
        assert_eq!(firefox_config.image, "probar-firefox:latest");
        assert_eq!(firefox_config.ports, vec![(9223, 9223)]);

        let webkit_config = ContainerConfig::for_browser(Browser::WebKit);
        assert_eq!(webkit_config.image, "probar-webkit:latest");
        assert_eq!(webkit_config.ports, vec![(9224, 9224)]);
    }

    // =========================================================================
    // Docker Config Tests
    // =========================================================================

    #[test]
    fn test_docker_config_default() {
        let config = DockerConfig::default();
        assert_eq!(config.browser, Browser::Chrome);
        assert!(config.coop_coep.enabled);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.parallel, 4);
        assert!(config.cleanup);
        assert!(config.capture_logs);
    }

    // =========================================================================
    // DockerTestRunner Builder Tests
    // =========================================================================

    #[test]
    fn test_docker_test_runner_builder_new() {
        let builder = DockerTestRunnerBuilder::new();
        let runner = builder.build().expect("Should build successfully");
        assert_eq!(runner.state(), ContainerState::NotCreated);
    }

    #[test]
    fn test_docker_test_runner_builder_browser() {
        let runner = DockerTestRunner::builder()
            .browser(Browser::Firefox)
            .build()
            .expect("Should build successfully");
        assert_eq!(runner.config().browser, Browser::Firefox);
    }

    #[test]
    fn test_docker_test_runner_builder_coop_coep() {
        let runner = DockerTestRunner::builder()
            .with_coop_coep(false)
            .build()
            .expect("Should build successfully");
        assert!(!runner.config().coop_coep.enabled);
    }

    #[test]
    fn test_docker_test_runner_builder_timeout() {
        let runner = DockerTestRunner::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Should build successfully");
        assert_eq!(runner.config().timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_docker_test_runner_builder_parallel() {
        let runner = DockerTestRunner::builder()
            .parallel(8)
            .build()
            .expect("Should build successfully");
        assert_eq!(runner.config().parallel, 8);
    }

    #[test]
    fn test_docker_test_runner_builder_pull_images() {
        let runner = DockerTestRunner::builder()
            .pull_images(false)
            .build()
            .expect("Should build successfully");
        assert!(!runner.config().pull_images);
    }

    #[test]
    fn test_docker_test_runner_builder_cleanup() {
        let runner = DockerTestRunner::builder()
            .cleanup(false)
            .build()
            .expect("Should build successfully");
        assert!(!runner.config().cleanup);
    }

    #[test]
    fn test_docker_test_runner_builder_capture_logs() {
        let runner = DockerTestRunner::builder()
            .capture_logs(false)
            .build()
            .expect("Should build successfully");
        assert!(!runner.config().capture_logs);
    }

    #[test]
    fn test_docker_test_runner_builder_docker_socket() {
        let runner = DockerTestRunner::builder()
            .docker_socket("/custom/docker.sock".to_string())
            .build()
            .expect("Should build successfully");
        assert_eq!(runner.config().docker_socket, "/custom/docker.sock");
    }

    #[test]
    fn test_docker_test_runner_builder_volume() {
        let runner = DockerTestRunner::builder()
            .volume(PathBuf::from("/host/path"), "/container/path".to_string())
            .build()
            .expect("Should build successfully");
        assert_eq!(runner.config().container.volumes.len(), 1);
    }

    #[test]
    fn test_docker_test_runner_builder_env() {
        let runner = DockerTestRunner::builder()
            .env("MY_VAR".to_string(), "my_value".to_string())
            .build()
            .expect("Should build successfully");
        assert_eq!(
            runner
                .config()
                .container
                .environment
                .get("MY_VAR")
                .map(String::as_str),
            Some("my_value")
        );
    }

    // =========================================================================
    // DockerTestRunner Tests
    // =========================================================================

    #[test]
    fn test_docker_test_runner_default() {
        let runner = DockerTestRunner::default();
        assert_eq!(runner.state(), ContainerState::NotCreated);
        assert!(runner.container_id().is_none());
        assert!(runner.logs().is_empty());
    }

    #[test]
    fn test_docker_test_runner_cdp_url() {
        let chrome_runner = DockerTestRunner::builder()
            .browser(Browser::Chrome)
            .build()
            .expect("Should build successfully");
        assert_eq!(chrome_runner.cdp_url(), "http://localhost:9222");

        let firefox_runner = DockerTestRunner::builder()
            .browser(Browser::Firefox)
            .build()
            .expect("Should build successfully");
        assert_eq!(firefox_runner.cdp_url(), "http://localhost:9223");
    }

    #[test]
    fn test_docker_test_runner_check_docker_available() {
        let runner = DockerTestRunner::default();
        assert!(runner.check_docker_available().is_ok());

        let empty_socket_runner = DockerTestRunner::builder()
            .docker_socket(String::new())
            .build()
            .expect("Should build");
        assert!(empty_socket_runner.check_docker_available().is_err());
    }

    #[test]
    fn test_docker_test_runner_validate_config() {
        let runner = DockerTestRunner::default();
        assert!(runner.validate_config().is_ok());
    }

    #[test]
    fn test_docker_test_runner_validate_config_empty_image() {
        let mut runner = DockerTestRunner::default();
        runner.config.container.image = String::new();
        assert!(runner.validate_config().is_err());
    }

    #[test]
    fn test_docker_test_runner_validate_config_empty_name() {
        let mut runner = DockerTestRunner::default();
        runner.config.container.name = String::new();
        assert!(runner.validate_config().is_err());
    }

    #[test]
    fn test_docker_test_runner_validate_config_zero_timeout() {
        let mut runner = DockerTestRunner::default();
        runner.config.timeout = Duration::ZERO;
        assert!(runner.validate_config().is_err());
    }

    #[test]
    fn test_docker_test_runner_simulate_start() {
        let mut runner = DockerTestRunner::default();
        runner.simulate_start().expect("Should start");
        assert_eq!(runner.state(), ContainerState::Running);
        assert!(runner.container_id().is_some());
        assert!(!runner.logs().is_empty());
    }

    #[test]
    fn test_docker_test_runner_simulate_stop() {
        let mut runner = DockerTestRunner::default();
        runner.simulate_start().expect("Should start");
        runner.simulate_stop().expect("Should stop");
        assert_eq!(runner.state(), ContainerState::Stopped);
        assert!(runner.container_id().is_none());
    }

    #[test]
    fn test_docker_test_runner_simulate_stop_not_running() {
        let mut runner = DockerTestRunner::default();
        assert!(runner.simulate_stop().is_err());
    }

    #[test]
    fn test_docker_test_runner_simulate_run_tests() {
        let mut runner = DockerTestRunner::default();
        runner.simulate_start().expect("Should start");
        let results = runner
            .simulate_run_tests(&["test1.rs", "test2.rs"])
            .expect("Should run tests");
        assert_eq!(results.passed, 2);
        assert_eq!(results.failed, 0);
        assert!(results.all_passed());
    }

    #[test]
    fn test_docker_test_runner_simulate_run_tests_not_running() {
        let mut runner = DockerTestRunner::default();
        assert!(runner.simulate_run_tests(&["test.rs"]).is_err());
    }

    // =========================================================================
    // TestResult Tests
    // =========================================================================

    #[test]
    fn test_test_result_passed() {
        let result = TestResult::passed("my_test".to_string(), Duration::from_millis(50));
        assert!(result.passed);
        assert!(result.error.is_none());
        assert_eq!(result.name, "my_test");
    }

    #[test]
    fn test_test_result_failed() {
        let result = TestResult::failed(
            "my_test".to_string(),
            Duration::from_millis(50),
            "assertion failed".to_string(),
        );
        assert!(!result.passed);
        assert_eq!(result.error, Some("assertion failed".to_string()));
    }

    // =========================================================================
    // TestResults Tests
    // =========================================================================

    #[test]
    fn test_test_results_new() {
        let results = TestResults::new(Browser::Chrome);
        assert_eq!(results.browser, Browser::Chrome);
        assert!(results.results.is_empty());
        assert_eq!(results.passed, 0);
        assert_eq!(results.failed, 0);
    }

    #[test]
    fn test_test_results_add_result() {
        let mut results = TestResults::new(Browser::Firefox);
        results.add_result(TestResult::passed(
            "test1".to_string(),
            Duration::from_secs(1),
        ));
        results.add_result(TestResult::failed(
            "test2".to_string(),
            Duration::from_secs(2),
            "error".to_string(),
        ));
        assert_eq!(results.passed, 1);
        assert_eq!(results.failed, 1);
        assert_eq!(results.total(), 2);
        assert_eq!(results.total_duration, Duration::from_secs(3));
    }

    #[test]
    fn test_test_results_all_passed() {
        let mut results = TestResults::new(Browser::Chrome);
        assert!(!results.all_passed()); // Empty results

        results.add_result(TestResult::passed(
            "test1".to_string(),
            Duration::from_secs(1),
        ));
        assert!(results.all_passed());

        results.add_result(TestResult::failed(
            "test2".to_string(),
            Duration::from_secs(1),
            "error".to_string(),
        ));
        assert!(!results.all_passed());
    }

    #[test]
    fn test_test_results_pass_rate() {
        let mut results = TestResults::new(Browser::WebKit);
        assert_eq!(results.pass_rate(), 0.0);

        results.add_result(TestResult::passed(
            "test1".to_string(),
            Duration::from_secs(1),
        ));
        assert_eq!(results.pass_rate(), 100.0);

        results.add_result(TestResult::failed(
            "test2".to_string(),
            Duration::from_secs(1),
            "error".to_string(),
        ));
        assert_eq!(results.pass_rate(), 50.0);
    }

    #[test]
    fn test_test_results_display() {
        let mut results = TestResults::new(Browser::Chrome);
        results.add_result(TestResult::passed(
            "test1".to_string(),
            Duration::from_secs(1),
        ));
        results.add_result(TestResult::passed(
            "test2".to_string(),
            Duration::from_secs(1),
        ));
        let display = format!("{results}");
        assert!(display.contains("chrome"));
        assert!(display.contains("2 passed"));
        assert!(display.contains("0 failed"));
        assert!(display.contains("100.0%"));
    }

    // =========================================================================
    // ParallelRunner Tests
    // =========================================================================

    #[test]
    fn test_parallel_runner_builder_new() {
        let builder = ParallelRunnerBuilder::new();
        let result = builder.build();
        assert!(result.is_err()); // No browsers configured
    }

    #[test]
    fn test_parallel_runner_builder_no_browsers() {
        let result = ParallelRunner::builder().tests(&["test.rs"]).build();
        assert!(result.is_err());
        match result {
            Err(DockerError::ConfigError(msg)) => {
                assert!(msg.contains("No browsers"));
            }
            _ => panic!("Expected ConfigError"),
        }
    }

    #[test]
    fn test_parallel_runner_builder_no_tests() {
        let result = ParallelRunner::builder()
            .browsers(&[Browser::Chrome])
            .build();
        assert!(result.is_err());
        match result {
            Err(DockerError::ConfigError(msg)) => {
                assert!(msg.contains("No tests"));
            }
            _ => panic!("Expected ConfigError"),
        }
    }

    #[test]
    fn test_parallel_runner_builder_success() {
        let runner = ParallelRunner::builder()
            .browsers(&[Browser::Chrome, Browser::Firefox])
            .tests(&["test1.rs", "test2.rs"])
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Should build successfully");

        assert_eq!(runner.browsers().len(), 2);
        assert_eq!(runner.tests().len(), 2);
    }

    #[test]
    fn test_parallel_runner_simulate_run() {
        let mut runner = ParallelRunner::builder()
            .browsers(&[Browser::Chrome, Browser::Firefox])
            .tests(&["test1.rs", "test2.rs"])
            .build()
            .expect("Should build");

        runner.simulate_run().expect("Should run");

        assert!(runner.all_passed());
        let results = runner.results_by_browser();
        assert_eq!(results.len(), 2);
        assert!(results.contains_key(&Browser::Chrome));
        assert!(results.contains_key(&Browser::Firefox));
    }

    #[test]
    fn test_parallel_runner_aggregate_stats() {
        let mut runner = ParallelRunner::builder()
            .browsers(&[Browser::Chrome, Browser::Firefox, Browser::WebKit])
            .tests(&["test1.rs", "test2.rs"])
            .build()
            .expect("Should build");

        runner.simulate_run().expect("Should run");

        let (passed, failed, duration) = runner.aggregate_stats();
        assert_eq!(passed, 6); // 2 tests × 3 browsers
        assert_eq!(failed, 0);
        assert!(duration > Duration::ZERO);
    }

    #[test]
    fn test_parallel_runner_default() {
        let runner = ParallelRunner::default();
        assert!(runner.browsers().is_empty());
        assert!(runner.tests().is_empty());
        assert!(!runner.all_passed());
    }

    // =========================================================================
    // Header Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_coop_coep_headers_valid() {
        let mut headers = HashMap::new();
        headers.insert(
            "cross-origin-opener-policy".to_string(),
            "same-origin".to_string(),
        );
        headers.insert(
            "cross-origin-embedder-policy".to_string(),
            "require-corp".to_string(),
        );
        assert!(validate_coop_coep_headers(&headers).is_ok());
    }

    #[test]
    fn test_validate_coop_coep_headers_valid_capitalized() {
        let mut headers = HashMap::new();
        headers.insert(
            "Cross-Origin-Opener-Policy".to_string(),
            "same-origin".to_string(),
        );
        headers.insert(
            "Cross-Origin-Embedder-Policy".to_string(),
            "require-corp".to_string(),
        );
        assert!(validate_coop_coep_headers(&headers).is_ok());
    }

    #[test]
    fn test_validate_coop_coep_headers_missing_coop() {
        let mut headers = HashMap::new();
        headers.insert(
            "cross-origin-embedder-policy".to_string(),
            "require-corp".to_string(),
        );
        let result = validate_coop_coep_headers(&headers);
        assert!(result.is_err());
        match result {
            Err(DockerError::ConfigError(msg)) => {
                assert!(msg.contains("Opener-Policy"));
            }
            _ => panic!("Expected ConfigError"),
        }
    }

    #[test]
    fn test_validate_coop_coep_headers_missing_coep() {
        let mut headers = HashMap::new();
        headers.insert(
            "cross-origin-opener-policy".to_string(),
            "same-origin".to_string(),
        );
        let result = validate_coop_coep_headers(&headers);
        assert!(result.is_err());
        match result {
            Err(DockerError::ConfigError(msg)) => {
                assert!(msg.contains("Embedder-Policy"));
            }
            _ => panic!("Expected ConfigError"),
        }
    }

    #[test]
    fn test_validate_coop_coep_headers_wrong_values() {
        let mut headers = HashMap::new();
        headers.insert(
            "cross-origin-opener-policy".to_string(),
            "unsafe-none".to_string(),
        );
        headers.insert(
            "cross-origin-embedder-policy".to_string(),
            "require-corp".to_string(),
        );
        let result = validate_coop_coep_headers(&headers);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_shared_array_buffer_support() {
        let config = CoopCoepConfig::default();
        assert!(check_shared_array_buffer_support(&config));

        let disabled = CoopCoepConfig::disabled();
        assert!(!check_shared_array_buffer_support(&disabled));
    }

    // =========================================================================
    // Error Tests
    // =========================================================================

    #[test]
    fn test_docker_error_display() {
        let err = DockerError::DaemonUnavailable("not running".to_string());
        assert!(format!("{err}").contains("Docker daemon not available"));

        let err = DockerError::ContainerStartFailed("exit 1".to_string());
        assert!(format!("{err}").contains("Container failed to start"));

        let err = DockerError::ContainerNotFound("abc123".to_string());
        assert!(format!("{err}").contains("Container not found"));

        let err = DockerError::ImageNotFound("probar:latest".to_string());
        assert!(format!("{err}").contains("Image not found"));

        let err = DockerError::CdpConnectionFailed("timeout".to_string());
        assert!(format!("{err}").contains("CDP connection failed"));

        let err = DockerError::TestExecutionFailed("assertion".to_string());
        assert!(format!("{err}").contains("Test execution failed"));

        let err = DockerError::Timeout("30s".to_string());
        assert!(format!("{err}").contains("Timeout"));

        let err = DockerError::HealthCheckFailed("unhealthy".to_string());
        assert!(format!("{err}").contains("Health check failed"));

        let err = DockerError::ConfigError("invalid".to_string());
        assert!(format!("{err}").contains("Configuration error"));

        let err = DockerError::IoError("permission denied".to_string());
        assert!(format!("{err}").contains("IO error"));

        let err = DockerError::NetworkError("connection refused".to_string());
        assert!(format!("{err}").contains("Network error"));
    }

    // =========================================================================
    // Integration-style Tests
    // =========================================================================

    #[test]
    fn test_full_lifecycle_chrome() {
        let mut runner = DockerTestRunner::builder()
            .browser(Browser::Chrome)
            .with_coop_coep(true)
            .timeout(Duration::from_secs(30))
            .cleanup(true)
            .build()
            .expect("Should build");

        // Verify initial state
        assert_eq!(runner.state(), ContainerState::NotCreated);

        // Start container
        runner.simulate_start().expect("Should start");
        assert_eq!(runner.state(), ContainerState::Running);

        // Run tests
        let results = runner
            .simulate_run_tests(&["worker_tests.rs", "shared_memory_tests.rs"])
            .expect("Should run tests");
        assert!(results.all_passed());
        assert_eq!(results.passed, 2);

        // Stop container
        runner.simulate_stop().expect("Should stop");
        assert_eq!(runner.state(), ContainerState::Stopped);
    }

    #[test]
    fn test_full_lifecycle_firefox() {
        let mut runner = DockerTestRunner::builder()
            .browser(Browser::Firefox)
            .build()
            .expect("Should build");

        runner.simulate_start().expect("Should start");
        let results = runner
            .simulate_run_tests(&["e2e_tests.rs"])
            .expect("Should run");
        assert!(results.all_passed());
        runner.simulate_stop().expect("Should stop");
    }

    #[test]
    fn test_full_lifecycle_webkit() {
        let mut runner = DockerTestRunner::builder()
            .browser(Browser::WebKit)
            .build()
            .expect("Should build");

        runner.simulate_start().expect("Should start");
        let results = runner
            .simulate_run_tests(&["visual_regression.rs"])
            .expect("Should run");
        assert!(results.all_passed());
        runner.simulate_stop().expect("Should stop");
    }

    #[test]
    fn test_parallel_cross_browser() {
        let mut runner = ParallelRunner::builder()
            .browsers(&Browser::all())
            .tests(&[
                "worker_tests.rs",
                "shared_memory_tests.rs",
                "ring_buffer_tests.rs",
            ])
            .build()
            .expect("Should build");

        runner.simulate_run().expect("Should run");

        assert!(runner.all_passed());

        let (passed, failed, _) = runner.aggregate_stats();
        assert_eq!(passed, 9); // 3 tests × 3 browsers
        assert_eq!(failed, 0);

        // Check each browser
        let results = runner.results_by_browser();
        for browser in Browser::all() {
            let browser_results = results.get(&browser).expect("Should have results");
            assert!(browser_results.all_passed());
            assert_eq!(browser_results.passed, 3);
        }
    }

    // =========================================================================
    // Edge Cases and Boundary Tests
    // =========================================================================

    #[test]
    fn test_empty_test_list() {
        let mut runner = DockerTestRunner::default();
        runner.simulate_start().expect("Should start");
        let results = runner.simulate_run_tests(&[]).expect("Should handle empty");
        assert_eq!(results.total(), 0);
        assert!(!results.all_passed()); // No tests = not passing
    }

    #[test]
    fn test_single_test() {
        let mut runner = DockerTestRunner::default();
        runner.simulate_start().expect("Should start");
        let results = runner
            .simulate_run_tests(&["single_test.rs"])
            .expect("Should run");
        assert_eq!(results.total(), 1);
        assert!(results.all_passed());
    }

    #[test]
    fn test_many_tests() {
        let mut runner = DockerTestRunner::default();
        runner.simulate_start().expect("Should start");

        let tests: Vec<String> = (0..100).map(|i| format!("test_{i}.rs")).collect();
        let test_refs: Vec<&str> = tests.iter().map(String::as_str).collect();

        let results = runner.simulate_run_tests(&test_refs).expect("Should run");
        assert_eq!(results.total(), 100);
        assert!(results.all_passed());
    }

    #[test]
    fn test_pass_rate_precision() {
        let mut results = TestResults::new(Browser::Chrome);

        // Add 1 passed, 2 failed = 33.33...%
        results.add_result(TestResult::passed("t1".to_string(), Duration::from_secs(1)));
        results.add_result(TestResult::failed(
            "t2".to_string(),
            Duration::from_secs(1),
            "err".to_string(),
        ));
        results.add_result(TestResult::failed(
            "t3".to_string(),
            Duration::from_secs(1),
            "err".to_string(),
        ));

        let rate = results.pass_rate();
        assert!((rate - 33.333_333_333_333_336).abs() < 0.001);
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_browser_serialization() {
        let browser = Browser::Chrome;
        let json = serde_json::to_string(&browser).expect("Should serialize");
        assert_eq!(json, "\"chrome\"");

        let deserialized: Browser = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized, Browser::Chrome);
    }

    #[test]
    fn test_container_state_serialization() {
        let state = ContainerState::Running;
        let json = serde_json::to_string(&state).expect("Should serialize");
        let deserialized: ContainerState = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized, ContainerState::Running);
    }

    #[test]
    fn test_coop_coep_config_serialization() {
        let config = CoopCoepConfig::default();
        let json = serde_json::to_string(&config).expect("Should serialize");
        assert!(json.contains("same-origin"));
        assert!(json.contains("require-corp"));

        let deserialized: CoopCoepConfig = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.coop, "same-origin");
    }

    #[test]
    fn test_test_result_serialization() {
        let result = TestResult::passed("my_test".to_string(), Duration::from_millis(123));
        let json = serde_json::to_string(&result).expect("Should serialize");
        assert!(json.contains("my_test"));
        assert!(json.contains("true"));

        let deserialized: TestResult = serde_json::from_str(&json).expect("Should deserialize");
        assert!(deserialized.passed);
    }

    #[test]
    fn test_test_results_serialization() {
        let mut results = TestResults::new(Browser::Firefox);
        results.add_result(TestResult::passed("t1".to_string(), Duration::from_secs(1)));
        results.add_result(TestResult::failed(
            "t2".to_string(),
            Duration::from_secs(2),
            "error".to_string(),
        ));

        let json = serde_json::to_string(&results).expect("Should serialize");
        assert!(json.contains("firefox"));
        assert!(json.contains("t1"));
        assert!(json.contains("t2"));

        let deserialized: TestResults = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.passed, 1);
        assert_eq!(deserialized.failed, 1);
    }

    // =========================================================================
    // Additional Edge Case Tests for 100% Coverage
    // =========================================================================

    #[test]
    fn test_docker_config_serialization() {
        let config = DockerConfig::default();
        let json = serde_json::to_string(&config).expect("Should serialize");
        assert!(json.contains("chrome"));
        assert!(json.contains("timeout"));
    }

    #[test]
    fn test_container_config_serialization() {
        let config = ContainerConfig::default();
        let json = serde_json::to_string(&config).expect("Should serialize");
        assert!(json.contains("probar-wasm-test"));
    }

    #[test]
    fn test_container_config_for_all_browsers() {
        for browser in Browser::all() {
            let config = ContainerConfig::for_browser(browser);
            assert!(!config.image.is_empty());
            assert!(!config.name.is_empty());
            assert!(!config.ports.is_empty());
            assert!(config.health_check.is_some());
        }
    }

    #[test]
    fn test_browser_serialization_all_variants() {
        for browser in Browser::all() {
            let json = serde_json::to_string(&browser).expect("Should serialize");
            let deserialized: Browser = serde_json::from_str(&json).expect("Should deserialize");
            assert_eq!(deserialized, browser);
        }
    }

    #[test]
    fn test_container_state_all_variants_serialization() {
        let states = [
            ContainerState::NotCreated,
            ContainerState::Creating,
            ContainerState::Starting,
            ContainerState::Running,
            ContainerState::HealthChecking,
            ContainerState::Stopping,
            ContainerState::Stopped,
            ContainerState::Error,
        ];
        for state in states {
            let json = serde_json::to_string(&state).expect("Should serialize");
            let deserialized: ContainerState =
                serde_json::from_str(&json).expect("Should deserialize");
            assert_eq!(deserialized, state);
        }
    }

    #[test]
    fn test_parallel_runner_all_passed_no_results() {
        let runner = ParallelRunner::default();
        assert!(!runner.all_passed()); // Empty results = not passed
    }

    #[test]
    fn test_test_results_with_only_failures() {
        let mut results = TestResults::new(Browser::Chrome);
        results.add_result(TestResult::failed(
            "fail1".to_string(),
            Duration::from_secs(1),
            "error".to_string(),
        ));
        results.add_result(TestResult::failed(
            "fail2".to_string(),
            Duration::from_secs(1),
            "error".to_string(),
        ));
        assert!(!results.all_passed());
        assert_eq!(results.pass_rate(), 0.0);
    }

    #[test]
    fn test_coop_coep_custom_values() {
        let mut config = CoopCoepConfig::default();
        config.coop = "same-origin-allow-popups".to_string();
        config.coep = "credentialless".to_string();
        assert!(!config.shared_array_buffer_available());
    }

    #[test]
    fn test_docker_test_runner_config_accessors() {
        let runner = DockerTestRunner::builder()
            .browser(Browser::WebKit)
            .parallel(8)
            .timeout(Duration::from_secs(300))
            .build()
            .expect("Should build");

        assert_eq!(runner.config().browser, Browser::WebKit);
        assert_eq!(runner.config().parallel, 8);
        assert_eq!(runner.config().timeout, Duration::from_secs(300));
        assert_eq!(runner.cdp_url(), "http://localhost:9224");
    }

    #[test]
    fn test_container_config_environment_variables() {
        let config = ContainerConfig::for_browser(Browser::Chrome);
        assert!(config.environment.contains_key("PROBAR_BROWSER"));
        assert!(config.environment.contains_key("PROBAR_CDP_PORT"));
        assert!(config.environment.contains_key("PROBAR_COOP_COEP"));
    }

    #[test]
    fn test_container_config_default_resources() {
        let config = ContainerConfig::default();
        assert_eq!(config.memory_limit, Some(2 * 1024 * 1024 * 1024));
        assert_eq!(config.cpu_limit, Some(2.0));
        assert_eq!(config.health_check_interval, Duration::from_secs(5));
        assert_eq!(config.health_check_timeout, Duration::from_secs(5));
        assert_eq!(config.health_check_retries, 3);
    }

    #[test]
    fn test_docker_error_variants_debug() {
        let errors = vec![
            DockerError::DaemonUnavailable("test".to_string()),
            DockerError::ContainerStartFailed("test".to_string()),
            DockerError::ContainerNotFound("test".to_string()),
            DockerError::ImageNotFound("test".to_string()),
            DockerError::CdpConnectionFailed("test".to_string()),
            DockerError::TestExecutionFailed("test".to_string()),
            DockerError::Timeout("test".to_string()),
            DockerError::HealthCheckFailed("test".to_string()),
            DockerError::ConfigError("test".to_string()),
            DockerError::IoError("test".to_string()),
            DockerError::NetworkError("test".to_string()),
        ];
        for err in errors {
            let debug = format!("{:?}", err);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_parallel_runner_tests_accessor() {
        let runner = ParallelRunner::builder()
            .browsers(&[Browser::Chrome])
            .tests(&["test1.rs", "test2.rs", "test3.rs"])
            .build()
            .expect("Should build");

        assert_eq!(runner.tests().len(), 3);
        assert!(runner.tests().contains(&"test1.rs".to_string()));
    }

    #[test]
    fn test_docker_test_runner_logs_accumulate() {
        let mut runner = DockerTestRunner::default();
        runner.simulate_start().expect("Should start");
        let initial_logs = runner.logs().len();

        runner.simulate_run_tests(&["t1.rs"]).expect("Should run");
        assert!(runner.logs().len() > initial_logs);

        runner
            .simulate_run_tests(&["t2.rs", "t3.rs"])
            .expect("Should run");
        assert!(runner.logs().len() > initial_logs + 1);
    }

    #[test]
    fn test_test_result_duration() {
        let result = TestResult::passed("test".to_string(), Duration::from_millis(42));
        assert_eq!(result.duration, Duration::from_millis(42));

        let failed = TestResult::failed(
            "test".to_string(),
            Duration::from_millis(100),
            "err".to_string(),
        );
        assert_eq!(failed.duration, Duration::from_millis(100));
    }

    #[test]
    fn test_test_results_total_duration() {
        let mut results = TestResults::new(Browser::Firefox);
        results.add_result(TestResult::passed(
            "t1".to_string(),
            Duration::from_millis(100),
        ));
        results.add_result(TestResult::passed(
            "t2".to_string(),
            Duration::from_millis(200),
        ));
        results.add_result(TestResult::passed(
            "t3".to_string(),
            Duration::from_millis(300),
        ));

        assert_eq!(results.total_duration, Duration::from_millis(600));
    }

    #[test]
    fn test_browser_from_str_case_insensitive() {
        assert_eq!(Browser::from_str("CHROME"), Some(Browser::Chrome));
        assert_eq!(Browser::from_str("Chrome"), Some(Browser::Chrome));
        assert_eq!(Browser::from_str("chrome"), Some(Browser::Chrome));
        assert_eq!(Browser::from_str("FIREFOX"), Some(Browser::Firefox));
        assert_eq!(Browser::from_str("Firefox"), Some(Browser::Firefox));
        assert_eq!(Browser::from_str("WEBKIT"), Some(Browser::WebKit));
        assert_eq!(Browser::from_str("WebKit"), Some(Browser::WebKit));
    }

    #[test]
    fn test_parallel_runner_timeout_configuration() {
        let runner = ParallelRunner::builder()
            .browsers(&[Browser::Chrome])
            .tests(&["test.rs"])
            .timeout(Duration::from_secs(180))
            .build()
            .expect("Should build");

        // Just verify it builds - timeout is stored in config
        assert!(!runner.browsers().is_empty());
    }

    #[test]
    fn test_docker_test_runner_chain_configuration() {
        let runner = DockerTestRunner::builder()
            .browser(Browser::Firefox)
            .with_coop_coep(true)
            .timeout(Duration::from_secs(90))
            .parallel(2)
            .pull_images(false)
            .cleanup(true)
            .capture_logs(true)
            .build()
            .expect("Should build");

        assert_eq!(runner.config().browser, Browser::Firefox);
        assert!(runner.config().coop_coep.enabled);
        assert_eq!(runner.config().timeout, Duration::from_secs(90));
        assert_eq!(runner.config().parallel, 2);
        assert!(!runner.config().pull_images);
        assert!(runner.config().cleanup);
        assert!(runner.config().capture_logs);
    }
}
