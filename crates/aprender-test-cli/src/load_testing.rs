//! Load Testing Module
//!
//! Implements the Load Testing features from the spec (Section D):
//!
//! - D.1: Command Interface (basic, scenario, ramp-up)
//! - D.2: Load Test Scenario Format (YAML)
//! - D.3: Load Test Results Format
//! - D.4: Implementation Requirements

#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::format_push_string)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::unwrap_used)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// =============================================================================
// D.4 Implementation Requirements
// =============================================================================

/// Load test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestConfig {
    /// Target URL to test
    pub target_url: String,
    /// User configuration
    pub users: UserConfig,
    /// Test duration in seconds
    pub duration_secs: u64,
    /// Optional scenario file path
    pub scenario: Option<PathBuf>,
    /// Output format
    pub output: LoadTestOutputFormat,
}

impl LoadTestConfig {
    /// Create a new load test config
    pub fn new(target_url: &str, users: u32, duration_secs: u64) -> Self {
        Self {
            target_url: target_url.to_string(),
            users: UserConfig::Fixed(users),
            duration_secs,
            scenario: None,
            output: LoadTestOutputFormat::Text,
        }
    }

    /// Create a ramp-up config
    pub fn with_ramp(
        target_url: &str,
        start_users: u32,
        end_users: u32,
        ramp_secs: u64,
        duration_secs: u64,
    ) -> Self {
        Self {
            target_url: target_url.to_string(),
            users: UserConfig::Ramp {
                start: start_users,
                end: end_users,
                ramp_secs,
            },
            duration_secs,
            scenario: None,
            output: LoadTestOutputFormat::Text,
        }
    }

    /// Create from scenario file
    pub fn from_scenario(scenario_path: PathBuf) -> Self {
        Self {
            target_url: String::new(),
            users: UserConfig::Fixed(1),
            duration_secs: 0,
            scenario: Some(scenario_path),
            output: LoadTestOutputFormat::Text,
        }
    }
}

/// User configuration for load test
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum UserConfig {
    /// Fixed number of concurrent users
    Fixed(u32),
    /// Ramp from start to end users
    Ramp {
        /// Starting user count
        start: u32,
        /// Ending user count
        end: u32,
        /// Ramp duration in seconds
        ramp_secs: u64,
    },
}

impl UserConfig {
    /// Get user count at a given time (seconds into test)
    pub fn users_at(&self, elapsed_secs: u64) -> u32 {
        match self {
            Self::Fixed(users) => *users,
            Self::Ramp {
                start,
                end,
                ramp_secs,
            } => {
                if elapsed_secs >= *ramp_secs {
                    *end
                } else {
                    let progress = elapsed_secs as f64 / *ramp_secs as f64;
                    let range = (*end as i64 - *start as i64) as f64;
                    (*start as f64 + range * progress) as u32
                }
            }
        }
    }
}

/// Output format for load test results
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
pub enum LoadTestOutputFormat {
    /// Human-readable text
    #[default]
    Text,
    /// JSON output
    Json,
    /// CSV output
    Csv,
}

// =============================================================================
// D.2 Load Test Scenario Format
// =============================================================================

/// Load test scenario (from YAML)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestScenario {
    /// Scenario name
    pub name: String,
    /// Description
    pub description: String,
    /// Citation reference
    #[serde(default)]
    pub citation: Option<String>,
    /// Test stages
    pub stages: Vec<LoadTestStage>,
    /// Request definitions
    pub requests: Vec<LoadTestRequest>,
}

impl LoadTestScenario {
    /// Create a new scenario
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            citation: None,
            stages: Vec::new(),
            requests: Vec::new(),
        }
    }

    /// Add a stage
    pub fn add_stage(&mut self, stage: LoadTestStage) {
        self.stages.push(stage);
    }

    /// Add a request
    pub fn add_request(&mut self, request: LoadTestRequest) {
        self.requests.push(request);
    }

    /// Calculate total duration in seconds
    pub fn total_duration_secs(&self) -> u64 {
        self.stages.iter().map(|s| s.duration_secs).sum()
    }

    /// Load from YAML string
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        serde_yaml_ng::from_str(yaml).map_err(|e| format!("Failed to parse YAML: {}", e))
    }

    /// Load from file
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_yaml(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Save to file
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let content = serde_yaml_ng::to_string(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, content)
    }
}

/// A stage in the load test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestStage {
    /// Stage name
    pub name: String,
    /// Duration in seconds
    pub duration_secs: u64,
    /// Starting users (for ramp stages)
    pub users_start: u32,
    /// Ending users (for ramp stages, same as start for steady)
    pub users_end: u32,
}

impl LoadTestStage {
    /// Create a steady-state stage
    pub fn steady(name: &str, duration_secs: u64, users: u32) -> Self {
        Self {
            name: name.to_string(),
            duration_secs,
            users_start: users,
            users_end: users,
        }
    }

    /// Create a ramp stage
    pub fn ramp(name: &str, duration_secs: u64, start_users: u32, end_users: u32) -> Self {
        Self {
            name: name.to_string(),
            duration_secs,
            users_start: start_users,
            users_end: end_users,
        }
    }

    /// Check if this is a ramp stage
    pub fn is_ramp(&self) -> bool {
        self.users_start != self.users_end
    }

    /// Get users at time offset within stage
    pub fn users_at(&self, offset_secs: u64) -> u32 {
        if !self.is_ramp() || self.duration_secs == 0 {
            return self.users_start;
        }
        let progress = (offset_secs as f64 / self.duration_secs as f64).min(1.0);
        let range = (self.users_end as i64 - self.users_start as i64) as f64;
        (self.users_start as f64 + range * progress) as u32
    }
}

/// A request definition in the scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestRequest {
    /// Request name
    pub name: String,
    /// HTTP method
    pub method: HttpMethod,
    /// Request path (relative to base URL)
    pub path: String,
    /// Request weight (probability of selection)
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// Headers to send
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Request body
    #[serde(default)]
    pub body: Option<String>,
    /// Assertions to check
    #[serde(default)]
    pub assertions: Vec<LoadTestAssertion>,
}

fn default_weight() -> f64 {
    1.0
}

impl LoadTestRequest {
    /// Create a GET request
    pub fn get(name: &str, path: &str) -> Self {
        Self {
            name: name.to_string(),
            method: HttpMethod::Get,
            path: path.to_string(),
            weight: 1.0,
            headers: HashMap::new(),
            body: None,
            assertions: Vec::new(),
        }
    }

    /// Create a POST request
    pub fn post(name: &str, path: &str, body: Option<String>) -> Self {
        Self {
            name: name.to_string(),
            method: HttpMethod::Post,
            path: path.to_string(),
            weight: 1.0,
            headers: HashMap::new(),
            body,
            assertions: Vec::new(),
        }
    }

    /// Add an assertion
    pub fn with_assertion(mut self, assertion: LoadTestAssertion) -> Self {
        self.assertions.push(assertion);
        self
    }

    /// Set weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

/// HTTP methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum HttpMethod {
    /// HTTP GET method
    #[default]
    Get,
    /// HTTP POST method
    Post,
    /// HTTP PUT method
    Put,
    /// HTTP DELETE method
    Delete,
    /// HTTP PATCH method
    Patch,
    /// HTTP HEAD method
    Head,
    /// HTTP OPTIONS method
    Options,
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Get => write!(f, "GET"),
            Self::Post => write!(f, "POST"),
            Self::Put => write!(f, "PUT"),
            Self::Delete => write!(f, "DELETE"),
            Self::Patch => write!(f, "PATCH"),
            Self::Head => write!(f, "HEAD"),
            Self::Options => write!(f, "OPTIONS"),
        }
    }
}

/// Assertion for load test
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LoadTestAssertion {
    /// Status code check
    Status {
        /// Expected status code
        expected: u16,
    },
    /// Latency percentile check
    LatencyPercentile {
        /// Percentile (0-100)
        percentile: u8,
        /// Max latency in milliseconds
        max_ms: u64,
    },
    /// Header value check
    Header {
        /// Header name
        name: String,
        /// Expected value
        expected: String,
    },
    /// Body contains check
    BodyContains {
        /// Expected substring
        substring: String,
    },
}

impl LoadTestAssertion {
    /// Create status assertion
    pub fn status(expected: u16) -> Self {
        Self::Status { expected }
    }

    /// Create latency percentile assertion
    pub fn latency_p95(max_ms: u64) -> Self {
        Self::LatencyPercentile {
            percentile: 95,
            max_ms,
        }
    }

    /// Create latency percentile assertion with custom percentile
    pub fn latency_percentile(percentile: u8, max_ms: u64) -> Self {
        Self::LatencyPercentile { percentile, max_ms }
    }

    /// Create header assertion
    pub fn header(name: &str, expected: &str) -> Self {
        Self::Header {
            name: name.to_string(),
            expected: expected.to_string(),
        }
    }

    /// Create body contains assertion
    pub fn body_contains(substring: &str) -> Self {
        Self::BodyContains {
            substring: substring.to_string(),
        }
    }

    /// Get description of this assertion
    pub fn description(&self) -> String {
        match self {
            Self::Status { expected } => format!("status == {}", expected),
            Self::LatencyPercentile { percentile, max_ms } => {
                format!("latency_p{} < {}ms", percentile, max_ms)
            }
            Self::Header { name, expected } => format!("{} == {}", name, expected),
            Self::BodyContains { substring } => format!("body contains '{}'", substring),
        }
    }
}

// =============================================================================
// D.3 Load Test Results
// =============================================================================

/// Load test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestResult {
    /// Scenario name
    pub scenario_name: String,
    /// Total duration in seconds
    pub duration_secs: u64,
    /// Total requests made
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Per-endpoint statistics
    pub endpoint_stats: Vec<EndpointStats>,
    /// Peak throughput (req/s)
    pub peak_throughput: f64,
    /// Peak throughput time (seconds into test)
    pub peak_throughput_time: u64,
    /// Average throughput (req/s)
    pub avg_throughput: f64,
    /// Resource usage stats
    pub resource_usage: ResourceUsage,
    /// Assertion results
    pub assertion_results: Vec<AssertionResult>,
    /// Errors encountered
    pub errors: Vec<LoadTestError>,
}

impl LoadTestResult {
    /// Create a new result
    pub fn new(scenario_name: &str) -> Self {
        Self {
            scenario_name: scenario_name.to_string(),
            duration_secs: 0,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            endpoint_stats: Vec::new(),
            peak_throughput: 0.0,
            peak_throughput_time: 0,
            avg_throughput: 0.0,
            resource_usage: ResourceUsage::default(),
            assertion_results: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Calculate error rate as percentage
    pub fn error_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.failed_requests as f64 / self.total_requests as f64) * 100.0
        }
    }

    /// Check if all assertions passed
    pub fn all_assertions_passed(&self) -> bool {
        self.assertion_results.iter().all(|r| r.passed)
    }

    /// Count passed assertions
    pub fn passed_assertions(&self) -> usize {
        self.assertion_results.iter().filter(|r| r.passed).count()
    }

    /// Count failed assertions
    pub fn failed_assertions(&self) -> usize {
        self.assertion_results.iter().filter(|r| !r.passed).count()
    }
}

/// Per-endpoint statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointStats {
    /// Endpoint name
    pub name: String,
    /// Request count
    pub count: u64,
    /// p50 latency in ms
    pub p50_ms: u64,
    /// p95 latency in ms
    pub p95_ms: u64,
    /// p99 latency in ms
    pub p99_ms: u64,
    /// Error count
    pub errors: u64,
    /// Min latency in ms
    pub min_ms: u64,
    /// Max latency in ms
    pub max_ms: u64,
    /// Avg latency in ms
    pub avg_ms: u64,
}

impl EndpointStats {
    /// Create new endpoint stats
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            count: 0,
            p50_ms: 0,
            p95_ms: 0,
            p99_ms: 0,
            errors: 0,
            min_ms: u64::MAX,
            max_ms: 0,
            avg_ms: 0,
        }
    }

    /// Create from latency samples
    pub fn from_samples(name: &str, samples: &[u64], errors: u64) -> Self {
        if samples.is_empty() {
            return Self::new(name);
        }

        let mut sorted = samples.to_vec();
        sorted.sort_unstable();

        let count = sorted.len() as u64;
        let sum: u64 = sorted.iter().sum();

        Self {
            name: name.to_string(),
            count,
            p50_ms: percentile(&sorted, 50),
            p95_ms: percentile(&sorted, 95),
            p99_ms: percentile(&sorted, 99),
            errors,
            min_ms: *sorted.first().unwrap_or(&0),
            max_ms: *sorted.last().unwrap_or(&0),
            avg_ms: sum / count,
        }
    }
}

/// Calculate percentile from sorted samples
fn percentile(sorted: &[u64], p: u8) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p as f64 / 100.0) * (sorted.len() - 1) as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceUsage {
    /// Average CPU usage percentage
    pub avg_cpu_percent: f64,
    /// Peak CPU usage percentage
    pub peak_cpu_percent: f64,
    /// Average memory in MB
    pub avg_memory_mb: u64,
    /// Peak memory in MB
    pub peak_memory_mb: u64,
}

/// Assertion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssertionResult {
    /// Endpoint name
    pub endpoint: String,
    /// Assertion description
    pub assertion: String,
    /// Whether it passed
    pub passed: bool,
    /// Actual value
    pub actual: String,
    /// Expected value
    pub expected: String,
}

impl AssertionResult {
    /// Create a passed assertion
    pub fn passed(endpoint: &str, assertion: &str, actual: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            assertion: assertion.to_string(),
            passed: true,
            actual: actual.to_string(),
            expected: actual.to_string(),
        }
    }

    /// Create a failed assertion
    pub fn failed(endpoint: &str, assertion: &str, expected: &str, actual: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            assertion: assertion.to_string(),
            passed: false,
            actual: actual.to_string(),
            expected: expected.to_string(),
        }
    }
}

/// Load test error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestError {
    /// Endpoint where error occurred
    pub endpoint: String,
    /// Error kind
    pub kind: LoadTestErrorKind,
    /// Error message
    pub message: String,
    /// Time of error (seconds into test)
    pub time_secs: u64,
}

impl LoadTestError {
    /// Create a new error
    pub fn new(endpoint: &str, kind: LoadTestErrorKind, message: &str, time_secs: u64) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            kind,
            message: message.to_string(),
            time_secs,
        }
    }
}

/// Load test error kinds
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LoadTestErrorKind {
    /// Connection failed
    Connection,
    /// Request timeout
    Timeout,
    /// HTTP error (non-2xx status)
    HttpError,
    /// DNS resolution failed
    DnsError,
    /// TLS/SSL error
    TlsError,
    /// Other error
    Other,
}

impl std::fmt::Display for LoadTestErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connection => write!(f, "Connection"),
            Self::Timeout => write!(f, "Timeout"),
            Self::HttpError => write!(f, "HTTP Error"),
            Self::DnsError => write!(f, "DNS Error"),
            Self::TlsError => write!(f, "TLS Error"),
            Self::Other => write!(f, "Other"),
        }
    }
}

// =============================================================================
// Latency Histogram for percentile tracking
// =============================================================================

/// Latency histogram for tracking percentiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyHistogram {
    /// Buckets (ms -> count)
    buckets: Vec<u64>,
    /// Bucket size in ms
    bucket_size: u64,
    /// Total samples
    count: u64,
    /// Sum of all samples
    sum: u64,
    /// Min value
    min: u64,
    /// Max value
    max: u64,
}

impl LatencyHistogram {
    /// Create a new histogram with bucket size in ms
    pub fn new(bucket_size: u64) -> Self {
        Self {
            buckets: vec![0; 1000], // Up to bucket_size * 1000 ms
            bucket_size,
            count: 0,
            sum: 0,
            min: u64::MAX,
            max: 0,
        }
    }

    /// Record a latency sample in ms
    pub fn record(&mut self, latency_ms: u64) {
        let bucket = (latency_ms / self.bucket_size) as usize;
        if bucket < self.buckets.len() {
            self.buckets[bucket] += 1;
        } else {
            // Overflow bucket (last)
            *self.buckets.last_mut().unwrap() += 1;
        }
        self.count += 1;
        self.sum += latency_ms;
        self.min = self.min.min(latency_ms);
        self.max = self.max.max(latency_ms);
    }

    /// Get percentile value
    pub fn percentile(&self, p: u8) -> u64 {
        if self.count == 0 {
            return 0;
        }

        let target = ((p as f64 / 100.0) * self.count as f64) as u64;
        let mut cumulative = 0u64;

        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return (i as u64 + 1) * self.bucket_size;
            }
        }

        self.max
    }

    /// Get mean latency
    pub fn mean(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            self.sum / self.count
        }
    }

    /// Get sample count
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get min value
    pub fn min(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            self.min
        }
    }

    /// Get max value
    pub fn max(&self) -> u64 {
        self.max
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new(1) // 1ms buckets
    }
}

// =============================================================================
// Rendering
// =============================================================================

/// Render load test results as text
pub fn render_load_test_report(result: &LoadTestResult) -> String {
    let mut output = String::new();

    output.push_str(&format!("LOAD TEST RESULTS: {}\n", result.scenario_name));
    output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    output.push_str(&format!(
        "Duration: {}s │ Total Requests: {} │ Failed: {} ({:.2}%)\n\n",
        result.duration_secs,
        result.total_requests,
        result.failed_requests,
        result.error_rate()
    ));

    // Request Statistics table
    output.push_str("Request Statistics:\n");
    output.push_str("┌─────────────────┬─────────┬─────────┬─────────┬─────────┬─────────┐\n");
    output.push_str("│ Endpoint        │ Count   │ p50     │ p95     │ p99     │ Errors  │\n");
    output.push_str("├─────────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤\n");

    for stat in &result.endpoint_stats {
        output.push_str(&format!(
            "│ {:<15} │ {:>7} │ {:>5}ms │ {:>5}ms │ {:>5}ms │ {:>7} │\n",
            truncate(&stat.name, 15),
            stat.count,
            stat.p50_ms,
            stat.p95_ms,
            stat.p99_ms,
            stat.errors
        ));
    }
    output.push_str("└─────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘\n\n");

    // Throughput
    output.push_str("Throughput:\n");
    output.push_str(&format!(
        "  Peak: {:.0} req/s at t={}s\n",
        result.peak_throughput, result.peak_throughput_time
    ));
    output.push_str(&format!("  Avg:  {:.0} req/s\n\n", result.avg_throughput));

    // Resource Usage
    output.push_str("Resource Usage:\n");
    output.push_str(&format!(
        "  Server CPU: avg {:.0}%, peak {:.0}%\n",
        result.resource_usage.avg_cpu_percent, result.resource_usage.peak_cpu_percent
    ));
    output.push_str(&format!(
        "  Server Memory: avg {}MB, peak {}MB\n\n",
        result.resource_usage.avg_memory_mb, result.resource_usage.peak_memory_mb
    ));

    // Assertions
    output.push_str("Assertions:\n");
    for assertion in &result.assertion_results {
        let symbol = if assertion.passed { "✓" } else { "✗" };
        output.push_str(&format!(
            "  {} {} {} (actual: {})\n",
            symbol, assertion.endpoint, assertion.assertion, assertion.actual
        ));
    }

    output
}

/// Render load test results as JSON
pub fn render_load_test_json(result: &LoadTestResult) -> String {
    serde_json::to_string_pretty(result).unwrap_or_else(|_| "{}".to_string())
}

/// Truncate string to max length
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}…", &s[..max_len - 1])
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_load_test_config_new() {
        let config = LoadTestConfig::new("http://localhost:8080", 100, 30);
        assert_eq!(config.target_url, "http://localhost:8080");
        assert_eq!(config.users, UserConfig::Fixed(100));
        assert_eq!(config.duration_secs, 30);
    }

    #[test]
    fn test_load_test_config_ramp() {
        let config = LoadTestConfig::with_ramp("http://localhost:8080", 1, 100, 60, 120);
        assert_eq!(
            config.users,
            UserConfig::Ramp {
                start: 1,
                end: 100,
                ramp_secs: 60
            }
        );
    }

    #[test]
    fn test_user_config_fixed() {
        let config = UserConfig::Fixed(50);
        assert_eq!(config.users_at(0), 50);
        assert_eq!(config.users_at(30), 50);
        assert_eq!(config.users_at(100), 50);
    }

    #[test]
    fn test_user_config_ramp() {
        let config = UserConfig::Ramp {
            start: 10,
            end: 100,
            ramp_secs: 90,
        };
        assert_eq!(config.users_at(0), 10);
        assert_eq!(config.users_at(45), 55); // Midpoint
        assert_eq!(config.users_at(90), 100);
        assert_eq!(config.users_at(100), 100); // Past ramp
    }

    #[test]
    fn test_scenario_new() {
        let mut scenario = LoadTestScenario::new("Test", "A test scenario");
        scenario.add_stage(LoadTestStage::steady("warmup", 30, 10));
        scenario.add_stage(LoadTestStage::ramp("ramp", 60, 10, 100));

        assert_eq!(scenario.stages.len(), 2);
        assert_eq!(scenario.total_duration_secs(), 90);
    }

    #[test]
    fn test_load_test_stage_steady() {
        let stage = LoadTestStage::steady("steady", 60, 50);
        assert!(!stage.is_ramp());
        assert_eq!(stage.users_at(0), 50);
        assert_eq!(stage.users_at(30), 50);
    }

    #[test]
    fn test_load_test_stage_ramp() {
        let stage = LoadTestStage::ramp("ramp", 60, 10, 100);
        assert!(stage.is_ramp());
        assert_eq!(stage.users_at(0), 10);
        assert_eq!(stage.users_at(30), 55);
        assert_eq!(stage.users_at(60), 100);
    }

    #[test]
    fn test_load_test_request_get() {
        let request = LoadTestRequest::get("home", "/")
            .with_assertion(LoadTestAssertion::status(200))
            .with_weight(2.0);

        assert_eq!(request.name, "home");
        assert_eq!(request.method, HttpMethod::Get);
        assert_eq!(request.weight, 2.0);
        assert_eq!(request.assertions.len(), 1);
    }

    #[test]
    fn test_http_method_display() {
        assert_eq!(HttpMethod::Get.to_string(), "GET");
        assert_eq!(HttpMethod::Post.to_string(), "POST");
        assert_eq!(HttpMethod::Delete.to_string(), "DELETE");
    }

    #[test]
    fn test_assertion_description() {
        assert_eq!(
            LoadTestAssertion::status(200).description(),
            "status == 200"
        );
        assert_eq!(
            LoadTestAssertion::latency_p95(100).description(),
            "latency_p95 < 100ms"
        );
        assert_eq!(
            LoadTestAssertion::header("content-type", "application/json").description(),
            "content-type == application/json"
        );
    }

    #[test]
    fn test_load_test_result_error_rate() {
        let mut result = LoadTestResult::new("Test");
        result.total_requests = 1000;
        result.failed_requests = 10;

        assert!((result.error_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_load_test_result_assertions() {
        let mut result = LoadTestResult::new("Test");
        result
            .assertion_results
            .push(AssertionResult::passed("ep1", "status", "200"));
        result
            .assertion_results
            .push(AssertionResult::passed("ep2", "status", "200"));
        result
            .assertion_results
            .push(AssertionResult::failed("ep3", "latency", "100ms", "200ms"));

        assert_eq!(result.passed_assertions(), 2);
        assert_eq!(result.failed_assertions(), 1);
        assert!(!result.all_assertions_passed());
    }

    #[test]
    fn test_endpoint_stats_from_samples() {
        let samples = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let stats = EndpointStats::from_samples("test", &samples, 2);

        assert_eq!(stats.count, 10);
        assert_eq!(stats.min_ms, 10);
        assert_eq!(stats.max_ms, 100);
        assert_eq!(stats.avg_ms, 55);
        assert_eq!(stats.errors, 2);
    }

    #[test]
    fn test_percentile() {
        let sorted = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(percentile(&sorted, 50), 5);
        assert_eq!(percentile(&sorted, 90), 9);
        assert_eq!(percentile(&sorted, 100), 10);
    }

    #[test]
    fn test_latency_histogram() {
        let mut hist = LatencyHistogram::new(10);

        for i in 0..100 {
            hist.record(i);
        }

        assert_eq!(hist.count(), 100);
        assert_eq!(hist.min(), 0);
        assert_eq!(hist.max(), 99);
    }

    #[test]
    fn test_latency_histogram_percentile() {
        let mut hist = LatencyHistogram::new(1);

        for i in 1..=100 {
            hist.record(i);
        }

        // Percentiles from histogram (bucketed, so may be slightly off)
        assert!(hist.percentile(50) >= 45 && hist.percentile(50) <= 55);
        assert!(hist.percentile(95) >= 90);
    }

    #[test]
    fn test_load_test_error() {
        let error = LoadTestError::new(
            "endpoint",
            LoadTestErrorKind::Timeout,
            "Request timed out",
            45,
        );
        assert_eq!(error.endpoint, "endpoint");
        assert_eq!(error.kind, LoadTestErrorKind::Timeout);
        assert_eq!(error.time_secs, 45);
    }

    #[test]
    fn test_error_kind_display() {
        assert_eq!(LoadTestErrorKind::Connection.to_string(), "Connection");
        assert_eq!(LoadTestErrorKind::Timeout.to_string(), "Timeout");
        assert_eq!(LoadTestErrorKind::HttpError.to_string(), "HTTP Error");
    }

    #[test]
    fn test_render_load_test_report() {
        let mut result = LoadTestResult::new("Test Scenario");
        result.duration_secs = 60;
        result.total_requests = 1000;
        result.successful_requests = 990;
        result.failed_requests = 10;
        result.avg_throughput = 16.67;
        result.peak_throughput = 25.0;
        result.peak_throughput_time = 30;

        result.endpoint_stats.push(EndpointStats {
            name: "homepage".to_string(),
            count: 500,
            p50_ms: 12,
            p95_ms: 45,
            p99_ms: 89,
            errors: 5,
            min_ms: 5,
            max_ms: 120,
            avg_ms: 25,
        });

        result
            .assertion_results
            .push(AssertionResult::passed("homepage", "status == 200", "200"));

        let report = render_load_test_report(&result);
        assert!(report.contains("Test Scenario"));
        assert!(report.contains("Duration: 60s"));
        assert!(report.contains("homepage"));
        assert!(report.contains("✓"));
    }

    #[test]
    fn test_render_load_test_json() {
        let result = LoadTestResult::new("JSON Test");
        let json = render_load_test_json(&result);
        assert!(json.contains("JSON Test"));
        assert!(json.contains("scenario_name"));
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is very long", 10), "this is v…");
    }

    #[test]
    fn test_scenario_yaml_roundtrip() {
        let mut scenario = LoadTestScenario::new("YAML Test", "Testing YAML serialization");
        scenario.add_stage(LoadTestStage::steady("warmup", 10, 5));
        scenario.add_request(LoadTestRequest::get("home", "/"));

        let yaml = serde_yaml_ng::to_string(&scenario).unwrap();
        let parsed: LoadTestScenario = serde_yaml_ng::from_str(&yaml).unwrap();

        assert_eq!(parsed.name, "YAML Test");
        assert_eq!(parsed.stages.len(), 1);
        assert_eq!(parsed.requests.len(), 1);
    }

    #[test]
    fn test_load_test_config_from_scenario() {
        let config = LoadTestConfig::from_scenario(PathBuf::from("scenario.yaml"));
        assert!(config.target_url.is_empty());
        assert_eq!(config.users, UserConfig::Fixed(1));
        assert_eq!(config.duration_secs, 0);
        assert_eq!(config.scenario, Some(PathBuf::from("scenario.yaml")));
    }

    #[test]
    fn test_load_test_assertion_body_contains() {
        let assertion = LoadTestAssertion::body_contains("success");
        assert_eq!(assertion.description(), "body contains 'success'");
    }

    #[test]
    fn test_load_test_assertion_latency_percentile() {
        let assertion = LoadTestAssertion::latency_percentile(99, 500);
        assert_eq!(assertion.description(), "latency_p99 < 500ms");
    }

    #[test]
    fn test_load_test_result_error_rate_zero() {
        let result = LoadTestResult::new("Zero Requests");
        assert_eq!(result.error_rate(), 0.0);
    }

    #[test]
    fn test_load_test_result_all_assertions_passed_empty() {
        let result = LoadTestResult::new("No Assertions");
        assert!(result.all_assertions_passed());
    }

    #[test]
    fn test_endpoint_stats_new() {
        let stats = EndpointStats::new("api/v1");
        assert_eq!(stats.name, "api/v1");
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn test_endpoint_stats_from_empty_samples() {
        let stats = EndpointStats::from_samples("empty", &[], 0);
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn test_percentile_empty() {
        let empty: Vec<u64> = vec![];
        assert_eq!(percentile(&empty, 50), 0);
    }

    #[test]
    fn test_http_method_all_variants() {
        assert_eq!(HttpMethod::Get.to_string(), "GET");
        assert_eq!(HttpMethod::Post.to_string(), "POST");
        assert_eq!(HttpMethod::Put.to_string(), "PUT");
        assert_eq!(HttpMethod::Delete.to_string(), "DELETE");
        assert_eq!(HttpMethod::Patch.to_string(), "PATCH");
        assert_eq!(HttpMethod::Head.to_string(), "HEAD");
        assert_eq!(HttpMethod::Options.to_string(), "OPTIONS");
    }

    #[test]
    fn test_error_kind_all_variants() {
        assert_eq!(LoadTestErrorKind::Connection.to_string(), "Connection");
        assert_eq!(LoadTestErrorKind::Timeout.to_string(), "Timeout");
        assert_eq!(LoadTestErrorKind::HttpError.to_string(), "HTTP Error");
        assert_eq!(LoadTestErrorKind::DnsError.to_string(), "DNS Error");
        assert_eq!(LoadTestErrorKind::TlsError.to_string(), "TLS Error");
        assert_eq!(LoadTestErrorKind::Other.to_string(), "Other");
    }

    #[test]
    fn test_load_test_request_post() {
        let request = LoadTestRequest::post(
            "create_user",
            "/users",
            Some(r#"{"name": "test"}"#.to_string()),
        );
        assert_eq!(request.method, HttpMethod::Post);
        assert!(request.body.is_some());
    }

    #[test]
    fn test_load_test_request_post_no_body() {
        let request = LoadTestRequest::post("create_empty", "/items", None);
        assert_eq!(request.method, HttpMethod::Post);
        assert!(request.body.is_none());
    }

    #[test]
    fn test_latency_histogram_empty_percentile() {
        let hist = LatencyHistogram::new(10);
        assert_eq!(hist.percentile(50), 0);
    }

    #[test]
    fn test_latency_histogram_empty_stats() {
        let hist = LatencyHistogram::new(10);
        assert_eq!(hist.count(), 0);
        assert_eq!(hist.min(), 0); // min() returns 0 for empty histogram
        assert_eq!(hist.max(), 0);
    }

    #[test]
    fn test_latency_histogram_overflow() {
        let mut hist = LatencyHistogram::new(10); // 10 buckets
        hist.record(10000); // Way beyond 10 buckets
        assert_eq!(hist.count(), 1);
    }

    #[test]
    fn test_assertion_result_passed() {
        let result = AssertionResult::passed("endpoint", "status == 200", "200");
        assert!(result.passed);
        assert_eq!(result.endpoint, "endpoint");
    }

    #[test]
    fn test_assertion_result_failed() {
        let result = AssertionResult::failed("endpoint", "latency_p95 < 100ms", "100ms", "150ms");
        assert!(!result.passed);
        assert!(!result.expected.is_empty());
        assert!(!result.actual.is_empty());
    }

    #[test]
    fn test_resource_usage_default() {
        let usage = ResourceUsage::default();
        assert_eq!(usage.avg_cpu_percent, 0.0);
        assert_eq!(usage.peak_cpu_percent, 0.0);
    }

    #[test]
    fn test_load_test_output_format_default() {
        let format = LoadTestOutputFormat::default();
        assert!(matches!(format, LoadTestOutputFormat::Text));
    }
}
