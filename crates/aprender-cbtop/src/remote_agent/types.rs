//! Type definitions for remote SSH/headless agent integration.

use std::collections::HashMap;
use std::time::Instant;

/// Result type for remote agent operations
pub type RemoteResult<T> = Result<T, RemoteError>;

/// Errors that can occur during remote operations
#[derive(Debug, Clone, PartialEq)]
pub enum RemoteError {
    /// SSH connection failed
    ConnectionFailed { host: String, reason: String },
    /// Authentication failed
    AuthenticationFailed { host: String },
    /// Command execution failed
    CommandFailed {
        host: String,
        exit_code: i32,
        stderr: String,
    },
    /// Timeout waiting for response
    Timeout { host: String, timeout_ms: u64 },
    /// Host not found in pool
    HostNotFound { host: String },
    /// All hosts failed
    AllHostsFailed { failures: Vec<String> },
    /// Invalid configuration
    InvalidConfig { reason: String },
    /// Result aggregation failed
    AggregationFailed { reason: String },
}

impl std::fmt::Display for RemoteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed { host, reason } => {
                write!(f, "Connection to {} failed: {}", host, reason)
            }
            Self::AuthenticationFailed { host } => {
                write!(f, "Authentication failed for {}", host)
            }
            Self::CommandFailed {
                host,
                exit_code,
                stderr,
            } => {
                write!(
                    f,
                    "Command failed on {} (exit {}): {}",
                    host, exit_code, stderr
                )
            }
            Self::Timeout { host, timeout_ms } => {
                write!(f, "Timeout after {}ms waiting for {}", timeout_ms, host)
            }
            Self::HostNotFound { host } => {
                write!(f, "Host {} not found in pool", host)
            }
            Self::AllHostsFailed { failures } => {
                write!(f, "All hosts failed: {:?}", failures)
            }
            Self::InvalidConfig { reason } => {
                write!(f, "Invalid configuration: {}", reason)
            }
            Self::AggregationFailed { reason } => {
                write!(f, "Result aggregation failed: {}", reason)
            }
        }
    }
}

impl std::error::Error for RemoteError {}

/// SSH authentication method
#[derive(Debug, Clone)]
pub enum AuthMethod {
    /// SSH key-based authentication
    Key {
        /// Path to private key file
        key_path: String,
        /// Optional passphrase (from environment or keyring, never stored)
        passphrase_env: Option<String>,
    },
    /// SSH agent forwarding
    Agent,
    /// Password from environment variable (never stored in config)
    PasswordEnv {
        /// Environment variable name containing password
        env_var: String,
    },
}

impl Default for AuthMethod {
    fn default() -> Self {
        Self::Agent
    }
}

/// Remote host configuration
#[derive(Debug, Clone)]
pub struct HostConfig {
    /// Hostname or IP address
    pub host: String,
    /// SSH port (default: 22)
    pub port: u16,
    /// Username for SSH connection
    pub username: String,
    /// Authentication method
    pub auth: AuthMethod,
    /// Connection timeout in milliseconds
    pub connect_timeout_ms: u64,
    /// Command execution timeout in milliseconds
    pub command_timeout_ms: u64,
    /// Host architecture (x86_64, aarch64, etc.)
    pub architecture: Option<String>,
    /// Host labels for grouping
    pub labels: HashMap<String, String>,
}

impl HostConfig {
    /// Create a new host configuration
    pub fn new(host: impl Into<String>, username: impl Into<String>) -> Self {
        Self {
            host: host.into(),
            port: 22,
            username: username.into(),
            auth: AuthMethod::default(),
            connect_timeout_ms: 10_000,
            command_timeout_ms: 60_000,
            architecture: None,
            labels: HashMap::new(),
        }
    }

    /// Set SSH port
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set authentication method
    pub fn with_auth(mut self, auth: AuthMethod) -> Self {
        self.auth = auth;
        self
    }

    /// Set connection timeout
    pub fn with_connect_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.connect_timeout_ms = timeout_ms;
        self
    }

    /// Set command timeout
    pub fn with_command_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.command_timeout_ms = timeout_ms;
        self
    }

    /// Set architecture
    pub fn with_architecture(mut self, arch: impl Into<String>) -> Self {
        self.architecture = Some(arch.into());
        self
    }

    /// Add a label
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }
}

/// Health status of a remote host
#[derive(Debug, Clone, PartialEq)]
pub enum HostHealth {
    /// Host is healthy and responding
    Healthy,
    /// Host is degraded (slow responses)
    Degraded { latency_ms: u64 },
    /// Host is unreachable
    Unreachable { last_error: String },
    /// Host status unknown (never checked)
    Unknown,
}

impl Default for HostHealth {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Remote host state
#[derive(Debug, Clone)]
pub struct HostState {
    /// Host configuration
    pub config: HostConfig,
    /// Current health status
    pub health: HostHealth,
    /// Last successful connection time
    pub last_success: Option<Instant>,
    /// Consecutive failure count
    pub failure_count: u32,
    /// Total commands executed
    pub commands_executed: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
}

impl HostState {
    /// Create new host state from configuration
    pub fn new(config: HostConfig) -> Self {
        Self {
            config,
            health: HostHealth::Unknown,
            last_success: None,
            failure_count: 0,
            commands_executed: 0,
            avg_latency_ms: 0.0,
        }
    }

    /// Check if host is available for commands
    pub fn is_available(&self) -> bool {
        matches!(
            self.health,
            HostHealth::Healthy | HostHealth::Degraded { .. } | HostHealth::Unknown
        )
    }

    /// Record a successful command execution
    pub fn record_success(&mut self, latency_ms: u64) {
        self.last_success = Some(Instant::now());
        self.failure_count = 0;
        self.commands_executed += 1;

        // Update running average
        let n = self.commands_executed as f64;
        self.avg_latency_ms = self.avg_latency_ms * ((n - 1.0) / n) + (latency_ms as f64) / n;

        // Update health based on latency
        self.health = if latency_ms > 5000 {
            HostHealth::Degraded { latency_ms }
        } else {
            HostHealth::Healthy
        };
    }

    /// Record a failed command
    pub fn record_failure(&mut self, error: &str) {
        self.failure_count += 1;
        if self.failure_count >= 3 {
            self.health = HostHealth::Unreachable {
                last_error: error.to_string(),
            };
        }
    }
}

/// Result from a single remote command
#[derive(Debug, Clone)]
pub struct CommandResult {
    /// Host that executed the command
    pub host: String,
    /// Exit code (0 = success)
    pub exit_code: i32,
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Execution time in milliseconds
    pub duration_ms: u64,
}

impl CommandResult {
    /// Check if command succeeded
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }
}

/// Aggregated benchmark result from multiple hosts
#[derive(Debug, Clone)]
pub struct AggregatedResult {
    /// Individual results from each host
    pub host_results: Vec<HostBenchmark>,
    /// Aggregated throughput (ops/sec, geometric mean)
    pub throughput_geomean: f64,
    /// Aggregated latency p50 (arithmetic mean)
    pub latency_p50_mean_us: f64,
    /// Aggregated latency p99 (max across hosts)
    pub latency_p99_max_us: f64,
    /// Number of successful hosts
    pub hosts_succeeded: usize,
    /// Number of failed hosts
    pub hosts_failed: usize,
    /// Total collection time
    pub collection_time_ms: u64,
}

impl AggregatedResult {
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.hosts_succeeded + self.hosts_failed;
        if total == 0 {
            0.0
        } else {
            self.hosts_succeeded as f64 / total as f64
        }
    }
}

/// Benchmark result from a single host
#[derive(Debug, Clone)]
pub struct HostBenchmark {
    /// Host identifier
    pub host: String,
    /// Host architecture
    pub architecture: String,
    /// Throughput in operations per second
    pub throughput_ops: f64,
    /// Latency p50 in microseconds
    pub latency_p50_us: f64,
    /// Latency p99 in microseconds
    pub latency_p99_us: f64,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// GPU utilization (0.0-1.0)
    pub gpu_utilization: Option<f64>,
    /// Collection timestamp
    pub timestamp_ns: u64,
}

/// Strategy for aggregating results from multiple hosts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Use geometric mean for throughput, arithmetic for latency
    GeometricMean,
    /// Use median values
    Median,
    /// Use minimum values (pessimistic)
    Minimum,
    /// Use maximum values (optimistic)
    Maximum,
}

impl Default for AggregationStrategy {
    fn default() -> Self {
        Self::GeometricMean
    }
}

/// Remote agent configuration
#[derive(Debug, Clone)]
pub struct RemoteAgentConfig {
    /// Maximum concurrent connections
    pub max_concurrent: usize,
    /// Retry count for failed commands
    pub retry_count: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Health check interval in seconds
    pub health_check_interval_sec: u64,
    /// Aggregation strategy
    pub aggregation: AggregationStrategy,
    /// Path to cbtop binary on remote hosts
    pub remote_binary_path: String,
}

impl Default for RemoteAgentConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 10,
            retry_count: 3,
            retry_delay_ms: 1000,
            health_check_interval_sec: 60,
            aggregation: AggregationStrategy::default(),
            remote_binary_path: "/usr/local/bin/cbtop".to_string(),
        }
    }
}

/// Default retry delay in milliseconds
pub const DEFAULT_RETRY_DELAY_MS: u64 = 1000;

/// Default maximum concurrent connections
pub const DEFAULT_MAX_CONCURRENT: usize = 10;

/// Default health check interval in seconds
pub const DEFAULT_HEALTH_CHECK_INTERVAL_SEC: u64 = 60;
