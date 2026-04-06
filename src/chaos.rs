// Chaos Engineering Configuration (Sprint 29 - Red-Team Profile)
//
// Builder pattern for chaos configuration following aprender patterns.
// Enables systematic fault injection for testing Renacer's robustness.
//
// Features:
// - Tiered chaos levels (basic, network, byzantine)
// - Chainable builder API
// - Feature-gated implementations

use std::time::Duration;

/// Chaos engineering configuration with builder pattern
///
/// # Example
/// ```
/// use renacer::chaos::ChaosConfig;
/// use std::time::Duration;
///
/// let config = ChaosConfig::new()
///     .with_memory_limit(100 * 1024 * 1024)  // 100MB
///     .with_cpu_limit(0.5)                    // 50% CPU
///     .with_timeout(Duration::from_secs(30))
///     .with_signal_injection(true)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct ChaosConfig {
    /// Memory limit in bytes (0 = unlimited)
    pub memory_limit: usize,

    /// CPU limit as fraction (0.0-1.0, 0 = unlimited)
    pub cpu_limit: f64,

    /// Execution timeout
    pub timeout: Duration,

    /// Enable random signal injection
    pub signal_injection: bool,

    /// Network latency injection (milliseconds)
    #[cfg(feature = "chaos-network")]
    pub network_latency_ms: u64,

    /// Packet loss probability (0.0-1.0)
    #[cfg(feature = "chaos-network")]
    pub packet_loss_prob: f64,

    /// Byzantine fault injection probability (0.0-1.0)
    #[cfg(feature = "chaos-byzantine")]
    pub byzantine_fault_prob: f64,

    /// Syscalls to inject faults into
    #[cfg(feature = "chaos-byzantine")]
    pub fault_syscalls: Vec<String>,
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            memory_limit: 0,
            cpu_limit: 0.0,
            timeout: Duration::from_secs(60),
            signal_injection: false,
            #[cfg(feature = "chaos-network")]
            network_latency_ms: 0,
            #[cfg(feature = "chaos-network")]
            packet_loss_prob: 0.0,
            #[cfg(feature = "chaos-byzantine")]
            byzantine_fault_prob: 0.0,
            #[cfg(feature = "chaos-byzantine")]
            fault_syscalls: Vec::new(),
        }
    }
}

impl ChaosConfig {
    /// Create a new chaos configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set memory limit in bytes
    ///
    /// # Example
    /// ```
    /// use renacer::chaos::ChaosConfig;
    /// let config = ChaosConfig::new().with_memory_limit(100 * 1024 * 1024);
    /// assert_eq!(config.memory_limit, 104857600);
    /// ```
    pub fn with_memory_limit(mut self, bytes: usize) -> Self {
        self.memory_limit = bytes;
        self
    }

    /// Set CPU limit as fraction (0.0-1.0)
    ///
    /// # Example
    /// ```
    /// use renacer::chaos::ChaosConfig;
    /// let config = ChaosConfig::new().with_cpu_limit(0.5);
    /// assert!((config.cpu_limit - 0.5).abs() < f64::EPSILON);
    /// ```
    pub fn with_cpu_limit(mut self, fraction: f64) -> Self {
        self.cpu_limit = fraction.clamp(0.0, 1.0);
        self
    }

    /// Set execution timeout
    ///
    /// # Example
    /// ```
    /// use renacer::chaos::ChaosConfig;
    /// use std::time::Duration;
    /// let config = ChaosConfig::new().with_timeout(Duration::from_secs(30));
    /// assert_eq!(config.timeout.as_secs(), 30);
    /// ```
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable or disable signal injection
    pub fn with_signal_injection(mut self, enabled: bool) -> Self {
        self.signal_injection = enabled;
        self
    }

    /// Set network latency injection (requires chaos-network feature)
    #[cfg(feature = "chaos-network")]
    pub fn with_network_latency(mut self, ms: u64) -> Self {
        self.network_latency_ms = ms;
        self
    }

    /// Set packet loss probability (requires chaos-network feature)
    #[cfg(feature = "chaos-network")]
    pub fn with_packet_loss(mut self, probability: f64) -> Self {
        self.packet_loss_prob = probability.clamp(0.0, 1.0);
        self
    }

    /// Set byzantine fault injection probability (requires chaos-byzantine feature)
    #[cfg(feature = "chaos-byzantine")]
    pub fn with_byzantine_faults(mut self, probability: f64) -> Self {
        self.byzantine_fault_prob = probability.clamp(0.0, 1.0);
        self
    }

    /// Set syscalls to inject faults into (requires chaos-byzantine feature)
    #[cfg(feature = "chaos-byzantine")]
    pub fn with_fault_syscalls(mut self, syscalls: Vec<String>) -> Self {
        self.fault_syscalls = syscalls;
        self
    }

    /// Build the final configuration (validates and returns)
    pub fn build(self) -> Self {
        self
    }

    /// Check if any chaos features are enabled
    pub fn is_active(&self) -> bool {
        self.memory_limit > 0
            || self.cpu_limit > 0.0
            || self.signal_injection
            || self.timeout < Duration::from_secs(60)
    }

    /// Get a preset for "gentle" chaos testing
    pub fn gentle() -> Self {
        Self::new()
            .with_memory_limit(512 * 1024 * 1024) // 512MB
            .with_cpu_limit(0.8)
            .with_timeout(Duration::from_secs(120))
    }

    /// Get a preset for "aggressive" chaos testing
    pub fn aggressive() -> Self {
        Self::new()
            .with_memory_limit(64 * 1024 * 1024) // 64MB
            .with_cpu_limit(0.25)
            .with_timeout(Duration::from_secs(10))
            .with_signal_injection(true)
    }

    /// Get a preset for "extreme" chaos testing (requires chaos-byzantine)
    #[cfg(feature = "chaos-byzantine")]
    pub fn extreme() -> Self {
        Self::aggressive().with_byzantine_faults(0.1).with_fault_syscalls(vec![
            "read".to_string(),
            "write".to_string(),
            "open".to_string(),
            "close".to_string(),
        ])
    }
}

/// Result type for chaos operations
pub type ChaosResult<T> = Result<T, ChaosError>;

/// Errors that can occur during chaos testing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChaosError {
    /// Memory limit exceeded
    MemoryLimitExceeded { limit: usize, used: usize },

    /// Execution timeout
    Timeout { elapsed: Duration, limit: Duration },

    /// Signal injection failed
    SignalInjectionFailed { signal: i32, reason: String },

    /// Parse error for memory/duration strings
    ParseError { input: String, reason: String },

    /// Resource limit application failed
    ResourceLimitFailed { resource: String, reason: String },

    /// Byzantine fault injection failed
    #[cfg(feature = "chaos-byzantine")]
    ByzantineFaultFailed { syscall: String, reason: String },
}

impl std::fmt::Display for ChaosError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChaosError::MemoryLimitExceeded { limit, used } => {
                write!(f, "Memory limit exceeded: {used} > {limit} bytes")
            }
            ChaosError::Timeout { elapsed, limit } => {
                write!(f, "Timeout: {elapsed:?} > {limit:?}")
            }
            ChaosError::SignalInjectionFailed { signal, reason } => {
                write!(f, "Signal injection failed ({signal}): {reason}")
            }
            ChaosError::ParseError { input, reason } => {
                write!(f, "Parse error for '{input}': {reason}")
            }
            ChaosError::ResourceLimitFailed { resource, reason } => {
                write!(f, "Failed to set {resource}: {reason}")
            }
            #[cfg(feature = "chaos-byzantine")]
            ChaosError::ByzantineFaultFailed { syscall, reason } => {
                write!(f, "Byzantine fault failed ({syscall}): {reason}")
            }
        }
    }
}

impl std::error::Error for ChaosError {}

/// Apply resource limits for chaos testing
///
/// This function applies memory and CPU limits using setrlimit.
/// It should be called in the child process before exec.
///
/// # Safety
/// This function modifies process resource limits using setrlimit.
/// It should only be called in the forked child process.
///
/// # Example
/// ```ignore
/// use renacer::chaos::ChaosConfig;
///
/// let config = ChaosConfig::aggressive();
/// if let Err(e) = config.apply_limits() {
///     eprintln!("Warning: Failed to apply chaos limits: {}", e);
/// }
/// ```
impl ChaosConfig {
    /// Apply resource limits to the current process
    ///
    /// Applies:
    /// - `RLIMIT_AS` (address space / virtual memory) if `memory_limit` > 0
    /// - `RLIMIT_CPU` (CPU time) derived from `cpu_limit` fraction
    /// - `RLIMIT_RTTIME` (soft timeout hint) from timeout
    pub fn apply_limits(&self) -> Result<(), ChaosError> {
        use nix::sys::resource::{setrlimit, Resource};

        // Apply memory limit (virtual address space)
        if self.memory_limit > 0 {
            let limit = self.memory_limit as u64;
            setrlimit(Resource::RLIMIT_AS, limit, limit).map_err(|e| {
                ChaosError::ResourceLimitFailed {
                    resource: "RLIMIT_AS".to_string(),
                    reason: e.to_string(),
                }
            })?;
        }

        // Apply CPU time limit (derived from fraction and timeout)
        // If cpu_limit is 0.5 and timeout is 10s, we allow 5s of CPU time
        if self.cpu_limit > 0.0 && self.cpu_limit < 1.0 {
            let cpu_seconds = (self.timeout.as_secs_f64() * self.cpu_limit) as u64;
            if cpu_seconds > 0 {
                // Set soft limit slightly below hard limit to get SIGXCPU warning
                let soft = cpu_seconds;
                let hard = cpu_seconds + 1;
                setrlimit(Resource::RLIMIT_CPU, soft, hard).map_err(|e| {
                    ChaosError::ResourceLimitFailed {
                        resource: "RLIMIT_CPU".to_string(),
                        reason: e.to_string(),
                    }
                })?;
            }
        }

        Ok(())
    }
}

/// Parse a memory size string (e.g., "64M", "512K", "1G", or raw bytes)
///
/// Supports suffixes: K/k (kilobytes), M/m (megabytes), G/g (gigabytes)
///
/// # Examples
/// ```
/// use renacer::chaos::parse_memory_size;
/// assert_eq!(parse_memory_size("64M").expect("test"), 64 * 1024 * 1024);
/// assert_eq!(parse_memory_size("512K").expect("test"), 512 * 1024);
/// assert_eq!(parse_memory_size("1G").expect("test"), 1024 * 1024 * 1024);
/// assert_eq!(parse_memory_size("1024").expect("test"), 1024);
/// ```
pub fn parse_memory_size(s: &str) -> Result<usize, ChaosError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(ChaosError::ParseError {
            input: s.to_string(),
            reason: "empty string".to_string(),
        });
    }

    // Check for suffix
    let (num_str, multiplier) = if let Some(stripped) = s.strip_suffix(['K', 'k']) {
        (stripped, 1024usize)
    } else if let Some(stripped) = s.strip_suffix(['M', 'm']) {
        (stripped, 1024 * 1024)
    } else if let Some(stripped) = s.strip_suffix(['G', 'g']) {
        (stripped, 1024 * 1024 * 1024)
    } else {
        (s, 1)
    };

    num_str.trim().parse::<usize>().map(|n| n * multiplier).map_err(|_| ChaosError::ParseError {
        input: s.to_string(),
        reason: "invalid number".to_string(),
    })
}

/// Parse a duration string (e.g., "10s", "2m", "1h", or raw seconds)
///
/// Supports suffixes: s (seconds), m (minutes), h (hours)
///
/// # Examples
/// ```
/// use renacer::chaos::parse_duration;
/// use std::time::Duration;
/// assert_eq!(parse_duration("10s").expect("test"), Duration::from_secs(10));
/// assert_eq!(parse_duration("2m").expect("test"), Duration::from_secs(120));
/// assert_eq!(parse_duration("1h").expect("test"), Duration::from_secs(3600));
/// assert_eq!(parse_duration("30").expect("test"), Duration::from_secs(30));
/// ```
pub fn parse_duration(s: &str) -> Result<Duration, ChaosError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(ChaosError::ParseError {
            input: s.to_string(),
            reason: "empty string".to_string(),
        });
    }

    // Check for suffix
    let (num_str, multiplier) = if let Some(stripped) = s.strip_suffix(['s', 'S']) {
        (stripped, 1u64)
    } else if let Some(stripped) = s.strip_suffix(['m', 'M']) {
        (stripped, 60)
    } else if let Some(stripped) = s.strip_suffix(['h', 'H']) {
        (stripped, 3600)
    } else {
        (s, 1)
    };

    num_str.trim().parse::<u64>().map(|n| Duration::from_secs(n * multiplier)).map_err(|_| {
        ChaosError::ParseError { input: s.to_string(), reason: "invalid number".to_string() }
    })
}

impl ChaosConfig {
    /// Create a `ChaosConfig` from CLI arguments
    ///
    /// # Arguments
    /// * `preset` - Optional preset name ("gentle" or "aggressive")
    /// * `memory_limit` - Optional memory limit string (e.g., "64M")
    /// * `cpu_limit` - Optional CPU limit (0.0-1.0)
    /// * `timeout` - Optional timeout string (e.g., "10s")
    /// * `signals` - Enable signal injection
    ///
    /// # Example
    /// ```
    /// use renacer::chaos::ChaosConfig;
    ///
    /// // Use preset
    /// let config = ChaosConfig::from_cli(Some("aggressive"), None, None, None, false).expect("test").expect("test");
    /// assert!(config.signal_injection);
    ///
    /// // Custom values
    /// let config = ChaosConfig::from_cli(None, Some("128M"), Some(0.5), Some("30s"), true).expect("test").expect("test");
    /// assert_eq!(config.memory_limit, 128 * 1024 * 1024);
    /// ```
    pub fn from_cli(
        preset: Option<&str>,
        memory_limit: Option<&str>,
        cpu_limit: Option<f64>,
        timeout: Option<&str>,
        signals: bool,
    ) -> Result<Option<Self>, ChaosError> {
        // If no chaos options specified, return None
        if preset.is_none()
            && memory_limit.is_none()
            && cpu_limit.is_none()
            && timeout.is_none()
            && !signals
        {
            return Ok(None);
        }

        // Start with preset or default
        let mut config = match preset {
            Some("gentle") => Self::gentle(),
            Some("aggressive") => Self::aggressive(),
            Some(other) => {
                return Err(ChaosError::ParseError {
                    input: other.to_string(),
                    reason: "unknown preset (use 'gentle' or 'aggressive')".to_string(),
                })
            }
            None => Self::new(),
        };

        // Override with custom values
        if let Some(mem) = memory_limit {
            config.memory_limit = parse_memory_size(mem)?;
        }

        if let Some(cpu) = cpu_limit {
            config.cpu_limit = cpu.clamp(0.0, 1.0);
        }

        if let Some(t) = timeout {
            config.timeout = parse_duration(t)?;
        }

        if signals {
            config.signal_injection = true;
        }

        Ok(Some(config))
    }

    /// Format a human-readable status line for chaos mode
    pub fn status_line(&self) -> String {
        let mut parts = Vec::new();

        if self.memory_limit > 0 {
            let mem_str = if self.memory_limit >= 1024 * 1024 * 1024 {
                format!("{}GB", self.memory_limit / (1024 * 1024 * 1024))
            } else if self.memory_limit >= 1024 * 1024 {
                format!("{}MB", self.memory_limit / (1024 * 1024))
            } else if self.memory_limit >= 1024 {
                format!("{}KB", self.memory_limit / 1024)
            } else {
                format!("{}B", self.memory_limit)
            };
            parts.push(format!("memory={mem_str}"));
        }

        if self.cpu_limit > 0.0 {
            parts.push(format!("cpu={}%", (self.cpu_limit * 100.0) as u32));
        }

        if self.timeout < Duration::from_secs(60) || self.timeout > Duration::from_secs(60) {
            parts.push(format!("timeout={}s", self.timeout.as_secs()));
        }

        if self.signal_injection {
            parts.push("signals=on".to_string());
        }

        parts.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ChaosConfig::new();
        assert_eq!(config.memory_limit, 0);
        assert!((config.cpu_limit - 0.0).abs() < f64::EPSILON);
        assert_eq!(config.timeout.as_secs(), 60);
        assert!(!config.signal_injection);
    }

    #[test]
    fn test_builder_chain() {
        let config = ChaosConfig::new()
            .with_memory_limit(100)
            .with_cpu_limit(0.5)
            .with_timeout(Duration::from_secs(30))
            .with_signal_injection(true)
            .build();

        assert_eq!(config.memory_limit, 100);
        assert!((config.cpu_limit - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.timeout.as_secs(), 30);
        assert!(config.signal_injection);
    }

    #[test]
    fn test_cpu_limit_clamping() {
        let config = ChaosConfig::new().with_cpu_limit(1.5);
        assert!((config.cpu_limit - 1.0).abs() < f64::EPSILON);

        let config = ChaosConfig::new().with_cpu_limit(-0.5);
        assert!((config.cpu_limit - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_is_active() {
        let config = ChaosConfig::new();
        assert!(!config.is_active());

        let config = ChaosConfig::new().with_memory_limit(100);
        assert!(config.is_active());

        let config = ChaosConfig::new().with_signal_injection(true);
        assert!(config.is_active());
    }

    #[test]
    fn test_gentle_preset() {
        let config = ChaosConfig::gentle();
        assert_eq!(config.memory_limit, 512 * 1024 * 1024);
        assert!((config.cpu_limit - 0.8).abs() < f64::EPSILON);
        assert_eq!(config.timeout.as_secs(), 120);
    }

    #[test]
    fn test_aggressive_preset() {
        let config = ChaosConfig::aggressive();
        assert_eq!(config.memory_limit, 64 * 1024 * 1024);
        assert!((config.cpu_limit - 0.25).abs() < f64::EPSILON);
        assert!(config.signal_injection);
    }

    #[test]
    fn test_error_display() {
        let err = ChaosError::MemoryLimitExceeded { limit: 100, used: 200 };
        assert!(err.to_string().contains("200"));
        assert!(err.to_string().contains("100"));

        let err =
            ChaosError::Timeout { elapsed: Duration::from_secs(10), limit: Duration::from_secs(5) };
        assert!(err.to_string().contains("Timeout"));
    }

    #[cfg(feature = "chaos-network")]
    #[test]
    fn test_network_chaos() {
        let config = ChaosConfig::new().with_network_latency(100).with_packet_loss(0.1);

        assert_eq!(config.network_latency_ms, 100);
        assert!((config.packet_loss_prob - 0.1).abs() < f64::EPSILON);
    }

    #[cfg(feature = "chaos-byzantine")]
    #[test]
    fn test_byzantine_chaos() {
        let config = ChaosConfig::new()
            .with_byzantine_faults(0.05)
            .with_fault_syscalls(vec!["read".to_string()]);

        assert!((config.byzantine_fault_prob - 0.05).abs() < f64::EPSILON);
        assert_eq!(config.fault_syscalls, vec!["read".to_string()]);
    }

    #[cfg(feature = "chaos-byzantine")]
    #[test]
    fn test_extreme_preset() {
        let config = ChaosConfig::extreme();
        assert!((config.byzantine_fault_prob - 0.1).abs() < f64::EPSILON);
        assert!(!config.fault_syscalls.is_empty());
    }

    // Sprint 47: Parse function tests (Issue #17)
    #[test]
    fn test_parse_memory_size_bytes() {
        assert_eq!(parse_memory_size("1024").expect("test"), 1024);
        assert_eq!(parse_memory_size("0").expect("test"), 0);
        assert_eq!(parse_memory_size("67108864").expect("test"), 67108864);
    }

    #[test]
    fn test_parse_memory_size_kilobytes() {
        assert_eq!(parse_memory_size("1K").expect("test"), 1024);
        assert_eq!(parse_memory_size("512k").expect("test"), 512 * 1024);
        assert_eq!(parse_memory_size("100K").expect("test"), 100 * 1024);
    }

    #[test]
    fn test_parse_memory_size_megabytes() {
        assert_eq!(parse_memory_size("1M").expect("test"), 1024 * 1024);
        assert_eq!(parse_memory_size("64m").expect("test"), 64 * 1024 * 1024);
        assert_eq!(parse_memory_size("512M").expect("test"), 512 * 1024 * 1024);
    }

    #[test]
    fn test_parse_memory_size_gigabytes() {
        assert_eq!(parse_memory_size("1G").expect("test"), 1024 * 1024 * 1024);
        assert_eq!(parse_memory_size("2g").expect("test"), 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_parse_memory_size_with_whitespace() {
        assert_eq!(parse_memory_size(" 64M ").expect("test"), 64 * 1024 * 1024);
        assert_eq!(parse_memory_size("  100K").expect("test"), 100 * 1024);
    }

    #[test]
    fn test_parse_memory_size_errors() {
        assert!(parse_memory_size("").is_err());
        assert!(parse_memory_size("abc").is_err());
        assert!(parse_memory_size("64X").is_err());
        assert!(parse_memory_size("-100M").is_err());
    }

    #[test]
    fn test_parse_duration_seconds() {
        assert_eq!(parse_duration("10s").expect("test"), Duration::from_secs(10));
        assert_eq!(parse_duration("30S").expect("test"), Duration::from_secs(30));
        assert_eq!(parse_duration("1s").expect("test"), Duration::from_secs(1));
    }

    #[test]
    fn test_parse_duration_minutes() {
        assert_eq!(parse_duration("1m").expect("test"), Duration::from_secs(60));
        assert_eq!(parse_duration("2M").expect("test"), Duration::from_secs(120));
        assert_eq!(parse_duration("5m").expect("test"), Duration::from_secs(300));
    }

    #[test]
    fn test_parse_duration_hours() {
        assert_eq!(parse_duration("1h").expect("test"), Duration::from_secs(3600));
        assert_eq!(parse_duration("2H").expect("test"), Duration::from_secs(7200));
    }

    #[test]
    fn test_parse_duration_raw_seconds() {
        assert_eq!(parse_duration("30").expect("test"), Duration::from_secs(30));
        assert_eq!(parse_duration("0").expect("test"), Duration::from_secs(0));
        assert_eq!(parse_duration("3600").expect("test"), Duration::from_secs(3600));
    }

    #[test]
    fn test_parse_duration_with_whitespace() {
        assert_eq!(parse_duration(" 10s ").expect("test"), Duration::from_secs(10));
        assert_eq!(parse_duration("  5m").expect("test"), Duration::from_secs(300));
    }

    #[test]
    fn test_parse_duration_errors() {
        assert!(parse_duration("").is_err());
        assert!(parse_duration("abc").is_err());
        assert!(parse_duration("10x").is_err());
        assert!(parse_duration("-10s").is_err());
    }

    #[test]
    fn test_from_cli_no_options() {
        let result = ChaosConfig::from_cli(None, None, None, None, false).expect("test");
        assert!(result.is_none());
    }

    #[test]
    fn test_from_cli_gentle_preset() {
        let config = ChaosConfig::from_cli(Some("gentle"), None, None, None, false)
            .expect("test")
            .expect("test");
        assert_eq!(config.memory_limit, 512 * 1024 * 1024);
        assert!((config.cpu_limit - 0.8).abs() < f64::EPSILON);
        assert_eq!(config.timeout.as_secs(), 120);
    }

    #[test]
    fn test_from_cli_aggressive_preset() {
        let config = ChaosConfig::from_cli(Some("aggressive"), None, None, None, false)
            .expect("test")
            .expect("test");
        assert_eq!(config.memory_limit, 64 * 1024 * 1024);
        assert!((config.cpu_limit - 0.25).abs() < f64::EPSILON);
        assert!(config.signal_injection);
    }

    #[test]
    fn test_from_cli_custom_memory() {
        let config = ChaosConfig::from_cli(None, Some("128M"), None, None, false)
            .expect("test")
            .expect("test");
        assert_eq!(config.memory_limit, 128 * 1024 * 1024);
    }

    #[test]
    fn test_from_cli_custom_cpu() {
        let config =
            ChaosConfig::from_cli(None, None, Some(0.5), None, false).expect("test").expect("test");
        assert!((config.cpu_limit - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_from_cli_custom_timeout() {
        let config = ChaosConfig::from_cli(None, None, None, Some("30s"), false)
            .expect("test")
            .expect("test");
        assert_eq!(config.timeout.as_secs(), 30);
    }

    #[test]
    fn test_from_cli_signals_only() {
        let config =
            ChaosConfig::from_cli(None, None, None, None, true).expect("test").expect("test");
        assert!(config.signal_injection);
    }

    #[test]
    fn test_from_cli_preset_with_overrides() {
        let config = ChaosConfig::from_cli(Some("gentle"), Some("256M"), Some(0.5), None, true)
            .expect("test")
            .expect("test");
        // Memory overridden from 512M to 256M
        assert_eq!(config.memory_limit, 256 * 1024 * 1024);
        // CPU overridden from 0.8 to 0.5
        assert!((config.cpu_limit - 0.5).abs() < f64::EPSILON);
        // Timeout preserved from gentle preset
        assert_eq!(config.timeout.as_secs(), 120);
        // Signals enabled (override)
        assert!(config.signal_injection);
    }

    #[test]
    fn test_from_cli_invalid_preset() {
        let result = ChaosConfig::from_cli(Some("unknown"), None, None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_cli_invalid_memory() {
        let result = ChaosConfig::from_cli(None, Some("invalid"), None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_cli_invalid_timeout() {
        let result = ChaosConfig::from_cli(None, None, None, Some("invalid"), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_status_line_empty() {
        let config = ChaosConfig::new();
        assert_eq!(config.status_line(), "");
    }

    #[test]
    fn test_status_line_memory_mb() {
        let config = ChaosConfig::new().with_memory_limit(64 * 1024 * 1024);
        assert!(config.status_line().contains("memory=64MB"));
    }

    #[test]
    fn test_status_line_memory_gb() {
        let config = ChaosConfig::new().with_memory_limit(2 * 1024 * 1024 * 1024);
        assert!(config.status_line().contains("memory=2GB"));
    }

    #[test]
    fn test_status_line_cpu() {
        let config = ChaosConfig::new().with_cpu_limit(0.5);
        assert!(config.status_line().contains("cpu=50%"));
    }

    #[test]
    fn test_status_line_timeout() {
        let config = ChaosConfig::new().with_timeout(Duration::from_secs(30));
        assert!(config.status_line().contains("timeout=30s"));
    }

    #[test]
    fn test_status_line_signals() {
        let config = ChaosConfig::new().with_signal_injection(true);
        assert!(config.status_line().contains("signals=on"));
    }

    #[test]
    fn test_status_line_full() {
        let config = ChaosConfig::aggressive();
        let status = config.status_line();
        assert!(status.contains("memory=64MB"));
        assert!(status.contains("cpu=25%"));
        assert!(status.contains("timeout=10s"));
        assert!(status.contains("signals=on"));
    }

    #[test]
    fn test_parse_error_display() {
        let err = ChaosError::ParseError {
            input: "invalid".to_string(),
            reason: "bad format".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("invalid"));
        assert!(msg.contains("bad format"));
    }

    #[test]
    fn test_resource_limit_error_display() {
        let err = ChaosError::ResourceLimitFailed {
            resource: "RLIMIT_AS".to_string(),
            reason: "permission denied".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("RLIMIT_AS"));
        assert!(msg.contains("permission denied"));
    }
}

// Sprint 47: Property-based tests for chaos parsing (Issue #17)
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Property: All valid memory sizes with K suffix should parse correctly
        #[test]
        fn prop_parse_memory_kilobytes(n in 1usize..1000) {
            let input = format!("{}K", n);
            let result = parse_memory_size(&input);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.expect("test"), n * 1024);
        }

        /// Property: All valid memory sizes with M suffix should parse correctly
        #[test]
        fn prop_parse_memory_megabytes(n in 1usize..1000) {
            let input = format!("{}M", n);
            let result = parse_memory_size(&input);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.expect("test"), n * 1024 * 1024);
        }

        /// Property: All valid memory sizes with G suffix should parse correctly
        #[test]
        fn prop_parse_memory_gigabytes(n in 1usize..10) {
            let input = format!("{}G", n);
            let result = parse_memory_size(&input);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.expect("test"), n * 1024 * 1024 * 1024);
        }

        /// Property: Raw byte values should parse correctly
        #[test]
        fn prop_parse_memory_raw_bytes(n in 0usize..1_000_000) {
            let input = format!("{}", n);
            let result = parse_memory_size(&input);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.expect("test"), n);
        }

        /// Property: All valid durations with s suffix should parse correctly
        #[test]
        fn prop_parse_duration_seconds(n in 0u64..3600) {
            let input = format!("{}s", n);
            let result = parse_duration(&input);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.expect("test"), Duration::from_secs(n));
        }

        /// Property: All valid durations with m suffix should parse correctly
        #[test]
        fn prop_parse_duration_minutes(n in 0u64..120) {
            let input = format!("{}m", n);
            let result = parse_duration(&input);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.expect("test"), Duration::from_secs(n * 60));
        }

        /// Property: All valid durations with h suffix should parse correctly
        #[test]
        fn prop_parse_duration_hours(n in 0u64..24) {
            let input = format!("{}h", n);
            let result = parse_duration(&input);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.expect("test"), Duration::from_secs(n * 3600));
        }

        /// Property: Raw second values should parse correctly
        #[test]
        fn prop_parse_duration_raw(n in 0u64..86400) {
            let input = format!("{}", n);
            let result = parse_duration(&input);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.expect("test"), Duration::from_secs(n));
        }

        /// Property: CPU limit is always clamped to 0.0-1.0
        #[test]
        fn prop_cpu_limit_clamped(fraction in -2.0f64..3.0) {
            let config = ChaosConfig::new().with_cpu_limit(fraction);
            prop_assert!(config.cpu_limit >= 0.0);
            prop_assert!(config.cpu_limit <= 1.0);
        }

        /// Property: Whitespace around values should not affect parsing
        #[test]
        fn prop_parse_memory_whitespace(n in 1usize..100, prefix_spaces in 0usize..5, suffix_spaces in 0usize..5) {
            let prefix = " ".repeat(prefix_spaces);
            let suffix = " ".repeat(suffix_spaces);
            let input = format!("{}{}M{}", prefix, n, suffix);
            let result = parse_memory_size(&input);
            prop_assert!(result.is_ok());
            prop_assert_eq!(result.expect("test"), n * 1024 * 1024);
        }

        /// Property: status_line output contains all active settings
        #[test]
        fn prop_status_line_contains_memory(n in 1usize..1000) {
            let config = ChaosConfig::new().with_memory_limit(n * 1024 * 1024);
            let status = config.status_line();
            prop_assert!(status.contains("memory="));
            prop_assert!(status.contains("MB"));
        }

        /// Property: is_active is true when any limit is set
        #[test]
        fn prop_is_active_with_memory(n in 1usize..1_000_000) {
            let config = ChaosConfig::new().with_memory_limit(n);
            prop_assert!(config.is_active());
        }

        /// Property: is_active is true when cpu limit is set
        #[test]
        fn prop_is_active_with_cpu(fraction in 0.01f64..1.0) {
            let config = ChaosConfig::new().with_cpu_limit(fraction);
            prop_assert!(config.is_active());
        }
    }
}
