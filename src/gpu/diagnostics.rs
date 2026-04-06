//! GPU Diagnostics Module (PMAT-802)
//!
//! Extracted from gpu/mod.rs - Logging, metrics, and debugging infrastructure.
//!
//! ## Contents
//! - `Logger`, `LogConfig`, `LogLevel`, `LogEntry` - Structured logging (IMP-079)
//! - `PhaseTimer`, `MemoryTracker`, `MemoryReport` - Performance tracking (IMP-080)
//! - `DiagnosticsCollector`, `DebugMode`, `RequestCapture`, `StateDump` - Debug tools (IMP-081)

use std::collections::HashMap;

// ============================================================================
// M32: Production Logging & Diagnostics (IMP-079, IMP-080, IMP-081)
// ============================================================================

/// Log level (IMP-079)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Trace level (most verbose)
    Trace,
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Warning level
    Warn,
    /// Error level
    Error,
}

impl LogLevel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }
}

/// Log entry (IMP-079)
#[derive(Debug, Clone)]
pub struct LogEntry {
    level: LogLevel,
    message: String,
    timestamp: u64,
    correlation_id: Option<String>,
    fields: HashMap<String, String>,
}

impl LogEntry {
    /// Create new log entry
    #[must_use]
    pub fn new(level: LogLevel, message: &str) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            level,
            message: message.to_string(),
            timestamp,
            correlation_id: None,
            fields: HashMap::new(),
        }
    }

    /// Set correlation ID
    #[must_use]
    pub fn with_correlation_id(mut self, id: &str) -> Self {
        self.correlation_id = Some(id.to_string());
        self
    }

    /// Add custom field
    #[must_use]
    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.fields.insert(key.to_string(), value.to_string());
        self
    }

    /// Get correlation ID
    #[must_use]
    pub fn correlation_id(&self) -> Option<&str> {
        self.correlation_id.as_deref()
    }

    /// Get log level
    #[must_use]
    pub fn level(&self) -> LogLevel {
        self.level
    }

    /// Get timestamp
    #[must_use]
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Convert to JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        use std::fmt::Write;

        let mut json = format!(
            "{{\"level\":\"{}\",\"message\":\"{}\",\"timestamp\":{}",
            self.level.as_str(),
            self.message,
            self.timestamp
        );

        if let Some(ref id) = self.correlation_id {
            let _ = write!(json, ",\"correlation_id\":\"{}\"", id);
        }

        for (key, value) in &self.fields {
            let _ = write!(json, ",\"{}\":\"{}\"", key, value);
        }

        json.push('}');
        json
    }
}

/// Log configuration (IMP-079)
#[derive(Debug, Clone)]
pub struct LogConfig {
    default_level: LogLevel,
    json_format: bool,
    module_levels: HashMap<String, LogLevel>,
}

impl LogConfig {
    /// Create new log config
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_level: LogLevel::Info,
            json_format: false,
            module_levels: HashMap::new(),
        }
    }

    /// Set default log level
    #[must_use]
    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.default_level = level;
        self
    }

    /// Enable JSON format
    #[must_use]
    pub fn with_json_format(mut self, enabled: bool) -> Self {
        self.json_format = enabled;
        self
    }

    /// Set module-specific log level
    #[must_use]
    pub fn with_module_level(mut self, module: &str, level: LogLevel) -> Self {
        self.module_levels.insert(module.to_string(), level);
        self
    }
}

impl Default for LogConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Logger (IMP-079)
pub struct Logger {
    config: LogConfig,
}

impl Logger {
    /// Create new logger
    #[must_use]
    pub fn new(config: LogConfig) -> Self {
        Self { config }
    }

    /// Check if log level is enabled for module
    #[must_use]
    pub fn is_enabled(&self, level: LogLevel, module: &str) -> bool {
        let min_level = self
            .config
            .module_levels
            .get(module)
            .copied()
            .unwrap_or(self.config.default_level);
        level >= min_level
    }
}

/// Phase timer for latency breakdown (IMP-080)
pub struct PhaseTimer {
    phases: std::sync::Mutex<HashMap<String, (Option<std::time::Instant>, u64)>>,
}

impl PhaseTimer {
    /// Create new phase timer
    #[must_use]
    pub fn new() -> Self {
        Self {
            phases: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Start timing a phase
    pub fn start_phase(&self, name: &str) {
        let mut phases = self.phases.lock().expect("mutex poisoned");
        phases.insert(name.to_string(), (Some(std::time::Instant::now()), 0));
    }

    /// End timing a phase
    pub fn end_phase(&self, name: &str) {
        let mut phases = self.phases.lock().expect("mutex poisoned");
        if let Some((Some(start_time), _)) = phases.get(name) {
            let elapsed = start_time.elapsed().as_micros() as u64;
            phases.insert(name.to_string(), (None, elapsed));
        }
    }

    /// Get timing breakdown
    #[must_use]
    pub fn breakdown(&self) -> HashMap<String, u64> {
        let phases = self.phases.lock().expect("mutex poisoned");
        phases.iter().map(|(k, (_, v))| (k.clone(), *v)).collect()
    }
}

impl Default for PhaseTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory report (IMP-080)
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Peak memory usage in bytes
    pub peak_bytes: u64,
    /// Current memory usage in bytes
    pub current_bytes: u64,
    /// Total allocation count
    pub allocation_count: u64,
}

/// Memory tracker (IMP-080)
pub struct MemoryTracker {
    current: std::sync::atomic::AtomicU64,
    peak: std::sync::atomic::AtomicU64,
    allocation_count: std::sync::atomic::AtomicU64,
}

impl MemoryTracker {
    /// Create new memory tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            current: std::sync::atomic::AtomicU64::new(0),
            peak: std::sync::atomic::AtomicU64::new(0),
            allocation_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Record memory allocation
    pub fn record_allocation(&self, _name: &str, bytes: u64) {
        let new_current = self
            .current
            .fetch_add(bytes, std::sync::atomic::Ordering::SeqCst)
            + bytes;
        self.allocation_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Update peak if necessary
        let mut peak = self.peak.load(std::sync::atomic::Ordering::SeqCst);
        while new_current > peak {
            match self.peak.compare_exchange_weak(
                peak,
                new_current,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current_peak) => peak = current_peak,
            }
        }
    }

    /// Record memory deallocation
    pub fn record_deallocation(&self, _name: &str, bytes: u64) {
        self.current
            .fetch_sub(bytes, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get memory report
    #[must_use]
    pub fn report(&self) -> MemoryReport {
        MemoryReport {
            peak_bytes: self.peak.load(std::sync::atomic::Ordering::SeqCst),
            current_bytes: self.current.load(std::sync::atomic::Ordering::SeqCst),
            allocation_count: self
                .allocation_count
                .load(std::sync::atomic::Ordering::SeqCst),
        }
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Diagnostics summary (IMP-080)
#[derive(Debug, Clone)]
pub struct DiagnosticsSummary {
    /// Number of requests tracked
    pub request_count: u64,
}

/// Diagnostics collector (IMP-080)
pub struct DiagnosticsCollector {
    /// Request count (pub(crate) for test access)
    pub(crate) request_count: std::sync::atomic::AtomicU64,
    timings: std::sync::Mutex<Vec<HashMap<String, u64>>>,
    memory_snapshots: std::sync::Mutex<Vec<MemoryReport>>,
}

impl DiagnosticsCollector {
    /// Create new diagnostics collector
    #[must_use]
    pub fn new() -> Self {
        Self {
            request_count: std::sync::atomic::AtomicU64::new(0),
            timings: std::sync::Mutex::new(Vec::new()),
            memory_snapshots: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Record request timing
    pub fn record_request_timing(&self, _request_id: &str, timing: HashMap<String, u64>) {
        self.request_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.timings.lock().expect("mutex poisoned").push(timing);
    }

    /// Record memory snapshot
    pub fn record_memory_snapshot(&self, report: MemoryReport) {
        self.memory_snapshots
            .lock()
            .expect("mutex poisoned")
            .push(report);
    }

    /// Get diagnostics summary
    #[must_use]
    pub fn summary(&self) -> DiagnosticsSummary {
        DiagnosticsSummary {
            request_count: self.request_count.load(std::sync::atomic::Ordering::SeqCst),
        }
    }
}

impl Default for DiagnosticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Debug mode controller (IMP-081)
pub struct DebugMode {
    enabled: std::sync::atomic::AtomicBool,
}

impl DebugMode {
    /// Create new debug mode
    #[must_use]
    pub fn new() -> Self {
        Self {
            enabled: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Check if debug mode is enabled
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Enable debug mode
    pub fn enable(&self) {
        self.enabled
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Disable debug mode
    pub fn disable(&self) {
        self.enabled
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Default for DebugMode {
    fn default() -> Self {
        Self::new()
    }
}

/// Request capture for replay (IMP-081)
#[derive(Debug, Clone)]
pub struct RequestCapture {
    input: String,
    params: HashMap<String, String>,
}

include!("request_capture.rs");
include!("diagnostics_request_capture.rs");
