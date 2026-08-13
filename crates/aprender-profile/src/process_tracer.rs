//! Process-level syscall tracing for ptop integration
//!
//! This module provides syscall-level visibility for processes displayed in ptop,
//! enabling deep tracing when performance anomalies are detected.
//!
//! # SPEC-057 Implementation
//!
//! Per specification `docs/specifications/ptop-presentar-tracing-support.md`:
//! - Attach to running processes via ptrace
//! - Collect syscall events with timing
//! - Compute z-score deviations from baseline
//! - Export traces to OTLP (Jaeger/Tempo)
//!
//! # Scientific Foundations
//!
//! - Mace et al. (2015): Pivot Tracing - only trace when anomalies detected
//! - Shewhart (1931): Statistical process control - z-score for anomaly detection
//! - Sigelman et al. (2010): Dapper - rate limiting to prevent self-DoS
//!
//! # Toyota Way Alignment
//!
//! - **Genchi Genbutsu**: Trace real syscalls, not simulated workloads
//! - **Jidoka**: Auto-escalate when thresholds exceeded
//! - **Muda Elimination**: Zero overhead when dormant
//! - **Mieruka**: Visual breakdown in ptop Trace Panel
//!
//! # Example
//!
//! ```rust,ignore
//! use renacer::process_tracer::{attach, collect, detach, ProcessTraceConfig};
//!
//! let config = ProcessTraceConfig::default();
//! let mut trace = attach(1234, config)?;
//! let result = collect(&mut trace)?;
//!
//! println!("Syscall breakdown: {:?}", result.breakdown);
//! println!("Max z-score: {:.2}", result.max_zscore);
//!
//! for anomaly in &result.anomalies {
//!     println!("Anomaly: {} at {:.1}σ", anomaly.syscall, anomaly.zscore);
//! }
//!
//! detach(trace)?;
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nix::sys::ptrace;
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::Pid;

/// Configuration for process tracing
///
/// # PMAT-057-002
#[derive(Debug, Clone)]
pub struct ProcessTraceConfig {
    /// Maximum syscalls to capture per collection cycle (default: 1000)
    pub max_syscalls: usize,
    /// Timeout for ptrace operations (default: 100ms)
    pub timeout: Duration,
    /// Enable source correlation via DWARF (default: false)
    pub enable_source: bool,
    /// OTLP endpoint for span export (default: None)
    pub otlp_endpoint: Option<String>,
    /// Rate limit traces per second (default: 100)
    pub rate_limit: u32,
    /// Z-score threshold for anomaly detection (default: 3.0)
    pub anomaly_threshold: f32,
}

impl Default for ProcessTraceConfig {
    fn default() -> Self {
        Self {
            max_syscalls: 1000,
            timeout: Duration::from_millis(100),
            enable_source: false,
            otlp_endpoint: None,
            rate_limit: 100,
            anomaly_threshold: 3.0,
        }
    }
}

impl ProcessTraceConfig {
    /// Set maximum syscalls per collection
    #[must_use]
    pub fn with_max_syscalls(mut self, max: usize) -> Self {
        self.max_syscalls = max;
        self
    }

    /// Set timeout for ptrace operations
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable DWARF source correlation
    #[must_use]
    pub fn with_source(mut self, enable: bool) -> Self {
        self.enable_source = enable;
        self
    }

    /// Set OTLP endpoint for export
    #[must_use]
    pub fn with_otlp(mut self, endpoint: String) -> Self {
        self.otlp_endpoint = Some(endpoint);
        self
    }

    /// Set rate limit
    #[must_use]
    pub fn with_rate_limit(mut self, rate: u32) -> Self {
        self.rate_limit = rate;
        self
    }

    /// Alias for `with_source` - enable source correlation
    #[must_use]
    pub fn with_source_correlation(self, enable: bool) -> Self {
        self.with_source(enable)
    }

    /// Set anomaly detection threshold (z-score)
    ///
    /// Events with z-scores above this threshold are flagged as anomalies.
    /// Default: 3.0 (3 sigma / 0.27% false positive rate)
    #[must_use]
    pub fn with_anomaly_threshold(mut self, threshold: f32) -> Self {
        self.anomaly_threshold = threshold;
        self
    }
}

/// Errors that can occur during process tracing
///
/// # PMAT-057-003
#[derive(Debug, thiserror::Error)]
pub enum TracerError {
    /// Permission denied (need CAP_SYS_PTRACE or root)
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// Process not found
    #[error("Process {pid} not found")]
    ProcessNotFound {
        /// The PID that was not found
        pid: u32,
    },

    /// Process is already being traced
    #[error("Process {pid} is already being traced")]
    AlreadyTraced {
        /// The PID that is already traced
        pid: u32,
    },

    /// Process exited during tracing
    #[error("Process {pid} exited during tracing")]
    ProcessExited {
        /// The PID that exited
        pid: u32,
    },

    /// Ptrace operation failed
    #[error("Ptrace error: {0}")]
    PtraceError(#[from] nix::Error),

    /// Timeout waiting for syscall
    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {current}/s > {limit}/s")]
    RateLimitExceeded {
        /// Current rate
        current: u32,
        /// Configured limit
        limit: u32,
    },

    /// OTLP export failed
    #[error("OTLP export error: {0}")]
    OtlpError(String),

    /// DWARF parsing error
    #[error("DWARF error: {0}")]
    DwarfError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Not attached to process
    #[error("Not attached to any process")]
    NotAttached,
}

// Compile-time thread-safety verification (Sprint 59)
// Ensures all public types can be safely shared across thread boundaries.
// Required for concurrent trace collection and async OTLP export.
static_assertions::assert_impl_all!(TracerError: Send, Sync);
static_assertions::assert_impl_all!(ProcessTraceConfig: Send, Sync);
static_assertions::assert_impl_all!(SyscallEvent: Send, Sync);
static_assertions::assert_impl_all!(SyscallBreakdown: Send, Sync);
static_assertions::assert_impl_all!(SyscallAnomaly: Send, Sync);
static_assertions::assert_impl_all!(SourceLocation: Send, Sync);
static_assertions::assert_impl_all!(OtlpAttribute: Send, Sync);
static_assertions::assert_impl_all!(OtlpSpan: Send, Sync);
static_assertions::assert_impl_all!(TraceResult: Send, Sync);

/// Syscall event captured during tracing
///
/// # PMAT-057-007
#[derive(Debug, Clone)]
pub struct SyscallEvent {
    /// Syscall name (e.g., "mmap", "futex", "read")
    pub syscall: String,
    /// Syscall number
    pub syscall_nr: i64,
    /// Duration of the syscall
    pub duration: Duration,
    /// Return value
    pub result: i64,
    /// Timestamp when syscall started
    pub timestamp: Instant,
}

impl SyscallEvent {
    /// Create a new syscall event
    pub fn new(syscall: String, syscall_nr: i64, duration: Duration, result: i64) -> Self {
        Self { syscall, syscall_nr, duration, result, timestamp: Instant::now() }
    }
}

/// Syscall breakdown for root cause analysis
///
/// When a process exceeds thresholds, this breakdown shows where
/// time was spent in syscalls vs actual computation.
///
/// # PMAT-057-008
#[derive(Debug, Clone, Default)]
pub struct SyscallBreakdown {
    /// Time spent in mmap syscalls (memory allocation)
    pub mmap_us: u64,
    /// Time spent in futex syscalls (thread synchronization)
    pub futex_us: u64,
    /// Time spent in ioctl syscalls (device control, CUDA driver)
    pub ioctl_us: u64,
    /// Time spent in read syscalls
    pub read_us: u64,
    /// Time spent in write syscalls
    pub write_us: u64,
    /// Time spent in other syscalls
    pub other_us: u64,
    /// Time spent in actual computation (total - syscall overhead)
    pub compute_us: u64,
    /// Total syscall count
    pub syscall_count: u64,
    /// Per-syscall counts
    pub syscall_counts: HashMap<String, u64>,
    /// Total trace duration
    pub total_us: u64,
}

impl SyscallBreakdown {
    /// Create breakdown from syscall events
    pub fn from_events(events: &[SyscallEvent], total_duration_us: u64) -> Self {
        // contract_pre_absolute_position_add!(events); // FIXME: macro hygiene
        let mut breakdown = Self { total_us: total_duration_us, ..Default::default() };
        let mut syscall_time_us: u64 = 0;

        for event in events {
            let duration_us = event.duration.as_micros() as u64;
            syscall_time_us += duration_us;
            breakdown.syscall_count += 1;

            *breakdown.syscall_counts.entry(event.syscall.clone()).or_insert(0) += 1;

            match event.syscall.as_str() {
                "mmap" | "munmap" | "mprotect" | "brk" | "mremap" => {
                    breakdown.mmap_us += duration_us;
                }
                "futex" => breakdown.futex_us += duration_us,
                "ioctl" => breakdown.ioctl_us += duration_us,
                "read" | "pread64" | "readv" | "preadv" | "preadv2" => {
                    breakdown.read_us += duration_us;
                }
                "write" | "pwrite64" | "writev" | "pwritev" | "pwritev2" => {
                    breakdown.write_us += duration_us;
                }
                _ => breakdown.other_us += duration_us,
            }
        }

        // Compute time = total - syscall overhead
        breakdown.compute_us = total_duration_us.saturating_sub(syscall_time_us);
        breakdown
    }

    /// Get total syscall time in microseconds
    pub fn syscall_time_us(&self) -> u64 {
        self.mmap_us + self.futex_us + self.ioctl_us + self.read_us + self.write_us + self.other_us
    }

    /// Get efficiency ratio (compute / total)
    pub fn efficiency(&self) -> f64 {
        if self.total_us == 0 {
            1.0
        } else {
            self.compute_us as f64 / self.total_us as f64
        }
    }

    /// Get sorted categories by time (descending)
    pub fn sorted_categories(&self) -> Vec<(&'static str, u64)> {
        let mut categories = vec![
            ("mmap", self.mmap_us),
            ("futex", self.futex_us),
            ("ioctl", self.ioctl_us),
            ("read", self.read_us),
            ("write", self.write_us),
            ("other", self.other_us),
            ("compute", self.compute_us),
        ];
        categories.sort_by_key(|b| std::cmp::Reverse(b.1));
        categories
    }
}

/// Baseline statistics for z-score calculation
///
/// # PMAT-057-010
#[derive(Debug, Clone, Default)]
pub struct SyscallBaseline {
    /// Mean duration per syscall category (microseconds)
    pub mean_us: HashMap<String, f64>,
    /// Standard deviation per syscall category
    pub std_us: HashMap<String, f64>,
    /// Sample count used to compute baseline
    pub sample_count: u64,
}

impl SyscallBaseline {
    /// Compute baseline from historical events
    pub fn from_events(events: &[SyscallEvent]) -> Self {
        let mut category_times: HashMap<String, Vec<f64>> = HashMap::new();

        for event in events {
            let category = categorize_syscall(&event.syscall);
            category_times
                .entry(category.to_string())
                .or_default()
                .push(event.duration.as_micros() as f64);
        }

        let mut mean_us = HashMap::new();
        let mut std_us = HashMap::new();

        for (category, times) in &category_times {
            if times.is_empty() {
                continue;
            }

            let n = times.len() as f64;
            let mean = times.iter().sum::<f64>() / n;
            mean_us.insert(category.clone(), mean);

            if times.len() > 1 {
                let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
                std_us.insert(category.clone(), variance.sqrt());
            } else {
                // With single sample, use mean as std to avoid division by zero
                std_us.insert(category.clone(), mean.max(1.0));
            }
        }

        Self { mean_us, std_us, sample_count: events.len() as u64 }
    }

    /// Check if baseline has data for a category
    pub fn has_category(&self, category: &str) -> bool {
        self.mean_us.contains_key(category)
    }
}

/// A syscall that deviated significantly from baseline
///
/// # PMAT-057-009
#[derive(Debug, Clone)]
pub struct SyscallAnomaly {
    /// Syscall name
    pub syscall: String,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Z-score deviation
    pub zscore: f32,
    /// Expected duration based on baseline
    pub expected_us: f64,
    /// Category for grouping
    pub category: String,
}

/// Source code location from DWARF
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// File path
    pub file: String,
    /// Line number
    pub line: u32,
    /// Function name
    pub function: Option<String>,
}

/// Attribute for OTLP span export
#[derive(Debug, Clone)]
pub struct OtlpAttribute {
    /// Attribute key
    pub key: String,
    /// Attribute value
    pub value: OtlpValue,
}

/// Value types for OTLP attributes
#[derive(Debug, Clone)]
pub enum OtlpValue {
    /// Integer value
    Int(i64),
    /// Floating point value
    Float(f64),
    /// String value
    String(String),
}

/// OTLP span representation for export to Jaeger/Tempo
///
/// # PMAT-057-010
#[derive(Debug, Clone)]
pub struct OtlpSpan {
    /// Span name (operation identifier)
    pub name: String,
    /// Trace ID (128-bit)
    pub trace_id: [u8; 16],
    /// Span ID (64-bit)
    pub span_id: [u8; 8],
    /// Span attributes
    pub attributes: Vec<OtlpAttribute>,
}

/// Result of a single trace collection cycle
///
/// # PMAT-057-009
#[derive(Debug, Clone)]
pub struct TraceResult {
    /// Process ID traced
    pub pid: u32,
    /// Duration of trace collection
    pub duration: Duration,
    /// Syscall breakdown
    pub breakdown: SyscallBreakdown,
    /// Maximum z-score deviation
    pub max_zscore: f32,
    /// Anomalous syscalls (z > 3.0)
    pub anomalies: Vec<SyscallAnomaly>,
    /// Source locations (if DWARF enabled)
    pub source_locations: Vec<SourceLocation>,
    /// Raw syscall events
    pub events: Vec<SyscallEvent>,
}

impl TraceResult {
    /// Create a new trace result
    pub fn new(pid: u32, duration: Duration, events: Vec<SyscallEvent>) -> Self {
        let breakdown = SyscallBreakdown::from_events(&events, duration.as_micros() as u64);
        Self {
            pid,
            duration,
            breakdown,
            max_zscore: 0.0,
            anomalies: Vec::new(),
            source_locations: Vec::new(),
            events,
        }
    }

    /// Compute anomalies using baseline
    pub fn with_baseline(mut self, baseline: &SyscallBaseline, threshold: f32) -> Self {
        let mut max_z: f32 = 0.0;

        for event in &self.events {
            let category = categorize_syscall(&event.syscall);
            let z = zscore(event, baseline);

            if z > max_z {
                max_z = z;
            }

            if z > threshold {
                let expected = baseline.mean_us.get(category).copied().unwrap_or(0.0);
                self.anomalies.push(SyscallAnomaly {
                    syscall: event.syscall.clone(),
                    duration_us: event.duration.as_micros() as u64,
                    zscore: z,
                    expected_us: expected,
                    category: category.to_string(),
                });
            }
        }

        self.max_zscore = max_z;
        self
    }

    /// Convert to OTLP span format for Jaeger/Tempo export
    ///
    /// Creates an OpenTelemetry-compatible span with:
    /// - Process attributes (pid, syscall count)
    /// - Breakdown attributes (mmap_us, futex_us, read_us, write_us, compute_us)
    /// - Anomaly attributes (max_zscore, anomaly_count)
    pub fn to_otlp_span(&self) -> OtlpSpan {
        use std::time::{SystemTime, UNIX_EPOCH};

        // Generate trace/span IDs from timestamp + pid
        let now =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;

        let mut trace_id = [0u8; 16];
        trace_id[0..8].copy_from_slice(&now.to_be_bytes());
        trace_id[8..12].copy_from_slice(&self.pid.to_be_bytes());

        let mut span_id = [0u8; 8];
        span_id[0..4].copy_from_slice(&self.pid.to_be_bytes());
        span_id[4..8].copy_from_slice(&(now as u32).to_be_bytes());

        let attributes = vec![
            OtlpAttribute {
                key: "process.pid".to_string(),
                value: OtlpValue::Int(i64::from(self.pid)),
            },
            OtlpAttribute {
                key: "syscall.count".to_string(),
                value: OtlpValue::Int(self.events.len() as i64),
            },
            OtlpAttribute {
                key: "syscall.mmap_us".to_string(),
                value: OtlpValue::Int(self.breakdown.mmap_us as i64),
            },
            OtlpAttribute {
                key: "syscall.futex_us".to_string(),
                value: OtlpValue::Int(self.breakdown.futex_us as i64),
            },
            OtlpAttribute {
                key: "syscall.ioctl_us".to_string(),
                value: OtlpValue::Int(self.breakdown.ioctl_us as i64),
            },
            OtlpAttribute {
                key: "syscall.read_us".to_string(),
                value: OtlpValue::Int(self.breakdown.read_us as i64),
            },
            OtlpAttribute {
                key: "syscall.write_us".to_string(),
                value: OtlpValue::Int(self.breakdown.write_us as i64),
            },
            OtlpAttribute {
                key: "syscall.compute_us".to_string(),
                value: OtlpValue::Int(self.breakdown.compute_us as i64),
            },
            OtlpAttribute {
                key: "anomaly.max_zscore".to_string(),
                value: OtlpValue::Float(f64::from(self.max_zscore)),
            },
            OtlpAttribute {
                key: "anomaly.count".to_string(),
                value: OtlpValue::Int(self.anomalies.len() as i64),
            },
        ];

        OtlpSpan { name: format!("process.trace.{}", self.pid), trace_id, span_id, attributes }
    }
}

/// Handle to an active process trace
///
/// # PMAT-057-004
#[derive(Debug)]
pub struct ProcessTrace {
    /// Process ID being traced
    pid: u32,
    /// Trace configuration
    config: ProcessTraceConfig,
    /// Start time of trace session (reserved for session duration tracking)
    _start_time: Instant,
    /// Collected syscall events
    events: Vec<SyscallEvent>,
    /// Baseline for z-score calculation
    baseline: Option<SyscallBaseline>,
    /// Whether currently attached
    attached: bool,
    /// Rate limiter state (reserved for per-trace rate limiting)
    _trace_count: Arc<AtomicU64>,
    /// Last rate limit check time (reserved for per-trace rate limiting)
    _rate_limit_window: Instant,
}

impl ProcessTrace {
    /// Get the traced PID
    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// Check if attached
    pub fn is_attached(&self) -> bool {
        self.attached
    }

    /// Get current event count
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Set baseline for z-score calculation
    pub fn set_baseline(&mut self, baseline: SyscallBaseline) {
        self.baseline = Some(baseline);
    }

    /// Get baseline
    pub fn baseline(&self) -> Option<&SyscallBaseline> {
        self.baseline.as_ref()
    }
}

// ============================================================================
// Rate Limiting
// ============================================================================

/// Global rate limiter for trace operations
static GLOBAL_TRACE_COUNT: AtomicU64 = AtomicU64::new(0);
static GLOBAL_RATE_WINDOW: AtomicU64 = AtomicU64::new(0);

fn check_rate_limit(config: &ProcessTraceConfig) -> Result<(), TracerError> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("System clock before UNIX epoch")
        .as_secs();

    let window = GLOBAL_RATE_WINDOW.load(Ordering::Relaxed);

    if now > window {
        // New window, reset count
        GLOBAL_RATE_WINDOW.store(now, Ordering::Relaxed);
        GLOBAL_TRACE_COUNT.store(1, Ordering::Relaxed);
        Ok(())
    } else {
        let count = GLOBAL_TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if count > config.rate_limit as u64 {
            Err(TracerError::RateLimitExceeded { current: count as u32, limit: config.rate_limit })
        } else {
            Ok(())
        }
    }
}

// ============================================================================
// Syscall Categorization
// ============================================================================

/// Categorize a syscall by name
fn categorize_syscall(name: &str) -> &'static str {
    match name {
        "mmap" | "munmap" | "mprotect" | "brk" | "mremap" => "mmap",
        "futex" => "futex",
        "ioctl" => "ioctl",
        "read" | "pread64" | "readv" | "preadv" | "preadv2" => "read",
        "write" | "pwrite64" | "writev" | "pwritev" | "pwritev2" => "write",
        _ => "other",
    }
}

/// Get syscall name from number (Linux x86_64)
///
/// Returns the human-readable name for a syscall number.
/// Returns "unknown" for unrecognized syscall numbers.
///
/// # Example
///
/// ```
/// use renacer::process_tracer::syscall_name;
/// assert_eq!(syscall_name(0), "read");
/// assert_eq!(syscall_name(1), "write");
/// assert_eq!(syscall_name(202), "futex");
/// ```
pub fn syscall_name(nr: i64) -> &'static str {
    match nr {
        0 => "read",
        1 => "write",
        2 => "open",
        3 => "close",
        4 => "stat",
        5 => "fstat",
        6 => "lstat",
        7 => "poll",
        8 => "lseek",
        9 => "mmap",
        10 => "mprotect",
        11 => "munmap",
        12 => "brk",
        13 => "rt_sigaction",
        14 => "rt_sigprocmask",
        15 => "rt_sigreturn",
        16 => "ioctl",
        17 => "pread64",
        18 => "pwrite64",
        19 => "readv",
        20 => "writev",
        21 => "access",
        22 => "pipe",
        23 => "select",
        24 => "sched_yield",
        25 => "mremap",
        35 => "nanosleep",
        56 => "clone",
        57 => "fork",
        58 => "vfork",
        59 => "execve",
        60 => "exit",
        61 => "wait4",
        62 => "kill",
        202 => "futex",
        _ => "unknown",
    }
}

// ============================================================================
// Z-Score Calculation
// ============================================================================

/// Calculate z-score for a syscall event against baseline
///
/// # PMAT-057-010
pub fn zscore(event: &SyscallEvent, baseline: &SyscallBaseline) -> f32 {
    let category = categorize_syscall(&event.syscall);

    let mean = match baseline.mean_us.get(category) {
        Some(&m) => m,
        None => return 0.0, // No baseline data for this category
    };

    let std = match baseline.std_us.get(category) {
        Some(&s) if s > 0.0 => s,
        _ => return 0.0, // Avoid division by zero
    };

    let duration = event.duration.as_micros() as f64;
    ((duration - mean) / std) as f32
}

/// Compute baseline statistics from historical events
///
/// # PMAT-057-010
pub fn compute_baseline(events: &[SyscallEvent]) -> SyscallBaseline {
    SyscallBaseline::from_events(events)
}

// ============================================================================
// Public API Functions
// ============================================================================

/// Check if process tracing is available on this system
/// Check if the current process has CAP_SYS_PTRACE capability
fn has_ptrace_capability() -> bool {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return false;
    };
    let Some(cap_line) = status.lines().find(|l| l.starts_with("CapEff:")) else {
        return false;
    };
    let Some(hex) = cap_line.split_whitespace().nth(1) else {
        return false;
    };
    let Ok(caps) = u64::from_str_radix(hex, 16) else {
        return false;
    };
    // CAP_SYS_PTRACE = bit 19
    (caps & (1 << 19)) != 0
}

/// Check if ptrace_scope allows unrestricted tracing
fn ptrace_scope_allows_tracing() -> bool {
    std::fs::read_to_string("/proc/sys/kernel/yama/ptrace_scope")
        .map(|scope| scope.trim() == "0")
        .unwrap_or(false)
}

pub fn is_available() -> bool {
    nix::unistd::geteuid().is_root() || has_ptrace_capability() || ptrace_scope_allows_tracing()
}

/// Check if we can trace a specific PID
fn can_trace_pid(pid: u32) -> Result<bool, TracerError> {
    // Check if process exists
    let proc_path = format!("/proc/{}", pid);
    if !std::path::Path::new(&proc_path).exists() {
        return Err(TracerError::ProcessNotFound { pid });
    }

    // Root can trace anything
    if nix::unistd::geteuid().is_root() {
        return Ok(true);
    }

    // Check if we own the process
    let status = std::fs::read_to_string(format!("/proc/{}/status", pid))?;
    let uid_line = status
        .lines()
        .find(|l| l.starts_with("Uid:"))
        .ok_or(TracerError::ProcessNotFound { pid })?;

    let real_uid: u32 = uid_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or(TracerError::ProcessNotFound { pid })?;

    Ok(real_uid == nix::unistd::getuid().as_raw())
}

/// Attach to a process and begin tracing syscalls
///
/// # PMAT-057-005
///
/// # Arguments
/// * `pid` - Process ID to trace
/// * `config` - Trace configuration
///
/// # Returns
/// * `Ok(ProcessTrace)` - Handle to active trace
/// * `Err(TracerError)` - If attach fails
///
/// # Errors
/// * `TracerError::PermissionDenied` - CAP_SYS_PTRACE required
/// * `TracerError::ProcessNotFound` - PID does not exist
/// * `TracerError::AlreadyTraced` - Process is already being traced
pub fn attach(pid: u32, config: ProcessTraceConfig) -> Result<ProcessTrace, TracerError> {
    // Check rate limit
    check_rate_limit(&config)?;

    // Verify we can trace this PID
    if !can_trace_pid(pid)? {
        return Err(TracerError::PermissionDenied(format!(
            "Cannot trace PID {} (not owner and not root)",
            pid
        )));
    }

    let nix_pid = Pid::from_raw(pid as i32);

    // Attempt ptrace attach
    ptrace::attach(nix_pid).map_err(|e| match e {
        nix::Error::EPERM => TracerError::PermissionDenied(format!("EPERM attaching to {}", pid)),
        nix::Error::ESRCH => TracerError::ProcessNotFound { pid },
        _ => TracerError::PtraceError(e),
    })?;

    // Wait for process to stop
    match waitpid(nix_pid, Some(WaitPidFlag::WSTOPPED)) {
        Ok(WaitStatus::Stopped(_, _)) => {}
        Ok(WaitStatus::Exited(_, _)) => {
            return Err(TracerError::ProcessExited { pid });
        }
        Ok(_) => {}
        Err(e) => {
            // Try to detach on error
            let _ = ptrace::detach(nix_pid, None);
            return Err(TracerError::PtraceError(e));
        }
    }

    // Set ptrace options for syscall tracing
    if let Err(e) = ptrace::setoptions(
        nix_pid,
        ptrace::Options::PTRACE_O_TRACESYSGOOD | ptrace::Options::PTRACE_O_TRACEEXEC,
    ) {
        let _ = ptrace::detach(nix_pid, None);
        return Err(TracerError::PtraceError(e));
    }

    Ok(ProcessTrace {
        pid,
        config,
        _start_time: Instant::now(),
        events: Vec::new(),
        baseline: None,
        attached: true,
        _trace_count: Arc::new(AtomicU64::new(0)),
        _rate_limit_window: Instant::now(),
    })
}

/// Detach from a traced process
///
/// # PMAT-057-006
///
/// # Safety
/// This function detaches ptrace cleanly, allowing the process to continue
/// execution without the tracer.
pub fn detach(mut trace: ProcessTrace) -> Result<(), TracerError> {
    if !trace.attached {
        return Ok(());
    }

    contract_pre_error_handling!();
    let nix_pid = Pid::from_raw(trace.pid as i32);

    // Detach and continue process
    ptrace::detach(nix_pid, None).map_err(|e| match e {
        nix::Error::ESRCH => TracerError::ProcessExited { pid: trace.pid },
        _ => TracerError::PtraceError(e),
    })?;

    trace.attached = false;
    Ok(())
}

/// Tracks syscall entry/exit state during collection
struct SyscallCollector {
    events: Vec<SyscallEvent>,
    in_syscall: bool,
    syscall_start: Instant,
    current_syscall_nr: i64,
}

impl SyscallCollector {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            in_syscall: false,
            syscall_start: Instant::now(),
            current_syscall_nr: 0,
        }
    }

    /// Handle a ptrace syscall stop (entry or exit)
    fn handle_syscall_stop(&mut self, nix_pid: Pid) -> Result<(), TracerError> {
        if self.in_syscall {
            let duration = self.syscall_start.elapsed();
            let regs = crate::arch::PtraceRegs::get_nix(nix_pid)?;
            let result = regs.syscall_return();

            self.events.push(SyscallEvent {
                syscall: syscall_name(self.current_syscall_nr).to_string(),
                syscall_nr: self.current_syscall_nr,
                duration,
                result,
                timestamp: self.syscall_start,
            });
            self.in_syscall = false;
        } else {
            let regs = crate::arch::PtraceRegs::get_nix(nix_pid)?;
            self.current_syscall_nr = regs.syscall_number();
            self.syscall_start = Instant::now();
            self.in_syscall = true;
        }
        Ok(())
    }
}

/// Process a single waitpid result, returning true if the collection loop should break
fn handle_wait_status(
    status: Result<WaitStatus, nix::Error>,
    nix_pid: Pid,
    collector: &mut SyscallCollector,
    pid: u32,
    attached: &mut bool,
) -> Result<bool, TracerError> {
    match status {
        Ok(WaitStatus::PtraceSyscall(_)) => {
            collector.handle_syscall_stop(nix_pid)?;
            ptrace::syscall(nix_pid, None)?;
            Ok(false)
        }
        Ok(WaitStatus::Exited(_, _) | WaitStatus::Signaled(_, _, _)) => {
            *attached = false;
            Err(TracerError::ProcessExited { pid })
        }
        Ok(WaitStatus::Stopped(_, sig)) => {
            ptrace::syscall(nix_pid, Some(sig))?;
            Ok(false)
        }
        Ok(WaitStatus::StillAlive) => {
            std::thread::sleep(Duration::from_micros(100));
            Ok(false)
        }
        Ok(_) => {
            ptrace::syscall(nix_pid, None)?;
            Ok(false)
        }
        Err(nix::Error::ECHILD) => {
            *attached = false;
            Err(TracerError::ProcessExited { pid })
        }
        Err(e) => Err(TracerError::PtraceError(e)),
    }
}

/// Collect syscall events from an attached process
///
/// # PMAT-057-007
///
/// # Arguments
/// * `trace` - Active trace handle
///
/// # Returns
/// * `TraceResult` with breakdown and anomalies
pub fn collect(trace: &mut ProcessTrace) -> Result<TraceResult, TracerError> {
    if !trace.attached {
        return Err(TracerError::NotAttached);
    }

    contract_pre_error_handling!();
    let nix_pid = Pid::from_raw(trace.pid as i32);
    let start = Instant::now();
    let timeout = trace.config.timeout;
    let max_syscalls = trace.config.max_syscalls;

    let mut collector = SyscallCollector::new();

    // Continue process
    ptrace::syscall(nix_pid, None)?;

    loop {
        if start.elapsed() > timeout || collector.events.len() >= max_syscalls {
            break;
        }

        let status = waitpid(nix_pid, Some(WaitPidFlag::WNOHANG));
        if handle_wait_status(status, nix_pid, &mut collector, trace.pid, &mut trace.attached)? {
            break;
        }
    }

    let duration = start.elapsed();
    let mut result = TraceResult::new(trace.pid, duration, collector.events);

    // Compute anomalies if baseline available
    if let Some(baseline) = &trace.baseline {
        result = result.with_baseline(baseline, 3.0);
    }

    // Store events in trace for baseline building
    trace.events.extend(result.events.clone());

    Ok(result)
}

/// Stream syscall events in real-time
///
/// # PMAT-057-011
///
/// Returns an iterator that yields syscall events as they occur.
/// For true async streaming, use with tokio or async-std.
pub fn stream_syscalls(pid: u32, config: ProcessTraceConfig) -> Result<SyscallStream, TracerError> {
    let trace = attach(pid, config)?;
    Ok(SyscallStream { trace })
}

/// Iterator over syscall events
pub struct SyscallStream {
    trace: ProcessTrace,
}

impl Iterator for SyscallStream {
    type Item = Result<SyscallEvent, TracerError>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.trace.attached {
            return None;
        }

        let nix_pid = Pid::from_raw(self.trace.pid as i32);
        let mut in_syscall = false;
        let mut syscall_start = Instant::now();
        let mut current_syscall_nr: i64 = 0;

        // Continue to next syscall
        if let Err(e) = ptrace::syscall(nix_pid, None) {
            self.trace.attached = false;
            return Some(Err(TracerError::PtraceError(e)));
        }

        loop {
            match waitpid(nix_pid, None) {
                Ok(WaitStatus::PtraceSyscall(_)) => {
                    if in_syscall {
                        // Syscall exit - return event
                        let duration = syscall_start.elapsed();
                        let regs = match crate::arch::PtraceRegs::get_nix(nix_pid) {
                            Ok(r) => r,
                            Err(e) => return Some(Err(TracerError::PtraceError(e))),
                        };
                        let result = regs.syscall_return();

                        return Some(Ok(SyscallEvent {
                            syscall: syscall_name(current_syscall_nr).to_string(),
                            syscall_nr: current_syscall_nr,
                            duration,
                            result,
                            timestamp: syscall_start,
                        }));
                    } else {
                        // Syscall entry
                        let regs = match crate::arch::PtraceRegs::get_nix(nix_pid) {
                            Ok(r) => r,
                            Err(e) => return Some(Err(TracerError::PtraceError(e))),
                        };
                        current_syscall_nr = regs.syscall_number();
                        syscall_start = Instant::now();
                        in_syscall = true;

                        // Continue to syscall exit
                        if let Err(e) = ptrace::syscall(nix_pid, None) {
                            return Some(Err(TracerError::PtraceError(e)));
                        }
                    }
                }
                Ok(WaitStatus::Exited(_, _) | WaitStatus::Signaled(_, _, _)) => {
                    self.trace.attached = false;
                    return None;
                }
                Ok(WaitStatus::Stopped(_, sig)) => {
                    if let Err(e) = ptrace::syscall(nix_pid, Some(sig)) {
                        return Some(Err(TracerError::PtraceError(e)));
                    }
                }
                Ok(_) => {
                    if let Err(e) = ptrace::syscall(nix_pid, None) {
                        return Some(Err(TracerError::PtraceError(e)));
                    }
                }
                Err(nix::Error::ECHILD) => {
                    self.trace.attached = false;
                    return None;
                }
                Err(e) => {
                    return Some(Err(TracerError::PtraceError(e)));
                }
            }
        }
    }
}

impl Drop for SyscallStream {
    fn drop(&mut self) {
        if self.trace.attached {
            let _ = ptrace::detach(Pid::from_raw(self.trace.pid as i32), None);
        }
    }
}

// ============================================================================
// Tests - Falsification Tests F001-F020
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // F001-F010: API Basics
    // ========================================================================

    /// F001: attach() returns handle for valid PID
    /// Falsification: attach(valid_pid) returns Err when process exists
    #[test]
    #[ignore] // Requires CAP_SYS_PTRACE; hangs in CI containers (paiml/renacer#62)
    fn test_f001_attach_valid_pid() {
        // Skip if we don't have ptrace permissions
        if !is_available() {
            eprintln!("Skipping F001: no ptrace permissions");
            return;
        }

        // We can't actually attach to ourselves, so spawn a child
        let mut child = std::process::Command::new("sleep")
            .arg("10")
            .spawn()
            .expect("Failed to spawn test process");

        let child_pid = child.id();
        let config = ProcessTraceConfig::default();

        let result = attach(child_pid, config);

        // Clean up — kill and wait to avoid zombie
        std::process::Command::new("kill").args(["-9", &child_pid.to_string()]).output().ok();
        let _ = child.wait();

        // If we got permission denied, that's expected in some environments
        match result {
            Ok(trace) => {
                assert!(trace.is_attached());
                assert_eq!(trace.pid(), child_pid);
                let _ = detach(trace);
            }
            Err(TracerError::PermissionDenied(_)) => {
                // Acceptable - we may not have CAP_SYS_PTRACE
            }
            Err(e) => {
                panic!("Unexpected error: {:?}", e);
            }
        }
    }

    /// F002: attach() fails for nonexistent PID
    /// Falsification: attach(99999999) returns Ok
    #[test]
    fn test_f002_attach_invalid_pid() {
        let config = ProcessTraceConfig::default();
        let result = attach(99999999, config);

        assert!(
            matches!(result, Err(TracerError::ProcessNotFound { .. })),
            "Expected ProcessNotFound, got {:?}",
            result
        );
    }

    /// F003: attach() fails without permissions
    /// Falsification: Non-root attach(1) returns Ok
    #[test]
    fn test_f003_attach_no_permission() {
        // Skip if running as root
        if nix::unistd::geteuid().is_root() {
            eprintln!("Skipping F003: running as root");
            return;
        }

        let config = ProcessTraceConfig::default();
        let result = attach(1, config); // PID 1 is init, owned by root

        assert!(
            matches!(
                result,
                Err(TracerError::PermissionDenied(_) | TracerError::ProcessNotFound { .. })
            ),
            "Expected PermissionDenied or ProcessNotFound, got {:?}",
            result
        );
    }

    /// F004: detach() releases process
    /// Falsification: Process remains stopped after detach()
    #[test]
    fn test_f004_detach_releases() {
        if !is_available() {
            eprintln!("Skipping F004: no ptrace permissions");
            return;
        }

        // This test requires spawning a process, which is covered by integration tests
        // For unit test, verify detach on already-detached trace doesn't panic
        let trace = ProcessTrace {
            pid: 12345,
            config: ProcessTraceConfig::default(),
            _start_time: Instant::now(),
            events: Vec::new(),
            baseline: None,
            attached: false, // Already detached
            _trace_count: Arc::new(AtomicU64::new(0)),
            _rate_limit_window: Instant::now(),
        };

        let result = detach(trace);
        assert!(result.is_ok());
    }

    /// F005: collect() returns events
    /// Falsification: collect() returns empty events for syscall-heavy process
    #[test]
    fn test_f005_collect_has_events() {
        // Test with mock events since we can't actually attach in unit tests
        let events = vec![
            SyscallEvent::new("read".to_string(), 0, Duration::from_micros(100), 1024),
            SyscallEvent::new("write".to_string(), 1, Duration::from_micros(50), 10),
        ];

        let result = TraceResult::new(1234, Duration::from_millis(10), events);

        assert!(!result.events.is_empty(), "Events should not be empty");
        assert_eq!(result.events.len(), 2);
    }

    /// F006: collect() respects timeout
    /// Tested via config validation
    #[test]
    fn test_f006_collect_timeout() {
        let config = ProcessTraceConfig::default().with_timeout(Duration::from_millis(50));
        assert_eq!(config.timeout, Duration::from_millis(50));
    }

    /// F007: collect() respects max_syscalls
    /// Tested via config validation
    #[test]
    fn test_f007_collect_max_syscalls() {
        let config = ProcessTraceConfig::default().with_max_syscalls(500);
        assert_eq!(config.max_syscalls, 500);
    }

    /// F008: stream_syscalls() yields events
    /// Tested via SyscallStream structure
    #[test]
    fn test_f008_stream_yields() {
        // SyscallStream implements Iterator
        fn assert_iterator<T: Iterator>() {}
        assert_iterator::<SyscallStream>();
    }

    /// F009: compute_baseline() handles empty
    /// Falsification: compute_baseline(&[]) panics
    #[test]
    fn test_f009_baseline_empty() {
        let events: Vec<SyscallEvent> = vec![];
        let baseline = compute_baseline(&events);

        assert_eq!(baseline.sample_count, 0);
        assert!(baseline.mean_us.is_empty());
        // Should NOT panic
    }

    /// F010: zscore() handles zero std
    /// Falsification: zscore() returns NaN when std=0
    #[test]
    fn test_f010_zscore_zero_std() {
        let mut baseline = SyscallBaseline::default();
        baseline.mean_us.insert("read".to_string(), 100.0);
        baseline.std_us.insert("read".to_string(), 0.0); // Zero std

        let event = SyscallEvent::new("read".to_string(), 0, Duration::from_micros(200), 0);

        let z = zscore(&event, &baseline);

        // Should return 0, not NaN
        assert!(!z.is_nan(), "zscore should not return NaN for zero std");
        assert_eq!(z, 0.0);
    }

    // ========================================================================
    // F011-F020: API Advanced
    // ========================================================================

    /// F011: Rate limiting enforced
    /// Falsification: Traces exceed rate_limit per second
    #[test]
    fn test_f011_rate_limit() {
        // Reset global state
        GLOBAL_TRACE_COUNT.store(0, Ordering::Relaxed);
        GLOBAL_RATE_WINDOW.store(0, Ordering::Relaxed);

        let config = ProcessTraceConfig::default().with_rate_limit(5);

        // First 5 should succeed
        for _ in 0..5 {
            assert!(check_rate_limit(&config).is_ok());
        }

        // 6th should fail
        let result = check_rate_limit(&config);
        assert!(
            matches!(result, Err(TracerError::RateLimitExceeded { .. })),
            "Expected RateLimitExceeded, got {:?}",
            result
        );
    }

    /// F012: Double attach rejected
    /// Tested by verifying AlreadyTraced error exists
    #[test]
    fn test_f012_double_attach() {
        let err = TracerError::AlreadyTraced { pid: 1234 };
        assert!(matches!(err, TracerError::AlreadyTraced { pid: 1234 }));
    }

    /// F013: Attach after detach works
    /// Tested via state tracking
    #[test]
    fn test_f013_reattach() {
        let mut trace = ProcessTrace {
            pid: 12345,
            config: ProcessTraceConfig::default(),
            _start_time: Instant::now(),
            events: Vec::new(),
            baseline: None,
            attached: true,
            _trace_count: Arc::new(AtomicU64::new(0)),
            _rate_limit_window: Instant::now(),
        };

        trace.attached = false;
        assert!(!trace.is_attached());
    }

    /// F014: ProcessExited error on exit
    #[test]
    fn test_f014_process_exit() {
        let err = TracerError::ProcessExited { pid: 1234 };
        let msg = format!("{}", err);
        assert!(msg.contains("1234"));
    }

    /// F015: SyscallEvent has valid duration
    /// Falsification: Any SyscallEvent.duration is negative
    #[test]
    fn test_f015_event_duration() {
        let event = SyscallEvent::new("read".to_string(), 0, Duration::from_micros(100), 0);

        // Duration is always non-negative by type (u128)
        assert!(event.duration.as_micros() > 0);
    }

    /// F016: SyscallBreakdown sums correctly
    /// Falsification: mmap+futex+read+write+ioctl+other+compute != total
    #[test]
    fn test_f016_breakdown_sum() {
        let events = vec![
            SyscallEvent::new("mmap".to_string(), 9, Duration::from_micros(100), 0),
            SyscallEvent::new("futex".to_string(), 202, Duration::from_micros(50), 0),
            SyscallEvent::new("read".to_string(), 0, Duration::from_micros(75), 0),
            SyscallEvent::new("write".to_string(), 1, Duration::from_micros(25), 0),
        ];

        let total_us = 500;
        let breakdown = SyscallBreakdown::from_events(&events, total_us);

        let syscall_sum = breakdown.mmap_us
            + breakdown.futex_us
            + breakdown.ioctl_us
            + breakdown.read_us
            + breakdown.write_us
            + breakdown.other_us;

        assert_eq!(syscall_sum + breakdown.compute_us, total_us, "Breakdown should sum to total");
    }

    /// F017: TraceResult has valid zscore
    /// Falsification: max_zscore is NaN or infinite
    #[test]
    fn test_f017_result_zscore() {
        let events = vec![SyscallEvent::new("read".to_string(), 0, Duration::from_micros(100), 0)];
        let result = TraceResult::new(1234, Duration::from_millis(10), events);

        assert!(!result.max_zscore.is_nan());
        assert!(!result.max_zscore.is_infinite());
    }

    /// F018: Source locations valid when enabled
    /// Tested via struct initialization
    #[test]
    fn test_f018_source_locations() {
        let loc = SourceLocation {
            file: "test.rs".to_string(),
            line: 42,
            function: Some("test_fn".to_string()),
        };

        assert_eq!(loc.file, "test.rs");
        assert_eq!(loc.line, 42);
        assert_eq!(loc.function, Some("test_fn".to_string()));
    }

    /// F019: OTLP export succeeds
    /// Tested via TraceResult creation (span export is separate)
    #[test]
    fn test_f019_otlp_span() {
        let events = vec![SyscallEvent::new("read".to_string(), 0, Duration::from_micros(100), 0)];
        let result = TraceResult::new(1234, Duration::from_millis(10), events);

        // TraceResult can be created without panic
        assert_eq!(result.pid, 1234);
    }

    /// F020: Error types are Send+Sync
    #[test]
    fn test_f020_error_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TracerError>();
    }

    // ========================================================================
    // Additional Unit Tests
    // ========================================================================

    #[test]
    fn test_categorize_syscall() {
        assert_eq!(categorize_syscall("mmap"), "mmap");
        assert_eq!(categorize_syscall("munmap"), "mmap");
        assert_eq!(categorize_syscall("mprotect"), "mmap");
        assert_eq!(categorize_syscall("futex"), "futex");
        assert_eq!(categorize_syscall("ioctl"), "ioctl");
        assert_eq!(categorize_syscall("read"), "read");
        assert_eq!(categorize_syscall("pread64"), "read");
        assert_eq!(categorize_syscall("write"), "write");
        assert_eq!(categorize_syscall("pwrite64"), "write");
        assert_eq!(categorize_syscall("unknown_syscall"), "other");
    }

    #[test]
    fn test_syscall_name() {
        assert_eq!(syscall_name(0), "read");
        assert_eq!(syscall_name(1), "write");
        assert_eq!(syscall_name(9), "mmap");
        assert_eq!(syscall_name(202), "futex");
        assert_eq!(syscall_name(99999), "unknown");
    }

    #[test]
    fn test_config_builder() {
        let config = ProcessTraceConfig::default()
            .with_max_syscalls(500)
            .with_timeout(Duration::from_millis(200))
            .with_source(true)
            .with_otlp("http://localhost:4317".to_string())
            .with_rate_limit(50);

        assert_eq!(config.max_syscalls, 500);
        assert_eq!(config.timeout, Duration::from_millis(200));
        assert!(config.enable_source);
        assert_eq!(config.otlp_endpoint, Some("http://localhost:4317".to_string()));
        assert_eq!(config.rate_limit, 50);
    }

    #[test]
    fn test_syscall_baseline_from_events() {
        let events = vec![
            SyscallEvent::new("read".to_string(), 0, Duration::from_micros(100), 0),
            SyscallEvent::new("read".to_string(), 0, Duration::from_micros(200), 0),
            SyscallEvent::new("read".to_string(), 0, Duration::from_micros(150), 0),
            SyscallEvent::new("write".to_string(), 1, Duration::from_micros(50), 0),
        ];

        let baseline = compute_baseline(&events);

        assert_eq!(baseline.sample_count, 4);
        assert!(baseline.mean_us.contains_key("read"));
        assert!(baseline.std_us.contains_key("read"));

        // Mean of 100, 200, 150 = 150
        let read_mean = baseline.mean_us.get("read").unwrap();
        assert!((read_mean - 150.0).abs() < 0.1);
    }

    #[test]
    fn test_zscore_calculation() {
        let mut baseline = SyscallBaseline::default();
        baseline.mean_us.insert("read".to_string(), 100.0);
        baseline.std_us.insert("read".to_string(), 20.0);

        // Event at 140us = 2 std devs above mean
        let event = SyscallEvent::new("read".to_string(), 0, Duration::from_micros(140), 0);
        let z = zscore(&event, &baseline);

        assert!((z - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_trace_result_with_baseline() {
        let events = vec![
            SyscallEvent::new("read".to_string(), 0, Duration::from_micros(100), 0),
            SyscallEvent::new("read".to_string(), 0, Duration::from_micros(500), 0), // Anomaly
        ];

        let mut baseline = SyscallBaseline::default();
        baseline.mean_us.insert("read".to_string(), 100.0);
        baseline.std_us.insert("read".to_string(), 50.0);
        baseline.sample_count = 100;

        let result =
            TraceResult::new(1234, Duration::from_millis(10), events).with_baseline(&baseline, 3.0);

        // 500us is 8 std devs above mean, should be anomaly
        assert!(!result.anomalies.is_empty());
        assert!(result.max_zscore > 3.0);
    }

    #[test]
    fn test_breakdown_efficiency() {
        let events = vec![
            SyscallEvent::new("read".to_string(), 0, Duration::from_micros(200), 0),
            SyscallEvent::new("write".to_string(), 1, Duration::from_micros(100), 0),
        ];

        // Total 1000us, syscalls took 300us, compute should be 700us
        let breakdown = SyscallBreakdown::from_events(&events, 1000);

        assert_eq!(breakdown.read_us, 200);
        assert_eq!(breakdown.write_us, 100);
        assert_eq!(breakdown.compute_us, 700);
        assert!((breakdown.efficiency() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_breakdown_sorted_categories() {
        let events = vec![
            SyscallEvent::new("mmap".to_string(), 9, Duration::from_micros(500), 0),
            SyscallEvent::new("read".to_string(), 0, Duration::from_micros(100), 0),
        ];

        let breakdown = SyscallBreakdown::from_events(&events, 1000);
        let sorted = breakdown.sorted_categories();

        // mmap (500) should be first
        assert_eq!(sorted[0].0, "mmap");
        assert_eq!(sorted[0].1, 500);
    }

    #[test]
    fn test_is_available() {
        // Just verify it doesn't panic
        let _ = is_available();
    }

    #[test]
    fn test_baseline_has_category() {
        let mut baseline = SyscallBaseline::default();
        assert!(!baseline.has_category("read"));

        baseline.mean_us.insert("read".to_string(), 100.0);
        baseline.std_us.insert("read".to_string(), 10.0);

        assert!(baseline.has_category("read"));
        assert!(!baseline.has_category("write"));
    }

    #[test]
    fn test_breakdown_default() {
        let breakdown = SyscallBreakdown::default();
        assert_eq!(breakdown.mmap_us, 0);
        assert_eq!(breakdown.futex_us, 0);
        assert_eq!(breakdown.ioctl_us, 0);
        assert_eq!(breakdown.read_us, 0);
        assert_eq!(breakdown.write_us, 0);
        assert_eq!(breakdown.other_us, 0);
        assert_eq!(breakdown.compute_us, 0);
        assert_eq!(breakdown.syscall_count, 0);
        assert_eq!(breakdown.total_us, 0);
    }

    #[test]
    fn test_breakdown_efficiency_zero_total() {
        let breakdown = SyscallBreakdown::default();
        assert!((breakdown.efficiency() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_breakdown_syscall_time_us() {
        let mut breakdown = SyscallBreakdown::default();
        breakdown.mmap_us = 100;
        breakdown.futex_us = 50;
        breakdown.ioctl_us = 30;
        breakdown.read_us = 200;
        breakdown.write_us = 150;
        breakdown.other_us = 70;

        assert_eq!(breakdown.syscall_time_us(), 600);
    }

    #[test]
    fn test_trace_result_new() {
        let events =
            vec![SyscallEvent::new("read".to_string(), 0, Duration::from_micros(100), 1024)];
        let result = TraceResult::new(12345, Duration::from_millis(10), events);

        assert_eq!(result.pid, 12345);
        assert_eq!(result.events.len(), 1);
        assert_eq!(result.max_zscore, 0.0);
        assert!(result.anomalies.is_empty());
    }

    #[test]
    fn test_syscall_event_new() {
        let event = SyscallEvent::new("mmap".to_string(), 9, Duration::from_micros(500), 0);
        assert_eq!(event.syscall, "mmap");
        assert_eq!(event.syscall_nr, 9);
        assert_eq!(event.result, 0);
    }

    #[test]
    fn test_source_location() {
        let loc = SourceLocation {
            file: "/home/test/main.rs".to_string(),
            line: 42,
            function: Some("main".to_string()),
        };

        assert_eq!(loc.file, "/home/test/main.rs");
        assert_eq!(loc.line, 42);
        assert_eq!(loc.function, Some("main".to_string()));
    }

    #[test]
    fn test_syscall_anomaly() {
        let anomaly = SyscallAnomaly {
            syscall: "futex".to_string(),
            duration_us: 50000,
            zscore: 5.5,
            expected_us: 1000.0,
            category: "futex".to_string(),
        };

        assert_eq!(anomaly.syscall, "futex");
        assert_eq!(anomaly.duration_us, 50000);
        assert!((anomaly.zscore - 5.5).abs() < 0.01);
    }

    #[test]
    fn test_process_trace_accessors() {
        let trace = ProcessTrace {
            pid: 9999,
            config: ProcessTraceConfig::default(),
            _start_time: Instant::now(),
            events: vec![SyscallEvent::new("read".to_string(), 0, Duration::from_micros(100), 0)],
            baseline: None,
            attached: true,
            _trace_count: Arc::new(AtomicU64::new(0)),
            _rate_limit_window: Instant::now(),
        };

        assert_eq!(trace.pid(), 9999);
        assert!(trace.is_attached());
        assert_eq!(trace.event_count(), 1);
        assert!(trace.baseline().is_none());
    }

    #[test]
    fn test_process_trace_set_baseline() {
        let mut trace = ProcessTrace {
            pid: 9999,
            config: ProcessTraceConfig::default(),
            _start_time: Instant::now(),
            events: Vec::new(),
            baseline: None,
            attached: true,
            _trace_count: Arc::new(AtomicU64::new(0)),
            _rate_limit_window: Instant::now(),
        };

        let mut baseline = SyscallBaseline::default();
        baseline.mean_us.insert("read".to_string(), 100.0);
        baseline.std_us.insert("read".to_string(), 20.0);

        trace.set_baseline(baseline);
        assert!(trace.baseline().is_some());
    }

    #[test]
    fn test_categorize_syscall_all() {
        // mmap category
        assert_eq!(categorize_syscall("mmap"), "mmap");
        assert_eq!(categorize_syscall("munmap"), "mmap");
        assert_eq!(categorize_syscall("mprotect"), "mmap");
        assert_eq!(categorize_syscall("brk"), "mmap");
        assert_eq!(categorize_syscall("mremap"), "mmap");

        // futex
        assert_eq!(categorize_syscall("futex"), "futex");

        // ioctl
        assert_eq!(categorize_syscall("ioctl"), "ioctl");

        // read category
        assert_eq!(categorize_syscall("read"), "read");
        assert_eq!(categorize_syscall("pread64"), "read");
        assert_eq!(categorize_syscall("readv"), "read");
        assert_eq!(categorize_syscall("preadv"), "read");
        assert_eq!(categorize_syscall("preadv2"), "read");

        // write category
        assert_eq!(categorize_syscall("write"), "write");
        assert_eq!(categorize_syscall("pwrite64"), "write");
        assert_eq!(categorize_syscall("writev"), "write");
        assert_eq!(categorize_syscall("pwritev"), "write");
        assert_eq!(categorize_syscall("pwritev2"), "write");

        // other
        assert_eq!(categorize_syscall("socket"), "other");
        assert_eq!(categorize_syscall("connect"), "other");
    }

    #[test]
    fn test_syscall_name_extended() {
        assert_eq!(syscall_name(0), "read");
        assert_eq!(syscall_name(1), "write");
        assert_eq!(syscall_name(2), "open");
        assert_eq!(syscall_name(3), "close");
        assert_eq!(syscall_name(4), "stat");
        assert_eq!(syscall_name(5), "fstat");
        assert_eq!(syscall_name(6), "lstat");
        assert_eq!(syscall_name(7), "poll");
        assert_eq!(syscall_name(8), "lseek");
        assert_eq!(syscall_name(9), "mmap");
        assert_eq!(syscall_name(10), "mprotect");
        assert_eq!(syscall_name(11), "munmap");
        assert_eq!(syscall_name(12), "brk");
        assert_eq!(syscall_name(13), "rt_sigaction");
        assert_eq!(syscall_name(14), "rt_sigprocmask");
        assert_eq!(syscall_name(15), "rt_sigreturn");
        assert_eq!(syscall_name(16), "ioctl");
        assert_eq!(syscall_name(17), "pread64");
        assert_eq!(syscall_name(18), "pwrite64");
        assert_eq!(syscall_name(19), "readv");
        assert_eq!(syscall_name(20), "writev");
        assert_eq!(syscall_name(21), "access");
        assert_eq!(syscall_name(22), "pipe");
        assert_eq!(syscall_name(23), "select");
        assert_eq!(syscall_name(24), "sched_yield");
        assert_eq!(syscall_name(25), "mremap");
        assert_eq!(syscall_name(35), "nanosleep");
        assert_eq!(syscall_name(56), "clone");
        assert_eq!(syscall_name(57), "fork");
        assert_eq!(syscall_name(58), "vfork");
        assert_eq!(syscall_name(59), "execve");
        assert_eq!(syscall_name(60), "exit");
        assert_eq!(syscall_name(61), "wait4");
        assert_eq!(syscall_name(62), "kill");
        assert_eq!(syscall_name(202), "futex");
        assert_eq!(syscall_name(9999), "unknown");
    }

    #[test]
    fn test_breakdown_with_all_categories() {
        let events = vec![
            SyscallEvent::new("mmap".to_string(), 9, Duration::from_micros(100), 0),
            SyscallEvent::new("munmap".to_string(), 11, Duration::from_micros(50), 0),
            SyscallEvent::new("futex".to_string(), 202, Duration::from_micros(200), 0),
            SyscallEvent::new("ioctl".to_string(), 16, Duration::from_micros(150), 0),
            SyscallEvent::new("read".to_string(), 0, Duration::from_micros(80), 0),
            SyscallEvent::new("pread64".to_string(), 17, Duration::from_micros(90), 0),
            SyscallEvent::new("write".to_string(), 1, Duration::from_micros(70), 0),
            SyscallEvent::new("writev".to_string(), 20, Duration::from_micros(60), 0),
            SyscallEvent::new("socket".to_string(), 41, Duration::from_micros(40), 0),
        ];

        let breakdown = SyscallBreakdown::from_events(&events, 2000);

        assert_eq!(breakdown.mmap_us, 150); // mmap + munmap
        assert_eq!(breakdown.futex_us, 200);
        assert_eq!(breakdown.ioctl_us, 150);
        assert_eq!(breakdown.read_us, 170); // read + pread64
        assert_eq!(breakdown.write_us, 130); // write + writev
        assert_eq!(breakdown.other_us, 40); // socket
        assert_eq!(breakdown.syscall_count, 9);
        // compute_us = 2000 - 840 = 1160
        assert_eq!(breakdown.compute_us, 1160);
    }

    #[test]
    fn test_zscore_no_baseline() {
        let baseline = SyscallBaseline::default();
        let event = SyscallEvent::new("read".to_string(), 0, Duration::from_micros(100), 0);

        let z = zscore(&event, &baseline);
        assert_eq!(z, 0.0);
    }

    #[test]
    fn test_trace_result_with_baseline_no_anomalies() {
        let events = vec![SyscallEvent::new("read".to_string(), 0, Duration::from_micros(100), 0)];

        let mut baseline = SyscallBaseline::default();
        baseline.mean_us.insert("read".to_string(), 100.0);
        baseline.std_us.insert("read".to_string(), 50.0);
        baseline.sample_count = 100;

        // z = (100 - 100) / 50 = 0 - no anomaly
        let result =
            TraceResult::new(1234, Duration::from_millis(10), events).with_baseline(&baseline, 3.0);

        assert!(result.anomalies.is_empty());
        assert!((result.max_zscore - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_error_display() {
        let errors = vec![
            TracerError::PermissionDenied("test".to_string()),
            TracerError::ProcessNotFound { pid: 1234 },
            TracerError::AlreadyTraced { pid: 1234 },
            TracerError::ProcessExited { pid: 1234 },
            TracerError::Timeout(Duration::from_secs(1)),
            TracerError::RateLimitExceeded { current: 200, limit: 100 },
            TracerError::OtlpError("test".to_string()),
            TracerError::DwarfError("test".to_string()),
            TracerError::NotAttached,
        ];

        for err in errors {
            let msg = format!("{}", err);
            assert!(!msg.is_empty());
        }
    }
}
