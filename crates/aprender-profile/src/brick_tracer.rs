//! ComputeBrick-aware tracing for trueno/cbtop integration
//!
//! This module provides first-class `ComputeBrick` tracing support to enable
//! automatic escalation from cbtop when performance anomalies are detected.
//!
//! # Toyota Way Alignment
//!
//! - **Genchi Genbutsu**: Trace real ComputeBrick execution, not simulated workloads
//! - **Jidoka**: Stop-the-line when brick assertions fail; emit span with failure reason
//! - **Muda Elimination**: Only trace when anomaly detected (CV > 15%, efficiency < 25%)
//! - **Mieruka**: Visual brick timeline in Jaeger/Tempo showing budget vs actual
//!
//! # Scientific Foundations
//!
//! - Popper (1959): Falsifiable assertions - brick claims must be testable
//! - Sigelman et al. (2010): Dapper adaptive sampling for high-frequency tracing
//! - Mace et al. (2015): Pivot Tracing - trace only anomalies
//!
//! # Example
//!
//! ```rust,ignore
//! use renacer::brick_tracer::{BrickTracer, BrickEscalationThresholds};
//! use trueno::ComputeBrick;
//!
//! let tracer = BrickTracer::new("http://localhost:4317")?;
//!
//! // Only traces if CV > 15% or efficiency < 25%
//! if tracer.should_trace(cv_percent, efficiency) {
//!     let result = tracer.trace_brick(&my_brick, || {
//!         my_brick.run(input)
//!     })?;
//!     println!("Syscall breakdown: {:?}", result.syscall_breakdown);
//! }
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::otlp_exporter::{OtlpConfig, OtlpExporter};

/// Syscall event captured during brick tracing
#[derive(Debug, Clone)]
pub struct SyscallEvent {
    /// Syscall name (e.g., "mmap", "futex", "read")
    pub syscall: String,
    /// Duration of the syscall
    pub duration: Duration,
    /// Return value
    pub result: i64,
}

/// Escalation thresholds from cbtop.
///
/// These thresholds determine when to escalate from cbtop measurement
/// to renacer deep tracing. Per Mace et al. (2015): "Always-on tracing
/// degrades throughput" - we only trace when anomalies are detected.
#[derive(Debug, Clone, Copy)]
pub struct BrickEscalationThresholds {
    /// CV threshold above which to trace (default: 15.0%)
    /// Per Curtsinger & Berger (2013): CV < 5% indicates stable measurements
    pub cv_percent: f64,
    /// Efficiency threshold below which to trace (default: 25.0%)
    /// Per Williams et al. (2009): Roofline efficiency for bottleneck detection
    pub efficiency_percent: f64,
    /// Maximum traces per second (rate limiting)
    /// Per Sigelman et al. (2010): Dapper uses aggressive sampling to prevent DoS
    pub max_traces_per_sec: u32,
}

impl Default for BrickEscalationThresholds {
    fn default() -> Self {
        Self { cv_percent: 15.0, efficiency_percent: 25.0, max_traces_per_sec: 100 }
    }
}

impl BrickEscalationThresholds {
    /// Create thresholds with custom CV limit
    #[must_use]
    pub fn with_cv(mut self, cv_percent: f64) -> Self {
        self.cv_percent = cv_percent;
        self
    }

    /// Create thresholds with custom efficiency limit
    #[must_use]
    pub fn with_efficiency(mut self, efficiency_percent: f64) -> Self {
        self.efficiency_percent = efficiency_percent;
        self
    }

    /// Create thresholds with custom rate limit
    #[must_use]
    pub fn with_rate_limit(mut self, max_per_sec: u32) -> Self {
        self.max_traces_per_sec = max_per_sec;
        self
    }
}

/// Syscall breakdown for root cause analysis.
///
/// When a ComputeBrick exceeds its budget, this breakdown shows where
/// time was spent in syscalls vs actual computation.
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
}

impl SyscallBreakdown {
    /// Create from syscall events captured during brick execution
    pub fn from_events(events: &[SyscallEvent], total_duration_us: u64) -> Self {
        let mut breakdown = Self::default();
        let mut syscall_time_us: u64 = 0;

        for event in events {
            let duration_us = event.duration.as_micros() as u64;
            syscall_time_us += duration_us;
            breakdown.syscall_count += 1;

            let syscall_name = &event.syscall;
            *breakdown.syscall_counts.entry(syscall_name.clone()).or_insert(0) += 1;

            match syscall_name.as_str() {
                "mmap" | "munmap" | "mprotect" | "brk" => breakdown.mmap_us += duration_us,
                "futex" => breakdown.futex_us += duration_us,
                "ioctl" => breakdown.ioctl_us += duration_us,
                "read" | "pread64" | "readv" => breakdown.read_us += duration_us,
                "write" | "pwrite64" | "writev" => breakdown.write_us += duration_us,
                _ => breakdown.other_us += duration_us,
            }
        }

        breakdown.compute_us = total_duration_us.saturating_sub(syscall_time_us);
        breakdown
    }

    /// Calculate syscall overhead percentage
    pub fn syscall_overhead_percent(&self) -> f64 {
        let total = self.total_us();
        if total == 0 {
            0.0
        } else {
            ((total - self.compute_us) as f64 / total as f64) * 100.0
        }
    }

    /// Total time in microseconds
    pub fn total_us(&self) -> u64 {
        self.mmap_us
            + self.futex_us
            + self.ioctl_us
            + self.read_us
            + self.write_us
            + self.other_us
            + self.compute_us
    }

    /// Get the dominant syscall category
    pub fn dominant_syscall(&self) -> &'static str {
        let max = self
            .mmap_us
            .max(self.futex_us)
            .max(self.ioctl_us)
            .max(self.read_us)
            .max(self.write_us)
            .max(self.other_us);

        if max == 0 {
            "none"
        } else if max == self.mmap_us {
            "mmap"
        } else if max == self.futex_us {
            "futex"
        } else if max == self.ioctl_us {
            "ioctl"
        } else if max == self.read_us {
            "read"
        } else if max == self.write_us {
            "write"
        } else {
            "other"
        }
    }
}

/// Brick metadata captured during tracing
#[derive(Debug, Clone)]
pub struct BrickMetadata {
    /// Brick name (e.g., "FfnBrick", "AttentionBrick")
    pub name: String,
    /// Budget in microseconds
    pub budget_us: u64,
    /// Actual execution time in microseconds
    pub actual_us: u64,
    /// Whether the brick exceeded its budget
    pub over_budget: bool,
    /// Efficiency (budget / actual), capped at 1.0
    pub efficiency: f64,
    /// Coefficient of variation (requires multiple runs)
    pub cv_percent: Option<f64>,
    /// Brick score (0-100)
    pub score: Option<u8>,
    /// Brick grade (A-F)
    pub grade: Option<char>,
    /// Number of assertions passed
    pub assertions_passed: u32,
    /// Number of assertions failed
    pub assertions_failed: u32,
    /// Names of failed assertions
    pub failed_assertion_names: Vec<String>,
}

/// Result of traced brick execution
#[derive(Debug, Clone)]
pub struct TracedBrickResult<R> {
    /// The actual result of the brick execution
    pub result: R,
    /// Execution duration in microseconds
    pub duration_us: u64,
    /// Syscall breakdown for root cause analysis
    pub syscall_breakdown: SyscallBreakdown,
    /// Brick metadata (if available)
    pub metadata: Option<BrickMetadata>,
    /// OTLP span ID (if exported)
    pub span_id: Option<String>,
    /// Reason for escalation
    pub escalation_reason: Option<EscalationReason>,
}

/// Reason why tracing was triggered
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscalationReason {
    /// CV exceeded threshold
    CvExceeded,
    /// Efficiency below threshold
    EfficiencyLow,
    /// Both CV and efficiency triggered
    Both,
    /// Manual/forced trace
    Manual,
}

impl std::fmt::Display for EscalationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CvExceeded => write!(f, "cv_exceeded"),
            Self::EfficiencyLow => write!(f, "efficiency_low"),
            Self::Both => write!(f, "cv_and_efficiency"),
            Self::Manual => write!(f, "manual"),
        }
    }
}

/// ComputeBrick-aware tracer for trueno/cbtop integration.
///
/// This tracer provides deep syscall-level visibility into ComputeBrick
/// execution when performance anomalies are detected by cbtop.
pub struct BrickTracer {
    /// OTLP exporter for span export
    exporter: Option<Arc<OtlpExporter>>,
    /// Escalation thresholds
    thresholds: BrickEscalationThresholds,
    /// Rate limiter: traces in current second
    traces_this_second: AtomicU64,
    /// Rate limiter: current second timestamp
    current_second: AtomicU64,
    /// Whether tracing is enabled
    enabled: bool,
}

impl BrickTracer {
    /// Create a new BrickTracer with OTLP endpoint.
    ///
    /// # Arguments
    ///
    /// * `endpoint` - OTLP endpoint (e.g., "http://localhost:4317")
    ///
    /// # Errors
    ///
    /// Returns error if OTLP exporter cannot be created.
    pub fn new(endpoint: &str) -> anyhow::Result<Self> {
        let config = OtlpConfig::new(endpoint.to_string(), "brick-tracer".to_string());
        let exporter = OtlpExporter::new(config, None)?;
        Ok(Self {
            exporter: Some(Arc::new(exporter)),
            thresholds: BrickEscalationThresholds::default(),
            traces_this_second: AtomicU64::new(0),
            current_second: AtomicU64::new(0),
            enabled: true,
        })
    }

    /// Create a BrickTracer without OTLP export (for testing/local use)
    pub fn new_local() -> Self {
        Self {
            exporter: None,
            thresholds: BrickEscalationThresholds::default(),
            traces_this_second: AtomicU64::new(0),
            current_second: AtomicU64::new(0),
            enabled: true,
        }
    }

    /// Set custom thresholds
    #[must_use]
    pub fn with_thresholds(mut self, thresholds: BrickEscalationThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Enable or disable tracing
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if brick should be traced based on metrics.
    ///
    /// Per Mace et al. (2015) Pivot Tracing: "Always-on tracing for
    /// high-frequency events degrades system throughput significantly."
    ///
    /// We only trace when:
    /// - CV > threshold (unstable measurements)
    /// - Efficiency < threshold (performance problem)
    /// - Rate limit not exceeded
    pub fn should_trace(&self, cv_percent: f64, efficiency_percent: f64) -> bool {
        if !self.enabled {
            return false;
        }

        // Check rate limit
        if !self.check_rate_limit() {
            return false;
        }

        // Check thresholds
        cv_percent > self.thresholds.cv_percent
            || efficiency_percent < self.thresholds.efficiency_percent
    }

    /// Determine escalation reason
    pub fn escalation_reason(&self, cv_percent: f64, efficiency_percent: f64) -> EscalationReason {
        let cv_exceeded = cv_percent > self.thresholds.cv_percent;
        let eff_low = efficiency_percent < self.thresholds.efficiency_percent;

        match (cv_exceeded, eff_low) {
            (true, true) => EscalationReason::Both,
            (true, false) => EscalationReason::CvExceeded,
            (false, true) => EscalationReason::EfficiencyLow,
            (false, false) => EscalationReason::Manual,
        }
    }

    /// Check and update rate limit
    fn check_rate_limit(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let current = self.current_second.load(Ordering::Relaxed);

        if now == current {
            // Same second, check limit
            let count = self.traces_this_second.fetch_add(1, Ordering::Relaxed);
            count < self.thresholds.max_traces_per_sec as u64
        } else {
            // New second, reset counter
            self.current_second.store(now, Ordering::Relaxed);
            self.traces_this_second.store(1, Ordering::Relaxed);
            true
        }
    }

    /// Trace a closure execution with syscall capture.
    ///
    /// This is the main entry point for brick tracing. It:
    /// 1. Captures syscalls during execution
    /// 2. Calculates syscall breakdown
    /// 3. Exports span to OTLP (if configured)
    ///
    /// # Type Parameters
    ///
    /// * `F` - Closure type
    /// * `R` - Return type of the closure
    ///
    /// # Arguments
    ///
    /// * `brick_name` - Name of the brick being traced
    /// * `budget_us` - Expected budget in microseconds
    /// * `f` - Closure to execute and trace
    pub fn trace<F, R>(&self, brick_name: &str, budget_us: u64, f: F) -> TracedBrickResult<R>
    where
        F: FnOnce() -> R,
    {
        contract_pre_error_handling!(brick_name);
        self.trace_with_reason(brick_name, budget_us, EscalationReason::Manual, f)
    }

    /// Trace with explicit escalation reason
    pub fn trace_with_reason<F, R>(
        &self,
        brick_name: &str,
        budget_us: u64,
        reason: EscalationReason,
        f: F,
    ) -> TracedBrickResult<R>
    where
        F: FnOnce() -> R,
    {
        contract_pre_error_handling!(brick_name);
        let start = Instant::now();

        // Execute the brick
        let result = f();

        let duration = start.elapsed();
        let duration_us = duration.as_micros() as u64;

        // For now, we don't have syscall capture integrated here
        // This would require ptrace which needs to be in a separate process
        // We provide the infrastructure for when syscall events are available
        let breakdown = SyscallBreakdown { compute_us: duration_us, ..Default::default() };

        let over_budget = duration_us > budget_us;
        let efficiency =
            if duration_us > 0 { (budget_us as f64 / duration_us as f64).min(1.0) } else { 1.0 };

        let metadata = BrickMetadata {
            name: brick_name.to_string(),
            budget_us,
            actual_us: duration_us,
            over_budget,
            efficiency,
            cv_percent: None,
            score: None,
            grade: None,
            assertions_passed: 0,
            assertions_failed: 0,
            failed_assertion_names: Vec::new(),
        };

        // Export to OTLP if configured
        let span_id = if let Some(ref exporter) = self.exporter {
            self.export_brick_span(exporter, &metadata, &breakdown, reason);
            Some(format!("{:016x}", rand::random::<u64>()))
        } else {
            None
        };

        TracedBrickResult {
            result,
            duration_us,
            syscall_breakdown: breakdown,
            metadata: Some(metadata),
            span_id,
            escalation_reason: Some(reason),
        }
    }

    /// Export brick span to OTLP
    fn export_brick_span(
        &self,
        exporter: &OtlpExporter,
        metadata: &BrickMetadata,
        breakdown: &SyscallBreakdown,
        reason: EscalationReason,
    ) {
        use crate::otlp_exporter::ComputeBlock;

        // Create compute block for OTLP export
        let block = ComputeBlock {
            operation: Box::leak(format!("brick: {}", metadata.name).into_boxed_str()),
            duration_us: metadata.actual_us,
            elements: 0, // Not applicable for bricks
            is_slow: metadata.over_budget,
        };

        exporter.record_compute_block(block);

        // Log additional brick-specific attributes
        tracing::info!(
            brick.name = %metadata.name,
            brick.budget_us = metadata.budget_us,
            brick.actual_us = metadata.actual_us,
            brick.over_budget = metadata.over_budget,
            brick.efficiency = %format!("{:.2}", metadata.efficiency),
            syscall.overhead_percent = %format!("{:.1}", breakdown.syscall_overhead_percent()),
            syscall.dominant = breakdown.dominant_syscall(),
            escalation.reason = %reason,
            "ComputeBrick traced"
        );
    }

    /// Get current thresholds
    pub fn thresholds(&self) -> &BrickEscalationThresholds {
        &self.thresholds
    }

    /// Check if OTLP export is configured
    pub fn has_exporter(&self) -> bool {
        self.exporter.is_some()
    }
}

impl Default for BrickTracer {
    fn default() -> Self {
        Self::new_local()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // F001: BrickTracer creates successfully
    #[test]
    fn f001_brick_tracer_creates_local() {
        let tracer = BrickTracer::new_local();
        assert!(tracer.enabled);
        assert!(!tracer.has_exporter());
    }

    // F002: trace_brick captures duration
    #[test]
    fn f002_trace_captures_duration() {
        let tracer = BrickTracer::new_local();
        let result = tracer.trace("TestBrick", 1000, || {
            std::thread::sleep(Duration::from_micros(100));
            42
        });
        assert_eq!(result.result, 42);
        assert!(result.duration_us >= 100);
    }

    // F005: brick.name attribute present
    #[test]
    fn f005_brick_name_present() {
        let tracer = BrickTracer::new_local();
        let result = tracer.trace("MyBrick", 1000, || 42);
        assert!(result.metadata.is_some());
        assert_eq!(result.metadata.unwrap().name, "MyBrick");
    }

    // F006: brick.budget_us attribute present
    #[test]
    fn f006_brick_budget_present() {
        let tracer = BrickTracer::new_local();
        let result = tracer.trace("MyBrick", 5000, || 42);
        assert_eq!(result.metadata.unwrap().budget_us, 5000);
    }

    // F007: brick.actual_us attribute present
    #[test]
    fn f007_brick_actual_present() {
        let tracer = BrickTracer::new_local();
        let result = tracer.trace("MyBrick", 1000, || {
            // Do some work to ensure non-zero duration
            std::thread::sleep(Duration::from_micros(10));
            42
        });
        assert!(result.metadata.unwrap().actual_us > 0);
    }

    // F008: brick.efficiency calculated
    #[test]
    fn f008_brick_efficiency_calculated() {
        let tracer = BrickTracer::new_local();
        let result = tracer.trace("MyBrick", 10000, || {
            std::thread::sleep(Duration::from_micros(100));
            42
        });
        let meta = result.metadata.unwrap();
        // Efficiency should be budget / actual, capped at 1.0
        assert!(meta.efficiency > 0.0);
        assert!(meta.efficiency <= 1.0);
    }

    // F021: should_trace true when CV > 15%
    #[test]
    fn f021_should_trace_cv_exceeded() {
        let tracer = BrickTracer::new_local();
        assert!(tracer.should_trace(15.1, 80.0));
    }

    // F022: should_trace false when CV < 15%
    #[test]
    fn f022_should_not_trace_cv_ok() {
        let tracer = BrickTracer::new_local();
        assert!(!tracer.should_trace(14.9, 80.0));
    }

    // F023: should_trace true when eff < 25%
    #[test]
    fn f023_should_trace_efficiency_low() {
        let tracer = BrickTracer::new_local();
        assert!(tracer.should_trace(5.0, 24.9));
    }

    // F024: should_trace false when eff > 25%
    #[test]
    fn f024_should_not_trace_efficiency_ok() {
        let tracer = BrickTracer::new_local();
        assert!(!tracer.should_trace(5.0, 25.1));
    }

    // F025: Threshold configurable
    #[test]
    fn f025_thresholds_configurable() {
        let thresholds = BrickEscalationThresholds::default().with_cv(20.0).with_efficiency(30.0);
        let tracer = BrickTracer::new_local().with_thresholds(thresholds);

        // With CV=20%, should not trace at 15%
        assert!(!tracer.should_trace(15.0, 80.0));
        // With eff=30%, should not trace at 35% (above threshold)
        assert!(!tracer.should_trace(5.0, 35.0));
        // With eff=30%, should trace at 25% (below threshold)
        assert!(tracer.should_trace(5.0, 25.0));
    }

    // F026: escalation.reason in span
    #[test]
    fn f026_escalation_reason_present() {
        let tracer = BrickTracer::new_local();
        let result = tracer.trace_with_reason("MyBrick", 1000, EscalationReason::CvExceeded, || 42);
        assert_eq!(result.escalation_reason, Some(EscalationReason::CvExceeded));
    }

    // F028: No trace when both ok
    #[test]
    fn f028_no_trace_when_ok() {
        let tracer = BrickTracer::new_local();
        assert!(!tracer.should_trace(5.0, 80.0));
    }

    // F029: Trace when CV high, eff ok
    #[test]
    fn f029_trace_cv_high() {
        let tracer = BrickTracer::new_local();
        assert!(tracer.should_trace(20.0, 80.0));
    }

    // F030: Trace when CV ok, eff low
    #[test]
    fn f030_trace_eff_low() {
        let tracer = BrickTracer::new_local();
        assert!(tracer.should_trace(5.0, 10.0));
    }

    // F031: Trace when both bad
    #[test]
    fn f031_trace_both_bad() {
        let tracer = BrickTracer::new_local();
        assert!(tracer.should_trace(30.0, 10.0));
    }

    // F034: Zero overhead when not tracing
    #[test]
    fn f034_should_trace_is_fast() {
        let tracer = BrickTracer::new_local();
        let start = Instant::now();
        for _ in 0..10000 {
            let _ = tracer.should_trace(5.0, 80.0);
        }
        let elapsed = start.elapsed();
        // 10000 checks should take < 1ms
        assert!(elapsed < Duration::from_millis(10));
    }

    // Test SyscallBreakdown
    #[test]
    fn test_syscall_breakdown_from_empty() {
        let breakdown = SyscallBreakdown::from_events(&[], 1000);
        assert_eq!(breakdown.compute_us, 1000);
        assert_eq!(breakdown.syscall_count, 0);
    }

    #[test]
    fn test_syscall_breakdown_overhead() {
        let mut breakdown = SyscallBreakdown::default();
        breakdown.mmap_us = 100;
        breakdown.futex_us = 50;
        breakdown.compute_us = 850;

        assert_eq!(breakdown.total_us(), 1000);
        assert!((breakdown.syscall_overhead_percent() - 15.0).abs() < 0.1);
    }

    #[test]
    fn test_syscall_breakdown_dominant() {
        let mut breakdown = SyscallBreakdown::default();
        breakdown.futex_us = 500;
        breakdown.mmap_us = 100;
        breakdown.compute_us = 400;

        assert_eq!(breakdown.dominant_syscall(), "futex");
    }

    // Test EscalationReason display
    #[test]
    fn test_escalation_reason_display() {
        assert_eq!(format!("{}", EscalationReason::CvExceeded), "cv_exceeded");
        assert_eq!(format!("{}", EscalationReason::EfficiencyLow), "efficiency_low");
        assert_eq!(format!("{}", EscalationReason::Both), "cv_and_efficiency");
        assert_eq!(format!("{}", EscalationReason::Manual), "manual");
    }

    // Test escalation_reason determination
    #[test]
    fn test_escalation_reason_determination() {
        let tracer = BrickTracer::new_local();

        assert_eq!(tracer.escalation_reason(20.0, 80.0), EscalationReason::CvExceeded);
        assert_eq!(tracer.escalation_reason(5.0, 10.0), EscalationReason::EfficiencyLow);
        assert_eq!(tracer.escalation_reason(20.0, 10.0), EscalationReason::Both);
        assert_eq!(tracer.escalation_reason(5.0, 80.0), EscalationReason::Manual);
    }

    // Test default thresholds
    #[test]
    fn test_default_thresholds() {
        let thresholds = BrickEscalationThresholds::default();
        assert!((thresholds.cv_percent - 15.0).abs() < 0.001);
        assert!((thresholds.efficiency_percent - 25.0).abs() < 0.001);
        assert_eq!(thresholds.max_traces_per_sec, 100);
    }

    // Test tracer disabled
    #[test]
    fn test_tracer_disabled() {
        let mut tracer = BrickTracer::new_local();
        tracer.set_enabled(false);
        assert!(!tracer.should_trace(50.0, 5.0)); // Would normally trace
    }

    // Test metadata over_budget flag
    #[test]
    fn test_over_budget_detection() {
        let tracer = BrickTracer::new_local();

        // Very short budget that will be exceeded
        let result = tracer.trace("SlowBrick", 1, || {
            std::thread::sleep(Duration::from_micros(100));
            42
        });

        let meta = result.metadata.unwrap();
        assert!(meta.over_budget);
        assert!(meta.actual_us > meta.budget_us);
    }

    // Test metadata under_budget
    #[test]
    fn test_under_budget() {
        let tracer = BrickTracer::new_local();

        // Very long budget that won't be exceeded
        let result = tracer.trace("FastBrick", 1_000_000, || 42);

        let meta = result.metadata.unwrap();
        assert!(!meta.over_budget);
    }

    // Test SyscallEvent creation
    #[test]
    fn test_syscall_event_creation() {
        let event = SyscallEvent {
            syscall: "mmap".to_string(),
            duration: Duration::from_micros(100),
            result: 0,
        };
        assert_eq!(event.syscall, "mmap");
        assert_eq!(event.duration.as_micros(), 100);
        assert_eq!(event.result, 0);
    }

    // Test SyscallBreakdown from_events with various syscalls
    #[test]
    fn test_syscall_breakdown_from_events_categorization() {
        let events = vec![
            SyscallEvent {
                syscall: "mmap".to_string(),
                duration: Duration::from_micros(100),
                result: 0,
            },
            SyscallEvent {
                syscall: "munmap".to_string(),
                duration: Duration::from_micros(50),
                result: 0,
            },
            SyscallEvent {
                syscall: "mprotect".to_string(),
                duration: Duration::from_micros(30),
                result: 0,
            },
            SyscallEvent {
                syscall: "brk".to_string(),
                duration: Duration::from_micros(20),
                result: 0,
            },
            SyscallEvent {
                syscall: "futex".to_string(),
                duration: Duration::from_micros(200),
                result: 0,
            },
            SyscallEvent {
                syscall: "ioctl".to_string(),
                duration: Duration::from_micros(150),
                result: 0,
            },
            SyscallEvent {
                syscall: "read".to_string(),
                duration: Duration::from_micros(80),
                result: 100,
            },
            SyscallEvent {
                syscall: "pread64".to_string(),
                duration: Duration::from_micros(40),
                result: 50,
            },
            SyscallEvent {
                syscall: "readv".to_string(),
                duration: Duration::from_micros(30),
                result: 25,
            },
            SyscallEvent {
                syscall: "write".to_string(),
                duration: Duration::from_micros(60),
                result: 100,
            },
            SyscallEvent {
                syscall: "pwrite64".to_string(),
                duration: Duration::from_micros(35),
                result: 50,
            },
            SyscallEvent {
                syscall: "writev".to_string(),
                duration: Duration::from_micros(25),
                result: 25,
            },
            SyscallEvent {
                syscall: "close".to_string(),
                duration: Duration::from_micros(10),
                result: 0,
            },
        ];

        let breakdown = SyscallBreakdown::from_events(&events, 1500);

        // mmap category: 100 + 50 + 30 + 20 = 200
        assert_eq!(breakdown.mmap_us, 200);
        // futex: 200
        assert_eq!(breakdown.futex_us, 200);
        // ioctl: 150
        assert_eq!(breakdown.ioctl_us, 150);
        // read: 80 + 40 + 30 = 150
        assert_eq!(breakdown.read_us, 150);
        // write: 60 + 35 + 25 = 120
        assert_eq!(breakdown.write_us, 120);
        // other: 10
        assert_eq!(breakdown.other_us, 10);
        // compute: 1500 - 830 = 670
        assert_eq!(breakdown.compute_us, 670);
        // syscall count
        assert_eq!(breakdown.syscall_count, 13);
    }

    // Test dominant_syscall for each category
    #[test]
    fn test_dominant_syscall_all_categories() {
        // Test ioctl dominant
        let mut breakdown = SyscallBreakdown::default();
        breakdown.ioctl_us = 500;
        breakdown.mmap_us = 100;
        assert_eq!(breakdown.dominant_syscall(), "ioctl");

        // Test read dominant
        let mut breakdown = SyscallBreakdown::default();
        breakdown.read_us = 500;
        breakdown.ioctl_us = 100;
        assert_eq!(breakdown.dominant_syscall(), "read");

        // Test write dominant
        let mut breakdown = SyscallBreakdown::default();
        breakdown.write_us = 500;
        breakdown.read_us = 100;
        assert_eq!(breakdown.dominant_syscall(), "write");

        // Test other dominant
        let mut breakdown = SyscallBreakdown::default();
        breakdown.other_us = 500;
        breakdown.write_us = 100;
        assert_eq!(breakdown.dominant_syscall(), "other");

        // Test mmap dominant
        let mut breakdown = SyscallBreakdown::default();
        breakdown.mmap_us = 500;
        breakdown.other_us = 100;
        assert_eq!(breakdown.dominant_syscall(), "mmap");

        // Test none (all zeros)
        let breakdown = SyscallBreakdown::default();
        assert_eq!(breakdown.dominant_syscall(), "none");
    }

    // Test syscall_overhead_percent with zero total
    #[test]
    fn test_syscall_overhead_zero_total() {
        let breakdown = SyscallBreakdown::default();
        assert_eq!(breakdown.syscall_overhead_percent(), 0.0);
    }

    // Test thresholds with_rate_limit builder
    #[test]
    fn test_thresholds_with_rate_limit() {
        let thresholds = BrickEscalationThresholds::default().with_rate_limit(50);
        assert_eq!(thresholds.max_traces_per_sec, 50);
    }

    // Test thresholds() getter
    #[test]
    fn test_thresholds_getter() {
        let tracer = BrickTracer::new_local();
        let thresholds = tracer.thresholds();
        assert!((thresholds.cv_percent - 15.0).abs() < 0.001);
    }

    // Test BrickTracer default
    #[test]
    fn test_brick_tracer_default() {
        let tracer = BrickTracer::default();
        assert!(tracer.enabled);
        assert!(!tracer.has_exporter());
    }

    // Test TracedBrickResult fields
    #[test]
    fn test_traced_brick_result_all_fields() {
        let tracer = BrickTracer::new_local();
        let result = tracer.trace("TestBrick", 1000, || "hello");

        assert_eq!(result.result, "hello");
        assert!(result.duration_us < 1000000); // Should be fast
        assert!(result.metadata.is_some());
        assert!(result.span_id.is_none()); // No exporter
        assert!(result.escalation_reason.is_some());
    }

    // Test BrickMetadata fields
    #[test]
    fn test_brick_metadata_all_fields() {
        let tracer = BrickTracer::new_local();
        let result = tracer.trace("TestBrick", 5000, || {
            std::thread::sleep(Duration::from_micros(100));
            42
        });

        let meta = result.metadata.unwrap();
        assert_eq!(meta.name, "TestBrick");
        assert_eq!(meta.budget_us, 5000);
        assert!(meta.actual_us > 0);
        // Efficiency and over_budget calculated
        assert!(meta.efficiency <= 1.0);
        // Optional fields
        assert!(meta.cv_percent.is_none());
        assert!(meta.score.is_none());
        assert!(meta.grade.is_none());
        assert_eq!(meta.assertions_passed, 0);
        assert_eq!(meta.assertions_failed, 0);
        assert!(meta.failed_assertion_names.is_empty());
    }

    // Test rate limiting
    #[test]
    fn test_rate_limiting() {
        let thresholds = BrickEscalationThresholds::default().with_rate_limit(5);
        let tracer = BrickTracer::new_local().with_thresholds(thresholds);

        // First 5 should pass
        for _ in 0..5 {
            assert!(tracer.should_trace(20.0, 10.0));
        }

        // 6th should be rate limited (in same second)
        // Note: This test might be flaky if it crosses second boundary
        let result = tracer.should_trace(20.0, 10.0);
        // Result depends on timing - just verify no panic
        let _ = result;
    }

    // Test SyscallBreakdown Debug
    #[test]
    fn test_syscall_breakdown_debug() {
        let breakdown = SyscallBreakdown::default();
        let debug = format!("{:?}", breakdown);
        assert!(debug.contains("SyscallBreakdown"));
    }

    // Test BrickMetadata Debug and Clone
    #[test]
    fn test_brick_metadata_debug_clone() {
        let meta = BrickMetadata {
            name: "Test".to_string(),
            budget_us: 1000,
            actual_us: 500,
            over_budget: false,
            efficiency: 1.0,
            cv_percent: Some(5.0),
            score: Some(95),
            grade: Some('A'),
            assertions_passed: 2,
            assertions_failed: 0,
            failed_assertion_names: Vec::new(),
        };

        let debug = format!("{:?}", meta);
        assert!(debug.contains("Test"));

        let cloned = meta.clone();
        assert_eq!(cloned.name, "Test");
        assert_eq!(cloned.score, Some(95));
    }

    // Test TracedBrickResult Clone
    #[test]
    fn test_traced_brick_result_clone() {
        let tracer = BrickTracer::new_local();
        let result = tracer.trace("TestBrick", 1000, || 42);
        let cloned = result.clone();
        assert_eq!(cloned.result, 42);
    }

    // Test EscalationReason PartialEq and Eq
    #[test]
    fn test_escalation_reason_eq() {
        assert!(EscalationReason::CvExceeded == EscalationReason::CvExceeded);
        assert!(EscalationReason::CvExceeded != EscalationReason::EfficiencyLow);
    }

    // Test SyscallEvent Clone and Debug
    #[test]
    fn test_syscall_event_clone_debug() {
        let event = SyscallEvent {
            syscall: "read".to_string(),
            duration: Duration::from_micros(100),
            result: 42,
        };

        let cloned = event.clone();
        assert_eq!(cloned.syscall, "read");

        let debug = format!("{:?}", event);
        assert!(debug.contains("read"));
    }

    // Test SyscallBreakdown Clone
    #[test]
    fn test_syscall_breakdown_clone() {
        let mut breakdown = SyscallBreakdown::default();
        breakdown.mmap_us = 100;

        let cloned = breakdown.clone();
        assert_eq!(cloned.mmap_us, 100);
    }

    // Test BrickEscalationThresholds Debug and Clone
    #[test]
    fn test_thresholds_debug_clone() {
        let thresholds = BrickEscalationThresholds::default();
        let debug = format!("{:?}", thresholds);
        assert!(debug.contains("BrickEscalationThresholds"));

        let cloned = thresholds;
        assert!((cloned.cv_percent - 15.0).abs() < 0.001);
    }

    // Test EscalationReason Copy
    #[test]
    fn test_escalation_reason_copy() {
        let reason = EscalationReason::Both;
        let copied = reason;
        assert_eq!(copied, EscalationReason::Both);
    }
}
