//! Core types for the stress testing framework.
//!
//! Contains frame profiles, reports, anomaly types, performance thresholds,
//! and the `verify_performance` function.

/// Frame profile data collected during stress testing
#[derive(Debug, Clone, Default)]
pub struct FrameProfile {
    /// Cycle number
    pub cycle: u32,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Memory usage estimate (bytes)
    pub memory_bytes: usize,
    /// Number of tests passed
    pub tests_passed: u32,
    /// Number of tests failed
    pub tests_failed: u32,
    /// Input seed used for this frame
    pub input_seed: u64,
    /// Input size used for this frame
    pub input_size: usize,
}

/// Cumulative stress test report
#[derive(Debug, Clone, Default)]
pub struct StressReport {
    /// All frame profiles
    pub frames: Vec<FrameProfile>,
    /// Total cycles completed
    pub cycles_completed: u32,
    /// Total tests passed across all cycles
    pub total_passed: u32,
    /// Total tests failed across all cycles
    pub total_failed: u32,
    /// Detected anomalies
    pub anomalies: Vec<Anomaly>,
}

impl StressReport {
    /// Calculate mean frame time in milliseconds
    #[must_use]
    pub fn mean_frame_time_ms(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.frames.iter().map(|f| f.duration_ms).sum();
        sum as f64 / self.frames.len() as f64
    }

    /// Calculate timing variance (coefficient of variation)
    #[must_use]
    pub fn timing_variance(&self) -> f64 {
        if self.frames.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_frame_time_ms();
        if mean == 0.0 {
            return 0.0;
        }
        let variance: f64 = self
            .frames
            .iter()
            .map(|f| {
                let diff = f.duration_ms as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.frames.len() as f64;
        variance.sqrt() / mean
    }

    /// Calculate max frame time
    #[must_use]
    pub fn max_frame_time_ms(&self) -> u64 {
        self.frames.iter().map(|f| f.duration_ms).max().unwrap_or(0)
    }

    /// Calculate pass rate (0.0 to 1.0)
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        let total = self.total_passed + self.total_failed;
        if total == 0 {
            return 1.0;
        }
        self.total_passed as f64 / total as f64
    }

    /// Add a frame to the report
    pub fn add_frame(&mut self, profile: FrameProfile) {
        self.total_passed += profile.tests_passed;
        self.total_failed += profile.tests_failed;
        self.cycles_completed += 1;
        self.frames.push(profile);
    }
}

/// Detected anomaly during stress testing
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Cycle where anomaly was detected
    pub cycle: u32,
    /// Type of anomaly
    pub kind: AnomalyKind,
    /// Description
    pub description: String,
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyKind {
    /// Frame took longer than threshold
    SlowFrame,
    /// Memory usage exceeded threshold
    HighMemory,
    /// Test failure detected
    TestFailure,
    /// Timing variance spike
    TimingSpike,
    /// Non-deterministic behavior
    NonDeterministic,
}

/// Performance thresholds for anomaly detection
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Max time per frame (ms)
    pub max_frame_time_ms: u64,
    /// Max memory per frame (bytes)
    pub max_memory_bytes: usize,
    /// Max variance in frame times (coefficient of variation)
    pub max_timing_variance: f64,
    /// Max allowed failure rate (0.0 to 1.0)
    pub max_failure_rate: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_frame_time_ms: 100,             // 10 FPS minimum
            max_memory_bytes: 64 * 1024 * 1024, // 64MB max
            max_timing_variance: 0.2,           // 20% max variance
            max_failure_rate: 0.01,             // 1% max failures
        }
    }
}

/// Performance verification result
#[derive(Debug, Clone)]
pub struct PerformanceResult {
    /// Whether all thresholds passed
    pub passed: bool,
    /// Max frame time observed
    pub max_frame_ms: u64,
    /// Mean frame time observed
    pub mean_frame_ms: f64,
    /// Timing variance observed
    pub variance: f64,
    /// Pass rate observed
    pub pass_rate: f64,
    /// List of threshold violations
    pub violations: Vec<String>,
}

/// Verify performance against thresholds
#[must_use]
pub fn verify_performance(
    report: &StressReport,
    thresholds: &PerformanceThresholds,
) -> PerformanceResult {
    let max_frame = report.max_frame_time_ms();
    let mean_frame = report.mean_frame_time_ms();
    let variance = report.timing_variance();
    let pass_rate = report.pass_rate();

    let mut violations = Vec::new();

    if max_frame > thresholds.max_frame_time_ms {
        violations.push(format!(
            "Max frame time {}ms exceeds threshold {}ms",
            max_frame, thresholds.max_frame_time_ms
        ));
    }

    if variance > thresholds.max_timing_variance {
        violations.push(format!(
            "Timing variance {:.3} exceeds threshold {:.3}",
            variance, thresholds.max_timing_variance
        ));
    }

    if pass_rate < (1.0 - thresholds.max_failure_rate) {
        violations.push(format!(
            "Pass rate {:.1}% below threshold {:.1}%",
            pass_rate * 100.0,
            (1.0 - thresholds.max_failure_rate) * 100.0
        ));
    }

    PerformanceResult {
        passed: violations.is_empty(),
        max_frame_ms: max_frame,
        mean_frame_ms: mean_frame,
        variance,
        pass_rate,
        violations,
    }
}
