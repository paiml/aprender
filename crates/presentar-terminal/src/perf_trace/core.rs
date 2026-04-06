use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

// =============================================================================
// GLOBAL ATOMIC STATE (trueno-viz pattern)
// =============================================================================

/// Global flag to enable/disable tracing (zero cost when disabled - 1 atomic load)
static TRACE_ENABLED: AtomicBool = AtomicBool::new(false);
/// Start time in microseconds for relative timestamps
static START_TIME_US: AtomicU64 = AtomicU64::new(0);

/// Enable global tracing
pub fn enable_tracing() {
    START_TIME_US.store(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64,
        Ordering::Relaxed,
    );
    TRACE_ENABLED.store(true, Ordering::Release);
}

/// Disable global tracing
pub fn disable_tracing() {
    TRACE_ENABLED.store(false, Ordering::Release);
}

/// Check if tracing is enabled
#[inline]
pub fn is_tracing_enabled() -> bool {
    TRACE_ENABLED.load(Ordering::Acquire)
}

// =============================================================================
// RAII TIMING GUARD (trueno-viz pattern)
// =============================================================================

/// Zero-cost RAII timing guard (trueno-viz pattern)
///
/// Automatically logs entry/exit on Drop. Zero overhead when tracing disabled.
///
/// # Example
/// ```rust,ignore
/// fn render_panel() {
///     let _guard = TimingGuard::new("render_panel", 1000); // 1ms budget
///     // ... rendering code ...
/// } // Automatically logs duration on drop
/// ```
pub struct TimingGuard {
    name: &'static str,
    start: Option<Instant>,
    budget_us: u64,
}

impl TimingGuard {
    /// Create a new timing guard with budget in microseconds
    ///
    /// If tracing is disabled, returns a no-op guard (zero cost).
    #[inline]
    pub fn new(name: &'static str, budget_us: u64) -> Self {
        if is_tracing_enabled() {
            Self {
                name,
                start: Some(Instant::now()),
                budget_us,
            }
        } else {
            Self {
                name,
                start: None,
                budget_us,
            }
        }
    }

    /// Create guard with default 1ms budget
    #[inline]
    pub fn with_default_budget(name: &'static str) -> Self {
        Self::new(name, 1000)
    }

    /// Create guard for render operations (16ms budget for 60fps)
    #[inline]
    pub fn render(name: &'static str) -> Self {
        Self::new(name, 16_000)
    }

    /// Create guard for collection operations (100ms budget)
    #[inline]
    pub fn collect(name: &'static str) -> Self {
        Self::new(name, 100_000)
    }
}

impl Drop for TimingGuard {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed = start.elapsed();
            let elapsed_us = elapsed.as_micros() as u64;
            let exceeded = elapsed_us > self.budget_us;

            // Log format: [+0042ms] [TRACE] [name] <- done (12.34ms)
            let relative_ms = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64)
                .saturating_sub(START_TIME_US.load(Ordering::Relaxed))
                / 1000;

            let status = if exceeded { "⚠️" } else { "✓" };
            eprintln!(
                "[+{:04}ms] [TRACE] {status} {} <- {:.2}ms (budget: {}μs)",
                relative_ms,
                self.name,
                elapsed_us as f64 / 1000.0,
                self.budget_us
            );
        }
    }
}

// =============================================================================
// O(1) SIMD-STYLE STATS (trueno-viz pattern)
// =============================================================================

/// Cache-aligned running statistics for O(1) access (trueno-viz SimdStats pattern)
///
/// All statistics are pre-computed during data collection, never during render.
/// This guarantees <1ms frame time per SPEC-024.
#[repr(C, align(64))] // Cache-aligned for multi-threaded access
#[derive(Debug, Clone, Default)]
pub struct SimdStats {
    /// Running count of samples
    pub count: u64,
    /// Running sum for mean calculation
    pub sum: f64,
    /// Running sum of squares for variance calculation
    pub sum_sq: f64,
    /// Running minimum value
    pub min: f64,
    /// Running maximum value
    pub max: f64,
}

impl SimdStats {
    /// Create new stats initialized to zero
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    /// Update stats with a new value (O(1))
    #[inline]
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_sq += value * value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Get mean (O(1))
    #[inline]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Get variance (O(1))
    #[inline]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        (self.sum_sq - (self.sum * self.sum) / n) / (n - 1.0)
    }

    /// Get standard deviation (O(1))
    #[inline]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get coefficient of variation (O(1))
    #[inline]
    pub fn cv_percent(&self) -> f64 {
        let mean = self.mean();
        if mean.abs() < 1e-9 {
            0.0
        } else {
            (self.std_dev() / mean) * 100.0
        }
    }

    /// Reset stats
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// =============================================================================
// BRICK PROFILER (renacer BrickTracer integration)
// =============================================================================

/// Brick types for profiling (aligns with renacer brick taxonomy)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BrickType {
    /// Data collection brick (I/O bound)
    Collect,
    /// Render brick (CPU bound, must be <16ms for 60fps)
    Render,
    /// Compute brick (SIMD optimized)
    Compute,
    /// Network brick (I/O bound with latency)
    Network,
    /// Storage brick (disk I/O)
    Storage,
}

impl BrickType {
    /// Get default budget for this brick type in microseconds
    #[must_use]
    pub fn default_budget_us(&self) -> u64 {
        match self {
            Self::Collect => 100_000, // 100ms for collection
            Self::Render => 16_000,   // 16ms for 60fps
            Self::Compute => 1_000,   // 1ms for compute
            Self::Network => 500_000, // 500ms for network
            Self::Storage => 50_000,  // 50ms for storage
        }
    }

    /// Get severity threshold (CV%) for escalation
    #[must_use]
    pub fn cv_threshold(&self) -> f64 {
        match self {
            Self::Render => 10.0,  // Strict for render
            Self::Compute => 15.0, // Standard
            Self::Collect => 25.0, // More lenient
            Self::Network => 50.0, // High variance expected
            Self::Storage => 30.0, // Moderate variance
        }
    }
}

/// Brick profiler for tracking computational units.
///
/// Provides higher-level profiling with brick-specific budgets and thresholds.
/// Compatible with renacer's BrickTracer for escalation.
#[derive(Debug)]
pub struct BrickProfiler {
    /// Stats per brick name
    stats: std::collections::HashMap<String, (BrickType, SimdStats)>,
    /// Whether profiling is enabled
    enabled: bool,
}

impl Default for BrickProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickProfiler {
    /// Create a new brick profiler
    #[must_use]
    pub fn new() -> Self {
        Self {
            stats: std::collections::HashMap::new(),
            enabled: is_tracing_enabled(),
        }
    }

    /// Profile a brick execution
    pub fn profile<F, R>(&mut self, name: &str, brick_type: BrickType, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.enabled {
            return f();
        }

        let start = std::time::Instant::now();
        let result = f();
        let elapsed_us = start.elapsed().as_micros() as f64;

        // Update stats
        let (_, stats) = self
            .stats
            .entry(name.to_string())
            .or_insert_with(|| (brick_type, SimdStats::new()));
        stats.update(elapsed_us);

        result
    }

    /// Check if a brick should escalate to deep profiling
    #[must_use]
    pub fn should_escalate(&self, name: &str) -> bool {
        if let Some((brick_type, stats)) = self.stats.get(name) {
            let cv = stats.cv_percent();
            cv > brick_type.cv_threshold()
        } else {
            false
        }
    }

    /// Get stats for a brick
    #[must_use]
    pub fn get_stats(&self, name: &str) -> Option<&SimdStats> {
        self.stats.get(name).map(|(_, s)| s)
    }

    /// Generate a summary report
    #[must_use]
    pub fn summary(&self) -> String {
        let mut lines = vec!["=== Brick Profiler Summary ===".to_string()];

        let mut sorted: Vec<_> = self.stats.iter().collect();
        sorted.sort_by(|a, b| {
            let a_total = a.1 .1.sum;
            let b_total = b.1 .1.sum;
            b_total
                .partial_cmp(&a_total)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (name, (brick_type, stats)) in sorted {
            let budget = brick_type.default_budget_us();
            let avg = stats.mean();
            let cv = stats.cv_percent();
            let status = if avg > budget as f64 { "⚠️" } else { "✓" };
            let escalate = if self.should_escalate(name) {
                " [ESCALATE]"
            } else {
                ""
            };

            lines.push(format!(
                "{status} {name} ({brick_type:?}): avg={avg:.0}μs cv={cv:.1}% n={}{escalate}",
                stats.count
            ));
        }

        lines.join("\n")
    }
}

/// Performance trace event (compatible with renacer `TraceEvent`)
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Event name (e.g., "`collect_metrics`", "`render_cpu_panel`")
    pub name: String,
    /// Duration of the event
    pub duration: Duration,
    /// Timestamp when event started
    pub timestamp_us: u64,
    /// Whether this exceeded the budget
    pub budget_exceeded: bool,
    /// Budget in microseconds (if set)
    pub budget_us: Option<u64>,
}

/// Aggregated statistics for a traced operation
#[derive(Debug, Clone, Default)]
pub struct TraceStats {
    /// Total number of invocations
    pub count: u64,
    /// Total duration across all invocations
    pub total_duration: Duration,
    /// Minimum duration
    pub min_duration: Duration,
    /// Maximum duration
    pub max_duration: Duration,
    /// Number of times budget was exceeded
    pub budget_violations: u64,
    /// Budget in microseconds
    pub budget_us: u64,
}

impl TraceStats {
    /// Create new stats with initial values
    fn new(duration: Duration, budget_us: u64, exceeded: bool) -> Self {
        Self {
            count: 1,
            total_duration: duration,
            min_duration: duration,
            max_duration: duration,
            budget_violations: u64::from(exceeded),
            budget_us,
        }
    }

    /// Update stats with a new measurement
    fn update(&mut self, duration: Duration, exceeded: bool) {
        self.count += 1;
        self.total_duration += duration;
        self.min_duration = self.min_duration.min(duration);
        self.max_duration = self.max_duration.max(duration);
        if exceeded {
            self.budget_violations += 1;
        }
    }

    /// Average duration
    pub fn avg_duration(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.total_duration / self.count as u32
        }
    }

    /// Coefficient of variation (CV) percentage
    /// CV > 15% indicates unstable performance per Curtsinger & Berger (2013)
    pub fn cv_percent(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let avg = self.avg_duration().as_secs_f64();
        if avg < 1e-9 {
            return 0.0;
        }
        let range = self
            .max_duration
            .checked_sub(self.min_duration)
            .unwrap_or_default()
            .as_secs_f64();
        (range / avg) * 50.0 // Simplified CV approximation
    }

    /// Budget efficiency percentage
    /// Efficiency < 25% indicates budget is too tight per Williams et al. (2009)
    pub fn efficiency_percent(&self) -> f64 {
        if self.budget_us == 0 {
            return 100.0;
        }
        let avg_us = self.avg_duration().as_micros() as f64;
        ((self.budget_us as f64) / avg_us * 100.0).min(100.0)
    }
}

/// Escalation thresholds for deep tracing (from renacer `BrickEscalationThresholds`)
#[derive(Debug, Clone, Copy)]
pub struct EscalationThresholds {
    /// CV threshold above which to escalate (default: 15.0%)
    pub cv_percent: f64,
    /// Efficiency threshold below which to escalate (default: 25.0%)
    pub efficiency_percent: f64,
    /// Maximum traces per second (rate limiting)
    pub max_traces_per_sec: u32,
}

impl Default for EscalationThresholds {
    fn default() -> Self {
        Self {
            cv_percent: 15.0,
            efficiency_percent: 25.0,
            max_traces_per_sec: 100,
        }
    }
}

/// Lightweight performance tracer for ptop
///
/// Provides timing measurements and statistics for `ComputeBlocks`.
/// Can be used standalone or integrated with renacer for deep analysis.
#[derive(Debug)]
pub struct PerfTracer {
    /// Aggregated stats per operation name
    stats: HashMap<String, TraceStats>,
    /// Recent events (ring buffer, last N)
    recent_events: Vec<TraceEvent>,
    /// Maximum recent events to keep
    max_recent: usize,
    /// Start time for relative timestamps
    start_time: Instant,
    /// Escalation thresholds
    thresholds: EscalationThresholds,
    /// Number of traces since last second
    traces_this_second: u32,
    /// Second when trace count was last reset
    last_second: u64,
}

impl Default for PerfTracer {
    fn default() -> Self {
        Self::new()
    }
}

impl PerfTracer {
    /// Create a new performance tracer
    #[must_use]
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            recent_events: Vec::with_capacity(100),
            max_recent: 100,
            start_time: Instant::now(),
            thresholds: EscalationThresholds::default(),
            traces_this_second: 0,
            last_second: 0,
        }
    }

    /// Create a tracer with custom thresholds
    #[must_use]
    pub fn with_thresholds(thresholds: EscalationThresholds) -> Self {
        Self {
            thresholds,
            ..Self::new()
        }
    }

    /// Trace a function execution with a name and budget
    pub fn trace_with_budget<F, R>(&mut self, name: &str, budget_us: u64, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();

        self.record_trace(name, duration, budget_us);
        result
    }

    /// Trace a function execution with default 1ms budget
    pub fn trace<F, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.trace_with_budget(name, 1000, f) // 1ms default budget
    }

    /// Record a trace event
    fn record_trace(&mut self, name: &str, duration: Duration, budget_us: u64) {
        let timestamp_us = self.start_time.elapsed().as_micros() as u64;
        let budget_exceeded = duration.as_micros() as u64 > budget_us;

        // Rate limiting
        let current_second = timestamp_us / 1_000_000;
        if current_second != self.last_second {
            self.traces_this_second = 0;
            self.last_second = current_second;
        }
        self.traces_this_second += 1;

        // Create event
        let event = TraceEvent {
            name: name.to_string(),
            duration,
            timestamp_us,
            budget_exceeded,
            budget_us: Some(budget_us),
        };

        // Update stats
        if let Some(stats) = self.stats.get_mut(name) {
            stats.update(duration, budget_exceeded);
        } else {
            self.stats.insert(
                name.to_string(),
                TraceStats::new(duration, budget_us, budget_exceeded),
            );
        }

        // Store recent event
        if self.recent_events.len() >= self.max_recent {
            self.recent_events.remove(0);
        }
        self.recent_events.push(event);
    }

    /// Check if an operation should escalate to deep tracing
    #[must_use]
    pub fn should_escalate(&self, name: &str) -> bool {
        if let Some(stats) = self.stats.get(name) {
            let cv = stats.cv_percent();
            let efficiency = stats.efficiency_percent();

            cv > self.thresholds.cv_percent || efficiency < self.thresholds.efficiency_percent
        } else {
            false
        }
    }

    /// Get stats for a specific operation
    #[must_use]
    pub fn get_stats(&self, name: &str) -> Option<&TraceStats> {
        self.stats.get(name)
    }

    /// Get all stats
    #[must_use]
    pub fn all_stats(&self) -> &HashMap<String, TraceStats> {
        &self.stats
    }

    /// Generate a summary report
    #[must_use]
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "=== Performance Trace Summary ===".to_string(),
            String::new(),
        ];

        let mut sorted: Vec<_> = self.stats.iter().collect();
        sorted.sort_by(|a, b| b.1.total_duration.cmp(&a.1.total_duration));

        for (name, stats) in sorted {
            let avg_us = stats.avg_duration().as_micros();
            let max_us = stats.max_duration.as_micros();
            let cv = stats.cv_percent();
            let eff = stats.efficiency_percent();
            let status = if stats.budget_violations > 0 {
                "⚠️"
            } else {
                "✓"
            };

            lines.push(format!(
                "{status} {name}: avg={avg_us}μs max={max_us}μs count={} cv={cv:.1}% eff={eff:.0}%",
                stats.count
            ));

            if self.should_escalate(name) {
                lines.push(format!(
                    "  └── ESCALATE: CV={cv:.1}% > {}% OR eff={eff:.0}% < {}%",
                    self.thresholds.cv_percent, self.thresholds.efficiency_percent
                ));
            }
        }

        lines.join("\n")
    }

    /// Export stats in a format compatible with renacer
    #[must_use]
    pub fn export_renacer_format(&self) -> String {
        let mut lines = vec!["# renacer-compatible trace export".to_string()];
        lines.push(format!("# timestamp: {:?}", self.start_time.elapsed()));

        for (name, stats) in &self.stats {
            lines.push(format!(
                "TRACE {} count={} total_us={} avg_us={} max_us={} cv={:.2} eff={:.2} violations={}",
                name,
                stats.count,
                stats.total_duration.as_micros(),
                stats.avg_duration().as_micros(),
                stats.max_duration.as_micros(),
                stats.cv_percent(),
                stats.efficiency_percent(),
                stats.budget_violations
            ));
        }

        lines.join("\n")
    }

    /// Clear all stats
    pub fn clear(&mut self) {
        self.stats.clear();
        self.recent_events.clear();
    }
}
