//! Training Trace Module (ITP-SPEC-001)
//!
//! Provides observability into the training pipeline for empirical analysis.
//! Used to falsify the "Kernel Launch Overhead" hypothesis.

use std::collections::HashMap;
use std::fmt;
use std::sync::{LazyLock, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// The lifecycle steps of a training operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TraceStep {
    /// Forward pass through model
    Forward,
    /// Backward pass (gradient computation)
    Backward,
    /// Matrix multiplication kernel
    Matmul,
    /// Attention computation
    Attention,
    /// CPU transpose operation
    Transpose,
    /// Memory allocation
    Alloc,
    /// Data transfer overhead
    Transfer,
    /// VRAM ledger reservation (GPU-SHARE-001)
    LedgerReserve,
    /// VRAM ledger dead-PID / expired-lease cleanup
    LedgerCleanup,
    /// VRAM query (cuMemGetInfo / NVML)
    VramQuery,
    /// Wait-for-VRAM poll iteration (GPU-SHARE-003)
    WaitPoll,
    /// VRAM ledger release
    LedgerRelease,
}

impl fmt::Display for TraceStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// A single timing measurement.
#[derive(Debug, Clone)]
pub struct TraceMeasurement {
    pub step: TraceStep,
    pub duration: Duration,
    pub metadata: String,
}

/// Thread-safe tracer for collecting timing measurements.
/// ALB-099: Uses aggregated counters instead of unbounded Vec to prevent
/// memory leak during long training runs (was ~2.8 GB at 28K steps).
pub struct Tracer {
    /// Legacy per-measurement storage (kept for backward compat, bounded)
    measurements: Mutex<Vec<TraceMeasurement>>,
    /// ALB-099: Aggregated totals — O(1) memory regardless of training length
    aggregated: Mutex<HashMap<TraceStep, (usize, Duration)>>,
    active_spans: Mutex<HashMap<TraceStep, Instant>>,
    enabled: Mutex<bool>,
}

impl Tracer {
    /// Create a new tracer.
    pub fn new() -> Self {
        Self {
            measurements: Mutex::new(Vec::new()),
            aggregated: Mutex::new(HashMap::new()),
            active_spans: Mutex::new(HashMap::new()),
            enabled: Mutex::new(false), // Disabled by default for performance
        }
    }

    /// Enable tracing.
    pub fn enable(&self) {
        *self.enabled.lock().unwrap_or_else(PoisonError::into_inner) = true;
    }

    /// Disable tracing.
    pub fn disable(&self) {
        *self.enabled.lock().unwrap_or_else(PoisonError::into_inner) = false;
    }

    /// Check if tracing is enabled.
    pub fn is_enabled(&self) -> bool {
        *self.enabled.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Start a timing span.
    pub fn start(&self, step: TraceStep) {
        if !self.is_enabled() {
            return;
        }
        let mut spans = self.active_spans.lock().unwrap_or_else(PoisonError::into_inner);
        spans.insert(step, Instant::now());
    }

    /// End a timing span and record measurement.
    /// ALB-099: Aggregates into counters (O(1) memory) instead of appending to Vec.
    pub fn end(&self, step: TraceStep, _metadata: impl Into<String>) {
        if !self.is_enabled() {
            return;
        }
        let mut spans = self.active_spans.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(start) = spans.remove(&step) {
            let duration = start.elapsed();
            let mut agg = self.aggregated.lock().unwrap_or_else(PoisonError::into_inner);
            let entry = agg.entry(step).or_insert((0, Duration::ZERO));
            entry.0 += 1;
            entry.1 += duration;
        }
    }

    /// Run a closure within a measured span.
    #[inline]
    pub fn span<F, R>(&self, step: TraceStep, metadata: impl Into<String>, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.is_enabled() {
            return f();
        }
        self.start(step);
        let result = f();
        self.end(step, metadata);
        result
    }

    /// Clear all measurements.
    pub fn clear(&self) {
        self.measurements.lock().unwrap_or_else(PoisonError::into_inner).clear();
        self.aggregated.lock().unwrap_or_else(PoisonError::into_inner).clear();
        self.active_spans.lock().unwrap_or_else(PoisonError::into_inner).clear();
    }

    /// Generate a report with Dr. Popper analysis.
    /// ALB-099: Reads from aggregated counters (O(1) memory).
    pub fn report(&self) -> String {
        let agg = self.aggregated.lock().unwrap_or_else(PoisonError::into_inner);
        if agg.is_empty() {
            return "No measurements recorded. Enable tracing with TRACER.enable()".to_string();
        }

        let mut totals: HashMap<TraceStep, Duration> = HashMap::new();
        let mut counts: HashMap<TraceStep, usize> = HashMap::new();
        let mut total_time = Duration::ZERO;

        for (&step, &(count, duration)) in agg.iter() {
            totals.insert(step, duration);
            counts.insert(step, count);
            total_time += duration;
        }

        let mut output =
            String::from("\n╔══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║       ENTRENAR TRACE REPORT (ITP-SPEC-001)                   ║\n");
        output.push_str("╚══════════════════════════════════════════════════════════════╝\n");
        output.push_str(&format!("Total Measured Time: {total_time:.2?}\n"));
        output.push_str("────────────────────────────────────────────────────────────────\n");
        output.push_str(&format!(
            "{:<15} | {:<8} | {:<15} | {:<8}\n",
            "Step", "Count", "Duration", "% Time"
        ));
        output.push_str("────────────────────────────────────────────────────────────────\n");

        // Sort by duration descending
        let mut sorted_steps: Vec<_> = totals.keys().collect();
        sorted_steps.sort_by(|a, b| totals[b].cmp(&totals[a]));

        for step in sorted_steps {
            let duration = totals[step];
            let count = counts[step];
            let percentage = if total_time.as_nanos() > 0 {
                (duration.as_secs_f64() / total_time.as_secs_f64()) * 100.0
            } else {
                0.0
            };
            output.push_str(&format!(
                "{:<15} | {:<8} | {:<15.2?} | {:>7.2}%\n",
                step.to_string(),
                count,
                duration,
                percentage
            ));
        }
        output.push_str("────────────────────────────────────────────────────────────────\n");

        // Dr. Popper Analysis
        let matmul_time = totals.get(&TraceStep::Matmul).copied().unwrap_or_default();
        let transpose_time = totals.get(&TraceStep::Transpose).copied().unwrap_or_default();
        let alloc_time = totals.get(&TraceStep::Alloc).copied().unwrap_or_default();
        let compute_time = matmul_time;
        let overhead_time = transpose_time + alloc_time;

        if compute_time.as_nanos() > 0 {
            let overhead_pct = (overhead_time.as_secs_f64()
                / (compute_time + overhead_time).as_secs_f64())
                * 100.0;

            output.push_str("\n[Dr. Popper Analysis]\n");
            output.push_str(&format!("CUDA Compute:   {compute_time:.2?}\n"));
            output.push_str(&format!("CPU Overhead:   {overhead_time:.2?} ({overhead_pct:.2}%)\n"));

            if overhead_pct > 50.0 {
                output.push_str("\n🔴 FALSIFICATION: Overhead > 50%. Kernel fusion required.\n");
            } else {
                output.push_str("\n🟢 CORROBORATED: Compute dominates. Current approach viable.\n");
            }
        }

        output
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}

/// Global tracer instance.
pub static TRACER: LazyLock<Tracer> = LazyLock::new(Tracer::new);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_step_display() {
        assert_eq!(TraceStep::Forward.to_string(), "Forward");
        assert_eq!(TraceStep::Backward.to_string(), "Backward");
        assert_eq!(TraceStep::Matmul.to_string(), "Matmul");
        assert_eq!(TraceStep::Attention.to_string(), "Attention");
        assert_eq!(TraceStep::Transpose.to_string(), "Transpose");
        assert_eq!(TraceStep::Alloc.to_string(), "Alloc");
        assert_eq!(TraceStep::Transfer.to_string(), "Transfer");
        assert_eq!(TraceStep::LedgerReserve.to_string(), "LedgerReserve");
        assert_eq!(TraceStep::LedgerCleanup.to_string(), "LedgerCleanup");
        assert_eq!(TraceStep::VramQuery.to_string(), "VramQuery");
        assert_eq!(TraceStep::WaitPoll.to_string(), "WaitPoll");
        assert_eq!(TraceStep::LedgerRelease.to_string(), "LedgerRelease");
    }

    #[test]
    fn test_trace_step_clone() {
        let step = TraceStep::Forward;
        let cloned = step;
        assert_eq!(step, cloned);
    }

    #[test]
    fn test_trace_step_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(TraceStep::Forward);
        set.insert(TraceStep::Forward);
        assert_eq!(set.len(), 1);
        set.insert(TraceStep::Backward);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_tracer_new() {
        let tracer = Tracer::new();
        assert!(!tracer.is_enabled());
    }

    #[test]
    fn test_tracer_default() {
        let tracer = Tracer::default();
        assert!(!tracer.is_enabled());
    }

    #[test]
    fn test_tracer_enable_disable() {
        let tracer = Tracer::new();
        assert!(!tracer.is_enabled());
        tracer.enable();
        assert!(tracer.is_enabled());
        tracer.disable();
        assert!(!tracer.is_enabled());
    }

    #[test]
    fn test_tracer_start_end_disabled() {
        let tracer = Tracer::new();
        // Should not panic when disabled
        tracer.start(TraceStep::Forward);
        tracer.end(TraceStep::Forward, "test");
    }

    #[test]
    fn test_tracer_start_end_enabled() {
        let tracer = Tracer::new();
        tracer.enable();
        tracer.start(TraceStep::Matmul);
        tracer.end(TraceStep::Matmul, "2x2");
        // Verify measurement was recorded
        let report = tracer.report();
        assert!(report.contains("Matmul"));
    }

    #[test]
    fn test_tracer_span_disabled() {
        let tracer = Tracer::new();
        let result = tracer.span(TraceStep::Forward, "test", || 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_tracer_span_enabled() {
        let tracer = Tracer::new();
        tracer.enable();
        let result = tracer.span(TraceStep::Attention, "4 heads", || "done");
        assert_eq!(result, "done");
        let report = tracer.report();
        assert!(report.contains("Attention"));
    }

    #[test]
    fn test_tracer_clear() {
        let tracer = Tracer::new();
        tracer.enable();
        tracer.start(TraceStep::Forward);
        tracer.end(TraceStep::Forward, "test");
        tracer.clear();
        let report = tracer.report();
        assert!(report.contains("No measurements recorded"));
    }

    #[test]
    fn test_tracer_report_empty() {
        let tracer = Tracer::new();
        let report = tracer.report();
        assert!(report.contains("No measurements recorded"));
    }

    #[test]
    fn test_tracer_report_with_measurements() {
        let tracer = Tracer::new();
        tracer.enable();

        tracer.start(TraceStep::Matmul);
        tracer.end(TraceStep::Matmul, "512x512");

        tracer.start(TraceStep::Transpose);
        tracer.end(TraceStep::Transpose, "256x256");

        let report = tracer.report();
        assert!(report.contains("ENTRENAR TRACE REPORT"));
        assert!(report.contains("Matmul"));
        assert!(report.contains("Transpose"));
        assert!(report.contains("% Time"));
    }

    #[test]
    fn test_tracer_report_dr_popper_analysis() {
        let tracer = Tracer::new();

        // Inject into aggregated counters (report reads from aggregated, not measurements)
        {
            let mut agg = tracer.aggregated.lock().expect("lock acquisition should succeed");
            agg.insert(TraceStep::Matmul, (1, Duration::from_millis(50)));
            agg.insert(TraceStep::Transpose, (1, Duration::from_millis(10)));
        }

        let report = tracer.report();
        assert!(report.contains("Dr. Popper Analysis"));
        assert!(report.contains("CUDA Compute:"));
        assert!(report.contains("CPU Overhead:"));
    }

    #[test]
    fn test_tracer_end_without_start() {
        let tracer = Tracer::new();
        tracer.enable();
        // Should not panic - just ignored
        tracer.end(TraceStep::Forward, "no start");
        let report = tracer.report();
        assert!(report.contains("No measurements recorded"));
    }

    #[test]
    fn test_trace_measurement_clone() {
        let measurement = TraceMeasurement {
            step: TraceStep::Forward,
            duration: Duration::from_millis(100),
            metadata: "test".to_string(),
        };
        let cloned = measurement.clone();
        assert_eq!(measurement.step, cloned.step);
        assert_eq!(measurement.duration, cloned.duration);
        assert_eq!(measurement.metadata, cloned.metadata);
    }

    #[test]
    fn test_trace_measurement_debug() {
        let measurement = TraceMeasurement {
            step: TraceStep::Backward,
            duration: Duration::from_micros(50),
            metadata: "grad".to_string(),
        };
        let debug_str = format!("{measurement:?}");
        assert!(debug_str.contains("TraceMeasurement"));
        assert!(debug_str.contains("Backward"));
    }
}
