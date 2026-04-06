//! PMAT-284: Lightweight sub-phase timing for CPU-GPU pipelining analysis
//!
//! `PhaseTimer` captures named sub-phases within a hot loop with minimal
//! overhead when disabled. Designed for iteration-scheduler-level instrumentation
//! where BrickProfiler (GPU-side) and InferenceTracer (model-level) don't apply.
//!
//! # Usage
//!
//! ```ignore
//! let mut timer = PhaseTimer::from_env("PMAT_283_TIMING");
//!
//! loop {
//!     timer.start();
//!     timer.mark("lock");      // write lock acquired
//!     // ... scheduling ...
//!     timer.mark("sched");     // scheduling done
//!     // ... decode step ...
//!     timer.mark("decode");    // GPU decode done
//!     // ... token distribution ...
//!     timer.mark("dist");      // tokens distributed
//!     timer.emit(iteration, batch_size);  // prints if enabled + interval
//! }
//! ```
//!
//! Output: `[PMAT-283] iter=42, m=4: lock=3µs sched=12µs decode=7412µs dist=84µs total=7511µs`

use std::time::Instant;

/// Lightweight sub-phase timer for hot-path instrumentation.
///
/// Zero overhead when disabled (all methods are no-ops).
/// When enabled, captures phase boundaries via `mark()` and
/// emits structured timing via `emit()`.
pub struct PhaseTimer {
    enabled: bool,
    start: Option<Instant>,
    marks: Vec<(&'static str, Instant)>,
    label: &'static str,
    /// Emit every N iterations (0 = first 5 then every 100)
    emit_interval: u64,
}

impl PhaseTimer {
    /// Create a timer controlled by an environment variable.
    ///
    /// Returns a disabled (zero-cost) timer if the env var is unset or != "1".
    pub fn from_env(env_var: &'static str, label: &'static str) -> Self {
        let enabled = std::env::var(env_var).map(|v| v == "1").unwrap_or(false);
        Self {
            enabled,
            start: None,
            marks: Vec::with_capacity(if enabled { 8 } else { 0 }),
            label,
            emit_interval: 0,
        }
    }

    /// Create an always-enabled timer (for testing).
    #[cfg(test)]
    pub fn enabled(label: &'static str) -> Self {
        Self { enabled: true, start: None, marks: Vec::with_capacity(8), label, emit_interval: 0 }
    }

    /// Start a new timing cycle. Call at the top of each iteration.
    #[inline]
    pub fn start(&mut self) {
        if self.enabled {
            self.marks.clear();
            self.start = Some(Instant::now());
        }
    }

    /// Mark the end of a named phase. The duration is from the previous
    /// mark (or start) to now.
    #[inline]
    pub fn mark(&mut self, phase: &'static str) {
        if self.enabled {
            self.marks.push((phase, Instant::now()));
        }
    }

    /// Emit timing data if enabled and the iteration matches the emit interval.
    ///
    /// Default interval: first 5 iterations, then every 100.
    pub fn emit(&self, iteration: u64, batch_size: usize) {
        if !self.enabled {
            return;
        }
        // Default: first 5, then every 100
        let should_emit = if self.emit_interval == 0 {
            iteration <= 5 || iteration % 100 == 0
        } else {
            iteration % self.emit_interval == 0
        };
        if !should_emit {
            return;
        }

        let Some(start) = self.start else { return };
        let mut parts = Vec::with_capacity(self.marks.len());
        let mut prev = start;
        for (name, ts) in &self.marks {
            let dur_us = ts.duration_since(prev).as_micros();
            parts.push(format!("{name}={dur_us}µs"));
            prev = *ts;
        }
        let total_us = prev.duration_since(start).as_micros();
        eprintln!(
            "[{}] iter={}, m={}: {} total={}µs",
            self.label,
            iteration,
            batch_size,
            parts.join(" "),
            total_us,
        );
    }

    /// Check if the timer is enabled.
    #[inline]
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_is_noop() {
        let mut t = PhaseTimer::from_env("NONEXISTENT_ENV_VAR_12345", "test");
        assert!(!t.is_enabled());
        t.start();
        t.mark("a");
        t.mark("b");
        t.emit(1, 4); // should not panic or print
    }

    #[test]
    fn test_enabled_captures_phases() {
        let mut t = PhaseTimer::enabled("test");
        t.start();
        std::thread::sleep(std::time::Duration::from_micros(100));
        t.mark("phase_a");
        std::thread::sleep(std::time::Duration::from_micros(100));
        t.mark("phase_b");
        assert_eq!(t.marks.len(), 2);
        assert_eq!(t.marks[0].0, "phase_a");
        assert_eq!(t.marks[1].0, "phase_b");
    }
}
