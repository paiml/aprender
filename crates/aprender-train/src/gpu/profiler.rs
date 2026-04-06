//! Brick-phase profiler for GPU sharing operations (GPU-SHARE-005).
//!
//! Follows the `StepProfiler` pattern from KAIZEN-047.
//!
//! # Contract C-GPUPROF-001
//!
//! Zero-overhead when disabled: all `begin`/`end` calls are no-ops
//! with zero `Instant::now()` calls.
//!
//! # Phases
//!
//! 1. `lock_acq`  — flock(LOCK_EX) acquisition time
//! 2. `ledger_rd` — JSON read + deserialize + PID prune
//! 3. `vram_qry`  — cuMemGetInfo / NVML call
//! 4. `ledger_wr` — Atomic write (temp + rename)
//! 5. `lock_rel`  — flock release (close fd)
//! 6. `wait_poll`  — Single poll iteration

use std::time::{Duration, Instant};

/// Phase indices.
pub const LOCK_ACQ: usize = 0;
pub const LEDGER_RD: usize = 1;
pub const VRAM_QRY: usize = 2;
pub const LEDGER_WR: usize = 3;
pub const LOCK_REL: usize = 4;
pub const WAIT_POLL: usize = 5;
const NUM_GPU_PHASES: usize = 6;

const GPU_PHASE_NAMES: [&str; NUM_GPU_PHASES] =
    ["lock_acq", "ledger_rd", "vram_qry", "ledger_wr", "lock_rel", "wait_poll"];

/// Brick-phase profiler for GPU sharing operations.
pub struct GpuProfiler {
    enabled: bool,
    phase_start: Option<Instant>,
    totals: [Duration; NUM_GPU_PHASES],
    counts: [usize; NUM_GPU_PHASES],
    op_count: usize,
}

impl GpuProfiler {
    /// Create a new profiler.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            phase_start: None,
            totals: [Duration::ZERO; NUM_GPU_PHASES],
            counts: [0; NUM_GPU_PHASES],
            op_count: 0,
        }
    }

    /// Disabled (no-op) profiler.
    pub fn disabled() -> Self {
        Self::new(false)
    }

    /// Start timing a phase.
    #[inline]
    pub fn begin(&mut self, _phase: usize) {
        if !self.enabled {
            return;
        }
        self.phase_start = Some(Instant::now());
    }

    /// End timing a phase and accumulate.
    #[inline]
    pub fn end(&mut self, phase: usize) {
        if !self.enabled {
            return;
        }
        if let Some(start) = self.phase_start.take() {
            self.totals[phase] += start.elapsed();
            self.counts[phase] += 1;
        }
    }

    /// Run a closure within a timed phase.
    #[inline]
    pub fn span<F, R>(&mut self, phase: usize, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.enabled {
            return f();
        }
        self.begin(phase);
        let result = f();
        self.end(phase);
        result
    }

    /// Mark an operation complete (for averaging).
    pub fn finish_op(&mut self) {
        if self.enabled {
            self.op_count += 1;
        }
    }

    /// Whether the profiler is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Total time across all phases.
    pub fn total_time(&self) -> Duration {
        self.totals.iter().copied().sum()
    }

    /// Generate a report string.
    pub fn report(&self) -> String {
        if self.op_count == 0 {
            return String::from("[GpuProfiler] No operations recorded.");
        }

        let total_us = self.total_time().as_micros() as f64;
        let mut out = format!(
            "\n┌─ GPU Sharing Profiler ({} ops, total {:.1} ms) ─┐\n",
            self.op_count,
            total_us / 1000.0
        );
        out.push_str(&format!(
            "│ {:>10} │ {:>6} │ {:>8} │ {:>6} │\n",
            "phase", "count", "total_ms", "pct"
        ));
        out.push_str(&format!("│ {:-<10} │ {:-<6} │ {:-<8} │ {:-<6} │\n", "", "", "", ""));

        for i in 0..NUM_GPU_PHASES {
            let ms = self.totals[i].as_micros() as f64 / 1000.0;
            let pct = if total_us > 0.0 {
                self.totals[i].as_micros() as f64 / total_us * 100.0
            } else {
                0.0
            };
            out.push_str(&format!(
                "│ {:>10} │ {:>6} │ {:>8.2} │ {:>5.1}% │\n",
                GPU_PHASE_NAMES[i], self.counts[i], ms, pct
            ));
        }
        out.push_str("└────────────┴────────┴──────────┴────────┘\n");
        out
    }

    // Phase constants for external callers.
    pub const LOCK_ACQ: usize = LOCK_ACQ;
    pub const LEDGER_RD: usize = LEDGER_RD;
    pub const VRAM_QRY: usize = VRAM_QRY;
    pub const LEDGER_WR: usize = LEDGER_WR;
    pub const LOCK_REL: usize = LOCK_REL;
    pub const WAIT_POLL: usize = WAIT_POLL;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_profiler_is_noop() {
        let mut p = GpuProfiler::disabled();
        p.begin(LOCK_ACQ);
        p.end(LOCK_ACQ);
        p.finish_op();
        // C-GPUPROF-001: op_count stays 0 when disabled
        assert_eq!(p.op_count, 0);
        assert!(!p.is_enabled());
    }

    #[test]
    fn enabled_profiler_records() {
        let mut p = GpuProfiler::new(true);
        p.begin(LOCK_ACQ);
        std::thread::sleep(Duration::from_millis(1));
        p.end(LOCK_ACQ);
        p.finish_op();

        assert!(p.is_enabled());
        assert_eq!(p.op_count, 1);
        assert_eq!(p.counts[LOCK_ACQ], 1);
        assert!(p.totals[LOCK_ACQ] >= Duration::from_micros(500));
    }

    #[test]
    fn span_returns_value() {
        let mut p = GpuProfiler::new(true);
        let result = p.span(VRAM_QRY, || 42);
        assert_eq!(result, 42);
        assert_eq!(p.counts[VRAM_QRY], 1);
    }

    #[test]
    fn report_empty_when_no_ops() {
        let p = GpuProfiler::new(true);
        let report = p.report();
        assert!(report.contains("No operations recorded"));
    }

    #[test]
    fn report_contains_phase_names() {
        let mut p = GpuProfiler::new(true);
        p.begin(LEDGER_RD);
        p.end(LEDGER_RD);
        p.finish_op();

        let report = p.report();
        assert!(report.contains("ledger_rd"));
        assert!(report.contains("1 ops"));
    }
}
