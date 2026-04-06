#![allow(missing_docs)]
//! BLIS Profiler Integration
//!
//! Performance tracking for BLIS operations at multiple granularity levels.
//! Supports Kaizen (continuous improvement) methodology.
//!
//! # Philosophy
//!
//! Kaizen (改善) means "continuous improvement." By tracking performance metrics
//! at each level of the BLIS hierarchy, we can identify bottlenecks and measure
//! the impact of optimizations.
//!
//! # Profiling Levels
//!
//! - **Macro**: L3 cache blocking level (NC x KC tiles)
//! - **Midi**: L2 cache blocking level (MC x KC tiles)
//! - **Micro**: Microkernel level (MR x NR tiles)
//! - **Pack**: Data packing operations
//!
//! # Usage
//!
//! ```
//! use trueno::blis::profiler::{BlisProfiler, BlisProfileLevel};
//!
//! let mut profiler = BlisProfiler::enabled();
//! profiler.record(BlisProfileLevel::Micro, 1000, 384);
//! println!("{}", profiler.summary());
//! ```

// ============================================================================
// Kaizen (Continuous Improvement) - Performance Tracking
// ============================================================================

/// Kaizen metrics for tracking improvement
#[derive(Debug, Clone, Default)]
pub struct KaizenMetrics {
    /// Total FLOP count
    pub flops: u64,
    /// Total time in nanoseconds
    pub time_ns: u64,
    /// Number of measurements
    pub samples: usize,
}

impl KaizenMetrics {
    /// Record a GEMM operation
    pub fn record(&mut self, m: usize, n: usize, k: usize, duration: std::time::Duration) {
        self.flops += 2 * m as u64 * n as u64 * k as u64;
        self.time_ns += duration.as_nanos() as u64;
        self.samples += 1;
    }

    /// Get achieved GFLOP/s
    pub fn gflops(&self) -> f64 {
        if self.time_ns == 0 {
            return 0.0;
        }
        self.flops as f64 / self.time_ns as f64
    }

    /// Reset metrics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

// ============================================================================
// BLIS Profiler Integration
// ============================================================================

/// Profiling level for BLIS operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlisProfileLevel {
    /// L3 block level (NC x KC tiles)
    Macro,
    /// L2 block level (MC x KC tiles)
    Midi,
    /// Microkernel level (MR x NR tiles)
    Micro,
    /// Packing operations
    Pack,
}

/// Statistics for a profiling level
#[derive(Debug, Clone, Default)]
pub struct BlisLevelStats {
    /// Total time in nanoseconds
    pub total_ns: u64,
    /// Number of invocations
    pub count: u64,
    /// Total FLOPs at this level
    pub flops: u64,
}

impl BlisLevelStats {
    /// Record a timing
    pub fn record(&mut self, duration_ns: u64, flops: u64) {
        self.total_ns += duration_ns;
        self.count += 1;
        self.flops += flops;
    }

    /// Get average time in microseconds
    pub fn avg_us(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.total_ns as f64 / self.count as f64 / 1000.0
    }

    /// Get GFLOP/s
    pub fn gflops(&self) -> f64 {
        if self.total_ns == 0 {
            return 0.0;
        }
        self.flops as f64 / self.total_ns as f64
    }
}

/// BLIS-aware profiler
#[derive(Debug, Clone, Default)]
pub struct BlisProfiler {
    /// Per-level statistics
    pub macro_stats: BlisLevelStats,
    pub midi_stats: BlisLevelStats,
    pub micro_stats: BlisLevelStats,
    pub pack_stats: BlisLevelStats,
    /// Whether profiling is enabled
    pub enabled: bool,
}

impl BlisProfiler {
    /// Create a new profiler (disabled by default)
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an enabled profiler
    pub fn enabled() -> Self {
        Self { enabled: true, ..Self::default() }
    }

    /// Record timing for a level
    pub fn record(&mut self, level: BlisProfileLevel, duration_ns: u64, flops: u64) {
        if !self.enabled {
            return;
        }
        match level {
            BlisProfileLevel::Macro => self.macro_stats.record(duration_ns, flops),
            BlisProfileLevel::Midi => self.midi_stats.record(duration_ns, flops),
            BlisProfileLevel::Micro => self.micro_stats.record(duration_ns, flops),
            BlisProfileLevel::Pack => self.pack_stats.record(duration_ns, 0),
        }
    }

    /// Record AVX-512 BLIS macro-level timing.
    /// Contract: avx512-blis-v1.yaml (C-AVX512-PROF-001)
    pub fn record_avx512_blis(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        duration: std::time::Duration,
    ) {
        if !self.enabled {
            return;
        }
        let flops = 2 * m as u64 * n as u64 * k as u64;
        let duration_ns = duration.as_nanos() as u64;
        self.macro_stats.record(duration_ns, flops);
    }

    /// Get total GFLOP/s
    pub fn total_gflops(&self) -> f64 {
        let total_ns = self.macro_stats.total_ns;
        let total_flops = self.macro_stats.flops;
        if total_ns == 0 {
            return 0.0;
        }
        total_flops as f64 / total_ns as f64
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("BLIS Profiler Summary\n");
        s.push_str("=====================\n");
        s.push_str(&format!(
            "Macro: {:.1}us avg, {:.1} GFLOP/s, {} calls\n",
            self.macro_stats.avg_us(),
            self.macro_stats.gflops(),
            self.macro_stats.count
        ));
        s.push_str(&format!(
            "Midi:  {:.1}us avg, {:.1} GFLOP/s, {} calls\n",
            self.midi_stats.avg_us(),
            self.midi_stats.gflops(),
            self.midi_stats.count
        ));
        s.push_str(&format!(
            "Micro: {:.1}us avg, {:.1} GFLOP/s, {} calls\n",
            self.micro_stats.avg_us(),
            self.micro_stats.gflops(),
            self.micro_stats.count
        ));
        s.push_str(&format!(
            "Pack:  {:.1}us avg, {} calls\n",
            self.pack_stats.avg_us(),
            self.pack_stats.count
        ));
        s.push_str(&format!("Total: {:.1} GFLOP/s\n", self.total_gflops()));
        s
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.macro_stats = BlisLevelStats::default();
        self.midi_stats = BlisLevelStats::default();
        self.micro_stats = BlisLevelStats::default();
        self.pack_stats = BlisLevelStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_kaizen_metrics_default() {
        let m = KaizenMetrics::default();
        assert_eq!(m.flops, 0);
        assert_eq!(m.time_ns, 0);
        assert_eq!(m.samples, 0);
    }

    #[test]
    fn test_kaizen_metrics_record() {
        let mut m = KaizenMetrics::default();
        m.record(2, 3, 4, Duration::from_nanos(100));
        assert_eq!(m.flops, 48); // 2 * 2 * 3 * 4
        assert_eq!(m.time_ns, 100);
        assert_eq!(m.samples, 1);
    }

    #[test]
    fn test_kaizen_metrics_gflops() {
        let m =
            KaizenMetrics { flops: 1_000_000_000, time_ns: 1_000_000_000, ..Default::default() };
        assert!((m.gflops() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kaizen_metrics_gflops_zero_time() {
        let m = KaizenMetrics::default();
        assert!((m.gflops() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_kaizen_metrics_reset() {
        let mut m = KaizenMetrics::default();
        m.record(2, 3, 4, Duration::from_nanos(100));
        m.reset();
        assert_eq!(m.flops, 0);
        assert_eq!(m.time_ns, 0);
        assert_eq!(m.samples, 0);
    }

    #[test]
    fn test_blis_level_stats_default() {
        let s = BlisLevelStats::default();
        assert_eq!(s.total_ns, 0);
        assert_eq!(s.count, 0);
        assert_eq!(s.flops, 0);
    }

    #[test]
    fn test_blis_level_stats_record() {
        let mut s = BlisLevelStats::default();
        s.record(1000, 500);
        assert_eq!(s.total_ns, 1000);
        assert_eq!(s.count, 1);
        assert_eq!(s.flops, 500);
    }

    #[test]
    fn test_blis_level_stats_avg_us() {
        let mut s = BlisLevelStats::default();
        s.record(2000, 0);
        s.record(4000, 0);
        assert!((s.avg_us() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_blis_level_stats_avg_us_zero_count() {
        let s = BlisLevelStats::default();
        assert!((s.avg_us() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_blis_level_stats_gflops() {
        let s =
            BlisLevelStats { total_ns: 1_000_000_000, flops: 1_000_000_000, ..Default::default() };
        assert!((s.gflops() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_blis_level_stats_gflops_zero_time() {
        let s = BlisLevelStats::default();
        assert!((s.gflops() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_blis_profiler_new() {
        let p = BlisProfiler::new();
        assert!(!p.enabled);
    }

    #[test]
    fn test_blis_profiler_enabled() {
        let p = BlisProfiler::enabled();
        assert!(p.enabled);
    }

    #[test]
    fn test_blis_profiler_record_disabled() {
        let mut p = BlisProfiler::new();
        p.record(BlisProfileLevel::Micro, 1000, 500);
        assert_eq!(p.micro_stats.count, 0);
    }

    #[test]
    fn test_blis_profiler_record_enabled() {
        let mut p = BlisProfiler::enabled();
        p.record(BlisProfileLevel::Micro, 1000, 500);
        assert_eq!(p.micro_stats.count, 1);
        assert_eq!(p.micro_stats.total_ns, 1000);
        assert_eq!(p.micro_stats.flops, 500);
    }

    #[test]
    fn test_blis_profiler_record_all_levels() {
        let mut p = BlisProfiler::enabled();
        p.record(BlisProfileLevel::Macro, 1000, 100);
        p.record(BlisProfileLevel::Midi, 2000, 200);
        p.record(BlisProfileLevel::Micro, 3000, 300);
        p.record(BlisProfileLevel::Pack, 4000, 400);

        assert_eq!(p.macro_stats.count, 1);
        assert_eq!(p.midi_stats.count, 1);
        assert_eq!(p.micro_stats.count, 1);
        assert_eq!(p.pack_stats.count, 1);
        assert_eq!(p.pack_stats.flops, 0); // Pack doesn't track flops
    }

    #[test]
    fn test_blis_profiler_total_gflops() {
        let mut p = BlisProfiler::enabled();
        p.macro_stats.total_ns = 1_000_000_000;
        p.macro_stats.flops = 1_000_000_000;
        assert!((p.total_gflops() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_blis_profiler_total_gflops_zero_time() {
        let p = BlisProfiler::enabled();
        assert!((p.total_gflops() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_blis_profiler_summary() {
        let p = BlisProfiler::enabled();
        let summary = p.summary();
        assert!(summary.contains("BLIS Profiler Summary"));
        assert!(summary.contains("Macro:"));
        assert!(summary.contains("Midi:"));
        assert!(summary.contains("Micro:"));
        assert!(summary.contains("Pack:"));
        assert!(summary.contains("Total:"));
    }

    #[test]
    fn test_blis_profiler_reset() {
        let mut p = BlisProfiler::enabled();
        p.record(BlisProfileLevel::Micro, 1000, 500);
        p.reset();
        assert_eq!(p.micro_stats.count, 0);
    }

    #[test]
    fn test_blis_profile_level_debug() {
        assert_eq!(format!("{:?}", BlisProfileLevel::Macro), "Macro");
        assert_eq!(format!("{:?}", BlisProfileLevel::Midi), "Midi");
        assert_eq!(format!("{:?}", BlisProfileLevel::Micro), "Micro");
        assert_eq!(format!("{:?}", BlisProfileLevel::Pack), "Pack");
    }

    #[test]
    fn test_blis_profile_level_eq() {
        assert_eq!(BlisProfileLevel::Macro, BlisProfileLevel::Macro);
        assert_ne!(BlisProfileLevel::Macro, BlisProfileLevel::Micro);
    }

    #[test]
    fn test_blis_profile_level_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(BlisProfileLevel::Macro);
        set.insert(BlisProfileLevel::Micro);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&BlisProfileLevel::Macro));
    }
}
