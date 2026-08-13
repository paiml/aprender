//! Syscall statistics tracking for -c mode
//!
//! Sprint 9-10: Statistics mode implementation
//! Sprint 32: Compute block tracing (Toyota Way v2.0)

use std::collections::HashMap;

/// Trace a compute block (multiple Trueno operations) - Sprint 32
///
/// This macro implements **block-level tracing** following Toyota Way principles:
/// - **Genchi Genbutsu**: No backend detection guessing
/// - **Jidoka**: Mandatory adaptive sampling (default: skip if duration < 100μs)
/// - **Muda**: Zero waste - no per-operation tracing overhead
/// - **Poka-Yoke**: Safe by default - cannot `DoS` tracing backend
///
/// # Usage
///
/// ```ignore
/// use crate::otlp_exporter::ComputeBlock;
///
/// let result = trace_compute_block!(otlp_exporter, "calculate_statistics", elements, {
///     // Block of Trueno operations
///     let v = trueno::Vector::from_slice(&data);
///     ExtendedStats {
///         mean: v.mean().unwrap_or(0.0),
///         stddev: v.stddev().unwrap_or(0.0),
///         // ... more operations
///     }
/// });
/// ```
///
/// # Adaptive Sampling
///
/// - If `duration < 100μs`: No span exported (too fast, not interesting)
/// - If `duration >= 100μs`: Export span to OTLP
/// - Debug mode: Use `--trace-compute-all` to bypass threshold
///
/// # Arguments
///
/// * `$exporter` - Optional reference to `OtlpExporter` (None if OTLP disabled)
/// * `$op_name` - Static string with operation name (e.g., "`calculate_statistics`")
/// * `$elements` - Number of elements being processed (usize)
/// * `$block` - Code block containing Trueno operations
///
/// # Returns
///
/// Returns the result of executing `$block`
#[macro_export]
macro_rules! trace_compute_block {
    ($exporter:expr, $op_name:expr, $elements:expr, $block:block) => {{
        let start = std::time::Instant::now();
        let result = $block;
        let duration_us = start.elapsed().as_micros() as u64;

        // Adaptive sampling: export trace data when duration exceeds threshold (Jidoka principle)
        if duration_us >= 100 {
            if let Some(exporter) = $exporter {
                #[cfg(feature = "otlp")]
                {
                    use $crate::otlp_exporter::ComputeBlock;
                    exporter.record_compute_block(ComputeBlock {
                        operation: $op_name,
                        duration_us,
                        elements: $elements,
                        is_slow: duration_us > 100,
                    });
                }
            }
        }

        result
    }};
}

/// Statistics for a single syscall type
#[derive(Debug, Clone, Default)]
pub struct SyscallStats {
    /// Number of times this syscall was called
    pub count: u64,
    /// Number of errors (negative return values)
    pub errors: u64,
    /// Total time spent in this syscall (microseconds)
    pub total_time_us: u64,
    /// Individual syscall durations (for percentile calculations)
    pub durations: Vec<u64>,
}

/// Summary totals for all syscalls
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatTotals {
    pub total_calls: u64,
    pub total_errors: u64,
    pub total_time_us: u64,
}

/// Extended statistics for a syscall (Sprint 19: Trueno integration)
#[derive(Debug, Clone, PartialEq)]
pub struct ExtendedStats {
    pub mean: f32,
    pub stddev: f32,
    pub min: f32,
    pub max: f32,
    pub median: f32, // P50
    pub p75: f32,
    pub p90: f32,
    pub p95: f32,
    pub p99: f32,
}

/// Tracks statistics for all syscalls
#[derive(Debug, Default)]
pub struct StatsTracker {
    /// Map from syscall name to statistics
    stats: HashMap<String, SyscallStats>,
}

impl StatsTracker {
    /// Create a new statistics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a syscall execution
    pub fn record(&mut self, syscall_name: &str, result: i64, duration_us: u64) {
        let entry = self.stats.entry(syscall_name.to_string()).or_default();
        entry.count += 1;
        entry.total_time_us += duration_us;
        entry.durations.push(duration_us); // Sprint 19: Track individual durations
        if result < 0 {
            entry.errors += 1;
        }
    }

    /// Get access to the stats map for CSV export
    pub fn stats_map(&self) -> &HashMap<String, SyscallStats> {
        &self.stats
    }

    /// Calculate totals using Trueno for high-performance SIMD operations
    pub fn calculate_totals_with_trueno(&self) -> StatTotals {
        if self.stats.is_empty() {
            return StatTotals { total_calls: 0, total_errors: 0, total_time_us: 0 };
        }

        // Extract data into vectors for SIMD processing
        let counts: Vec<f32> = self.stats.values().map(|s| s.count as f32).collect();
        let errors: Vec<f32> = self.stats.values().map(|s| s.errors as f32).collect();
        let times: Vec<f32> = self.stats.values().map(|s| s.total_time_us as f32).collect();

        // Use Trueno for SIMD-accelerated sums
        let total_calls = trueno::Vector::from_slice(&counts).sum().unwrap_or(0.0) as u64;
        let total_errors = trueno::Vector::from_slice(&errors).sum().unwrap_or(0.0) as u64;
        let total_time_us = trueno::Vector::from_slice(&times).sum().unwrap_or(0.0) as u64;

        StatTotals { total_calls, total_errors, total_time_us }
    }

    /// Calculate percentile from sorted data
    fn calculate_percentile(sorted_data: &[f32], percentile: f32) -> f32 {
        if sorted_data.is_empty() {
            return 0.0;
        }
        if sorted_data.len() == 1 {
            return sorted_data[0];
        }

        let index = (percentile / 100.0) * (sorted_data.len() - 1) as f32;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            sorted_data[lower]
        } else {
            let weight = index - lower as f32;
            sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
        }
    }

    /// Calculate extended statistics for a syscall using Trueno
    ///
    /// Sprint 32: Now accepts optional `OtlpExporter` for compute block tracing
    ///
    /// # Arguments
    ///
    /// * `syscall_name` - Name of the syscall to compute stats for
    /// * `otlp_exporter` - Optional OTLP exporter for tracing (Sprint 32)
    pub fn calculate_extended_statistics(
        &self,
        syscall_name: &str,
        #[cfg(feature = "otlp")] otlp_exporter: Option<&crate::otlp_exporter::OtlpExporter>,
        #[cfg(not(feature = "otlp"))] _otlp_exporter: Option<&()>,
    ) -> Option<ExtendedStats> {
        let stats = self.stats.get(syscall_name)?;

        if stats.durations.is_empty() {
            return None;
        }

        // Convert durations to f32 for Trueno
        let durations: Vec<f32> = stats.durations.iter().map(|&d| d as f32).collect();
        #[allow(unused_variables)]
        let elements = durations.len();

        // Trace entire compute block (Sprint 32: Block-level tracing)
        #[cfg(feature = "otlp")]
        let result = trace_compute_block!(otlp_exporter, "calculate_statistics", elements, {
            Self::compute_extended_stats_block(&durations)
        });

        #[cfg(not(feature = "otlp"))]
        let result = Self::compute_extended_stats_block(&durations);

        Some(result)
    }

    /// Internal: Compute extended stats block (extracted for tracing)
    fn compute_extended_stats_block(durations: &[f32]) -> ExtendedStats {
        let v = trueno::Vector::from_slice(durations);

        // Use Trueno for basic statistics
        let mean = v.mean().unwrap_or(0.0);
        let stddev = v.stddev().unwrap_or(0.0);
        let min = v.min().unwrap_or(0.0);
        let max = v.max().unwrap_or(0.0);

        // Calculate percentiles (Trueno doesn't have built-in percentile function)
        let mut sorted = durations.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = Self::calculate_percentile(&sorted, 50.0);
        let p75 = Self::calculate_percentile(&sorted, 75.0);
        let p90 = Self::calculate_percentile(&sorted, 90.0);
        let p95 = Self::calculate_percentile(&sorted, 95.0);
        let p99 = Self::calculate_percentile(&sorted, 99.0);

        ExtendedStats { mean, stddev, min, max, median, p75, p90, p95, p99 }
    }

    /// Check if a duration is an anomaly (>threshold σ from mean)
    pub fn is_anomaly(&self, syscall_name: &str, duration_us: u64, threshold: f32) -> bool {
        // Note: Pass None for tracing - this is a quick check, not a compute block
        #[cfg(feature = "otlp")]
        let extended = self.calculate_extended_statistics(syscall_name, None);
        #[cfg(not(feature = "otlp"))]
        let extended = self.calculate_extended_statistics(syscall_name, None);

        if let Some(extended) = extended {
            if extended.stddev > 0.0 {
                let z_score = ((duration_us as f32 - extended.mean) / extended.stddev).abs();
                return z_score > threshold;
            }
        }
        false
    }

    /// Print extended statistics summary to stderr
    ///
    /// Sprint 32: Now accepts optional `OtlpExporter` for compute block tracing
    pub fn print_extended_summary(
        &self,
        threshold: f32,
        #[cfg(feature = "otlp")] otlp_exporter: Option<&crate::otlp_exporter::OtlpExporter>,
        #[cfg(not(feature = "otlp"))] _otlp_exporter: Option<&()>,
    ) {
        if self.stats.is_empty() {
            eprintln!("No syscalls traced.");
            return;
        }

        eprintln!("\n=== Extended Statistics (SIMD-accelerated via Trueno) ===\n");

        // Sort by call count
        let mut sorted: Vec<_> = self.stats.iter().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.1.count));

        for (name, stats) in sorted {
            #[cfg(feature = "otlp")]
            let extended = self.calculate_extended_statistics(name, otlp_exporter);
            #[cfg(not(feature = "otlp"))]
            let extended = self.calculate_extended_statistics(name, _otlp_exporter);

            if let Some(extended) = extended {
                eprintln!("{} ({} calls):", name, stats.count);
                eprintln!("  Mean:         {:.2} μs", extended.mean);
                eprintln!("  Std Dev:      {:.2} μs", extended.stddev);
                eprintln!("  Min:          {:.2} μs", extended.min);
                eprintln!("  Max:          {:.2} μs", extended.max);
                eprintln!("  Median (P50): {:.2} μs", extended.median);
                eprintln!("  P75:          {:.2} μs", extended.p75);
                eprintln!("  P90:          {:.2} μs", extended.p90);
                eprintln!("  P95:          {:.2} μs", extended.p95);
                eprintln!("  P99:          {:.2} μs", extended.p99);

                // Check for anomalies in the data
                if extended.stddev > 0.0 {
                    let max_z = (extended.max - extended.mean) / extended.stddev;
                    if max_z > threshold {
                        eprintln!("  ⚠️  ANOMALY DETECTED: Max duration is {max_z:.1}σ above mean");
                    }
                }
                eprintln!();
            }
        }
    }

    /// Print statistics summary to stderr (matching strace behavior)
    pub fn print_summary(&self) {
        if self.stats.is_empty() {
            eprintln!("No syscalls traced.");
            return;
        }

        // Calculate totals using Trueno for SIMD acceleration
        let totals = self.calculate_totals_with_trueno();
        let total_calls = totals.total_calls;
        let total_errors = totals.total_errors;
        let total_time_us = totals.total_time_us;

        // Sort by call count (descending)
        let mut sorted: Vec<_> = self.stats.iter().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.1.count));

        // Print header
        eprintln!("% time     seconds  usecs/call     calls    errors syscall");
        eprintln!("------ ----------- ----------- --------- --------- ----------------");

        // Print each syscall
        for (name, stats) in sorted {
            let time_percent = if total_time_us > 0 {
                (stats.total_time_us as f64 / total_time_us as f64) * 100.0
            } else {
                0.0
            };
            let seconds = stats.total_time_us as f64 / 1_000_000.0;
            let usecs_per_call = stats.total_time_us.checked_div(stats.count).unwrap_or(0);

            eprintln!(
                "{:6.2} {:>11.6} {:>11} {:>9} {:>9} {}",
                time_percent,
                seconds,
                usecs_per_call,
                stats.count,
                if stats.errors > 0 { stats.errors.to_string() } else { String::new() },
                name
            );
        }

        // Print summary line
        eprintln!("------ ----------- ----------- --------- --------- ----------------");
        let total_seconds = total_time_us as f64 / 1_000_000.0;
        let avg_usecs = total_time_us.checked_div(total_calls).unwrap_or(0);
        eprintln!(
            "100.00 {:>11.6} {:>11} {:>9} {:>9} total",
            total_seconds,
            avg_usecs,
            total_calls,
            if total_errors > 0 { total_errors.to_string() } else { String::new() }
        );
    }
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(SyscallStats: Send, Sync);
static_assertions::assert_impl_all!(StatTotals: Send, Sync);
static_assertions::assert_impl_all!(ExtendedStats: Send, Sync);
static_assertions::assert_impl_all!(StatsTracker: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use trueno::Vector;

    #[test]
    fn test_stats_tracker_records_calls() {
        let mut tracker = StatsTracker::new();
        tracker.record("open", 3, 100);
        tracker.record("read", 10, 50);
        tracker.record("read", 10, 75);

        assert_eq!(tracker.stats.get("open").expect("test").count, 1);
        assert_eq!(tracker.stats.get("read").expect("test").count, 2);
        assert_eq!(tracker.stats.get("read").expect("test").total_time_us, 125);
    }

    #[test]
    fn test_stats_tracker_records_errors() {
        let mut tracker = StatsTracker::new();
        tracker.record("open", 3, 100); // success
        tracker.record("open", -2, 50); // error (ENOENT)
        tracker.record("open", -13, 25); // error (EACCES)

        let stats = tracker.stats.get("open").expect("test");
        assert_eq!(stats.count, 3);
        assert_eq!(stats.errors, 2);
        assert_eq!(stats.total_time_us, 175);
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = StatsTracker::new();
        // Should not panic
        tracker.print_summary();
    }

    #[test]
    fn test_syscall_stats_default() {
        let stats = SyscallStats::default();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.total_time_us, 0);
    }

    #[test]
    fn test_syscall_stats_clone() {
        let stats1 = SyscallStats {
            count: 42,
            errors: 3,
            total_time_us: 1234,
            durations: vec![100, 200, 934], // Sprint 19
        };
        let stats2 = stats1.clone();
        assert_eq!(stats2.count, 42);
        assert_eq!(stats2.errors, 3);
        assert_eq!(stats2.total_time_us, 1234);
    }

    #[test]
    fn test_syscall_stats_debug() {
        let stats = SyscallStats {
            count: 10,
            errors: 2,
            total_time_us: 5000,
            durations: vec![500, 500, 1000, 1000, 2000], // Sprint 19
        };
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("count"));
        assert!(debug_str.contains("10"));
    }

    #[test]
    fn test_stats_tracker_debug() {
        let mut tracker = StatsTracker::new();
        tracker.record("test", 0, 100);
        let debug_str = format!("{:?}", tracker);
        assert!(debug_str.contains("StatsTracker"));
    }

    #[test]
    fn test_stats_tracker_multiple_syscalls() {
        let mut tracker = StatsTracker::new();
        tracker.record("open", 3, 100);
        tracker.record("read", 10, 200);
        tracker.record("write", 20, 150);
        tracker.record("close", 0, 50);

        assert_eq!(tracker.stats.len(), 4);
        assert_eq!(tracker.stats.get("open").expect("test").count, 1);
        assert_eq!(tracker.stats.get("read").expect("test").count, 1);
        assert_eq!(tracker.stats.get("write").expect("test").count, 1);
        assert_eq!(tracker.stats.get("close").expect("test").count, 1);
    }

    #[test]
    fn test_stats_tracker_zero_time() {
        let mut tracker = StatsTracker::new();
        tracker.record("test", 0, 0);
        tracker.record("test", 0, 0);

        let stats = tracker.stats.get("test").expect("test");
        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_time_us, 0);
        // Should not panic on division by zero
        tracker.print_summary();
    }

    #[test]
    fn test_stats_tracker_all_errors() {
        let mut tracker = StatsTracker::new();
        tracker.record("fail", -1, 10);
        tracker.record("fail", -2, 20);
        tracker.record("fail", -13, 30);

        let stats = tracker.stats.get("fail").expect("test");
        assert_eq!(stats.count, 3);
        assert_eq!(stats.errors, 3);
        assert_eq!(stats.total_time_us, 60);
    }

    #[test]
    fn test_stats_tracker_mixed_success_failure() {
        let mut tracker = StatsTracker::new();
        tracker.record("open", 3, 100); // success
        tracker.record("open", -2, 50); // error
        tracker.record("open", 5, 75); // success
        tracker.record("open", -13, 25); // error

        let stats = tracker.stats.get("open").expect("test");
        assert_eq!(stats.count, 4);
        assert_eq!(stats.errors, 2);
        assert_eq!(stats.total_time_us, 250);
    }

    #[test]
    fn test_stats_tracker_large_numbers() {
        let mut tracker = StatsTracker::new();
        // Test with large time values
        tracker.record("big", 0, u64::MAX / 2);
        tracker.record("big", 0, u64::MAX / 2);

        let stats = tracker.stats.get("big").expect("test");
        assert_eq!(stats.count, 2);
        // Should handle large numbers without overflow
        assert!(stats.total_time_us > 0);
    }

    #[test]
    fn test_stats_tracker_sorting_by_count() {
        let mut tracker = StatsTracker::new();
        // Add syscalls with different counts
        tracker.record("rare", 0, 10);

        tracker.record("common", 0, 20);
        tracker.record("common", 0, 30);
        tracker.record("common", 0, 40);

        tracker.record("medium", 0, 50);
        tracker.record("medium", 0, 60);

        // Verify counts
        assert_eq!(tracker.stats.get("rare").expect("test").count, 1);
        assert_eq!(tracker.stats.get("medium").expect("test").count, 2);
        assert_eq!(tracker.stats.get("common").expect("test").count, 3);

        // Print should sort by count (descending)
        tracker.print_summary();
    }

    #[test]
    fn test_stats_tracker_percentage_calculation() {
        let mut tracker = StatsTracker::new();
        tracker.record("half", 0, 500);
        tracker.record("quarter", 0, 250);
        tracker.record("quarter", 0, 250);

        // Total time: 1000us
        // half: 500us = 50%
        // quarter: 500us = 50% (but 2 calls)
        let half_stats = tracker.stats.get("half").expect("test");
        let quarter_stats = tracker.stats.get("quarter").expect("test");

        assert_eq!(half_stats.total_time_us, 500);
        assert_eq!(quarter_stats.total_time_us, 500);

        tracker.print_summary();
    }

    #[test]
    fn test_stats_tracker_record_zero_result() {
        let mut tracker = StatsTracker::new();
        tracker.record("success", 0, 100);

        let stats = tracker.stats.get("success").expect("test");
        assert_eq!(stats.count, 1);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_stats_tracker_record_positive_result() {
        let mut tracker = StatsTracker::new();
        tracker.record("read", 1024, 100); // read 1024 bytes

        let stats = tracker.stats.get("read").expect("test");
        assert_eq!(stats.count, 1);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_stats_tracker_record_negative_result() {
        let mut tracker = StatsTracker::new();
        tracker.record("open", -2, 50); // ENOENT error

        let stats = tracker.stats.get("open").expect("test");
        assert_eq!(stats.count, 1);
        assert_eq!(stats.errors, 1);
    }

    #[test]
    fn test_trueno_sum_integration() {
        // RED phase: Test that we can use Trueno for sum operations
        let tracker = StatsTracker::new();

        // Create sample data
        let counts = vec![10.0_f32, 20.0, 30.0, 40.0];
        let v = Vector::from_slice(&counts);

        // Use Trueno to sum
        let result = v.sum().expect("test");
        assert_eq!(result, 100.0);

        // This test passes - now we need to actually integrate Trueno into StatsTracker
        let _ = tracker; // Use tracker to avoid warning
    }

    #[test]
    fn test_stats_tracker_uses_trueno_for_sums() {
        // RED phase: Test that StatsTracker uses Trueno for sum calculations
        let mut tracker = StatsTracker::new();

        // Record some syscalls with timing data
        tracker.record("open", 3, 100);
        tracker.record("read", 10, 200);
        tracker.record("write", 20, 300);
        tracker.record("close", 0, 400);

        // Calculate totals using Trueno (this will fail until we implement it)
        let totals = tracker.calculate_totals_with_trueno();

        assert_eq!(totals.total_calls, 4);
        assert_eq!(totals.total_time_us, 1000);
        assert_eq!(totals.total_errors, 0);
    }

    #[test]
    fn test_calculate_extended_statistics() {
        let mut tracker = StatsTracker::new();

        // Add multiple durations for statistical analysis
        tracker.record("read", 10, 100);
        tracker.record("read", 10, 200);
        tracker.record("read", 10, 300);
        tracker.record("read", 10, 400);
        tracker.record("read", 10, 500);

        #[cfg(feature = "otlp")]
        let extended = tracker.calculate_extended_statistics("read", None);
        #[cfg(not(feature = "otlp"))]
        let extended = tracker.calculate_extended_statistics("read", None);

        let stats = extended.expect("Should calculate extended stats");
        assert!((stats.mean - 300.0).abs() < 1.0);
        assert!(stats.min >= 100.0);
        assert!(stats.max <= 500.0);
        assert!((stats.median - 300.0).abs() < 1.0);
    }

    #[test]
    fn test_calculate_extended_statistics_empty() {
        let tracker = StatsTracker::new();

        #[cfg(feature = "otlp")]
        let extended = tracker.calculate_extended_statistics("nonexistent", None);
        #[cfg(not(feature = "otlp"))]
        let extended = tracker.calculate_extended_statistics("nonexistent", None);

        assert!(extended.is_none());
    }

    #[test]
    fn test_calculate_extended_statistics_single_value() {
        let mut tracker = StatsTracker::new();
        tracker.record("single", 0, 100);

        #[cfg(feature = "otlp")]
        let extended = tracker.calculate_extended_statistics("single", None);
        #[cfg(not(feature = "otlp"))]
        let extended = tracker.calculate_extended_statistics("single", None);

        let stats = extended.expect("Should calculate stats for single value");
        assert!((stats.mean - 100.0).abs() < f32::EPSILON);
        assert!((stats.min - 100.0).abs() < f32::EPSILON);
        assert!((stats.max - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_is_anomaly_detection() {
        let mut tracker = StatsTracker::new();

        // Add baseline with consistent times
        for _ in 0..50 {
            tracker.record("consistent", 0, 100);
        }
        // Add one outlier
        tracker.record("consistent", 0, 1000);

        // The 1000μs call should be an anomaly (> 3σ)
        let is_anomaly = tracker.is_anomaly("consistent", 1000, 3.0);
        // Note: depends on whether there's enough variance
        // With 50 x 100μs and 1 x 1000μs, the stddev is significant
        assert!(is_anomaly);

        // A normal call should not be an anomaly
        let is_normal = tracker.is_anomaly("consistent", 100, 3.0);
        assert!(!is_normal);
    }

    #[test]
    fn test_is_anomaly_no_stddev() {
        let mut tracker = StatsTracker::new();

        // All same values - stddev is 0
        tracker.record("same", 0, 100);
        tracker.record("same", 0, 100);
        tracker.record("same", 0, 100);

        // Should not detect anomaly when stddev is 0
        let result = tracker.is_anomaly("same", 100, 2.0);
        assert!(!result);
    }

    #[test]
    fn test_is_anomaly_nonexistent_syscall() {
        let tracker = StatsTracker::new();

        // Should return false for nonexistent syscall
        let result = tracker.is_anomaly("nonexistent", 100, 2.0);
        assert!(!result);
    }

    #[test]
    fn test_stats_map_access() {
        let mut tracker = StatsTracker::new();
        tracker.record("test", 0, 100);

        let stats_map = tracker.stats_map();
        assert_eq!(stats_map.len(), 1);
        assert!(stats_map.contains_key("test"));
    }

    #[test]
    fn test_calculate_percentile_empty() {
        let percentile = StatsTracker::calculate_percentile(&[], 50.0);
        assert!((percentile - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_calculate_percentile_single() {
        let data = vec![42.0];
        let percentile = StatsTracker::calculate_percentile(&data, 50.0);
        assert!((percentile - 42.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_calculate_percentile_multiple() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let p50 = StatsTracker::calculate_percentile(&data, 50.0);
        assert!((p50 - 30.0).abs() < f32::EPSILON);

        let p0 = StatsTracker::calculate_percentile(&data, 0.0);
        assert!((p0 - 10.0).abs() < f32::EPSILON);

        let p100 = StatsTracker::calculate_percentile(&data, 100.0);
        assert!((p100 - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_calculate_percentile_interpolation() {
        let data = vec![10.0, 20.0, 30.0, 40.0];
        // P50 should interpolate between 20 and 30
        let p50 = StatsTracker::calculate_percentile(&data, 50.0);
        assert!((p50 - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_print_extended_summary() {
        let mut tracker = StatsTracker::new();

        // Add varied data to trigger anomaly detection
        for _ in 0..10 {
            tracker.record("read", 10, 100);
        }
        tracker.record("read", 10, 10000); // anomaly

        // Should not panic
        #[cfg(feature = "otlp")]
        tracker.print_extended_summary(2.0, None);
        #[cfg(not(feature = "otlp"))]
        tracker.print_extended_summary(2.0, None);
    }

    #[test]
    fn test_print_extended_summary_empty() {
        let tracker = StatsTracker::new();

        // Should not panic on empty tracker
        #[cfg(feature = "otlp")]
        tracker.print_extended_summary(2.0, None);
        #[cfg(not(feature = "otlp"))]
        tracker.print_extended_summary(2.0, None);
    }

    #[test]
    fn test_totals_with_errors() {
        let mut tracker = StatsTracker::new();

        tracker.record("open", 3, 100);
        tracker.record("open", -2, 50);
        tracker.record("read", 10, 200);
        tracker.record("read", -1, 100);

        let totals = tracker.calculate_totals_with_trueno();

        assert_eq!(totals.total_calls, 4);
        assert_eq!(totals.total_errors, 2);
        assert_eq!(totals.total_time_us, 450);
    }

    #[test]
    fn test_extended_stats_struct() {
        let stats = ExtendedStats {
            mean: 100.0,
            stddev: 10.0,
            min: 80.0,
            max: 120.0,
            median: 100.0,
            p75: 105.0,
            p90: 115.0,
            p95: 118.0,
            p99: 119.0,
        };

        // Test PartialEq
        let stats2 = stats.clone();
        assert_eq!(stats, stats2);

        // Test Debug
        let debug = format!("{:?}", stats);
        assert!(debug.contains("mean"));
    }

    #[test]
    fn test_stat_totals_equality() {
        let totals1 = StatTotals { total_calls: 10, total_errors: 2, total_time_us: 1000 };
        let totals2 = StatTotals { total_calls: 10, total_errors: 2, total_time_us: 1000 };
        assert_eq!(totals1, totals2);
    }
}
