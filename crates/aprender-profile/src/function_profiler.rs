//! Function-level profiling with DWARF correlation
//!
//! GitHub Issue #1: Function-level profiling for performance analysis
//!
//! This module provides function-level timing aggregation by correlating
//! syscalls with DWARF debug info to identify performance bottlenecks.
//!
//! Features:
//! - Function-level syscall attribution and timing
//! - Call graph tracking (parent-child relationships)
//! - Hot path analysis (most frequently executed paths)
//! - I/O bottleneck detection (slow operations)
//! - Flamegraph export support

use std::collections::HashMap;

/// I/O syscalls that should be tracked for bottleneck detection
const IO_SYSCALLS: &[&str] = &[
    "read",
    "write",
    "readv",
    "writev",
    "pread64",
    "pwrite64",
    "openat",
    "open",
    "close",
    "fsync",
    "fdatasync",
    "sync",
    "sendfile",
    "splice",
    "tee",
    "vmsplice",
];

/// Threshold for slow I/O operations (1ms = 1000 microseconds)
const SLOW_IO_THRESHOLD_US: u64 = 1000;

/// Statistics for a single function
#[derive(Debug, Clone, Default)]
pub struct FunctionStats {
    /// Number of syscalls attributed to this function
    pub syscall_count: u64,
    /// Total time spent in syscalls from this function (microseconds)
    pub total_time_us: u64,
    /// Functions called by this function (call graph) - Reserved for future use
    pub callees: HashMap<String, u64>,
    /// Number of times this is an I/O syscall - Reserved for future use
    pub io_syscalls: u64,
    /// Number of slow I/O operations (>1ms) - Reserved for future use
    pub slow_io_count: u64,
}

/// Tracks function-level profiling statistics
#[derive(Debug, Default)]
pub struct FunctionProfiler {
    /// Map from function name to statistics
    stats: HashMap<String, FunctionStats>,
}

impl FunctionProfiler {
    /// Create a new function profiler
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a syscall execution attributed to a function
    ///
    /// # Arguments
    /// * `function_name` - Name of the function making the syscall
    /// * `syscall_name` - Name of the syscall being made (for I/O detection)
    /// * `duration_us` - Duration of the syscall in microseconds
    /// * `caller_name` - Optional name of the function that called this function (for call graph)
    pub fn record(
        &mut self,
        function_name: &str,
        syscall_name: &str,
        duration_us: u64,
        caller_name: Option<&str>,
    ) {
        let entry = self.stats.entry(function_name.to_string()).or_default();
        entry.syscall_count += 1;
        entry.total_time_us += duration_us;

        // Track I/O syscalls
        if IO_SYSCALLS.contains(&syscall_name) {
            entry.io_syscalls += 1;

            // Track I/O operations exceeding 1ms threshold
            if duration_us > SLOW_IO_THRESHOLD_US {
                entry.slow_io_count += 1;
            }
        }

        // Track call graph (parent -> child relationship)
        if let Some(caller) = caller_name {
            let caller_entry = self.stats.entry(caller.to_string()).or_default();
            *caller_entry.callees.entry(function_name.to_string()).or_insert(0) += 1;
        }
    }

    /// Export profiling data in flamegraph format
    ///
    /// Generates flamegraph-compatible output format (folded stacks)
    /// Each line: "func1;func2;func3 samples"
    ///
    /// # Arguments
    /// * `writer` - Where to write the flamegraph data
    pub fn export_flamegraph<W: std::io::Write>(&self, mut writer: W) -> std::io::Result<()> {
        // Build flamegraph samples from call graph
        // Format: "caller;callee sample_count"

        for (function, stats) in &self.stats {
            // Add root-level functions (no callers)
            if !self.has_caller(function) {
                writeln!(writer, "{} {}", function, stats.syscall_count)?;
            }

            // Add caller->callee relationships
            for (callee, count) in &stats.callees {
                writeln!(writer, "{function};{callee} {count}")?;
            }
        }

        Ok(())
    }

    /// Check if a function has any callers
    fn has_caller(&self, function: &str) -> bool {
        self.stats.values().any(|stats| stats.callees.contains_key(function))
    }

    /// Print function timing summary to stderr
    pub fn print_summary(&self) {
        if self.stats.is_empty() {
            eprintln!("\nNo function profiling data collected.");
            return;
        }

        // Sort by total time (descending)
        let mut sorted: Vec<_> = self.stats.iter().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.1.total_time_us));

        eprintln!("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  Function Timing Summary (sorted by total time)                                                   ║");
        eprintln!("╚════════════════════════════════════════════════════════════════════════════════════════════════════╝");
        eprintln!();
        eprintln!(
            "{:<40} {:>10} {:>12} {:>12} {:>10} {:>10}",
            "Function", "Calls", "Total Time", "Avg Time", "I/O Ops", "Slow I/O"
        );
        eprintln!("{}", "─".repeat(104));

        for (function, stats) in &sorted {
            let total_seconds = stats.total_time_us as f64 / 1_000_000.0;
            let avg_us = stats.total_time_us.checked_div(stats.syscall_count).unwrap_or(0);
            let avg_seconds = avg_us as f64 / 1_000_000.0;

            // Highlight functions with high-latency I/O
            let marker = if stats.slow_io_count > 0 { "⚠️ " } else { "   " };

            eprintln!(
                "{}{:<37} {:>10} {:>11.6}s {:>11.6}s {:>10} {:>10}",
                marker,
                function,
                stats.syscall_count,
                total_seconds,
                avg_seconds,
                stats.io_syscalls,
                stats.slow_io_count
            );
        }

        eprintln!("{}", "─".repeat(104));

        // Print hot path analysis (top 10 most active functions)
        if sorted.len() > 1 {
            eprintln!();
            eprintln!("╔════════════════════════════════════════════════════════════════════════════════════════════════════╗");
            eprintln!("║  Hot Path Analysis (top 10 most active functions)                                                 ║");
            eprintln!("╚════════════════════════════════════════════════════════════════════════════════════════════════════╝");
            eprintln!();

            // Already sorted by total_time_us descending
            let hot_functions = sorted.iter().take(10);

            for (rank, (function, stats)) in hot_functions.enumerate() {
                let total_seconds = stats.total_time_us as f64 / 1_000_000.0;
                let percent = if self.stats.values().map(|s| s.total_time_us).sum::<u64>() > 0 {
                    (stats.total_time_us as f64
                        / self.stats.values().map(|s| s.total_time_us).sum::<u64>() as f64)
                        * 100.0
                } else {
                    0.0
                };

                eprintln!(
                    "{}. {} - {:.2}% of total time ({:.6}s, {} syscalls)",
                    rank + 1,
                    function,
                    percent,
                    total_seconds,
                    stats.syscall_count
                );

                // Show call graph for this hot function
                if !stats.callees.is_empty() {
                    let mut callees: Vec<_> = stats.callees.iter().collect();
                    callees.sort_by(|a, b| b.1.cmp(a.1));

                    for (callee, count) in callees.iter().take(5) {
                        // Top 5 callees
                        eprintln!(
                            "   └─> {} ({} call{})",
                            callee,
                            count,
                            if **count == 1 { "" } else { "s" }
                        );
                    }
                }
                eprintln!();
            }
        }

        // Print call graph if available
        let has_call_graph = sorted.iter().any(|(_, stats)| !stats.callees.is_empty());
        if has_call_graph {
            eprintln!("╔════════════════════════════════════════════════════════════════════════════════════════════════════╗");
            eprintln!("║  Call Graph (parent → child relationships)                                                        ║");
            eprintln!("╚════════════════════════════════════════════════════════════════════════════════════════════════════╝");
            eprintln!();

            for (function, stats) in &sorted {
                if !stats.callees.is_empty() {
                    eprintln!("{function} calls:");

                    // Sort callees by call count (descending)
                    let mut callees: Vec<_> = stats.callees.iter().collect();
                    callees.sort_by(|a, b| b.1.cmp(a.1));

                    for (callee, count) in callees {
                        eprintln!(
                            "  └─> {} ({} call{})",
                            callee,
                            count,
                            if *count == 1 { "" } else { "s" }
                        );
                    }
                    eprintln!();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_profiler_new() {
        let profiler = FunctionProfiler::new();
        assert!(profiler.stats.is_empty());
    }

    #[test]
    fn test_function_profiler_record() {
        let mut profiler = FunctionProfiler::new();
        profiler.record("main", "write", 1000, None);
        profiler.record("main", "read", 2000, None);
        profiler.record("helper", "open", 500, None);

        assert_eq!(profiler.stats.len(), 2);
        assert_eq!(profiler.stats.get("main").expect("test").syscall_count, 2);
        assert_eq!(profiler.stats.get("main").expect("test").total_time_us, 3000);
        assert_eq!(profiler.stats.get("helper").expect("test").syscall_count, 1);
        assert_eq!(profiler.stats.get("helper").expect("test").total_time_us, 500);
    }

    #[test]
    fn test_function_profiler_empty_summary() {
        let profiler = FunctionProfiler::new();
        // Should not panic
        profiler.print_summary();
    }

    #[test]
    fn test_function_stats_default() {
        let stats = FunctionStats::default();
        assert_eq!(stats.syscall_count, 0);
        assert_eq!(stats.total_time_us, 0);
    }

    #[test]
    fn test_function_profiler_print_summary_with_data() {
        let mut profiler = FunctionProfiler::new();
        profiler.record("main", "write", 1000000, None); // 1 second
        profiler.record("main", "read", 500000, None); // 0.5 seconds
        profiler.record("helper", "open", 250000, None); // 0.25 seconds
        profiler.record("foo", "close", 100000, None); // 0.1 seconds

        // This exercises the print_summary() code path with data
        // including sorting, formatting, and calculations
        profiler.print_summary();

        // Verify internal state
        assert_eq!(profiler.stats.len(), 3);
    }

    #[test]
    fn test_function_profiler_sorting_by_total_time() {
        let mut profiler = FunctionProfiler::new();
        // Record in non-sorted order
        profiler.record("slow_func", "write", 5000000, None); // 5 seconds total
        profiler.record("fast_func", "brk", 100000, None); // 0.1 seconds total
        profiler.record("medium_func", "read", 1000000, None); // 1 second total

        // print_summary() should sort by total time (descending)
        profiler.print_summary();

        // Verify data is present
        assert_eq!(profiler.stats.get("slow_func").expect("test").total_time_us, 5000000);
        assert_eq!(profiler.stats.get("fast_func").expect("test").total_time_us, 100000);
        assert_eq!(profiler.stats.get("medium_func").expect("test").total_time_us, 1000000);
    }

    #[test]
    fn test_function_profiler_average_calculation() {
        let mut profiler = FunctionProfiler::new();
        // Record multiple calls to test average calculation
        profiler.record("test_func", "write", 1000, None);
        profiler.record("test_func", "read", 2000, None);
        profiler.record("test_func", "open", 3000, None);

        let stats = profiler.stats.get("test_func").expect("test");
        assert_eq!(stats.syscall_count, 3);
        assert_eq!(stats.total_time_us, 6000);

        // Average should be 2000 microseconds
        let avg = stats.total_time_us / stats.syscall_count;
        assert_eq!(avg, 2000);

        profiler.print_summary();
    }

    #[test]
    fn test_function_profiler_zero_syscalls_edge_case() {
        let mut profiler = FunctionProfiler::new();
        // Manually insert a function with 0 syscalls (edge case)
        profiler.stats.insert(
            "never_called".to_string(),
            FunctionStats {
                syscall_count: 0,
                total_time_us: 0,
                callees: HashMap::new(),
                io_syscalls: 0,
                slow_io_count: 0,
            },
        );

        // Should handle division by zero gracefully
        profiler.print_summary();

        let stats = profiler.stats.get("never_called").expect("test");
        assert_eq!(stats.syscall_count, 0);
    }

    #[test]
    fn test_io_syscall_tracking() {
        let mut profiler = FunctionProfiler::new();

        // Record I/O syscalls
        profiler.record("io_func", "read", 500, None);
        profiler.record("io_func", "write", 600, None);
        profiler.record("io_func", "open", 700, None);
        profiler.record("io_func", "close", 400, None);

        let stats = profiler.stats.get("io_func").expect("test");
        assert_eq!(stats.io_syscalls, 4);
        assert_eq!(stats.syscall_count, 4);
        assert_eq!(stats.slow_io_count, 0); // All under 1ms
    }

    #[test]
    fn test_non_io_syscalls_not_tracked() {
        let mut profiler = FunctionProfiler::new();

        // Record non-I/O syscalls
        profiler.record("compute_func", "brk", 500, None);
        profiler.record("compute_func", "mmap", 600, None);
        profiler.record("compute_func", "getpid", 100, None);

        let stats = profiler.stats.get("compute_func").expect("test");
        assert_eq!(stats.io_syscalls, 0);
        assert_eq!(stats.syscall_count, 3);
        assert_eq!(stats.slow_io_count, 0);
    }

    #[test]
    fn test_slow_io_detection() {
        let mut profiler = FunctionProfiler::new();

        // Record slow I/O operations (>1ms = >1000us)
        profiler.record("slow_io_func", "read", 2000, None); // 2ms - SLOW
        profiler.record("slow_io_func", "write", 5000, None); // 5ms - SLOW
        profiler.record("slow_io_func", "fsync", 10000, None); // 10ms - SLOW

        let stats = profiler.stats.get("slow_io_func").expect("test");
        assert_eq!(stats.io_syscalls, 3);
        assert_eq!(stats.slow_io_count, 3);
        assert_eq!(stats.syscall_count, 3);
    }

    #[test]
    fn test_fast_io_not_marked_slow() {
        let mut profiler = FunctionProfiler::new();

        // Record fast I/O operations (<1ms)
        profiler.record("fast_io_func", "read", 100, None); // 0.1ms - fast
        profiler.record("fast_io_func", "write", 500, None); // 0.5ms - fast
        profiler.record("fast_io_func", "close", 50, None); // 0.05ms - fast

        let stats = profiler.stats.get("fast_io_func").expect("test");
        assert_eq!(stats.io_syscalls, 3);
        assert_eq!(stats.slow_io_count, 0); // None are slow
        assert_eq!(stats.syscall_count, 3);
    }

    #[test]
    fn test_mixed_io_and_non_io_operations() {
        let mut profiler = FunctionProfiler::new();

        // Mix of I/O and non-I/O syscalls
        profiler.record("mixed_func", "read", 1500, None); // I/O, slow
        profiler.record("mixed_func", "brk", 100, None); // Non-I/O
        profiler.record("mixed_func", "write", 500, None); // I/O, fast
        profiler.record("mixed_func", "mmap", 200, None); // Non-I/O
        profiler.record("mixed_func", "fsync", 3000, None); // I/O, slow

        let stats = profiler.stats.get("mixed_func").expect("test");
        assert_eq!(stats.syscall_count, 5);
        assert_eq!(stats.io_syscalls, 3); // read, write, fsync
        assert_eq!(stats.slow_io_count, 2); // read and fsync
        assert_eq!(stats.total_time_us, 5300);
    }

    #[test]
    fn test_slow_io_threshold_boundary() {
        let mut profiler = FunctionProfiler::new();

        // Test exactly at threshold
        profiler.record("boundary_func", "read", 1000, None); // Exactly 1ms - NOT slow (>1ms)
        profiler.record("boundary_func", "write", 999, None); // Just under - NOT slow
        profiler.record("boundary_func", "open", 1001, None); // Just over - SLOW

        let stats = profiler.stats.get("boundary_func").expect("test");
        assert_eq!(stats.io_syscalls, 3);
        assert_eq!(stats.slow_io_count, 1); // Only the 1001us operation
    }

    #[test]
    fn test_all_io_syscall_types() {
        let mut profiler = FunctionProfiler::new();

        // Test all I/O syscall types from IO_SYSCALLS constant
        profiler.record("io_types", "read", 100, None);
        profiler.record("io_types", "write", 100, None);
        profiler.record("io_types", "readv", 100, None);
        profiler.record("io_types", "writev", 100, None);
        profiler.record("io_types", "pread64", 100, None);
        profiler.record("io_types", "pwrite64", 100, None);
        profiler.record("io_types", "openat", 100, None);
        profiler.record("io_types", "open", 100, None);
        profiler.record("io_types", "close", 100, None);
        profiler.record("io_types", "fsync", 100, None);
        profiler.record("io_types", "fdatasync", 100, None);
        profiler.record("io_types", "sync", 100, None);
        profiler.record("io_types", "sendfile", 100, None);
        profiler.record("io_types", "splice", 100, None);
        profiler.record("io_types", "tee", 100, None);
        profiler.record("io_types", "vmsplice", 100, None);

        let stats = profiler.stats.get("io_types").expect("test");
        assert_eq!(stats.io_syscalls, 16); // All 16 I/O syscall types
        assert_eq!(stats.syscall_count, 16);
        assert_eq!(stats.slow_io_count, 0); // All fast
    }

    #[test]
    fn test_call_graph_single_relationship() {
        let mut profiler = FunctionProfiler::new();

        // main calls helper
        profiler.record("helper", "write", 1000, Some("main"));

        let main_stats = profiler.stats.get("main").expect("test");
        assert_eq!(main_stats.callees.len(), 1);
        assert_eq!(*main_stats.callees.get("helper").expect("test"), 1);
    }

    #[test]
    fn test_call_graph_multiple_calls() {
        let mut profiler = FunctionProfiler::new();

        // main calls helper multiple times
        profiler.record("helper", "write", 1000, Some("main"));
        profiler.record("helper", "read", 2000, Some("main"));
        profiler.record("helper", "open", 500, Some("main"));

        let main_stats = profiler.stats.get("main").expect("test");
        assert_eq!(main_stats.callees.len(), 1);
        assert_eq!(*main_stats.callees.get("helper").expect("test"), 3);
    }

    #[test]
    fn test_call_graph_multiple_callees() {
        let mut profiler = FunctionProfiler::new();

        // main calls multiple different functions
        profiler.record("helper_a", "write", 1000, Some("main"));
        profiler.record("helper_b", "read", 2000, Some("main"));
        profiler.record("helper_c", "open", 500, Some("main"));

        let main_stats = profiler.stats.get("main").expect("test");
        assert_eq!(main_stats.callees.len(), 3);
        assert_eq!(*main_stats.callees.get("helper_a").expect("test"), 1);
        assert_eq!(*main_stats.callees.get("helper_b").expect("test"), 1);
        assert_eq!(*main_stats.callees.get("helper_c").expect("test"), 1);
    }

    #[test]
    fn test_call_graph_nested_calls() {
        let mut profiler = FunctionProfiler::new();

        // main -> helper_a -> helper_b
        profiler.record("helper_a", "write", 1000, Some("main"));
        profiler.record("helper_b", "read", 2000, Some("helper_a"));

        let main_stats = profiler.stats.get("main").expect("test");
        assert_eq!(main_stats.callees.len(), 1);
        assert_eq!(*main_stats.callees.get("helper_a").expect("test"), 1);

        let helper_a_stats = profiler.stats.get("helper_a").expect("test");
        assert_eq!(helper_a_stats.callees.len(), 1);
        assert_eq!(*helper_a_stats.callees.get("helper_b").expect("test"), 1);
    }

    #[test]
    fn test_call_graph_with_no_caller() {
        let mut profiler = FunctionProfiler::new();

        // Function called with no caller (e.g., from main)
        profiler.record("main", "write", 1000, None);

        let main_stats = profiler.stats.get("main").expect("test");
        assert_eq!(main_stats.callees.len(), 0);
        assert_eq!(main_stats.syscall_count, 1);
    }

    #[test]
    fn test_call_graph_mixed_with_without_caller() {
        let mut profiler = FunctionProfiler::new();

        // main is called without caller
        profiler.record("main", "write", 1000, None);
        // main calls helper
        profiler.record("helper", "read", 2000, Some("main"));
        // helper is also called without caller (e.g., recursion or external call)
        profiler.record("helper", "open", 500, None);

        let main_stats = profiler.stats.get("main").expect("test");
        assert_eq!(main_stats.syscall_count, 1);
        assert_eq!(main_stats.callees.len(), 1);
        assert_eq!(*main_stats.callees.get("helper").expect("test"), 1);

        let helper_stats = profiler.stats.get("helper").expect("test");
        assert_eq!(helper_stats.syscall_count, 2);
        assert_eq!(helper_stats.callees.len(), 0);
    }

    #[test]
    fn test_hot_path_analysis_sorting() {
        let mut profiler = FunctionProfiler::new();

        // Create functions with varying execution times
        profiler.record("func_slow", "write", 5000000, None); // 5s
        profiler.record("func_medium", "read", 1000000, None); // 1s
        profiler.record("func_fast", "open", 100000, None); // 0.1s
        profiler.record("func_very_slow", "fsync", 10000000, None); // 10s
        profiler.record("func_quick", "close", 50000, None); // 0.05s

        // print_summary() should display hot path analysis with top functions
        // sorted by total_time_us (descending)
        profiler.print_summary();

        // Verify sorting order in stats
        let mut sorted: Vec<_> = profiler.stats.iter().collect();
        sorted.sort_by(|a, b| b.1.total_time_us.cmp(&a.1.total_time_us));

        assert_eq!(sorted[0].0, "func_very_slow");
        assert_eq!(sorted[1].0, "func_slow");
        assert_eq!(sorted[2].0, "func_medium");
    }

    #[test]
    fn test_hot_path_analysis_with_few_functions() {
        let mut profiler = FunctionProfiler::new();

        // Only 3 functions (less than 10)
        profiler.record("func_a", "write", 3000000, None);
        profiler.record("func_b", "read", 2000000, None);
        profiler.record("func_c", "open", 1000000, None);

        // Should handle fewer than 10 functions gracefully
        profiler.print_summary();

        assert_eq!(profiler.stats.len(), 3);
    }

    #[test]
    fn test_hot_path_analysis_with_call_graph() {
        let mut profiler = FunctionProfiler::new();

        // Hot function with callees
        profiler.record("hot_main", "write", 5000000, None);
        profiler.record("helper_a", "read", 1000000, Some("hot_main"));
        profiler.record("helper_b", "open", 500000, Some("hot_main"));
        profiler.record("helper_c", "close", 250000, Some("hot_main"));

        // Should show call graph for hot functions
        profiler.print_summary();

        let hot_main_stats = profiler.stats.get("hot_main").expect("test");
        assert_eq!(hot_main_stats.callees.len(), 3);
        assert_eq!(hot_main_stats.total_time_us, 5000000);
    }

    #[test]
    fn test_hot_path_analysis_percentage_calculation() {
        let mut profiler = FunctionProfiler::new();

        // Total: 10 seconds
        profiler.record("func_50", "write", 5000000, None); // 50% of total
        profiler.record("func_30", "read", 3000000, None); // 30% of total
        profiler.record("func_20", "open", 2000000, None); // 20% of total

        let total: u64 = profiler.stats.values().map(|s| s.total_time_us).sum();
        assert_eq!(total, 10000000);

        // Verify percentages would be calculated correctly
        let func_50_stats = profiler.stats.get("func_50").expect("test");
        let percent_50 = (func_50_stats.total_time_us as f64 / total as f64) * 100.0;
        assert!((percent_50 - 50.0).abs() < 0.01);

        profiler.print_summary();
    }

    #[test]
    fn test_hot_path_analysis_more_than_10_functions() {
        let mut profiler = FunctionProfiler::new();

        // Create 15 functions to test "top 10" limit
        for i in 0..15 {
            let time = (15 - i) * 100000; // Descending times
            profiler.record(&format!("func_{}", i), "write", time, None);
        }

        assert_eq!(profiler.stats.len(), 15);

        // print_summary() should only show top 10 in hot path analysis
        profiler.print_summary();
    }

    #[test]
    fn test_flamegraph_export_simple() {
        let mut profiler = FunctionProfiler::new();
        profiler.record("main", "write", 1000, None);

        let mut output = Vec::new();
        profiler.export_flamegraph(&mut output).expect("test");

        let flamegraph = String::from_utf8(output).expect("test");
        assert!(flamegraph.contains("main 1"));
    }

    #[test]
    fn test_flamegraph_export_with_call_graph() {
        let mut profiler = FunctionProfiler::new();

        // main calls helper
        profiler.record("main", "write", 1000, None);
        profiler.record("helper", "read", 2000, Some("main"));

        let mut output = Vec::new();
        profiler.export_flamegraph(&mut output).expect("test");

        let flamegraph = String::from_utf8(output).expect("test");

        // Should have root function
        assert!(flamegraph.contains("main 1"));
        // Should have caller->callee
        assert!(flamegraph.contains("main;helper 1"));
    }

    #[test]
    fn test_flamegraph_export_multiple_callees() {
        let mut profiler = FunctionProfiler::new();

        // main calls multiple functions
        profiler.record("main", "write", 1000, None);
        profiler.record("helper_a", "read", 2000, Some("main"));
        profiler.record("helper_b", "open", 3000, Some("main"));
        profiler.record("helper_c", "close", 4000, Some("main"));

        let mut output = Vec::new();
        profiler.export_flamegraph(&mut output).expect("test");

        let flamegraph = String::from_utf8(output).expect("test");

        // Should have all call paths
        assert!(flamegraph.contains("main;helper_a 1"));
        assert!(flamegraph.contains("main;helper_b 1"));
        assert!(flamegraph.contains("main;helper_c 1"));
    }

    #[test]
    fn test_flamegraph_export_nested_calls() {
        let mut profiler = FunctionProfiler::new();

        // main -> helper_a -> helper_b
        profiler.record("main", "write", 1000, None);
        profiler.record("helper_a", "read", 2000, Some("main"));
        profiler.record("helper_b", "open", 3000, Some("helper_a"));

        let mut output = Vec::new();
        profiler.export_flamegraph(&mut output).expect("test");

        let flamegraph = String::from_utf8(output).expect("test");

        // Should have main as root
        assert!(flamegraph.contains("main 1"));
        // Should have first level
        assert!(flamegraph.contains("main;helper_a 1"));
        // Should have second level
        assert!(flamegraph.contains("helper_a;helper_b 1"));
    }

    #[test]
    fn test_flamegraph_export_multiple_calls_same_function() {
        let mut profiler = FunctionProfiler::new();

        // main calls helper multiple times
        profiler.record("main", "write", 1000, None);
        profiler.record("helper", "read", 1000, Some("main"));
        profiler.record("helper", "read", 2000, Some("main"));
        profiler.record("helper", "open", 3000, Some("main"));

        let mut output = Vec::new();
        profiler.export_flamegraph(&mut output).expect("test");

        let flamegraph = String::from_utf8(output).expect("test");

        // Should show aggregated call count
        assert!(flamegraph.contains("main;helper 3"));
    }

    #[test]
    fn test_flamegraph_export_empty() {
        let profiler = FunctionProfiler::new();

        let mut output = Vec::new();
        profiler.export_flamegraph(&mut output).expect("test");

        let flamegraph = String::from_utf8(output).expect("test");
        assert!(flamegraph.is_empty());
    }

    #[test]
    fn test_flamegraph_export_multiple_roots() {
        let mut profiler = FunctionProfiler::new();

        // Multiple root functions (no callers)
        profiler.record("main", "write", 1000, None);
        profiler.record("worker_thread", "read", 2000, None);
        profiler.record("signal_handler", "open", 3000, None);

        let mut output = Vec::new();
        profiler.export_flamegraph(&mut output).expect("test");

        let flamegraph = String::from_utf8(output).expect("test");

        // All should appear as roots
        assert!(flamegraph.contains("main 1"));
        assert!(flamegraph.contains("worker_thread 1"));
        assert!(flamegraph.contains("signal_handler 1"));
    }

    #[test]
    fn test_has_caller_helper() {
        let mut profiler = FunctionProfiler::new();

        profiler.record("main", "write", 1000, None);
        profiler.record("helper", "read", 2000, Some("main"));

        // main has no caller
        assert!(!profiler.has_caller("main"));
        // helper has caller (main)
        assert!(profiler.has_caller("helper"));
    }

    #[test]
    fn test_flamegraph_format_correctness() {
        let mut profiler = FunctionProfiler::new();

        profiler.record("main", "write", 1000, None);
        profiler.record("helper", "read", 2000, Some("main"));

        let mut output = Vec::new();
        profiler.export_flamegraph(&mut output).expect("test");

        let flamegraph = String::from_utf8(output).expect("test");
        let lines: Vec<&str> = flamegraph.lines().collect();

        // Each line should follow "stack count" format
        for line in lines {
            let parts: Vec<&str> = line.split_whitespace().collect();
            assert!(parts.len() >= 2, "Line should have stack and count: {}", line);

            // Last part should be a number (count)
            let count_str = parts.last().expect("test");
            assert!(count_str.parse::<u64>().is_ok(), "Count should be a number: {}", count_str);
        }
    }
}
