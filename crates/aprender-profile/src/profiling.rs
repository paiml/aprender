//! Self-profiling infrastructure for Renacer
//!
//! GitHub Issue #3: Built-in profiling and performance analysis
//! Sprint 13-14: Internal timing infrastructure
//!
//! This module provides instrumentation to measure Renacer's own performance
//! overhead when tracing programs. It tracks time spent in different operations
//! to help identify optimization opportunities.

use std::time::{Duration, Instant};

/// Categories of operations that can be profiled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProfilingCategory {
    /// Time spent in ptrace system calls (getregs, setregs, etc)
    Ptrace,
    /// Time spent formatting syscall output strings
    Formatting,
    /// Time spent reading memory from traced process (filenames, etc)
    MemoryRead,
    /// Time spent in DWARF debug info lookups
    DwarfLookup,
    /// Time spent in statistics tracking
    Statistics,
    /// Time spent in JSON serialization
    JsonSerialization,
    /// Other miscellaneous operations
    Other,
}

/// Profiling context that tracks time spent in various operations
#[derive(Debug, Default)]
pub struct ProfilingContext {
    /// Total number of syscalls traced
    syscall_count: u64,
    /// Time spent in ptrace operations
    ptrace_time: Duration,
    /// Time spent formatting output
    formatting_time: Duration,
    /// Time spent reading memory
    memory_read_time: Duration,
    /// Time spent in DWARF lookups
    dwarf_time: Duration,
    /// Time spent in statistics
    stats_time: Duration,
    /// Time spent in JSON serialization
    json_time: Duration,
    /// Time spent in other operations
    other_time: Duration,
    /// Total wall clock time
    start_time: Option<Instant>,
}

impl ProfilingContext {
    /// Create a new profiling context
    pub fn new() -> Self {
        Self { start_time: Some(Instant::now()), ..Default::default() }
    }

    /// Record that a syscall was traced
    pub fn record_syscall(&mut self) {
        self.syscall_count += 1;
    }

    /// Measure the time taken by an operation
    ///
    /// # Example
    /// ```
    /// use renacer::profiling::{ProfilingContext, ProfilingCategory};
    ///
    /// let mut ctx = ProfilingContext::new();
    /// let result = ctx.measure(ProfilingCategory::Formatting, || {
    ///     format!("test")
    /// });
    /// assert_eq!(result, "test");
    /// ```
    pub fn measure<F, R>(&mut self, category: ProfilingCategory, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        self.record_time(category, elapsed);
        result
    }

    /// Record time spent in a category
    pub fn record_time(&mut self, category: ProfilingCategory, duration: Duration) {
        match category {
            ProfilingCategory::Ptrace => self.ptrace_time += duration,
            ProfilingCategory::Formatting => self.formatting_time += duration,
            ProfilingCategory::MemoryRead => self.memory_read_time += duration,
            ProfilingCategory::DwarfLookup => self.dwarf_time += duration,
            ProfilingCategory::Statistics => self.stats_time += duration,
            ProfilingCategory::JsonSerialization => self.json_time += duration,
            ProfilingCategory::Other => self.other_time += duration,
        }
    }

    /// Get the total wall clock time since profiling started
    pub fn wall_time(&self) -> Duration {
        self.start_time.map(|start| start.elapsed()).unwrap_or_default()
    }

    /// Get the total number of syscalls traced
    pub fn syscall_count(&self) -> u64 {
        self.syscall_count
    }

    /// Get time spent in a specific category
    pub fn time_in_category(&self, category: ProfilingCategory) -> Duration {
        match category {
            ProfilingCategory::Ptrace => self.ptrace_time,
            ProfilingCategory::Formatting => self.formatting_time,
            ProfilingCategory::MemoryRead => self.memory_read_time,
            ProfilingCategory::DwarfLookup => self.dwarf_time,
            ProfilingCategory::Statistics => self.stats_time,
            ProfilingCategory::JsonSerialization => self.json_time,
            ProfilingCategory::Other => self.other_time,
        }
    }

    /// Get total user time (sum of all categories)
    pub fn user_time(&self) -> Duration {
        self.ptrace_time
            + self.formatting_time
            + self.memory_read_time
            + self.dwarf_time
            + self.stats_time
            + self.json_time
            + self.other_time
    }

    /// Print profiling summary to stderr
    pub fn print_summary(&self) {
        let wall = self.wall_time();
        let user = self.user_time();
        let kernel = wall.saturating_sub(user);

        eprintln!("\n╔════════════════════════════════════════════════════════════╗");
        eprintln!("║  Renacer Self-Profiling Results                           ║");
        eprintln!("╚════════════════════════════════════════════════════════════╝");
        eprintln!();
        eprintln!("Total syscalls traced:     {}", self.syscall_count);
        eprintln!("Total wall time:           {:.3}s", wall.as_secs_f64());
        eprintln!(
            "  - Kernel time (ptrace):  {:.3}s ({:.1}%)",
            kernel.as_secs_f64(),
            kernel.as_secs_f64() / wall.as_secs_f64() * 100.0
        );
        eprintln!(
            "  - User time (renacer):   {:.3}s ({:.1}%)",
            user.as_secs_f64(),
            user.as_secs_f64() / wall.as_secs_f64() * 100.0
        );
        eprintln!();
        eprintln!("User-space breakdown:");
        self.print_category("Ptrace ops", self.ptrace_time, user);
        self.print_category("String formatting", self.formatting_time, user);
        self.print_category("Memory reads", self.memory_read_time, user);
        self.print_category("DWARF lookups", self.dwarf_time, user);
        self.print_category("Statistics", self.stats_time, user);
        self.print_category("JSON output", self.json_time, user);
        self.print_category("Other", self.other_time, user);
        eprintln!();
    }

    fn print_category(&self, name: &str, time: Duration, total: Duration) {
        if time > Duration::ZERO {
            eprintln!(
                "  - {:20} {:.3}s ({:.1}%)",
                format!("{}:", name),
                time.as_secs_f64(),
                time.as_secs_f64() / total.as_secs_f64() * 100.0
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiling_context_new() {
        let ctx = ProfilingContext::new();
        assert_eq!(ctx.syscall_count(), 0);
        assert!(ctx.start_time.is_some());
    }

    #[test]
    fn test_record_syscall() {
        let mut ctx = ProfilingContext::new();
        assert_eq!(ctx.syscall_count(), 0);

        ctx.record_syscall();
        assert_eq!(ctx.syscall_count(), 1);

        ctx.record_syscall();
        ctx.record_syscall();
        assert_eq!(ctx.syscall_count(), 3);
    }

    #[test]
    fn test_record_time() {
        let mut ctx = ProfilingContext::new();
        let duration = Duration::from_millis(100);

        ctx.record_time(ProfilingCategory::Formatting, duration);
        assert_eq!(ctx.time_in_category(ProfilingCategory::Formatting), duration);

        ctx.record_time(ProfilingCategory::Formatting, duration);
        assert_eq!(ctx.time_in_category(ProfilingCategory::Formatting), duration + duration);
    }

    #[test]
    fn test_measure() {
        let mut ctx = ProfilingContext::new();

        let result = ctx.measure(ProfilingCategory::Formatting, || {
            thread::sleep(Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        let formatting_time = ctx.time_in_category(ProfilingCategory::Formatting);
        assert!(formatting_time >= Duration::from_millis(10));
        assert!(formatting_time < Duration::from_millis(50)); // Allow some slack
    }

    #[test]
    fn test_user_time_sum() {
        let mut ctx = ProfilingContext::new();

        ctx.record_time(ProfilingCategory::Ptrace, Duration::from_millis(10));
        ctx.record_time(ProfilingCategory::Formatting, Duration::from_millis(20));
        ctx.record_time(ProfilingCategory::MemoryRead, Duration::from_millis(30));

        assert_eq!(ctx.user_time(), Duration::from_millis(60));
    }

    #[test]
    fn test_wall_time() {
        let ctx = ProfilingContext::new();
        thread::sleep(Duration::from_millis(10));

        let wall = ctx.wall_time();
        assert!(wall >= Duration::from_millis(10));
        assert!(wall < Duration::from_millis(100));
    }

    #[test]
    fn test_all_categories() {
        let mut ctx = ProfilingContext::new();
        let duration = Duration::from_millis(5);

        ctx.record_time(ProfilingCategory::Ptrace, duration);
        ctx.record_time(ProfilingCategory::Formatting, duration);
        ctx.record_time(ProfilingCategory::MemoryRead, duration);
        ctx.record_time(ProfilingCategory::DwarfLookup, duration);
        ctx.record_time(ProfilingCategory::Statistics, duration);
        ctx.record_time(ProfilingCategory::JsonSerialization, duration);
        ctx.record_time(ProfilingCategory::Other, duration);

        assert_eq!(ctx.user_time(), Duration::from_millis(35));
    }

    #[test]
    fn test_print_summary_does_not_panic() {
        let mut ctx = ProfilingContext::new();
        ctx.record_syscall();
        ctx.record_time(ProfilingCategory::Formatting, Duration::from_millis(10));

        // Should not panic
        ctx.print_summary();
    }

    #[test]
    fn test_profiling_category_equality() {
        assert_eq!(ProfilingCategory::Ptrace, ProfilingCategory::Ptrace);
        assert_ne!(ProfilingCategory::Ptrace, ProfilingCategory::Formatting);
    }
}
