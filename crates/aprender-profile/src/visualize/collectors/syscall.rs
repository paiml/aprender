//! Syscall collector for real-time syscall metrics
//!
//! Receives syscall events from the tracer and aggregates them
//! into per-syscall and per-category statistics for visualization.

use super::{Collector, MetricValue, Metrics};
use crate::visualize::app::SyscallCategory;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

/// Syscall event received from tracer
#[derive(Debug, Clone)]
pub struct SyscallEvent {
    /// Syscall name (e.g., "read", "write")
    pub name: String,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Return value (negative for errors)
    pub result: i64,
    /// Process ID
    pub pid: i32,
    /// Source file (from DWARF)
    pub source_file: Option<String>,
    /// Source line (from DWARF)
    pub source_line: Option<u32>,
}

impl SyscallEvent {
    /// Create new syscall event
    pub fn new(name: &str, duration_us: u64) -> Self {
        Self {
            name: name.to_string(),
            duration_us,
            result: 0,
            pid: 0,
            source_file: None,
            source_line: None,
        }
    }

    /// Set result value
    pub fn with_result(mut self, result: i64) -> Self {
        self.result = result;
        self
    }

    /// Set process ID
    pub fn with_pid(mut self, pid: i32) -> Self {
        self.pid = pid;
        self
    }

    /// Set source location
    pub fn with_source(mut self, file: &str, line: u32) -> Self {
        self.source_file = Some(file.to_string());
        self.source_line = Some(line);
        self
    }

    /// Check if syscall resulted in error
    pub fn is_error(&self) -> bool {
        self.result < 0
    }

    /// Get syscall category
    pub fn category(&self) -> SyscallCategory {
        SyscallCategory::from_name(&self.name)
    }
}

/// Per-syscall statistics
#[derive(Debug, Clone, Default)]
pub struct SyscallStats {
    /// Total call count in current window
    pub count: u64,
    /// Error count in current window
    pub errors: u64,
    /// Sum of durations (for average)
    pub duration_sum: u64,
    /// Min duration
    pub duration_min: u64,
    /// Max duration
    pub duration_max: u64,
    /// Last update time
    pub last_update: Option<Instant>,
}

impl SyscallStats {
    /// Update stats with new event
    pub fn update(&mut self, event: &SyscallEvent) {
        self.count += 1;
        if event.is_error() {
            self.errors += 1;
        }
        self.duration_sum += event.duration_us;
        self.duration_min = self.duration_min.min(event.duration_us);
        self.duration_max = self.duration_max.max(event.duration_us);
        self.last_update = Some(Instant::now());
    }

    /// Calculate average duration
    pub fn avg_duration(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.duration_sum as f64 / self.count as f64
        }
    }

    /// Calculate rate (calls per second) given time window
    pub fn rate(&self, window: Duration) -> f64 {
        if window.as_secs_f64() == 0.0 {
            0.0
        } else {
            self.count as f64 / window.as_secs_f64()
        }
    }

    /// Reset stats for new window
    pub fn reset(&mut self) {
        self.count = 0;
        self.errors = 0;
        self.duration_sum = 0;
        self.duration_min = u64::MAX;
        self.duration_max = 0;
    }
}

/// Syscall collector for visualization
pub struct SyscallCollector {
    /// Receiver for syscall events from tracer
    rx: Option<Receiver<SyscallEvent>>,
    /// Per-syscall statistics
    stats: HashMap<String, SyscallStats>,
    /// Per-category statistics
    category_stats: HashMap<SyscallCategory, SyscallStats>,
    /// Collection window start time
    window_start: Instant,
    /// Collection _interval (reserved for future rate limiting)
    _interval: Duration,
    /// Total syscalls in current window
    total_count: u64,
    /// Total errors in current window
    total_errors: u64,
    /// Whether collector is running
    available: bool,
}

impl SyscallCollector {
    /// Create new syscall collector with event receiver
    pub fn new(rx: Receiver<SyscallEvent>) -> Self {
        Self {
            rx: Some(rx),
            stats: HashMap::new(),
            category_stats: HashMap::new(),
            window_start: Instant::now(),
            _interval: Duration::from_secs(1),
            total_count: 0,
            total_errors: 0,
            available: true,
        }
    }

    /// Create mock collector for testing (no receiver)
    pub fn mock() -> Self {
        Self {
            rx: None,
            stats: HashMap::new(),
            category_stats: HashMap::new(),
            window_start: Instant::now(),
            _interval: Duration::from_secs(1),
            total_count: 0,
            total_errors: 0,
            available: true,
        }
    }

    /// Inject a syscall event (for testing)
    pub fn inject(&mut self, event: SyscallEvent) {
        self.process_event(&event);
    }

    /// Process a single syscall event
    fn process_event(&mut self, event: &SyscallEvent) {
        // Update per-syscall stats
        self.stats.entry(event.name.clone()).or_default().update(event);

        // Update per-category stats
        self.category_stats.entry(event.category()).or_default().update(event);

        // Update totals
        self.total_count += 1;
        if event.is_error() {
            self.total_errors += 1;
        }
    }

    /// Drain pending events from receiver
    fn drain_events(&mut self) {
        // Collect events first to avoid borrow conflict
        let events: Vec<SyscallEvent> = if let Some(ref rx) = self.rx {
            let mut events = Vec::new();
            loop {
                match rx.try_recv() {
                    Ok(event) => events.push(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.available = false;
                        break;
                    }
                }
            }
            events
        } else {
            Vec::new()
        };

        // Process collected events
        for event in events {
            self.process_event(&event);
        }
    }

    /// Get current stats for a syscall
    pub fn get_stats(&self, name: &str) -> Option<&SyscallStats> {
        self.stats.get(name)
    }

    /// Get current stats for a category
    pub fn get_category_stats(&self, category: &SyscallCategory) -> Option<&SyscallStats> {
        self.category_stats.get(category)
    }

    /// Get all syscall names sorted by count (descending)
    pub fn top_syscalls(&self, limit: usize) -> Vec<(&str, u64)> {
        let mut syscalls: Vec<_> =
            self.stats.iter().map(|(name, stats)| (name.as_str(), stats.count)).collect();
        syscalls.sort_by(|a, b| b.1.cmp(&a.1));
        syscalls.truncate(limit);
        syscalls
    }

    /// Get total syscall rate
    pub fn total_rate(&self) -> f64 {
        let elapsed = self.window_start.elapsed();
        if elapsed.as_secs_f64() == 0.0 {
            0.0
        } else {
            self.total_count as f64 / elapsed.as_secs_f64()
        }
    }
}

impl Collector for SyscallCollector {
    fn collect(&mut self) -> Result<Metrics> {
        // Drain any pending events
        self.drain_events();

        let elapsed = self.window_start.elapsed();
        let mut values = HashMap::new();

        // Add total metrics
        values.insert("syscall.total.rate".to_string(), MetricValue::Rate(self.total_rate()));
        values.insert("syscall.total.count".to_string(), MetricValue::Counter(self.total_count));
        values.insert("syscall.total.errors".to_string(), MetricValue::Counter(self.total_errors));

        // Add per-category rates
        for (category, stats) in &self.category_stats {
            let key = format!("syscall.category.{}.rate", category.name());
            values.insert(key, MetricValue::Rate(stats.rate(elapsed)));
        }

        // Add top syscall rates
        for (name, stats) in &self.stats {
            let key = format!("syscall.{}.rate", name);
            values.insert(key, MetricValue::Rate(stats.rate(elapsed)));

            let key = format!("syscall.{}.avg_duration", name);
            values.insert(key, MetricValue::Gauge(stats.avg_duration()));
        }

        Ok(Metrics::new(values))
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn name(&self) -> &'static str {
        "syscall"
    }

    fn reset(&mut self) {
        self.stats.clear();
        self.category_stats.clear();
        self.window_start = Instant::now();
        self.total_count = 0;
        self.total_errors = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syscall_event_new() {
        let event = SyscallEvent::new("read", 100);
        assert_eq!(event.name, "read");
        assert_eq!(event.duration_us, 100);
        assert_eq!(event.result, 0);
        assert!(!event.is_error());
    }

    #[test]
    fn test_syscall_event_with_error() {
        let event = SyscallEvent::new("open", 50).with_result(-1);
        assert!(event.is_error());
    }

    #[test]
    fn test_syscall_event_with_source() {
        let event = SyscallEvent::new("write", 200).with_source("main.c", 42);
        assert_eq!(event.source_file, Some("main.c".to_string()));
        assert_eq!(event.source_line, Some(42));
    }

    #[test]
    fn test_syscall_event_category() {
        assert_eq!(SyscallEvent::new("read", 0).category(), SyscallCategory::File);
        assert_eq!(SyscallEvent::new("socket", 0).category(), SyscallCategory::Network);
        assert_eq!(SyscallEvent::new("mmap", 0).category(), SyscallCategory::Memory);
        assert_eq!(SyscallEvent::new("fork", 0).category(), SyscallCategory::Process);
    }

    #[test]
    fn test_syscall_stats_update() {
        let mut stats = SyscallStats::default();
        stats.duration_min = u64::MAX;

        let event1 = SyscallEvent::new("read", 100);
        let event2 = SyscallEvent::new("read", 200).with_result(-1);

        stats.update(&event1);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.duration_sum, 100);

        stats.update(&event2);
        assert_eq!(stats.count, 2);
        assert_eq!(stats.errors, 1);
        assert_eq!(stats.duration_sum, 300);
        assert!((stats.avg_duration() - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_syscall_stats_reset() {
        let mut stats = SyscallStats::default();
        stats.count = 100;
        stats.errors = 10;

        stats.reset();
        assert_eq!(stats.count, 0);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_syscall_collector_mock() {
        let mut collector = SyscallCollector::mock();
        assert!(collector.is_available());
        assert_eq!(collector.name(), "syscall");

        collector.inject(SyscallEvent::new("read", 100));
        collector.inject(SyscallEvent::new("write", 200));
        collector.inject(SyscallEvent::new("read", 150));

        let stats = collector.get_stats("read").unwrap();
        assert_eq!(stats.count, 2);

        let top = collector.top_syscalls(10);
        assert_eq!(top[0].0, "read");
        assert_eq!(top[0].1, 2);
    }

    #[test]
    fn test_syscall_collector_collect() {
        let mut collector = SyscallCollector::mock();
        collector.inject(SyscallEvent::new("read", 100));

        let metrics = collector.collect().unwrap();
        assert!(metrics.values.contains_key("syscall.total.count"));
    }

    #[test]
    fn test_syscall_collector_category_stats() {
        let mut collector = SyscallCollector::mock();
        collector.inject(SyscallEvent::new("read", 100));
        collector.inject(SyscallEvent::new("open", 50));
        collector.inject(SyscallEvent::new("socket", 200));

        let file_stats = collector.get_category_stats(&SyscallCategory::File).unwrap();
        assert_eq!(file_stats.count, 2); // read + open

        let net_stats = collector.get_category_stats(&SyscallCategory::Network).unwrap();
        assert_eq!(net_stats.count, 1);
    }

    #[test]
    fn test_syscall_collector_reset() {
        let mut collector = SyscallCollector::mock();
        collector.inject(SyscallEvent::new("read", 100));
        assert_eq!(collector.total_count, 1);

        collector.reset();
        assert_eq!(collector.total_count, 0);
        assert!(collector.stats.is_empty());
    }
}
