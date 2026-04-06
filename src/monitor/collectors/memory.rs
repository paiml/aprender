//! Memory metrics collector.
//!
//! Parses `/proc/meminfo` on Linux to collect memory utilization metrics.

use crate::monitor::error::{MonitorError, Result};
use crate::monitor::ring_buffer::RingBuffer;
#[cfg(target_os = "macos")]
use crate::monitor::subprocess::run_with_timeout_stdout;
use crate::monitor::types::{Collector, MetricValue, Metrics};
use std::time::Duration;

/// Collector for memory metrics.
#[derive(Debug)]
pub struct MemoryCollector {
    /// History of memory usage percentage.
    history: RingBuffer<f64>,
}

impl MemoryCollector {
    /// Creates a new memory collector.
    #[must_use]
    pub fn new() -> Self {
        Self { history: RingBuffer::new(300) }
    }

    /// Parses /proc/meminfo on Linux.
    #[cfg(target_os = "linux")]
    fn parse_meminfo() -> Result<Metrics> {
        let content = std::fs::read_to_string("/proc/meminfo").map_err(|e| {
            MonitorError::CollectionFailed {
                collector: "memory",
                message: format!("Failed to read /proc/meminfo: {e}"),
            }
        })?;

        let mut metrics = Metrics::new();
        let mut total: u64 = 0;
        let mut free: u64 = 0;
        let mut available: u64 = 0;
        let mut buffers: u64 = 0;
        let mut cached: u64 = 0;
        let mut swap_total: u64 = 0;
        let mut swap_free: u64 = 0;

        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            let value: u64 = parts[1].parse().unwrap_or(0) * 1024; // Convert from KB to bytes

            match parts[0] {
                "MemTotal:" => total = value,
                "MemFree:" => free = value,
                "MemAvailable:" => available = value,
                "Buffers:" => buffers = value,
                "Cached:" => cached = value,
                "SwapTotal:" => swap_total = value,
                "SwapFree:" => swap_free = value,
                _ => {}
            }
        }

        let used = total.saturating_sub(available);
        let swap_used = swap_total.saturating_sub(swap_free);

        metrics.insert("memory.total", MetricValue::Counter(total));
        metrics.insert("memory.free", MetricValue::Counter(free));
        metrics.insert("memory.available", MetricValue::Counter(available));
        metrics.insert("memory.used", MetricValue::Counter(used));
        metrics.insert("memory.buffers", MetricValue::Counter(buffers));
        metrics.insert("memory.cached", MetricValue::Counter(cached));
        metrics.insert("memory.swap.total", MetricValue::Counter(swap_total));
        metrics.insert("memory.swap.free", MetricValue::Counter(swap_free));
        metrics.insert("memory.swap.used", MetricValue::Counter(swap_used));

        // Calculate percentages
        if total > 0 {
            let used_percent = (used as f64 / total as f64) * 100.0;
            metrics.insert("memory.used.percent", used_percent);
        }

        if swap_total > 0 {
            let swap_percent = (swap_used as f64 / swap_total as f64) * 100.0;
            metrics.insert("memory.swap.percent", swap_percent);
        }

        Ok(metrics)
    }

    #[cfg(target_os = "macos")]
    fn parse_meminfo() -> Result<Metrics> {
        let mut metrics = Metrics::new();

        // Get total memory from sysctl (fast, 1s timeout)
        let total: u64 =
            run_with_timeout_stdout("sysctl", &["-n", "hw.memsize"], Duration::from_secs(1))
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);

        // Parse vm_stat for memory breakdown (fast, 2s timeout)
        let content =
            run_with_timeout_stdout("vm_stat", &[], Duration::from_secs(2)).ok_or_else(|| {
                MonitorError::CollectionFailed {
                    collector: "memory",
                    message: "vm_stat timed out or failed".to_string(),
                }
            })?;

        // Parse page size (first line: "Mach Virtual Memory Statistics: (page size of XXXX bytes)")
        let page_size: u64 = content
            .lines()
            .next()
            .and_then(|line| {
                line.split("page size of ")
                    .nth(1)
                    .and_then(|s| s.split(' ').next())
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(4096);

        let mut pages_free: u64 = 0;
        let mut pages_active: u64 = 0;
        let mut pages_inactive: u64 = 0;
        let mut pages_speculative: u64 = 0;
        let mut pages_wired: u64 = 0;
        let mut pages_compressed: u64 = 0;
        let mut pages_purgeable: u64 = 0;

        for line in content.lines() {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() != 2 {
                continue;
            }

            let value: u64 = parts[1].trim().trim_end_matches('.').parse().unwrap_or(0);

            match parts[0].trim() {
                "Pages free" => pages_free = value,
                "Pages active" => pages_active = value,
                "Pages inactive" => pages_inactive = value,
                "Pages speculative" => pages_speculative = value,
                "Pages wired down" => pages_wired = value,
                "Pages occupied by compressor" => pages_compressed = value,
                "Pages purgeable" => pages_purgeable = value,
                _ => {}
            }
        }

        let free = (pages_free + pages_speculative) * page_size;
        let cached = pages_purgeable * page_size;
        let active = pages_active * page_size;
        let inactive = pages_inactive * page_size;
        let wired = pages_wired * page_size;
        let compressed = pages_compressed * page_size;

        // Available = free + inactive (can be reclaimed)
        let available = free + inactive + cached;
        let used = total.saturating_sub(available);

        metrics.insert("memory.total", MetricValue::Counter(total));
        metrics.insert("memory.free", MetricValue::Counter(free));
        metrics.insert("memory.available", MetricValue::Counter(available));
        metrics.insert("memory.used", MetricValue::Counter(used));
        metrics.insert("memory.cached", MetricValue::Counter(cached));
        metrics.insert("memory.buffers", MetricValue::Counter(0)); // Not applicable on macOS
        metrics.insert("memory.active", MetricValue::Counter(active));
        metrics.insert("memory.inactive", MetricValue::Counter(inactive));
        metrics.insert("memory.wired", MetricValue::Counter(wired));
        metrics.insert("memory.compressed", MetricValue::Counter(compressed));

        // Swap info from sysctl (fast, 1s timeout)
        let swap_output =
            run_with_timeout_stdout("sysctl", &["-n", "vm.swapusage"], Duration::from_secs(1));

        let (swap_total, swap_used, swap_free) = swap_output
            .map(|content| {
                // Format: "total = 2048.00M  used = 1024.00M  free = 1024.00M  ..."
                let mut total: u64 = 0;
                let mut used: u64 = 0;
                let mut free: u64 = 0;

                for part in content.split_whitespace() {
                    if part.ends_with('M') {
                        let val = part.trim_end_matches('M').parse::<f64>().unwrap_or(0.0);
                        let bytes = (val * 1024.0 * 1024.0) as u64;
                        if total == 0 {
                            total = bytes;
                        } else if used == 0 {
                            used = bytes;
                        } else if free == 0 {
                            free = bytes;
                            break;
                        }
                    } else if part.ends_with('G') {
                        let val = part.trim_end_matches('G').parse::<f64>().unwrap_or(0.0);
                        let bytes = (val * 1024.0 * 1024.0 * 1024.0) as u64;
                        if total == 0 {
                            total = bytes;
                        } else if used == 0 {
                            used = bytes;
                        } else if free == 0 {
                            free = bytes;
                            break;
                        }
                    }
                }
                (total, used, free)
            })
            .unwrap_or((0, 0, 0));

        metrics.insert("memory.swap.total", MetricValue::Counter(swap_total));
        metrics.insert("memory.swap.used", MetricValue::Counter(swap_used));
        metrics.insert("memory.swap.free", MetricValue::Counter(swap_free));

        // Calculate percentages
        if total > 0 {
            let used_percent = (used as f64 / total as f64) * 100.0;
            metrics.insert("memory.used.percent", used_percent);
        }

        if swap_total > 0 {
            let swap_percent = (swap_used as f64 / swap_total as f64) * 100.0;
            metrics.insert("memory.swap.percent", swap_percent);
        }

        Ok(metrics)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn parse_meminfo() -> Result<Metrics> {
        // Return dummy data on other systems
        let mut metrics = Metrics::new();
        metrics.insert("memory.total", MetricValue::Counter(8 * 1024 * 1024 * 1024));
        metrics.insert("memory.used", MetricValue::Counter(4 * 1024 * 1024 * 1024));
        metrics.insert("memory.used.percent", 50.0);
        Ok(metrics)
    }

    /// Returns the memory usage history.
    #[must_use]
    pub fn history(&self) -> &RingBuffer<f64> {
        &self.history
    }
}

impl Default for MemoryCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for MemoryCollector {
    fn id(&self) -> &'static str {
        "memory"
    }

    fn collect(&mut self) -> Result<Metrics> {
        let metrics = Self::parse_meminfo()?;

        // Update history with normalized usage
        if let Some(percent) = metrics.get_gauge("memory.used.percent") {
            self.history.push(percent / 100.0);
        }

        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/proc/meminfo").exists()
        }
        #[cfg(target_os = "macos")]
        {
            true // macOS uses vm_stat which is always available
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            false
        }
    }

    fn interval_hint(&self) -> Duration {
        Duration::from_millis(1000)
    }

    fn display_name(&self) -> &'static str {
        "Memory"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_collector_new() {
        let collector = MemoryCollector::new();
        assert!(collector.history.is_empty());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_memory_collector_is_available() {
        let collector = MemoryCollector::new();
        assert!(collector.is_available());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_memory_collector_collect() {
        let mut collector = MemoryCollector::new();
        let metrics = collector.collect();

        assert!(metrics.is_ok());
        let m = metrics.expect("operation should succeed");

        assert!(m.get_counter("memory.total").is_some());
        assert!(m.get_gauge("memory.used.percent").is_some());
    }

    #[test]
    fn test_memory_collector_interval() {
        let collector = MemoryCollector::new();
        assert_eq!(collector.interval_hint(), Duration::from_millis(1000));
    }
}
