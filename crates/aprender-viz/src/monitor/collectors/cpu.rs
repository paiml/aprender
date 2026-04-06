//! CPU metrics collector.
//!
//! Parses `/proc/stat` on Linux to collect CPU utilization metrics.
//!
//! ## Metrics Collected
//!
//! - Total and per-core CPU utilization
//! - Load average (1, 5, 15 minute)
//! - CPU frequency per core
//! - System uptime

use crate::monitor::error::{MonitorError, Result};
use crate::monitor::ring_buffer::RingBuffer;
#[cfg(target_os = "macos")]
use crate::monitor::subprocess::run_with_timeout_stdout;
use crate::monitor::types::{Collector, MetricValue, Metrics};
use std::time::Duration;

/// Load average values.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadAverage {
    /// 1-minute load average.
    pub one: f64,
    /// 5-minute load average.
    pub five: f64,
    /// 15-minute load average.
    pub fifteen: f64,
}

/// CPU frequency information.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuFrequency {
    /// Current frequency in MHz.
    pub current_mhz: u64,
    /// Minimum frequency in MHz.
    pub min_mhz: u64,
    /// Maximum frequency in MHz.
    pub max_mhz: u64,
}

/// CPU statistics from /proc/stat.
#[derive(Debug, Clone, Default)]
struct CpuStats {
    user: u64,
    nice: u64,
    system: u64,
    idle: u64,
    iowait: u64,
    irq: u64,
    softirq: u64,
    steal: u64,
}

impl CpuStats {
    /// Total CPU time.
    fn total(&self) -> u64 {
        self.user
            + self.nice
            + self.system
            + self.idle
            + self.iowait
            + self.irq
            + self.softirq
            + self.steal
    }

    /// Idle time (idle + iowait).
    fn idle_time(&self) -> u64 {
        self.idle + self.iowait
    }
}

/// Collector for CPU metrics.
#[derive(Debug)]
pub struct CpuCollector {
    /// Previous stats for delta calculation.
    prev_total: Option<CpuStats>,
    /// Per-core previous stats.
    prev_cores: Vec<CpuStats>,
    /// History of total CPU usage.
    history: RingBuffer<f64>,
    /// Per-core usage history.
    core_history: Vec<RingBuffer<f64>>,
    /// Number of CPU cores.
    core_count: usize,
    /// Latest load average.
    load_average: LoadAverage,
    /// Per-core frequency.
    frequencies: Vec<CpuFrequency>,
    /// System uptime in seconds.
    uptime_secs: f64,
}

impl CpuCollector {
    /// Creates a new CPU collector.
    #[must_use]
    pub fn new() -> Self {
        let core_count = Self::detect_core_count();
        let mut core_history = Vec::with_capacity(core_count);
        for _ in 0..core_count {
            core_history.push(RingBuffer::new(300));
        }

        Self {
            prev_total: None,
            prev_cores: Vec::new(),
            history: RingBuffer::new(300),
            core_history,
            core_count,
            load_average: LoadAverage::default(),
            frequencies: vec![CpuFrequency::default(); core_count],
            uptime_secs: 0.0,
        }
    }

    /// Detects the number of CPU cores.
    fn detect_core_count() -> usize {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/stat")
                .map(|content| {
                    content
                        .lines()
                        .filter(|line| line.starts_with("cpu") && !line.starts_with("cpu "))
                        .count()
                })
                .unwrap_or(1)
        }
        #[cfg(target_os = "macos")]
        {
            run_with_timeout_stdout("sysctl", &["-n", "hw.ncpu"], Duration::from_secs(1))
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(1)
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            1
        }
    }

    /// Parses /proc/stat and returns CPU stats.
    #[cfg(target_os = "linux")]
    fn parse_proc_stat() -> Result<(CpuStats, Vec<CpuStats>)> {
        let content =
            std::fs::read_to_string("/proc/stat").map_err(|e| MonitorError::CollectionFailed {
                collector: "cpu",
                message: format!("Failed to read /proc/stat: {e}"),
            })?;

        let mut total = CpuStats::default();
        let mut cores = Vec::new();

        for line in content.lines() {
            if line.starts_with("cpu ") {
                total = Self::parse_cpu_line(line)?;
            } else if line.starts_with("cpu") {
                cores.push(Self::parse_cpu_line(line)?);
            }
        }

        Ok((total, cores))
    }

    #[cfg(target_os = "macos")]
    fn parse_proc_stat() -> Result<(CpuStats, Vec<CpuStats>)> {
        // Use Mach kernel API for REAL per-core CPU stats (like Linux /proc/stat)
        use std::mem;

        #[repr(C)]
        #[derive(Copy, Clone, Default)]
        struct CpuTickInfo {
            user: u32,
            system: u32,
            idle: u32,
            nice: u32,
        }

        extern "C" {
            fn host_processor_info(
                host: u32,
                flavor: i32,
                out_processor_count: *mut u32,
                out_processor_info: *mut *mut i32,
                out_processor_info_count: *mut u32,
            ) -> i32;
            fn mach_host_self() -> u32;
            fn vm_deallocate(target: u32, address: usize, size: usize) -> i32;
            fn mach_task_self() -> u32;
        }

        const PROCESSOR_CPU_LOAD_INFO: i32 = 2;
        const CPU_STATE_USER: usize = 0;
        const CPU_STATE_SYSTEM: usize = 1;
        const CPU_STATE_IDLE: usize = 2;
        const CPU_STATE_NICE: usize = 3;
        const CPU_STATE_MAX: usize = 4;

        let mut processor_count: u32 = 0;
        let mut processor_info: *mut i32 = std::ptr::null_mut();
        let mut processor_info_count: u32 = 0;

        // SAFETY: Calling Mach kernel API with valid pointers
        let result = unsafe {
            host_processor_info(
                mach_host_self(),
                PROCESSOR_CPU_LOAD_INFO,
                &mut processor_count,
                &mut processor_info,
                &mut processor_info_count,
            )
        };

        if result != 0 || processor_info.is_null() {
            // Fallback to top command if Mach API fails
            return Self::parse_proc_stat_fallback();
        }

        let mut cores = Vec::with_capacity(processor_count as usize);
        let mut total_user: u64 = 0;
        let mut total_system: u64 = 0;
        let mut total_idle: u64 = 0;
        let mut total_nice: u64 = 0;

        // SAFETY: processor_info is valid and has processor_count * CPU_STATE_MAX elements
        unsafe {
            for i in 0..processor_count as usize {
                let base = i * CPU_STATE_MAX;
                let user = *processor_info.add(base + CPU_STATE_USER) as u64;
                let system = *processor_info.add(base + CPU_STATE_SYSTEM) as u64;
                let idle = *processor_info.add(base + CPU_STATE_IDLE) as u64;
                let nice = *processor_info.add(base + CPU_STATE_NICE) as u64;

                cores.push(CpuStats {
                    user,
                    nice,
                    system,
                    idle,
                    iowait: 0,
                    irq: 0,
                    softirq: 0,
                    steal: 0,
                });

                total_user += user;
                total_system += system;
                total_idle += idle;
                total_nice += nice;
            }

            // Deallocate the processor info
            let info_size = (processor_info_count as usize) * mem::size_of::<i32>();
            vm_deallocate(mach_task_self(), processor_info as usize, info_size);
        }

        let total = CpuStats {
            user: total_user,
            nice: total_nice,
            system: total_system,
            idle: total_idle,
            iowait: 0,
            irq: 0,
            softirq: 0,
            steal: 0,
        };

        Ok((total, cores))
    }

    /// Fallback to top command if Mach API fails
    #[cfg(target_os = "macos")]
    fn parse_proc_stat_fallback() -> Result<(CpuStats, Vec<CpuStats>)> {
        // top can be slow, 5s timeout
        let content = run_with_timeout_stdout(
            "top",
            &["-l", "1", "-n", "0", "-s", "0"],
            Duration::from_secs(5),
        )
        .ok_or_else(|| MonitorError::CollectionFailed {
            collector: "cpu",
            message: "top timed out or failed".to_string(),
        })?;
        let mut user: f64 = 0.0;
        let mut sys: f64 = 0.0;
        let mut idle: f64 = 0.0;

        for line in content.lines() {
            if line.starts_with("CPU usage:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                for (i, part) in parts.iter().enumerate() {
                    if *part == "user," || *part == "user" {
                        if let Some(val) = parts.get(i - 1) {
                            user = val.trim_end_matches('%').parse::<f64>().unwrap_or(0.0);
                        }
                    } else if *part == "sys," || *part == "sys" {
                        if let Some(val) = parts.get(i - 1) {
                            sys = val.trim_end_matches('%').parse::<f64>().unwrap_or(0.0);
                        }
                    } else if *part == "idle" {
                        if let Some(val) = parts.get(i - 1) {
                            idle = val.trim_end_matches('%').parse::<f64>().unwrap_or(0.0);
                        }
                    }
                }
                break;
            }
        }

        let total = CpuStats {
            user: (user * 100.0) as u64,
            nice: 0,
            system: (sys * 100.0) as u64,
            idle: (idle * 100.0) as u64,
            iowait: 0,
            irq: 0,
            softirq: 0,
            steal: 0,
        };

        let core_count = Self::detect_core_count();
        let cores = vec![total.clone(); core_count];

        Ok((total, cores))
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn parse_proc_stat() -> Result<(CpuStats, Vec<CpuStats>)> {
        // Return dummy data on other systems
        Ok((CpuStats::default(), vec![CpuStats::default()]))
    }

    /// Parses a single CPU line from /proc/stat.
    fn parse_cpu_line(line: &str) -> Result<CpuStats> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 8 {
            return Err(MonitorError::CollectionFailed {
                collector: "cpu",
                message: "Invalid /proc/stat format".to_string(),
            });
        }

        Ok(CpuStats {
            user: parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0),
            nice: parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0),
            system: parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0),
            idle: parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0),
            iowait: parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(0),
            irq: parts.get(6).and_then(|s| s.parse().ok()).unwrap_or(0),
            softirq: parts.get(7).and_then(|s| s.parse().ok()).unwrap_or(0),
            steal: parts.get(8).and_then(|s| s.parse().ok()).unwrap_or(0),
        })
    }

    /// Calculates CPU percentage from delta.
    fn calculate_percentage(prev: &CpuStats, curr: &CpuStats) -> f64 {
        let total_delta = curr.total().saturating_sub(prev.total());
        let idle_delta = curr.idle_time().saturating_sub(prev.idle_time());

        if total_delta == 0 {
            return 0.0;
        }

        let used_delta = total_delta.saturating_sub(idle_delta);
        (used_delta as f64 / total_delta as f64) * 100.0
    }

    /// Returns the CPU usage history.
    #[must_use]
    pub fn history(&self) -> &RingBuffer<f64> {
        &self.history
    }

    /// Returns the per-core usage history.
    #[must_use]
    pub fn core_history(&self, core: usize) -> Option<&RingBuffer<f64>> {
        self.core_history.get(core)
    }

    /// Returns the number of CPU cores.
    #[must_use]
    pub fn core_count(&self) -> usize {
        self.core_count
    }

    /// Returns the latest load average.
    #[must_use]
    pub fn load_average(&self) -> LoadAverage {
        self.load_average
    }

    /// Returns the per-core frequency information.
    #[must_use]
    pub fn frequencies(&self) -> &[CpuFrequency] {
        &self.frequencies
    }

    /// Returns the system uptime in seconds.
    #[must_use]
    pub fn uptime_secs(&self) -> f64 {
        self.uptime_secs
    }

    /// Reads load average from /proc/loadavg.
    #[cfg(target_os = "linux")]
    fn read_load_average() -> LoadAverage {
        std::fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|content| {
                let parts: Vec<&str> = content.split_whitespace().collect();
                if parts.len() >= 3 {
                    Some(LoadAverage {
                        one: parts[0].parse().unwrap_or(0.0),
                        five: parts[1].parse().unwrap_or(0.0),
                        fifteen: parts[2].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .unwrap_or_default()
    }

    #[cfg(target_os = "macos")]
    fn read_load_average() -> LoadAverage {
        run_with_timeout_stdout("sysctl", &["-n", "vm.loadavg"], Duration::from_secs(1))
            .and_then(|content| {
                // Format: "{ 1.23 4.56 7.89 }"
                let trimmed = content.trim().trim_start_matches('{').trim_end_matches('}');
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 3 {
                    Some(LoadAverage {
                        one: parts[0].parse().unwrap_or(0.0),
                        five: parts[1].parse().unwrap_or(0.0),
                        fifteen: parts[2].parse().unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .unwrap_or_default()
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn read_load_average() -> LoadAverage {
        LoadAverage::default()
    }

    /// Reads CPU frequency for a specific core.
    #[cfg(target_os = "linux")]
    fn read_frequency(core: usize) -> CpuFrequency {
        let base = format!("/sys/devices/system/cpu/cpu{core}/cpufreq");

        let current = std::fs::read_to_string(format!("{base}/scaling_cur_freq"))
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(0)
            / 1000; // Convert from kHz to MHz

        let min = std::fs::read_to_string(format!("{base}/scaling_min_freq"))
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(0)
            / 1000;

        let max = std::fs::read_to_string(format!("{base}/scaling_max_freq"))
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(0)
            / 1000;

        CpuFrequency { current_mhz: current, min_mhz: min, max_mhz: max }
    }

    #[cfg(target_os = "macos")]
    fn read_frequency(_core: usize) -> CpuFrequency {
        // macOS doesn't expose per-core frequency easily, use sysctl for base frequency
        let current =
            run_with_timeout_stdout("sysctl", &["-n", "hw.cpufrequency"], Duration::from_secs(1))
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0)
                / 1_000_000; // Convert Hz to MHz

        CpuFrequency { current_mhz: current, min_mhz: current, max_mhz: current }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn read_frequency(_core: usize) -> CpuFrequency {
        CpuFrequency::default()
    }

    /// Reads system uptime from /proc/uptime.
    #[cfg(target_os = "linux")]
    fn read_uptime() -> f64 {
        std::fs::read_to_string("/proc/uptime")
            .ok()
            .and_then(|content| content.split_whitespace().next().and_then(|s| s.parse().ok()))
            .unwrap_or(0.0)
    }

    #[cfg(target_os = "macos")]
    fn read_uptime() -> f64 {
        run_with_timeout_stdout("sysctl", &["-n", "kern.boottime"], Duration::from_secs(1))
            .and_then(|content| {
                // Format: "{ sec = 1234567890, usec = 123456 } Mon Jan 1 00:00:00 2024"
                content
                    .split("sec = ")
                    .nth(1)
                    .and_then(|s| s.split(',').next())
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .map(|boot_time| {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        (now - boot_time) as f64
                    })
            })
            .unwrap_or(0.0)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn read_uptime() -> f64 {
        0.0
    }
}

impl Default for CpuCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for CpuCollector {
    fn id(&self) -> &'static str {
        "cpu"
    }

    fn collect(&mut self) -> Result<Metrics> {
        let (curr_total, curr_cores) = Self::parse_proc_stat()?;

        let mut metrics = Metrics::new();

        // Calculate total CPU percentage using delta-based calculation
        // Both Linux (/proc/stat) and macOS (Mach kernel) provide cumulative tick counts
        if let Some(prev) = &self.prev_total {
            let percent = Self::calculate_percentage(prev, &curr_total);
            metrics.insert("cpu.total", percent);
            self.history.push(percent / 100.0); // Normalized for graphs
        }

        // Calculate per-core percentages using delta
        for (i, curr) in curr_cores.iter().enumerate() {
            if let Some(prev) = self.prev_cores.get(i) {
                let percent = Self::calculate_percentage(prev, curr);
                metrics.insert(format!("cpu.core.{i}"), percent);

                // Update per-core history
                if let Some(history) = self.core_history.get_mut(i) {
                    history.push(percent / 100.0);
                }
            }
        }

        // Update previous values for next delta calculation
        self.prev_total = Some(curr_total);
        self.prev_cores = curr_cores;

        // Add core count
        metrics.insert("cpu.cores", MetricValue::Counter(self.core_count as u64));

        // Load average
        self.load_average = Self::read_load_average();
        metrics.insert("cpu.load.1", self.load_average.one);
        metrics.insert("cpu.load.5", self.load_average.five);
        metrics.insert("cpu.load.15", self.load_average.fifteen);

        // CPU frequency per core
        for i in 0..self.core_count {
            let freq = Self::read_frequency(i);
            if let Some(f) = self.frequencies.get_mut(i) {
                *f = freq;
            }
            metrics.insert(format!("cpu.freq.{i}"), MetricValue::Counter(freq.current_mhz));
        }

        // Average frequency across all cores
        let avg_freq: u64 = self
            .frequencies
            .iter()
            .map(|f| f.current_mhz)
            .sum::<u64>()
            .checked_div(self.core_count as u64)
            .unwrap_or(0);
        metrics.insert("cpu.freq.avg", MetricValue::Counter(avg_freq));

        // Uptime
        self.uptime_secs = Self::read_uptime();
        metrics.insert("cpu.uptime", self.uptime_secs);

        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/proc/stat").exists()
        }
        #[cfg(target_os = "macos")]
        {
            true // macOS uses sysctl and top, which are always available
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
        "CPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_collector_new() {
        let collector = CpuCollector::new();
        assert!(collector.core_count >= 1);
        assert_eq!(collector.history.len(), 0);
        assert_eq!(collector.core_history.len(), collector.core_count);
    }

    #[test]
    fn test_cpu_stats_total() {
        let stats = CpuStats {
            user: 100,
            nice: 10,
            system: 50,
            idle: 800,
            iowait: 20,
            irq: 5,
            softirq: 5,
            steal: 10,
        };

        assert_eq!(stats.total(), 1000);
        assert_eq!(stats.idle_time(), 820);
    }

    #[test]
    fn test_cpu_stats_default() {
        let stats = CpuStats::default();
        assert_eq!(stats.total(), 0);
        assert_eq!(stats.idle_time(), 0);
    }

    #[test]
    fn test_calculate_percentage() {
        let prev = CpuStats {
            user: 100,
            nice: 0,
            system: 0,
            idle: 900,
            iowait: 0,
            irq: 0,
            softirq: 0,
            steal: 0,
        };

        let curr = CpuStats {
            user: 200,
            nice: 0,
            system: 0,
            idle: 1800,
            iowait: 0,
            irq: 0,
            softirq: 0,
            steal: 0,
        };

        let percent = CpuCollector::calculate_percentage(&prev, &curr);
        // 100 user delta, 1000 total delta = 10%
        assert!((percent - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_calculate_percentage_zero_delta() {
        let prev = CpuStats {
            user: 100,
            nice: 0,
            system: 0,
            idle: 900,
            iowait: 0,
            irq: 0,
            softirq: 0,
            steal: 0,
        };
        // Same as prev - zero delta
        let percent = CpuCollector::calculate_percentage(&prev, &prev);
        assert!((percent - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_calculate_percentage_full_cpu() {
        let prev = CpuStats {
            user: 0,
            nice: 0,
            system: 0,
            idle: 1000,
            iowait: 0,
            irq: 0,
            softirq: 0,
            steal: 0,
        };

        let curr = CpuStats {
            user: 1000,
            nice: 0,
            system: 0,
            idle: 1000, // No change in idle
            iowait: 0,
            irq: 0,
            softirq: 0,
            steal: 0,
        };

        let percent = CpuCollector::calculate_percentage(&prev, &curr);
        assert!((percent - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_cpu_line() {
        let line = "cpu0 100 10 50 800 20 5 5 10 0 0";
        let stats = CpuCollector::parse_cpu_line(line).expect("parsing should succeed");

        assert_eq!(stats.user, 100);
        assert_eq!(stats.nice, 10);
        assert_eq!(stats.system, 50);
        assert_eq!(stats.idle, 800);
        assert_eq!(stats.iowait, 20);
        assert_eq!(stats.irq, 5);
        assert_eq!(stats.softirq, 5);
        assert_eq!(stats.steal, 10);
    }

    #[test]
    fn test_parse_cpu_line_total() {
        let line = "cpu  12345 678 9012 345678 901 23 45 67 0 0";
        let stats = CpuCollector::parse_cpu_line(line).expect("parsing should succeed");

        assert_eq!(stats.user, 12345);
        assert_eq!(stats.nice, 678);
        assert_eq!(stats.system, 9012);
        assert_eq!(stats.idle, 345678);
    }

    #[test]
    fn test_parse_cpu_line_minimal() {
        // Minimal line with just enough fields
        let line = "cpu0 1 2 3 4 5 6 7 8";
        let stats = CpuCollector::parse_cpu_line(line).expect("parsing should succeed");

        assert_eq!(stats.user, 1);
        assert_eq!(stats.nice, 2);
        assert_eq!(stats.system, 3);
        assert_eq!(stats.idle, 4);
    }

    #[test]
    fn test_parse_cpu_line_invalid() {
        let line = "not a cpu line";
        let result = CpuCollector::parse_cpu_line(line);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_cpu_line_too_few_fields() {
        let line = "cpu0 1 2 3"; // Not enough fields
        let result = CpuCollector::parse_cpu_line(line);
        assert!(result.is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_cpu_collector_is_available() {
        let collector = CpuCollector::new();
        assert!(collector.is_available());
    }

    #[test]
    fn test_cpu_collector_interval() {
        let collector = CpuCollector::new();
        assert_eq!(collector.interval_hint(), Duration::from_millis(1000));
    }

    #[test]
    fn test_cpu_collector_display_name() {
        let collector = CpuCollector::new();
        assert_eq!(collector.display_name(), "CPU");
    }

    #[test]
    fn test_load_average_default() {
        let load = LoadAverage::default();
        assert_eq!(load.one, 0.0);
        assert_eq!(load.five, 0.0);
        assert_eq!(load.fifteen, 0.0);
    }

    #[test]
    fn test_cpu_frequency_default() {
        let freq = CpuFrequency::default();
        assert_eq!(freq.current_mhz, 0);
        assert_eq!(freq.min_mhz, 0);
        assert_eq!(freq.max_mhz, 0);
    }

    #[test]
    fn test_cpu_collector_history() {
        let mut collector = CpuCollector::new();
        // First collection initializes prev stats
        let _ = collector.collect();
        // Second collection should produce percentage
        std::thread::sleep(Duration::from_millis(50));
        let _ = collector.collect();

        // History should have at least one entry
        assert!(!collector.history.is_empty());
    }

    #[test]
    fn test_cpu_collector_core_count() {
        let collector = CpuCollector::new();
        assert_eq!(collector.core_count(), collector.core_count);
        assert!(collector.core_count() >= 1);
    }

    #[test]
    fn test_cpu_collector_core_history() {
        let mut collector = CpuCollector::new();
        let _ = collector.collect();
        std::thread::sleep(Duration::from_millis(50));
        let _ = collector.collect();

        // Core 0 should have history
        let core_hist = collector.core_history(0);
        assert!(core_hist.is_some());

        // Invalid core should return None
        let invalid = collector.core_history(9999);
        assert!(invalid.is_none());
    }

    #[test]
    fn test_cpu_stats_all_fields() {
        let stats = CpuStats {
            user: 1000,
            nice: 200,
            system: 300,
            idle: 5000,
            iowait: 100,
            irq: 50,
            softirq: 30,
            steal: 20,
        };

        // Total should be sum of all fields
        assert_eq!(stats.total(), 6700);
        // Idle time is idle + iowait
        assert_eq!(stats.idle_time(), 5100);
    }

    #[test]
    fn test_cpu_collector_load_average() {
        let mut collector = CpuCollector::new();
        let _ = collector.collect();

        let load = collector.load_average();
        // Load average should be non-negative
        assert!(load.one >= 0.0);
        assert!(load.five >= 0.0);
        assert!(load.fifteen >= 0.0);
    }

    #[test]
    fn test_cpu_collector_uptime() {
        let mut collector = CpuCollector::new();
        let _ = collector.collect();

        let uptime = collector.uptime_secs();
        assert!(uptime >= 0.0);
    }

    #[test]
    fn test_cpu_collector_history_accessor() {
        let mut collector = CpuCollector::new();
        let _ = collector.collect();
        std::thread::sleep(Duration::from_millis(50));
        let _ = collector.collect();

        let history = collector.history();
        assert!(!history.is_empty());
    }

    #[test]
    fn test_cpu_collector_frequencies() {
        let mut collector = CpuCollector::new();
        let _ = collector.collect();

        let freqs = collector.frequencies();
        assert_eq!(freqs.len(), collector.core_count());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_proc_stat() {
        let result = CpuCollector::parse_proc_stat();
        assert!(result.is_ok());
        let (total, cores) = result.expect("should parse");
        assert!(total.total() > 0);
        assert!(!cores.is_empty());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_read_load_average() {
        let load = CpuCollector::read_load_average();
        assert!(load.one >= 0.0);
        assert!(load.five >= 0.0);
        assert!(load.fifteen >= 0.0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_read_uptime() {
        let uptime = CpuCollector::read_uptime();
        assert!(uptime > 0.0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_read_frequency() {
        let freq = CpuCollector::read_frequency(0);
        // Just verify we got a valid frequency structure (values are always non-negative for u64)
        let _ = freq.current_mhz;
        let _ = freq.min_mhz;
        let _ = freq.max_mhz;
    }

    #[test]
    fn test_cpu_collector_collect_twice() {
        let mut collector = CpuCollector::new();

        // First collect
        let result1 = collector.collect();
        assert!(result1.is_ok());

        std::thread::sleep(Duration::from_millis(50));

        // Second collect - should calculate percentages
        let result2 = collector.collect();
        assert!(result2.is_ok());

        let metrics = result2.expect("should collect");
        assert!(metrics.get_gauge("cpu.total").is_some());
    }

    #[test]
    fn test_cpu_collector_metrics_structure() {
        let mut collector = CpuCollector::new();
        let _ = collector.collect();
        std::thread::sleep(Duration::from_millis(50));
        let result = collector.collect();

        assert!(result.is_ok());
        let metrics = result.expect("should collect");

        // Check expected metrics exist
        assert!(metrics.get_gauge("cpu.total").is_some());
        assert!(metrics.get_gauge("cpu.load.1").is_some());
        assert!(metrics.get_gauge("cpu.load.5").is_some());
        assert!(metrics.get_gauge("cpu.load.15").is_some());
        assert!(metrics.get_counter("cpu.cores").is_some());
    }

    #[test]
    fn test_calculate_percentage_high_usage() {
        let prev = CpuStats {
            user: 0,
            nice: 0,
            system: 0,
            idle: 1000,
            iowait: 0,
            irq: 0,
            softirq: 0,
            steal: 0,
        };

        let curr = CpuStats {
            user: 800,
            nice: 50,
            system: 100,
            idle: 1050, // Only 50 more idle ticks
            iowait: 0,
            irq: 0,
            softirq: 0,
            steal: 0,
        };

        let percent = CpuCollector::calculate_percentage(&prev, &curr);
        // 950 used out of 1000 delta = 95%
        assert!((percent - 95.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_cpu_line_with_extra_fields() {
        // Some systems have extra fields beyond steal
        let line = "cpu0 100 10 50 800 20 5 5 10 100 200";
        let stats = CpuCollector::parse_cpu_line(line).expect("parsing should succeed");

        assert_eq!(stats.user, 100);
        assert_eq!(stats.steal, 10);
    }

    #[test]
    fn test_cpu_collector_id() {
        let collector = CpuCollector::new();
        assert_eq!(collector.id(), "cpu");
    }

    #[test]
    fn test_cpu_stats_clone() {
        let stats = CpuStats {
            user: 100,
            nice: 10,
            system: 50,
            idle: 800,
            iowait: 20,
            irq: 5,
            softirq: 5,
            steal: 10,
        };

        let cloned = stats.clone();
        assert_eq!(stats.total(), cloned.total());
        assert_eq!(stats.idle_time(), cloned.idle_time());
    }

    #[test]
    fn test_load_average_values() {
        let load = LoadAverage { one: 1.5, five: 2.0, fifteen: 1.8 };

        assert!((load.one - 1.5).abs() < f64::EPSILON);
        assert!((load.five - 2.0).abs() < f64::EPSILON);
        assert!((load.fifteen - 1.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cpu_frequency_values() {
        let freq = CpuFrequency { current_mhz: 2400, min_mhz: 800, max_mhz: 3600 };

        assert_eq!(freq.current_mhz, 2400);
        assert_eq!(freq.min_mhz, 800);
        assert_eq!(freq.max_mhz, 3600);
    }

    #[test]
    fn test_cpu_collector_default() {
        let collector1 = CpuCollector::new();
        let collector2 = CpuCollector::default();

        assert_eq!(collector1.core_count(), collector2.core_count());
    }
}
