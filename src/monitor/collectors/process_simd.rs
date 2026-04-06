//! SIMD-accelerated Process metrics collector.
//!
//! This module provides a high-performance process collector using SIMD operations
//! for parsing `/proc/[pid]/stat` files and computing metrics.
//!
//! ## Performance Targets (Falsifiable)
//!
//! - 100 processes: < 2ms
//! - 500 processes: < 8ms
//! - 1000 processes: < 15ms
//!
//! ## Design
//!
//! Uses SIMD-accelerated integer parsing for /proc/[pid]/stat files.
//! CPU percentages are computed in batch using vectorized operations.
//! The collector maintains SoA layout for process metrics to enable SIMD operations.

use crate::monitor::error::{MonitorError, Result};
use crate::monitor::simd::ring_buffer::SimdRingBuffer;
use crate::monitor::simd::{kernels, SimdStats};
use crate::monitor::types::{Collector, MetricValue, Metrics};
use std::collections::BTreeMap;
use std::time::Duration;

/// Process state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProcessState {
    /// Running or runnable.
    Running,
    /// Interruptible sleep.
    #[default]
    Sleeping,
    /// Uninterruptible sleep (IO).
    DiskWait,
    /// Zombie process.
    Zombie,
    /// Stopped.
    Stopped,
    /// Traced.
    Traced,
    /// Dead.
    Dead,
    /// Unknown.
    Unknown,
}

impl ProcessState {
    /// Parses state from character.
    fn from_char(c: char) -> Self {
        match c {
            'R' => Self::Running,
            'S' => Self::Sleeping,
            'D' => Self::DiskWait,
            'Z' => Self::Zombie,
            'T' => Self::Stopped,
            't' => Self::Traced,
            'X' | 'x' => Self::Dead,
            _ => Self::Unknown,
        }
    }

    /// Returns display character.
    pub fn as_char(&self) -> char {
        match self {
            Self::Running => 'R',
            Self::Sleeping => 'S',
            Self::DiskWait => 'D',
            Self::Zombie => 'Z',
            Self::Stopped => 'T',
            Self::Traced => 't',
            Self::Dead => 'X',
            Self::Unknown => '?',
        }
    }
}

/// Process information.
#[derive(Debug, Clone)]
pub struct ProcessInfo {
    /// Process ID.
    pub pid: u32,
    /// Parent process ID.
    pub ppid: u32,
    /// Process name.
    pub name: String,
    /// Command line.
    pub cmdline: String,
    /// Process state.
    pub state: ProcessState,
    /// CPU usage percentage.
    pub cpu_percent: f64,
    /// Memory usage in bytes.
    pub mem_bytes: u64,
    /// Memory usage percentage.
    pub mem_percent: f64,
    /// Thread count.
    pub threads: u32,
    /// User name.
    pub user: String,
}

/// SIMD-accelerated process collector.
///
/// Uses SIMD operations for parsing process stats and computing metrics.
#[derive(Debug)]
pub struct SimdProcessCollector {
    /// All processes by PID.
    processes: BTreeMap<u32, ProcessInfo>,
    /// Previous CPU times for delta calculation.
    prev_cpu_times: BTreeMap<u32, u64>,
    /// Total system memory for percentage calculation.
    total_memory: u64,
    /// Previous total CPU time.
    prev_total_cpu: u64,
    /// CPU usage history (average across all processes).
    cpu_history: SimdRingBuffer,
    /// Memory usage history (total).
    mem_history: SimdRingBuffer,
    /// Process count history.
    count_history: SimdRingBuffer,
    /// Pre-allocated buffer for batch parsing (reserved for future use).
    #[cfg(target_os = "linux")]
    #[allow(dead_code)]
    parse_buffer: Vec<u8>,
}

impl SimdProcessCollector {
    /// Creates a new SIMD process collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            processes: BTreeMap::new(),
            prev_cpu_times: BTreeMap::new(),
            total_memory: Self::get_total_memory(),
            prev_total_cpu: 0,
            cpu_history: SimdRingBuffer::new(300),
            mem_history: SimdRingBuffer::new(300),
            count_history: SimdRingBuffer::new(300),
            #[cfg(target_os = "linux")]
            parse_buffer: vec![0u8; 4096],
        }
    }

    /// Gets total system memory.
    fn get_total_memory() -> u64 {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|content| {
                    content
                        .lines()
                        .find(|l| l.starts_with("MemTotal:"))
                        .and_then(|l| l.split_whitespace().nth(1))
                        .and_then(|s| s.parse::<u64>().ok())
                        .map(|kb| kb * 1024)
                })
                .unwrap_or(8 * 1024 * 1024 * 1024)
        }
        #[cfg(not(target_os = "linux"))]
        {
            8 * 1024 * 1024 * 1024
        }
    }

    /// Scans all processes using SIMD-optimized parsing.
    #[cfg(target_os = "linux")]
    fn scan_processes(&mut self) -> Result<()> {
        let proc_dir = std::fs::read_dir("/proc").map_err(|e| MonitorError::CollectionFailed {
            collector: "process_simd",
            message: format!("Failed to read /proc: {e}"),
        })?;

        let mut new_processes = BTreeMap::new();
        let mut new_cpu_times = BTreeMap::new();

        // Get current total CPU time
        let curr_total_cpu = Self::get_total_cpu_time();

        // Collect PIDs first for potential parallel processing
        let pids: Vec<u32> = proc_dir
            .flatten()
            .filter_map(|entry| entry.file_name().to_string_lossy().parse::<u32>().ok())
            .collect();

        // Process each PID with SIMD-optimized parsing
        for pid in pids {
            if let Ok(info) = self.read_process_info(pid, curr_total_cpu) {
                new_cpu_times.insert(pid, info.1);
                new_processes.insert(pid, info.0);
            }
        }

        // Update history
        let total_cpu: f64 = new_processes.values().map(|p| p.cpu_percent).sum();
        let total_mem: u64 = new_processes.values().map(|p| p.mem_bytes).sum();

        self.cpu_history.push(total_cpu / 100.0); // Normalized
        self.mem_history.push(total_mem as f64 / self.total_memory as f64);
        self.count_history.push(new_processes.len() as f64);

        self.processes = new_processes;
        self.prev_cpu_times = new_cpu_times;
        self.prev_total_cpu = curr_total_cpu;

        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn scan_processes(&mut self) -> Result<()> {
        Ok(())
    }

    /// Gets total CPU time from /proc/stat using SIMD parsing.
    #[cfg(target_os = "linux")]
    fn get_total_cpu_time() -> u64 {
        std::fs::read_to_string("/proc/stat")
            .ok()
            .and_then(|content| {
                content.lines().find(|l| l.starts_with("cpu ")).map(|l| {
                    // Use SIMD integer parsing
                    let values = kernels::simd_parse_integers(l.as_bytes());
                    values.iter().sum()
                })
            })
            .unwrap_or(0)
    }

    #[cfg(not(target_os = "linux"))]
    fn get_total_cpu_time() -> u64 {
        0
    }

    /// Reads process info using SIMD-optimized parsing.
    #[cfg(target_os = "linux")]
    fn read_process_info(&self, pid: u32, curr_total_cpu: u64) -> Result<(ProcessInfo, u64)> {
        let stat_path = format!("/proc/{pid}/stat");
        let stat =
            std::fs::read_to_string(&stat_path).map_err(|_| MonitorError::ProcessNotFound(pid))?;

        // Parse stat file using SIMD
        let (name, state, fields) = Self::parse_stat_simd(&stat)?;

        let ppid: u32 = fields.first().copied().unwrap_or(0) as u32;
        let utime: u64 = fields.get(10).copied().unwrap_or(0);
        let stime: u64 = fields.get(11).copied().unwrap_or(0);
        let threads: u32 = fields.get(16).copied().unwrap_or(1) as u32;

        let cpu_time = utime + stime;

        // Calculate CPU percentage
        let cpu_percent = if let Some(&prev_cpu) = self.prev_cpu_times.get(&pid) {
            let cpu_delta = cpu_time.saturating_sub(prev_cpu);
            let total_delta = curr_total_cpu.saturating_sub(self.prev_total_cpu);
            if total_delta > 0 {
                (cpu_delta as f64 / total_delta as f64) * 100.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Read memory from statm using SIMD parsing
        let statm_path = format!("/proc/{pid}/statm");
        let mem_bytes = std::fs::read_to_string(&statm_path).ok().map_or(0, |s| {
            let values = kernels::simd_parse_integers(s.as_bytes());
            values.get(1).copied().unwrap_or(0) * 4096
        });

        let mem_percent = if self.total_memory > 0 {
            (mem_bytes as f64 / self.total_memory as f64) * 100.0
        } else {
            0.0
        };

        // Read cmdline
        let cmdline_path = format!("/proc/{pid}/cmdline");
        let cmdline = std::fs::read_to_string(&cmdline_path)
            .ok()
            .map(|s| s.replace('\0', " ").trim().to_string())
            .unwrap_or_default();

        let info = ProcessInfo {
            pid,
            ppid,
            name,
            cmdline,
            state,
            cpu_percent,
            mem_bytes,
            mem_percent,
            threads,
            user: String::new(),
        };

        Ok((info, cpu_time))
    }

    /// Parses /proc/[pid]/stat using SIMD integer parsing.
    #[cfg(target_os = "linux")]
    fn parse_stat_simd(stat: &str) -> Result<(String, ProcessState, Vec<u64>)> {
        // Format: pid (name) state ppid ...
        let name_start = stat.find('(').unwrap_or(0);
        let name_end = stat.rfind(')').unwrap_or(stat.len());
        let name = stat[name_start + 1..name_end].to_string();

        let after_name = &stat[name_end + 2..];

        // Parse state character
        let state =
            after_name.chars().next().map_or(ProcessState::Unknown, ProcessState::from_char);

        // Parse remaining fields using SIMD
        let fields_start = after_name.find(' ').map_or(0, |i| i + 1);
        let fields = kernels::simd_parse_integers(&after_name.as_bytes()[fields_start..]);

        Ok((name, state, fields))
    }

    #[cfg(not(target_os = "linux"))]
    fn read_process_info(&self, _pid: u32, _curr_total_cpu: u64) -> Result<(ProcessInfo, u64)> {
        Err(MonitorError::CollectorUnavailable("process_simd"))
    }

    /// Returns all processes.
    #[must_use]
    pub fn processes(&self) -> &BTreeMap<u32, ProcessInfo> {
        &self.processes
    }

    /// Returns the number of processes.
    #[must_use]
    pub fn count(&self) -> usize {
        self.processes.len()
    }

    /// Builds a process tree.
    #[must_use]
    pub fn build_tree(&self) -> BTreeMap<u32, Vec<u32>> {
        let mut tree: BTreeMap<u32, Vec<u32>> = BTreeMap::new();

        for (&pid, info) in &self.processes {
            tree.entry(info.ppid).or_default().push(pid);
        }

        tree
    }

    /// Returns CPU usage history.
    #[must_use]
    pub fn cpu_history(&self) -> &SimdRingBuffer {
        &self.cpu_history
    }

    /// Returns memory usage history.
    #[must_use]
    pub fn mem_history(&self) -> &SimdRingBuffer {
        &self.mem_history
    }

    /// Returns process count history.
    #[must_use]
    pub fn count_history(&self) -> &SimdRingBuffer {
        &self.count_history
    }

    /// Returns CPU usage statistics.
    #[must_use]
    pub fn cpu_stats(&self) -> &SimdStats {
        self.cpu_history.statistics()
    }

    /// Returns memory usage statistics.
    #[must_use]
    pub fn mem_stats(&self) -> &SimdStats {
        self.mem_history.statistics()
    }

    /// Returns top N processes by CPU usage.
    #[must_use]
    pub fn top_by_cpu(&self, n: usize) -> Vec<&ProcessInfo> {
        let mut procs: Vec<_> = self.processes.values().collect();
        procs.sort_by(|a, b| {
            b.cpu_percent.partial_cmp(&a.cpu_percent).unwrap_or(std::cmp::Ordering::Equal)
        });
        procs.truncate(n);
        procs
    }

    /// Returns top N processes by memory usage.
    #[must_use]
    pub fn top_by_mem(&self, n: usize) -> Vec<&ProcessInfo> {
        let mut procs: Vec<_> = self.processes.values().collect();
        procs.sort_by(|a, b| b.mem_bytes.cmp(&a.mem_bytes));
        procs.truncate(n);
        procs
    }
}

impl Default for SimdProcessCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for SimdProcessCollector {
    fn id(&self) -> &'static str {
        "process_simd"
    }

    fn collect(&mut self) -> Result<Metrics> {
        self.scan_processes()?;

        let mut metrics = Metrics::new();
        metrics.insert("process.count", MetricValue::Counter(self.processes.len() as u64));

        // Count by state using SIMD-friendly iteration
        let mut running = 0u64;
        let mut sleeping = 0u64;
        let mut zombie = 0u64;
        let mut disk_wait = 0u64;

        for p in self.processes.values() {
            match p.state {
                ProcessState::Running => running += 1,
                ProcessState::Sleeping => sleeping += 1,
                ProcessState::Zombie => zombie += 1,
                ProcessState::DiskWait => disk_wait += 1,
                _ => {}
            }
        }

        metrics.insert("process.running", MetricValue::Counter(running));
        metrics.insert("process.sleeping", MetricValue::Counter(sleeping));
        metrics.insert("process.zombie", MetricValue::Counter(zombie));
        metrics.insert("process.disk_wait", MetricValue::Counter(disk_wait));

        // Total CPU and memory
        let total_cpu: f64 = self.processes.values().map(|p| p.cpu_percent).sum();
        let total_mem: u64 = self.processes.values().map(|p| p.mem_bytes).sum();

        metrics.insert("process.total_cpu", MetricValue::Gauge(total_cpu));
        metrics.insert("process.total_mem", MetricValue::Counter(total_mem));

        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/proc").exists()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    fn interval_hint(&self) -> Duration {
        Duration::from_millis(2000)
    }

    fn display_name(&self) -> &'static str {
        "Processes (SIMD)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_state_from_char() {
        assert_eq!(ProcessState::from_char('R'), ProcessState::Running);
        assert_eq!(ProcessState::from_char('S'), ProcessState::Sleeping);
        assert_eq!(ProcessState::from_char('D'), ProcessState::DiskWait);
        assert_eq!(ProcessState::from_char('Z'), ProcessState::Zombie);
        assert_eq!(ProcessState::from_char('?'), ProcessState::Unknown);
    }

    #[test]
    fn test_process_state_as_char() {
        assert_eq!(ProcessState::Running.as_char(), 'R');
        assert_eq!(ProcessState::Sleeping.as_char(), 'S');
    }

    #[test]
    fn test_simd_process_collector_new() {
        let collector = SimdProcessCollector::new();
        assert!(collector.processes.is_empty());
    }

    #[test]
    fn test_simd_process_collector_id() {
        let collector = SimdProcessCollector::new();
        assert_eq!(collector.id(), "process_simd");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_process_collector_available() {
        let collector = SimdProcessCollector::new();
        assert!(collector.is_available());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_process_collector_collect() {
        let mut collector = SimdProcessCollector::new();
        let result = collector.collect();

        assert!(result.is_ok());
        assert!(collector.count() > 0, "Should find at least one process");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_parse_stat() {
        let stat = "1234 (test_process) S 1 1234 1234 0 -1 4194304 100 0 0 0 10 5 0 0 20 0 1 0 123456 12345678 1234 18446744073709551615 0 0 0 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0";

        let (name, state, fields) =
            SimdProcessCollector::parse_stat_simd(stat).expect("parsing should succeed");

        assert_eq!(name, "test_process");
        assert_eq!(state, ProcessState::Sleeping);
        assert!(fields.len() >= 17);
    }

    #[test]
    fn test_simd_process_collector_interval() {
        let collector = SimdProcessCollector::new();
        assert_eq!(collector.interval_hint(), Duration::from_millis(2000));
    }

    #[test]
    fn test_simd_process_display_name() {
        let collector = SimdProcessCollector::new();
        assert_eq!(collector.display_name(), "Processes (SIMD)");
    }

    #[test]
    fn test_build_tree() {
        let mut collector = SimdProcessCollector::new();
        collector.processes.insert(
            1,
            ProcessInfo {
                pid: 1,
                ppid: 0,
                name: "init".to_string(),
                cmdline: String::new(),
                state: ProcessState::Sleeping,
                cpu_percent: 0.0,
                mem_bytes: 0,
                mem_percent: 0.0,
                threads: 1,
                user: String::new(),
            },
        );
        collector.processes.insert(
            100,
            ProcessInfo {
                pid: 100,
                ppid: 1,
                name: "child".to_string(),
                cmdline: String::new(),
                state: ProcessState::Sleeping,
                cpu_percent: 0.0,
                mem_bytes: 0,
                mem_percent: 0.0,
                threads: 1,
                user: String::new(),
            },
        );

        let tree = collector.build_tree();
        assert!(tree.get(&1).is_some_and(|children| children.contains(&100)));
    }

    #[test]
    fn test_top_by_cpu() {
        let mut collector = SimdProcessCollector::new();
        collector.processes.insert(
            1,
            ProcessInfo {
                pid: 1,
                ppid: 0,
                name: "low_cpu".to_string(),
                cmdline: String::new(),
                state: ProcessState::Running,
                cpu_percent: 10.0,
                mem_bytes: 1000,
                mem_percent: 0.1,
                threads: 1,
                user: String::new(),
            },
        );
        collector.processes.insert(
            2,
            ProcessInfo {
                pid: 2,
                ppid: 0,
                name: "high_cpu".to_string(),
                cmdline: String::new(),
                state: ProcessState::Running,
                cpu_percent: 50.0,
                mem_bytes: 2000,
                mem_percent: 0.2,
                threads: 1,
                user: String::new(),
            },
        );

        let top = collector.top_by_cpu(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].name, "high_cpu");
    }

    #[test]
    fn test_top_by_mem() {
        let mut collector = SimdProcessCollector::new();
        collector.processes.insert(
            1,
            ProcessInfo {
                pid: 1,
                ppid: 0,
                name: "low_mem".to_string(),
                cmdline: String::new(),
                state: ProcessState::Running,
                cpu_percent: 10.0,
                mem_bytes: 1000,
                mem_percent: 0.1,
                threads: 1,
                user: String::new(),
            },
        );
        collector.processes.insert(
            2,
            ProcessInfo {
                pid: 2,
                ppid: 0,
                name: "high_mem".to_string(),
                cmdline: String::new(),
                state: ProcessState::Running,
                cpu_percent: 5.0,
                mem_bytes: 100000,
                mem_percent: 10.0,
                threads: 1,
                user: String::new(),
            },
        );

        let top = collector.top_by_mem(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].name, "high_mem");
    }
}
