//! Process metrics collector.
//!
//! Parses `/proc/[pid]/*` on Linux to collect process information.

use crate::monitor::error::{MonitorError, Result};
#[cfg(target_os = "macos")]
use crate::monitor::subprocess::run_with_timeout;
use crate::monitor::types::{Collector, MetricValue, Metrics};
use std::collections::BTreeMap;
use std::time::Duration;

/// Process state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessState {
    /// Running or runnable (on run queue).
    Running,
    /// Interruptible sleep (waiting for an event).
    Sleeping,
    /// Uninterruptible sleep (usually IO).
    DiskWait,
    /// Defunct/zombie process.
    Zombie,
    /// Stopped (on signal or by debugger).
    Stopped,
    /// Tracing stop (by debugger).
    Traced,
    /// Dead (should never be seen).
    Dead,
    /// Unknown state.
    Unknown,
}

impl ProcessState {
    /// Parses a state character from /proc/[pid]/stat.
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

    /// Returns a display character.
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

/// Information about a single process.
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

/// Collector for process metrics.
#[derive(Debug)]
pub struct ProcessCollector {
    /// All processes by PID.
    processes: BTreeMap<u32, ProcessInfo>,
    /// Previous CPU times for delta calculation.
    prev_cpu_times: BTreeMap<u32, u64>,
    /// Total system memory for percentage calculation.
    total_memory: u64,
    /// Previous total CPU time.
    prev_total_cpu: u64,
}

impl ProcessCollector {
    /// Creates a new process collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            processes: BTreeMap::new(),
            prev_cpu_times: BTreeMap::new(),
            total_memory: Self::get_total_memory(),
            prev_total_cpu: 0,
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
        #[cfg(target_os = "macos")]
        {
            run_with_timeout("sysctl", &["-n", "hw.memsize"], Duration::from_secs(2))
                .stdout_string()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(8 * 1024 * 1024 * 1024)
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            8 * 1024 * 1024 * 1024
        }
    }

    /// Scans all processes.
    #[cfg(target_os = "linux")]
    fn scan_processes(&mut self) -> Result<()> {
        let proc_dir = std::fs::read_dir("/proc").map_err(|e| MonitorError::CollectionFailed {
            collector: "process",
            message: format!("Failed to read /proc: {e}"),
        })?;

        let mut new_processes = BTreeMap::new();
        let mut new_cpu_times = BTreeMap::new();

        // Get current total CPU time
        let curr_total_cpu = Self::get_total_cpu_time();

        for entry in proc_dir.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Only process numeric directories (PIDs)
            if let Ok(pid) = name_str.parse::<u32>() {
                if let Ok(info) = self.read_process_info(pid, curr_total_cpu) {
                    new_cpu_times.insert(pid, info.1);
                    new_processes.insert(pid, info.0);
                }
            }
        }

        self.processes = new_processes;
        self.prev_cpu_times = new_cpu_times;
        self.prev_total_cpu = curr_total_cpu;

        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn scan_processes(&mut self) -> Result<()> {
        // Use ps to get process information on macOS
        // Note: macOS ps doesn't support nlwp, so we skip thread count
        // Use timeout to prevent hangs on slow systems
        let result = run_with_timeout(
            "ps",
            &["-axo", "pid,ppid,state,%cpu,%mem,rss,user,comm"],
            Duration::from_secs(5),
        );

        let content = match result.stdout_string() {
            Some(s) => s,
            None => {
                // Timeout or error - return empty but don't fail
                return Ok(());
            }
        };
        let mut new_processes = BTreeMap::new();

        for line in content.lines().skip(1) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 8 {
                continue;
            }

            let pid: u32 = match parts[0].parse() {
                Ok(p) => p,
                Err(_) => continue,
            };
            let ppid: u32 = parts[1].parse().unwrap_or(0);

            // macOS state chars: R=running, S=sleeping, I=idle, U=uninterruptible, Z=zombie, T=stopped
            let state = match parts[2].chars().next().unwrap_or('?') {
                'R' => ProcessState::Running,
                'S' | 'I' => ProcessState::Sleeping,
                'U' => ProcessState::DiskWait,
                'Z' => ProcessState::Zombie,
                'T' => ProcessState::Stopped,
                _ => ProcessState::Unknown,
            };

            let cpu_percent: f64 = parts[3].parse().unwrap_or(0.0);
            let mem_percent: f64 = parts[4].parse().unwrap_or(0.0);
            let rss_kb: u64 = parts[5].parse().unwrap_or(0);
            let user = parts[6].to_string();
            let name = parts[7..].join(" ");

            let mem_bytes = rss_kb * 1024;

            new_processes.insert(
                pid,
                ProcessInfo {
                    pid,
                    ppid,
                    name: name.clone(),
                    cmdline: name, // ps doesn't give full cmdline easily
                    state,
                    cpu_percent,
                    mem_bytes,
                    mem_percent,
                    threads: 1, // Thread count not available via ps on macOS
                    user,
                },
            );
        }

        self.processes = new_processes;
        Ok(())
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn scan_processes(&mut self) -> Result<()> {
        Ok(())
    }

    /// Gets total CPU time from /proc/stat.
    #[cfg(target_os = "linux")]
    fn get_total_cpu_time() -> u64 {
        std::fs::read_to_string("/proc/stat")
            .ok()
            .and_then(|content| {
                content.lines().find(|l| l.starts_with("cpu ")).map(|l| {
                    l.split_whitespace().skip(1).filter_map(|s| s.parse::<u64>().ok()).sum()
                })
            })
            .unwrap_or(0)
    }

    #[cfg(not(target_os = "linux"))]
    fn get_total_cpu_time() -> u64 {
        0
    }

    /// Reads information about a single process.
    #[cfg(target_os = "linux")]
    fn read_process_info(&self, pid: u32, curr_total_cpu: u64) -> Result<(ProcessInfo, u64)> {
        let stat_path = format!("/proc/{pid}/stat");
        let stat =
            std::fs::read_to_string(&stat_path).map_err(|_| MonitorError::ProcessNotFound(pid))?;

        // Parse stat file - format: pid (name) state ppid ... utime stime ...
        let name_start = stat.find('(').unwrap_or(0);
        let name_end = stat.rfind(')').unwrap_or(stat.len());
        let name = stat[name_start + 1..name_end].to_string();

        let after_name = &stat[name_end + 2..];
        let fields: Vec<&str> = after_name.split_whitespace().collect();

        let state = fields
            .first()
            .and_then(|s| s.chars().next())
            .map_or(ProcessState::Unknown, ProcessState::from_char);
        let ppid: u32 = fields.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        let utime: u64 = fields.get(11).and_then(|s| s.parse().ok()).unwrap_or(0);
        let stime: u64 = fields.get(12).and_then(|s| s.parse().ok()).unwrap_or(0);
        let threads: u32 = fields.get(17).and_then(|s| s.parse().ok()).unwrap_or(1);

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

        // Read memory from statm
        let statm_path = format!("/proc/{pid}/statm");
        let mem_bytes = std::fs::read_to_string(&statm_path)
            .ok()
            .and_then(|s| s.split_whitespace().nth(1).and_then(|s| s.parse::<u64>().ok()))
            .map_or(0, |pages| pages * 4096);

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
            user: String::new(), // TODO: Read from /proc/[pid]/status
        };

        Ok((info, cpu_time))
    }

    #[cfg(not(target_os = "linux"))]
    fn read_process_info(&self, _pid: u32, _curr_total_cpu: u64) -> Result<(ProcessInfo, u64)> {
        Err(MonitorError::CollectorUnavailable("process"))
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

    /// Builds a process tree (parent -> children mapping).
    #[must_use]
    pub fn build_tree(&self) -> BTreeMap<u32, Vec<u32>> {
        let mut tree: BTreeMap<u32, Vec<u32>> = BTreeMap::new();

        for (&pid, info) in &self.processes {
            tree.entry(info.ppid).or_default().push(pid);
        }

        tree
    }
}

impl Default for ProcessCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for ProcessCollector {
    fn id(&self) -> &'static str {
        "process"
    }

    fn collect(&mut self) -> Result<Metrics> {
        self.scan_processes()?;

        let mut metrics = Metrics::new();
        metrics.insert("process.count", MetricValue::Counter(self.processes.len() as u64));

        // Count by state
        let running = self.processes.values().filter(|p| p.state == ProcessState::Running).count();
        let sleeping =
            self.processes.values().filter(|p| p.state == ProcessState::Sleeping).count();
        let zombie = self.processes.values().filter(|p| p.state == ProcessState::Zombie).count();

        metrics.insert("process.running", MetricValue::Counter(running as u64));
        metrics.insert("process.sleeping", MetricValue::Counter(sleeping as u64));
        metrics.insert("process.zombie", MetricValue::Counter(zombie as u64));

        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/proc").exists()
        }
        #[cfg(target_os = "macos")]
        {
            true // macOS uses ps which is always available
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            false
        }
    }

    fn interval_hint(&self) -> Duration {
        Duration::from_millis(2000) // Process scanning is more expensive
    }

    fn display_name(&self) -> &'static str {
        "Processes"
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
    fn test_process_collector_new() {
        let collector = ProcessCollector::new();
        assert!(collector.processes.is_empty());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_process_collector_is_available() {
        let collector = ProcessCollector::new();
        assert!(collector.is_available());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_process_collector_collect() {
        let mut collector = ProcessCollector::new();
        let result = collector.collect();

        assert!(result.is_ok());
        assert!(collector.count() > 0, "Should find at least one process");
    }

    #[test]
    fn test_process_collector_interval() {
        let collector = ProcessCollector::new();
        assert_eq!(collector.interval_hint(), Duration::from_millis(2000));
    }

    #[test]
    fn test_build_tree() {
        let mut collector = ProcessCollector::new();
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
}
