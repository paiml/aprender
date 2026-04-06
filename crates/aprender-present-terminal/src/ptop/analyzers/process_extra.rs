//! Extended Process Information Analyzer
//!
//! Reads detailed process information from `/proc/[pid]/`:
//! - cgroup: Container/slice membership
//! - `oom_score`: OOM killer score (0-1000)
//! - `oom_score_adj`: OOM adjustment (-1000 to +1000)
//! - io: I/O statistics and priority
//! - status: CPU affinity, scheduler info

#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::single_char_pattern)]
#![allow(clippy::manual_let_else)]

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Duration;

use super::{Analyzer, AnalyzerError};

/// I/O priority class (from Linux)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IoPriorityClass {
    /// Real-time I/O (highest priority)
    RealTime,
    /// Best-effort (default for normal processes)
    #[default]
    BestEffort,
    /// Idle (lowest priority, only when system is idle)
    Idle,
    /// None/unknown
    None,
}

impl IoPriorityClass {
    /// Get display string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RealTime => "RT",
            Self::BestEffort => "BE",
            Self::Idle => "IDLE",
            Self::None => "-",
        }
    }
}

/// Extended process information
#[derive(Debug, Clone, Default)]
pub struct ProcessExtra {
    /// Process ID
    pub pid: u32,
    /// cgroup path (v2 unified hierarchy)
    pub cgroup: String,
    /// Container name extracted from cgroup (PMAT-GAP-032)
    pub container: Option<String>,
    /// OOM score (0-1000, higher = more likely to be killed)
    pub oom_score: i32,
    /// OOM score adjustment (-1000 to +1000)
    pub oom_score_adj: i32,
    /// Nice value (-20 to +19)
    pub nice: i32,
    /// CPU affinity mask (bit per CPU)
    pub cpu_affinity: Vec<bool>,
    /// I/O priority class
    pub io_class: IoPriorityClass,
    /// I/O priority level (0-7, lower = higher priority)
    pub io_priority: u8,
    /// Number of threads
    pub num_threads: u32,
    /// Voluntary context switches
    pub voluntary_ctxt_switches: u64,
    /// Involuntary context switches
    pub nonvoluntary_ctxt_switches: u64,
}

impl ProcessExtra {
    /// Get OOM risk level (0-100%)
    pub fn oom_risk_percent(&self) -> f64 {
        // oom_score is 0-1000
        self.oom_score as f64 / 10.0
    }

    /// Check if process is protected from OOM killer
    pub fn is_oom_protected(&self) -> bool {
        self.oom_score_adj == -1000
    }

    /// Get container badge string (PMAT-GAP-032 - ttop parity).
    ///
    /// Format: `[container_name]` or `[truncated…]` if > 12 chars.
    #[must_use]
    pub fn container_badge(&self) -> Option<String> {
        self.container.as_ref().map(|c| {
            if c.len() > 12 {
                format!("[{}…]", &c[..11])
            } else {
                format!("[{}]", c)
            }
        })
    }

    /// Check if process is running in a container.
    #[must_use]
    pub fn is_containerized(&self) -> bool {
        self.container.is_some()
    }

    /// Format cgroup for display (short form)
    pub fn cgroup_short(&self) -> String {
        if self.cgroup.is_empty() {
            return "-".to_string();
        }

        // Extract last component of cgroup path
        self.cgroup
            .rsplit('/')
            .find(|s| !s.is_empty())
            .map(|s| {
                if s.len() > 30 {
                    format!("{}...", &s[..27])
                } else {
                    s.to_string()
                }
            })
            .unwrap_or_else(|| "-".to_string())
    }

    /// Format CPU affinity for display
    pub fn affinity_display(&self) -> String {
        if self.cpu_affinity.is_empty() {
            return "-".to_string();
        }

        // Check if all CPUs are allowed
        if self.cpu_affinity.iter().all(|&x| x) {
            return "all".to_string();
        }

        // List specific CPUs
        let cpus: Vec<usize> = self
            .cpu_affinity
            .iter()
            .enumerate()
            .filter_map(|(i, &allowed)| if allowed { Some(i) } else { None })
            .collect();

        if cpus.len() <= 4 {
            cpus.iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(",")
        } else {
            format!("{} CPUs", cpus.len())
        }
    }
}

/// Collection of extended process info
#[derive(Debug, Clone, Default)]
pub struct ProcessExtraData {
    /// Map of PID to extra info
    pub processes: HashMap<u32, ProcessExtra>,
}

impl ProcessExtraData {
    /// Get extra info for a specific PID
    pub fn get(&self, pid: u32) -> Option<&ProcessExtra> {
        self.processes.get(&pid)
    }

    /// Get processes sorted by OOM score (highest first)
    pub fn by_oom_score(&self) -> Vec<&ProcessExtra> {
        let mut procs: Vec<_> = self.processes.values().collect();
        procs.sort_by(|a, b| b.oom_score.cmp(&a.oom_score));
        procs
    }

    /// Count of processes with high OOM risk (>50%)
    pub fn high_oom_risk_count(&self) -> usize {
        self.processes
            .values()
            .filter(|p| p.oom_risk_percent() > 50.0)
            .count()
    }
}

/// Analyzer for extended process information
pub struct ProcessExtraAnalyzer {
    data: ProcessExtraData,
    interval: Duration,
}

impl Default for ProcessExtraAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessExtraAnalyzer {
    /// Create a new process extra analyzer
    pub fn new() -> Self {
        Self {
            data: ProcessExtraData::default(),
            interval: Duration::from_secs(2),
        }
    }

    /// Get the current data
    pub fn data(&self) -> &ProcessExtraData {
        &self.data
    }

    /// Read extra info for a single process
    fn read_process_extra(&self, pid: u32) -> Option<ProcessExtra> {
        let proc_path = Path::new("/proc").join(pid.to_string());

        if !proc_path.exists() {
            return None;
        }

        let mut extra = ProcessExtra {
            pid,
            ..Default::default()
        };

        // Read cgroup
        if let Ok(content) = fs::read_to_string(proc_path.join("cgroup")) {
            // cgroup v2 format: "0::/path"
            // cgroup v1 format: "hierarchy:controller:path"
            extra.cgroup = content
                .lines()
                .next()
                .and_then(|line| line.split("::").nth(1).or_else(|| line.rsplit(':').next()))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
        }

        // Read oom_score
        if let Ok(content) = fs::read_to_string(proc_path.join("oom_score")) {
            extra.oom_score = content.trim().parse().unwrap_or(0);
        }

        // Read oom_score_adj
        if let Ok(content) = fs::read_to_string(proc_path.join("oom_score_adj")) {
            extra.oom_score_adj = content.trim().parse().unwrap_or(0);
        }

        // Read status for various fields
        if let Ok(content) = fs::read_to_string(proc_path.join("status")) {
            for line in content.lines() {
                if let Some((key, value)) = line.split_once(':') {
                    let value = value.trim();
                    match key {
                        "Threads" => {
                            extra.num_threads = value.parse().unwrap_or(1);
                        }
                        "voluntary_ctxt_switches" => {
                            extra.voluntary_ctxt_switches = value.parse().unwrap_or(0);
                        }
                        "nonvoluntary_ctxt_switches" => {
                            extra.nonvoluntary_ctxt_switches = value.parse().unwrap_or(0);
                        }
                        "Cpus_allowed" => {
                            // Parse hex CPU mask
                            extra.cpu_affinity = Self::parse_cpu_mask(value);
                        }
                        _ => {}
                    }
                }
            }
        }

        // Read stat for nice value
        if let Ok(content) = fs::read_to_string(proc_path.join("stat")) {
            // stat format: pid (comm) state ... nice ...
            // nice is field 19 (0-indexed: 18)
            let parts: Vec<&str> = content.split_whitespace().collect();
            if parts.len() > 18 {
                extra.nice = parts[18].parse().unwrap_or(0);
            }
        }

        // Read io priority from ionice or /proc/[pid]/io
        // Note: ionice info requires CAP_SYS_NICE or root, so we default to BestEffort
        extra.io_class = IoPriorityClass::BestEffort;
        extra.io_priority = 4; // Default best-effort priority

        Some(extra)
    }

    /// Parse CPU affinity hex mask
    fn parse_cpu_mask(hex: &str) -> Vec<bool> {
        let hex = hex.trim().replace(",", "");
        let mut cpus = Vec::new();

        // Parse from right to left (LSB first)
        for (i, c) in hex.chars().rev().enumerate() {
            let nibble = match c.to_digit(16) {
                Some(n) => n,
                None => continue,
            };

            for bit in 0..4 {
                let cpu_idx = i * 4 + bit;
                if cpu_idx < 256 {
                    // Reasonable max CPU count
                    while cpus.len() <= cpu_idx {
                        cpus.push(false);
                    }
                    cpus[cpu_idx] = (nibble & (1 << bit)) != 0;
                }
            }
        }

        // Trim trailing false values
        while cpus.last() == Some(&false) {
            cpus.pop();
        }

        cpus
    }
}

impl Analyzer for ProcessExtraAnalyzer {
    fn name(&self) -> &'static str {
        "process_extra"
    }

    fn collect(&mut self) -> Result<(), AnalyzerError> {
        let mut processes = HashMap::new();

        // Iterate over all processes
        let proc_path = Path::new("/proc");
        let Ok(entries) = fs::read_dir(proc_path) else {
            return Ok(());
        };

        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Only process numeric directories (PIDs)
            let Ok(pid) = name_str.parse::<u32>() else {
                continue;
            };

            if let Some(extra) = self.read_process_extra(pid) {
                processes.insert(pid, extra);
            }
        }

        self.data = ProcessExtraData { processes };
        Ok(())
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn available(&self) -> bool {
        Path::new("/proc/self/cgroup").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oom_risk_percent() {
        let mut extra = ProcessExtra::default();
        extra.oom_score = 500;
        assert!((extra.oom_risk_percent() - 50.0).abs() < 0.1);

        extra.oom_score = 1000;
        assert!((extra.oom_risk_percent() - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_oom_protected() {
        let mut extra = ProcessExtra::default();
        assert!(!extra.is_oom_protected());

        extra.oom_score_adj = -1000;
        assert!(extra.is_oom_protected());
    }

    #[test]
    fn test_cgroup_short() {
        let mut extra = ProcessExtra::default();
        extra.cgroup = "/user.slice/user-1000.slice/session-1.scope".to_string();
        assert_eq!(extra.cgroup_short(), "session-1.scope");

        extra.cgroup = "".to_string();
        assert_eq!(extra.cgroup_short(), "-");
    }

    #[test]
    fn test_affinity_display() {
        let mut extra = ProcessExtra::default();

        // Empty
        assert_eq!(extra.affinity_display(), "-");

        // All CPUs
        extra.cpu_affinity = vec![true, true, true, true];
        assert_eq!(extra.affinity_display(), "all");

        // Specific CPUs
        extra.cpu_affinity = vec![true, false, true, false];
        assert_eq!(extra.affinity_display(), "0,2");

        // Many CPUs
        extra.cpu_affinity = vec![true; 16];
        extra.cpu_affinity[0] = false;
        extra.cpu_affinity[1] = false;
        assert_eq!(extra.affinity_display(), "14 CPUs");
    }

    #[test]
    fn test_parse_cpu_mask() {
        // Single CPU (CPU 0)
        let mask = ProcessExtraAnalyzer::parse_cpu_mask("1");
        assert_eq!(mask, vec![true]);

        // CPUs 0 and 1
        let mask = ProcessExtraAnalyzer::parse_cpu_mask("3");
        assert_eq!(mask, vec![true, true]);

        // CPUs 0, 2 (binary: 0101 = 5)
        let mask = ProcessExtraAnalyzer::parse_cpu_mask("5");
        assert_eq!(mask, vec![true, false, true]);

        // All 8 CPUs (0xff)
        let mask = ProcessExtraAnalyzer::parse_cpu_mask("ff");
        assert_eq!(mask, vec![true; 8]);
    }

    #[test]
    fn test_io_priority_class_display() {
        assert_eq!(IoPriorityClass::RealTime.as_str(), "RT");
        assert_eq!(IoPriorityClass::BestEffort.as_str(), "BE");
        assert_eq!(IoPriorityClass::Idle.as_str(), "IDLE");
    }

    #[test]
    fn test_analyzer_available() {
        let analyzer = ProcessExtraAnalyzer::new();
        #[cfg(target_os = "linux")]
        assert!(analyzer.available());
    }

    #[test]
    fn test_analyzer_collect() {
        let mut analyzer = ProcessExtraAnalyzer::new();
        let result = analyzer.collect();
        assert!(result.is_ok());

        // Should have collected at least one process (ourselves)
        #[cfg(target_os = "linux")]
        {
            let data = analyzer.data();
            assert!(!data.processes.is_empty());

            // Our own process should be there
            let pid = std::process::id();
            assert!(data.get(pid).is_some());
        }
    }

    #[test]
    fn test_data_by_oom_score() {
        let mut data = ProcessExtraData::default();

        let mut p1 = ProcessExtra::default();
        p1.pid = 1;
        p1.oom_score = 100;

        let mut p2 = ProcessExtra::default();
        p2.pid = 2;
        p2.oom_score = 500;

        let mut p3 = ProcessExtra::default();
        p3.pid = 3;
        p3.oom_score = 300;

        data.processes.insert(1, p1);
        data.processes.insert(2, p2);
        data.processes.insert(3, p3);

        let sorted = data.by_oom_score();
        assert_eq!(sorted[0].pid, 2); // Highest OOM score
        assert_eq!(sorted[1].pid, 3);
        assert_eq!(sorted[2].pid, 1); // Lowest OOM score
    }

    #[test]
    fn test_io_priority_class_default() {
        let default = IoPriorityClass::default();
        assert_eq!(default, IoPriorityClass::BestEffort);
    }

    #[test]
    fn test_io_priority_class_none() {
        assert_eq!(IoPriorityClass::None.as_str(), "-");
    }

    #[test]
    fn test_process_extra_default() {
        let extra = ProcessExtra::default();
        assert_eq!(extra.pid, 0);
        assert!(extra.cgroup.is_empty());
        assert_eq!(extra.oom_score, 0);
        assert_eq!(extra.oom_score_adj, 0);
        assert_eq!(extra.nice, 0);
        assert!(extra.cpu_affinity.is_empty());
        assert_eq!(extra.io_class, IoPriorityClass::BestEffort);
        assert_eq!(extra.io_priority, 0);
        assert_eq!(extra.num_threads, 0);
    }

    #[test]
    fn test_process_extra_data_default() {
        let data = ProcessExtraData::default();
        assert!(data.processes.is_empty());
    }

    #[test]
    fn test_process_extra_data_get_none() {
        let data = ProcessExtraData::default();
        assert!(data.get(999).is_none());
    }

    #[test]
    fn test_high_oom_risk_count() {
        let mut data = ProcessExtraData::default();

        // Add process with low OOM risk
        let mut p1 = ProcessExtra::default();
        p1.pid = 1;
        p1.oom_score = 200; // 20%
        data.processes.insert(1, p1);

        // Add process with high OOM risk
        let mut p2 = ProcessExtra::default();
        p2.pid = 2;
        p2.oom_score = 600; // 60%
        data.processes.insert(2, p2);

        // Add another high risk process
        let mut p3 = ProcessExtra::default();
        p3.pid = 3;
        p3.oom_score = 800; // 80%
        data.processes.insert(3, p3);

        assert_eq!(data.high_oom_risk_count(), 2);
    }

    #[test]
    fn test_analyzer_default() {
        let analyzer = ProcessExtraAnalyzer::default();
        assert_eq!(analyzer.name(), "process_extra");
    }

    #[test]
    fn test_analyzer_interval() {
        let analyzer = ProcessExtraAnalyzer::new();
        assert_eq!(analyzer.interval(), Duration::from_secs(2));
    }

    #[test]
    fn test_analyzer_data() {
        let analyzer = ProcessExtraAnalyzer::new();
        let data = analyzer.data();
        assert!(data.processes.is_empty());
    }

    #[test]
    fn test_cgroup_short_long_name() {
        let mut extra = ProcessExtra::default();
        // Create a cgroup name longer than 30 chars
        extra.cgroup = "/very/long/path/to/a/very_very_very_very_very_long_cgroup_name".to_string();
        let short = extra.cgroup_short();
        assert!(short.contains("..."));
        assert!(short.len() <= 30);
    }

    #[test]
    fn test_cgroup_short_trailing_slash() {
        let mut extra = ProcessExtra::default();
        extra.cgroup = "/user.slice/session-1.scope/".to_string();
        assert_eq!(extra.cgroup_short(), "session-1.scope");
    }

    #[test]
    fn test_affinity_display_three_cpus() {
        let mut extra = ProcessExtra::default();
        extra.cpu_affinity = vec![true, true, true, false, false];
        assert_eq!(extra.affinity_display(), "0,1,2");
    }

    #[test]
    fn test_affinity_display_four_cpus() {
        let mut extra = ProcessExtra::default();
        extra.cpu_affinity = vec![true, true, true, true, false];
        assert_eq!(extra.affinity_display(), "0,1,2,3");
    }

    #[test]
    fn test_parse_cpu_mask_with_comma() {
        // Large masks often have commas
        let mask = ProcessExtraAnalyzer::parse_cpu_mask("ff,ff");
        assert_eq!(mask.len(), 16);
        assert!(mask.iter().all(|&x| x));
    }

    #[test]
    fn test_parse_cpu_mask_invalid_char() {
        let mask = ProcessExtraAnalyzer::parse_cpu_mask("z");
        assert!(mask.is_empty());
    }

    #[test]
    fn test_parse_cpu_mask_empty() {
        let mask = ProcessExtraAnalyzer::parse_cpu_mask("");
        assert!(mask.is_empty());
    }

    #[test]
    fn test_process_extra_clone() {
        let mut extra = ProcessExtra::default();
        extra.pid = 123;
        extra.oom_score = 500;
        extra.cgroup = "/test".to_string();
        extra.cpu_affinity = vec![true, false, true];

        let cloned = extra.clone();
        assert_eq!(cloned.pid, 123);
        assert_eq!(cloned.oom_score, 500);
        assert_eq!(cloned.cgroup, "/test");
        assert_eq!(cloned.cpu_affinity, vec![true, false, true]);
    }

    #[test]
    fn test_process_extra_data_clone() {
        let mut data = ProcessExtraData::default();
        let mut p1 = ProcessExtra::default();
        p1.pid = 1;
        data.processes.insert(1, p1);

        let cloned = data.clone();
        assert_eq!(cloned.processes.len(), 1);
    }

    #[test]
    fn test_process_extra_debug() {
        let extra = ProcessExtra::default();
        let debug = format!("{extra:?}");
        assert!(debug.contains("ProcessExtra"));
    }

    #[test]
    fn test_io_priority_class_eq() {
        assert_eq!(IoPriorityClass::RealTime, IoPriorityClass::RealTime);
        assert_ne!(IoPriorityClass::RealTime, IoPriorityClass::Idle);
    }

    // =========================================================================
    // Container badge tests (PMAT-GAP-032)
    // =========================================================================

    #[test]
    fn test_container_badge_none() {
        let extra = ProcessExtra::default();
        assert!(extra.container_badge().is_none());
    }

    #[test]
    fn test_container_badge_short() {
        let mut extra = ProcessExtra::default();
        extra.container = Some("nginx".to_string());
        assert_eq!(extra.container_badge(), Some("[nginx]".to_string()));
    }

    #[test]
    fn test_container_badge_exact_12() {
        let mut extra = ProcessExtra::default();
        extra.container = Some("exactly12chr".to_string()); // 12 chars
        assert_eq!(extra.container_badge(), Some("[exactly12chr]".to_string()));
    }

    #[test]
    fn test_container_badge_truncated() {
        let mut extra = ProcessExtra::default();
        extra.container = Some("very-long-container-name".to_string());
        // Should truncate to first 11 chars + "…"
        assert_eq!(extra.container_badge(), Some("[very-long-c…]".to_string()));
    }

    #[test]
    fn test_container_badge_13_chars() {
        let mut extra = ProcessExtra::default();
        extra.container = Some("1234567890123".to_string()); // 13 chars
        assert_eq!(extra.container_badge(), Some("[12345678901…]".to_string()));
    }

    #[test]
    fn test_is_containerized_false() {
        let extra = ProcessExtra::default();
        assert!(!extra.is_containerized());
    }

    #[test]
    fn test_is_containerized_true() {
        let mut extra = ProcessExtra::default();
        extra.container = Some("docker-abc123".to_string());
        assert!(extra.is_containerized());
    }
}
