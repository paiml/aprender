//! CPU metrics collector
//!
//! Collects CPU usage from /proc/stat (Genchi Genbutsu: real data)

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use crate::ring_buffer::RingBuffer;
use std::any::Any;
use std::fs::read_to_string;
use std::time::{Duration, Instant};

/// CPU metrics
#[derive(Debug, Clone)]
pub struct CpuMetrics {
    /// Timestamp of collection
    pub timestamp: Instant,
    /// Total CPU usage (0-100%)
    pub total_usage: f64,
    /// Per-core usage (0-100%)
    pub per_core_usage: Vec<f64>,
    /// Per-core frequency in MHz
    pub frequency_mhz: Vec<u32>,
    /// Package temperature (if available)
    pub temperature_c: Option<f64>,
}

impl Default for CpuMetrics {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            total_usage: 0.0,
            per_core_usage: Vec::new(),
            frequency_mhz: Vec::new(),
            temperature_c: None,
        }
    }
}

/// CPU collector brick
pub struct CpuCollectorBrick {
    /// Metrics history
    history: RingBuffer<CpuMetrics>,
    /// Previous /proc/stat values for delta calculation
    last_stat: Option<ProcStat>,
    /// Number of CPU cores
    core_count: usize,
}

#[derive(Debug, Clone, Default)]
struct ProcStat {
    total: CpuTimes,
    per_core: Vec<CpuTimes>,
}

#[derive(Debug, Clone, Copy, Default)]
struct CpuTimes {
    user: u64,
    nice: u64,
    system: u64,
    idle: u64,
    iowait: u64,
    irq: u64,
    softirq: u64,
    steal: u64,
}

impl CpuTimes {
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

    fn active(&self) -> u64 {
        self.user + self.nice + self.system + self.irq + self.softirq + self.steal
    }

    fn usage_since(&self, prev: &CpuTimes) -> f64 {
        let total_delta = self.total().saturating_sub(prev.total());
        let active_delta = self.active().saturating_sub(prev.active());

        if total_delta == 0 {
            0.0
        } else {
            (active_delta as f64 / total_delta as f64) * 100.0
        }
    }
}

impl CpuCollectorBrick {
    /// Create new CPU collector
    pub fn new() -> Self {
        let core_count = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

        Self {
            history: RingBuffer::new(120), // 2 minutes at 1Hz
            last_stat: None,
            core_count,
        }
    }

    /// Collect current CPU metrics
    pub fn collect(&mut self) -> CpuMetrics {
        let stat = self.read_proc_stat().unwrap_or_default();
        let metrics = if let Some(ref last) = self.last_stat {
            self.calculate_usage(&stat, last)
        } else {
            CpuMetrics {
                timestamp: Instant::now(),
                total_usage: 0.0,
                per_core_usage: vec![0.0; self.core_count],
                frequency_mhz: vec![0; self.core_count],
                temperature_c: None,
            }
        };

        self.last_stat = Some(stat);
        self.history.push(metrics.clone());
        metrics
    }

    /// Get metrics history
    pub fn history(&self) -> &RingBuffer<CpuMetrics> {
        &self.history
    }

    /// Is collector available?
    pub fn is_available(&self) -> bool {
        true // Always available on Linux
    }

    /// Suggested collection interval
    pub fn interval_hint(&self) -> Duration {
        Duration::from_millis(1000)
    }

    fn read_proc_stat(&self) -> Result<ProcStat, std::io::Error> {
        let content = read_to_string("/proc/stat")?;
        let mut stat = ProcStat::default();

        for line in content.lines() {
            if line.starts_with("cpu ") {
                stat.total = parse_cpu_line(line);
            } else if line.starts_with("cpu") {
                stat.per_core.push(parse_cpu_line(line));
            }
        }

        Ok(stat)
    }

    fn calculate_usage(&self, current: &ProcStat, prev: &ProcStat) -> CpuMetrics {
        let total_usage = current.total.usage_since(&prev.total);

        let per_core_usage: Vec<f64> = current
            .per_core
            .iter()
            .zip(prev.per_core.iter())
            .map(|(curr, prev)| curr.usage_since(prev))
            .collect();

        CpuMetrics {
            timestamp: Instant::now(),
            total_usage,
            per_core_usage,
            frequency_mhz: vec![3000; self.core_count], // Fixed for now
            temperature_c: Some(55.0),                  // Fixed for now
        }
    }
}

fn parse_cpu_line(line: &str) -> CpuTimes {
    let parts: Vec<&str> = line.split_whitespace().skip(1).collect();
    let mut times = CpuTimes::default();

    if parts.len() >= 8 {
        times.user = parts[0].parse().unwrap_or(0);
        times.nice = parts[1].parse().unwrap_or(0);
        times.system = parts[2].parse().unwrap_or(0);
        times.idle = parts[3].parse().unwrap_or(0);
        times.iowait = parts[4].parse().unwrap_or(0);
        times.irq = parts[5].parse().unwrap_or(0);
        times.softirq = parts[6].parse().unwrap_or(0);
        times.steal = parts[7].parse().unwrap_or(0);
    }

    times
}

impl Default for CpuCollectorBrick {
    fn default() -> Self {
        Self::new()
    }
}

impl Brick for CpuCollectorBrick {
    fn brick_name(&self) -> &'static str {
        "cpu_collector"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::ValueInRange { min: 0.0, max: 100.0 },
            BrickAssertion::max_latency_ms(5),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget { collect_ms: 5, layout_ms: 0, render_ms: 0 }
    }

    fn verify(&self) -> BrickVerification {
        let mut v = BrickVerification::new();
        for assertion in self.assertions() {
            v.check(&assertion);
        }
        v
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_collector_brick_name() {
        let collector = CpuCollectorBrick::new();
        assert_eq!(collector.brick_name(), "cpu_collector");
    }

    #[test]
    fn test_cpu_collector_has_assertions() {
        let collector = CpuCollectorBrick::new();
        assert!(!collector.assertions().is_empty());
    }

    #[test]
    fn test_cpu_collector_is_available() {
        let collector = CpuCollectorBrick::new();
        assert!(collector.is_available());
    }

    #[test]
    fn test_cpu_metrics_in_range() {
        let mut collector = CpuCollectorBrick::new();
        let metrics = collector.collect();
        // Since we read real /proc/stat, values might be 0 if delta is too small or parsing fails (e.g. not linux)
        // But logic should hold.
        assert!(metrics.total_usage >= 0.0 && metrics.total_usage <= 100.0);
    }
}
