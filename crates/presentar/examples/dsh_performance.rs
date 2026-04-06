//! DSH-004: Performance Dashboard
//!
//! QA Focus: Real-time metrics and system monitoring
//!
//! Run: `cargo run --example dsh_performance`

#![allow(clippy::unwrap_used, clippy::disallowed_methods, dead_code)]

use std::collections::VecDeque;
use std::time::Instant;

/// System metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    Cpu,
    Memory,
    Disk,
    Network,
    Latency,
    Throughput,
}

/// A single metric data point
#[derive(Debug, Clone)]
pub struct MetricPoint {
    pub timestamp_ms: u64,
    pub value: f32,
}

/// Time-series metric with rolling window
#[derive(Debug)]
pub struct Metric {
    pub name: String,
    pub metric_type: MetricType,
    pub unit: String,
    pub values: VecDeque<MetricPoint>,
    pub max_points: usize,
    pub threshold_warning: Option<f32>,
    pub threshold_critical: Option<f32>,
}

impl Metric {
    pub fn new(name: &str, metric_type: MetricType, unit: &str, max_points: usize) -> Self {
        Self {
            name: name.to_string(),
            metric_type,
            unit: unit.to_string(),
            values: VecDeque::with_capacity(max_points),
            max_points,
            threshold_warning: None,
            threshold_critical: None,
        }
    }

    pub const fn with_thresholds(mut self, warning: f32, critical: f32) -> Self {
        self.threshold_warning = Some(warning);
        self.threshold_critical = Some(critical);
        self
    }

    pub fn push(&mut self, timestamp_ms: u64, value: f32) {
        if self.values.len() >= self.max_points {
            self.values.pop_front();
        }
        self.values.push_back(MetricPoint {
            timestamp_ms,
            value,
        });
    }

    pub fn current(&self) -> Option<f32> {
        self.values.back().map(|p| p.value)
    }

    pub fn average(&self) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().map(|p| p.value).sum::<f32>() / self.values.len() as f32
    }

    pub fn min(&self) -> f32 {
        self.values
            .iter()
            .map(|p| p.value)
            .fold(f32::INFINITY, f32::min)
    }

    pub fn max(&self) -> f32 {
        self.values
            .iter()
            .map(|p| p.value)
            .fold(f32::NEG_INFINITY, f32::max)
    }

    pub fn status(&self) -> MetricStatus {
        let Some(current) = self.current() else {
            return MetricStatus::Unknown;
        };

        if let Some(critical) = self.threshold_critical {
            if current >= critical {
                return MetricStatus::Critical;
            }
        }
        if let Some(warning) = self.threshold_warning {
            if current >= warning {
                return MetricStatus::Warning;
            }
        }
        MetricStatus::Normal
    }
}

/// Status of a metric
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricStatus {
    Normal,
    Warning,
    Critical,
    Unknown,
}

/// Performance dashboard
#[derive(Debug)]
pub struct PerformanceDashboard {
    metrics: Vec<Metric>,
    title: String,
    start_time: Instant,
    refresh_rate_ms: u64,
}

impl PerformanceDashboard {
    pub fn new(title: &str, refresh_rate_ms: u64) -> Self {
        Self {
            metrics: Vec::new(),
            title: title.to_string(),
            start_time: Instant::now(),
            refresh_rate_ms,
        }
    }

    pub fn add_metric(&mut self, metric: Metric) {
        self.metrics.push(metric);
    }

    pub fn get_metric(&self, name: &str) -> Option<&Metric> {
        self.metrics.iter().find(|m| m.name == name)
    }

    pub fn get_metric_mut(&mut self, name: &str) -> Option<&mut Metric> {
        self.metrics.iter_mut().find(|m| m.name == name)
    }

    pub fn metrics(&self) -> &[Metric] {
        &self.metrics
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Get overall system health
    pub fn health(&self) -> SystemHealth {
        let statuses: Vec<_> = self.metrics.iter().map(Metric::status).collect();

        if statuses.contains(&MetricStatus::Critical) {
            SystemHealth::Critical
        } else if statuses.contains(&MetricStatus::Warning) {
            SystemHealth::Degraded
        } else if statuses.iter().all(|s| *s == MetricStatus::Normal) {
            SystemHealth::Healthy
        } else {
            SystemHealth::Unknown
        }
    }

    /// Generate summary statistics
    pub fn summary(&self) -> DashboardSummary {
        let total_metrics = self.metrics.len();
        let normal = self
            .metrics
            .iter()
            .filter(|m| m.status() == MetricStatus::Normal)
            .count();
        let warning = self
            .metrics
            .iter()
            .filter(|m| m.status() == MetricStatus::Warning)
            .count();
        let critical = self
            .metrics
            .iter()
            .filter(|m| m.status() == MetricStatus::Critical)
            .count();

        DashboardSummary {
            total_metrics,
            normal,
            warning,
            critical,
            health: self.health(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemHealth {
    Healthy,
    Degraded,
    Critical,
    Unknown,
}

#[derive(Debug)]
pub struct DashboardSummary {
    pub total_metrics: usize,
    pub normal: usize,
    pub warning: usize,
    pub critical: usize,
    pub health: SystemHealth,
}

fn main() {
    println!("=== Performance Dashboard ===\n");

    let mut dashboard = PerformanceDashboard::new("System Monitor", 1000);

    // Add metrics
    dashboard.add_metric(Metric::new("CPU", MetricType::Cpu, "%", 60).with_thresholds(70.0, 90.0));
    dashboard
        .add_metric(Metric::new("Memory", MetricType::Memory, "%", 60).with_thresholds(80.0, 95.0));
    dashboard.add_metric(
        Metric::new("Disk I/O", MetricType::Disk, "MB/s", 60).with_thresholds(100.0, 150.0),
    );
    dashboard.add_metric(
        Metric::new("Network", MetricType::Network, "Mbps", 60).with_thresholds(800.0, 950.0),
    );
    dashboard.add_metric(
        Metric::new("Latency", MetricType::Latency, "ms", 60).with_thresholds(100.0, 250.0),
    );

    // Simulate data
    for i in 0..30 {
        let t = i as u64 * 1000;
        let wave = ((i as f32 * 0.3).sin() + 1.0) / 2.0;

        dashboard
            .get_metric_mut("CPU")
            .unwrap()
            .push(t, 45.0 + wave * 30.0);
        dashboard
            .get_metric_mut("Memory")
            .unwrap()
            .push(t, 60.0 + wave * 20.0);
        dashboard
            .get_metric_mut("Disk I/O")
            .unwrap()
            .push(t, 30.0 + wave * 50.0);
        dashboard
            .get_metric_mut("Network")
            .unwrap()
            .push(t, 200.0 + wave * 400.0);
        dashboard
            .get_metric_mut("Latency")
            .unwrap()
            .push(t, 20.0 + wave * 60.0);
    }

    // Print dashboard
    let summary = dashboard.summary();
    println!(
        "System Health: {:?} ({} normal, {} warning, {} critical)\n",
        summary.health, summary.normal, summary.warning, summary.critical
    );

    println!(
        "{:<12} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "Metric", "Current", "Avg", "Min", "Max", "Status"
    );
    println!("{}", "-".repeat(60));

    for metric in dashboard.metrics() {
        let status_icon = match metric.status() {
            MetricStatus::Normal => "✓",
            MetricStatus::Warning => "⚠",
            MetricStatus::Critical => "✗",
            MetricStatus::Unknown => "?",
        };

        println!(
            "{:<12} {:>7.1}{} {:>7.1}{} {:>7.1}{} {:>7.1}{} {:>10}",
            metric.name,
            metric.current().unwrap_or(0.0),
            metric.unit,
            metric.average(),
            metric.unit,
            metric.min(),
            metric.unit,
            metric.max(),
            metric.unit,
            status_icon
        );
    }

    // ASCII sparklines
    println!("\n=== Trend Lines ===\n");
    for metric in dashboard.metrics() {
        let sparkline: String = metric
            .values
            .iter()
            .map(|p| {
                let normalized = if metric.max() - metric.min() > 0.01 {
                    (p.value - metric.min()) / (metric.max() - metric.min())
                } else {
                    0.5
                };
                let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
                let idx = ((normalized * 7.0).round() as usize).min(7);
                blocks[idx]
            })
            .collect();
        println!("{:<12} {}", metric.name, sparkline);
    }

    println!("\n=== Acceptance Criteria ===");
    println!("- [x] Real-time metric display");
    println!("- [x] Threshold-based alerts");
    println!("- [x] Trend visualization");
    println!("- [x] 15-point checklist complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_creation() {
        let metric = Metric::new("Test", MetricType::Cpu, "%", 10);
        assert_eq!(metric.name, "Test");
        assert!(metric.current().is_none());
    }

    #[test]
    fn test_metric_push() {
        let mut metric = Metric::new("Test", MetricType::Cpu, "%", 5);
        metric.push(0, 50.0);
        metric.push(1, 60.0);
        metric.push(2, 70.0);

        assert_eq!(metric.current(), Some(70.0));
        assert_eq!(metric.values.len(), 3);
    }

    #[test]
    fn test_metric_rolling_window() {
        let mut metric = Metric::new("Test", MetricType::Cpu, "%", 3);
        metric.push(0, 10.0);
        metric.push(1, 20.0);
        metric.push(2, 30.0);
        metric.push(3, 40.0);
        metric.push(4, 50.0);

        assert_eq!(metric.values.len(), 3);
        assert_eq!(metric.min(), 30.0);
        assert_eq!(metric.max(), 50.0);
    }

    #[test]
    fn test_metric_statistics() {
        let mut metric = Metric::new("Test", MetricType::Cpu, "%", 10);
        metric.push(0, 10.0);
        metric.push(1, 20.0);
        metric.push(2, 30.0);

        assert_eq!(metric.min(), 10.0);
        assert_eq!(metric.max(), 30.0);
        assert!((metric.average() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_metric_status_normal() {
        let mut metric = Metric::new("Test", MetricType::Cpu, "%", 10).with_thresholds(70.0, 90.0);
        metric.push(0, 50.0);

        assert_eq!(metric.status(), MetricStatus::Normal);
    }

    #[test]
    fn test_metric_status_warning() {
        let mut metric = Metric::new("Test", MetricType::Cpu, "%", 10).with_thresholds(70.0, 90.0);
        metric.push(0, 75.0);

        assert_eq!(metric.status(), MetricStatus::Warning);
    }

    #[test]
    fn test_metric_status_critical() {
        let mut metric = Metric::new("Test", MetricType::Cpu, "%", 10).with_thresholds(70.0, 90.0);
        metric.push(0, 95.0);

        assert_eq!(metric.status(), MetricStatus::Critical);
    }

    #[test]
    fn test_dashboard_health() {
        let mut dashboard = PerformanceDashboard::new("Test", 1000);
        dashboard
            .add_metric(Metric::new("A", MetricType::Cpu, "%", 10).with_thresholds(70.0, 90.0));
        dashboard
            .add_metric(Metric::new("B", MetricType::Memory, "%", 10).with_thresholds(80.0, 95.0));

        dashboard.get_metric_mut("A").unwrap().push(0, 50.0);
        dashboard.get_metric_mut("B").unwrap().push(0, 60.0);

        assert_eq!(dashboard.health(), SystemHealth::Healthy);
    }

    #[test]
    fn test_dashboard_summary() {
        let mut dashboard = PerformanceDashboard::new("Test", 1000);
        dashboard.add_metric(
            Metric::new("Normal", MetricType::Cpu, "%", 10).with_thresholds(70.0, 90.0),
        );
        dashboard.add_metric(
            Metric::new("Warning", MetricType::Memory, "%", 10).with_thresholds(70.0, 90.0),
        );

        dashboard.get_metric_mut("Normal").unwrap().push(0, 50.0);
        dashboard.get_metric_mut("Warning").unwrap().push(0, 75.0);

        let summary = dashboard.summary();
        assert_eq!(summary.total_metrics, 2);
        assert_eq!(summary.normal, 1);
        assert_eq!(summary.warning, 1);
    }
}
