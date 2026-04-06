//! Dashboard Module (ENT-043)
//!
//! Terminal visualization using trueno-viz patterns.
//! Displays training metrics in real-time ASCII format.

use super::MetricsSummary;

/// Dashboard configuration
#[derive(Debug, Clone)]
pub struct DashboardConfig {
    /// Width in characters
    pub width: usize,
    /// Height in characters
    pub height: usize,
    /// Refresh interval in milliseconds
    pub refresh_ms: u64,
    /// Show ASCII mode (for SSH)
    pub ascii_mode: bool,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self { width: 80, height: 24, refresh_ms: 1000, ascii_mode: true }
    }
}

/// Training dashboard for real-time visualization
pub struct Dashboard {
    config: DashboardConfig,
    history: Vec<MetricsSummary>,
    max_history: usize,
}

impl Dashboard {
    /// Create a new dashboard with default config
    pub fn new() -> Self {
        Self::with_config(DashboardConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: DashboardConfig) -> Self {
        Self { config, history: Vec::new(), max_history: 100 }
    }

    /// Update with new metrics
    pub fn update(&mut self, summary: MetricsSummary) {
        self.history.push(summary);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Render to ASCII string
    pub fn render_ascii(&self) -> String {
        let mut output = String::new();
        output.push_str(&"═".repeat(self.config.width));
        output.push('\n');
        output.push_str("  TRAINING MONITOR\n");
        output.push_str(&"─".repeat(self.config.width));
        output.push('\n');

        if let Some(latest) = self.history.last() {
            for (metric, stats) in latest {
                output.push_str(&format!(
                    "  {:<15} mean={:.4} std={:.4} min={:.4} max={:.4}\n",
                    metric.as_str(),
                    stats.mean,
                    stats.std,
                    stats.min,
                    stats.max
                ));
            }
        } else {
            output.push_str("  No metrics recorded yet\n");
        }

        output.push_str(&"═".repeat(self.config.width));
        output.push('\n');
        output
    }

    /// Render simple sparkline for a metric
    pub fn sparkline(&self, metric: &super::Metric) -> String {
        let chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let values: Vec<f64> =
            self.history.iter().filter_map(|s| s.get(metric).map(|st| st.mean)).collect();

        if values.is_empty() {
            return String::new();
        }

        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        if range == 0.0 {
            return chars[4].to_string().repeat(values.len());
        }

        values
            .iter()
            .map(|v| {
                let idx = (((v - min) / range) * 7.0).round() as usize;
                chars[idx.min(7)]
            })
            .collect()
    }
}

impl Default for Dashboard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::{Metric, MetricStats};
    use std::collections::HashMap;

    #[test]
    fn test_dashboard_new() {
        let dashboard = Dashboard::new();
        assert_eq!(dashboard.history.len(), 0);
    }

    #[test]
    fn test_dashboard_update() {
        let mut dashboard = Dashboard::new();
        let mut summary = HashMap::new();
        summary.insert(
            Metric::Loss,
            MetricStats {
                count: 1,
                mean: 0.5,
                std: 0.0,
                min: 0.5,
                max: 0.5,
                sum: 0.5,
                has_nan: false,
                has_inf: false,
            },
        );
        dashboard.update(summary);
        assert_eq!(dashboard.history.len(), 1);
    }

    #[test]
    fn test_render_ascii_empty() {
        let dashboard = Dashboard::new();
        let output = dashboard.render_ascii();
        assert!(output.contains("No metrics"));
    }

    #[test]
    fn test_render_ascii_with_data() {
        let mut dashboard = Dashboard::new();
        let mut summary = HashMap::new();
        summary.insert(
            Metric::Loss,
            MetricStats {
                count: 10,
                mean: 0.25,
                std: 0.1,
                min: 0.1,
                max: 0.5,
                sum: 2.5,
                has_nan: false,
                has_inf: false,
            },
        );
        dashboard.update(summary);
        let output = dashboard.render_ascii();
        assert!(output.contains("loss"));
        assert!(output.contains("0.25"));
    }

    #[test]
    fn test_sparkline() {
        let mut dashboard = Dashboard::new();

        // Add decreasing loss values
        for i in 0..10 {
            let mut summary = HashMap::new();
            summary.insert(
                Metric::Loss,
                MetricStats {
                    count: 1,
                    mean: 1.0 - (f64::from(i) * 0.1),
                    std: 0.0,
                    min: 0.0,
                    max: 1.0,
                    sum: 0.0,
                    has_nan: false,
                    has_inf: false,
                },
            );
            dashboard.update(summary);
        }

        let spark = dashboard.sparkline(&Metric::Loss);
        assert_eq!(spark.chars().count(), 10);
    }

    #[test]
    fn test_dashboard_config_default() {
        let config = DashboardConfig::default();
        assert_eq!(config.width, 80);
        assert_eq!(config.height, 24);
        assert_eq!(config.refresh_ms, 1000);
        assert!(config.ascii_mode);
    }

    #[test]
    fn test_dashboard_with_config() {
        let config = DashboardConfig { width: 100, height: 30, refresh_ms: 500, ascii_mode: false };
        let dashboard = Dashboard::with_config(config.clone());
        assert_eq!(dashboard.config.width, 100);
    }

    #[test]
    fn test_sparkline_empty() {
        let dashboard = Dashboard::new();
        let spark = dashboard.sparkline(&Metric::Loss);
        assert!(spark.is_empty());
    }

    #[test]
    fn test_dashboard_default() {
        let dashboard = Dashboard::default();
        assert!(dashboard.history.is_empty());
    }
}
