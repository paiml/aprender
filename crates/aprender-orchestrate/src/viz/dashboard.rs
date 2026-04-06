//! Presentar Dashboard Configuration Generator
//!
//! Generates YAML configurations for Presentar monitoring dashboards
//! with Trueno-DB and Prometheus data source support.

use serde::{Deserialize, Serialize};

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DashboardConfig {
    /// Application metadata
    pub app: AppConfig,
    /// Data source configuration
    pub data_source: DataSourceConfig,
    /// Dashboard panels
    pub panels: Vec<Panel>,
    /// Layout configuration
    pub layout: Layout,
}

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AppConfig {
    /// Dashboard name
    pub name: String,
    /// Version
    pub version: String,
    /// Server port
    pub port: u16,
    /// Theme (light/dark)
    pub theme: String,
}

/// Data source configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataSourceConfig {
    /// Source type (trueno-db, prometheus, file)
    #[serde(rename = "type")]
    pub source_type: String,
    /// Connection path/URL
    pub path: String,
    /// Refresh interval in milliseconds
    pub refresh_interval_ms: u64,
}

/// Dashboard panel
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Panel {
    /// Panel ID
    pub id: String,
    /// Display title
    pub title: String,
    /// Panel type (timeseries, gauge, bar, stat, table)
    #[serde(rename = "type")]
    pub panel_type: String,
    /// Query to execute
    pub query: String,
    /// Y-axis label (for timeseries)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y_axis: Option<String>,
    /// Max value (for gauge)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<u64>,
    /// Unit (for stat)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// Thresholds for coloring
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thresholds: Option<Vec<Threshold>>,
}

/// Threshold for panel coloring
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Threshold {
    /// Value at which threshold activates
    pub value: u64,
    /// Color to use (red, yellow, green)
    pub color: String,
}

/// Dashboard layout
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Layout {
    /// Row configurations
    pub rows: Vec<LayoutRow>,
}

/// Layout row
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayoutRow {
    /// Row height (CSS)
    pub height: String,
    /// Panel IDs in this row
    pub panels: Vec<String>,
}

/// Builder for dashboard configuration
#[derive(Debug, Default)]
pub struct DashboardBuilder {
    name: String,
    version: String,
    port: u16,
    theme: String,
    source_type: String,
    source_path: String,
    refresh_ms: u64,
    panels: Vec<Panel>,
    rows: Vec<LayoutRow>,
}

impl DashboardBuilder {
    /// Create a new dashboard builder
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            port: 3000,
            theme: "dark".to_string(),
            source_type: "trueno-db".to_string(),
            source_path: "metrics".to_string(),
            refresh_ms: 1000,
            panels: Vec::new(),
            rows: Vec::new(),
        }
    }

    /// Set port
    #[must_use]
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set theme
    #[must_use]
    pub fn theme(mut self, theme: &str) -> Self {
        self.theme = theme.to_string();
        self
    }

    /// Set data source
    #[must_use]
    pub fn data_source(mut self, source_type: &str, path: &str) -> Self {
        self.source_type = source_type.to_string();
        self.source_path = path.to_string();
        self
    }

    /// Set refresh interval
    #[must_use]
    pub fn refresh_interval_ms(mut self, ms: u64) -> Self {
        self.refresh_ms = ms;
        self
    }

    /// Add a timeseries panel
    #[must_use]
    pub fn add_timeseries(mut self, id: &str, title: &str, query: &str, y_axis: &str) -> Self {
        self.panels.push(Panel {
            id: id.to_string(),
            title: title.to_string(),
            panel_type: "timeseries".to_string(),
            query: query.to_string(),
            y_axis: Some(y_axis.to_string()),
            max: None,
            unit: None,
            thresholds: None,
        });
        self
    }

    /// Add a gauge panel
    #[must_use]
    pub fn add_gauge(mut self, id: &str, title: &str, query: &str, max: u64) -> Self {
        self.panels.push(Panel {
            id: id.to_string(),
            title: title.to_string(),
            panel_type: "gauge".to_string(),
            query: query.to_string(),
            y_axis: None,
            max: Some(max),
            unit: None,
            thresholds: None,
        });
        self
    }

    /// Add a stat panel
    #[must_use]
    pub fn add_stat(mut self, id: &str, title: &str, query: &str, unit: &str) -> Self {
        self.panels.push(Panel {
            id: id.to_string(),
            title: title.to_string(),
            panel_type: "stat".to_string(),
            query: query.to_string(),
            y_axis: None,
            max: None,
            unit: Some(unit.to_string()),
            thresholds: None,
        });
        self
    }

    /// Add a table panel
    #[must_use]
    pub fn add_table(mut self, id: &str, title: &str, query: &str) -> Self {
        self.panels.push(Panel {
            id: id.to_string(),
            title: title.to_string(),
            panel_type: "table".to_string(),
            query: query.to_string(),
            y_axis: None,
            max: None,
            unit: None,
            thresholds: None,
        });
        self
    }

    /// Add a layout row
    #[must_use]
    pub fn add_row(mut self, height: &str, panels: &[&str]) -> Self {
        self.rows.push(LayoutRow {
            height: height.to_string(),
            panels: panels.iter().map(|s| (*s).to_string()).collect(),
        });
        self
    }

    /// Build the dashboard configuration
    #[must_use]
    pub fn build(self) -> DashboardConfig {
        DashboardConfig {
            app: AppConfig {
                name: self.name,
                version: self.version,
                port: self.port,
                theme: self.theme,
            },
            data_source: DataSourceConfig {
                source_type: self.source_type,
                path: self.source_path,
                refresh_interval_ms: self.refresh_ms,
            },
            panels: self.panels,
            layout: Layout { rows: self.rows },
        }
    }
}

/// Create a default monitoring dashboard for Realizar
#[must_use]
pub fn default_realizar_dashboard() -> DashboardConfig {
    DashboardBuilder::new("Realizar Monitoring")
        .port(3000)
        .theme("dark")
        .data_source("trueno-db", "metrics")
        .refresh_interval_ms(1000)
        .add_timeseries(
            "inference_latency",
            "Inference Latency",
            "SELECT time, p50, p95, p99 FROM realizar_metrics WHERE metric = 'inference_latency_ms'",
            "Latency (ms)",
        )
        .add_gauge(
            "throughput",
            "Token Throughput",
            "SELECT avg(tokens_per_second) FROM realizar_metrics WHERE metric = 'throughput'",
            1000,
        )
        .add_stat(
            "error_rate",
            "Error Rate",
            "SELECT (count(*) FILTER (WHERE status = 'error')) * 100.0 / count(*) FROM realizar_metrics",
            "%",
        )
        .add_table(
            "ab_tests",
            "A/B Test Results",
            "SELECT test_name, variant, requests, success_rate FROM ab_test_results",
        )
        .add_row("300px", &["inference_latency", "throughput"])
        .add_row("200px", &["error_rate", "ab_tests"])
        .build()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_builder() {
        let dashboard = DashboardBuilder::new("Test Dashboard")
            .port(8080)
            .theme("light")
            .data_source("prometheus", "localhost:9090")
            .build();

        assert_eq!(dashboard.app.name, "Test Dashboard");
        assert_eq!(dashboard.app.port, 8080);
        assert_eq!(dashboard.app.theme, "light");
        assert_eq!(dashboard.data_source.source_type, "prometheus");
        assert_eq!(dashboard.data_source.path, "localhost:9090");
    }

    #[test]
    fn test_dashboard_builder_defaults() {
        let dashboard = DashboardBuilder::new("Default").build();

        assert_eq!(dashboard.app.port, 3000);
        assert_eq!(dashboard.app.theme, "dark");
        assert_eq!(dashboard.data_source.source_type, "trueno-db");
        assert_eq!(dashboard.data_source.refresh_interval_ms, 1000);
    }

    #[test]
    fn test_add_timeseries_panel() {
        let dashboard = DashboardBuilder::new("Test")
            .add_timeseries("latency", "Latency", "SELECT * FROM metrics", "ms")
            .build();

        assert_eq!(dashboard.panels.len(), 1);
        assert_eq!(dashboard.panels[0].id, "latency");
        assert_eq!(dashboard.panels[0].panel_type, "timeseries");
        assert_eq!(dashboard.panels[0].y_axis, Some("ms".to_string()));
    }

    #[test]
    fn test_add_gauge_panel() {
        let dashboard = DashboardBuilder::new("Test")
            .add_gauge("throughput", "Throughput", "SELECT avg(tps) FROM metrics", 1000)
            .build();

        assert_eq!(dashboard.panels.len(), 1);
        assert_eq!(dashboard.panels[0].panel_type, "gauge");
        assert_eq!(dashboard.panels[0].max, Some(1000));
    }

    #[test]
    fn test_add_stat_panel() {
        let dashboard = DashboardBuilder::new("Test")
            .add_stat("errors", "Error Rate", "SELECT error_pct FROM metrics", "%")
            .build();

        assert_eq!(dashboard.panels.len(), 1);
        assert_eq!(dashboard.panels[0].panel_type, "stat");
        assert_eq!(dashboard.panels[0].unit, Some("%".to_string()));
    }

    #[test]
    fn test_add_table_panel() {
        let dashboard = DashboardBuilder::new("Test")
            .add_table("results", "Results", "SELECT * FROM results")
            .build();

        assert_eq!(dashboard.panels.len(), 1);
        assert_eq!(dashboard.panels[0].panel_type, "table");
    }

    #[test]
    fn test_layout_rows() {
        let dashboard = DashboardBuilder::new("Test")
            .add_timeseries("a", "A", "SELECT 1", "y")
            .add_gauge("b", "B", "SELECT 2", 100)
            .add_row("300px", &["a", "b"])
            .build();

        assert_eq!(dashboard.layout.rows.len(), 1);
        assert_eq!(dashboard.layout.rows[0].height, "300px");
        assert_eq!(dashboard.layout.rows[0].panels, vec!["a", "b"]);
    }

    #[test]
    fn test_default_realizar_dashboard() {
        let dashboard = default_realizar_dashboard();

        assert_eq!(dashboard.app.name, "Realizar Monitoring");
        assert_eq!(dashboard.panels.len(), 4);
        assert_eq!(dashboard.layout.rows.len(), 2);

        // Check panel types
        let panel_types: Vec<_> = dashboard.panels.iter().map(|p| p.panel_type.as_str()).collect();
        assert!(panel_types.contains(&"timeseries"));
        assert!(panel_types.contains(&"gauge"));
        assert!(panel_types.contains(&"stat"));
        assert!(panel_types.contains(&"table"));
    }

    #[test]
    fn test_dashboard_serialization() {
        let dashboard = DashboardBuilder::new("Test")
            .add_timeseries("m1", "Metric 1", "SELECT 1", "value")
            .add_row("200px", &["m1"])
            .build();

        let yaml = serde_yaml_ng::to_string(&dashboard).expect("yaml serialize failed");
        assert!(yaml.contains("name: Test"));
        assert!(yaml.contains("type: timeseries"));
        assert!(yaml.contains("height: 200px"));
    }

    #[test]
    fn test_dashboard_deserialization() {
        let yaml = r#"
app:
  name: "Deserialized"
  version: "1.0.0"
  port: 9000
  theme: light
data_source:
  type: file
  path: /tmp/metrics.db
  refresh_interval_ms: 5000
panels: []
layout:
  rows: []
"#;

        let dashboard: DashboardConfig =
            serde_yaml_ng::from_str(yaml).expect("yaml deserialize failed");
        assert_eq!(dashboard.app.name, "Deserialized");
        assert_eq!(dashboard.app.port, 9000);
        assert_eq!(dashboard.data_source.source_type, "file");
        assert_eq!(dashboard.data_source.refresh_interval_ms, 5000);
    }

    #[test]
    fn test_threshold_serialization() {
        let threshold = Threshold { value: 50, color: "yellow".to_string() };

        let json = serde_json::to_string(&threshold).expect("json serialize failed");
        assert!(json.contains("50"));
        assert!(json.contains("yellow"));
    }

    #[test]
    fn test_multiple_panels_and_rows() {
        let dashboard = DashboardBuilder::new("Complex")
            .add_timeseries("ts1", "Time Series 1", "Q1", "Y1")
            .add_timeseries("ts2", "Time Series 2", "Q2", "Y2")
            .add_gauge("g1", "Gauge 1", "Q3", 100)
            .add_stat("s1", "Stat 1", "Q4", "units")
            .add_table("t1", "Table 1", "Q5")
            .add_row("300px", &["ts1", "ts2"])
            .add_row("200px", &["g1", "s1"])
            .add_row("250px", &["t1"])
            .build();

        assert_eq!(dashboard.panels.len(), 5);
        assert_eq!(dashboard.layout.rows.len(), 3);
        assert_eq!(dashboard.layout.rows[0].panels.len(), 2);
        assert_eq!(dashboard.layout.rows[1].panels.len(), 2);
        assert_eq!(dashboard.layout.rows[2].panels.len(), 1);
    }

    #[test]
    fn test_refresh_interval() {
        let dashboard = DashboardBuilder::new("Refresh Test").refresh_interval_ms(500).build();

        assert_eq!(dashboard.data_source.refresh_interval_ms, 500);
    }

    #[test]
    fn test_data_source_types() {
        let trueno = DashboardBuilder::new("TruenoDB").data_source("trueno-db", "metrics").build();
        assert_eq!(trueno.data_source.source_type, "trueno-db");

        let prometheus =
            DashboardBuilder::new("Prometheus").data_source("prometheus", "localhost:9090").build();
        assert_eq!(prometheus.data_source.source_type, "prometheus");

        let file = DashboardBuilder::new("File").data_source("file", "/path/to/db").build();
        assert_eq!(file.data_source.source_type, "file");
    }

    #[test]
    fn test_dashboard_builder_default_trait() {
        let builder = DashboardBuilder::default();
        let dashboard = builder.build();
        // Default builder has empty name
        assert_eq!(dashboard.app.name, "");
        assert_eq!(dashboard.app.version, "");
    }

    #[test]
    fn test_dashboard_config_clone() {
        let dashboard = DashboardBuilder::new("Clone Test").build();
        let cloned = dashboard.clone();
        assert_eq!(dashboard, cloned);
    }

    #[test]
    fn test_dashboard_config_debug() {
        let dashboard = DashboardBuilder::new("Debug Test").build();
        let debug_str = format!("{:?}", dashboard);
        assert!(debug_str.contains("Debug Test"));
        assert!(debug_str.contains("DashboardConfig"));
    }

    #[test]
    fn test_app_config_clone() {
        let app = AppConfig {
            name: "Test".to_string(),
            version: "1.0".to_string(),
            port: 8080,
            theme: "dark".to_string(),
        };
        let cloned = app.clone();
        assert_eq!(app, cloned);
    }

    #[test]
    fn test_app_config_debug() {
        let app = AppConfig {
            name: "Debug".to_string(),
            version: "2.0".to_string(),
            port: 3000,
            theme: "light".to_string(),
        };
        let debug_str = format!("{:?}", app);
        assert!(debug_str.contains("Debug"));
        assert!(debug_str.contains("AppConfig"));
    }

    #[test]
    fn test_data_source_config_clone() {
        let ds = DataSourceConfig {
            source_type: "prometheus".to_string(),
            path: "localhost:9090".to_string(),
            refresh_interval_ms: 2000,
        };
        let cloned = ds.clone();
        assert_eq!(ds, cloned);
    }

    #[test]
    fn test_data_source_config_debug() {
        let ds = DataSourceConfig {
            source_type: "trueno-db".to_string(),
            path: "metrics".to_string(),
            refresh_interval_ms: 1000,
        };
        let debug_str = format!("{:?}", ds);
        assert!(debug_str.contains("trueno-db"));
    }

    #[test]
    fn test_panel_clone() {
        let panel = Panel {
            id: "test_id".to_string(),
            title: "Test Panel".to_string(),
            panel_type: "gauge".to_string(),
            query: "SELECT 1".to_string(),
            y_axis: None,
            max: Some(100),
            unit: None,
            thresholds: Some(vec![Threshold { value: 50, color: "yellow".to_string() }]),
        };
        let cloned = panel.clone();
        assert_eq!(panel, cloned);
    }

    #[test]
    fn test_panel_debug() {
        let panel = Panel {
            id: "debug_panel".to_string(),
            title: "Debug".to_string(),
            panel_type: "stat".to_string(),
            query: "Q".to_string(),
            y_axis: None,
            max: None,
            unit: Some("ms".to_string()),
            thresholds: None,
        };
        let debug_str = format!("{:?}", panel);
        assert!(debug_str.contains("debug_panel"));
    }

    #[test]
    fn test_threshold_clone() {
        let threshold = Threshold { value: 75, color: "red".to_string() };
        let cloned = threshold.clone();
        assert_eq!(threshold, cloned);
    }

    #[test]
    fn test_threshold_debug() {
        let threshold = Threshold { value: 100, color: "green".to_string() };
        let debug_str = format!("{:?}", threshold);
        assert!(debug_str.contains("100"));
        assert!(debug_str.contains("green"));
    }

    #[test]
    fn test_layout_clone() {
        let layout = Layout {
            rows: vec![LayoutRow {
                height: "300px".to_string(),
                panels: vec!["a".to_string(), "b".to_string()],
            }],
        };
        let cloned = layout.clone();
        assert_eq!(layout, cloned);
    }

    #[test]
    fn test_layout_debug() {
        let layout = Layout { rows: vec![] };
        let debug_str = format!("{:?}", layout);
        assert!(debug_str.contains("Layout"));
    }

    #[test]
    fn test_layout_row_clone() {
        let row = LayoutRow { height: "200px".to_string(), panels: vec!["panel1".to_string()] };
        let cloned = row.clone();
        assert_eq!(row, cloned);
    }

    #[test]
    fn test_layout_row_debug() {
        let row = LayoutRow { height: "150px".to_string(), panels: vec!["x".to_string()] };
        let debug_str = format!("{:?}", row);
        assert!(debug_str.contains("150px"));
    }

    #[test]
    fn test_dashboard_builder_debug() {
        let builder = DashboardBuilder::new("Builder Debug");
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("DashboardBuilder"));
        assert!(debug_str.contains("Builder Debug"));
    }

    #[test]
    fn test_dashboard_config_equality() {
        let d1 = DashboardBuilder::new("Test").port(8080).build();
        let d2 = DashboardBuilder::new("Test").port(8080).build();
        let d3 = DashboardBuilder::new("Test").port(9090).build();
        assert_eq!(d1, d2);
        assert_ne!(d1, d3);
    }

    #[test]
    fn test_panel_with_thresholds() {
        let panel = Panel {
            id: "threshold_panel".to_string(),
            title: "With Thresholds".to_string(),
            panel_type: "gauge".to_string(),
            query: "SELECT value FROM metrics".to_string(),
            y_axis: None,
            max: Some(100),
            unit: None,
            thresholds: Some(vec![
                Threshold { value: 25, color: "green".to_string() },
                Threshold { value: 50, color: "yellow".to_string() },
                Threshold { value: 75, color: "red".to_string() },
            ]),
        };
        assert_eq!(panel.thresholds.as_ref().expect("unexpected failure").len(), 3);
    }

    #[test]
    fn test_dashboard_roundtrip_serialization() {
        let original = DashboardBuilder::new("Roundtrip")
            .port(4000)
            .theme("light")
            .data_source("file", "/data/metrics.db")
            .refresh_interval_ms(2000)
            .add_timeseries("ts", "Time Series", "SELECT * FROM ts", "Y")
            .add_gauge("g", "Gauge", "SELECT val", 500)
            .add_row("250px", &["ts", "g"])
            .build();

        let yaml = serde_yaml_ng::to_string(&original).expect("yaml serialize failed");
        let deserialized: DashboardConfig =
            serde_yaml_ng::from_str(&yaml).expect("yaml deserialize failed");
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_panel_json_roundtrip() {
        let panel = Panel {
            id: "json_panel".to_string(),
            title: "JSON Test".to_string(),
            panel_type: "table".to_string(),
            query: "SELECT * FROM data".to_string(),
            y_axis: None,
            max: None,
            unit: None,
            thresholds: None,
        };
        let json = serde_json::to_string(&panel).expect("json serialize failed");
        let deserialized: Panel = serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(panel, deserialized);
    }

    #[test]
    fn test_empty_dashboard() {
        let dashboard = DashboardBuilder::new("Empty").build();
        assert!(dashboard.panels.is_empty());
        assert!(dashboard.layout.rows.is_empty());
    }

    #[test]
    fn test_dashboard_version_default() {
        let dashboard = DashboardBuilder::new("Version Test").build();
        assert_eq!(dashboard.app.version, "1.0.0");
    }
}
