# Dashboard

Complete dashboard examples with real-time monitoring, pipeline visualization, and alert systems.

## Dashboard Types

| Type | Use Case | Example |
|------|----------|---------|
| Performance | System monitoring | `dsh_performance` |
| Pipeline | Data flow tracking | `dsh_pipeline` |
| Infrastructure | Server/container status | `dsh_infrastructure` |
| Research | Experiment tracking | `dsh_research` |
| Alerts | Severity-based notifications | `dsh_alerts` |

## Performance Dashboard (DSH-004)

Real-time system metrics with threshold-based alerts:

```rust
// From dsh_performance.rs
pub struct Metric {
    pub name: String,
    pub metric_type: MetricType,
    pub values: VecDeque<MetricPoint>,
    pub threshold_warning: Option<f32>,
    pub threshold_critical: Option<f32>,
}

impl Metric {
    pub fn status(&self) -> MetricStatus {
        let current = match self.current() {
            Some(v) => v,
            None => return MetricStatus::Unknown,
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
```

Run: `cargo run --example dsh_performance`

## Data Pipeline Dashboard (DSH-006)

Visualize ETL pipeline stages and data flow:

```rust
// From dsh_pipeline.rs
pub struct Pipeline {
    pub name: String,
    stages: Vec<PipelineStage>,
}

impl Pipeline {
    pub fn bottleneck(&self) -> Option<&PipelineStage> {
        self.stages
            .iter()
            .max_by_key(|s| s.duration.unwrap_or(Duration::ZERO))
    }

    pub fn overall_drop_rate(&self) -> f32 {
        let in_count = self.total_records_in();
        let out_count = self.total_records_out();
        ((in_count - out_count) as f32 / in_count as f32) * 100.0
    }
}
```

Run: `cargo run --example dsh_pipeline`

## Infrastructure Dashboard (DSH-007)

Server and container monitoring with health scoring:

```rust
// From dsh_infrastructure.rs
pub struct InfrastructureDashboard {
    nodes: Vec<Node>,
}

impl InfrastructureDashboard {
    pub fn health_score(&self) -> f32 {
        let healthy = self.nodes_by_status(NodeStatus::Healthy).len();
        let total = self.nodes.len();
        (healthy as f32 / total as f32) * 100.0
    }

    pub fn average_utilization(&self) -> ResourceUsage {
        // Aggregates CPU, memory, disk across all nodes
    }

    pub fn needs_attention(&self) -> Vec<&Node> {
        self.nodes.iter()
            .filter(|n| n.needs_attention())
            .collect()
    }
}
```

Run: `cargo run --example dsh_infrastructure`

## Research Dashboard (DSH-009)

ML experiment tracking and comparison:

```rust
// From dsh_research.rs
pub struct ResearchDashboard {
    experiments: Vec<Experiment>,
    primary_metric: String,
    higher_is_better: bool,
}

impl ResearchDashboard {
    pub fn best_experiment(&self) -> Option<&Experiment> {
        let completed = self.by_status(ExperimentStatus::Completed);
        completed.into_iter().max_by(|a, b| {
            let val_a = a.get_metric(&self.primary_metric).unwrap_or(f32::NEG_INFINITY);
            let val_b = b.get_metric(&self.primary_metric).unwrap_or(f32::NEG_INFINITY);
            if self.higher_is_better {
                val_a.partial_cmp(&val_b).unwrap()
            } else {
                val_b.partial_cmp(&val_a).unwrap()
            }
        })
    }

    pub fn hyperparam_impact(&self, param: &str, metric: &str) -> Vec<(f32, f32)> {
        // Returns (param_value, metric_value) pairs for analysis
    }
}
```

Run: `cargo run --example dsh_research`

## Alert Dashboard (DSH-010)

Severity-based alert system with acknowledgment workflow:

```rust
// From dsh_alerts.rs
pub enum AlertSeverity {
    Info, Warning, Error, Critical
}

pub struct AlertDashboard {
    alerts: VecDeque<Alert>,
    rules: Vec<AlertRule>,
}

impl AlertDashboard {
    pub fn active_sorted(&self) -> Vec<&Alert> {
        let mut active = self.by_status(AlertStatus::Active);
        active.sort_by(|a, b| b.severity.cmp(&a.severity));
        active
    }

    pub fn acknowledge_all(&mut self, user: &str) {
        for alert in self.alerts.iter_mut() {
            if alert.status == AlertStatus::Active {
                alert.acknowledge(user);
            }
        }
    }
}
```

Run: `cargo run --example dsh_alerts`

## YAML Configuration

### Basic Dashboard Layout

```yaml
app:
  name: "Analytics Dashboard"
  root:
    type: Column
    children:
      - type: Row
        children:
          - type: DataCard
            title: "Users"
            value: "{{ metrics.users }}"
          - type: DataCard
            title: "Revenue"
            value: "{{ metrics.revenue | currency }}"
      - type: Row
        children:
          - type: Chart
            chart_type: line
            data: "{{ timeseries }}"
          - type: DataTable
            data: "{{ top_products }}"
```

### Data Sources with Refresh

```yaml
data:
  metrics:
    source: "metrics.ald"
    refresh: 60s

  live_metrics:
    source: "api/metrics"
    refresh: 5s
    on_update:
      action: animate
      duration: 300ms
```

## Responsive Grid

| Breakpoint | Columns |
|------------|---------|
| < 600px | 1 |
| 600-1200px | 2 |
| > 1200px | 3 |

## Test Coverage

| Example | Tests | Coverage |
|---------|-------|----------|
| dsh_performance | 9 | Metrics, thresholds, status |
| dsh_pipeline | 10 | Stages, bottlenecks, drop rates |
| dsh_infrastructure | 9 | Nodes, health scores, utilization |
| dsh_research | 9 | Experiments, metrics, comparison |
| dsh_alerts | 9 | Severity, acknowledgment, rules |

## Verified Test

```rust
#[test]
fn test_dashboard_health_score() {
    let mut dashboard = InfrastructureDashboard::new("Test");
    dashboard.add_node(
        Node::new("1", "a", NodeType::Server, "us")
            .with_status(NodeStatus::Healthy)
    );
    dashboard.add_node(
        Node::new("2", "b", NodeType::Server, "us")
            .with_status(NodeStatus::Warning)
    );

    // 1 healthy out of 2 = 50%
    assert!((dashboard.health_score() - 50.0).abs() < 0.01);
}
```
