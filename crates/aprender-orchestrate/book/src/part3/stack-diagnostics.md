# Stack Diagnostics & ML Insights

The Stack Diagnostics module provides ML-driven insights for monitoring PAIML stack health, implementing Toyota Way principles for observability.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  SOVEREIGN AI STACK HEALTH DASHBOARD                    │
│                  Timestamp: 2024-12-07 15:30:45                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ANDON STATUS: 🟢 All systems healthy                                   │
│                                                                         │
│  STACK SUMMARY                                                          │
│  Total Components:    24                                                │
│  Healthy:             22 (92%)                                          │
│  Warnings:             2 (8%)                                           │
│  Critical:             0 (0%)                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Toyota Way Principles

The diagnostics system implements several Toyota Production System concepts:

| Principle | Implementation |
|-----------|----------------|
| **Mieruka** | ASCII dashboards make health visible at a glance |
| **Jidoka** | ML anomaly detection surfaces issues automatically |
| **Genchi Genbutsu** | Evidence-based diagnosis from actual dependency data |
| **Andon** | Red/Yellow/Green status with stop-the-line alerts |
| **Yokoten** | Cross-component insight sharing via knowledge graph |

## Andon Status System

The Andon system provides visual health indicators:

```rust
use batuta::{HealthStatus, QualityGrade};

// Status from quality grade
let status = HealthStatus::from_grade(QualityGrade::A);
assert_eq!(status, HealthStatus::Green);

// Visual indicators
println!("{} Green  - All systems healthy", HealthStatus::Green.icon());
println!("{} Yellow - Attention needed", HealthStatus::Yellow.icon());
println!("{} Red    - Stop-the-line", HealthStatus::Red.icon());
```

### Status Transitions

| Quality Grade | Health Status | Action |
|---------------|---------------|--------|
| A+, A | 🟢 Green | Normal operation |
| A-, B+ | 🟡 Yellow | Attention needed |
| B, C, D, F | 🔴 Red | Stop-the-line |

## Component Metrics

Each stack component tracks key quality metrics:

```rust
use batuta::{ComponentMetrics, ComponentNode, QualityStackLayer as StackLayer};

// Create component with metrics
let mut node = ComponentNode::new("trueno", "0.7.4", StackLayer::Compute);
node.metrics = ComponentMetrics {
    demo_score: 95.5,      // PMAT quality score
    coverage: 92.0,         // Test coverage %
    mutation_score: 85.0,   // Mutation testing kill rate
    complexity_avg: 4.2,    // Cyclomatic complexity
    satd_count: 3,          // Self-Admitted Technical Debt
    dead_code_pct: 0.5,     // Dead code percentage
    grade: QualityGrade::APlus,
};
node.update_health();
```

## Graph Analytics

The system computes graph-level metrics for dependency analysis:

### PageRank

Identifies critical components based on dependency centrality:

```rust
use batuta::StackDiagnostics;

let mut diag = StackDiagnostics::new();
// Add components...

let metrics = diag.compute_metrics()?;

// Top components by PageRank
for (name, score) in metrics.top_by_pagerank(5) {
    println!("{}: {:.3}", name, score);
}
```

### Betweenness Centrality

Finds bottleneck components that many paths pass through:

```rust
// Find components with high betweenness (potential bottlenecks)
let bottlenecks = metrics.bottlenecks(0.5);
for name in bottlenecks {
    println!("Bottleneck: {}", name);
}
```

### Depth Analysis

Measures dependency chain depth from root nodes:

```rust
for (name, depth) in &metrics.depth_map {
    println!("{} at depth {}", name, depth);
}
println!("Maximum depth: {}", metrics.max_depth);
```

## ML Anomaly Detection

### Isolation Forest

The Isolation Forest algorithm detects anomalies by measuring isolation:

```rust
use batuta::IsolationForest;

let mut forest = IsolationForest::new(100, 256, 42);

// Fit on component metrics
let data = vec![
    vec![90.0, 85.0, 80.0, 5.0],  // Normal
    vec![88.0, 82.0, 78.0, 5.5],  // Normal
    vec![30.0, 20.0, 15.0, 25.0], // Anomaly!
];
forest.fit(&data);

// Score data points (higher = more anomalous)
let scores = forest.score(&data);
```

### Detecting Anomalies in Stack

```rust
// Detect anomalies in component metrics
let anomalies = forest.detect_anomalies(&diagnostics, 0.5);

for anomaly in &anomalies {
    println!("{}: {} (score: {:.3})",
        anomaly.component,
        anomaly.description,
        anomaly.score
    );

    if let Some(rec) = &anomaly.recommendation {
        println!("  Recommendation: {}", rec);
    }
}
```

### Anomaly Categories

| Category | Trigger | Example |
|----------|---------|---------|
| `QualityRegression` | Demo score < 70 | "Score dropped from 90 to 65" |
| `CoverageDrop` | Coverage < 50% | "Coverage at 45% (target: 80%)" |
| `ComplexityIncrease` | Avg complexity > 15 | "Complexity grew to 18.5" |
| `DependencyRisk` | Dead code > 10% | "15% dead code detected" |
| `BuildTimeSpike` | Build time increase | "Build time +40%" |

## Error Forecasting

Predict future error trends using exponential smoothing:

```rust
use batuta::ErrorForecaster;

let mut forecaster = ErrorForecaster::new(0.3);

// Add historical observations
forecaster.observe(5.0);
forecaster.observe(8.0);
forecaster.observe(12.0);
forecaster.observe(10.0);

// Forecast next 4 periods
let forecast = forecaster.forecast(4);
println!("Predicted errors: {:?}", forecast);

// Check accuracy metrics
let metrics = forecaster.error_metrics();
println!("MAE: {:.2}", metrics.mae);
println!("RMSE: {:.2}", metrics.rmse);
```

## Dashboard Rendering

Generate ASCII dashboards for terminal display:

```rust
use batuta::{render_dashboard, StackDiagnostics};

let diag = StackDiagnostics::new();
// Add components and anomalies...

let output = render_dashboard(&diag);
println!("{}", output);
```

## Running the Demo

```bash
cargo run --example stack_diagnostics_demo --features native
```

This demonstrates:

1. **Phase 1**: Andon Status Board
2. **Phase 2**: Component Metrics
3. **Phase 3**: Graph Analytics
4. **Phase 4**: Isolation Forest Anomaly Detection
5. **Phase 5**: Error Forecasting
6. **Phase 6**: Dashboard Rendering

## Integration with CLI

The diagnostics system integrates with `batuta stack`:

```bash
# Stack health dashboard
batuta stack status --diagnostics

# Run anomaly detection
batuta stack check --ml

# Forecast error trends
batuta stack forecast --days 7
```

## Best Practices

1. **Regular Monitoring**: Run diagnostics as part of CI/CD
2. **Threshold Tuning**: Adjust anomaly threshold based on stack maturity
3. **Evidence Collection**: Always include evidence in anomaly reports
4. **Action Items**: Provide actionable recommendations

## See Also

- [PAIML Ecosystem Comparison](../appendix/comparison.md)
- [PMAT Integration](./pmat.md)
- [Toyota Way Principles](../part1/toyota-way.md)
