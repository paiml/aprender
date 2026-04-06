//! Stack Diagnostics & ML Anomaly Detection Demo
//!
//! Demonstrates the diagnostics system for monitoring PAIML stack health
//! using Toyota Way principles (Mieruka, Jidoka, Genchi Genbutsu, Andon).
//!
//! ## Features
//!
//! - **Andon Status Board**: Visual health indicators (Green/Yellow/Red)
//! - **Graph Analytics**: PageRank, Betweenness Centrality, Depth analysis
//! - **Isolation Forest**: ML-based anomaly detection
//! - **Error Forecasting**: Time series prediction for proactive alerts
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example stack_diagnostics_demo --features native
//! ```

#[cfg(feature = "native")]
use batuta::{
    render_dashboard, Anomaly, AnomalyCategory, ComponentMetrics, ComponentNode, ErrorForecaster,
    HealthStatus, IsolationForest, QualityGrade, QualityStackLayer as StackLayer, StackDiagnostics,
};

#[cfg(feature = "native")]
fn main() -> anyhow::Result<()> {
    println!("===============================================================");
    println!("     Stack Diagnostics & ML Anomaly Detection - Demo");
    println!("===============================================================\n");

    // =========================================================================
    // Phase 1: Andon Status Board
    // =========================================================================
    println!("+-------------------------------------------------------------+");
    println!("| Phase 1: Andon Status Board (Visual Health Control)         |");
    println!("+-------------------------------------------------------------+\n");

    demo_andon_status();

    // =========================================================================
    // Phase 2: Component Metrics
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 2: Component Metrics                                  |");
    println!("+-------------------------------------------------------------+\n");

    demo_component_metrics();

    // =========================================================================
    // Phase 3: Graph Analytics
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 3: Graph Analytics (PageRank, Betweenness)            |");
    println!("+-------------------------------------------------------------+\n");

    demo_graph_analytics()?;

    // =========================================================================
    // Phase 4: Isolation Forest Anomaly Detection
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 4: Isolation Forest Anomaly Detection                 |");
    println!("+-------------------------------------------------------------+\n");

    demo_isolation_forest();

    // =========================================================================
    // Phase 5: Error Forecasting
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 5: Error Forecasting (Exponential Smoothing)          |");
    println!("+-------------------------------------------------------------+\n");

    demo_error_forecasting();

    // =========================================================================
    // Phase 6: Dashboard Rendering
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 6: Dashboard Rendering (ASCII Mieruka)                |");
    println!("+-------------------------------------------------------------+\n");

    demo_dashboard();

    println!("\n===============================================================");
    println!("     Stack Diagnostics demo completed!");
    println!("===============================================================\n");

    Ok(())
}

#[cfg(feature = "native")]
fn demo_andon_status() {
    println!("  Toyota Andon System - Visual Health Indicators:");
    println!();
    println!("  {} Green  - All systems healthy, normal operation", HealthStatus::Green.icon());
    println!("  {} Yellow - Attention needed, warnings present", HealthStatus::Yellow.icon());
    println!("  {} Red    - Critical issues, stop-the-line", HealthStatus::Red.icon());
    println!("  {} Unknown- Not yet analyzed", HealthStatus::Unknown.icon());
    println!();
    println!("  Health Status from Quality Grades:");
    println!();
    println!(
        "    A+/A  → {} Green  (Release ready)",
        HealthStatus::from_grade(QualityGrade::A).icon()
    );
    println!(
        "    A-/B+ → {} Yellow (Needs attention)",
        HealthStatus::from_grade(QualityGrade::AMinus).icon()
    );
    println!("    B-/C  → {} Red    (Blocked)", HealthStatus::from_grade(QualityGrade::B).icon());
    println!();
    println!("  ASCII Terminal Symbols (for CI/CD logs):");
    println!(
        "    Green:  {}  Yellow: {}  Red: {}  Unknown: {}",
        HealthStatus::Green.symbol(),
        HealthStatus::Yellow.symbol(),
        HealthStatus::Red.symbol(),
        HealthStatus::Unknown.symbol()
    );
}

#[cfg(feature = "native")]
fn demo_component_metrics() {
    println!("  Component Quality Metrics:");
    println!();

    // Create sample components
    let components = [
        ("trueno", 95.5, 92.0, 85.0, 4.2, QualityGrade::APlus),
        ("aprender", 92.0, 88.5, 82.0, 5.1, QualityGrade::A),
        ("batuta", 88.0, 85.0, 78.0, 6.3, QualityGrade::AMinus),
        ("weak-crate", 65.0, 45.0, 30.0, 12.5, QualityGrade::C),
    ];

    println!("  ┌────────────────┬───────────┬──────────┬──────────┬───────────┬───────┐");
    println!("  │ Component      │ Demo Score│ Coverage │ Mutation │ Complexity│ Grade │");
    println!("  ├────────────────┼───────────┼──────────┼──────────┼───────────┼───────┤");

    for (name, demo, cov, mut_score, complexity, grade) in components {
        let status = HealthStatus::from_grade(grade);
        println!(
            "  │ {:14} │ {:>8.1} │ {:>7.1}% │ {:>7.1}% │ {:>8.1} │ {} {:3} │",
            name,
            demo,
            cov,
            mut_score,
            complexity,
            status.icon(),
            grade.symbol()
        );
    }

    println!("  └────────────────┴───────────┴──────────┴──────────┴───────────┴───────┘");
    println!();
    println!("  Metrics Guide:");
    println!("    Demo Score:  PMAT normalized quality score (0-100)");
    println!("    Coverage:    Test coverage percentage");
    println!("    Mutation:    Mutation testing kill rate");
    println!("    Complexity:  Average cyclomatic complexity");
}

#[cfg(feature = "native")]
fn demo_graph_analytics() -> anyhow::Result<()> {
    let mut diag = StackDiagnostics::new();

    // Add stack components
    let components = [
        ("trueno", StackLayer::Compute, 95.0),
        ("aprender", StackLayer::Ml, 92.0),
        ("realizar", StackLayer::Training, 88.0),
        ("batuta", StackLayer::Orchestration, 90.0),
        ("depyler", StackLayer::Transpilers, 87.0),
    ];

    for (name, layer, score) in components {
        let mut node = ComponentNode::new(name, "1.0.0", layer);
        node.metrics = ComponentMetrics::with_demo_score(score);
        node.update_health();
        diag.add_component(node);
    }

    // Compute metrics
    let metrics = diag.compute_metrics()?;

    println!("  Graph-Level Statistics:");
    println!();
    println!("    Total Nodes:     {}", metrics.total_nodes);
    println!("    Total Edges:     {}", metrics.total_edges);
    println!("    Graph Density:   {:.4}", metrics.density);
    println!("    Average Degree:  {:.2}", metrics.avg_degree);
    println!("    Max Depth:       {}", metrics.max_depth);
    println!();
    println!("  PageRank Scores (Importance):");
    println!();

    let top = metrics.top_by_pagerank(5);
    for (name, score) in &top {
        let bar_len = (*score * 50.0).round() as usize;
        let bar: String = "█".repeat(bar_len.min(50));
        println!("    {:12} │ {:5.3} │ {}", name, score, bar);
    }

    println!();
    println!("  Depth from Root:");
    println!();
    for (name, depth) in &metrics.depth_map {
        let indent = "  ".repeat(*depth as usize);
        println!("    {}├── {} (depth {})", indent, name, depth);
    }

    Ok(())
}

#[cfg(feature = "native")]
fn demo_isolation_forest() {
    println!("  Isolation Forest Algorithm:");
    println!();
    println!("  The Isolation Forest detects anomalies by measuring how");
    println!("  easily data points can be isolated using random splits.");
    println!();
    println!("  - Anomalies require fewer splits (shorter path length)");
    println!("  - Normal points require more splits (longer path length)");
    println!("  - Score ∈ [0, 1]: Higher = more anomalous");
    println!();

    // Train on sample data
    let mut forest = IsolationForest::new(50, 64, 42);

    // Normal data: high quality scores
    let mut data = Vec::new();
    for _ in 0..20 {
        data.push(vec![90.0, 85.0, 80.0, 5.0, 2.0, 1.0]);
    }
    // Add an anomaly: poor quality
    data.push(vec![30.0, 20.0, 15.0, 25.0, 50.0, 30.0]);

    forest.fit(&data);
    let scores = forest.score(&data);

    println!("  Sample Anomaly Detection:");
    println!();
    println!("  ┌─────────────────┬────────────┬─────────────────────────────────┐");
    println!("  │ Data Point      │ Score      │ Interpretation                  │");
    println!("  ├─────────────────┼────────────┼─────────────────────────────────┤");

    for (i, &score) in scores.iter().enumerate() {
        let label = if i == 20 { "Anomaly (low quality)" } else { "Normal (high quality)" };
        let interp = if score > 0.6 {
            "🔴 Anomalous"
        } else if score > 0.5 {
            "🟡 Borderline"
        } else {
            "🟢 Normal"
        };
        if i == 0 || i == 10 || i == 20 {
            println!("  │ {:15} │ {:>10.4} │ {:31} │", label, score, interp);
        }
    }

    println!("  └─────────────────┴────────────┴─────────────────────────────────┘");
    println!();
    println!("  Anomaly Categories:");
    println!();
    println!(
        "    {} - Quality score significantly below threshold",
        AnomalyCategory::QualityRegression
    );
    println!("    {} - Test coverage dropped unexpectedly", AnomalyCategory::CoverageDrop);
    println!("    {} - Build time increased significantly", AnomalyCategory::BuildTimeSpike);
    println!("    {} - High-risk dependency change", AnomalyCategory::DependencyRisk);
    println!("    {} - Code complexity grew too high", AnomalyCategory::ComplexityIncrease);
}

#[cfg(feature = "native")]
fn demo_error_forecasting() {
    println!("  Exponential Smoothing Forecaster:");
    println!();
    println!("  Formula: L_t = α·Y_t + (1-α)·L_{{t-1}}");
    println!("  Where: α=0.3 (smoothing parameter), Y=observation, L=level");
    println!();

    let mut forecaster = ErrorForecaster::new(0.3);

    // Simulate historical error counts
    let observations = [5.0, 8.0, 12.0, 10.0, 15.0, 18.0, 14.0, 20.0];

    println!("  Historical Error Counts:");
    println!();
    for (i, &obs) in observations.iter().enumerate() {
        forecaster.observe(obs);
        println!(
            "    Week {}: {:>5.1} errors  (smoothed level: {:>5.2})",
            i + 1,
            obs,
            forecaster.current_level()
        );
    }

    println!();
    println!("  Forecast (next 4 weeks):");
    let forecast = forecaster.forecast(4);
    for (i, &f) in forecast.iter().enumerate() {
        println!("    Week {}: {:>5.2} errors (predicted)", i + 9, f);
    }

    println!();
    let metrics = forecaster.error_metrics();
    println!("  Forecast Accuracy Metrics:");
    println!("    MAE  (Mean Absolute Error):    {:>6.2}", metrics.mae);
    println!("    RMSE (Root Mean Square Error): {:>6.2}", metrics.rmse);
    println!("    MAPE (Mean Absolute % Error):  {:>6.2}%", metrics.mape);
}

#[cfg(feature = "native")]
fn demo_dashboard() {
    let mut diag = StackDiagnostics::new();

    // Add sample components
    let mut trueno = ComponentNode::new("trueno", "0.7.4", StackLayer::Compute);
    trueno.metrics = ComponentMetrics::with_demo_score(95.5);
    trueno.health = HealthStatus::Green;
    diag.add_component(trueno);

    let mut aprender = ComponentNode::new("aprender", "0.9.0", StackLayer::Ml);
    aprender.metrics = ComponentMetrics::with_demo_score(92.0);
    aprender.health = HealthStatus::Green;
    diag.add_component(aprender);

    let mut weak = ComponentNode::new("weak-crate", "0.1.0", StackLayer::Quality);
    weak.metrics = ComponentMetrics::with_demo_score(65.0);
    weak.health = HealthStatus::Red;
    diag.add_component(weak);

    // Add anomaly
    diag.add_anomaly(
        Anomaly::new(
            "weak-crate",
            0.85,
            AnomalyCategory::QualityRegression,
            "Quality score 65.0 is below A- threshold",
        )
        .with_evidence("Demo score dropped from 85.0 to 65.0")
        .with_evidence("Coverage at 45% (below 80% target)")
        .with_recommendation("Run `pmat demo-score` and address top issues"),
    );

    println!("  ASCII Dashboard (Mieruka - Visual Control):");
    println!();
    let dashboard = render_dashboard(&diag);
    println!("{}", dashboard);

    println!("  Dashboard Features:");
    println!("    - Real-time Andon status");
    println!("    - Component health counts");
    println!("    - Average quality metrics");
    println!("    - Top anomalies with recommendations");
}

#[cfg(not(feature = "native"))]
fn main() {
    println!("This example requires the 'native' feature.");
    println!("Run with: cargo run --example stack_diagnostics_demo --features native");
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_from_grade() {
        assert_eq!(HealthStatus::from_grade(QualityGrade::APlus), HealthStatus::Green);
        assert_eq!(HealthStatus::from_grade(QualityGrade::AMinus), HealthStatus::Yellow);
        assert_eq!(HealthStatus::from_grade(QualityGrade::B), HealthStatus::Red);
    }

    #[test]
    fn test_isolation_forest_scoring() {
        let mut forest = IsolationForest::new(10, 32, 42);
        let data = vec![
            vec![90.0, 85.0],
            vec![88.0, 82.0],
            vec![10.0, 5.0], // Outlier
        ];
        forest.fit(&data);
        let scores = forest.score(&data);
        assert_eq!(scores.len(), 3);
        for score in scores {
            assert!(score >= 0.0 && score <= 1.0);
        }
    }

    #[test]
    fn test_error_forecaster() {
        let mut forecaster = ErrorForecaster::new(0.5);
        forecaster.observe(10.0);
        forecaster.observe(20.0);
        assert_eq!(forecaster.current_level(), 15.0);
    }
}
