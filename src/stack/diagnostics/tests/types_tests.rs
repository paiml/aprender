//! Tests for diagnostics types
//!
//! Tests for HealthStatus, ComponentNode, ComponentMetrics, GraphMetrics,
//! HealthSummary, Anomaly, and AndonStatus types.

use crate::stack::diagnostics::*;
use crate::stack::quality::{QualityGrade, StackLayer};

// ========================================================================
// HealthStatus Tests
// ========================================================================

#[test]
fn test_health_status_from_grade_green() {
    assert_eq!(HealthStatus::from_grade(QualityGrade::APlus), HealthStatus::Green);
    assert_eq!(HealthStatus::from_grade(QualityGrade::A), HealthStatus::Green);
}

#[test]
fn test_health_status_from_grade_yellow() {
    assert_eq!(HealthStatus::from_grade(QualityGrade::AMinus), HealthStatus::Yellow);
    assert_eq!(HealthStatus::from_grade(QualityGrade::BPlus), HealthStatus::Yellow);
}

#[test]
fn test_health_status_from_grade_red() {
    assert_eq!(HealthStatus::from_grade(QualityGrade::B), HealthStatus::Red);
    assert_eq!(HealthStatus::from_grade(QualityGrade::C), HealthStatus::Red);
    assert_eq!(HealthStatus::from_grade(QualityGrade::F), HealthStatus::Red);
}

#[test]
fn test_health_status_icons() {
    assert_eq!(HealthStatus::Green.icon(), "🟢");
    assert_eq!(HealthStatus::Yellow.icon(), "🟡");
    assert_eq!(HealthStatus::Red.icon(), "🔴");
    assert_eq!(HealthStatus::Unknown.icon(), "⚪");
}

#[test]
fn test_health_status_symbols() {
    assert_eq!(HealthStatus::Green.symbol(), "●");
    assert_eq!(HealthStatus::Yellow.symbol(), "◐");
    assert_eq!(HealthStatus::Red.symbol(), "○");
    assert_eq!(HealthStatus::Unknown.symbol(), "◌");
}

// ========================================================================
// ComponentNode Tests
// ========================================================================

#[test]
fn test_component_node_creation() {
    let node = ComponentNode::new("trueno", "0.7.4", StackLayer::Compute);
    assert_eq!(node.name, "trueno");
    assert_eq!(node.version, "0.7.4");
    assert_eq!(node.layer, StackLayer::Compute);
    assert_eq!(node.health, HealthStatus::Unknown);
}

#[test]
fn test_component_node_update_health() {
    let mut node = ComponentNode::new("trueno", "0.7.4", StackLayer::Compute);
    node.metrics = ComponentMetrics::with_demo_score(95.0);
    node.update_health();
    assert_eq!(node.health, HealthStatus::Green);
}

#[test]
fn test_component_node_update_health_yellow() {
    let mut node = ComponentNode::new("test", "1.0.0", StackLayer::Ml);
    node.metrics = ComponentMetrics::with_demo_score(85.0);
    node.update_health();
    assert_eq!(node.health, HealthStatus::Yellow);
}

#[test]
fn test_component_node_update_health_red() {
    let mut node = ComponentNode::new("test", "1.0.0", StackLayer::Ml);
    node.metrics = ComponentMetrics::with_demo_score(65.0);
    node.update_health();
    assert_eq!(node.health, HealthStatus::Red);
}

// ========================================================================
// ComponentMetrics Tests
// ========================================================================

#[test]
fn test_component_metrics_default() {
    let metrics = ComponentMetrics::default();
    assert_eq!(metrics.demo_score, 0.0);
    assert_eq!(metrics.coverage, 0.0);
    assert!(!metrics.meets_threshold());
}

#[test]
fn test_component_metrics_with_demo_score() {
    let metrics = ComponentMetrics::with_demo_score(90.0);
    assert_eq!(metrics.demo_score, 90.0);
    assert!(metrics.meets_threshold());
}

#[test]
fn test_component_metrics_threshold() {
    assert!(ComponentMetrics::with_demo_score(85.0).meets_threshold());
    assert!(ComponentMetrics::with_demo_score(100.0).meets_threshold());
    assert!(!ComponentMetrics::with_demo_score(84.9).meets_threshold());
}

// ========================================================================
// GraphMetrics Tests
// ========================================================================

#[test]
fn test_graph_metrics_top_by_pagerank() {
    let mut metrics = GraphMetrics::default();
    metrics.pagerank.insert("trueno".to_string(), 0.25);
    metrics.pagerank.insert("aprender".to_string(), 0.15);
    metrics.pagerank.insert("batuta".to_string(), 0.10);

    let top = metrics.top_by_pagerank(2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].0, "trueno");
    assert_eq!(top[1].0, "aprender");
}

#[test]
fn test_graph_metrics_bottlenecks() {
    let mut metrics = GraphMetrics::default();
    metrics.betweenness.insert("trueno".to_string(), 0.8);
    metrics.betweenness.insert("aprender".to_string(), 0.3);
    metrics.betweenness.insert("batuta".to_string(), 0.1);

    let bottlenecks = metrics.bottlenecks(0.5);
    assert_eq!(bottlenecks.len(), 1);
    assert!(bottlenecks.contains(&&"trueno".to_string()));
}

#[test]
fn test_graph_metrics_top_by_pagerank_empty() {
    let metrics = GraphMetrics::default();
    let top = metrics.top_by_pagerank(5);
    assert!(top.is_empty());
}

#[test]
fn test_graph_metrics_bottlenecks_empty() {
    let metrics = GraphMetrics::default();
    let bottlenecks = metrics.bottlenecks(0.5);
    assert!(bottlenecks.is_empty());
}

// ========================================================================
// HealthSummary Tests
// ========================================================================

#[test]
fn test_health_summary_percentage() {
    let summary = HealthSummary {
        total_components: 20,
        green_count: 17,
        yellow_count: 3,
        red_count: 0,
        unknown_count: 0,
        avg_demo_score: 85.0,
        avg_coverage: 90.0,
        andon_status: AndonStatus::Yellow,
    };

    assert_eq!(summary.health_percentage(), 85.0);
    assert!(!summary.all_healthy());
}

#[test]
fn test_health_summary_percentage_empty() {
    let summary = HealthSummary {
        total_components: 0,
        green_count: 0,
        yellow_count: 0,
        red_count: 0,
        unknown_count: 0,
        avg_demo_score: 0.0,
        avg_coverage: 0.0,
        andon_status: AndonStatus::Unknown,
    };

    assert_eq!(summary.health_percentage(), 0.0);
}

// ========================================================================
// Anomaly Tests
// ========================================================================

#[test]
fn test_anomaly_creation() {
    let anomaly = Anomaly::new("test", 0.65, AnomalyCategory::QualityRegression, "Score dropped");

    assert_eq!(anomaly.component, "test");
    assert_eq!(anomaly.score, 0.65);
    assert!(!anomaly.is_critical());
    assert!(anomaly.evidence.is_empty());
    assert!(anomaly.recommendation.is_none());
}

#[test]
fn test_anomaly_critical() {
    let critical = Anomaly::new("test", 0.85, AnomalyCategory::DependencyRisk, "High risk");
    assert!(critical.is_critical());

    let non_critical = Anomaly::new("test", 0.79, AnomalyCategory::Other, "Low risk");
    assert!(!non_critical.is_critical());
}

#[test]
fn test_anomaly_with_details() {
    let anomaly = Anomaly::new("test", 0.7, AnomalyCategory::BuildTimeSpike, "Build slow")
        .with_evidence("Time increased 40%")
        .with_evidence("New macro expansion")
        .with_recommendation("Enable incremental compilation");

    assert_eq!(anomaly.evidence.len(), 2);
    assert!(anomaly.recommendation.is_some());
}

#[test]
fn test_anomaly_category_display() {
    assert_eq!(format!("{}", AnomalyCategory::QualityRegression), "Quality Regression");
    assert_eq!(format!("{}", AnomalyCategory::CoverageDrop), "Coverage Drop");
    assert_eq!(format!("{}", AnomalyCategory::BuildTimeSpike), "Build Time Spike");
}

// ========================================================================
// AndonStatus Tests
// ========================================================================

#[test]
fn test_andon_status_messages() {
    assert_eq!(AndonStatus::Green.message(), "All systems healthy");
    assert_eq!(AndonStatus::Yellow.message(), "Attention needed");
    assert_eq!(AndonStatus::Red.message(), "Stop-the-line");
    assert_eq!(AndonStatus::Unknown.message(), "Analysis pending");
}

#[test]
fn test_andon_status_display() {
    let green = format!("{}", AndonStatus::Green);
    assert!(green.contains("🟢"));
    assert!(green.contains("healthy"));
}

// ========================================================================
// Coverage: Display for HealthStatus (0% -> 100%)
// ========================================================================

#[test]
fn test_health_status_display_all_variants() {
    let green = format!("{}", HealthStatus::Green);
    assert!(green.contains("🟢"));

    let yellow = format!("{}", HealthStatus::Yellow);
    assert!(yellow.contains("🟡"));

    let red = format!("{}", HealthStatus::Red);
    assert!(red.contains("🔴"));

    let unknown = format!("{}", HealthStatus::Unknown);
    assert!(unknown.contains("⚪"));
}

// ========================================================================
// Coverage: Display for AndonStatus (75% -> 100%)
// ========================================================================

#[test]
fn test_andon_status_display_all_variants() {
    let yellow = format!("{}", AndonStatus::Yellow);
    assert!(yellow.contains("🟡"));
    assert!(yellow.contains("Attention"));

    let red = format!("{}", AndonStatus::Red);
    assert!(red.contains("🔴"));
    assert!(red.contains("Stop"));

    let unknown = format!("{}", AndonStatus::Unknown);
    assert!(unknown.contains("⚪"));
    assert!(unknown.contains("pending"));
}

// ========================================================================
// Coverage: Display for AnomalyCategory (89% -> 100%)
// ========================================================================

#[test]
fn test_anomaly_category_display_all_variants() {
    assert_eq!(format!("{}", AnomalyCategory::DependencyRisk), "Dependency Risk");
    assert_eq!(format!("{}", AnomalyCategory::ComplexityIncrease), "Complexity Increase");
    assert_eq!(format!("{}", AnomalyCategory::Other), "Other");
}
