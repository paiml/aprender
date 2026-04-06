//! Tests for dashboard rendering
//!
//! Tests for the ASCII dashboard rendering functionality.

use crate::stack::diagnostics::*;
use crate::stack::quality::StackLayer;

// ========================================================================
// Dashboard Renderer Tests
// ========================================================================

#[test]
fn test_render_dashboard_empty() {
    let diag = StackDiagnostics::new();
    let output = render_dashboard(&diag);

    assert!(output.contains("SOVEREIGN AI STACK"));
    assert!(output.contains("ANDON STATUS"));
    assert!(output.contains("Total Components"));
}

#[test]
fn test_render_dashboard_with_components() {
    let mut diag = StackDiagnostics::new();

    let mut node = ComponentNode::new("trueno", "0.7.4", StackLayer::Compute);
    node.health = HealthStatus::Green;
    node.metrics = ComponentMetrics::with_demo_score(92.0);
    diag.add_component(node);

    let output = render_dashboard(&diag);

    assert!(output.contains("Total Components:      1"));
    assert!(output.contains("Healthy:               1"));
}

#[test]
fn test_render_dashboard_with_anomalies() {
    let mut diag = StackDiagnostics::new();

    diag.add_anomaly(Anomaly::new(
        "trueno-graph",
        0.75,
        AnomalyCategory::CoverageDrop,
        "Coverage dropped",
    ));

    let output = render_dashboard(&diag);
    assert!(output.contains("ANOMALIES DETECTED"));
    assert!(output.contains("trueno-graph"));
}
