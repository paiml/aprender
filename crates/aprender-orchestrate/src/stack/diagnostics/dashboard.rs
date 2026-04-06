//! Dashboard rendering for stack diagnostics
//!
//! This module provides ASCII dashboard rendering functionality for
//! visualizing stack health status in a terminal-friendly format.

use super::engine::StackDiagnostics;

// ============================================================================
// Dashboard Renderer
// ============================================================================

/// Render diagnostics as ASCII dashboard
pub fn render_dashboard(diagnostics: &StackDiagnostics) -> String {
    let mut output = String::new();
    let summary = diagnostics.health_summary();

    // Header
    output
        .push_str("┌─────────────────────────────────────────────────────────────────────────┐\n");
    output
        .push_str("│                  SOVEREIGN AI STACK HEALTH DASHBOARD                    │\n");
    output.push_str(&format!(
        "│                  Timestamp: {:40} │\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S")
    ));
    output
        .push_str("├─────────────────────────────────────────────────────────────────────────┤\n");

    // Andon Status
    output
        .push_str("│                                                                         │\n");
    output.push_str(&format!("│  ANDON STATUS: {} {:55}│\n", summary.andon_status, ""));
    output
        .push_str("│                                                                         │\n");

    // Stack Summary
    output.push_str("│  ═══════════════════════════════════════════════════════════════════   │\n");
    output
        .push_str("│  STACK SUMMARY                                                          │\n");
    output.push_str("│  ═══════════════════════════════════════════════════════════════════   │\n");
    output
        .push_str("│                                                                         │\n");
    output.push_str(&format!(
        "│  Total Components:    {:3}                                              │\n",
        summary.total_components
    ));
    output.push_str(&format!(
        "│  Healthy:             {:3} ({:.0}%)                                         │\n",
        summary.green_count,
        summary.health_percentage()
    ));
    output.push_str(&format!(
        "│  Warnings:            {:3} ({:.0}%)                                         │\n",
        summary.yellow_count,
        if summary.total_components > 0 {
            (summary.yellow_count as f64 / summary.total_components as f64) * 100.0
        } else {
            0.0
        }
    ));
    output.push_str(&format!(
        "│  Critical:            {:3} ({:.0}%)                                         │\n",
        summary.red_count,
        if summary.total_components > 0 {
            (summary.red_count as f64 / summary.total_components as f64) * 100.0
        } else {
            0.0
        }
    ));
    output.push_str(&format!(
        "│  Average Demo Score:  {:.1}/100                                          │\n",
        summary.avg_demo_score
    ));
    output.push_str(&format!(
        "│  Average Coverage:    {:.1}%                                             │\n",
        summary.avg_coverage
    ));
    output
        .push_str("│                                                                         │\n");

    // Anomalies
    let anomalies = diagnostics.anomalies();
    if !anomalies.is_empty() {
        output.push_str(
            "│  ═══════════════════════════════════════════════════════════════════   │\n",
        );
        output.push_str(
            "│  ANOMALIES DETECTED                                                     │\n",
        );
        output.push_str(
            "│  ═══════════════════════════════════════════════════════════════════   │\n",
        );
        output.push_str(
            "│                                                                         │\n",
        );

        for anomaly in anomalies.iter().take(5) {
            let icon = if anomaly.is_critical() { "🔴" } else { "⚠️" };
            output.push_str(&format!(
                "│  {}  {}: {}                               │\n",
                icon, anomaly.component, anomaly.description
            ));
        }
        output.push_str(
            "│                                                                         │\n",
        );
    }

    output
        .push_str("└─────────────────────────────────────────────────────────────────────────┘\n");

    output
}
