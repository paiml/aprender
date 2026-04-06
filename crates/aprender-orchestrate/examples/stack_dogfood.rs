#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    dead_code,
    clippy::needless_borrows_for_generic_args,
    clippy::match_single_binding,
    clippy::single_char_pattern
)]
//! Stack Dogfooding - Real PAIML Stack Health Analysis
//!
//! This example uses REAL pmat demo-score data from the actual PAIML stack
//! to demonstrate the diagnostics system eating its own dog food.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example stack_dogfood --features native
//! ```

#[cfg(feature = "native")]
use batuta::{
    render_dashboard, AndonStatus, Anomaly, AnomalyCategory, ComponentMetrics, ComponentNode,
    ErrorForecaster, HealthStatus, IsolationForest, QualityGrade, QualityStackLayer as StackLayer,
    StackDiagnostics,
};

#[cfg(feature = "native")]
type StackEntry = (&'static str, &'static str, StackLayer, f64, QualityGrade);

#[cfg(feature = "native")]
fn print_phase_header(num: u32, title: &str) {
    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│ PHASE {}: {:62}│", num, title);
    println!("└─────────────────────────────────────────────────────────────────────┘\n");
}

#[cfg(feature = "native")]
fn get_stack_data() -> Vec<StackEntry> {
    vec![
        ("trueno", "0.7.4", StackLayer::Compute, 89.9, QualityGrade::AMinus),
        ("realizar", "0.9.0", StackLayer::Training, 85.4, QualityGrade::AMinus),
        ("pforge", "0.1.4", StackLayer::Quality, 85.2, QualityGrade::AMinus),
        ("batuta", "0.1.4", StackLayer::Orchestration, 84.9, QualityGrade::BPlus),
        ("aprender", "0.11.0", StackLayer::Ml, 83.1, QualityGrade::BPlus),
        ("trueno-db", "0.5.0", StackLayer::DataMlops, 80.4, QualityGrade::BPlus),
        ("bashrs", "0.2.0", StackLayer::Transpilers, 79.0, QualityGrade::B),
        ("depyler", "0.4.0", StackLayer::Transpilers, 74.6, QualityGrade::B),
        ("entrenar", "1.8.0", StackLayer::Training, 73.5, QualityGrade::B),
    ]
}

#[cfg(feature = "native")]
fn build_diagnostics(stack_data: &[StackEntry]) -> StackDiagnostics {
    let mut diag = StackDiagnostics::new();
    for (name, version, layer, score, grade) in stack_data {
        let mut node = ComponentNode::new(*name, *version, *layer);
        node.metrics = ComponentMetrics::with_demo_score(*score);
        node.metrics.grade = *grade;
        node.update_health();
        diag.add_component(node);
    }
    diag
}

#[cfg(feature = "native")]
fn phase1_andon_status(diag: &StackDiagnostics) {
    print_phase_header(1, "ANDON STATUS OVERVIEW");

    let summary = diag.health_summary();
    let andon_icon = match summary.andon_status {
        AndonStatus::Green => "🟢",
        AndonStatus::Yellow => "🟡",
        AndonStatus::Red => "🔴",
        AndonStatus::Unknown => "⚪",
    };

    println!("  ANDON STATUS: {} {:?}\n", andon_icon, summary.andon_status);
    println!("  Components by Health:");
    println!("    🟢 Green (A-/A/A+):  {}", summary.green_count);
    println!("    🟡 Yellow (B+/A-):   {}", summary.yellow_count);
    println!("    🔴 Red (≤B):         {}", summary.red_count);
    println!("    ⚪ Unknown:          {}", summary.unknown_count);
    println!();
}

#[cfg(feature = "native")]
fn phase2_quality_matrix(stack_data: &[StackEntry]) {
    print_phase_header(2, "COMPONENT QUALITY MATRIX");

    println!("  ┌──────────────┬─────────┬───────────┬────────┬────────────────────┐");
    println!("  │ Component    │ Version │ Demo Score│ Grade  │ Quality Gate       │");
    println!("  ├──────────────┼─────────┼───────────┼────────┼────────────────────┤");

    for (name, version, _, score, grade) in stack_data {
        let status = HealthStatus::from_grade(*grade);
        let gate = if *score >= 85.0 { "✅ PASS" } else { "❌ FAIL" };
        println!(
            "  │ {:12} │ {:7} │ {:>9.1} │ {} {:4} │ {:18} │",
            name,
            version,
            score,
            status.icon(),
            grade.symbol(),
            gate
        );
    }
    println!("  └──────────────┴─────────┴───────────┴────────┴────────────────────┘\n");

    let scores: Vec<f64> = stack_data.iter().map(|(_, _, _, s, _)| *s).collect();
    let avg = scores.iter().sum::<f64>() / scores.len() as f64;
    let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let pass = scores.iter().filter(|&&s| s >= 85.0).count();

    println!("  STATISTICS:");
    println!("    Average Score:    {:.1}/100", avg);
    println!("    Min Score:        {:.1}/100 (entrenar)", min);
    println!("    Max Score:        {:.1}/100 (trueno)", max);
    println!(
        "    Passing (≥85.0):  {}/{} ({:.0}%)\n",
        pass,
        scores.len(),
        (pass as f64 / scores.len() as f64) * 100.0
    );
}

#[cfg(feature = "native")]
fn phase3_anomaly_detection(stack_data: &[StackEntry], diag: &mut StackDiagnostics) {
    print_phase_header(3, "ISOLATION FOREST ANOMALY DETECTION");

    let data: Vec<Vec<f64>> = stack_data.iter().map(|(_, _, _, score, _)| vec![*score]).collect();
    let mut forest = IsolationForest::new(100, 256, 42);
    forest.fit(&data);
    let anomaly_scores = forest.score(&data);

    println!("  Isolation Forest Results (threshold: 0.5):\n");
    println!("  ┌──────────────┬──────────────┬────────────────────────────────────┐");
    println!("  │ Component    │ Anomaly Score│ Status                             │");
    println!("  ├──────────────┼──────────────┼────────────────────────────────────┤");

    let mut anomalies = Vec::new();
    for ((name, _, _, demo_score, _), &score) in stack_data.iter().zip(anomaly_scores.iter()) {
        let status = if score > 0.6 {
            anomalies.push((*name, score, *demo_score));
            "🔴 ANOMALY - Quality regression"
        } else if score > 0.5 {
            "🟡 BORDERLINE - Monitor closely"
        } else {
            "🟢 NORMAL - Within expected range"
        };
        println!("  │ {:12} │ {:>12.4} │ {:34} │", name, score, status);
    }
    println!("  └──────────────┴──────────────┴────────────────────────────────────┘\n");

    for (name, score, demo_score) in &anomalies {
        diag.add_anomaly(
            Anomaly::new(
                *name,
                *score,
                AnomalyCategory::QualityRegression,
                format!("Demo score {:.1} below A- threshold (85.0)", demo_score),
            )
            .with_evidence(format!("Isolation Forest anomaly score: {:.4}", score))
            .with_recommendation("Run `pmat demo-score` and address top issues"),
        );
    }

    println!("  ANOMALIES DETECTED: {}", anomalies.len());
    for (name, score, demo_score) in &anomalies {
        println!("    - {} (score: {:.1}, anomaly: {:.4})", name, demo_score, score);
    }
    println!();
}

#[cfg(feature = "native")]
fn phase4_graph_analytics(diag: &mut StackDiagnostics) -> anyhow::Result<()> {
    print_phase_header(4, "GRAPH ANALYTICS");

    let metrics = diag.compute_metrics()?;
    println!("  Graph-Level Metrics:");
    println!("    Total Nodes:     {}", metrics.total_nodes);
    println!("    Total Edges:     {}", metrics.total_edges);
    println!("    Graph Density:   {:.4}", metrics.density);
    println!("    Average Degree:  {:.2}\n", metrics.avg_degree);

    println!("  PageRank Importance:");
    for (name, score) in metrics.top_by_pagerank(9) {
        let bar: String = "█".repeat((score * 50.0).round() as usize).chars().take(20).collect();
        println!("    {:12} │ {:5.3} │ {}", name, score, bar);
    }
    println!();
    Ok(())
}

#[cfg(feature = "native")]
fn phase5_error_forecasting() {
    print_phase_header(5, "ERROR FORECASTING");

    let historical = [2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0];
    let mut forecaster = ErrorForecaster::new(0.3);
    for &obs in &historical {
        forecaster.observe(obs);
    }

    println!("  Historical Quality Gate Failures:");
    for (i, &obs) in historical.iter().enumerate() {
        println!("    Week {}: {} failures", i + 1, obs as i32);
    }

    println!("\n  Forecast (next 4 weeks):");
    for (i, &f) in forecaster.forecast(4).iter().enumerate() {
        let trend = if f > historical[7] {
            "↗"
        } else if f < historical[7] {
            "↘"
        } else {
            "→"
        };
        println!("    Week {}: {:.1} failures predicted {}", i + 9, f, trend);
    }

    let err = forecaster.error_metrics();
    println!("\n  Forecast Accuracy:");
    println!("    MAE:  {:.2}", err.mae);
    println!("    RMSE: {:.2}\n", err.rmse);
}

#[cfg(feature = "native")]
fn phase6_dashboard(diag: &StackDiagnostics) {
    print_phase_header(6, "ASCII DASHBOARD (MIERUKA)");
    println!("{}", render_dashboard(diag));
}

#[cfg(feature = "native")]
fn phase7_insights(stack_data: &[StackEntry], diag: &StackDiagnostics) {
    print_phase_header(7, "INSIGHTS & RECOMMENDATIONS");

    let scores: Vec<f64> = stack_data.iter().map(|(_, _, _, s, _)| *s).collect();
    let pass_count = scores.iter().filter(|&&s| s >= 85.0).count();
    let summary = diag.health_summary();

    let overall = if pass_count >= 6 {
        "🟢 GOOD"
    } else if pass_count >= 3 {
        "🟡 ATTENTION NEEDED"
    } else {
        "🔴 CRITICAL"
    };
    println!("  🎯 KEY INSIGHTS:\n");
    println!(
        "  1. OVERALL HEALTH: {} ({:.1}% passing)\n",
        overall,
        (pass_count as f64 / scores.len() as f64) * 100.0
    );

    println!("  2. TOP PERFORMERS (A- or better):");
    for (name, _, _, score, grade) in stack_data.iter().filter(|(_, _, _, s, _)| *s >= 85.0) {
        println!("     ✅ {} ({:.1} - {})", name, score, grade.symbol());
    }

    println!("\n  3. NEEDS ATTENTION (below A-):");
    for (name, _, _, score, grade) in stack_data.iter().filter(|(_, _, _, s, _)| *s < 85.0) {
        println!(
            "     ❌ {} ({:.1} - {}) - needs +{:.1} points",
            name,
            score,
            grade.symbol(),
            85.0 - score
        );
    }

    println!("\n  4. PRIORITY ORDER (by gap to A-):");
    let mut below: Vec<_> = stack_data.iter().filter(|(_, _, _, s, _)| *s < 85.0).collect();
    below.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    for (i, (name, _, _, score, _)) in below.iter().enumerate() {
        println!("     {}. {} (+{:.1} to pass)", i + 1, name, 85.0 - score);
    }

    println!("\n  5. RECOMMENDED ACTIONS:");
    println!("     a. Focus on batuta first (only +0.1 to pass)");
    println!("     b. Improve aprender demo runtime quality (currently 75%)");
    println!("     c. Address trueno-db quality standards");
    println!("     d. Review entrenar for technical debt (lowest score)\n");

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                  DOGFOODING ANALYSIS COMPLETE                     ║");
    println!(
        "║  Stack Health: {} Green, {} Yellow, {} Red                        ║",
        summary.green_count, summary.yellow_count, summary.red_count
    );
    println!(
        "║  Quality Gate: {}/9 passing ({:.0}%)                               ║",
        pass_count,
        (pass_count as f64 / 9.0) * 100.0
    );
    println!("╚═══════════════════════════════════════════════════════════════════╝");
}

#[cfg(feature = "native")]
fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║        PAIML SOVEREIGN AI STACK - REAL HEALTH ANALYSIS            ║");
    println!("║               Dogfooding Stack Diagnostics v0.1.0                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let stack_data = get_stack_data();
    let mut diag = build_diagnostics(&stack_data);

    phase1_andon_status(&diag);
    phase2_quality_matrix(&stack_data);
    phase3_anomaly_detection(&stack_data, &mut diag);
    phase4_graph_analytics(&mut diag)?;
    phase5_error_forecasting();
    phase6_dashboard(&diag);
    phase7_insights(&stack_data, &diag);

    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    println!("This example requires the 'native' feature.");
    println!("Run with: cargo run --example stack_dogfood --features native");
}
