//! Dashboard Widgets Example
//!
//! Demonstrates the experiment dashboard widgets: Sparkline, `ResourceBar`, and `RunTable`.
//!
//! Run with: cargo run --example `dashboard_widgets`

use trueno_viz::prelude::*;

fn main() {
    println!("Dashboard Widgets Example");
    println!("=========================\n");

    // =========================================================================
    // Sparkline Widget
    // =========================================================================
    println!("1. Sparkline Widget");
    println!("   ----------------");

    let loss_values = vec![0.9, 0.75, 0.6, 0.45, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05];

    let sparkline = Sparkline::new(&loss_values)
        .dimensions(100, 20)
        .color(Rgba::rgb(66, 133, 244))
        .with_trend_indicator();

    let trend = sparkline.trend();
    println!(
        "   Loss trend: {} {}",
        trend.indicator(),
        match trend {
            TrendDirection::Rising => "Rising (bad for loss!)",
            TrendDirection::Falling => "Falling (good for loss!)",
            TrendDirection::Stable => "Stable",
        }
    );

    // Render sparkline
    match sparkline.to_framebuffer() {
        Ok(fb) => println!("   Rendered sparkline: {}x{} pixels", fb.width(), fb.height()),
        Err(e) => println!("   Error: {e}"),
    }

    // =========================================================================
    // ResourceBar Widget
    // =========================================================================
    println!("\n2. ResourceBar Widget");
    println!("   ------------------");

    let resources = vec![
        ("GPU Hours", 100.0, 75.0, "hours"),
        ("Training Time", 24.0, 18.5, "hours"),
        ("Compute Cost", 500.0, 620.0, "$"), // Over budget!
        ("Memory", 32.0, 28.0, "GB"),
    ];

    for (label, planned, actual, unit) in resources {
        let bar = ResourceBar::new(label, planned, actual, unit);
        let status = if bar.is_over_budget() { "OVER BUDGET" } else { "OK" };
        println!(
            "   {}: {:.1}/{:.1} {} ({:.1}%) [{}]",
            label,
            actual,
            planned,
            unit,
            bar.percentage(),
            status
        );
    }

    // =========================================================================
    // RunTable Widget
    // =========================================================================
    println!("\n3. RunTable Widget");
    println!("   ----------------");

    let runs = vec![
        RunRow::new("exp-001", RunStatus::Completed)
            .with_duration(3600.0)
            .with_metric("loss", 0.05)
            .with_metric("accuracy", 0.95),
        RunRow::new("exp-002", RunStatus::Completed)
            .with_duration(4200.0)
            .with_metric("loss", 0.03)
            .with_metric("accuracy", 0.97),
        RunRow::new("exp-003", RunStatus::Running).with_duration(1800.0).with_metric("loss", 0.08),
        RunRow::new("exp-004", RunStatus::Pending),
        RunRow::new("exp-005", RunStatus::Failed).with_duration(600.0).with_metric("loss", 0.45),
    ];

    let table = RunTable::from_runs(runs);

    // Status summary
    let counts = table.status_counts();
    println!("   Status Summary:");
    println!("   - Completed: {}", counts.get(&RunStatus::Completed).unwrap_or(&0));
    println!("   - Running: {}", counts.get(&RunStatus::Running).unwrap_or(&0));
    println!("   - Pending: {}", counts.get(&RunStatus::Pending).unwrap_or(&0));
    println!("   - Failed: {}", counts.get(&RunStatus::Failed).unwrap_or(&0));

    // Render table
    println!("\n   Rendered Table:");
    for line in table.render().lines() {
        println!("   {line}");
    }

    // Sort by loss and show best run
    let mut sorted_table = RunTable::from_runs(vec![
        RunRow::new("exp-001", RunStatus::Completed).with_metric("loss", 0.05),
        RunRow::new("exp-002", RunStatus::Completed).with_metric("loss", 0.03),
        RunRow::new("exp-005", RunStatus::Failed).with_metric("loss", 0.45),
    ]);
    sorted_table.sort_by_metric("loss");

    println!(
        "\n   Best run by loss: {} (loss={:.4})",
        sorted_table.runs()[0].id,
        sorted_table.runs()[0].metric("loss").unwrap_or(f64::NAN)
    );

    println!("\n--- Example Complete ---");
}
