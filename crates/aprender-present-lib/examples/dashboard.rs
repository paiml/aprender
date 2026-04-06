//! Dashboard example demonstrating data visualization widgets.
//!
//! Run with: `cargo run --example dashboard`

#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    unused_variables,
    clippy::iter_without_into_iter,
    clippy::cast_possible_wrap,
    dead_code,
    clippy::too_many_lines
)]

use presentar::widgets::{
    row::MainAxisAlignment, Chart, Column, DataSeries, ProgressBar, Row, Text,
};
use presentar::{Color, Constraints, DrawCommand, RecordingCanvas, Rect, Size, Widget};

fn main() {
    println!("=== Presentar Dashboard Example ===\n");

    // Create sample data series for chart
    let sales_series = DataSeries::new("Sales")
        .points(vec![
            (1.0, 120.0),
            (2.0, 180.0),
            (3.0, 150.0),
            (4.0, 220.0),
            (5.0, 280.0),
            (6.0, 250.0),
        ])
        .color(Color::from_hex("#6366f1").expect("valid hex"));

    // Build dashboard layout
    let mut dashboard = Column::new()
        .gap(24.0)
        // Header
        .child(
            Text::new("Sales Dashboard")
                .font_size(28.0)
                .color(Color::from_hex("#111827").expect("valid hex")),
        )
        // KPI Row using Text widgets
        .child(
            Row::new()
                .gap(32.0)
                .main_axis_alignment(MainAxisAlignment::SpaceEvenly)
                .child(
                    Column::new()
                        .gap(4.0)
                        .child(
                            Text::new("Total Revenue")
                                .font_size(12.0)
                                .color(Color::from_hex("#6b7280").expect("hex")),
                        )
                        .child(
                            Text::new("$1,200,000")
                                .font_size(24.0)
                                .color(Color::from_hex("#111827").expect("hex")),
                        )
                        .child(Text::new("+12.5%").font_size(14.0).color(Color::GREEN)),
                )
                .child(
                    Column::new()
                        .gap(4.0)
                        .child(
                            Text::new("Orders")
                                .font_size(12.0)
                                .color(Color::from_hex("#6b7280").expect("hex")),
                        )
                        .child(
                            Text::new("3,847")
                                .font_size(24.0)
                                .color(Color::from_hex("#111827").expect("hex")),
                        )
                        .child(Text::new("+8.2%").font_size(14.0).color(Color::GREEN)),
                )
                .child(
                    Column::new()
                        .gap(4.0)
                        .child(
                            Text::new("Customers")
                                .font_size(12.0)
                                .color(Color::from_hex("#6b7280").expect("hex")),
                        )
                        .child(
                            Text::new("12,493")
                                .font_size(24.0)
                                .color(Color::from_hex("#111827").expect("hex")),
                        )
                        .child(Text::new("-2.1%").font_size(14.0).color(Color::RED)),
                ),
        )
        // Chart
        .child(
            Chart::line()
                .title("Monthly Sales")
                .series(sales_series)
                .width(400.0)
                .height(200.0),
        )
        // Progress indicators
        .child(
            Column::new()
                .gap(12.0)
                .child(Text::new("Quarterly Goals").font_size(18.0))
                .child(
                    Row::new()
                        .gap(8.0)
                        .child(Text::new("Revenue Target").font_size(14.0))
                        .child(ProgressBar::new().value(0.75).fill_color(Color::GREEN)),
                )
                .child(
                    Row::new()
                        .gap(8.0)
                        .child(Text::new("Customer Growth").font_size(14.0))
                        .child(ProgressBar::new().value(0.45).fill_color(Color::BLUE)),
                ),
        );

    // Measure and layout
    let constraints = Constraints::loose(Size::new(800.0, 600.0));
    let size = dashboard.measure(constraints);
    println!("Dashboard size: {}x{}", size.width, size.height);

    let bounds = Rect::new(0.0, 0.0, size.width, size.height);
    let result = dashboard.layout(bounds);
    println!(
        "Layout complete: {}x{}",
        result.size.width, result.size.height
    );

    // Paint and count commands
    let mut canvas = RecordingCanvas::new();
    dashboard.paint(&mut canvas);
    println!("\nGenerated {} draw commands", canvas.command_count());

    // Print summary of draw commands by type
    let mut rect_count = 0;
    let mut text_count = 0;
    for cmd in canvas.commands() {
        match cmd {
            DrawCommand::Rect { .. } => rect_count += 1,
            DrawCommand::Text { .. } => text_count += 1,
            _ => {}
        }
    }
    println!("  - Rect commands: {rect_count}");
    println!("  - Text commands: {text_count}");

    println!("\n=== Dashboard Example Complete ===");
}
