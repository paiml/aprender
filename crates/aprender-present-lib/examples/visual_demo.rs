//! Visual Demo - Exports UI to SVG for browser viewing.
//!
//! Run with: `cargo run --example visual_demo -p presentar`
//! Then open: `demo_output.svg` in your browser

#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::match_same_arms,
    clippy::needless_range_loop,
    clippy::if_same_then_else,
    clippy::too_many_lines,
    clippy::or_fun_call,
    clippy::format_push_string,
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::struct_field_names,
    clippy::println_empty_string,
    unused_variables
)]

use presentar::widgets::{
    row::MainAxisAlignment, Chart, Column, DataSeries, ProgressBar, Row, Text,
};
use presentar::{Color, Constraints, DrawCommand, RecordingCanvas, Rect, Size, Widget};
use std::fs::File;
use std::io::Write;

/// Convert draw commands to SVG
fn to_svg(commands: &[DrawCommand], width: f32, height: f32) -> String {
    let mut svg = String::new();
    svg.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    svg.push('\n');
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">"#
    ));
    svg.push('\n');
    svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"#f8fafc\"/>\n");
    svg.push_str("  <style>text { font-family: system-ui, -apple-system, sans-serif; }</style>\n");

    for cmd in commands {
        match cmd {
            DrawCommand::Rect {
                bounds,
                style,
                radius,
            } => {
                let fill = style.fill.map_or("none".to_string(), |c| {
                    format!(
                        "rgb({},{},{})",
                        (c.r * 255.0) as u8,
                        (c.g * 255.0) as u8,
                        (c.b * 255.0) as u8
                    )
                });
                let stroke = style
                    .stroke
                    .as_ref()
                    .map_or(("none".to_string(), 0.0), |s| {
                        (
                            format!(
                                "rgb({},{},{})",
                                (s.color.r * 255.0) as u8,
                                (s.color.g * 255.0) as u8,
                                (s.color.b * 255.0) as u8
                            ),
                            s.width,
                        )
                    });
                let rx = radius.top_left.max(0.0);
                svg.push_str(&format!(
                    r#"  <rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" rx="{:.1}" fill="{}" stroke="{}" stroke-width="{:.1}"/>
"#,
                    bounds.x, bounds.y, bounds.width, bounds.height, rx, fill, stroke.0, stroke.1
                ));
            }
            DrawCommand::Text {
                content,
                position,
                style,
            } => {
                let color = format!(
                    "rgb({},{},{})",
                    (style.color.r * 255.0) as u8,
                    (style.color.g * 255.0) as u8,
                    (style.color.b * 255.0) as u8
                );
                let weight = match style.weight {
                    presentar::FontWeight::Bold => "bold",
                    presentar::FontWeight::Medium => "500",
                    presentar::FontWeight::Semibold => "600",
                    _ => "normal",
                };
                // Adjust y position for baseline
                let y = style.size.mul_add(0.85, position.y);
                svg.push_str(&format!(
                    r#"  <text x="{:.1}" y="{:.1}" font-size="{:.1}" font-weight="{}" fill="{}">{}</text>
"#,
                    position.x, y, style.size, weight, color,
                    html_escape(content)
                ));
            }
            DrawCommand::Circle {
                center,
                radius,
                style,
            } => {
                let fill = style.fill.map_or("none".to_string(), |c| {
                    format!(
                        "rgb({},{},{})",
                        (c.r * 255.0) as u8,
                        (c.g * 255.0) as u8,
                        (c.b * 255.0) as u8
                    )
                });
                svg.push_str(&format!(
                    r#"  <circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="{}"/>
"#,
                    center.x, center.y, radius, fill
                ));
            }
            _ => {}
        }
    }

    svg.push_str("</svg>\n");
    svg
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn main() {
    println!("=== Presentar Visual Demo ===\n");

    // Create chart data
    let sales_series = DataSeries::new("Sales")
        .points(vec![
            (1.0, 120.0),
            (2.0, 180.0),
            (3.0, 150.0),
            (4.0, 220.0),
            (5.0, 280.0),
            (6.0, 250.0),
        ])
        .color(Color::from_hex("#6366f1").expect("hex"));

    // Build dashboard
    let mut dashboard = Column::new()
        .gap(24.0)
        .child(
            Text::new("Presentar Dashboard")
                .font_size(32.0)
                .color(Color::from_hex("#111827").expect("hex")),
        )
        .child(
            Text::new("Real-time analytics powered by WASM")
                .font_size(14.0)
                .color(Color::from_hex("#6b7280").expect("hex")),
        )
        .child(
            Row::new()
                .gap(32.0)
                .main_axis_alignment(MainAxisAlignment::SpaceEvenly)
                .child(
                    Column::new()
                        .gap(4.0)
                        .child(
                            Text::new("Revenue")
                                .font_size(12.0)
                                .color(Color::from_hex("#6b7280").expect("hex")),
                        )
                        .child(
                            Text::new("$1.2M")
                                .font_size(28.0)
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
                                .font_size(28.0)
                                .color(Color::from_hex("#111827").expect("hex")),
                        )
                        .child(Text::new("+8.2%").font_size(14.0).color(Color::GREEN)),
                )
                .child(
                    Column::new()
                        .gap(4.0)
                        .child(
                            Text::new("Users")
                                .font_size(12.0)
                                .color(Color::from_hex("#6b7280").expect("hex")),
                        )
                        .child(
                            Text::new("12.4K")
                                .font_size(28.0)
                                .color(Color::from_hex("#111827").expect("hex")),
                        )
                        .child(Text::new("-2.1%").font_size(14.0).color(Color::RED)),
                ),
        )
        .child(
            Chart::line()
                .title("Monthly Sales")
                .series(sales_series)
                .width(500.0)
                .height(200.0),
        )
        .child(
            Column::new()
                .gap(12.0)
                .child(Text::new("Goals Progress").font_size(18.0))
                .child(
                    Row::new()
                        .gap(12.0)
                        .child(Text::new("Revenue").font_size(14.0))
                        .child(ProgressBar::new().value(0.75).fill_color(Color::GREEN)),
                )
                .child(
                    Row::new()
                        .gap(12.0)
                        .child(Text::new("Growth").font_size(14.0))
                        .child(ProgressBar::new().value(0.45).fill_color(Color::BLUE)),
                ),
        );

    // Measure and layout
    let viewport = Size::new(600.0, 600.0);
    let constraints = Constraints::loose(viewport);
    let size = dashboard.measure(constraints);
    let bounds = Rect::new(20.0, 20.0, size.width, size.height);
    dashboard.layout(bounds);

    // Paint to recording canvas
    let mut canvas = RecordingCanvas::new();
    dashboard.paint(&mut canvas);

    // Convert to SVG
    let svg_width = size.width + 40.0;
    let svg_height = size.height + 40.0;
    let svg = to_svg(canvas.commands(), svg_width, svg_height);

    // Write to file
    let output_path = "demo_output.svg";
    let mut file = File::create(output_path).expect("create file");
    file.write_all(svg.as_bytes()).expect("write file");

    println!("Generated {} draw commands", canvas.command_count());
    println!("\n✅ SVG output saved to: {output_path}");
    println!("\n📂 Open in browser:");
    println!("   firefox {output_path} &");
    println!("   google-chrome {output_path} &");
    println!("   xdg-open {output_path}");

    println!("\n=== Visual Demo Complete ===");
}
