//! Demonstrates basic presentar-core types: geometry, colors, constraints,
//! and the `RecordingCanvas`.
//!
//! This example runs in native Rust (no WASM required) and shows how the
//! core building blocks fit together.
//!
//! Run with: `cargo run --example hello_widget -p presentar-core`

use presentar_core::widget::Canvas;
use presentar_core::{Color, Constraints, Point, RecordingCanvas, Rect, Size};

fn main() {
    println!("=== Presentar Core Widget Demo ===\n");

    // --- Geometry primitives ---
    let origin = Point::new(0.0, 0.0);
    let size = Size::new(200.0, 100.0);
    let rect = Rect::new(origin.x, origin.y, size.width, size.height);
    println!(
        "Rect: ({}, {}) {}x{}",
        rect.x, rect.y, rect.width, rect.height
    );

    // --- Colors ---
    let red = Color::RED;
    let blue = Color::BLUE;
    let custom = Color::from_hex("#6366f1").expect("valid hex color");
    println!("Red:    rgba({}, {}, {}, {})", red.r, red.g, red.b, red.a);
    println!(
        "Blue:   rgba({}, {}, {}, {})",
        blue.r, blue.g, blue.b, blue.a
    );
    println!(
        "Custom: rgba({:.2}, {:.2}, {:.2}, {:.2})",
        custom.r, custom.g, custom.b, custom.a
    );

    // WCAG contrast ratio (accessibility)
    let white = Color::WHITE;
    let black = Color::BLACK;
    let contrast = white.contrast_ratio(&black);
    println!("White/Black contrast ratio: {contrast:.1}:1 (WCAG AAA requires 7:1)");

    // --- Constraints ---
    let constraints = Constraints::new(0.0, 800.0, 0.0, 600.0);
    let desired = Size::new(1000.0, 400.0);
    let bounded = constraints.constrain(desired);
    println!(
        "\nConstrained {}x{} to {}x{}",
        desired.width, desired.height, bounded.width, bounded.height
    );

    // --- RecordingCanvas ---
    let mut canvas = RecordingCanvas::new();
    canvas.fill_rect(rect, red);
    canvas.draw_text(
        "Hello, Presentar!",
        Point::new(10.0, 30.0),
        &presentar_core::widget::TextStyle {
            size: 16.0,
            color: white,
            weight: presentar_core::widget::FontWeight::Bold,
            ..Default::default()
        },
    );
    canvas.fill_circle(Point::new(100.0, 50.0), 20.0, blue);

    println!(
        "\nRecordingCanvas captured {} draw commands:",
        canvas.command_count()
    );
    for (i, cmd) in canvas.commands().iter().enumerate() {
        println!("  [{i}] {cmd:?}");
    }

    println!("\nDone.");
}
