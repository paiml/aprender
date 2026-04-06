//! Hello World example demonstrating basic Presentar widget usage.
//!
//! Run with: `cargo run --example hello_world`

#![allow(clippy::unwrap_used, clippy::disallowed_methods)]

use presentar::widgets::{row::MainAxisAlignment, Button, Column, Text};
use presentar::{Color, Constraints, RecordingCanvas, Rect, Size, Widget};

fn main() {
    println!("=== Presentar Hello World Example ===\n");

    // Create a simple UI with Text and Button widgets
    let mut ui = Column::new()
        .main_axis_alignment(MainAxisAlignment::Center)
        .gap(16.0)
        .child(
            Text::new("Hello, Presentar!")
                .font_size(24.0)
                .color(Color::from_hex("#1f2937").unwrap()),
        )
        .child(
            Text::new("A WASM-first visualization framework")
                .font_size(14.0)
                .color(Color::from_hex("#6b7280").unwrap()),
        )
        .child(
            Button::new("Click me!")
                .background(Color::from_hex("#4f46e5").unwrap())
                .text_color(Color::WHITE)
                .padding(12.0),
        );

    // Measure the UI
    let constraints = Constraints::loose(Size::new(400.0, 300.0));
    let size = ui.measure(constraints);
    println!("Measured UI size: {}x{}", size.width, size.height);

    // Layout the UI
    let bounds = Rect::new(0.0, 0.0, size.width, size.height);
    let result = ui.layout(bounds);
    println!(
        "Layout result: {}x{}",
        result.size.width, result.size.height
    );

    // Paint to a recording canvas (captures draw commands)
    let mut canvas = RecordingCanvas::new();
    ui.paint(&mut canvas);
    println!("\nGenerated {} draw commands:", canvas.command_count());

    for (i, cmd) in canvas.commands().iter().enumerate() {
        println!("  {}: {:?}", i + 1, cmd);
    }

    println!("\n=== Example Complete ===");
}
