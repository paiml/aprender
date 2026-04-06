//! Form Inputs example demonstrating interactive widgets.
//!
//! Run with: `cargo run --example form_inputs`

use presentar::widgets::{Button, Checkbox, Column, Row, Slider, Text, TextInput, Toggle};
use presentar::{Color, Constraints, DrawCommand, RecordingCanvas, Rect, Size, Widget};

fn main() {
    println!("=== Presentar Form Inputs Example ===\n");

    // Build form layout with various input widgets
    let mut form = Column::new()
        .gap(20.0)
        .child(
            Text::new("User Settings")
                .font_size(24.0)
                .color(Color::from_hex("#1f2937").expect("hex")),
        )
        // Text input
        .child(
            Column::new()
                .gap(8.0)
                .child(Text::new("Username").font_size(14.0))
                .child(TextInput::new().placeholder("Enter your username")),
        )
        // Checkbox
        .child(Checkbox::new().label("Enable dark mode").checked(true))
        // Toggle
        .child(
            Row::new()
                .gap(12.0)
                .child(Text::new("Notifications").font_size(14.0))
                .child(Toggle::new().on(true)),
        )
        // Slider
        .child(
            Column::new()
                .gap(8.0)
                .child(Text::new("Volume").font_size(14.0))
                .child(
                    Row::new()
                        .gap(12.0)
                        .child(Slider::new().min(0.0).max(100.0).value(75.0))
                        .child(Text::new("75%").font_size(14.0)),
                ),
        )
        // Buttons
        .child(
            Row::new()
                .gap(12.0)
                .child(
                    Button::new("Save")
                        .background(Color::from_hex("#4f46e5").expect("hex"))
                        .text_color(Color::WHITE)
                        .padding(12.0),
                )
                .child(
                    Button::new("Cancel")
                        .background(Color::from_hex("#e5e7eb").expect("hex"))
                        .text_color(Color::from_hex("#374151").expect("hex"))
                        .padding(12.0),
                ),
        );

    // Measure and layout
    let constraints = Constraints::loose(Size::new(400.0, 600.0));
    let size = form.measure(constraints);
    println!("Form size: {}x{}", size.width, size.height);

    let bounds = Rect::new(0.0, 0.0, size.width, size.height);
    let result = form.layout(bounds);
    println!(
        "Layout complete: {}x{}",
        result.size.width, result.size.height
    );

    // Paint and analyze commands
    let mut canvas = RecordingCanvas::new();
    form.paint(&mut canvas);
    println!("\nGenerated {} draw commands", canvas.command_count());

    // Print summary
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

    println!("\n=== Form Inputs Example Complete ===");
}
