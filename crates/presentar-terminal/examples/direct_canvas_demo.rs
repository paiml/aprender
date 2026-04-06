#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Direct Canvas Demo
//!
//! Demonstrates the zero-allocation direct terminal backend.
//!
//! Run with: cargo run -p presentar-terminal --example `direct_canvas_demo`

use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Transform2D};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::ColorMode;
use std::io::{stdout, Write};

fn main() {
    println!("=== Direct Canvas Demo ===\n");

    // Create a buffer (simulating an 40x12 terminal)
    let mut buffer = CellBuffer::new(40, 12);
    let mut renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);

    // Draw using the Canvas trait
    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        // Fill background
        canvas.fill_rect(
            Rect::new(0.0, 0.0, 40.0, 12.0),
            Color::new(0.1, 0.1, 0.2, 1.0),
        );

        // Draw a border
        canvas.stroke_rect(
            Rect::new(1.0, 1.0, 38.0, 10.0),
            Color::new(0.3, 0.6, 0.9, 1.0),
            1.0,
        );

        // Draw title
        let title_style = TextStyle {
            color: Color::new(1.0, 0.9, 0.3, 1.0),
            ..Default::default()
        };
        canvas.draw_text("Direct TUI Backend", Point::new(10.0, 2.0), &title_style);

        // Draw some content with transforms
        canvas.push_transform(Transform2D::translate(5.0, 4.0));

        let content_style = TextStyle {
            color: Color::WHITE,
            ..Default::default()
        };
        canvas.draw_text(
            "Zero-allocation rendering",
            Point::new(0.0, 0.0),
            &content_style,
        );
        canvas.draw_text(
            "Smart differential updates",
            Point::new(0.0, 1.0),
            &content_style,
        );
        canvas.draw_text(
            "Direct crossterm output",
            Point::new(0.0, 2.0),
            &content_style,
        );

        canvas.pop_transform();

        // Draw a colored bar
        for i in 0..30 {
            let hue = i as f32 / 30.0;
            let color = hsv_to_rgb(hue, 0.8, 0.9);
            canvas.fill_rect(Rect::new(5.0 + i as f32, 8.0, 1.0, 1.0), color);
        }
    }

    // Render to a string buffer (for demo purposes)
    let mut output = Vec::with_capacity(8192);
    let cells_written = renderer.flush(&mut buffer, &mut output).unwrap();

    // Print statistics
    println!(
        "Buffer size: {}x{} = {} cells",
        buffer.width(),
        buffer.height(),
        buffer.len()
    );
    println!("Cells written: {cells_written}");
    println!("Cursor moves: {}", renderer.cursor_moves());
    println!("Style changes: {}", renderer.style_changes());
    println!("Output bytes: {}", output.len());
    println!();

    // Print the rendered output
    println!("Rendered output:");
    println!("{}", "─".repeat(42));
    stdout().write_all(&output).unwrap();
    println!();
    println!("{}", "─".repeat(42));
}

/// Convert HSV to RGB color
#[allow(clippy::many_single_char_names)]
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color {
    let h = h * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = v * (1.0 - s);
    let q = v * s.mul_add(-f, 1.0);
    let t = v * s.mul_add(-(1.0 - f), 1.0);

    let (r, g, b) = match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };

    Color::new(r, g, b, 1.0)
}
