#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Real-time ttop-style CPU pane demo.
//!
//! Run with: cargo run -p presentar-terminal --example `ttop_cpu_pane`

use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::widgets::{BrailleGraph, CpuGrid, GraphMode};
use presentar_terminal::ColorMode;
use std::io::{self, Write};
use std::time::{Duration, Instant};

fn main() {
    // Setup terminal
    print!("\x1b[?25l"); // Hide cursor
    print!("\x1b[2J"); // Clear screen
    print!("\x1b[H"); // Move to top-left
    io::stdout().flush().unwrap();

    let width = 60;
    let height = 12;

    let mut buffer = CellBuffer::new(width, height);
    let mut renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);
    let mut cpu_history: Vec<f64> = vec![0.0; 60];
    let start = Instant::now();

    // Simulated 8-core CPU usage
    let mut core_usage: [f64; 8] = [0.0; 8];

    loop {
        let elapsed = start.elapsed().as_secs_f64();

        // Simulate varying CPU loads per core
        for (i, usage) in core_usage.iter_mut().enumerate() {
            let phase = i as f64 * 0.5;
            let base = (i as f64).mul_add(8.0, 20.0);
            *usage = 30.0f64.mul_add(elapsed.mul_add(0.5, phase).sin().abs(), base);
            *usage = usage.clamp(5.0, 95.0);
        }

        // Calculate average for history graph
        let avg: f64 = core_usage.iter().sum::<f64>() / 8.0;
        cpu_history.remove(0);
        cpu_history.push(avg);

        // Clear buffer
        buffer.clear();

        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);

            // Draw panel border (ttop style with rounded corners)
            draw_panel(
                &mut canvas,
                "CPU",
                Rect::new(0.0, 0.0, f32::from(width), f32::from(height)),
            );

            // Draw CPU grid (per-core sparklines)
            let grid_values: Vec<f64> = core_usage.to_vec();
            let mut cpu_grid = CpuGrid::new(grid_values).with_columns(8).compact();
            cpu_grid.layout(Rect::new(2.0, 2.0, 50.0, 2.0));
            cpu_grid.paint(&mut canvas);

            // Draw braille graph (history)
            let mut graph = BrailleGraph::new(cpu_history.clone())
                .with_mode(GraphMode::Braille)
                .with_range(0.0, 100.0)
                .with_color(Color::new(0.3, 0.8, 0.5, 1.0));
            graph.layout(Rect::new(2.0, 5.0, 56.0, 4.0));
            graph.paint(&mut canvas);

            // Draw average percentage
            let avg_text = format!("Avg: {avg:5.1}%");
            canvas.draw_text(
                &avg_text,
                Point::new(2.0, 10.0),
                &TextStyle {
                    color: Color::new(0.4, 0.9, 0.6, 1.0),
                    ..Default::default()
                },
            );

            // Draw uptime
            let uptime = format!("Uptime: {elapsed:.1}s");
            canvas.draw_text(
                &uptime,
                Point::new(45.0, 10.0),
                &TextStyle {
                    color: Color::new(0.6, 0.6, 0.6, 1.0),
                    ..Default::default()
                },
            );
        }

        // Render to terminal
        print!("\x1b[H"); // Move cursor to top-left
        let mut output = Vec::with_capacity(4096);
        renderer.flush(&mut buffer, &mut output).unwrap();
        io::stdout().write_all(&output).unwrap();
        io::stdout().flush().unwrap();

        // ~30 FPS
        std::thread::sleep(Duration::from_millis(33));

        // Exit after 30 seconds or on any key (simplified: just timeout)
        if elapsed > 30.0 {
            break;
        }
    }

    // Cleanup
    print!("\x1b[?25h"); // Show cursor
    print!("\x1b[2J\x1b[H"); // Clear and home
    println!("Demo complete!");
}

fn draw_panel(canvas: &mut DirectTerminalCanvas, title: &str, bounds: Rect) {
    let border_color = Color::new(0.4, 0.6, 0.8, 1.0);
    let title_color = Color::new(0.6, 0.8, 1.0, 1.0);

    let style = TextStyle {
        color: border_color,
        ..Default::default()
    };
    let title_style = TextStyle {
        color: title_color,
        ..Default::default()
    };

    let w = bounds.width as usize;
    let h = bounds.height as usize;

    // Top border: ╭─ TITLE ─────────────────╮
    canvas.draw_text("╭", Point::new(bounds.x, bounds.y), &style);
    canvas.draw_text("─", Point::new(bounds.x + 1.0, bounds.y), &style);
    canvas.draw_text(
        &format!(" {title} "),
        Point::new(bounds.x + 2.0, bounds.y),
        &title_style,
    );
    let title_end = 2 + title.len() + 2;
    for x in title_end..w - 1 {
        canvas.draw_text("─", Point::new(bounds.x + x as f32, bounds.y), &style);
    }
    canvas.draw_text("╮", Point::new(bounds.x + (w - 1) as f32, bounds.y), &style);

    // Side borders
    for y in 1..h - 1 {
        canvas.draw_text("│", Point::new(bounds.x, bounds.y + y as f32), &style);
        canvas.draw_text(
            "│",
            Point::new(bounds.x + (w - 1) as f32, bounds.y + y as f32),
            &style,
        );
    }

    // Bottom border: ╰──────────────────────────╯
    canvas.draw_text("╰", Point::new(bounds.x, bounds.y + (h - 1) as f32), &style);
    for x in 1..w - 1 {
        canvas.draw_text(
            "─",
            Point::new(bounds.x + x as f32, bounds.y + (h - 1) as f32),
            &style,
        );
    }
    canvas.draw_text(
        "╯",
        Point::new(bounds.x + (w - 1) as f32, bounds.y + (h - 1) as f32),
        &style,
    );
}
