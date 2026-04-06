#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! CPU Monitor Example
//!
//! Demonstrates real-time CPU usage monitoring with braille graphs.
//! Similar to btop/htop CPU visualization.
//!
//! Run with: cargo run -p presentar-terminal --example `cpu_monitor`

use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::{BrailleGraph, ColorMode, GraphMode};

fn main() {
    println!("=== CPU Monitor Example ===\n");

    // Simulate CPU metrics
    let cpu_history = simulate_cpu_history(60);
    let per_core = simulate_per_core_usage(8);

    // Create buffer (80x24 terminal)
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);

    // Draw CPU monitor UI
    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        // Background
        canvas.fill_rect(
            Rect::new(0.0, 0.0, 80.0, 24.0),
            Color::new(0.05, 0.05, 0.1, 1.0),
        );

        // Title
        let title_style = TextStyle {
            color: Color::new(0.4, 0.8, 1.0, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            "CPU Monitor - Real-time Usage",
            Point::new(2.0, 1.0),
            &title_style,
        );

        // Overall CPU graph (top section)
        draw_cpu_graph(&mut canvas, &cpu_history, Rect::new(2.0, 3.0, 50.0, 8.0));

        // Per-core meters (right side)
        draw_core_meters(&mut canvas, &per_core, 55.0, 3.0);

        // Statistics
        draw_statistics(&mut canvas, &cpu_history, 2.0, 13.0);

        // Load average
        draw_load_average(&mut canvas, 2.0, 18.0);
    }

    // Render output
    let mut output = Vec::with_capacity(8192);
    let cells_written = renderer.flush(&mut buffer, &mut output).unwrap();

    // Print statistics
    println!(
        "Buffer: {}x{} ({} cells)",
        buffer.width(),
        buffer.height(),
        buffer.len()
    );
    println!("Cells written: {cells_written}");
    println!("Output bytes: {}\n", output.len());

    // Display rendered output
    println!("Rendered output:");
    println!("{}", "─".repeat(82));
    std::io::Write::write_all(&mut std::io::stdout(), &output).unwrap();
    println!();
    println!("{}", "─".repeat(82));
}

fn draw_cpu_graph(canvas: &mut DirectTerminalCanvas<'_>, history: &[f64], bounds: Rect) {
    let label_style = TextStyle {
        color: Color::new(0.7, 0.7, 0.7, 1.0),
        ..Default::default()
    };
    canvas.draw_text("CPU Total [", Point::new(bounds.x, bounds.y), &label_style);

    let current = history.last().copied().unwrap_or(0.0);
    let pct_style = TextStyle {
        color: cpu_color(current),
        ..Default::default()
    };
    canvas.draw_text(
        &format!("{current:5.1}%"),
        Point::new(bounds.x + 11.0, bounds.y),
        &pct_style,
    );
    canvas.draw_text("]", Point::new(bounds.x + 17.0, bounds.y), &label_style);

    // Draw graph using BrailleGraph
    let mut graph = BrailleGraph::new(history.to_vec())
        .with_color(cpu_color(current))
        .with_range(0.0, 100.0)
        .with_mode(GraphMode::Braille);

    graph.layout(Rect::new(
        bounds.x,
        bounds.y + 1.0,
        bounds.width,
        bounds.height - 1.0,
    ));
    graph.paint(canvas);
}

fn draw_core_meters(canvas: &mut DirectTerminalCanvas<'_>, per_core: &[f64], x: f32, y: f32) {
    use std::fmt::Write;

    let label_style = TextStyle {
        color: Color::new(0.7, 0.7, 0.7, 1.0),
        ..Default::default()
    };
    canvas.draw_text("Per-Core Usage:", Point::new(x, y), &label_style);

    for (i, &usage) in per_core.iter().enumerate() {
        let core_y = y + 1.0 + i as f32;
        let color = cpu_color(usage);

        // Draw mini bar
        let label = format!("Core {i}: ");
        canvas.draw_text(&label, Point::new(x, core_y), &label_style);

        let bar_width = 12;
        let filled = ((usage / 100.0) * bar_width as f64).round() as usize;
        let mut bar = String::with_capacity(bar_width + 7);
        bar.push('[');
        for j in 0..bar_width {
            bar.push(if j < filled { '█' } else { '░' });
        }
        let _ = write!(bar, "] {usage:5.1}%");

        let bar_style = TextStyle {
            color,
            ..Default::default()
        };
        canvas.draw_text(&bar, Point::new(x + 8.0, core_y), &bar_style);
    }
}

fn draw_statistics(canvas: &mut DirectTerminalCanvas<'_>, history: &[f64], x: f32, y: f32) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    let value_style = TextStyle {
        color: Color::new(0.9, 0.9, 0.9, 1.0),
        ..Default::default()
    };

    let avg = history.iter().sum::<f64>() / history.len() as f64;
    let max = history.iter().fold(0.0_f64, |a, &b| a.max(b));
    let min = history.iter().fold(100.0_f64, |a, &b| a.min(b));

    canvas.draw_text("Statistics:", Point::new(x, y), &label_style);
    canvas.draw_text(
        &format!("Average: {avg:5.1}%"),
        Point::new(x, y + 1.0),
        &value_style,
    );
    canvas.draw_text(
        &format!("Maximum: {max:5.1}%"),
        Point::new(x, y + 2.0),
        &value_style,
    );
    canvas.draw_text(
        &format!("Minimum: {min:5.1}%"),
        Point::new(x, y + 3.0),
        &value_style,
    );

    canvas.draw_text(
        &format!("Samples:  {:>5}", history.len()),
        Point::new(x + 25.0, y + 1.0),
        &value_style,
    );
    canvas.draw_text(
        "Interval:  1.0s",
        Point::new(x + 25.0, y + 2.0),
        &value_style,
    );
}

fn draw_load_average(canvas: &mut DirectTerminalCanvas<'_>, x: f32, y: f32) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };

    // Simulated load average
    let load_1 = 2.45;
    let load_5 = 1.89;
    let load_15 = 1.52;

    canvas.draw_text("Load Average:", Point::new(x, y), &label_style);

    let load_style = TextStyle {
        color: Color::new(0.9, 0.7, 0.3, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        &format!("1min: {load_1:.2}  5min: {load_5:.2}  15min: {load_15:.2}"),
        Point::new(x, y + 1.0),
        &load_style,
    );

    // Uptime
    canvas.draw_text(
        "Uptime: 5d 12h 34m",
        Point::new(x + 45.0, y + 1.0),
        &label_style,
    );
}

fn cpu_color(usage: f64) -> Color {
    if usage > 90.0 {
        Color::new(1.0, 0.3, 0.3, 1.0) // Red
    } else if usage > 70.0 {
        Color::new(1.0, 0.7, 0.2, 1.0) // Orange
    } else if usage > 50.0 {
        Color::new(1.0, 1.0, 0.3, 1.0) // Yellow
    } else {
        Color::new(0.3, 1.0, 0.5, 1.0) // Green
    }
}

fn simulate_cpu_history(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let t = i as f64 / count as f64;
            let base = 20.0f64.mul_add((t * 3.0).sin(), 35.0);
            let noise = ((i * 7919 + 104_729) % 100) as f64 / 5.0;
            (base + noise).clamp(5.0, 95.0)
        })
        .collect()
}

fn simulate_per_core_usage(cores: usize) -> Vec<f64> {
    (0..cores)
        .map(|i| {
            let base = (i as f64).mul_add(10.0, 20.0);
            let noise = ((i * 6971 + 7723) % 40) as f64;
            (base + noise).clamp(5.0, 95.0)
        })
        .collect()
}
