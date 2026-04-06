#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Memory Monitor Example
//!
//! Demonstrates memory usage visualization with meters and graphs.
//! Similar to btop/htop memory panels.
//!
//! Run with: cargo run -p presentar-terminal --example `memory_monitor`

use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::{BrailleGraph, ColorMode, GraphMode};

fn main() {
    println!("=== Memory Monitor Example ===\n");

    // Simulate memory metrics
    let mem_history = simulate_memory_history(60);
    let swap_history = simulate_swap_history(60);

    // Memory breakdown (in GB)
    let total_mem = 32.0;
    let used_mem = 18.5;
    let cached = 8.2;
    let buffers = 2.1;
    let available = total_mem - used_mem;

    let total_swap = 8.0;
    let used_swap = 1.2;

    // Create buffer
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);

    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        // Background
        canvas.fill_rect(
            Rect::new(0.0, 0.0, 80.0, 24.0),
            Color::new(0.05, 0.05, 0.1, 1.0),
        );

        // Title
        let title_style = TextStyle {
            color: Color::new(0.6, 0.9, 0.6, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            "Memory Monitor - RAM & Swap Usage",
            Point::new(2.0, 1.0),
            &title_style,
        );

        // RAM section
        draw_memory_section(
            &mut canvas,
            "RAM",
            used_mem,
            total_mem,
            &mem_history,
            Rect::new(2.0, 3.0, 76.0, 8.0),
            Color::new(0.3, 0.7, 1.0, 1.0),
        );

        // Memory breakdown
        draw_memory_breakdown(
            &mut canvas,
            used_mem,
            cached,
            buffers,
            available,
            total_mem,
            2.0,
            11.0,
        );

        // Swap section
        draw_memory_section(
            &mut canvas,
            "Swap",
            used_swap,
            total_swap,
            &swap_history,
            Rect::new(2.0, 15.0, 76.0, 5.0),
            Color::new(0.9, 0.6, 0.3, 1.0),
        );

        // Process memory usage (top consumers)
        draw_top_processes(&mut canvas, 2.0, 21.0);
    }

    // Render
    let mut output = Vec::with_capacity(8192);
    let cells_written = renderer.flush(&mut buffer, &mut output).unwrap();

    println!("Buffer: {}x{}", buffer.width(), buffer.height());
    println!("Cells written: {cells_written}");
    println!("Output bytes: {}\n", output.len());

    println!("Rendered output:");
    println!("{}", "─".repeat(82));
    std::io::Write::write_all(&mut std::io::stdout(), &output).unwrap();
    println!();
    println!("{}", "─".repeat(82));
}

fn draw_memory_section(
    canvas: &mut DirectTerminalCanvas<'_>,
    label: &str,
    used: f64,
    total: f64,
    history: &[f64],
    bounds: Rect,
    color: Color,
) {
    let label_style = TextStyle {
        color: Color::new(0.7, 0.7, 0.7, 1.0),
        ..Default::default()
    };
    let value_style = TextStyle {
        color,
        ..Default::default()
    };

    let pct = (used / total) * 100.0;
    canvas.draw_text(
        &format!("{label} [{pct:5.1}% - {used:.1}/{total:.1} GB]"),
        Point::new(bounds.x, bounds.y),
        &label_style,
    );

    // Draw meter bar
    let bar_y = bounds.y + 1.0;
    let bar_width = 50;
    let filled = ((pct / 100.0) * bar_width as f64).round() as usize;

    let mut bar = String::with_capacity(bar_width + 2);
    bar.push('[');
    for i in 0..bar_width {
        if i < filled {
            bar.push('█');
        } else {
            bar.push('░');
        }
    }
    bar.push(']');

    canvas.draw_text(&bar, Point::new(bounds.x, bar_y), &value_style);

    // Draw history graph
    if bounds.height > 3.0 {
        let graph_data: Vec<f64> = history.iter().map(|&v| (v / total) * 100.0).collect();
        let mut graph = BrailleGraph::new(graph_data)
            .with_color(color)
            .with_range(0.0, 100.0)
            .with_mode(GraphMode::Block);

        graph.layout(Rect::new(
            bounds.x + 55.0,
            bounds.y,
            bounds.width - 55.0,
            bounds.height - 1.0,
        ));
        graph.paint(canvas);
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_memory_breakdown(
    canvas: &mut DirectTerminalCanvas<'_>,
    used: f64,
    cached: f64,
    buffers: f64,
    available: f64,
    total: f64,
    x: f32,
    y: f32,
) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };

    canvas.draw_text("Memory Breakdown:", Point::new(x, y), &label_style);

    let items = [
        ("Used", used, Color::new(1.0, 0.4, 0.4, 1.0)),
        ("Cached", cached, Color::new(0.4, 0.8, 1.0, 1.0)),
        ("Buffers", buffers, Color::new(0.8, 0.6, 1.0, 1.0)),
        ("Available", available, Color::new(0.4, 1.0, 0.6, 1.0)),
    ];

    for (i, (name, value, color)) in items.iter().enumerate() {
        let col = (i % 2) as f32 * 38.0;
        let row = (i / 2) as f32;

        let item_style = TextStyle {
            color: *color,
            ..Default::default()
        };

        let pct = (value / total) * 100.0;
        canvas.draw_text(
            &format!("{name:>10}: {value:6.2} GB ({pct:5.1}%)"),
            Point::new(x + col, y + 1.0 + row),
            &item_style,
        );
    }
}

fn draw_top_processes(canvas: &mut DirectTerminalCanvas<'_>, x: f32, y: f32) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    let value_style = TextStyle {
        color: Color::new(0.9, 0.9, 0.9, 1.0),
        ..Default::default()
    };

    canvas.draw_text("Top Memory Consumers:", Point::new(x, y), &label_style);
    canvas.draw_text(
        "1. firefox (2.8GB)  2. code (1.5GB)  3. slack (0.9GB)  4. docker (0.7GB)",
        Point::new(x, y + 1.0),
        &value_style,
    );
}

fn simulate_memory_history(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 4.0f64.mul_add((i as f64 / 20.0).sin(), 16.0);
            let noise = ((i * 7919) % 20) as f64 / 10.0;
            base + noise
        })
        .collect()
}

fn simulate_swap_history(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 0.5f64.mul_add((i as f64 / 15.0).sin(), 0.8);
            let noise = ((i * 6971) % 10) as f64 / 20.0;
            base + noise
        })
        .collect()
}
