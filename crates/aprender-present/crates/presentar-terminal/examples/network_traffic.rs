#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Network Traffic Monitor Example
//!
//! Demonstrates real-time network RX/TX monitoring with dual graphs.
//! Similar to btop/nethogs network visualization.
//!
//! Run with: cargo run -p presentar-terminal --example `network_traffic`

use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::{BrailleGraph, ColorMode, GraphMode};

fn main() {
    println!("=== Network Traffic Monitor ===\n");

    // Simulate network metrics (in MB/s)
    let rx_history = simulate_rx_history(60);
    let tx_history = simulate_tx_history(60);

    // Interface stats
    let interfaces = vec![
        ("eth0", 125.4, 45.2, true),
        ("wlan0", 0.0, 0.0, false),
        ("docker0", 12.3, 8.7, true),
        ("lo", 0.5, 0.5, true),
    ];

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
            color: Color::new(0.8, 0.6, 1.0, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            "Network Traffic Monitor",
            Point::new(2.0, 1.0),
            &title_style,
        );

        // RX Graph
        draw_traffic_graph(
            &mut canvas,
            "Download (RX)",
            &rx_history,
            Rect::new(2.0, 3.0, 36.0, 7.0),
            Color::new(0.3, 0.9, 0.5, 1.0),
        );

        // TX Graph
        draw_traffic_graph(
            &mut canvas,
            "Upload (TX)",
            &tx_history,
            Rect::new(42.0, 3.0, 36.0, 7.0),
            Color::new(0.9, 0.5, 0.3, 1.0),
        );

        // Interface table
        draw_interface_table(&mut canvas, &interfaces, 2.0, 11.0);

        // Totals
        draw_totals(&mut canvas, &rx_history, &tx_history, 2.0, 19.0);

        // Connection stats
        draw_connection_stats(&mut canvas, 50.0, 19.0);
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

fn draw_traffic_graph(
    canvas: &mut DirectTerminalCanvas<'_>,
    title: &str,
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

    let current = history.last().copied().unwrap_or(0.0);
    canvas.draw_text(title, Point::new(bounds.x, bounds.y), &label_style);
    canvas.draw_text(
        &format!("{current:7.2} MB/s"),
        Point::new(bounds.x + bounds.width - 12.0, bounds.y),
        &value_style,
    );

    // Draw graph
    let max_val = history.iter().fold(10.0_f64, |a, &b| a.max(b));
    let mut graph = BrailleGraph::new(history.to_vec())
        .with_color(color)
        .with_range(0.0, max_val * 1.1)
        .with_mode(GraphMode::Braille);

    graph.layout(Rect::new(
        bounds.x,
        bounds.y + 1.0,
        bounds.width,
        bounds.height - 1.0,
    ));
    graph.paint(canvas);
}

fn draw_interface_table(
    canvas: &mut DirectTerminalCanvas<'_>,
    interfaces: &[(&str, f64, f64, bool)],
    x: f32,
    y: f32,
) {
    let header_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };

    canvas.draw_text("Interface Statistics:", Point::new(x, y), &header_style);
    canvas.draw_text(
        "Interface      Status       RX (MB/s)    TX (MB/s)    Total",
        Point::new(x, y + 1.0),
        &header_style,
    );
    canvas.draw_text(&"─".repeat(70), Point::new(x, y + 2.0), &header_style);

    for (i, (name, rx, tx, up)) in interfaces.iter().enumerate() {
        let row_y = y + 3.0 + i as f32;

        let status_color = if *up {
            Color::new(0.3, 1.0, 0.5, 1.0)
        } else {
            Color::new(0.5, 0.5, 0.5, 1.0)
        };
        let status = if *up { "UP  " } else { "DOWN" };

        let name_style = TextStyle {
            color: Color::new(0.9, 0.9, 0.9, 1.0),
            ..Default::default()
        };
        let status_style = TextStyle {
            color: status_color,
            ..Default::default()
        };
        let value_style = TextStyle {
            color: Color::new(0.8, 0.8, 0.8, 1.0),
            ..Default::default()
        };

        canvas.draw_text(&format!("{name:<12}"), Point::new(x, row_y), &name_style);
        canvas.draw_text(status, Point::new(x + 15.0, row_y), &status_style);
        canvas.draw_text(
            &format!("{rx:>10.2}"),
            Point::new(x + 26.0, row_y),
            &value_style,
        );
        canvas.draw_text(
            &format!("{tx:>10.2}"),
            Point::new(x + 39.0, row_y),
            &value_style,
        );
        canvas.draw_text(
            &format!("{:>10.2}", rx + tx),
            Point::new(x + 52.0, row_y),
            &value_style,
        );
    }
}

fn draw_totals(canvas: &mut DirectTerminalCanvas<'_>, rx: &[f64], tx: &[f64], x: f32, y: f32) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };

    let total_rx: f64 = rx.iter().sum();
    let total_tx: f64 = tx.iter().sum();
    let avg_rx = total_rx / rx.len() as f64;
    let avg_tx = total_tx / tx.len() as f64;

    canvas.draw_text("Session Totals:", Point::new(x, y), &label_style);

    let rx_style = TextStyle {
        color: Color::new(0.3, 0.9, 0.5, 1.0),
        ..Default::default()
    };
    let tx_style = TextStyle {
        color: Color::new(0.9, 0.5, 0.3, 1.0),
        ..Default::default()
    };

    canvas.draw_text(
        &format!("RX: {total_rx:.1} MB (avg: {avg_rx:.2} MB/s)"),
        Point::new(x, y + 1.0),
        &rx_style,
    );
    canvas.draw_text(
        &format!("TX: {total_tx:.1} MB (avg: {avg_tx:.2} MB/s)"),
        Point::new(x, y + 2.0),
        &tx_style,
    );
}

fn draw_connection_stats(canvas: &mut DirectTerminalCanvas<'_>, x: f32, y: f32) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    let value_style = TextStyle {
        color: Color::new(0.8, 0.8, 0.8, 1.0),
        ..Default::default()
    };

    canvas.draw_text("Connections:", Point::new(x, y), &label_style);
    canvas.draw_text("ESTABLISHED: 42", Point::new(x, y + 1.0), &value_style);
    canvas.draw_text("LISTEN:      12", Point::new(x, y + 2.0), &value_style);
    canvas.draw_text("TIME_WAIT:    8", Point::new(x, y + 3.0), &value_style);
}

fn simulate_rx_history(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 30.0f64.mul_add((i as f64 / 10.0).sin(), 50.0);
            let spike = if i % 15 == 0 { 40.0 } else { 0.0 };
            let noise = ((i * 7919) % 20) as f64;
            (base + spike + noise).max(0.0)
        })
        .collect()
}

fn simulate_tx_history(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 10.0f64.mul_add((i as f64 / 8.0).cos(), 15.0);
            let noise = ((i * 6971) % 15) as f64;
            (base + noise).max(0.0)
        })
        .collect()
}
