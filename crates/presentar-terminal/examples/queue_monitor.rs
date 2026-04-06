#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Message Queue Monitor Example
//!
//! Demonstrates real-time monitoring of message queues with
//! depth, throughput, and latency visualization.
//!
//! Run with: cargo run -p presentar-terminal --example `queue_monitor`

use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::{BrailleGraph, ColorMode, GraphMode};

fn main() {
    println!("=== Message Queue Monitor ===\n");

    // Simulate queue metrics
    let queues = vec![
        QueueMetrics::new("orders.new", 1250, 450.5, 12.3),
        QueueMetrics::new("orders.processing", 89, 420.2, 45.8),
        QueueMetrics::new("notifications.email", 3420, 180.5, 120.5),
        QueueMetrics::new("notifications.sms", 156, 95.2, 85.3),
        QueueMetrics::new("analytics.events", 15600, 2500.0, 5.2),
        QueueMetrics::new("dlq.orders", 23, 2.1, 0.0),
    ];

    let throughput_history = simulate_throughput(60);
    let latency_history = simulate_latency(60);

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
            color: Color::new(0.6, 0.8, 1.0, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            "Message Queue Monitor - RabbitMQ Cluster",
            Point::new(2.0, 1.0),
            &title_style,
        );

        // Queue table
        draw_queue_table(&mut canvas, &queues, 2.0, 3.0);

        // Throughput graph
        draw_throughput_graph(
            &mut canvas,
            &throughput_history,
            Rect::new(2.0, 12.0, 36.0, 5.0),
        );

        // Latency graph
        draw_latency_graph(
            &mut canvas,
            &latency_history,
            Rect::new(42.0, 12.0, 36.0, 5.0),
        );

        // Summary stats
        draw_summary(&mut canvas, &queues, 2.0, 18.0);

        // Connection info
        draw_connection_info(&mut canvas, 2.0, 21.0);
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

struct QueueMetrics {
    name: String,
    depth: u64,
    msgs_per_sec: f64,
    avg_latency_ms: f64,
}

impl QueueMetrics {
    fn new(name: &str, depth: u64, msgs_per_sec: f64, avg_latency_ms: f64) -> Self {
        Self {
            name: name.to_string(),
            depth,
            msgs_per_sec,
            avg_latency_ms,
        }
    }

    fn depth_color(&self) -> Color {
        if self.depth > 10000 {
            Color::new(1.0, 0.3, 0.3, 1.0) // Red - critical
        } else if self.depth > 1000 {
            Color::new(1.0, 0.7, 0.2, 1.0) // Orange - warning
        } else if self.depth > 100 {
            Color::new(1.0, 1.0, 0.3, 1.0) // Yellow - elevated
        } else {
            Color::new(0.3, 1.0, 0.5, 1.0) // Green - healthy
        }
    }

    fn latency_color(&self) -> Color {
        if self.avg_latency_ms > 100.0 {
            Color::new(1.0, 0.3, 0.3, 1.0)
        } else if self.avg_latency_ms > 50.0 {
            Color::new(1.0, 0.7, 0.2, 1.0)
        } else {
            Color::new(0.3, 1.0, 0.5, 1.0)
        }
    }
}

fn draw_queue_table(
    canvas: &mut DirectTerminalCanvas<'_>,
    queues: &[QueueMetrics],
    x: f32,
    y: f32,
) {
    let header_style = TextStyle {
        color: Color::new(0.5, 0.5, 0.5, 1.0),
        ..Default::default()
    };

    canvas.draw_text(
        "Queue Name               Depth       Rate (msg/s)  Latency (ms)  Status",
        Point::new(x, y),
        &header_style,
    );
    canvas.draw_text(&"─".repeat(76), Point::new(x, y + 1.0), &header_style);

    for (i, queue) in queues.iter().enumerate() {
        let row_y = y + 2.0 + i as f32;

        // Name
        let name_style = TextStyle {
            color: Color::new(0.9, 0.9, 0.9, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:<22}", queue.name),
            Point::new(x, row_y),
            &name_style,
        );

        // Depth with color
        let depth_style = TextStyle {
            color: queue.depth_color(),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:>8}", queue.depth),
            Point::new(x + 23.0, row_y),
            &depth_style,
        );

        // Rate
        let rate_style = TextStyle {
            color: Color::new(0.3, 0.7, 1.0, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:>12.1}", queue.msgs_per_sec),
            Point::new(x + 34.0, row_y),
            &rate_style,
        );

        // Latency with color
        let latency_style = TextStyle {
            color: queue.latency_color(),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:>12.1}", queue.avg_latency_ms),
            Point::new(x + 48.0, row_y),
            &latency_style,
        );

        // Status indicator
        let status = if queue.name.starts_with("dlq.") {
            ("⚠", Color::new(1.0, 0.7, 0.2, 1.0))
        } else if queue.depth > 10000 {
            ("●", Color::new(1.0, 0.3, 0.3, 1.0))
        } else if queue.depth > 1000 {
            ("●", Color::new(1.0, 0.7, 0.2, 1.0))
        } else {
            ("●", Color::new(0.3, 1.0, 0.5, 1.0))
        };
        let status_style = TextStyle {
            color: status.1,
            ..Default::default()
        };
        canvas.draw_text(status.0, Point::new(x + 66.0, row_y), &status_style);
    }
}

fn draw_throughput_graph(canvas: &mut DirectTerminalCanvas<'_>, history: &[f64], bounds: Rect) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "Throughput (msgs/sec)",
        Point::new(bounds.x, bounds.y),
        &label_style,
    );

    let current = history.last().copied().unwrap_or(0.0);
    let value_style = TextStyle {
        color: Color::new(0.3, 0.9, 0.5, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        &format!("{current:.0}"),
        Point::new(bounds.x + 24.0, bounds.y),
        &value_style,
    );

    let max_val = history.iter().fold(100.0_f64, |a, &b| a.max(b));
    let mut graph = BrailleGraph::new(history.to_vec())
        .with_color(Color::new(0.3, 0.9, 0.5, 1.0))
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

fn draw_latency_graph(canvas: &mut DirectTerminalCanvas<'_>, history: &[f64], bounds: Rect) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "P99 Latency (ms)",
        Point::new(bounds.x, bounds.y),
        &label_style,
    );

    let current = history.last().copied().unwrap_or(0.0);
    let color = if current > 100.0 {
        Color::new(1.0, 0.3, 0.3, 1.0)
    } else if current > 50.0 {
        Color::new(1.0, 0.7, 0.2, 1.0)
    } else {
        Color::new(0.9, 0.7, 0.3, 1.0)
    };

    let value_style = TextStyle {
        color,
        ..Default::default()
    };
    canvas.draw_text(
        &format!("{current:.1}ms"),
        Point::new(bounds.x + 20.0, bounds.y),
        &value_style,
    );

    let max_val = history.iter().fold(50.0_f64, |a, &b| a.max(b));
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

fn draw_summary(canvas: &mut DirectTerminalCanvas<'_>, queues: &[QueueMetrics], x: f32, y: f32) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };

    let total_depth: u64 = queues.iter().map(|q| q.depth).sum();
    let total_rate: f64 = queues.iter().map(|q| q.msgs_per_sec).sum();
    let avg_latency: f64 =
        queues.iter().map(|q| q.avg_latency_ms).sum::<f64>() / queues.len() as f64;
    let dlq_depth: u64 = queues
        .iter()
        .filter(|q| q.name.starts_with("dlq."))
        .map(|q| q.depth)
        .sum();

    canvas.draw_text("Summary:", Point::new(x, y), &label_style);

    let value_style = TextStyle {
        color: Color::new(0.8, 0.8, 0.8, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        &format!(
            "Total Depth: {total_depth}  |  Total Rate: {total_rate:.0} msg/s  |  Avg Latency: {avg_latency:.1}ms  |  DLQ: {dlq_depth}"
        ),
        Point::new(x, y + 1.0),
        &value_style,
    );
}

fn draw_connection_info(canvas: &mut DirectTerminalCanvas<'_>, x: f32, y: f32) {
    let label_style = TextStyle {
        color: Color::new(0.4, 0.4, 0.4, 1.0),
        ..Default::default()
    };

    canvas.draw_text(
        "Cluster: rabbitmq-prod (3 nodes)  |  Connections: 156  |  Channels: 312  |  Consumers: 48",
        Point::new(x, y),
        &label_style,
    );
    canvas.draw_text(
        "[q] quit  [r] refresh  [d] details  [p] purge  [h] help",
        Point::new(x, y + 1.0),
        &label_style,
    );
}

fn simulate_throughput(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 500.0f64.mul_add((i as f64 / 10.0).sin(), 3000.0);
            let noise = ((i * 7919) % 300) as f64;
            (base + noise).max(100.0)
        })
        .collect()
}

fn simulate_latency(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 15.0f64.mul_add((i as f64 / 8.0).sin(), 25.0);
            let spike = if i % 20 == 0 { 40.0 } else { 0.0 };
            let noise = ((i * 6971) % 20) as f64;
            (base + spike + noise).max(5.0)
        })
        .collect()
}
