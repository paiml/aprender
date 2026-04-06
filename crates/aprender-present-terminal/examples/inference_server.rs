#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! ML Inference Server Monitor
//!
//! Demonstrates monitoring an ML inference server with request
//! latency, throughput, and model performance metrics.
//!
//! Run with: cargo run -p presentar-terminal --example `inference_server`

use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::{BrailleGraph, ColorMode, GraphMode};

fn main() {
    println!("=== ML Inference Server Monitor ===\n");

    // Simulate inference metrics
    let latency_history = simulate_latency(60);
    let throughput_history = simulate_throughput(60);
    let queue_history = simulate_queue_depth(60);

    let models = vec![
        ModelInfo::new("llama-3-70b", "loaded", 45.2, 12.5, 850, 15.2),
        ModelInfo::new("gpt-4-turbo", "loaded", 38.5, 8.2, 1200, 22.5),
        ModelInfo::new("whisper-large", "loaded", 12.8, 4.5, 320, 180.5),
        ModelInfo::new("stable-diffusion", "unloading", 8.2, 18.0, 45, 2500.0),
    ];

    // Create buffer
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);

    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        // Background
        canvas.fill_rect(
            Rect::new(0.0, 0.0, 80.0, 24.0),
            Color::new(0.02, 0.04, 0.06, 1.0),
        );

        // Title
        let title_style = TextStyle {
            color: Color::new(0.5, 0.9, 0.7, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            "Inference Server Monitor - triton-inference:8001",
            Point::new(2.0, 1.0),
            &title_style,
        );

        // Latency graph (P50/P99)
        draw_latency_panel(
            &mut canvas,
            &latency_history,
            Rect::new(2.0, 3.0, 36.0, 5.0),
        );

        // Throughput graph
        draw_throughput_panel(
            &mut canvas,
            &throughput_history,
            Rect::new(42.0, 3.0, 36.0, 5.0),
        );

        // Queue depth
        draw_queue_panel(&mut canvas, &queue_history, Rect::new(2.0, 9.0, 36.0, 4.0));

        // Request breakdown
        draw_request_breakdown(&mut canvas, 42.0, 9.0);

        // Model status table
        draw_model_table(&mut canvas, &models, 2.0, 14.0);

        // Footer
        draw_footer(&mut canvas);
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

struct ModelInfo {
    name: String,
    status: String,
    req_per_sec: f64,
    vram_gb: f64,
    queue: u32,
    avg_latency_ms: f64,
}

impl ModelInfo {
    fn new(name: &str, status: &str, rps: f64, vram: f64, queue: u32, latency: f64) -> Self {
        Self {
            name: name.to_string(),
            status: status.to_string(),
            req_per_sec: rps,
            vram_gb: vram,
            queue,
            avg_latency_ms: latency,
        }
    }

    fn status_color(&self) -> Color {
        match self.status.as_str() {
            "loaded" => Color::new(0.3, 1.0, 0.5, 1.0),
            "loading" => Color::new(0.9, 0.9, 0.3, 1.0),
            "unloading" => Color::new(0.9, 0.6, 0.3, 1.0),
            "error" => Color::new(1.0, 0.3, 0.3, 1.0),
            _ => Color::new(0.5, 0.5, 0.5, 1.0),
        }
    }
}

fn draw_latency_panel(canvas: &mut DirectTerminalCanvas<'_>, history: &[f64], bounds: Rect) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "Request Latency (ms)",
        Point::new(bounds.x, bounds.y),
        &label_style,
    );

    let p99 = history.last().copied().unwrap_or(0.0);
    let p50 = p99 * 0.6; // Simulated P50

    let p50_style = TextStyle {
        color: Color::new(0.3, 0.9, 0.5, 1.0),
        ..Default::default()
    };
    let p99_style = TextStyle {
        color: Color::new(0.9, 0.6, 0.3, 1.0),
        ..Default::default()
    };

    canvas.draw_text(
        &format!("P50:{p50:.0}"),
        Point::new(bounds.x + 22.0, bounds.y),
        &p50_style,
    );
    canvas.draw_text(
        &format!("P99:{p99:.0}"),
        Point::new(bounds.x + 30.0, bounds.y),
        &p99_style,
    );

    let color = if p99 > 200.0 {
        Color::new(1.0, 0.3, 0.3, 1.0)
    } else if p99 > 100.0 {
        Color::new(0.9, 0.6, 0.3, 1.0)
    } else {
        Color::new(0.3, 0.9, 0.5, 1.0)
    };

    let mut graph = BrailleGraph::new(history.to_vec())
        .with_color(color)
        .with_range(0.0, 300.0)
        .with_mode(GraphMode::Braille);

    graph.layout(Rect::new(
        bounds.x,
        bounds.y + 1.0,
        bounds.width,
        bounds.height - 1.0,
    ));
    graph.paint(canvas);
}

fn draw_throughput_panel(canvas: &mut DirectTerminalCanvas<'_>, history: &[f64], bounds: Rect) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "Throughput (req/s)",
        Point::new(bounds.x, bounds.y),
        &label_style,
    );

    let current = history.last().copied().unwrap_or(0.0);
    let value_style = TextStyle {
        color: Color::new(0.3, 0.7, 1.0, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        &format!("{current:.0}"),
        Point::new(bounds.x + 22.0, bounds.y),
        &value_style,
    );

    let max_val = history.iter().fold(100.0_f64, |a, &b| a.max(b));
    let mut graph = BrailleGraph::new(history.to_vec())
        .with_color(Color::new(0.3, 0.7, 1.0, 1.0))
        .with_range(0.0, max_val * 1.2)
        .with_mode(GraphMode::Braille);

    graph.layout(Rect::new(
        bounds.x,
        bounds.y + 1.0,
        bounds.width,
        bounds.height - 1.0,
    ));
    graph.paint(canvas);
}

fn draw_queue_panel(canvas: &mut DirectTerminalCanvas<'_>, history: &[f64], bounds: Rect) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "Request Queue Depth",
        Point::new(bounds.x, bounds.y),
        &label_style,
    );

    let current = history.last().copied().unwrap_or(0.0);
    let color = if current > 100.0 {
        Color::new(1.0, 0.3, 0.3, 1.0)
    } else if current > 50.0 {
        Color::new(0.9, 0.6, 0.3, 1.0)
    } else {
        Color::new(0.6, 0.3, 0.9, 1.0)
    };

    let value_style = TextStyle {
        color,
        ..Default::default()
    };
    canvas.draw_text(
        &format!("{current:.0}"),
        Point::new(bounds.x + 22.0, bounds.y),
        &value_style,
    );

    let mut graph = BrailleGraph::new(history.to_vec())
        .with_color(color)
        .with_range(0.0, 150.0)
        .with_mode(GraphMode::Block);

    graph.layout(Rect::new(
        bounds.x,
        bounds.y + 1.0,
        bounds.width,
        bounds.height - 1.0,
    ));
    graph.paint(canvas);
}

fn draw_request_breakdown(canvas: &mut DirectTerminalCanvas<'_>, x: f32, y: f32) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    canvas.draw_text("Request Status (1min):", Point::new(x, y), &label_style);

    let items = [
        ("Success", 15420, Color::new(0.3, 1.0, 0.5, 1.0)),
        ("Error", 23, Color::new(1.0, 0.3, 0.3, 1.0)),
        ("Timeout", 8, Color::new(0.9, 0.6, 0.3, 1.0)),
        ("Queued", 45, Color::new(0.6, 0.3, 0.9, 1.0)),
    ];

    for (i, (name, count, color)) in items.iter().enumerate() {
        let row_y = y + 1.0 + (i / 2) as f32;
        let col_x = ((i % 2) as f32).mul_add(18.0, x);

        let style = TextStyle {
            color: *color,
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{name}: {count}"),
            Point::new(col_x, row_y),
            &style,
        );
    }
}

fn draw_model_table(canvas: &mut DirectTerminalCanvas<'_>, models: &[ModelInfo], x: f32, y: f32) {
    let header_style = TextStyle {
        color: Color::new(0.5, 0.5, 0.5, 1.0),
        ..Default::default()
    };

    canvas.draw_text(
        "Model                  Status      RPS      VRAM     Queue    Latency(ms)",
        Point::new(x, y),
        &header_style,
    );
    canvas.draw_text(&"─".repeat(76), Point::new(x, y + 1.0), &header_style);

    for (i, model) in models.iter().enumerate() {
        let row_y = y + 2.0 + i as f32;

        // Name
        let name_style = TextStyle {
            color: Color::new(0.9, 0.9, 0.9, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:<20}", model.name),
            Point::new(x, row_y),
            &name_style,
        );

        // Status
        let status_style = TextStyle {
            color: model.status_color(),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:<10}", model.status),
            Point::new(x + 21.0, row_y),
            &status_style,
        );

        // RPS
        let rps_style = TextStyle {
            color: Color::new(0.3, 0.7, 1.0, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:>6.1}", model.req_per_sec),
            Point::new(x + 32.0, row_y),
            &rps_style,
        );

        // VRAM
        let vram_style = TextStyle {
            color: Color::new(0.6, 0.3, 0.9, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:>6.1}GB", model.vram_gb),
            Point::new(x + 40.0, row_y),
            &vram_style,
        );

        // Queue
        let queue_color = if model.queue > 500 {
            Color::new(1.0, 0.3, 0.3, 1.0)
        } else if model.queue > 100 {
            Color::new(0.9, 0.6, 0.3, 1.0)
        } else {
            Color::new(0.7, 0.7, 0.7, 1.0)
        };
        let queue_style = TextStyle {
            color: queue_color,
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:>8}", model.queue),
            Point::new(x + 51.0, row_y),
            &queue_style,
        );

        // Latency
        let latency_color = if model.avg_latency_ms > 500.0 {
            Color::new(1.0, 0.3, 0.3, 1.0)
        } else if model.avg_latency_ms > 100.0 {
            Color::new(0.9, 0.6, 0.3, 1.0)
        } else {
            Color::new(0.3, 0.9, 0.5, 1.0)
        };
        let latency_style = TextStyle {
            color: latency_color,
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:>10.1}", model.avg_latency_ms),
            Point::new(x + 62.0, row_y),
            &latency_style,
        );
    }
}

fn draw_footer(canvas: &mut DirectTerminalCanvas<'_>) {
    let label_style = TextStyle {
        color: Color::new(0.4, 0.4, 0.4, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "Server: 4x RTX 4090 | VRAM: 43.5/96 GB | Uptime: 12d 5h | Version: 2.38.0",
        Point::new(2.0, 20.0),
        &label_style,
    );
    canvas.draw_text(
        "[q] quit  [r] refresh  [l] logs  [m] models  [s] scaling  [h] help",
        Point::new(2.0, 21.0),
        &label_style,
    );
}

fn simulate_latency(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 30.0f64.mul_add((i as f64 / 15.0).sin(), 45.0);
            let spike = if i % 18 == 0 { 80.0 } else { 0.0 };
            let noise = ((i * 7919) % 30) as f64;
            (base + spike + noise).max(10.0)
        })
        .collect()
}

fn simulate_throughput(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 30.0f64.mul_add((i as f64 / 12.0).cos(), 100.0);
            let noise = ((i * 6971) % 40) as f64;
            (base + noise).max(20.0)
        })
        .collect()
}

fn simulate_queue_depth(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 20.0f64.mul_add((i as f64 / 8.0).sin(), 25.0);
            let spike = if i % 12 == 0 { 50.0 } else { 0.0 };
            let noise = ((i * 1103) % 20) as f64;
            (base + spike + noise).max(0.0)
        })
        .collect()
}
