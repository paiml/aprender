#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Kubernetes Cluster Status Monitor
//!
//! Demonstrates monitoring a Kubernetes cluster with node status,
//! pod health, and resource utilization.
//!
//! Run with: cargo run -p presentar-terminal --example `cluster_status`

use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::{BrailleGraph, ColorMode, GraphMode};

fn main() {
    println!("=== Kubernetes Cluster Monitor ===\n");

    // Simulate cluster metrics
    let cpu_history = simulate_cluster_cpu(60);
    let mem_history = simulate_cluster_mem(60);

    let nodes = vec![
        NodeInfo::new("worker-01", "Ready", 78.5, 82.3, 45, 8, 64),
        NodeInfo::new("worker-02", "Ready", 65.2, 71.8, 38, 8, 64),
        NodeInfo::new("worker-03", "Ready", 92.1, 88.5, 52, 8, 64),
        NodeInfo::new("worker-04", "NotReady", 0.0, 45.2, 12, 8, 64),
        NodeInfo::new("master-01", "Ready", 25.3, 35.8, 18, 4, 32),
    ];

    // Create buffer
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);

    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        // Background
        canvas.fill_rect(
            Rect::new(0.0, 0.0, 80.0, 24.0),
            Color::new(0.02, 0.02, 0.06, 1.0),
        );

        // Title
        let title_style = TextStyle {
            color: Color::new(0.3, 0.6, 1.0, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            "Kubernetes Cluster: prod-us-east-1",
            Point::new(2.0, 1.0),
            &title_style,
        );

        // Cluster CPU graph
        draw_resource_graph(
            &mut canvas,
            "Cluster CPU",
            &cpu_history,
            Rect::new(2.0, 3.0, 36.0, 5.0),
            Color::new(0.3, 0.8, 0.5, 1.0),
        );

        // Cluster Memory graph
        draw_resource_graph(
            &mut canvas,
            "Cluster Memory",
            &mem_history,
            Rect::new(42.0, 3.0, 36.0, 5.0),
            Color::new(0.5, 0.3, 0.9, 1.0),
        );

        // Node table
        draw_node_table(&mut canvas, &nodes, 2.0, 9.0);

        // Pod summary
        draw_pod_summary(&mut canvas, 2.0, 17.0);

        // Namespace breakdown
        draw_namespace_breakdown(&mut canvas, 42.0, 17.0);

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

struct NodeInfo {
    name: String,
    status: String,
    cpu_pct: f64,
    mem_pct: f64,
    pods: u32,
    cores: u32,
    mem_gb: u32,
}

impl NodeInfo {
    fn new(
        name: &str,
        status: &str,
        cpu: f64,
        mem: f64,
        pods: u32,
        cores: u32,
        mem_gb: u32,
    ) -> Self {
        Self {
            name: name.to_string(),
            status: status.to_string(),
            cpu_pct: cpu,
            mem_pct: mem,
            pods,
            cores,
            mem_gb,
        }
    }

    fn status_color(&self) -> Color {
        match self.status.as_str() {
            "Ready" => Color::new(0.3, 1.0, 0.5, 1.0),
            "NotReady" => Color::new(1.0, 0.3, 0.3, 1.0),
            "Unknown" => Color::new(0.9, 0.6, 0.3, 1.0),
            _ => Color::new(0.5, 0.5, 0.5, 1.0),
        }
    }

    fn resource_color(pct: f64) -> Color {
        if pct > 90.0 {
            Color::new(1.0, 0.3, 0.3, 1.0)
        } else if pct > 75.0 {
            Color::new(0.9, 0.6, 0.3, 1.0)
        } else if pct > 50.0 {
            Color::new(0.9, 0.9, 0.3, 1.0)
        } else {
            Color::new(0.3, 0.9, 0.5, 1.0)
        }
    }
}

fn draw_resource_graph(
    canvas: &mut DirectTerminalCanvas<'_>,
    title: &str,
    history: &[f64],
    bounds: Rect,
    color: Color,
) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    canvas.draw_text(title, Point::new(bounds.x, bounds.y), &label_style);

    let current = history.last().copied().unwrap_or(0.0);
    let value_style = TextStyle {
        color,
        ..Default::default()
    };
    canvas.draw_text(
        &format!("{current:5.1}%"),
        Point::new(bounds.x + bounds.width - 8.0, bounds.y),
        &value_style,
    );

    let mut graph = BrailleGraph::new(history.to_vec())
        .with_color(color)
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

fn draw_node_table(canvas: &mut DirectTerminalCanvas<'_>, nodes: &[NodeInfo], x: f32, y: f32) {
    let header_style = TextStyle {
        color: Color::new(0.5, 0.5, 0.5, 1.0),
        ..Default::default()
    };

    canvas.draw_text(
        "Node             Status     CPU              Memory           Pods   Resources",
        Point::new(x, y),
        &header_style,
    );
    canvas.draw_text(&"─".repeat(76), Point::new(x, y + 1.0), &header_style);

    for (i, node) in nodes.iter().enumerate() {
        let row_y = y + 2.0 + i as f32;

        // Name
        let name_style = TextStyle {
            color: Color::new(0.9, 0.9, 0.9, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:<14}", node.name),
            Point::new(x, row_y),
            &name_style,
        );

        // Status
        let status_style = TextStyle {
            color: node.status_color(),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:<9}", node.status),
            Point::new(x + 15.0, row_y),
            &status_style,
        );

        // CPU bar
        draw_inline_meter(
            canvas,
            node.cpu_pct,
            x + 25.0,
            row_y,
            10,
            NodeInfo::resource_color(node.cpu_pct),
        );

        // Memory bar
        draw_inline_meter(
            canvas,
            node.mem_pct,
            x + 42.0,
            row_y,
            10,
            NodeInfo::resource_color(node.mem_pct),
        );

        // Pods
        let pods_style = TextStyle {
            color: Color::new(0.7, 0.7, 0.7, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:>4}", node.pods),
            Point::new(x + 57.0, row_y),
            &pods_style,
        );

        // Resources
        canvas.draw_text(
            &format!("{}c/{}G", node.cores, node.mem_gb),
            Point::new(x + 64.0, row_y),
            &pods_style,
        );
    }
}

fn draw_inline_meter(
    canvas: &mut DirectTerminalCanvas<'_>,
    pct: f64,
    x: f32,
    y: f32,
    width: usize,
    color: Color,
) {
    use std::fmt::Write;
    let filled = ((pct / 100.0) * width as f64).round() as usize;
    let mut bar = String::with_capacity(width + 8);
    for i in 0..width {
        bar.push(if i < filled { '█' } else { '░' });
    }
    let _ = write!(bar, " {pct:>3.0}%");

    let style = TextStyle {
        color,
        ..Default::default()
    };
    canvas.draw_text(&bar, Point::new(x, y), &style);
}

fn draw_pod_summary(canvas: &mut DirectTerminalCanvas<'_>, x: f32, y: f32) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    canvas.draw_text("Pod Status:", Point::new(x, y), &label_style);

    let items = [
        ("Running", 156, Color::new(0.3, 1.0, 0.5, 1.0)),
        ("Pending", 3, Color::new(0.9, 0.9, 0.3, 1.0)),
        ("Failed", 2, Color::new(1.0, 0.3, 0.3, 1.0)),
        ("Succeeded", 45, Color::new(0.5, 0.5, 0.5, 1.0)),
    ];

    let mut offset = 13.0;
    for (name, count, color) in items {
        let style = TextStyle {
            color,
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{name}:{count}"),
            Point::new(x + offset, y),
            &style,
        );
        offset += name.len() as f32 + 5.0;
    }

    // Deployments/Services
    canvas.draw_text(
        "Deployments: 28 | Services: 42 | Ingresses: 8 | ConfigMaps: 65",
        Point::new(x, y + 1.0),
        &label_style,
    );
}

fn draw_namespace_breakdown(canvas: &mut DirectTerminalCanvas<'_>, x: f32, y: f32) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    canvas.draw_text("Top Namespaces:", Point::new(x, y), &label_style);

    let namespaces = [
        ("production", 45, Color::new(0.3, 0.7, 1.0, 1.0)),
        ("staging", 28, Color::new(0.9, 0.6, 0.3, 1.0)),
        ("monitoring", 18, Color::new(0.6, 0.3, 0.9, 1.0)),
        ("default", 12, Color::new(0.5, 0.5, 0.5, 1.0)),
    ];

    for (i, (ns, pods, color)) in namespaces.iter().enumerate() {
        let row_y = y + 1.0 + (i / 2) as f32;
        let col_x = ((i % 2) as f32).mul_add(18.0, x);

        let style = TextStyle {
            color: *color,
            ..Default::default()
        };
        canvas.draw_text(&format!("{ns}: {pods}"), Point::new(col_x, row_y), &style);
    }
}

fn draw_footer(canvas: &mut DirectTerminalCanvas<'_>) {
    let key_style = TextStyle {
        color: Color::new(0.4, 0.4, 0.4, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "Context: prod-cluster | Version: 1.28.4 | Nodes: 5 | Total Pods: 206",
        Point::new(2.0, 20.0),
        &key_style,
    );
    canvas.draw_text(
        "[q] quit  [n] nodes  [p] pods  [d] deployments  [l] logs  [h] help",
        Point::new(2.0, 21.0),
        &key_style,
    );
}

fn simulate_cluster_cpu(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 15.0f64.mul_add((i as f64 / 12.0).sin(), 65.0);
            let noise = ((i * 7919) % 20) as f64;
            (base + noise).clamp(20.0, 95.0)
        })
        .collect()
}

fn simulate_cluster_mem(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 10.0f64.mul_add((i as f64 / 15.0).cos(), 72.0);
            let noise = ((i * 6971) % 15) as f64;
            (base + noise).clamp(40.0, 95.0)
        })
        .collect()
}
