#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! GPU Compute Monitor Example
//!
//! Demonstrates GPU utilization monitoring for CUDA/ML workloads
//! with memory, compute, and temperature visualization.
//!
//! Run with: cargo run -p presentar-terminal --example `gpu_compute`

use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::{BrailleGraph, ColorMode, GraphMode};

fn main() {
    println!("=== GPU Compute Monitor ===\n");

    // Simulate GPU metrics
    let gpus = vec![
        GpuInfo::new(0, "NVIDIA RTX 4090", 89.5, 18.2, 24.0, 72, 350, 450, 2520),
        GpuInfo::new(1, "NVIDIA RTX 4090", 92.3, 20.1, 24.0, 78, 380, 450, 2505),
        GpuInfo::new(2, "NVIDIA RTX 4090", 45.2, 12.8, 24.0, 58, 220, 450, 2490),
        GpuInfo::new(3, "NVIDIA RTX 4090", 78.6, 16.5, 24.0, 68, 310, 450, 2510),
    ];

    let compute_history = simulate_compute_usage(60);
    let memory_history = simulate_memory_usage(60);
    let power_history = simulate_power(60);

    // Create buffer
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);

    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        // Background
        canvas.fill_rect(
            Rect::new(0.0, 0.0, 80.0, 24.0),
            Color::new(0.02, 0.02, 0.05, 1.0),
        );

        // Title
        let title_style = TextStyle {
            color: Color::new(0.4, 0.9, 0.4, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            "GPU Compute Cluster Monitor",
            Point::new(2.0, 1.0),
            &title_style,
        );

        // Compute utilization graph
        draw_utilization_graph(
            &mut canvas,
            &compute_history,
            Rect::new(2.0, 3.0, 50.0, 5.0),
        );

        // Memory usage graph
        draw_memory_graph(
            &mut canvas,
            &memory_history,
            Rect::new(54.0, 3.0, 24.0, 5.0),
        );

        // GPU cards info
        draw_gpu_cards(&mut canvas, &gpus, 2.0, 9.0);

        // Power consumption
        draw_power_graph(&mut canvas, &power_history, Rect::new(2.0, 17.0, 50.0, 4.0));

        // Summary stats
        draw_cluster_summary(&mut canvas, &gpus, 54.0, 17.0);

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

struct GpuInfo {
    id: u32,
    name: String,
    compute_pct: f64,
    mem_used: f64,
    mem_total: f64,
    temp: u32,
    power: u32,
    power_limit: u32,
    clock_mhz: u32,
}

impl GpuInfo {
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: u32,
        name: &str,
        compute: f64,
        mem_used: f64,
        mem_total: f64,
        temp: u32,
        power: u32,
        power_limit: u32,
        clock: u32,
    ) -> Self {
        Self {
            id,
            name: name.to_string(),
            compute_pct: compute,
            mem_used,
            mem_total,
            temp,
            power,
            power_limit,
            clock_mhz: clock,
        }
    }

    fn compute_color(&self) -> Color {
        if self.compute_pct > 90.0 {
            Color::new(0.3, 1.0, 0.5, 1.0) // Green - fully utilized
        } else if self.compute_pct > 70.0 {
            Color::new(0.9, 0.9, 0.3, 1.0) // Yellow - good
        } else if self.compute_pct > 30.0 {
            Color::new(0.9, 0.6, 0.3, 1.0) // Orange - underutilized
        } else {
            Color::new(0.5, 0.5, 0.5, 1.0) // Gray - idle
        }
    }

    fn temp_color(&self) -> Color {
        if self.temp > 80 {
            Color::new(1.0, 0.3, 0.3, 1.0) // Red - critical
        } else if self.temp > 70 {
            Color::new(1.0, 0.7, 0.2, 1.0) // Orange - warm
        } else {
            Color::new(0.3, 0.8, 1.0, 1.0) // Blue - cool
        }
    }
}

fn draw_utilization_graph(canvas: &mut DirectTerminalCanvas<'_>, history: &[f64], bounds: Rect) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "Cluster Compute Utilization",
        Point::new(bounds.x, bounds.y),
        &label_style,
    );

    let current = history.last().copied().unwrap_or(0.0);
    let color = if current > 80.0 {
        Color::new(0.3, 1.0, 0.5, 1.0)
    } else if current > 50.0 {
        Color::new(0.9, 0.9, 0.3, 1.0)
    } else {
        Color::new(0.9, 0.6, 0.3, 1.0)
    };

    let value_style = TextStyle {
        color,
        ..Default::default()
    };
    canvas.draw_text(
        &format!("{current:5.1}%"),
        Point::new(bounds.x + 30.0, bounds.y),
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

#[allow(clippy::no_effect_underscore_binding)]
fn draw_memory_graph(canvas: &mut DirectTerminalCanvas<'_>, history: &[f64], bounds: Rect) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    canvas.draw_text("VRAM", Point::new(bounds.x, bounds.y), &label_style);

    let current = history.last().copied().unwrap_or(0.0);
    let total = 96.0; // 4x 24GB
    let _pct = (current / total) * 100.0;

    let color = Color::new(0.6, 0.3, 1.0, 1.0);
    let value_style = TextStyle {
        color,
        ..Default::default()
    };
    canvas.draw_text(
        &format!("{current:.0}/{total:.0}GB"),
        Point::new(bounds.x + 6.0, bounds.y),
        &value_style,
    );

    let pct_history: Vec<f64> = history.iter().map(|&v| (v / total) * 100.0).collect();
    let mut graph = BrailleGraph::new(pct_history)
        .with_color(color)
        .with_range(0.0, 100.0)
        .with_mode(GraphMode::Block);

    graph.layout(Rect::new(
        bounds.x,
        bounds.y + 1.0,
        bounds.width,
        bounds.height - 1.0,
    ));
    graph.paint(canvas);
}

fn draw_gpu_cards(canvas: &mut DirectTerminalCanvas<'_>, gpus: &[GpuInfo], x: f32, y: f32) {
    let header_style = TextStyle {
        color: Color::new(0.5, 0.5, 0.5, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "GPU  Model              Compute    Memory        Temp   Power    Clock",
        Point::new(x, y),
        &header_style,
    );
    canvas.draw_text(&"─".repeat(76), Point::new(x, y + 1.0), &header_style);

    for (i, gpu) in gpus.iter().enumerate() {
        let row_y = y + 2.0 + i as f32;

        // GPU ID
        let id_style = TextStyle {
            color: Color::new(0.9, 0.9, 0.9, 1.0),
            ..Default::default()
        };
        canvas.draw_text(&format!("[{}]", gpu.id), Point::new(x, row_y), &id_style);

        // Model name (truncated)
        let name_style = TextStyle {
            color: Color::new(0.7, 0.7, 0.7, 1.0),
            ..Default::default()
        };
        let short_name: String = gpu.name.chars().take(16).collect();
        canvas.draw_text(
            &format!("{short_name:<16}"),
            Point::new(x + 5.0, row_y),
            &name_style,
        );

        // Compute usage with bar
        let compute_style = TextStyle {
            color: gpu.compute_color(),
            ..Default::default()
        };
        let bar_width = 8;
        let filled = ((gpu.compute_pct / 100.0) * bar_width as f64).round() as usize;
        let mut bar = String::with_capacity(bar_width);
        for j in 0..bar_width {
            bar.push(if j < filled { '█' } else { '░' });
        }
        canvas.draw_text(&bar, Point::new(x + 22.0, row_y), &compute_style);
        canvas.draw_text(
            &format!("{:>3.0}%", gpu.compute_pct),
            Point::new(x + 31.0, row_y),
            &compute_style,
        );

        // Memory
        #[allow(clippy::no_effect_underscore_binding)]
        let _mem_pct = (gpu.mem_used / gpu.mem_total) * 100.0;
        let mem_style = TextStyle {
            color: Color::new(0.6, 0.3, 1.0, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:.0}/{:.0}GB", gpu.mem_used, gpu.mem_total),
            Point::new(x + 37.0, row_y),
            &mem_style,
        );

        // Temperature
        let temp_style = TextStyle {
            color: gpu.temp_color(),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:>3}°C", gpu.temp),
            Point::new(x + 50.0, row_y),
            &temp_style,
        );

        // Power
        let power_style = TextStyle {
            color: Color::new(0.9, 0.7, 0.3, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:>3}/{}W", gpu.power, gpu.power_limit),
            Point::new(x + 57.0, row_y),
            &power_style,
        );

        // Clock
        let clock_style = TextStyle {
            color: Color::new(0.7, 0.7, 0.7, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{}MHz", gpu.clock_mhz),
            Point::new(x + 68.0, row_y),
            &clock_style,
        );
    }
}

fn draw_power_graph(canvas: &mut DirectTerminalCanvas<'_>, history: &[f64], bounds: Rect) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "Total Power Draw",
        Point::new(bounds.x, bounds.y),
        &label_style,
    );

    let current = history.last().copied().unwrap_or(0.0);
    let max_power = 1800.0; // 4x 450W

    let color = Color::new(0.9, 0.7, 0.3, 1.0);
    let value_style = TextStyle {
        color,
        ..Default::default()
    };
    canvas.draw_text(
        &format!("{current:.0}W / {max_power:.0}W"),
        Point::new(bounds.x + 20.0, bounds.y),
        &value_style,
    );

    let mut graph = BrailleGraph::new(history.to_vec())
        .with_color(color)
        .with_range(0.0, max_power)
        .with_mode(GraphMode::Braille);

    graph.layout(Rect::new(
        bounds.x,
        bounds.y + 1.0,
        bounds.width,
        bounds.height - 1.0,
    ));
    graph.paint(canvas);
}

fn draw_cluster_summary(canvas: &mut DirectTerminalCanvas<'_>, gpus: &[GpuInfo], x: f32, y: f32) {
    let label_style = TextStyle {
        color: Color::new(0.5, 0.5, 0.5, 1.0),
        ..Default::default()
    };
    let value_style = TextStyle {
        color: Color::new(0.8, 0.8, 0.8, 1.0),
        ..Default::default()
    };

    let avg_compute: f64 = gpus.iter().map(|g| g.compute_pct).sum::<f64>() / gpus.len() as f64;
    let total_mem: f64 = gpus.iter().map(|g| g.mem_used).sum();
    let max_temp: u32 = gpus.iter().map(|g| g.temp).max().unwrap_or(0);
    let total_power: u32 = gpus.iter().map(|g| g.power).sum();

    canvas.draw_text("Cluster Stats:", Point::new(x, y), &label_style);
    canvas.draw_text(
        &format!("Avg Util: {avg_compute:.0}%"),
        Point::new(x, y + 1.0),
        &value_style,
    );
    canvas.draw_text(
        &format!("VRAM: {total_mem:.0} GB"),
        Point::new(x, y + 2.0),
        &value_style,
    );
    canvas.draw_text(
        &format!("Max Temp: {max_temp}°C"),
        Point::new(x, y + 3.0),
        &value_style,
    );
    canvas.draw_text(
        &format!("Power: {total_power} W"),
        Point::new(x, y + 4.0),
        &value_style,
    );
}

fn draw_footer(canvas: &mut DirectTerminalCanvas<'_>) {
    let key_style = TextStyle {
        color: Color::new(0.4, 0.4, 0.4, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "[q] quit  [r] refresh  [p] processes  [f] fans  [h] help",
        Point::new(2.0, 22.0),
        &key_style,
    );
}

fn simulate_compute_usage(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 15.0f64.mul_add((i as f64 / 10.0).sin(), 75.0);
            let noise = ((i * 7919) % 20) as f64;
            (base + noise).clamp(20.0, 98.0)
        })
        .collect()
}

fn simulate_memory_usage(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 10.0f64.mul_add((i as f64 / 15.0).sin(), 65.0);
            let noise = ((i * 6971) % 10) as f64;
            base + noise
        })
        .collect()
}

fn simulate_power(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let base = 200.0f64.mul_add((i as f64 / 12.0).sin(), 1200.0);
            let noise = ((i * 1103) % 100) as f64;
            (base + noise).clamp(400.0, 1700.0)
        })
        .collect()
}
