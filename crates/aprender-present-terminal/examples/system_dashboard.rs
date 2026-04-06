#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! System Dashboard Example - cbtop/ttop Clone
//!
//! Pixel-perfect recreation of the cbtop/ttop terminal interface.
//!
//! Run with: cargo run -p presentar-terminal --example `system_dashboard`

use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::{
    Border, BorderStyle, BrailleGraph, ColorMode, CpuGrid, GraphMode, MemoryBar, NetworkInterface,
    NetworkPanel, ProcessEntry, ProcessTable,
};

// Layout constants from spec
const WIDTH: f32 = 82.0;
const HEIGHT: f32 = 24.0;

fn main() {
    // Simulate system metrics
    let cpu_history = simulate_cpu(60);
    let mem_history = simulate_memory(60);
    let net_rx = simulate_network_rx(60);
    let net_tx = simulate_network_tx(60);

    // Create buffer matching spec dimensions (roughly 82x24 for standard terminals)
    let mut buffer = CellBuffer::new(WIDTH as u16, HEIGHT as u16);
    let mut renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);

    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        // Main Window Border
        let mut main_border = Border::new()
            .with_title("cbtop - Compute Block System Monitor")
            .with_style(BorderStyle::Rounded)
            .with_color(Color::new(0.4, 0.4, 0.4, 1.0));

        // We draw the uptime manually in the title bar area since Border doesn't support right-aligned text yet
        main_border.layout(Rect::new(0.0, 0.0, WIDTH, HEIGHT));
        main_border.paint(&mut canvas);

        // Uptime (manual placement on top border)
        let time_style = TextStyle {
            color: Color::new(0.8, 0.8, 0.8, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            "uptime: 5d 12:34:56",
            Point::new(WIDTH - 20.0, 0.0),
            &time_style,
        );

        // === Row 1: CPU & Memory ===
        let row1_y = 1.0;
        let row1_h = 8.0;
        let col1_w = 40.0;
        let col2_w = WIDTH - col1_w - 2.0; // Accounting for main border padding

        // CPU Panel
        let cpu_rect = Rect::new(1.0, row1_y, col1_w, row1_h);
        draw_cpu_panel(&mut canvas, &cpu_history, cpu_rect);

        // Memory Panel
        let mem_rect = Rect::new(col1_w + 1.0, row1_y, col2_w, row1_h);
        draw_memory_panel(&mut canvas, &mem_history, mem_rect);

        // === Row 2: Network & Disk ===
        let row2_y = row1_y + row1_h;
        let row2_h = 7.0;

        // Network Panel
        let net_rect = Rect::new(1.0, row2_y, col1_w, row2_h);
        draw_network_panel(&mut canvas, &net_rx, &net_tx, net_rect);

        // Disk Panel
        let disk_rect = Rect::new(col1_w + 1.0, row2_y, col2_w, row2_h);
        draw_disk_panel(&mut canvas, disk_rect);

        // === Row 3: Processes ===
        let row3_y = row2_y + row2_h;
        let row3_h = HEIGHT - row3_y - 2.0; // Reserve space for footer
        let proc_rect = Rect::new(1.0, row3_y, WIDTH - 2.0, row3_h);
        draw_process_panel(&mut canvas, proc_rect);

        // Footer
        draw_footer(&mut canvas, Rect::new(1.0, HEIGHT - 2.0, WIDTH - 2.0, 1.0));
    }

    // Render to stdout
    let mut output = Vec::with_capacity(16384);
    renderer.flush(&mut buffer, &mut output).unwrap();
    std::io::Write::write_all(&mut std::io::stdout(), &output).unwrap();
    println!(); // Ensure newline at end
}

fn draw_cpu_panel(canvas: &mut DirectTerminalCanvas<'_>, history: &[f64], bounds: Rect) {
    // Border
    let mut border = Border::new()
        .with_title("CPU")
        .with_style(BorderStyle::Rounded)
        .with_color(Color::new(0.3, 0.6, 1.0, 1.0)); // Blue-ish
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    // Usage Text
    let current = history.last().copied().unwrap_or(0.0);
    let color = cpu_usage_color(current);
    let pct_style = TextStyle {
        color,
        ..Default::default()
    };
    canvas.draw_text(
        &format!("{current:.1}%"),
        Point::new(bounds.x + 6.0, bounds.y), // Right of "CPU" title
        &pct_style,
    );

    // Braille Graph
    let mut graph = BrailleGraph::new(history.to_vec())
        .with_color(color)
        .with_range(0.0, 100.0)
        .with_mode(GraphMode::Braille);
    graph.layout(Rect::new(inner.x, inner.y, inner.width, 3.0));
    graph.paint(canvas);

    // CPU Grid
    let core_usage: Vec<f64> = (0..16)
        .map(|i| {
            let base = 35.0f64.mul_add(
                f64::from(i).mul_add(0.7, history.len() as f64 / 10.0).sin(),
                45.0,
            );
            let noise = f64::from((i * 7919) % 40);
            (base + noise).clamp(5.0, 98.0)
        })
        .collect();
    let mut grid = CpuGrid::new(core_usage).with_columns(8).compact();
    grid.layout(Rect::new(
        inner.x,
        inner.y + 3.0,
        inner.width,
        inner.height - 3.0,
    ));
    grid.paint(canvas);
}

fn draw_memory_panel(canvas: &mut DirectTerminalCanvas<'_>, history: &[f64], bounds: Rect) {
    let mut border = Border::new()
        .with_title("Memory")
        .with_style(BorderStyle::Rounded)
        .with_color(Color::new(0.8, 0.3, 0.8, 1.0)); // Purple-ish
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    let total_gb = 128.0;
    let used_gb = history.last().copied().unwrap_or(0.0);

    // Usage Text
    let info_style = TextStyle {
        color: Color::new(0.8, 0.8, 0.8, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        &format!("{used_gb:.1}/{total_gb:.0} GB"),
        Point::new(bounds.x + 9.0, bounds.y),
        &info_style,
    );

    // Graph
    let pct_history: Vec<f64> = history.iter().map(|&v| (v / total_gb) * 100.0).collect();
    let mut graph = BrailleGraph::new(pct_history)
        .with_color(Color::new(0.8, 0.3, 0.8, 1.0))
        .with_range(0.0, 100.0)
        .with_mode(GraphMode::Braille);
    graph.layout(Rect::new(inner.x, inner.y, inner.width, 3.0));
    graph.paint(canvas);

    // Memory Bar
    let total_bytes = (total_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let mut memory_bar = MemoryBar::from_usage(
        (50.0 * 1024.0 * 1024.0 * 1024.0) as u64,
        (30.0 * 1024.0 * 1024.0 * 1024.0) as u64,
        (2.0 * 1024.0 * 1024.0 * 1024.0) as u64,
        (16.0 * 1024.0 * 1024.0 * 1024.0) as u64,
        total_bytes,
    )
    .with_bar_width(40);
    memory_bar.layout(Rect::new(
        inner.x,
        inner.y + 3.0,
        inner.width,
        inner.height - 3.0,
    ));
    memory_bar.paint(canvas);
}

fn draw_network_panel(canvas: &mut DirectTerminalCanvas<'_>, rx: &[f64], tx: &[f64], bounds: Rect) {
    let mut border = Border::new()
        .with_title("Network")
        .with_style(BorderStyle::Rounded)
        .with_color(Color::new(0.3, 0.8, 0.5, 1.0)); // Green-ish
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    let mut eth0 = NetworkInterface::new("eth0");
    for (&r, &t) in rx.iter().zip(tx.iter()) {
        eth0.update(r * 1024.0 * 1024.0, t * 1024.0 * 1024.0);
    }
    eth0.set_totals(50 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024);

    let mut wlan0 = NetworkInterface::new("wlan0");
    // Simulate low traffic
    wlan0.update(3.6 * 1024.0 * 1024.0, 0.5 * 1024.0 * 1024.0);

    let mut panel = NetworkPanel::new().with_spark_width(10).compact();
    panel.set_interfaces(vec![eth0, wlan0]);
    panel.layout(inner);
    panel.paint(canvas);
}

fn draw_disk_panel(canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    let mut border = Border::new()
        .with_title("Disk")
        .with_style(BorderStyle::Rounded)
        .with_color(Color::new(0.9, 0.7, 0.3, 1.0)); // Orange-ish
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    let disks = [("/", 70.5), ("/home", 62.7), ("/data", 72.5)];

    let style = TextStyle {
        color: Color::new(0.8, 0.8, 0.8, 1.0),
        ..Default::default()
    };
    for (i, (mount, pct)) in disks.iter().enumerate() {
        let y = inner.y + i as f32;
        // Label
        canvas.draw_text(&format!("{mount:<6}"), Point::new(inner.x, y), &style);

        // Bar
        let bar_width = 15;
        let filled = ((*pct / 100.0) * bar_width as f64).round() as usize;
        let mut bar = String::new();
        for j in 0..bar_width {
            bar.push(if j < filled { '█' } else { '░' });
        }
        let bar_color = if *pct > 70.0 {
            Color::new(0.9, 0.7, 0.3, 1.0)
        } else {
            Color::GREEN
        };
        canvas.draw_text(
            &bar,
            Point::new(inner.x + 7.0, y),
            &TextStyle {
                color: bar_color,
                ..Default::default()
            },
        );

        // Pct
        canvas.draw_text(&format!("{pct}%"), Point::new(inner.x + 23.0, y), &style);
    }
}

fn draw_process_panel(canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    let mut border = Border::new()
        .with_title("Processes")
        .with_style(BorderStyle::Rounded)
        .with_color(Color::new(0.5, 0.7, 0.9, 1.0)); // Cyan-ish
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    let processes = vec![
        ProcessEntry::new(1234, "noah", 25.3, 5.5, "firefox"),
        ProcessEntry::new(5678, "noah", 18.7, 12.3, "rustc"),
        ProcessEntry::new(9012, "noah", 15.2, 8.1, "code"),
        ProcessEntry::new(3456, "root", 12.8, 3.2, "dockerd"),
    ];

    let mut table = ProcessTable::new();
    table.set_processes(processes);
    table.layout(inner);
    table.paint(canvas);
}

fn draw_footer(canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    let key_style = TextStyle {
        color: Color::new(0.3, 0.7, 0.9, 1.0),
        ..Default::default()
    };
    let desc_style = TextStyle {
        color: Color::new(0.5, 0.5, 0.5, 1.0),
        ..Default::default()
    };

    let keys = [
        ("[q]", "quit"),
        ("[h]", "help"),
        ("[c]", "sort:cpu"),
        ("[m]", "sort:mem"),
        ("[p]", "sort:pid"),
        ("[k]", "kill"),
    ];

    let mut x = bounds.x + 1.0;
    for (k, d) in keys {
        canvas.draw_text(k, Point::new(x, bounds.y), &key_style);
        x += k.len() as f32 + 1.0;
        canvas.draw_text(d, Point::new(x, bounds.y), &desc_style);
        x += d.len() as f32 + 2.0;
    }
}

fn cpu_usage_color(usage: f64) -> Color {
    if usage > 90.0 {
        Color::new(1.0, 0.3, 0.3, 1.0)
    } else if usage > 50.0 {
        Color::new(1.0, 1.0, 0.3, 1.0)
    } else {
        Color::new(0.3, 0.8, 1.0, 1.0)
    }
}

// Simple simulation helpers
fn simulate_cpu(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 35.0f64.mul_add((i as f64 / 10.0).sin(), 45.0))
        .collect()
}
fn simulate_memory(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 10.0f64.mul_add((i as f64 / 20.0).sin(), 65.0))
        .collect()
}
fn simulate_network_rx(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 30.0f64.mul_add((i as f64 / 8.0).sin(), 40.0))
        .collect()
}
fn simulate_network_tx(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 10.0f64.mul_add((i as f64 / 6.0).cos(), 15.0))
        .collect()
}
