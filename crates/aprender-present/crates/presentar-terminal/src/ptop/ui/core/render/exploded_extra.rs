use super::*;

/// Get color for I/O rate display (read or write)
fn io_rate_color(rate: f64, is_read: bool) -> Color {
    if rate > 10_000_000.0 {
        if is_read {
            Color::new(0.3, 0.9, 0.5, 1.0)
        } else {
            Color::new(0.9, 0.6, 0.3, 1.0)
        }
    } else {
        Color::new(0.7, 0.7, 0.7, 1.0)
    }
}

/// Draw disk I/O section for exploded view
fn draw_disk_io_section(
    canvas: &mut dyn Canvas,
    io: &crate::ptop::analyzers::DiskIoData,
    inner: Rect,
    mut y: f32,
    io_section_height: usize,
    header_bg: Color,
    border_color: Color,
) {
    use crate::widgets::display_rules::{format_column, ColumnAlign, TruncateStrategy};
    use crate::widgets::selection::DIMMED_BG;

    canvas.draw_text(
        "I/O RATES BY DEVICE",
        Point::new(inner.x, y),
        &TextStyle {
            color: border_color,
            ..Default::default()
        },
    );
    y += 1.0;

    let io_col_dev = 12;
    let io_col_read = 12;
    let io_col_write = 12;
    let io_col_iops = 10;
    canvas.fill_rect(Rect::new(inner.x, y, inner.width, 1.0), header_bg);
    let io_headers = ["DEVICE", "READ/s", "WRITE/s", "IOPS"];
    let io_widths = [io_col_dev, io_col_read, io_col_write, io_col_iops];
    let mut ihx = inner.x;
    for (h, w) in io_headers.iter().zip(io_widths.iter()) {
        canvas.draw_text(
            &format_column(h, *w, ColumnAlign::Left, TruncateStrategy::End),
            Point::new(ihx, y),
            &TextStyle {
                color: border_color,
                ..Default::default()
            },
        );
        ihx += *w as f32 + 1.0;
    }
    y += 1.0;

    let mut devices: Vec<_> = io.physical_disks().collect();
    devices.sort_by(|a, b| a.0.cmp(b.0));

    for (dev_name, _stats) in devices.iter().take(io_section_height.saturating_sub(2)) {
        let rates = io.rates.get(*dev_name);
        let read_rate = rates.map_or(0.0, |r| r.read_bytes_per_sec);
        let write_rate = rates.map_or(0.0, |r| r.write_bytes_per_sec);
        let iops = rates.map_or(0.0, |r| r.reads_per_sec + r.writes_per_sec);

        canvas.fill_rect(Rect::new(inner.x, y, inner.width, 1.0), DIMMED_BG);
        let mut col_x = inner.x;

        canvas.draw_text(
            &format_column(
                dev_name,
                io_col_dev,
                ColumnAlign::Left,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &TextStyle {
                color: Color::new(0.85, 0.85, 0.85, 1.0),
                ..Default::default()
            },
        );
        col_x += io_col_dev as f32 + 1.0;

        canvas.draw_text(
            &format_column(
                &format_bytes_rate(read_rate),
                io_col_read,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &TextStyle {
                color: io_rate_color(read_rate, true),
                ..Default::default()
            },
        );
        col_x += io_col_read as f32 + 1.0;

        canvas.draw_text(
            &format_column(
                &format_bytes_rate(write_rate),
                io_col_write,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &TextStyle {
                color: io_rate_color(write_rate, false),
                ..Default::default()
            },
        );
        col_x += io_col_write as f32 + 1.0;

        canvas.draw_text(
            &format_column(
                &format!("{:.0}", iops),
                io_col_iops,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &TextStyle {
                color: Color::new(0.7, 0.7, 0.7, 1.0),
                ..Default::default()
            },
        );

        y += 1.0;
    }
}

/// FULL SCREEN disk exploded view
pub(super) fn draw_disk_exploded(app: &App, canvas: &mut DirectTerminalCanvas, area: Rect) {
    use crate::widgets::display_rules::{
        format_bytes_si, format_column, format_percent, ColumnAlign, TruncateStrategy,
    };
    use crate::widgets::selection::RowHighlight;
    use crate::HeatScheme;

    let border_color = DISK_COLOR;
    let disk_io = app.disk_io_data();
    let (total_read, total_write) = disk_io.map_or((0.0, 0.0), |io| {
        (io.total_read_bytes_per_sec, io.total_write_bytes_per_sec)
    });

    let title = format!(
        "▼ DISK │ R: {} │ W: {} │ {} Volumes",
        format_bytes_rate(total_read),
        format_bytes_rate(total_write),
        app.disks.len()
    );

    let mut border = create_panel_border(&title, border_color, true);
    border.layout(area);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }

    let disk_count = app.disks.len();
    let disk_section_height = (disk_count.min(inner.height as usize / 2)).max(4);
    let io_section_height = (inner.height as usize).saturating_sub(disk_section_height + 2);

    let dim_style = TextStyle {
        color: Color::new(0.5, 0.5, 0.5, 1.0),
        ..Default::default()
    };
    let header_bg = Color::new(0.12, 0.15, 0.22, 1.0);

    let mut y = inner.y;
    let col_mount = 25.min(inner.width as usize / 4);
    let col_fs = 10;
    let col_used = 10;
    let col_total = 10;
    let col_pct = 8;
    let col_bar = (inner.width as usize)
        .saturating_sub(col_mount + col_fs + col_used + col_total + col_pct + 8);

    canvas.fill_rect(Rect::new(inner.x, y, inner.width, 1.0), header_bg);
    let headers = ["MOUNT", "FS", "USED", "TOTAL", "USE%", ""];
    let widths = [col_mount, col_fs, col_used, col_total, col_pct, col_bar];
    let mut hx = inner.x;
    for (header, width) in headers.iter().zip(widths.iter()) {
        canvas.draw_text(
            &format_column(header, *width, ColumnAlign::Left, TruncateStrategy::End),
            Point::new(hx, y),
            &TextStyle {
                color: border_color,
                ..Default::default()
            },
        );
        hx += *width as f32 + 1.0;
    }
    y += 1.0;

    for (i, disk) in app.disks.iter().enumerate() {
        if (y - inner.y) as usize >= disk_section_height {
            break;
        }

        let row_hl = RowHighlight::new(Rect::new(inner.x, y, inner.width, 1.0), i == 0);
        row_hl.paint(canvas);
        let text_style = row_hl.text_style();

        let mount = disk.mount_point().to_string_lossy().to_string();
        let fs_type = disk.file_system().to_string_lossy().to_string();
        let total = disk.total_space();
        let used = total.saturating_sub(disk.available_space());
        let use_pct = if total > 0 {
            (used as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let mut col_x = inner.x;
        canvas.draw_text(
            &format_column(
                &mount,
                col_mount,
                ColumnAlign::Left,
                TruncateStrategy::Command,
            ),
            Point::new(col_x, y),
            &text_style,
        );
        col_x += col_mount as f32 + 1.0;
        canvas.draw_text(
            &format_column(&fs_type, col_fs, ColumnAlign::Left, TruncateStrategy::End),
            Point::new(col_x, y),
            &text_style,
        );
        col_x += col_fs as f32 + 1.0;
        canvas.draw_text(
            &format_column(
                &format_bytes_si(used),
                col_used,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &text_style,
        );
        col_x += col_used as f32 + 1.0;
        canvas.draw_text(
            &format_column(
                &format_bytes_si(total),
                col_total,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &text_style,
        );
        col_x += col_total as f32 + 1.0;

        let pct_color = HeatScheme::Warm.color_for_percent(use_pct);
        canvas.draw_text(
            &format_column(
                &format_percent(use_pct as f32),
                col_pct,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &TextStyle {
                color: if i == 0 { Color::WHITE } else { pct_color },
                ..Default::default()
            },
        );
        col_x += col_pct as f32 + 1.0;

        if col_bar >= 3 {
            let bar_str = make_bar(use_pct / 100.0, col_bar);
            canvas.draw_text(
                &bar_str,
                Point::new(col_x, y),
                &TextStyle {
                    color: HeatScheme::Warm.color_for_percent(use_pct),
                    ..Default::default()
                },
            );
        }
        y += 1.0;
    }

    y += 0.5;
    canvas.draw_text(
        &"─".repeat(inner.width as usize),
        Point::new(inner.x, y),
        &dim_style,
    );
    y += 1.0;

    if let Some(io) = disk_io {
        draw_disk_io_section(
            canvas,
            io,
            inner,
            y,
            io_section_height,
            header_bg,
            border_color,
        );
    }
}

/// Get color for network rate display (high rate = highlighted color).
#[inline]
fn network_rate_color(rate: u64, is_rx: bool, is_selected: bool) -> Color {
    if rate > 1_000_000 {
        if is_rx {
            Color::new(0.3, 0.9, 0.5, 1.0) // Green for RX
        } else {
            Color::new(0.9, 0.6, 0.3, 1.0) // Orange for TX
        }
    } else if is_selected {
        Color::WHITE
    } else {
        Color::new(0.7, 0.7, 0.7, 1.0)
    }
}

/// FULL SCREEN network exploded view
/// SPEC-024 Section 30: Exploded views fill the screen
pub(super) fn draw_network_exploded(app: &App, canvas: &mut DirectTerminalCanvas, area: Rect) {
    use crate::widgets::display_rules::{
        format_bytes_si, format_column, ColumnAlign, TruncateStrategy,
    };
    use crate::widgets::selection::RowHighlight;

    let border_color = NETWORK_COLOR;

    // Calculate totals
    let (rx_total, tx_total): (u64, u64) = if app.deterministic {
        (0, 0)
    } else {
        app.networks
            .values()
            .map(|d| (d.received(), d.transmitted()))
            .fold((0, 0), |(ar, at), (r, t)| (ar + r, at + t))
    };

    let title = format!(
        "▼ NETWORK │ ↓ {}/s │ ↑ {}/s │ {} Interfaces",
        format_bytes(rx_total),
        format_bytes(tx_total),
        app.networks.len()
    );

    let mut border = create_panel_border(&title, border_color, true);
    border.layout(area);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }

    let mut y = inner.y;

    // Column widths
    let col_iface = 15;
    let col_rx = 12;
    let col_tx = 12;
    let col_rx_total = 14;
    let col_tx_total = 14;
    let col_bar = (inner.width as usize)
        .saturating_sub(col_iface + col_rx + col_tx + col_rx_total + col_tx_total + 10);

    // Header
    let header_bg = Color::new(0.12, 0.15, 0.22, 1.0);
    canvas.fill_rect(Rect::new(inner.x, y, inner.width, 1.0), header_bg);

    let headers = ["INTERFACE", "↓ RX/s", "↑ TX/s", "TOTAL RX", "TOTAL TX", ""];
    let widths = [
        col_iface,
        col_rx,
        col_tx,
        col_rx_total,
        col_tx_total,
        col_bar,
    ];
    let mut hx = inner.x;
    for (header, width) in headers.iter().zip(widths.iter()) {
        canvas.draw_text(
            &format_column(header, *width, ColumnAlign::Left, TruncateStrategy::End),
            Point::new(hx, y),
            &TextStyle {
                color: border_color,
                ..Default::default()
            },
        );
        hx += *width as f32 + 1.0;
    }
    y += 1.0;

    // Network interface rows
    let mut interfaces: Vec<_> = app.networks.iter().collect();
    interfaces.sort_by(|a, b| b.1.received().cmp(&a.1.received()));

    for (i, (iface_name, data)) in interfaces.iter().enumerate() {
        if (y - inner.y) as usize >= (inner.height as usize).saturating_sub(2) {
            break;
        }

        let row_rect = Rect::new(inner.x, y, inner.width, 1.0);
        let is_selected = i == 0;

        let row_hl = RowHighlight::new(row_rect, is_selected);
        row_hl.paint(canvas);
        let text_style = row_hl.text_style();

        let rx_rate = data.received();
        let tx_rate = data.transmitted();
        let total_rx = data.total_received();
        let total_tx = data.total_transmitted();

        let mut col_x = inner.x;

        // Interface name
        canvas.draw_text(
            &format_column(
                iface_name,
                col_iface,
                ColumnAlign::Left,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &text_style,
        );
        col_x += col_iface as f32 + 1.0;

        // RX rate
        canvas.draw_text(
            &format_column(
                &format_bytes_rate(rx_rate as f64),
                col_rx,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &TextStyle {
                color: network_rate_color(rx_rate, true, is_selected),
                ..Default::default()
            },
        );
        col_x += col_rx as f32 + 1.0;

        // TX rate
        canvas.draw_text(
            &format_column(
                &format_bytes_rate(tx_rate as f64),
                col_tx,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &TextStyle {
                color: network_rate_color(tx_rate, false, is_selected),
                ..Default::default()
            },
        );
        col_x += col_tx as f32 + 1.0;

        // Total RX
        canvas.draw_text(
            &format_column(
                &format_bytes_si(total_rx),
                col_rx_total,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &text_style,
        );
        col_x += col_rx_total as f32 + 1.0;

        // Total TX
        canvas.draw_text(
            &format_column(
                &format_bytes_si(total_tx),
                col_tx_total,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &text_style,
        );
        col_x += col_tx_total as f32 + 1.0;

        // Activity bar
        if col_bar >= 3 {
            let max_rate = 100_000_000.0; // 100MB/s scale
            let rx_pct = ((rx_rate as f64 / max_rate) * 100.0).min(100.0);
            let tx_pct = ((tx_rate as f64 / max_rate) * 100.0).min(100.0);
            let rx_chars = ((rx_pct / 100.0) * (col_bar / 2) as f64) as usize;
            let tx_chars = ((tx_pct / 100.0) * (col_bar / 2) as f64) as usize;

            let rx_bar: String = "▁".repeat(rx_chars);
            let tx_bar: String = "▁".repeat(tx_chars);

            canvas.draw_text(
                &rx_bar,
                Point::new(col_x, y),
                &TextStyle {
                    color: Color::new(0.3, 0.9, 0.5, 1.0),
                    ..Default::default()
                },
            );
            canvas.draw_text(
                &tx_bar,
                Point::new(col_x + (col_bar / 2) as f32, y),
                &TextStyle {
                    color: Color::new(0.9, 0.6, 0.3, 1.0),
                    ..Default::default()
                },
            );
        }

        y += 1.0;
    }
}

/// FULL SCREEN GPU exploded view
/// SPEC-024 Section 30: Exploded views fill the screen
pub(super) fn draw_gpu_exploded(app: &App, canvas: &mut DirectTerminalCanvas, area: Rect) {
    use crate::widgets::display_rules::format_bytes_si;
    use crate::HeatScheme;

    let border_color = GPU_COLOR;

    let gpu = app.gpu_info.clone();
    let title = gpu
        .as_ref()
        .map(|g| {
            let temp_str = g
                .temperature
                .map(|t| format!(" │ {}°C", t as i32))
                .unwrap_or_default();
            let power_str = g
                .power_watts
                .map(|p| format!(" │ {:.0}W", p))
                .unwrap_or_default();
            let util_str = g
                .utilization
                .map(|u| format!(" │ {}%", u as i32))
                .unwrap_or_default();
            format!("▼ GPU: {}{}{}{}", g.name, util_str, temp_str, power_str)
        })
        .unwrap_or_else(|| "▼ GPU │ No GPU detected".to_string());

    let mut border = create_panel_border(&title, border_color, true);
    border.layout(area);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }

    let mut y = inner.y;

    if let Some(g) = gpu {
        // GPU Utilization bar
        let util = g.utilization.unwrap_or(0) as f64;
        let util_bar_width = (inner.width as usize).min(60);
        let bar = make_bar(util / 100.0, util_bar_width);
        let util_color = HeatScheme::Warm.color_for_percent(util);

        canvas.draw_text(
            &format!("GPU Utilization: {} {:>5.1}%", bar, util),
            Point::new(inner.x, y),
            &TextStyle {
                color: util_color,
                ..Default::default()
            },
        );
        y += 1.0;

        // VRAM bar
        let vram_used = g.vram_used.unwrap_or(0);
        let vram_total = g.vram_total.unwrap_or(0);
        let vram_pct = safe_pct(vram_used, vram_total);
        let vram_bar = make_bar(vram_pct / 100.0, util_bar_width);
        let vram_color = HeatScheme::Warm.color_for_percent(vram_pct);

        canvas.draw_text(
            &format!(
                "VRAM:            {} {:>5.1}% ({} / {})",
                vram_bar,
                vram_pct,
                format_bytes_si(vram_used),
                format_bytes_si(vram_total)
            ),
            Point::new(inner.x, y),
            &TextStyle {
                color: vram_color,
                ..Default::default()
            },
        );
        y += 1.0;

        // Temperature and Power
        if let Some(temp) = g.temperature {
            let temp_color = HeatScheme::Thermal.color_for_percent((temp as f64 / 100.0) * 100.0);
            canvas.draw_text(
                &format!("Temperature: {}°C", temp as i32),
                Point::new(inner.x, y),
                &TextStyle {
                    color: temp_color,
                    ..Default::default()
                },
            );
        }
        if let Some(power) = g.power_watts {
            canvas.draw_text(
                &format!("    Power: {:.1}W", power),
                Point::new(inner.x + 25.0, y),
                &TextStyle {
                    color: Color::new(0.9, 0.7, 0.3, 1.0),
                    ..Default::default()
                },
            );
        }
        y += 2.0;

        // GPU processes header
        let header_bg = Color::new(0.12, 0.15, 0.22, 1.0);
        canvas.fill_rect(Rect::new(inner.x, y, inner.width, 1.0), header_bg);
        canvas.draw_text(
            "GPU PROCESSES",
            Point::new(inner.x, y),
            &TextStyle {
                color: border_color,
                ..Default::default()
            },
        );
        y += 1.0;

        // Note: GPU processes would come from app.analyzers.gpu_procs_data()
        // For now show placeholder
        canvas.draw_text(
            "  (GPU process list requires nvidia-smi)",
            Point::new(inner.x, y),
            &TextStyle {
                color: Color::new(0.5, 0.5, 0.5, 1.0),
                ..Default::default()
            },
        );
    } else {
        canvas.draw_text(
            "No GPU detected or nvidia-smi not available",
            Point::new(inner.x, y),
            &TextStyle {
                color: Color::new(0.5, 0.5, 0.5, 1.0),
                ..Default::default()
            },
        );
    }
}

/// Get color for sensor value based on type.
#[inline]
fn sensor_value_display_color(sensor_type: SensorType, value: f64, is_selected: bool) -> Color {
    use crate::HeatScheme;
    match sensor_type {
        SensorType::Temperature => HeatScheme::Thermal.color_for_percent(value),
        SensorType::Fan => Color::new(0.3, 0.7, 0.9, 1.0),
        _ => {
            if is_selected {
                Color::WHITE
            } else {
                Color::new(0.8, 0.8, 0.8, 1.0)
            }
        }
    }
}

/// Get color for sensor status.
#[inline]
fn sensor_status_display_color(status: SensorStatus) -> Color {
    match status {
        SensorStatus::Normal => Color::new(0.3, 0.9, 0.3, 1.0),
        SensorStatus::Warning => Color::new(0.9, 0.7, 0.2, 1.0),
        SensorStatus::Critical => Color::new(0.9, 0.2, 0.2, 1.0),
        SensorStatus::Low => Color::new(0.3, 0.5, 0.9, 1.0),
        SensorStatus::Fault => Color::new(0.5, 0.5, 0.5, 1.0),
    }
}

/// FULL SCREEN sensors exploded view
/// SPEC-024 Section 30: Exploded views fill the screen
pub(super) fn draw_sensors_exploded(app: &App, canvas: &mut DirectTerminalCanvas, area: Rect) {
    use crate::widgets::display_rules::{format_column, ColumnAlign, TruncateStrategy};
    use crate::widgets::selection::RowHighlight;

    let border_color = SENSORS_COLOR;

    // Get sensor data
    let sensor_data = app.snapshot_sensor_health.as_ref();
    let sensor_count = sensor_data.map(|d| d.sensors.len()).unwrap_or(0);

    let title = format!("▼ SENSORS │ {} readings", sensor_count);

    let mut border = create_panel_border(&title, border_color, true);
    border.layout(area);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }

    let mut y = inner.y;

    // Column widths
    let col_name = 25;
    let col_value = 15;
    let col_status = 10;
    let col_bar = (inner.width as usize).saturating_sub(col_name + col_value + col_status + 6);

    // Header
    let header_bg = Color::new(0.12, 0.15, 0.22, 1.0);
    canvas.fill_rect(Rect::new(inner.x, y, inner.width, 1.0), header_bg);

    let headers = ["SENSOR", "VALUE", "STATUS", ""];
    let widths = [col_name, col_value, col_status, col_bar];
    let mut hx = inner.x;
    for (header, width) in headers.iter().zip(widths.iter()) {
        canvas.draw_text(
            &format_column(header, *width, ColumnAlign::Left, TruncateStrategy::End),
            Point::new(hx, y),
            &TextStyle {
                color: border_color,
                ..Default::default()
            },
        );
        hx += *width as f32 + 1.0;
    }
    y += 1.0;

    if let Some(data) = sensor_data {
        for (i, reading) in data.sensors.iter().enumerate() {
            if (y - inner.y) as usize >= (inner.height as usize).saturating_sub(2) {
                break;
            }

            let row_rect = Rect::new(inner.x, y, inner.width, 1.0);
            let is_selected = i == 0;

            let row_hl = RowHighlight::new(row_rect, is_selected);
            row_hl.paint(canvas);
            let text_style = row_hl.text_style();

            let mut col_x = inner.x;

            // Name (label)
            canvas.draw_text(
                &format_column(
                    &reading.label,
                    col_name,
                    ColumnAlign::Left,
                    TruncateStrategy::End,
                ),
                Point::new(col_x, y),
                &text_style,
            );
            col_x += col_name as f32 + 1.0;

            // Value with thermal coloring (includes unit from value_display)
            let value_color =
                sensor_value_display_color(reading.sensor_type, reading.value, is_selected);
            canvas.draw_text(
                &format_column(
                    &reading.value_display(),
                    col_value,
                    ColumnAlign::Right,
                    TruncateStrategy::End,
                ),
                Point::new(col_x, y),
                &TextStyle {
                    color: value_color,
                    ..Default::default()
                },
            );
            col_x += col_value as f32 + 1.0;

            // Status
            let status_color = if is_selected {
                Color::WHITE
            } else {
                sensor_status_display_color(reading.status)
            };
            canvas.draw_text(
                &format_column(
                    reading.status.as_str(),
                    col_status,
                    ColumnAlign::Left,
                    TruncateStrategy::End,
                ),
                Point::new(col_x, y),
                &TextStyle {
                    color: status_color,
                    ..Default::default()
                },
            );

            y += 1.0;
        }
    } else {
        canvas.draw_text(
            "No sensor data available",
            Point::new(inner.x, y),
            &TextStyle {
                color: Color::new(0.5, 0.5, 0.5, 1.0),
                ..Default::default()
            },
        );
    }
}

/// FULL SCREEN process exploded view
/// SPEC-024 Section 30: Exploded views fill the screen
pub(super) fn draw_process_exploded(app: &App, canvas: &mut DirectTerminalCanvas, area: Rect) {
    use crate::widgets::display_rules::{
        format_bytes_si, format_column, ColumnAlign, TruncateStrategy,
    };
    use crate::widgets::selection::RowHighlight;
    use crate::HeatScheme;

    let border_color = PROCESS_COLOR;
    let proc_count = app.system.processes().len();

    let title = format!("▼ PROCESSES │ {} total", proc_count);

    let mut border = create_panel_border(&title, border_color, true);
    border.layout(area);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }

    let mut y = inner.y;

    // Column widths - full screen allows more columns
    let col_pid = 8;
    let col_user = 12;
    let col_cpu = 7;
    let col_mem = 7;
    let col_virt = 10;
    let col_rss = 10;
    let col_state = 6;
    let col_time = 10;
    let col_cmd = (inner.width as usize).saturating_sub(
        col_pid + col_user + col_cpu + col_mem + col_virt + col_rss + col_state + col_time + 12,
    );

    // Header
    let header_bg = Color::new(0.12, 0.15, 0.22, 1.0);
    canvas.fill_rect(Rect::new(inner.x, y, inner.width, 1.0), header_bg);

    let headers = [
        "PID", "USER", "CPU%", "MEM%", "VIRT", "RSS", "STATE", "TIME", "COMMAND",
    ];
    let widths = [
        col_pid, col_user, col_cpu, col_mem, col_virt, col_rss, col_state, col_time, col_cmd,
    ];
    let mut hx = inner.x;
    for (header, width) in headers.iter().zip(widths.iter()) {
        canvas.draw_text(
            &format_column(header, *width, ColumnAlign::Left, TruncateStrategy::End),
            Point::new(hx, y),
            &TextStyle {
                color: border_color,
                ..Default::default()
            },
        );
        hx += *width as f32 + 1.0;
    }
    y += 1.0;

    // Sort processes by CPU
    let mut procs: Vec<_> = app.system.processes().iter().collect();
    procs.sort_by(|a, b| {
        b.1.cpu_usage()
            .partial_cmp(&a.1.cpu_usage())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (i, (_pid, proc)) in procs.iter().enumerate() {
        if (y - inner.y) as usize >= (inner.height as usize).saturating_sub(2) {
            break;
        }

        let row_rect = Rect::new(inner.x, y, inner.width, 1.0);
        let is_selected = app.process_selected == i;

        let row_hl = RowHighlight::new(row_rect, is_selected);
        row_hl.paint(canvas);
        let text_style = row_hl.text_style();

        let cpu_pct = proc.cpu_usage() as f64;
        let mem_pct = (proc.memory() as f64 / app.mem_total as f64) * 100.0;

        let mut col_x = inner.x;

        // PID
        canvas.draw_text(
            &format_column(
                &proc.pid().to_string(),
                col_pid,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &text_style,
        );
        col_x += col_pid as f32 + 1.0;

        // USER
        let user = proc
            .user_id()
            .map(|u| u.to_string())
            .unwrap_or_else(|| "-".to_string());
        canvas.draw_text(
            &format_column(&user, col_user, ColumnAlign::Left, TruncateStrategy::End),
            Point::new(col_x, y),
            &text_style,
        );
        col_x += col_user as f32 + 1.0;

        // CPU%
        let cpu_color = HeatScheme::Warm.color_for_percent(cpu_pct);
        canvas.draw_text(
            &format_column(
                &format!("{:.1}", cpu_pct),
                col_cpu,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &TextStyle {
                color: if is_selected { Color::WHITE } else { cpu_color },
                ..Default::default()
            },
        );
        col_x += col_cpu as f32 + 1.0;

        // MEM%
        let mem_color = HeatScheme::Warm.color_for_percent(mem_pct);
        canvas.draw_text(
            &format_column(
                &format!("{:.1}", mem_pct),
                col_mem,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &TextStyle {
                color: if is_selected { Color::WHITE } else { mem_color },
                ..Default::default()
            },
        );
        col_x += col_mem as f32 + 1.0;

        // VIRT
        canvas.draw_text(
            &format_column(
                &format_bytes_si(proc.virtual_memory()),
                col_virt,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &text_style,
        );
        col_x += col_virt as f32 + 1.0;

        // RSS
        canvas.draw_text(
            &format_column(
                &format_bytes_si(proc.memory()),
                col_rss,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &text_style,
        );
        col_x += col_rss as f32 + 1.0;

        // STATE
        let state = format!("{:?}", proc.status());
        canvas.draw_text(
            &format_column(&state, col_state, ColumnAlign::Left, TruncateStrategy::End),
            Point::new(col_x, y),
            &text_style,
        );
        col_x += col_state as f32 + 1.0;

        // TIME
        let run_time = proc.run_time();
        let time_str = format!(
            "{}:{:02}:{:02}",
            run_time / 3600,
            (run_time % 3600) / 60,
            run_time % 60
        );
        canvas.draw_text(
            &format_column(
                &time_str,
                col_time,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, y),
            &text_style,
        );
        col_x += col_time as f32 + 1.0;

        // COMMAND
        let cmd = proc.name().to_string_lossy().to_string();
        canvas.draw_text(
            &format_column(&cmd, col_cmd, ColumnAlign::Left, TruncateStrategy::Command),
            Point::new(col_x, y),
            &text_style,
        );

        y += 1.0;
    }
}

/// FULL SCREEN connections exploded view
/// SPEC-024 Section 30: Exploded views fill the screen
pub(super) fn draw_connections_exploded(app: &App, canvas: &mut DirectTerminalCanvas, area: Rect) {
    use crate::widgets::display_rules::{format_column, ColumnAlign, TruncateStrategy};
    use crate::widgets::selection::RowHighlight;

    let border_color = CONNECTIONS_COLOR;

    let conn_data = app.snapshot_connections.as_ref();
    let conn_count = conn_data.map(|d| d.connections.len()).unwrap_or(0);

    let title = format!("▼ CONNECTIONS │ {} active", conn_count);

    let mut border = create_panel_border(&title, border_color, true);
    border.layout(area);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }

    let mut y = inner.y;

    // Column widths
    let col_proto = 6;
    let col_local = 25;
    let col_remote = 25;
    let col_state = 12;
    let col_pid = 8;
    let col_proc = (inner.width as usize)
        .saturating_sub(col_proto + col_local + col_remote + col_state + col_pid + 10);

    // Header
    let header_bg = Color::new(0.12, 0.15, 0.22, 1.0);
    canvas.fill_rect(Rect::new(inner.x, y, inner.width, 1.0), header_bg);

    let headers = [
        "PROTO",
        "LOCAL ADDRESS",
        "REMOTE ADDRESS",
        "STATE",
        "PID",
        "PROCESS",
    ];
    let widths = [
        col_proto, col_local, col_remote, col_state, col_pid, col_proc,
    ];
    let mut hx = inner.x;
    for (header, width) in headers.iter().zip(widths.iter()) {
        canvas.draw_text(
            &format_column(header, *width, ColumnAlign::Left, TruncateStrategy::End),
            Point::new(hx, y),
            &TextStyle {
                color: border_color,
                ..Default::default()
            },
        );
        hx += *width as f32 + 1.0;
    }
    y += 1.0;

    if let Some(data) = conn_data {
        for (i, conn) in data.connections.iter().enumerate() {
            if (y - inner.y) as usize >= (inner.height as usize).saturating_sub(2) {
                break;
            }

            let row_rect = Rect::new(inner.x, y, inner.width, 1.0);
            let is_selected = i == 0;

            let row_hl = RowHighlight::new(row_rect, is_selected);
            row_hl.paint(canvas);
            let text_style = row_hl.text_style();

            let mut col_x = inner.x;

            // Protocol
            canvas.draw_text(
                &format_column("TCP", col_proto, ColumnAlign::Left, TruncateStrategy::End),
                Point::new(col_x, y),
                &text_style,
            );
            col_x += col_proto as f32 + 1.0;

            // Local address
            let local = format!("{}:{}", conn.local_addr, conn.local_port);
            canvas.draw_text(
                &format_column(&local, col_local, ColumnAlign::Left, TruncateStrategy::End),
                Point::new(col_x, y),
                &text_style,
            );
            col_x += col_local as f32 + 1.0;

            // Remote address
            let remote = format!("{}:{}", conn.remote_addr, conn.remote_port);
            canvas.draw_text(
                &format_column(
                    &remote,
                    col_remote,
                    ColumnAlign::Left,
                    TruncateStrategy::End,
                ),
                Point::new(col_x, y),
                &text_style,
            );
            col_x += col_remote as f32 + 1.0;

            // State with color
            let state_str = format!("{:?}", conn.state);
            let state_color = match conn.state {
                TcpState::Established => Color::new(0.3, 0.9, 0.3, 1.0),
                TcpState::Listen => Color::new(0.3, 0.7, 0.9, 1.0),
                TcpState::TimeWait => Color::new(0.7, 0.7, 0.3, 1.0),
                TcpState::CloseWait => Color::new(0.9, 0.5, 0.3, 1.0),
                _ => Color::new(0.6, 0.6, 0.6, 1.0),
            };
            canvas.draw_text(
                &format_column(
                    &state_str,
                    col_state,
                    ColumnAlign::Left,
                    TruncateStrategy::End,
                ),
                Point::new(col_x, y),
                &TextStyle {
                    color: if is_selected {
                        Color::WHITE
                    } else {
                        state_color
                    },
                    ..Default::default()
                },
            );
            col_x += col_state as f32 + 1.0;

            // PID
            let pid_str = conn
                .pid
                .map(|p| p.to_string())
                .unwrap_or_else(|| "-".to_string());
            canvas.draw_text(
                &format_column(&pid_str, col_pid, ColumnAlign::Right, TruncateStrategy::End),
                Point::new(col_x, y),
                &text_style,
            );
            col_x += col_pid as f32 + 1.0;

            // Process name
            let proc_name = conn.process_name.as_deref().unwrap_or("-");
            canvas.draw_text(
                &format_column(
                    proc_name,
                    col_proc,
                    ColumnAlign::Left,
                    TruncateStrategy::Command,
                ),
                Point::new(col_x, y),
                &text_style,
            );

            y += 1.0;
        }
    } else {
        canvas.draw_text(
            "No connection data available",
            Point::new(inner.x, y),
            &TextStyle {
                color: Color::new(0.5, 0.5, 0.5, 1.0),
                ..Default::default()
            },
        );
    }
}
