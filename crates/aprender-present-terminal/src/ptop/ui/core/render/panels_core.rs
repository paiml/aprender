use super::*;

fn get_cpu_load_freq(app: &App) -> (sysinfo::LoadAvg, u64) {
    use sysinfo::Cpu;
    if app.deterministic {
        (
            sysinfo::LoadAvg {
                one: 0.0,
                five: 0.0,
                fifteen: 0.0,
            },
            0,
        )
    } else {
        let freq = app
            .system
            .cpus()
            .iter()
            .map(Cpu::frequency)
            .max()
            .unwrap_or(0);
        (app.load_avg.clone(), freq)
    }
}

/// Draw CPU meters and history graph.
fn draw_cpu_meters_graph(
    app: &App,
    canvas: &mut DirectTerminalCanvas<'_>,
    inner: Rect,
    core_area_height: f32,
    max_freq_mhz: u64,
) {
    let core_count = app.per_core_percent.len();
    let is_exploded = inner.width > 100.0;

    let layout = CpuMeterLayout::calculate(core_count, core_area_height, is_exploded);
    let max_meter_ratio = if is_exploded { 0.70 } else { 0.5 };
    let meters_width =
        (layout.num_meter_cols as f32 * layout.meter_bar_width).min(inner.width * max_meter_ratio);

    let mut grid = CpuGrid::new(app.per_core_percent.clone())
        .with_frequencies(
            app.per_core_freq.iter().map(|&f| f as u32).collect(),
            vec![max_freq_mhz as u32; core_count],
        )
        .with_freq_indicators();

    if is_exploded {
        grid = grid.with_percentages();
    }

    grid.layout(Rect::new(inner.x, inner.y, meters_width, core_area_height));
    grid.paint(canvas);

    let graph_x = inner.x + meters_width + 1.0;
    let graph_width = inner.width - meters_width - 1.0;

    if graph_width > 5.0 && !app.cpu_history.as_slice().is_empty() {
        let history: Vec<f64> = app
            .cpu_history
            .as_slice()
            .iter()
            .map(|&v| v * 100.0)
            .collect();
        let mut graph = BrailleGraph::new(history)
            .with_color(CPU_COLOR)
            .with_range(0.0, 100.0)
            .with_mode(GraphMode::Block);
        graph.layout(Rect::new(graph_x, inner.y, graph_width, core_area_height));
        graph.paint(canvas);
    }
}

/// Format load average string based on available width.
fn format_load_string(
    load: &sysinfo::LoadAvg,
    core_count: usize,
    freq_ghz: f64,
    width: usize,
    deterministic: bool,
) -> String {
    let load_normalized = load.one / core_count as f64;
    let trend_1_5 = load_trend_arrow(load.one, load.five);
    let trend_5_15 = load_trend_arrow(load.five, load.fifteen);
    let load_pct = (load_normalized / 2.0).min(1.0);

    if deterministic {
        let bar = build_load_bar(load_pct, 10);
        format!(
            "Load {bar} {:.2}{trend_1_5} {:.2}{trend_5_15} {:.2} │ Fre",
            load.one, load.five, load.fifteen
        )
    } else if width >= 45 && freq_ghz > 0.0 {
        let bar = build_load_bar(load_pct, 10);
        format!(
            "Load {bar} {:.2}{trend_1_5} {:.2}{trend_5_15} {:.2}→ │ {freq_ghz:.1}GHz",
            load.one, load.five, load.fifteen
        )
    } else if width >= 35 {
        let bar = build_load_bar(load_pct, 10);
        format!(
            "Load {bar} {:.2}{trend_1_5} {:.2}{trend_5_15} {:.2}→",
            load.one, load.five, load.fifteen
        )
    } else {
        let bar = build_load_bar(load_pct, 4);
        format!(
            "Load {bar} {:.1}{trend_1_5} {:.1}{trend_5_15} {:.1}→",
            load.one, load.five, load.fifteen
        )
    }
}

/// Draw load average gauge row.
fn draw_load_gauge(
    canvas: &mut DirectTerminalCanvas<'_>,
    inner: Rect,
    load_y: f32,
    load: &sysinfo::LoadAvg,
    core_count: usize,
    freq_ghz: f64,
    deterministic: bool,
) {
    if load_y >= inner.y + inner.height || inner.width <= 20.0 {
        return;
    }

    let load_normalized = load.one / core_count as f64;
    let load_str = format_load_string(
        load,
        core_count,
        freq_ghz,
        inner.width as usize,
        deterministic,
    );

    canvas.draw_text(
        &load_str,
        Point::new(inner.x, load_y),
        &TextStyle {
            color: load_color(load_normalized),
            ..Default::default()
        },
    );
}

/// Draw top CPU consumers row.
fn draw_top_consumers(
    app: &App,
    canvas: &mut DirectTerminalCanvas<'_>,
    inner: Rect,
    consumers_y: f32,
) {
    if app.deterministic || consumers_y >= inner.y + inner.height || inner.width <= 20.0 {
        return;
    }

    let mut top_procs: Vec<_> = app
        .system
        .processes()
        .values()
        .filter(|p| p.cpu_usage() > 0.1)
        .collect();
    top_procs.sort_by(|a, b| {
        b.cpu_usage()
            .partial_cmp(&a.cpu_usage())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if top_procs.is_empty() {
        return;
    }

    canvas.draw_text(
        "Top ",
        Point::new(inner.x, consumers_y),
        &TextStyle {
            color: DIM_LABEL_COLOR,
            ..Default::default()
        },
    );

    let mut x_offset = 4.0;
    for (i, proc) in top_procs.iter().take(3).enumerate() {
        let cpu = proc.cpu_usage() as f64;
        let name: String = proc.name().to_string_lossy().chars().take(12).collect();

        if i > 0 {
            canvas.draw_text(
                " │ ",
                Point::new(inner.x + x_offset, consumers_y),
                &TextStyle {
                    color: DIM_LABEL_COLOR,
                    ..Default::default()
                },
            );
            x_offset += 3.0;
        }

        let cpu_str = format!("{cpu:.0}%");
        canvas.draw_text(
            &cpu_str,
            Point::new(inner.x + x_offset, consumers_y),
            &TextStyle {
                color: consumer_cpu_color(cpu),
                ..Default::default()
            },
        );
        x_offset += cpu_str.len() as f32;

        canvas.draw_text(
            &format!(" {name}"),
            Point::new(inner.x + x_offset, consumers_y),
            &TextStyle {
                color: PROCESS_NAME_COLOR,
                ..Default::default()
            },
        );
        x_offset += 1.0 + name.len() as f32;
    }
}

pub(super) fn draw_cpu_panel(app: &App, canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    let cpu_pct = app.cpu_history.last().copied().unwrap_or(0.0) * 100.0;
    let core_count = app.per_core_percent.len();
    let uptime = app.uptime();
    let (load, max_freq_mhz) = get_cpu_load_freq(app);

    let is_boosting = max_freq_mhz > 3000;
    let freq_ghz = max_freq_mhz as f64 / 1000.0;

    let title = if bounds.width < 35.0 {
        build_cpu_title_compact(cpu_pct, core_count, freq_ghz, is_boosting)
    } else {
        build_cpu_title(
            cpu_pct,
            core_count,
            freq_ghz,
            is_boosting,
            uptime,
            load.one,
            app.deterministic,
        )
    };

    let is_focused = app.is_panel_focused(PanelType::Cpu);
    let mut border = create_panel_border(&title, CPU_COLOR, is_focused);
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if panel_too_small(&inner, 2.0, 10.0) {
        return;
    }

    let reserved_bottom = 2.0_f32;
    let core_area_height = (inner.height - reserved_bottom).max(1.0);
    let has_cpu_data = !app.deterministic || app.per_core_percent.iter().any(|&p| p > 0.0);

    if has_cpu_data {
        draw_cpu_meters_graph(app, canvas, inner, core_area_height, max_freq_mhz);
    }

    draw_load_gauge(
        canvas,
        inner,
        inner.y + core_area_height,
        &load,
        core_count,
        freq_ghz,
        app.deterministic,
    );
    draw_top_consumers(app, canvas, inner, inner.y + core_area_height + 1.0);
}

// ============================================================================
// Memory Panel - uses extracted helpers from panel_memory module
// Local drawing functions use imported types and colors
// ============================================================================

/// Memory statistics for deterministic rendering (legacy wrapper).
struct MemoryStats {
    used_gb: f64,
    cached_gb: f64,
    free_gb: f64,
}

impl MemoryStats {
    fn from_app(app: &App) -> Self {
        let stats = MemStats::from_bytes(
            app.mem_used,
            app.mem_cached,
            app.mem_available,
            app.mem_total,
        );
        Self {
            used_gb: stats.used_gb,
            cached_gb: stats.cached_gb,
            free_gb: stats.free_gb,
        }
    }
}

/// Draw stacked memory bar (Used|Cached|Free).
fn draw_memory_stacked_bar(canvas: &mut DirectTerminalCanvas<'_>, inner: Rect, y: f32, app: &App) {
    let bar_width = inner.width as usize;
    let used_actual_pct = if app.mem_total > 0 {
        ((app.mem_total - app.mem_available) as f64 / app.mem_total as f64) * 100.0
    } else {
        0.0
    };
    let cached_pct = if app.mem_total > 0 {
        (app.mem_cached as f64 / app.mem_total as f64) * 100.0
    } else {
        0.0
    };

    let used_chars = ((used_actual_pct / 100.0) * bar_width as f64) as usize;
    let cached_chars = ((cached_pct / 100.0) * bar_width as f64) as usize;
    let free_chars = bar_width.saturating_sub(used_chars + cached_chars);

    let used_color = percent_color(used_actual_pct);
    let free_color = Color::new(0.3, 0.3, 0.3, 1.0);

    // Draw used segment
    if used_chars > 0 {
        let used_bar: String = "█".repeat(used_chars);
        canvas.draw_text(
            &used_bar,
            Point::new(inner.x, y),
            &TextStyle {
                color: used_color,
                ..Default::default()
            },
        );
    }

    // Draw cached segment
    if cached_chars > 0 {
        let cached_bar: String = "█".repeat(cached_chars);
        canvas.draw_text(
            &cached_bar,
            Point::new(inner.x + used_chars as f32, y),
            &TextStyle {
                color: CACHED_COLOR,
                ..Default::default()
            },
        );
    }

    // Draw free segment
    if free_chars > 0 {
        let free_bar: String = "░".repeat(free_chars);
        canvas.draw_text(
            &free_bar,
            Point::new(inner.x + used_chars as f32 + cached_chars as f32, y),
            &TextStyle {
                color: free_color,
                ..Default::default()
            },
        );
    }
}

/// Draw memory rows in deterministic mode (ttop style).
fn draw_memory_rows_deterministic(
    canvas: &mut DirectTerminalCanvas<'_>,
    inner: Rect,
    mut y: f32,
    stats: &MemoryStats,
) -> f32 {
    // Used row
    canvas.draw_text(
        &format!("  Used:   {:.1}G  0", stats.used_gb),
        Point::new(inner.x, y),
        &TextStyle {
            color: percent_color(0.0),
            ..Default::default()
        },
    );
    y += 1.0;

    // Cached row
    if y < inner.y + inner.height {
        canvas.draw_text(
            &format!("Cached:   {:.1}G  0", stats.cached_gb),
            Point::new(inner.x, y),
            &TextStyle {
                color: CACHED_COLOR,
                ..Default::default()
            },
        );
        y += 1.0;
    }

    // Free row
    if y < inner.y + inner.height {
        canvas.draw_text(
            &format!("  Free:   {:.1}G  0", stats.free_gb),
            Point::new(inner.x, y),
            &TextStyle {
                color: FREE_COLOR,
                ..Default::default()
            },
        );
        y += 1.0;
    }

    // PSI row (ttop style)
    if y < inner.y + inner.height {
        canvas.draw_text(
            "PSI ○ 0.0 cpu ○ 0.0 mem ○ 0.0 io",
            Point::new(inner.x, y),
            &TextStyle {
                color: DIM_COLOR,
                ..Default::default()
            },
        );
        y += 1.0;
    }

    // Top Memory Consumers header
    if y < inner.y + inner.height {
        canvas.draw_text(
            "── Top Memory Consumers ──────────────",
            Point::new(inner.x, y),
            &TextStyle {
                color: DIM_COLOR,
                ..Default::default()
            },
        );
    }

    y
}

// ============================================================================

#[allow(clippy::too_many_lines)]
/// Compute memory percentages from app state.
fn compute_mem_percentages(app: &App) -> (f64, f64, f64, f64) {
    let mem_total = app.mem_total;
    let swap_total = app.swap_total;
    let used_pct = safe_pct(app.mem_used, mem_total);
    let cached_pct = safe_pct(app.mem_cached, mem_total);
    let free_pct = safe_pct(app.mem_available, mem_total);
    let swap_pct = safe_pct(app.swap_used, swap_total);
    (used_pct, cached_pct, free_pct, swap_pct)
}

/// Build memory rows vector with optional ZRAM row.
fn build_memory_rows(app: &App, has_zram: bool) -> Vec<(&'static str, f64, f64, Color)> {
    let gb = |b: u64| b as f64 / 1024.0 / 1024.0 / 1024.0;
    let (used_pct, cached_pct, free_pct, swap_pct) = compute_mem_percentages(app);
    let mut rows: Vec<(&str, f64, f64, Color)> = vec![
        ("Used", gb(app.mem_used), used_pct, percent_color(used_pct)),
        ("Swap", gb(app.swap_used), swap_pct, swap_color(swap_pct)),
        ("Cached", gb(app.mem_cached), cached_pct, CACHED_COLOR),
        ("Free", gb(app.mem_available), free_pct, FREE_COLOR),
    ];
    if has_zram {
        rows.insert(
            2,
            (
                "ZRAM",
                0.0,
                0.0,
                Color {
                    r: 0.8,
                    g: 0.4,
                    b: 1.0,
                    a: 1.0,
                },
            ),
        );
    }
    rows
}

/// Draw ZRAM row in ttop style.
fn draw_zram_row(
    canvas: &mut DirectTerminalCanvas<'_>,
    inner: Rect,
    y: f32,
    zram_data: &(f64, f64, f64, &str),
) {
    let (orig_gb, compr_gb, ratio, algo) = zram_data;
    let orig_str = ZramDisplay::format_size(*orig_gb);
    let compr_str = ZramDisplay::format_size(*compr_gb);
    canvas.draw_text(
        "  ZRAM ",
        Point::new(inner.x, y),
        &TextStyle {
            color: DIM_COLOR,
            ..Default::default()
        },
    );
    canvas.draw_text(
        &format!("{orig_str}→{compr_str} "),
        Point::new(inner.x + 7.0, y),
        &TextStyle {
            color: ZRAM_COLOR,
            ..Default::default()
        },
    );
    let ratio_x = inner.x + 7.0 + orig_str.len() as f32 + 1.0 + compr_str.len() as f32 + 1.0;
    canvas.draw_text(
        &format!("{ratio:.1}x"),
        Point::new(ratio_x, y),
        &TextStyle {
            color: RATIO_COLOR,
            ..Default::default()
        },
    );
    canvas.draw_text(
        &format!(" {algo}"),
        Point::new(ratio_x + 4.0, y),
        &TextStyle {
            color: DIM_COLOR,
            ..Default::default()
        },
    );
}

/// Draw a single memory row with progress bar.
fn draw_memory_row_bar(
    canvas: &mut DirectTerminalCanvas<'_>,
    inner: Rect,
    y: f32,
    label: &str,
    value: f64,
    pct: f64,
    color: Color,
) {
    let bar_width = 10.min((inner.width as usize).saturating_sub(22));
    let bar = make_bar(pct / 100.0, bar_width);
    let text = format!("{label:>6} {value:>5.1}G {bar} {pct:>5.1}%");
    canvas.draw_text(
        &text,
        Point::new(inner.x, y),
        &TextStyle {
            color,
            ..Default::default()
        },
    );
}

/// Draw swap thrashing indicator if active.
fn draw_swap_thrash_indicator(
    app: &App,
    canvas: &mut DirectTerminalCanvas<'_>,
    inner: Rect,
    y: f32,
) {
    if let Some(swap_data) = app.analyzers.swap_data() {
        let (is_thrashing, severity) = swap_data.is_thrashing();
        if has_swap_activity(
            is_thrashing,
            swap_data.swap_in_rate,
            swap_data.swap_out_rate,
        ) {
            let (indicator, ind_color) = thrashing_indicator(severity);
            let bar_width = 10.min((inner.width as usize).saturating_sub(22));
            let thrash_x = inner.x + 28.0 + bar_width as f32;
            let thrash_text = format!(
                " {indicator} I:{:.0}/O:{:.0}",
                swap_data.swap_in_rate, swap_data.swap_out_rate
            );
            canvas.draw_text(
                &thrash_text,
                Point::new(thrash_x, y),
                &TextStyle {
                    color: ind_color,
                    ..Default::default()
                },
            );
        }
    }
}

/// Draw PSI memory pressure indicator.
fn draw_mem_psi_indicator(app: &App, canvas: &mut DirectTerminalCanvas<'_>, inner: Rect, y: f32) {
    if let Some(psi) = app.psi_data() {
        let mem_some = psi.memory.some.avg10;
        let mem_full = psi.memory.full.as_ref().map_or(0.0, |f| f.avg10);
        let (symbol, color) = psi_memory_indicator(mem_some, mem_full);
        let psi_text = format!("   PSI {symbol} {mem_some:>5.1}% some {mem_full:>5.1}% full");
        canvas.draw_text(
            &psi_text,
            Point::new(inner.x, y),
            &TextStyle {
                color,
                ..Default::default()
            },
        );
    }
}

/// Draw memory rows in normal mode with bars and indicators.
fn draw_memory_rows_normal(
    app: &App,
    canvas: &mut DirectTerminalCanvas<'_>,
    inner: Rect,
    mut y: f32,
    rows: &[(&str, f64, f64, Color)],
    zram_data: Option<(f64, f64, f64, &str)>,
) {
    for (label, value, pct, color) in rows {
        if y >= inner.y + inner.height {
            break;
        }
        if *label == "ZRAM" {
            if let Some(ref data) = zram_data {
                draw_zram_row(canvas, inner, y, data);
            }
            y += 1.0;
            continue;
        }
        draw_memory_row_bar(canvas, inner, y, label, *value, *pct, *color);
        if *label == "Swap" {
            draw_swap_thrash_indicator(app, canvas, inner, y);
        }
        y += 1.0;
    }
    if y < inner.y + inner.height {
        draw_mem_psi_indicator(app, canvas, inner, y);
    }
}

pub(super) fn draw_memory_panel(app: &App, canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    let _detail_level = DetailLevel::for_height(bounds.height as u16);
    let gb = |b: u64| b as f64 / 1024.0 / 1024.0 / 1024.0;
    let mem_pct = if app.mem_total > 0 {
        (app.mem_used as f64 / app.mem_total as f64) * 100.0
    } else {
        0.0
    };

    let zram_stats = if app.deterministic {
        None
    } else {
        read_zram_stats()
    };
    let zram_info = zram_stats
        .as_ref()
        .filter(|z| z.is_active())
        .map(|z| format!(" │ ZRAM:{:.1}x", z.ratio()))
        .unwrap_or_default();
    let title = format!(
        "Memory │ {:.1}G / {:.1}G ({:.0}%){}",
        gb(app.mem_used),
        gb(app.mem_total),
        mem_pct,
        zram_info
    );

    let is_focused = app.is_panel_focused(PanelType::Memory);
    let mut border = create_panel_border(&title, MEMORY_COLOR, is_focused);
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 || inner.width < 10.0 {
        return;
    }

    let mut y = inner.y;
    draw_memory_stacked_bar(canvas, inner, y, app);
    y += 1.0;

    if y >= inner.y + inner.height {
        return;
    }

    let zram_row_data = zram_stats.as_ref().filter(|z| z.is_active()).map(|z| {
        (
            gb(z.orig_data_size),
            gb(z.compr_data_size),
            z.ratio(),
            z.algorithm.as_str(),
        )
    });
    let rows = build_memory_rows(app, zram_row_data.is_some());

    if app.deterministic {
        let stats = MemoryStats::from_app(app);
        draw_memory_rows_deterministic(canvas, inner, y, &stats);
    } else {
        draw_memory_rows_normal(app, canvas, inner, y, &rows, zram_row_data);
    }
}

/// Compute total disk stats (used, space, read_rate, write_rate).
fn compute_disk_stats(app: &App) -> (u64, u64, f64, f64) {
    if app.deterministic {
        return (0, 0, 0.0, 0.0);
    }
    let disk_io = app.disk_io_data();
    let (used, space): (u64, u64) = app
        .disks
        .iter()
        .map(|d| (d.total_space() - d.available_space(), d.total_space()))
        .fold((0, 0), |(au, at), (u, t)| (au + u, at + t));
    let r_rate = disk_io.map_or(0.0, |d| d.total_read_bytes_per_sec);
    let w_rate = disk_io.map_or(0.0, |d| d.total_write_bytes_per_sec);
    (used, space, r_rate, w_rate)
}

/// Format disk panel title.
fn format_disk_title(
    deterministic: bool,
    used: u64,
    space: u64,
    r_rate: f64,
    w_rate: f64,
) -> String {
    let gb = |b: u64| b as f64 / 1024.0 / 1024.0 / 1024.0;
    if deterministic {
        "Disk │ R: 0B/s │ W: 0B/s │ -0 IOPS │".to_string()
    } else if r_rate > 0.0 || w_rate > 0.0 {
        format!(
            "Disk │ R: {} │ W: {} │ {:.0}G / {:.0}G",
            format_bytes_rate(r_rate),
            format_bytes_rate(w_rate),
            gb(used),
            gb(space)
        )
    } else {
        let pct = if space > 0 {
            (used as f64 / space as f64) * 100.0
        } else {
            0.0
        };
        format!("Disk │ {:.0}G / {:.0}G ({:.0}%)", gb(used), gb(space), pct)
    }
}

/// Draw disk panel in deterministic mode.
fn draw_disk_deterministic(canvas: &mut DirectTerminalCanvas<'_>, inner: Rect) {
    let dim_color = Color {
        r: 0.3,
        g: 0.3,
        b: 0.3,
        a: 1.0,
    };
    canvas.draw_text(
        "I/O Pressure ○  0.0% some    0.0% full",
        Point::new(inner.x, inner.y),
        &TextStyle {
            color: dim_color,
            ..Default::default()
        },
    );
    if inner.height >= 2.0 {
        canvas.draw_text(
            "── Top Active Processes ──────────────",
            Point::new(inner.x, inner.y + 1.0),
            &TextStyle {
                color: dim_color,
                ..Default::default()
            },
        );
    }
}

/// Get I/O rates for a specific disk device.
fn get_disk_io_rates(app: &App, device_name: &str) -> (f64, f64) {
    app.disk_io_data()
        .and_then(|data| data.rates.get(device_name))
        .map_or((0.0, 0.0), |rate| {
            (rate.read_bytes_per_sec, rate.write_bytes_per_sec)
        })
}

/// Draw a single disk row.
fn draw_disk_row(
    canvas: &mut DirectTerminalCanvas<'_>,
    inner: Rect,
    y: f32,
    disk: &sysinfo::Disk,
    d_read: f64,
    d_write: f64,
) {
    let mount = disk.mount_point().to_string_lossy();
    let mount_short: String = if mount == "/" {
        "/".to_string()
    } else {
        mount
            .split('/')
            .next_back()
            .unwrap_or(&mount)
            .chars()
            .take(8)
            .collect()
    };
    let total = disk.total_space();
    let used = total - disk.available_space();
    let pct = safe_pct(used, total);
    let total_gb = total as f64 / 1024.0 / 1024.0 / 1024.0;
    let io_str = if d_read > 0.0 || d_write > 0.0 {
        format!(
            " R:{} W:{}",
            format_bytes_rate(d_read),
            format_bytes_rate(d_write)
        )
    } else {
        String::new()
    };
    let bar_width = (inner.width as usize)
        .saturating_sub(24 + io_str.len())
        .max(2);
    let bar = make_bar(pct / 100.0, bar_width);
    let text = format!("{mount_short:<8} {total_gb:>5.0}G {bar} {pct:>5.1}%{io_str}");
    let color = if d_read > 1024.0 || d_write > 1024.0 {
        Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        }
    } else {
        percent_color(pct)
    };
    canvas.draw_text(
        &text,
        Point::new(inner.x, y),
        &TextStyle {
            color,
            ..Default::default()
        },
    );
}

pub(super) fn draw_disk_panel(app: &App, canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    let (total_used, total_space, read_rate, write_rate) = compute_disk_stats(app);
    let title = format_disk_title(
        app.deterministic,
        total_used,
        total_space,
        read_rate,
        write_rate,
    );

    let is_focused = app.is_panel_focused(PanelType::Disk);
    let mut border = create_panel_border(&title, DISK_COLOR, is_focused);
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }
    if app.deterministic {
        draw_disk_deterministic(canvas, inner);
        return;
    }

    let max_disks = inner.height as usize;
    for (i, disk) in app.disks.iter().take(max_disks).enumerate() {
        let y = inner.y + i as f32;
        if y >= inner.y + inner.height {
            break;
        }
        let disk_name = disk.name().to_string_lossy();
        let device_name = disk_name.trim_start_matches("/dev/");
        let (d_read, d_write) = get_disk_io_rates(app, device_name);
        draw_disk_row(canvas, inner, y, disk, d_read, d_write);
    }
}

/// Compute network stats (rx_total, tx_total, primary_iface).
fn compute_network_stats(app: &App) -> (u64, u64, &str) {
    if app.deterministic {
        return (0, 0, "none");
    }
    let (rx, tx): (u64, u64) = app
        .networks
        .values()
        .map(|d| (d.received(), d.transmitted()))
        .fold((0, 0), |(ar, at), (r, t)| (ar + r, at + t));
    let iface = app
        .networks
        .iter()
        .filter(|(name, _)| !name.starts_with("lo"))
        .max_by_key(|(_, data)| data.received() + data.transmitted())
        .map_or("none", |(name, _)| name.as_str());
    (rx, tx, iface)
}

/// Draw network deterministic download/upload rows.
fn draw_net_dl_ul_rows(canvas: &mut DirectTerminalCanvas<'_>, inner: Rect, y: &mut f32) {
    let cyan = Color {
        r: 0.3,
        g: 0.8,
        b: 0.9,
        a: 1.0,
    };
    let red = Color {
        r: 1.0,
        g: 0.3,
        b: 0.3,
        a: 1.0,
    };
    let white = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    canvas.draw_text(
        "↓",
        Point::new(inner.x, *y),
        &TextStyle {
            color: cyan,
            ..Default::default()
        },
    );
    canvas.draw_text(
        " Download ",
        Point::new(inner.x + 1.0, *y),
        &TextStyle {
            color: cyan,
            ..Default::default()
        },
    );
    canvas.draw_text(
        "0B/s",
        Point::new(inner.x + 11.0, *y),
        &TextStyle {
            color: white,
            ..Default::default()
        },
    );
    *y += 1.0;
    if *y < inner.y + inner.height {
        canvas.draw_text(
            &"⠀".repeat(inner.width as usize),
            Point::new(inner.x, *y),
            &TextStyle {
                color: cyan,
                ..Default::default()
            },
        );
        *y += 1.0;
    }
    if *y < inner.y + inner.height {
        canvas.draw_text(
            "↑",
            Point::new(inner.x, *y),
            &TextStyle {
                color: red,
                ..Default::default()
            },
        );
        canvas.draw_text(
            " Upload   ",
            Point::new(inner.x + 1.0, *y),
            &TextStyle {
                color: red,
                ..Default::default()
            },
        );
        canvas.draw_text(
            "0B/s",
            Point::new(inner.x + 11.0, *y),
            &TextStyle {
                color: white,
                ..Default::default()
            },
        );
        *y += 1.0;
    }
    for _ in 0..2 {
        if *y < inner.y + inner.height {
            canvas.draw_text(
                &"⠀".repeat(inner.width as usize),
                Point::new(inner.x, *y),
                &TextStyle {
                    color: red,
                    ..Default::default()
                },
            );
            *y += 1.0;
        }
    }
}

/// Draw network deterministic session and TCP/UDP rows.
fn draw_net_session_stats(canvas: &mut DirectTerminalCanvas<'_>, inner: Rect, y: f32) {
    let cyan = Color {
        r: 0.3,
        g: 0.8,
        b: 0.9,
        a: 1.0,
    };
    let red = Color {
        r: 1.0,
        g: 0.3,
        b: 0.3,
        a: 1.0,
    };
    let dim = Color {
        r: 0.3,
        g: 0.3,
        b: 0.3,
        a: 1.0,
    };
    let white = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    let green = Color {
        r: 0.3,
        g: 0.9,
        b: 0.3,
        a: 1.0,
    };
    let mut y = y;
    if y < inner.y + inner.height {
        canvas.draw_text(
            "Session ",
            Point::new(inner.x, y),
            &TextStyle {
                color: dim,
                ..Default::default()
            },
        );
        canvas.draw_text(
            "↓",
            Point::new(inner.x + 8.0, y),
            &TextStyle {
                color: cyan,
                ..Default::default()
            },
        );
        canvas.draw_text(
            "0B",
            Point::new(inner.x + 9.0, y),
            &TextStyle {
                color: white,
                ..Default::default()
            },
        );
        canvas.draw_text(
            " ↑",
            Point::new(inner.x + 11.0, y),
            &TextStyle {
                color: red,
                ..Default::default()
            },
        );
        canvas.draw_text(
            "0B",
            Point::new(inner.x + 13.0, y),
            &TextStyle {
                color: white,
                ..Default::default()
            },
        );
        y += 1.0;
    }
    if y < inner.y + inner.height {
        let tcp_col = Color {
            r: 0.3,
            g: 0.7,
            b: 0.9,
            a: 1.0,
        };
        let udp_col = Color {
            r: 0.8,
            g: 0.3,
            b: 0.8,
            a: 1.0,
        };
        canvas.draw_text(
            "TCP ",
            Point::new(inner.x, y),
            &TextStyle {
                color: tcp_col,
                ..Default::default()
            },
        );
        canvas.draw_text(
            "0",
            Point::new(inner.x + 4.0, y),
            &TextStyle {
                color: green,
                ..Default::default()
            },
        );
        canvas.draw_text(
            "/",
            Point::new(inner.x + 5.0, y),
            &TextStyle {
                color: dim,
                ..Default::default()
            },
        );
        canvas.draw_text(
            "0",
            Point::new(inner.x + 6.0, y),
            &TextStyle {
                color: tcp_col,
                ..Default::default()
            },
        );
        canvas.draw_text(
            " UDP ",
            Point::new(inner.x + 7.0, y),
            &TextStyle {
                color: udp_col,
                ..Default::default()
            },
        );
        canvas.draw_text(
            "0",
            Point::new(inner.x + 12.0, y),
            &TextStyle {
                color: white,
                ..Default::default()
            },
        );
        canvas.draw_text(
            " │ RTT ",
            Point::new(inner.x + 13.0, y),
            &TextStyle {
                color: dim,
                ..Default::default()
            },
        );
        canvas.draw_text(
            "●●●●●",
            Point::new(inner.x + 20.0, y),
            &TextStyle {
                color: green,
                ..Default::default()
            },
        );
    }
}

/// Draw network panel in deterministic mode.
fn draw_network_deterministic(canvas: &mut DirectTerminalCanvas<'_>, inner: Rect) {
    let mut y = inner.y;
    draw_net_dl_ul_rows(canvas, inner, &mut y);
    draw_net_session_stats(canvas, inner, y);
}

/// Build network interfaces list with stats.
fn build_network_interfaces(app: &App) -> Vec<NetworkInterface> {
    let network_stats_data = app.analyzers.network_stats_data();
    let mut interfaces: Vec<NetworkInterface> = Vec::new();
    for (name, data) in &app.networks {
        let mut iface = NetworkInterface::new(name);
        iface.update(data.received() as f64, data.transmitted() as f64);
        iface.set_totals(data.total_received(), data.total_transmitted());
        if let Some(stats_data) = network_stats_data {
            if let Some(stats) = stats_data.stats.get(name.as_str()) {
                iface.set_stats(
                    stats.rx_errors,
                    stats.tx_errors,
                    stats.rx_dropped,
                    stats.tx_dropped,
                );
            }
            if let Some(rates) = stats_data.rates.get(name.as_str()) {
                iface.set_rates(rates.errors_per_sec, rates.drops_per_sec);
                iface.set_utilization(rates.utilization_percent());
            }
        }
        interfaces.push(iface);
    }
    interfaces.sort_by(|a, b| {
        (b.rx_bps + b.tx_bps)
            .partial_cmp(&(a.rx_bps + a.tx_bps))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    interfaces
}

pub(super) fn draw_network_panel(app: &App, canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    let (rx_total, tx_total, primary_iface) = compute_network_stats(app);
    let title = format!(
        "Network ({}) │ ↓ {}/s │ ↑ {}/s",
        primary_iface,
        format_bytes(rx_total),
        format_bytes(tx_total)
    );

    let is_focused = app.is_panel_focused(PanelType::Network);
    let mut border = create_panel_border(&title, NETWORK_COLOR, is_focused);
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if app.deterministic {
        draw_network_deterministic(canvas, inner);
        return;
    }

    let mut interfaces = build_network_interfaces(app);
    for iface in interfaces.iter_mut() {
        if let Some((rx_hist, tx_hist)) = app.net_iface_history.get(&iface.name) {
            iface.rx_history = rx_hist.as_slice().to_vec();
            iface.tx_history = tx_hist.as_slice().to_vec();
        }
    }
    interfaces.truncate(4);

    if !interfaces.is_empty() && inner.height > 0.0 {
        let spark_w = (inner.width as usize / 4).max(5);
        let mut panel = NetworkPanel::new()
            .with_spark_width(spark_w)
            .with_rx_color(NET_RX_COLOR)
            .with_tx_color(NET_TX_COLOR)
            .compact();
        panel.set_interfaces(interfaces);
        panel.layout(inner);
        panel.paint(canvas);
    }
}

/// Get display name for sort column
fn sort_column_name(col: ProcessSortColumn) -> &'static str {
    match col {
        ProcessSortColumn::Cpu => "CPU%",
        ProcessSortColumn::Mem => "MEM%",
        ProcessSortColumn::Pid => "PID",
        ProcessSortColumn::User => "USER",
        ProcessSortColumn::Command => "CMD",
    }
}

/// Convert sysinfo process status to ProcessState
fn convert_process_status(status: sysinfo::ProcessStatus) -> ProcessState {
    match status {
        sysinfo::ProcessStatus::Run => ProcessState::Running,
        sysinfo::ProcessStatus::Sleep => ProcessState::Sleeping,
        sysinfo::ProcessStatus::Idle => ProcessState::Idle,
        sysinfo::ProcessStatus::Zombie => ProcessState::Zombie,
        sysinfo::ProcessStatus::Stop => ProcessState::Stopped,
        sysinfo::ProcessStatus::UninterruptibleDiskSleep => ProcessState::DiskWait,
        _ => ProcessState::Sleeping,
    }
}

/// Get process command string (either name or full cmdline in exploded mode)
fn get_process_command(p: &sysinfo::Process, is_exploded: bool, max_len: usize) -> String {
    if is_exploded {
        let cmdline: Vec<String> = p
            .cmd()
            .iter()
            .map(|s| s.to_string_lossy().to_string())
            .collect();
        if cmdline.is_empty() {
            p.name().to_string_lossy().chars().take(max_len).collect()
        } else {
            cmdline.join(" ").chars().take(max_len).collect()
        }
    } else {
        p.name().to_string_lossy().chars().take(max_len).collect()
    }
}

pub(super) fn draw_process_panel(app: &App, canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    let sort_name = sort_column_name(app.sort_column);
    let arrow = if app.sort_descending { "▼" } else { "▲" };
    let filter_str = if app.filter.is_empty() {
        String::new()
    } else {
        format!(" │ Filter: \"{}\"", app.filter)
    };
    let title = format!(
        "Processes ({}) │ Sort: {} {}{}",
        app.process_count(),
        sort_name,
        arrow,
        filter_str
    );

    let is_focused = app.is_panel_focused(PanelType::Process);
    let mut border = create_panel_border(&title, PROCESS_COLOR, is_focused);
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 2.0 {
        return;
    }

    if app.deterministic {
        canvas.draw_text(
            "PID    S  C%   M%   COMMAND",
            Point::new(inner.x, inner.y),
            &TextStyle {
                color: PROCESS_COLOR,
                ..Default::default()
            },
        );
        return;
    }

    let procs = app.sorted_processes();
    let total_mem = app.mem_total as f64;
    let process_extra_data = app.analyzers.process_extra_data();
    let is_exploded = inner.height > 30.0 || inner.width > 100.0;
    let max_cmd_len = if is_exploded { 200 } else { 40 };

    let entries: Vec<ProcessEntry> = procs
        .iter()
        .take(if is_exploded { 500 } else { 100 })
        .map(|p| {
            let pid = p.pid().as_u32();
            let mem_pct = if total_mem > 0.0 {
                (p.memory() as f64 / total_mem) * 100.0
            } else {
                0.0
            };
            let user = p
                .user_id()
                .and_then(|uid| app.users.get_user_by_id(uid))
                .map(|u| u.name().to_string())
                .unwrap_or_else(|| "-".to_string());
            let user_short: String = user.chars().take(8).collect();
            let cmd = get_process_command(p, is_exploded, max_cmd_len);
            let state = convert_process_status(p.status());

            let mut entry =
                ProcessEntry::new(pid, &user_short, p.cpu_usage(), mem_pct as f32, &cmd)
                    .with_state(state);
            if let Some(extra_data) = process_extra_data {
                if let Some(extra) = extra_data.get(pid) {
                    entry = entry
                        .with_oom_score(extra.oom_score)
                        .with_cgroup(extra.cgroup_short())
                        .with_nice(extra.nice)
                        .with_threads(extra.num_threads);
                }
            }
            entry
        })
        .collect();

    let mut table = if is_exploded {
        ProcessTable::new().with_cmdline().with_threads_column()
    } else {
        ProcessTable::new().compact().with_threads_column()
    };
    table.set_processes(entries);
    table.select(app.process_selected);
    table.layout(inner);
    table.paint(canvas);
}

pub(super) fn draw_help_overlay(canvas: &mut DirectTerminalCanvas<'_>, w: f32, h: f32) {
    let popup_w = 55.0;
    let popup_h = 27.0; // Expanded for signal keybindings (SPEC-024 Appendix G.6)
    let px = (w - popup_w) / 2.0;
    let py = (h - popup_h) / 2.0;

    // Clear background
    for y in 0..popup_h as u16 {
        let spaces: String = (0..popup_w as usize).map(|_| ' ').collect();
        canvas.draw_text(
            &spaces,
            Point::new(px, py + y as f32),
            &TextStyle {
                color: Color::new(0.1, 0.1, 0.15, 1.0),
                ..Default::default()
            },
        );
    }

    let mut border = Border::new()
        .with_title(" Help ")
        .with_style(BorderStyle::Double)
        .with_color(Color::new(0.3, 0.8, 0.9, 1.0));
    border.layout(Rect::new(px, py, popup_w, popup_h));
    border.paint(canvas);

    let text_style = TextStyle {
        color: Color::new(0.9, 0.9, 0.9, 1.0),
        ..Default::default()
    };
    let key_style = TextStyle {
        color: Color::new(0.3, 0.8, 0.9, 1.0),
        ..Default::default()
    };
    let section_style = TextStyle {
        color: Color::new(0.8, 0.8, 0.2, 1.0),
        ..Default::default()
    };

    // Help content with section headers (SPEC-024 v5.0 Feature D, Appendix G.6)
    let help_lines: &[(&str, &str, bool)] = &[
        ("", "-- General --", true),
        ("q, Esc, Ctrl+C", "Quit", false),
        ("h, ?", "Toggle help", false),
        ("", "-- Panel Navigation --", true),
        ("Tab", "Focus next panel", false),
        ("Shift+Tab", "Focus previous panel", false),
        ("hjkl", "Vim-style focus navigation", false),
        ("Enter, z", "Explode/zoom focused panel", false),
        ("", "-- Process List --", true),
        ("j/k, ↑/↓", "Navigate processes", false),
        ("PgUp/PgDn", "Page up/down", false),
        ("g/G", "Go to top/bottom", false),
        ("c/m/p", "Sort by CPU/Memory/PID", false),
        ("s", "Cycle sort column", false),
        ("r", "Reverse sort", false),
        ("/, f", "Filter processes", false),
        ("Delete", "Clear filter", false),
        ("", "-- Signals --", true),
        ("x", "SIGTERM (graceful stop)", false),
        ("X", "SIGKILL (force kill)", false),
        ("", "-- Panels --", true),
        ("1-5", "Toggle panels", false),
        ("0", "Reset panels", false),
    ];

    for (i, (key, desc, is_section)) in help_lines.iter().enumerate() {
        let y = py + 1.0 + i as f32;
        if *is_section {
            canvas.draw_text(desc, Point::new(px + 2.0, y), &section_style);
        } else {
            canvas.draw_text(&format!("{key:>14}"), Point::new(px + 2.0, y), &key_style);
            canvas.draw_text(desc, Point::new(px + 18.0, y), &text_style);
        }
    }
}

/// Draw signal confirmation dialog (SPEC-024 Appendix G.6 P0)
pub(super) fn draw_signal_dialog(app: &App, canvas: &mut DirectTerminalCanvas<'_>, w: f32, h: f32) {
    use crate::ptop::config::SignalType;

    let Some((pid, ref name, signal)) = app.pending_signal else {
        return;
    };

    let popup_w = 50.0;
    let popup_h = 7.0;
    let px = (w - popup_w) / 2.0;
    let py = (h - popup_h) / 2.0;

    // Clear background
    for y in 0..popup_h as u16 {
        let spaces: String = (0..popup_w as usize).map(|_| ' ').collect();
        canvas.draw_text(
            &spaces,
            Point::new(px, py + y as f32),
            &TextStyle {
                color: Color::new(0.15, 0.1, 0.1, 1.0),
                ..Default::default()
            },
        );
    }

    // Border color based on signal severity
    let border_color = match signal {
        SignalType::Kill => Color::new(1.0, 0.3, 0.3, 1.0), // Red for SIGKILL
        SignalType::Term => Color::new(1.0, 0.8, 0.2, 1.0), // Yellow for SIGTERM
        SignalType::Stop => Color::new(0.8, 0.4, 1.0, 1.0), // Purple for SIGSTOP
        _ => Color::new(0.3, 0.8, 0.9, 1.0),                // Cyan for others
    };

    let mut border = Border::new()
        .with_title(format!(" Send SIG{} ", signal.name()))
        .with_style(BorderStyle::Double)
        .with_color(border_color);
    border.layout(Rect::new(px, py, popup_w, popup_h));
    border.paint(canvas);

    let text_style = TextStyle {
        color: Color::new(0.9, 0.9, 0.9, 1.0),
        ..Default::default()
    };
    let warning_style = TextStyle {
        color: border_color,
        ..Default::default()
    };
    let hint_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };

    // Truncate process name if too long
    let max_name_len = 25;
    let display_name = if name.len() > max_name_len {
        format!("{}...", &name[..max_name_len - 3])
    } else {
        name.clone()
    };

    // Dialog content
    canvas.draw_text(
        &format!("Process: {} (PID {})", display_name, pid),
        Point::new(px + 2.0, py + 1.0),
        &text_style,
    );
    canvas.draw_text(
        &format!("Signal: {} - {}", signal.name(), signal.description()),
        Point::new(px + 2.0, py + 2.0),
        &warning_style,
    );

    // Confirmation prompt
    canvas.draw_text("", Point::new(px + 2.0, py + 3.0), &text_style);
    canvas.draw_text(
        "Send signal? [Y]es / [n]o / [Esc] cancel",
        Point::new(px + 2.0, py + 4.0),
        &text_style,
    );
    canvas.draw_text(
        "x=TERM  K=KILL  H=HUP  i=INT  p=STOP",
        Point::new(px + 2.0, py + 5.0),
        &hint_style,
    );
}

pub(super) fn draw_filter_overlay(
    app: &App,
    canvas: &mut DirectTerminalCanvas<'_>,
    w: f32,
    h: f32,
) {
    let popup_w = 45.0;
    let popup_h = 3.0;
    let px = (w - popup_w) / 2.0;
    let py = (h - popup_h) / 2.0;

    let mut border = Border::new()
        .with_title(" Filter Processes ")
        .with_style(BorderStyle::Rounded)
        .with_color(Color::new(0.3, 0.8, 0.9, 1.0));
    border.layout(Rect::new(px, py, popup_w, popup_h));
    border.paint(canvas);

    let filter_display = format!("{}_", app.filter);
    canvas.draw_text(
        &filter_display,
        Point::new(px + 2.0, py + 1.0),
        &TextStyle {
            color: Color::new(1.0, 1.0, 1.0, 1.0),
            ..Default::default()
        },
    );
}
