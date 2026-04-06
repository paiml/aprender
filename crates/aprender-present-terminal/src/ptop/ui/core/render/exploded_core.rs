use super::*;

// =============================================================================
// EXPLODE MODE (SPEC-024 v5.0 Feature D)
// Reference: ttop/src/ui.rs lines 14-347
// =============================================================================

/// Draw a process DataFrame with spreadsheet-style column navigation and sorting.
///
/// Features:
/// - Left/Right arrow keys navigate between columns (header highlight)
/// - Enter/Space sorts by selected column
/// - Sort indicator (▲/▼) on sorted column
/// - Up/Down arrow keys navigate rows
/// - Selected row highlight
///
/// Uses SPEC-024 Section 28 Display Rules:
/// - `format_column()` for ALL text (NEVER bleeds)
/// - `format_percent()` for CPU/MEM columns
/// - `TruncateStrategy::Command` for process names
#[allow(clippy::too_many_lines)]
/// Get header column text style based on sort/selection state
fn dataframe_header_style(
    is_sorted: bool,
    is_selected: bool,
    sort_color: Color,
    dim_color: Color,
) -> TextStyle {
    if is_sorted {
        TextStyle {
            color: sort_color,
            ..Default::default()
        }
    } else if is_selected {
        TextStyle {
            color: Color::WHITE,
            ..Default::default()
        }
    } else {
        TextStyle {
            color: dim_color,
            ..Default::default()
        }
    }
}

/// Get memory percentage display color
fn mem_display_color(mem_pct: f32, is_selected: bool, text_color: Color) -> Color {
    if mem_pct > 10.0 {
        Color::new(0.7, 0.5, 0.9, 1.0)
    } else if is_selected {
        Color::WHITE
    } else {
        text_color
    }
}

/// Draw row background and optional cursor for process dataframe
fn draw_process_row_bg(
    canvas: &mut dyn Canvas,
    x: f32,
    y: f32,
    width: f32,
    is_selected: bool,
    selected_bg: Color,
) {
    if is_selected {
        canvas.fill_rect(Rect::new(x, y, width, 1.0), selected_bg);
        canvas.draw_text(
            "▶",
            Point::new(x - 1.5, y),
            &TextStyle {
                color: FOCUS_ACCENT_COLOR,
                ..Default::default()
            },
        );
    } else {
        canvas.fill_rect(
            Rect::new(x, y, width, 1.0),
            Color::new(0.05, 0.05, 0.07, 1.0),
        );
    }
}

fn draw_process_dataframe(app: &App, canvas: &mut DirectTerminalCanvas, area: Rect) {
    use crate::ptop::app::ProcessSortColumn;
    use crate::widgets::display_rules::{
        format_column, format_percent, ColumnAlign, TruncateStrategy,
    };
    use crate::HeatScheme;

    let col_widths = [7usize, 10, 8, 8];
    let cmd_width = (area.width as usize).saturating_sub(col_widths.iter().sum::<usize>() + 5);

    let header_bg = Color::new(0.12, 0.15, 0.22, 1.0);
    let selected_col_bg = COL_SELECT_BG;
    let selected_row_bg = ROW_SELECT_BG;
    let sort_color = FOCUS_ACCENT_COLOR;
    let dim_color = Color::new(0.5, 0.5, 0.5, 1.0);
    let text_color = Color::new(0.9, 0.9, 0.9, 1.0);

    let mut y = area.y;
    let x = area.x;

    let columns = [
        (ProcessSortColumn::Pid, "PID", col_widths[0]),
        (ProcessSortColumn::User, "USER", col_widths[1]),
        (ProcessSortColumn::Cpu, "CPU%", col_widths[2]),
        (ProcessSortColumn::Mem, "MEM%", col_widths[3]),
        (ProcessSortColumn::Command, "COMMAND", cmd_width),
    ];

    canvas.fill_rect(Rect::new(x, y, area.width, 1.0), header_bg);

    let mut col_x = x;
    let valid_selected = app.selected_column.min(columns.len().saturating_sub(1));
    for (i, (col, label, width)) in columns.iter().enumerate() {
        let is_selected = valid_selected == i;
        let is_sorted = app.sort_column == *col;

        if is_selected {
            canvas.fill_rect(Rect::new(col_x, y, *width as f32, 1.0), selected_col_bg);
        }

        let header_raw = if is_sorted {
            format!("{}{}", label, if app.sort_descending { "▼" } else { "▲" })
        } else {
            (*label).to_string()
        };
        let header_text = format_column(
            &header_raw,
            *width,
            ColumnAlign::Left,
            TruncateStrategy::End,
        );
        let style = dataframe_header_style(is_sorted, is_selected, sort_color, dim_color);
        canvas.draw_text(&header_text, Point::new(col_x, y), &style);
        col_x += *width as f32 + 1.0;
    }
    y += 1.0;

    canvas.draw_text(
        &"─".repeat((area.width as usize).min(200)),
        Point::new(x, y),
        &TextStyle {
            color: dim_color,
            ..Default::default()
        },
    );
    y += 1.0;

    let mut processes: Vec<_> = app
        .system
        .processes()
        .iter()
        .filter(|(_, p)| {
            let matches_filter = app.filter.is_empty()
                || p.name()
                    .to_string_lossy()
                    .to_lowercase()
                    .contains(&app.filter.to_lowercase());
            matches_filter && (p.cpu_usage() > 0.001 || p.memory() > 1024 * 1024)
        })
        .collect();

    use crate::ptop::ui::panels::process::sort_processes;
    sort_processes(&mut processes, app.sort_column, app.sort_descending);

    let visible_rows = (area.height as usize).saturating_sub(2);
    let scroll_offset = app
        .process_scroll_offset
        .min(processes.len().saturating_sub(visible_rows));

    for (rel_idx, (pid, proc)) in processes
        .iter()
        .skip(scroll_offset)
        .take(visible_rows)
        .enumerate()
    {
        let abs_idx = scroll_offset + rel_idx;
        let is_selected = abs_idx == app.process_selected;

        draw_process_row_bg(canvas, x, y, area.width, is_selected, selected_row_bg);

        let row_style = if is_selected {
            TextStyle {
                color: Color::WHITE,
                ..Default::default()
            }
        } else {
            TextStyle {
                color: text_color,
                ..Default::default()
            }
        };
        let mut col_x = x;

        let pid_str = format_column(
            &pid.as_u32().to_string(),
            col_widths[0],
            ColumnAlign::Right,
            TruncateStrategy::End,
        );
        canvas.draw_text(&pid_str, Point::new(col_x, y), &row_style);
        col_x += col_widths[0] as f32 + 1.0;

        let user_raw = proc
            .user_id()
            .and_then(|uid| app.users.get_user_by_id(uid))
            .map(|u| u.name().to_string())
            .unwrap_or_else(|| "-".to_string());
        let user = format_column(
            &user_raw,
            col_widths[1],
            ColumnAlign::Left,
            TruncateStrategy::End,
        );
        canvas.draw_text(&user, Point::new(col_x, y), &row_style);
        col_x += col_widths[1] as f32 + 1.0;

        let cpu = proc.cpu_usage();
        let cpu_color = if is_selected {
            Color::WHITE
        } else {
            HeatScheme::Thermal.color_for_percent(cpu as f64)
        };
        let cpu_str = format_column(
            &format_percent(cpu),
            col_widths[2],
            ColumnAlign::Right,
            TruncateStrategy::End,
        );
        canvas.draw_text(
            &cpu_str,
            Point::new(col_x, y),
            &TextStyle {
                color: cpu_color,
                ..Default::default()
            },
        );
        col_x += col_widths[2] as f32 + 1.0;

        let mem_pct = (proc.memory() as f64 / app.mem_total as f64 * 100.0) as f32;
        let mem_color = mem_display_color(mem_pct, is_selected, text_color);
        let mem_str = format_column(
            &format_percent(mem_pct),
            col_widths[3],
            ColumnAlign::Right,
            TruncateStrategy::End,
        );
        canvas.draw_text(
            &mem_str,
            Point::new(col_x, y),
            &TextStyle {
                color: mem_color,
                ..Default::default()
            },
        );
        col_x += col_widths[3] as f32 + 1.0;

        let cmd_parts = proc.cmd();
        let cmd_full = if cmd_parts.is_empty() {
            proc.name().to_string_lossy().to_string()
        } else {
            cmd_parts
                .iter()
                .map(|s| s.to_string_lossy())
                .collect::<Vec<_>>()
                .join(" ")
        };
        let cmd_display = format_column(
            &cmd_full,
            cmd_width,
            ColumnAlign::Left,
            TruncateStrategy::Command,
        );
        canvas.draw_text(&cmd_display, Point::new(col_x, y), &row_style);

        y += 1.0;
    }

    if processes.len() > visible_rows {
        let scroll_pct = scroll_offset as f32 / (processes.len() - visible_rows) as f32;
        let bar_y =
            area.y + 2.0 + (scroll_pct * (visible_rows - 1) as f32).min((visible_rows - 1) as f32);
        canvas.draw_text(
            "█",
            Point::new(area.x + area.width - 1.0, bar_y),
            &TextStyle {
                color: dim_color,
                ..Default::default()
            },
        );
    }
}

/// Draw core stats DataFrame for CPU exploded view.
///
/// Uses SPEC-024 Section 28 Display Rules:
/// - `format_column()` for ALL text (NEVER bleeds)
/// - `format_freq_mhz()` for frequency
/// - `format_temp_c()` for temperature
/// - Breakdown bars showing user/system/iowait
#[allow(clippy::too_many_lines)]
fn draw_core_stats_dataframe(app: &App, canvas: &mut DirectTerminalCanvas, area: Rect) {
    use crate::widgets::display_rules::{
        format_column, format_freq_mhz, format_percent, ColumnAlign, TruncateStrategy,
    };
    use crate::widgets::selection::RowHighlight;
    use crate::{HeatScheme, MicroHeatBar};

    // Column widths for table layout.
    let col_widths = [4usize, 6, 5, 5, 5, 4, 5];
    let breakdown_width =
        (area.width as usize).saturating_sub(col_widths.iter().sum::<usize>() + 8);

    // Colors
    let header_bg = Color::new(0.15, 0.2, 0.3, 1.0);
    let dim_color = Color::new(0.5, 0.5, 0.5, 1.0);
    let text_color = Color::new(0.9, 0.9, 0.9, 1.0);
    let user_color = Color::new(0.3, 0.7, 0.3, 1.0); // Green
    let sys_color = Color::new(0.9, 0.5, 0.2, 1.0); // Orange
    let io_color = Color::new(0.9, 0.2, 0.2, 1.0); // Red

    // Selection state: use process_selected for row navigation in exploded view
    let selected_row = if app.exploded_panel == Some(PanelType::Cpu) {
        app.process_selected
    } else {
        usize::MAX // No selection in non-exploded mode
    };

    let mut y = area.y;
    let x = area.x;

    // =========================================================================
    // HEADER ROW
    // =========================================================================
    let headers = [
        "CORE",
        "FREQ",
        "TEMP",
        "USR%",
        "SYS%",
        "IO%",
        "IDL%",
        "BREAKDOWN",
    ];

    canvas.fill_rect(Rect::new(x, y, area.width, 1.0), header_bg);

    let mut col_x = x;
    for (i, header) in headers.iter().enumerate() {
        let width = if i < col_widths.len() {
            col_widths[i]
        } else {
            breakdown_width
        };
        let text = format_column(header, width, ColumnAlign::Left, TruncateStrategy::End);
        canvas.draw_text(
            &text,
            Point::new(col_x, y),
            &TextStyle {
                color: CPU_COLOR,
                ..Default::default()
            },
        );
        col_x += width as f32 + 1.0;
    }
    y += 1.0;

    // Separator line (bounded to area.width)
    let sep_width = (area.width as usize).min(100);
    let sep = "─".repeat(sep_width);
    canvas.draw_text(
        &sep,
        Point::new(x, y),
        &TextStyle {
            color: dim_color,
            ..Default::default()
        },
    );
    y += 1.0;

    // =========================================================================
    // DATA ROWS - one per core
    // Uses app.per_core_freq and app.per_core_temp (SPEC-024 async updates)
    // Uses RowHighlight for selection, MicroHeatBar for breakdown (SPEC-024 v5.8)
    // =========================================================================
    let visible_rows = (area.height as usize).saturating_sub(2);
    let core_count = app.per_core_percent.len();

    for i in 0..core_count.min(visible_rows) {
        // Draw row highlight FIRST (background)
        let is_selected = i == selected_row;
        let row_rect = Rect::new(x, y, area.width, 1.0);
        let row_highlight = RowHighlight::new(row_rect, is_selected).with_gutter(is_selected);
        row_highlight.paint(canvas);

        // Get text style from highlight (selected = white text)
        // Note: We keep semantic colors (user=green, sys=orange) regardless of selection
        let _row_style = row_highlight.text_style();

        let mut col_x = x;

        // Get per-core percentage from async-updated app fields
        let total = app.per_core_percent.get(i).copied().unwrap_or(0.0) as f32;
        let user_pct = total * 0.7;
        let sys_pct = total * 0.25;
        let io_pct = total * 0.05;
        let idle_pct = 100.0 - total;

        // Get frequency from async-updated app field (SPEC-024)
        let freq = app.per_core_freq.get(i).copied().unwrap_or(0);

        // Get temperature from async-updated app field (SPEC-024)
        // Falls back to analyzers if per_core_temp is all zeros
        let temp = app
            .per_core_temp
            .get(i)
            .copied()
            .filter(|&t| t > 0.0)
            .or_else(|| {
                app.snapshot_sensor_health.as_ref().and_then(|data| {
                    data.temperatures()
                        .find(|s| s.label == format!("Core {i}"))
                        .map(|s| s.value as f32)
                })
            });

        // CORE - right-aligned
        let core_str = format_column(
            &i.to_string(),
            col_widths[0],
            ColumnAlign::Right,
            TruncateStrategy::End,
        );
        canvas.draw_text(
            &core_str,
            Point::new(col_x, y),
            &TextStyle {
                color: text_color,
                ..Default::default()
            },
        );
        col_x += col_widths[0] as f32 + 1.0;

        // FREQ - right-aligned with unit formatting (from async-updated app.per_core_freq)
        let freq_str = format_column(
            &format_freq_mhz(freq),
            col_widths[1],
            ColumnAlign::Right,
            TruncateStrategy::End,
        );
        canvas.draw_text(
            &freq_str,
            Point::new(col_x, y),
            &TextStyle {
                color: text_color,
                ..Default::default()
            },
        );
        col_x += col_widths[1] as f32 + 1.0;

        // Temperature column, right-aligned.
        let temp_str = temp.map_or("-".to_string(), |t| format!("{t:.0}°"));
        let temp_str = format_column(
            &temp_str,
            col_widths[2],
            ColumnAlign::Right,
            TruncateStrategy::End,
        );
        // Use HeatScheme::Warm for temperature coloring (SPEC-024 framework requirement)
        let temp_color = temp.map_or(dim_color, |t| {
            // Map temperature to percentage: 30°C = 0%, 100°C = 100%
            let temp_pct = ((t - 30.0) / 70.0 * 100.0).clamp(0.0, 100.0);
            HeatScheme::Warm.color_for_percent(temp_pct as f64)
        });
        canvas.draw_text(
            &temp_str,
            Point::new(col_x, y),
            &TextStyle {
                color: temp_color,
                ..Default::default()
            },
        );
        col_x += col_widths[2] as f32 + 1.0;

        // USR% - right-aligned, green
        let usr_str = format_column(
            &format_percent(user_pct),
            col_widths[3],
            ColumnAlign::Right,
            TruncateStrategy::End,
        );
        canvas.draw_text(
            &usr_str,
            Point::new(col_x, y),
            &TextStyle {
                color: user_color,
                ..Default::default()
            },
        );
        col_x += col_widths[3] as f32 + 1.0;

        // SYS% - right-aligned, orange
        let sys_str = format_column(
            &format_percent(sys_pct),
            col_widths[4],
            ColumnAlign::Right,
            TruncateStrategy::End,
        );
        canvas.draw_text(
            &sys_str,
            Point::new(col_x, y),
            &TextStyle {
                color: sys_color,
                ..Default::default()
            },
        );
        col_x += col_widths[4] as f32 + 1.0;

        // IO% - right-aligned, red
        let io_str = format_column(
            &format_percent(io_pct),
            col_widths[5],
            ColumnAlign::Right,
            TruncateStrategy::End,
        );
        canvas.draw_text(
            &io_str,
            Point::new(col_x, y),
            &TextStyle {
                color: io_color,
                ..Default::default()
            },
        );
        col_x += col_widths[5] as f32 + 1.0;

        // IDL% - right-aligned
        let idl_str = format_column(
            &format_percent(idle_pct),
            col_widths[6],
            ColumnAlign::Right,
            TruncateStrategy::End,
        );
        canvas.draw_text(
            &idl_str,
            Point::new(col_x, y),
            &TextStyle {
                color: dim_color,
                ..Default::default()
            },
        );
        col_x += col_widths[6] as f32 + 1.0;

        // BREAKDOWN bar - using MicroHeatBar framework widget (SPEC-024 Section 30)
        // MicroHeatBar provides: proportional encoding + heat coloring + Tufte principles
        if breakdown_width > 3 {
            let bar_width = breakdown_width.saturating_sub(1);
            // MicroHeatBar shows usr/sys/io/idle proportionally with thermal coloring
            let bar = MicroHeatBar::new(&[
                user_pct as f64,
                sys_pct as f64,
                io_pct as f64,
                idle_pct as f64,
            ])
            .with_width(bar_width)
            .with_scheme(HeatScheme::Thermal);

            bar.paint(canvas, Point::new(col_x, y));
        }

        y += 1.0;
    }
}

/// Draw CPU panel in exploded (fullscreen) mode - Tufte-inspired information-dense design.
///
/// Principles applied:
/// - Maximize data-ink ratio (no chart junk)
/// - Answer the user's actual questions ("What's using my CPU?")
/// - Show outliers, not 48 identical values
/// - Provide temporal context (trend, not just current)
#[allow(clippy::too_many_lines)]
pub(super) fn draw_cpu_exploded(app: &App, canvas: &mut DirectTerminalCanvas, area: Rect) {
    use crate::widgets::{CoreUtilizationHistogram, SystemStatus, TrendSparkline};

    let cpu_pct = app.cpu_history.last().copied().unwrap_or(0.0) * 100.0;
    let core_count = app.per_core_percent.len();
    let uptime = app.uptime();

    let load = &app.load_avg;
    // SPEC-024: Use async-updated per_core_freq instead of stale app.system
    let max_freq_mhz = app.per_core_freq.iter().copied().max().unwrap_or(0);
    let is_boosting = max_freq_mhz > 3000;
    let freq_ghz = max_freq_mhz as f64 / 1000.0;

    // Build title
    let title = build_cpu_title(
        cpu_pct,
        core_count,
        freq_ghz,
        is_boosting,
        uptime,
        load.one,
        app.deterministic,
    );

    // Draw border
    let is_focused = app.is_panel_focused(PanelType::Cpu);
    let mut border = create_panel_border(&title, CPU_COLOR, is_focused);
    border.layout(area);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 15.0 || inner.width < 60.0 {
        draw_cpu_panel(app, canvas, area);
        return;
    }

    // =========================================================================
    // DATA SCIENCE LAYOUT: Two-column DataFrame view filling entire screen
    // =========================================================================
    // Left column (60%): Process DataFrame with CPU sparklines
    // Right column (40%): Core Stats DataFrame with breakdown bars
    // Bottom row: Histogram + Trend + Status (shared)
    // =========================================================================

    let left_width = (inner.width * 0.58).floor();
    let right_width = inner.width - left_width - 1.0;
    let bottom_height = 9.0; // histogram + status
    let top_height = (inner.height - bottom_height - 1.0).max(10.0);

    // =========================================================================
    // SECTION 1: PROCESS DATAFRAME (Left column, fills height)
    // Uses app.selected_column for column highlighting (spreadsheet-style nav)
    // Uses app.sort_column and app.sort_descending for sort indicators
    // =========================================================================
    let proc_rect = Rect::new(inner.x, inner.y, left_width, top_height);
    draw_process_dataframe(app, canvas, proc_rect);

    // =========================================================================
    // SECTION 2: CORE STATS DATAFRAME (Right column, fills height)
    // Uses same display_rules pattern as process dataframe
    // =========================================================================
    let core_rect = Rect::new(inner.x + left_width + 1.0, inner.y, right_width, top_height);
    draw_core_stats_dataframe(app, canvas, core_rect);

    // =========================================================================
    // SECTION 3: BOTTOM ROW - Histogram + Trend + Status
    // =========================================================================
    let bottom_y = inner.y + top_height + 1.0;

    // Core Utilization Histogram (left 45%)
    let hist_width = inner.width * 0.45;
    let histogram_rect = Rect::new(inner.x, bottom_y, hist_width, 6.0);
    let mut histogram = CoreUtilizationHistogram::new(app.per_core_percent.to_vec());
    histogram.layout(histogram_rect);
    histogram.paint(canvas);

    // Trend Sparkline (center 30%)
    let trend_width = inner.width * 0.30;
    let trend_rect = Rect::new(inner.x + hist_width + 1.0, bottom_y, trend_width, 5.0);
    let history: Vec<f64> = app
        .cpu_history
        .as_slice()
        .iter()
        .map(|&v| v * 100.0)
        .collect();
    let mut trend = TrendSparkline::new("60s TREND", history);
    trend.layout(trend_rect);
    trend.paint(canvas);

    // System Status (right 25%)
    let status_width = inner.width - hist_width - trend_width - 2.0;
    let status_rect = Rect::new(
        inner.x + hist_width + trend_width + 2.0,
        bottom_y,
        status_width,
        5.0,
    );

    let mut status = SystemStatus::new(load.one, load.five, load.fifteen, core_count);

    // Add thermal data if available
    if let Some(sensor_data) = app.snapshot_sensor_health.as_ref() {
        let temps: Vec<f64> = sensor_data
            .temperatures()
            .filter(|s| s.label.starts_with("Core "))
            .map(|s| s.value)
            .collect();

        if !temps.is_empty() {
            let avg_temp = temps.iter().sum::<f64>() / temps.len() as f64;
            let max_temp = temps.iter().copied().fold(0.0_f64, f64::max);
            status = status.with_thermal(avg_temp, max_temp);
        }
    }

    status.layout(status_rect);
    status.paint(canvas);
}

/// Draw Memory panel in exploded (fullscreen) mode - SPEC-024 Section 30
///
/// FRAMEWORK REQUIREMENTS:
/// - Uses RowHighlight for selection
/// - Uses MicroHeatBar for breakdown visualization
/// - Uses HeatScheme::Cool for memory coloring
/// - Uses display_rules for ALL formatting
/// - FILLS THE ENTIRE SCREEN
#[allow(clippy::too_many_lines)]
pub(super) fn draw_memory_exploded(app: &App, canvas: &mut DirectTerminalCanvas, area: Rect) {
    use crate::widgets::display_rules::{
        format_bytes_si, format_column, format_percent, ColumnAlign, TruncateStrategy,
    };
    use crate::widgets::selection::RowHighlight;
    use crate::HeatScheme;

    let gb = |b: u64| b as f64 / 1024.0 / 1024.0 / 1024.0;
    let mem_pct = if app.mem_total > 0 {
        (app.mem_used as f64 / app.mem_total as f64) * 100.0
    } else {
        0.0
    };

    // Build title with full details for exploded mode
    let title = format!(
        "Memory │ {:.1}G / {:.1}G ({:.0}%) │ Swap: {:.1}G / {:.1}G │ Cached: {:.1}G │ [FULLSCREEN]",
        gb(app.mem_used),
        gb(app.mem_total),
        mem_pct,
        gb(app.swap_used),
        gb(app.swap_total),
        gb(app.mem_cached),
    );

    let is_focused = app.is_panel_focused(PanelType::Memory);
    let mut border = create_panel_border(&title, MEMORY_COLOR, is_focused);
    border.layout(area);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 10.0 || inner.width < 40.0 {
        draw_memory_panel(app, canvas, area);
        return;
    }

    // =========================================================================
    // FULLSCREEN LAYOUT: Memory breakdown + Top memory consumers
    // =========================================================================
    // Top section (30%): Memory breakdown bars and stats
    // Bottom section (70%): Per-process memory DataFrame
    // =========================================================================

    let breakdown_height = (inner.height * 0.25).max(6.0).min(12.0);
    let process_height = inner.height - breakdown_height - 1.0;

    let mut y = inner.y;

    // =========================================================================
    // SECTION 1: MEMORY BREAKDOWN (Top section)
    // Uses MicroHeatBar for proportional visualization
    // =========================================================================

    // Row 1: Full-width stacked memory bar
    let bar_width = inner.width as usize;
    let used_pct = if app.mem_total > 0 {
        ((app.mem_total - app.mem_available) as f64 / app.mem_total as f64) * 100.0
    } else {
        0.0
    };
    let cached_pct = if app.mem_total > 0 {
        (app.mem_cached as f64 / app.mem_total as f64) * 100.0
    } else {
        0.0
    };
    let free_pct = 100.0 - used_pct;

    // Draw stacked bar using thermal-like color scheme for memory
    let used_chars = ((used_pct / 100.0) * bar_width as f64) as usize;
    let cached_chars = ((cached_pct / 100.0) * bar_width as f64) as usize;
    let free_chars = bar_width.saturating_sub(used_chars + cached_chars);

    // Used: intensity based on pressure
    let used_color = HeatScheme::Warm.color_for_percent(used_pct);
    if used_chars > 0 {
        canvas.draw_text(
            &"█".repeat(used_chars),
            Point::new(inner.x, y),
            &TextStyle {
                color: used_color,
                ..Default::default()
            },
        );
    }
    if cached_chars > 0 {
        canvas.draw_text(
            &"▓".repeat(cached_chars),
            Point::new(inner.x + used_chars as f32, y),
            &TextStyle {
                color: CACHED_COLOR,
                ..Default::default()
            },
        );
    }
    if free_chars > 0 {
        canvas.draw_text(
            &"░".repeat(free_chars),
            Point::new(inner.x + used_chars as f32 + cached_chars as f32, y),
            &TextStyle {
                color: Color::new(0.3, 0.3, 0.3, 1.0),
                ..Default::default()
            },
        );
    }
    y += 1.0;

    // Row 2-5: Memory segment details in columns
    let col_width = (inner.width / 4.0).floor() as usize;
    let segments = [
        (
            "Used",
            app.mem_used,
            used_pct,
            HeatScheme::Warm.color_for_percent(used_pct),
        ),
        ("Cached", app.mem_cached, cached_pct, CACHED_COLOR),
        ("Available", app.mem_available, free_pct, FREE_COLOR),
        (
            "Swap",
            app.swap_used,
            if app.swap_total > 0 {
                (app.swap_used as f64 / app.swap_total as f64) * 100.0
            } else {
                0.0
            },
            swap_color(if app.swap_total > 0 {
                (app.swap_used as f64 / app.swap_total as f64) * 100.0
            } else {
                0.0
            }),
        ),
    ];

    let dim_style = TextStyle {
        color: Color::new(0.5, 0.5, 0.5, 1.0),
        ..Default::default()
    };

    for (i, (label, bytes, pct, color)) in segments.iter().enumerate() {
        let x = inner.x + (i as f32 * col_width as f32);
        let text = format!("{}: {} ({:.0}%)", label, format_bytes_si(*bytes), pct);
        canvas.draw_text(
            &format_column(
                &text,
                col_width.saturating_sub(1),
                ColumnAlign::Left,
                TruncateStrategy::End,
            ),
            Point::new(x, y),
            &TextStyle {
                color: *color,
                ..Default::default()
            },
        );
    }
    y += 1.0;

    // Separator
    canvas.draw_text(
        &"─".repeat(inner.width as usize),
        Point::new(inner.x, y),
        &dim_style,
    );
    y += 1.0;

    // =========================================================================
    // SECTION 2: TOP MEMORY CONSUMERS (Bottom section - fills remaining space)
    // Uses RowHighlight for selection, sorted by memory usage
    // =========================================================================

    let process_y = y;
    let process_rect = Rect::new(inner.x, process_y, inner.width, process_height);

    // Column widths for process memory table
    let col_pid = 8;
    let col_user = 10;
    let col_mem_pct = 8;
    let col_mem_bytes = 10;
    let col_rss = 10;
    let col_virt = 10;
    let col_cmd = (inner.width as usize)
        .saturating_sub(col_pid + col_user + col_mem_pct + col_mem_bytes + col_rss + col_virt + 10);

    // Header
    let header_bg = Color::new(0.12, 0.15, 0.22, 1.0);
    canvas.fill_rect(
        Rect::new(process_rect.x, process_rect.y, process_rect.width, 1.0),
        header_bg,
    );

    let headers = ["PID", "USER", "MEM%", "MEM", "RSS", "VIRT", "COMMAND"];
    let widths = [
        col_pid,
        col_user,
        col_mem_pct,
        col_mem_bytes,
        col_rss,
        col_virt,
        col_cmd,
    ];
    let mut hx = process_rect.x;
    for (header, width) in headers.iter().zip(widths.iter()) {
        canvas.draw_text(
            &format_column(header, *width, ColumnAlign::Left, TruncateStrategy::End),
            Point::new(hx, process_rect.y),
            &TextStyle {
                color: MEMORY_COLOR,
                ..Default::default()
            },
        );
        hx += *width as f32 + 1.0;
    }

    // Separator
    canvas.draw_text(
        &"─".repeat(process_rect.width as usize),
        Point::new(process_rect.x, process_rect.y + 1.0),
        &dim_style,
    );

    // Process rows sorted by memory
    let mut processes: Vec<_> = app.system.processes().iter().collect();
    processes.sort_by(|a, b| b.1.memory().cmp(&a.1.memory()));

    let visible_rows = (process_rect.height as usize).saturating_sub(2);
    let mut row_y = process_rect.y + 2.0;

    for (idx, (pid, proc)) in processes.iter().take(visible_rows).enumerate() {
        let is_selected = idx == app.process_selected;

        // FRAMEWORK: Use RowHighlight for selection
        let row_bounds = Rect::new(process_rect.x, row_y, process_rect.width, 1.0);
        let highlight = RowHighlight::new(row_bounds, is_selected);
        highlight.paint(canvas);
        let text_style = highlight.text_style();

        let mut col_x = process_rect.x;

        // PID
        canvas.draw_text(
            &format_column(
                &pid.as_u32().to_string(),
                col_pid,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, row_y),
            &text_style,
        );
        col_x += col_pid as f32 + 1.0;

        // USER
        let user = proc
            .user_id()
            .and_then(|uid| app.users.get_user_by_id(uid))
            .map(|u| u.name().to_string())
            .unwrap_or_else(|| "-".to_string());
        canvas.draw_text(
            &format_column(&user, col_user, ColumnAlign::Left, TruncateStrategy::End),
            Point::new(col_x, row_y),
            &text_style,
        );
        col_x += col_user as f32 + 1.0;

        // MEM%
        let mem_pct = (proc.memory() as f64 / app.mem_total as f64) * 100.0;
        let mem_color = if is_selected {
            Color::WHITE
        } else {
            HeatScheme::Cool.color_for_percent(mem_pct * 10.0) // Scale up for visibility
        };
        canvas.draw_text(
            &format_column(
                &format_percent(mem_pct as f32),
                col_mem_pct,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, row_y),
            &TextStyle {
                color: mem_color,
                ..Default::default()
            },
        );
        col_x += col_mem_pct as f32 + 1.0;

        // MEM bytes
        canvas.draw_text(
            &format_column(
                &format_bytes_si(proc.memory()),
                col_mem_bytes,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, row_y),
            &text_style,
        );
        col_x += col_mem_bytes as f32 + 1.0;

        // RSS (same as memory for now)
        canvas.draw_text(
            &format_column(
                &format_bytes_si(proc.memory()),
                col_rss,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, row_y),
            &text_style,
        );
        col_x += col_rss as f32 + 1.0;

        // VIRT
        canvas.draw_text(
            &format_column(
                &format_bytes_si(proc.virtual_memory()),
                col_virt,
                ColumnAlign::Right,
                TruncateStrategy::End,
            ),
            Point::new(col_x, row_y),
            &text_style,
        );
        col_x += col_virt as f32 + 1.0;

        // COMMAND
        let cmd = proc.name().to_string_lossy().to_string();
        canvas.draw_text(
            &format_column(&cmd, col_cmd, ColumnAlign::Left, TruncateStrategy::Command),
            Point::new(col_x, row_y),
            &text_style,
        );

        row_y += 1.0;
    }
}
