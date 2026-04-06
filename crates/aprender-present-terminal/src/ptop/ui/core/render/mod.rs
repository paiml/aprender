//! UI layout and rendering for ptop.
//!
//! Pixel-perfect ttop clone using presentar-terminal widgets.

// Allow style-only clippy warnings that don't affect correctness
#![allow(clippy::too_many_lines)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::option_map_or_none)]

use crate::direct::{CellBuffer, DirectTerminalCanvas};
use crate::{
    Border, BorderStyle, BrailleGraph, CpuGrid, GraphMode, NetworkInterface, NetworkPanel,
    ProcessEntry, ProcessState, ProcessTable, TitleBar, Treemap, TreemapNode,
};
use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};

use crate::ptop::analyzers::{ContainerState, SensorStatus, SensorType, TcpState};
use crate::ptop::app::{App, ProcessSortColumn};
use crate::ptop::config::{calculate_grid_layout, snap_to_grid, DetailLevel, PanelType};
use crate::ptop::ui::core::layout::push_if_visible;
use crate::ptop::ui::core::panel_cpu::{
    build_cpu_title, build_cpu_title_compact, build_load_bar, consumer_cpu_color, load_color,
    load_trend_arrow, CpuMeterLayout, DIM_LABEL_COLOR, PROCESS_NAME_COLOR,
};
#[allow(unused_imports)]
use crate::ptop::ui::core::panel_gpu::{
    build_gpu_bar, build_gpu_title, format_proc_util, gpu_proc_badge, gpu_temp_color,
    truncate_name, HEADER_COLOR, POWER_COLOR, PROC_INFO_COLOR, VRAM_GRAPH_COLOR,
};
use crate::ptop::ui::core::panel_memory::{
    has_swap_activity, psi_memory_indicator, swap_color, thrashing_indicator,
    MemoryStats as MemStats, ZramDisplay, CACHED_COLOR, DIM_COLOR, FREE_COLOR, RATIO_COLOR,
    ZRAM_COLOR,
};
use crate::ptop::ui::panels::connections::{
    build_sparkline, ACTIVE_COLOR, DIM_COLOR as CONN_DIM_COLOR, LISTEN_COLOR,
};
// Atomic widget helpers (available for incremental adoption)
#[allow(unused_imports)]
use crate::ptop::ui_atoms::{draw_colored_text, severity_color, usage_color};

// ── Shared helpers (PMAT-016: DataTransformation, PMAT-017: ControlFlow) ──

/// Build a progress bar string of `width` characters using █ and ░.
/// `ratio` is 0.0..=1.0 (percentage / 100).
fn make_bar(ratio: f64, width: usize) -> String {
    let filled = ((ratio * width as f64) as usize).min(width);
    "█".repeat(filled) + &"░".repeat(width.saturating_sub(filled))
}

/// Safe percentage: returns `(numerator / denominator) * 100`, or 0 if denominator is 0.
fn safe_pct(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        (numerator as f64 / denominator as f64) * 100.0
    }
}

/// Returns true when a row at `y` still fits inside `inner`.
fn can_draw_row(y: f32, inner: &Rect) -> bool {
    y < inner.y + inner.height
}

/// Returns true when a panel is too small to render.
fn panel_too_small(inner: &Rect, min_h: f32, min_w: f32) -> bool {
    inner.height < min_h || inner.width < min_w
}

// ttop panel border colors (exact RGB values from theme.rs)
const CPU_COLOR: Color = Color {
    r: 0.392,
    g: 0.784,
    b: 1.0,
    a: 1.0,
}; // #64C8FF (100,200,255)
const MEMORY_COLOR: Color = Color {
    r: 0.706,
    g: 0.471,
    b: 1.0,
    a: 1.0,
}; // #B478FF (180,120,255)
const DISK_COLOR: Color = Color {
    r: 0.392,
    g: 0.706,
    b: 1.0,
    a: 1.0,
}; // #64B4FF (100,180,255)
const NETWORK_COLOR: Color = Color {
    r: 1.0,
    g: 0.588,
    b: 0.392,
    a: 1.0,
}; // #FF9664 (255,150,100)
const PROCESS_COLOR: Color = Color {
    r: 0.863,
    g: 0.706,
    b: 0.392,
    a: 1.0,
}; // #DCC464 (220,180,100)
const GPU_COLOR: Color = Color {
    r: 0.392,
    g: 1.0,
    b: 0.588,
    a: 1.0,
}; // #64FF96 (100,255,150)
const BATTERY_COLOR: Color = Color {
    r: 1.0,
    g: 0.863,
    b: 0.392,
    a: 1.0,
}; // #FFDC64 (255,220,100)
const SENSORS_COLOR: Color = Color {
    r: 1.0,
    g: 0.392,
    b: 0.588,
    a: 1.0,
}; // #FF6496 (255,100,150)
const PSI_COLOR: Color = Color {
    r: 0.784,
    g: 0.314,
    b: 0.314,
    a: 1.0,
}; // #C85050 (200,80,80)
const CONNECTIONS_COLOR: Color = Color {
    r: 0.471,
    g: 0.706,
    b: 0.863,
    a: 1.0,
}; // #78B4DC (120,180,220)
const FILES_COLOR: Color = Color {
    r: 0.706,
    g: 0.549,
    b: 0.392,
    a: 1.0,
}; // #B48C64 (180,140,100)
const CONTAINERS_COLOR: Color = Color {
    r: 0.392,
    g: 0.706,
    b: 0.863,
    a: 1.0,
}; // #64B4DC (100,180,220) - Docker blue

// Network graph colors (distinct from panel border)
const NET_RX_COLOR: Color = Color {
    r: 0.392,
    g: 0.784,
    b: 1.0,
    a: 1.0,
}; // Cyan (download)
const NET_TX_COLOR: Color = Color {
    r: 1.0,
    g: 0.392,
    b: 0.392,
    a: 1.0,
}; // Red (upload)

// =============================================================================
// TUFTE-INSPIRED SELECTION: Use framework widgets from widgets/selection.rs
// =============================================================================
use crate::widgets::selection::{SELECTION_ACCENT, SELECTION_BG};

// Re-export for local use (framework-first design)
const FOCUS_ACCENT_COLOR: Color = SELECTION_ACCENT;
const ROW_SELECT_BG: Color = SELECTION_BG;

/// Column header selection background (slightly different from row)
const COL_SELECT_BG: Color = Color {
    r: 0.15,
    g: 0.4,
    b: 0.65,
    a: 1.0,
};

/// Status bar background
const STATUS_BAR_BG: Color = Color {
    r: 0.08,
    g: 0.08,
    b: 0.12,
    a: 1.0,
};

/// btop-style color gradient for percentage values (0-100)
/// Uses smooth transition: cyan -> green -> yellow -> orange -> red
fn percent_color(percent: f64) -> Color {
    let p = percent.clamp(0.0, 100.0);

    if p >= 90.0 {
        // Critical: bright red
        Color {
            r: 1.0,
            g: 0.25,
            b: 0.25,
            a: 1.0,
        }
    } else if p >= 75.0 {
        // High: orange-red gradient
        let t = (p - 75.0) / 15.0;
        Color {
            r: 1.0,
            g: (0.706 - t * 0.456) as f32,
            b: 0.25,
            a: 1.0,
        }
    } else if p >= 50.0 {
        // Medium-high: yellow to orange
        let t = (p - 50.0) / 25.0;
        Color {
            r: 1.0,
            g: (0.863 - t * 0.157) as f32,
            b: 0.25,
            a: 1.0,
        }
    } else if p >= 25.0 {
        // Medium-low: green to yellow
        let t = (p - 25.0) / 25.0;
        Color {
            r: (0.392 + t * 0.608) as f32,
            g: 0.863,
            b: (0.392 - t * 0.142) as f32,
            a: 1.0,
        }
    } else {
        // Low: cyan to green
        let t = p / 25.0;
        Color {
            r: (0.25 + t * 0.142) as f32,
            g: (0.706 + t * 0.157) as f32,
            b: (0.863 - t * 0.471) as f32,
            a: 1.0,
        }
    }
}

/// Format bytes to human-readable string
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.1}T", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.1}G", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}M", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}K", bytes as f64 / KB as f64)
    } else {
        format!("{bytes}B")
    }
}

/// Format bytes per second rate (for disk/network I/O)
fn format_bytes_rate(bytes_per_sec: f64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    if bytes_per_sec >= GB {
        format!("{:.1}G", bytes_per_sec / GB)
    } else if bytes_per_sec >= MB {
        format!("{:.1}M", bytes_per_sec / MB)
    } else if bytes_per_sec >= KB {
        format!("{:.0}K", bytes_per_sec / KB)
    } else {
        format!("{bytes_per_sec:.0}B")
    }
}

// Note: format_uptime moved to format.rs

/// Create a border with PATTERN 5 HYBRID focus indication
/// Focus indicators (WCAG AAA compliant):
/// 1. Double-line border (vs rounded for unfocused) - works in monochrome
/// 2. Bright cyan accent color for focused - high visibility
/// 3. Focus indicator arrow `►` prepended to title
/// 4. Unfocused panels are dimmed for contrast
fn create_panel_border(title: &str, color: Color, is_focused: bool) -> Border {
    let style = if is_focused {
        BorderStyle::Double // Double border for focused panel (Pattern 1)
    } else {
        BorderStyle::Rounded // Normal rounded border
    };

    // PATTERN 5: Use accent color for focused, dim for unfocused
    let border_color = if is_focused {
        // Blend panel color with cyan accent for focused state
        Color {
            r: (color.r * 0.5 + FOCUS_ACCENT_COLOR.r * 0.5).min(1.0),
            g: (color.g * 0.5 + FOCUS_ACCENT_COLOR.g * 0.5).min(1.0),
            b: (color.b * 0.5 + FOCUS_ACCENT_COLOR.b * 0.5).min(1.0),
            a: color.a,
        }
    } else {
        // Dim unfocused panels significantly for contrast
        Color {
            r: color.r * 0.5,
            g: color.g * 0.5,
            b: color.b * 0.5,
            a: color.a,
        }
    };

    // Add focus indicator to title
    let display_title = if is_focused {
        format!("► {title}")
    } else {
        title.to_string()
    };

    Border::new()
        .with_title(&display_title)
        .with_style(style)
        .with_color(border_color)
        .with_title_left_aligned()
}

/// FRAMEWORK-LEVEL: Paint a panel with automatic clipping
///
/// This function:
/// 1. Creates and paints the border
/// 2. Pushes a clip region for the inner content area
/// 3. Calls the content painter closure
/// 4. Pops the clip region
///
/// All panel content is automatically constrained to the inner bounds,
/// preventing text overflow into adjacent panels.
#[allow(dead_code)] // Framework function for future use
fn paint_panel_with_clip<F>(
    canvas: &mut DirectTerminalCanvas<'_>,
    title: &str,
    color: Color,
    is_focused: bool,
    bounds: Rect,
    content_painter: F,
) where
    F: FnOnce(&mut DirectTerminalCanvas<'_>, Rect),
{
    let mut border = create_panel_border(title, color, is_focused);
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 || inner.width < 1.0 {
        return;
    }

    // Push clip to constrain content to panel bounds
    canvas.push_clip(inner);
    content_painter(canvas, inner);
    canvas.pop_clip();
}

/// ZRAM statistics from /sys/block/zram*
#[derive(Debug, Default)]
struct ZramStats {
    /// Original (uncompressed) data size in bytes
    orig_data_size: u64,
    /// Compressed data size in bytes
    compr_data_size: u64,
    /// Compression algorithm (lzo, lz4, zstd, etc.)
    algorithm: String,
}

impl ZramStats {
    /// Get compression ratio (original / compressed)
    fn ratio(&self) -> f64 {
        if self.compr_data_size == 0 {
            1.0
        } else {
            self.orig_data_size as f64 / self.compr_data_size as f64
        }
    }

    /// Check if ZRAM is active
    fn is_active(&self) -> bool {
        self.orig_data_size > 0
    }
}

/// Parse compression algorithm from comp_algorithm file content.
/// Format: "lzo lzo-rle [lz4] zstd" - bracketed = active
#[cfg(target_os = "linux")]
fn parse_zram_algorithm(content: &str) -> String {
    content
        .split_whitespace()
        .find(|p| p.starts_with('[') && p.ends_with(']'))
        .map(|p| p.trim_matches(|c| c == '[' || c == ']').to_string())
        .unwrap_or_else(|| "?".to_string())
}

/// Read ZRAM stats from a specific device path.
#[cfg(target_os = "linux")]
fn read_zram_device(base_path: &str) -> Option<ZramStats> {
    use std::fs;

    let mm_stat_path = format!("{base_path}/mm_stat");
    let content = fs::read_to_string(&mm_stat_path).ok()?;
    let parts: Vec<&str> = content.split_whitespace().collect();

    if parts.len() < 2 {
        return None;
    }

    let orig = parts[0].parse::<u64>().unwrap_or(0);
    let compr = parts[1].parse::<u64>().unwrap_or(0);

    if orig == 0 {
        return None;
    }

    let algo_path = format!("{base_path}/comp_algorithm");
    let algorithm = fs::read_to_string(&algo_path)
        .map(|s| parse_zram_algorithm(&s))
        .unwrap_or_else(|_| "?".to_string());

    Some(ZramStats {
        orig_data_size: orig,
        compr_data_size: compr,
        algorithm,
    })
}

/// Read ZRAM statistics from /sys/block/zram* (Linux only)
fn read_zram_stats() -> Option<ZramStats> {
    #[cfg(target_os = "linux")]
    {
        for i in 0..4 {
            let base_path = format!("/sys/block/zram{i}");
            if !std::path::Path::new(&base_path).exists() {
                continue;
            }
            if let Some(stats) = read_zram_device(&base_path) {
                return Some(stats);
            }
        }
        None
    }

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Main draw function - called each frame
/// Get keybinds based on current mode.
fn get_keybinds(exploded: bool) -> &'static [(&'static str, &'static str)] {
    if exploded {
        &[
            ("←→", "Column"),
            ("↵", "Sort"),
            ("↑↓", "Row"),
            ("Esc", "Exit"),
        ]
    } else {
        &[
            ("q", "Quit"),
            ("?", "Help"),
            ("/", "Filter"),
            ("Tab", "Nav"),
        ]
    }
}

/// Draw title bar with app name and search.
fn draw_title_bar(app: &App, canvas: &mut DirectTerminalCanvas<'_>, w: f32) {
    let keybinds = get_keybinds(app.exploded_panel.is_some());
    let mut title_bar = TitleBar::new("ptop")
        .with_version(env!("CARGO_PKG_VERSION"))
        .with_search_placeholder("Filter processes...")
        .with_search_text(&app.filter)
        .with_search_active(app.show_filter_input)
        .with_keybinds(keybinds)
        .with_primary_color(CPU_COLOR);
    if app.exploded_panel.is_some() {
        title_bar = title_bar.with_mode_indicator("[▣]");
    }
    title_bar.layout(Rect::new(0.0, 0.0, w, 1.0));
    title_bar.paint(canvas);
}

/// Compute layout heights for top/bottom panels.
fn compute_panel_layout(content_h: f32, top_count: u32, has_process: bool) -> (f32, f32) {
    let top_h = if top_count > 0 && has_process {
        (content_h * 0.45).max(8.0)
    } else if top_count > 0 {
        content_h
    } else {
        0.0
    };
    (top_h, content_h - top_h)
}

/// Draw bottom row panels (process, connections, files/treemap).
fn draw_bottom_row(
    app: &App,
    canvas: &mut DirectTerminalCanvas<'_>,
    bottom_y: f32,
    bottom_h: f32,
    w: f32,
) {
    if !app.panels.process || bottom_h <= 3.0 {
        return;
    }
    let proc_w = (w * 0.4).round();
    let remaining = w - proc_w;
    let conn_w = (remaining / 2.0).floor();
    let files_w = remaining - conn_w;
    draw_process_panel(app, canvas, Rect::new(0.0, bottom_y, proc_w, bottom_h));
    if app.panels.connections {
        draw_connections_panel(app, canvas, Rect::new(proc_w, bottom_y, conn_w, bottom_h));
    }
    if app.panels.files {
        draw_files_panel(
            app,
            canvas,
            Rect::new(proc_w + conn_w, bottom_y, files_w, bottom_h),
        );
    } else if app.panels.treemap {
        draw_treemap_panel(
            app,
            canvas,
            Rect::new(proc_w + conn_w, bottom_y, files_w, bottom_h),
        );
    }
}

/// Draw overlay dialogs (help, signal, filter, fps).
fn draw_overlays(app: &App, canvas: &mut DirectTerminalCanvas<'_>, w: f32, h: f32) {
    if app.show_help {
        draw_help_overlay(canvas, w, h);
    }
    if app.pending_signal.is_some() {
        draw_signal_dialog(app, canvas, w, h);
    }
    if app.show_filter_input {
        draw_filter_overlay(app, canvas, w, h);
    }
    if app.show_fps {
        draw_fps_overlay(app, canvas, w);
    }
}

pub fn draw(app: &App, buffer: &mut CellBuffer) {
    let w = buffer.width() as f32;
    let h = buffer.height() as f32;
    if w < 10.0 || h < 5.0 {
        return;
    }

    let mut canvas = DirectTerminalCanvas::new(buffer);
    draw_title_bar(app, &mut canvas, w);

    let content_y = 1.0_f32;
    let content_h = h - 2.0; // 1 title + 1 status

    if let Some(panel) = app.exploded_panel {
        draw_exploded_panel(
            app,
            &mut canvas,
            Rect::new(0.0, content_y, w, content_h),
            panel,
        );
        draw_status_bar(app, &mut canvas, w, h);
        return;
    }

    let top_count = count_top_panels(app);
    let (top_h, bottom_h) = compute_panel_layout(content_h, top_count, app.panels.process);

    if top_count > 0 {
        draw_top_panels(app, &mut canvas, Rect::new(0.0, content_y, w, top_h));
    }
    draw_bottom_row(app, &mut canvas, content_y + top_h, bottom_h, w);
    draw_overlays(app, &mut canvas, w, h);
    draw_status_bar(app, &mut canvas, w, h);
}

fn count_top_panels(app: &App) -> u32 {
    let mut count = 0;
    if app.panels.cpu {
        count += 1;
    }
    if app.panels.memory {
        count += 1;
    }
    if app.panels.disk {
        count += 1;
    }
    if app.panels.network {
        count += 1;
    }
    if app.panels.gpu {
        count += 1;
    }
    if app.panels.battery {
        count += 1;
    }
    if app.panels.sensors {
        count += 1;
    }
    if app.panels.psi {
        count += 1;
    }
    count
}

fn draw_fps_overlay(app: &App, canvas: &mut DirectTerminalCanvas<'_>, w: f32) {
    let fps_str = format!(" Frame: {}μs ", app.avg_frame_time_us);
    let style = TextStyle {
        color: Color::new(0.4, 1.0, 0.4, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        &fps_str,
        Point::new(w - fps_str.len() as f32 - 1.0, 0.0),
        &style,
    );
}

/// Status bar with navigation hints (SPEC-024 v5.5 Section 11.6)
/// Shows: [Tab] Navigate [Enter] Explode [?] Help [q] Quit | Focused: CPU
fn draw_status_bar(app: &App, canvas: &mut DirectTerminalCanvas<'_>, w: f32, h: f32) {
    let y = h - 1.0;

    // PATTERN 5 HYBRID: Status bar colors
    let bracket_style = TextStyle {
        color: Color::new(0.5, 0.5, 0.5, 1.0), // Brackets
        ..Default::default()
    };
    let key_style = TextStyle {
        color: FOCUS_ACCENT_COLOR, // Bright cyan for keys
        ..Default::default()
    };
    let action_style = TextStyle {
        color: Color::new(0.7, 0.7, 0.7, 1.0), // Action text
        ..Default::default()
    };
    let focus_indicator_style = TextStyle {
        color: FOCUS_ACCENT_COLOR, // Bright cyan for focus indicator
        ..Default::default()
    };

    // Draw background bar
    canvas.fill_rect(Rect::new(0.0, y, w, 1.0), STATUS_BAR_BG);

    // Navigation hints - different for exploded vs normal view
    let hints = if app.exploded_panel.is_some() {
        " [Esc]Exit  [↑↓]Row  [←→]Col  [?]Help  [q]Quit "
    } else {
        " [Tab]Panel  [Enter]Explode  [↑↓]Row  [/]Filter  [?]Help  [q]Quit "
    };

    // Draw hints with bracket highlighting
    let mut x = 0.0;
    let mut in_bracket = false;
    for ch in hints.chars() {
        let style = if ch == '[' {
            in_bracket = true;
            &bracket_style
        } else if ch == ']' {
            in_bracket = false;
            &bracket_style
        } else if in_bracket {
            &key_style // Key inside brackets - bright cyan
        } else {
            &action_style
        };
        canvas.draw_text(&ch.to_string(), Point::new(x, y), style);
        x += 1.0;
    }

    // PATTERN 5: Prominent focused panel indicator on right with ► cursor
    if let Some(panel) = app.focused_panel {
        let panel_name = match panel {
            PanelType::Cpu => "CPU",
            PanelType::Memory => "Memory",
            PanelType::Disk => "Disk",
            PanelType::Network => "Network",
            PanelType::Process => "Process",
            PanelType::Gpu => "GPU",
            PanelType::Battery => "Battery",
            PanelType::Sensors => "Sensors",
            PanelType::Files => "Files",
            PanelType::Connections => "Connections",
            PanelType::Psi => "PSI",
            PanelType::Containers => "Containers",
        };
        // Prominent focus indicator: "► CPU"
        let focus_text = format!("► {panel_name} ");
        let focus_x = w - focus_text.chars().count() as f32 - 1.0;
        if focus_x > x {
            canvas.draw_text(&focus_text, Point::new(focus_x, y), &focus_indicator_style);
        }
    }
}

/// Check if app is configured for ttop-style 6-panel layout.
fn is_ttop_layout(app: &App) -> bool {
    app.panels.cpu
        && app.panels.memory
        && app.panels.disk
        && app.panels.network
        && app.panels.gpu
        && app.panels.sensors
        && !app.panels.psi
        && !app.panels.battery
}

/// Draw ttop-specific 3x2 grid layout with stacked third column.
fn draw_ttop_grid(app: &App, canvas: &mut DirectTerminalCanvas<'_>, area: Rect) {
    let cell_w = area.width / 3.0;
    let cell_h = area.height / 2.0;

    // Row 0: CPU, Memory, Disk
    draw_cpu_panel(app, canvas, Rect::new(area.x, area.y, cell_w, cell_h));
    draw_memory_panel(
        app,
        canvas,
        Rect::new(area.x + cell_w, area.y, cell_w, cell_h),
    );
    draw_disk_panel(
        app,
        canvas,
        Rect::new(area.x + 2.0 * cell_w, area.y, cell_w, cell_h),
    );

    // Row 1: Network, GPU, Sensors+Containers stacked
    let row1_y = area.y + cell_h;
    draw_network_panel(app, canvas, Rect::new(area.x, row1_y, cell_w, cell_h));
    draw_gpu_panel(
        app,
        canvas,
        Rect::new(area.x + cell_w, row1_y, cell_w, cell_h),
    );

    // Third column: Sensors (33%) + Containers (67%)
    let col3_x = area.x + 2.0 * cell_w;
    let sensors_h = (cell_h / 3.0).round();
    draw_sensors_panel(app, canvas, Rect::new(col3_x, row1_y, cell_w, sensors_h));
    draw_containers_panel(
        app,
        canvas,
        Rect::new(col3_x, row1_y + sensors_h, cell_w, cell_h - sensors_h),
    );
}

/// Build list of panel draw functions based on app configuration.
#[allow(clippy::type_complexity)]
fn build_panel_list(app: &App) -> Vec<fn(&App, &mut DirectTerminalCanvas<'_>, Rect)> {
    let mut panels: Vec<fn(&App, &mut DirectTerminalCanvas<'_>, Rect)> = Vec::new();

    if app.panels.cpu {
        panels.push(draw_cpu_panel);
    }
    if app.panels.memory {
        panels.push(draw_memory_panel);
    }
    if app.panels.disk {
        panels.push(draw_disk_panel);
    }
    if app.panels.network {
        panels.push(draw_network_panel);
    }

    push_if_visible(
        &mut panels,
        app,
        app.panels.gpu,
        PanelType::Gpu,
        draw_gpu_panel,
        None,
    );
    push_if_visible(
        &mut panels,
        app,
        app.panels.sensors,
        PanelType::Sensors,
        draw_sensors_panel,
        Some(draw_sensors_compact_panel),
    );
    push_if_visible(
        &mut panels,
        app,
        app.panels.psi,
        PanelType::Psi,
        draw_psi_panel,
        None,
    );
    push_if_visible(
        &mut panels,
        app,
        app.panels.battery,
        PanelType::Battery,
        draw_battery_panel,
        None,
    );

    if app.panels.sensors_compact {
        panels.push(draw_sensors_compact_panel);
    }
    if app.panels.system {
        panels.push(draw_system_panel);
    }

    panels
}

fn draw_top_panels(app: &App, canvas: &mut DirectTerminalCanvas<'_>, area: Rect) {
    if is_ttop_layout(app) && area.width >= 100.0 {
        draw_ttop_grid(app, canvas, area);
        return;
    }

    let panels = build_panel_list(app);
    if panels.is_empty() {
        return;
    }

    let layout_config = &app.config.layout;
    let grid_rects = calculate_grid_layout(
        panels.len() as u32,
        area.width as u16,
        area.height as u16,
        layout_config,
    );

    for (i, draw_fn) in panels.iter().enumerate() {
        if let Some(rect) = grid_rects.get(i) {
            let snapped_x = snap_to_grid(rect.x, layout_config.grid_size);
            let snapped_y = snap_to_grid(rect.y, layout_config.grid_size);
            let bounds = Rect::new(
                area.x + snapped_x as f32,
                area.y + snapped_y as f32,
                rect.width as f32,
                rect.height as f32,
            );
            draw_fn(app, canvas, bounds);
        }
    }
}

// ============================================================================
// CPU Panel - uses extracted helpers from panel_cpu module
// ============================================================================

mod exploded_core;
mod exploded_extra;
/// Get CPU load average and max frequency.
mod panels_core;
mod panels_extra;

use exploded_core::*;
use exploded_extra::*;
use panels_core::*;
use panels_extra::*;

// Re-export types needed by app.rs
pub use panels_extra::{read_gpu_info, GpuInfo};

/// Draw a single panel in fullscreen (exploded) mode
fn draw_exploded_panel(app: &App, canvas: &mut DirectTerminalCanvas, area: Rect, panel: PanelType) {
    match panel {
        PanelType::Cpu => draw_cpu_exploded(app, canvas, area),
        PanelType::Memory => draw_memory_exploded(app, canvas, area),
        PanelType::Disk => draw_disk_exploded(app, canvas, area),
        PanelType::Network => draw_network_exploded(app, canvas, area),
        PanelType::Process => draw_process_exploded(app, canvas, area),
        PanelType::Gpu => draw_gpu_exploded(app, canvas, area),
        PanelType::Sensors => draw_sensors_exploded(app, canvas, area),
        PanelType::Connections => draw_connections_exploded(app, canvas, area),
        PanelType::Psi => draw_psi_panel(app, canvas, area),
        PanelType::Files => draw_files_panel(app, canvas, area),
        PanelType::Battery => draw_battery_panel(app, canvas, area),
        PanelType::Containers => draw_containers_panel(app, canvas, area),
    }
}

/// Get the border color for a panel type
pub fn panel_border_color(panel: PanelType) -> Color {
    match panel {
        PanelType::Cpu => CPU_COLOR,
        PanelType::Memory => MEMORY_COLOR,
        PanelType::Disk => DISK_COLOR,
        PanelType::Network => NETWORK_COLOR,
        PanelType::Process => PROCESS_COLOR,
        PanelType::Gpu => GPU_COLOR,
        PanelType::Battery => BATTERY_COLOR,
        PanelType::Sensors => SENSORS_COLOR,
        PanelType::Psi => PSI_COLOR,
        PanelType::Connections => CONNECTIONS_COLOR,
        PanelType::Files => FILES_COLOR,
        PanelType::Containers => CONTAINERS_COLOR,
    }
}
