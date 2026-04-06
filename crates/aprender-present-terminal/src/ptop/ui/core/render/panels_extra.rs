use super::*;

// ============================================================================
// NEW PANELS (F006-F014): GPU, Battery, Sensors, PSI, Connections, etc.
// ============================================================================

/// GPU information from sysfs or nvidia-smi
/// GPU information structure used by both app.rs and ui.rs
#[derive(Debug, Default, Clone)]
pub struct GpuInfo {
    /// GPU name/model
    pub name: String,
    /// GPU utilization (0-100)
    pub utilization: Option<u8>,
    /// Temperature in Celsius
    pub temperature: Option<u32>,
    /// Power consumption in Watts
    pub power_watts: Option<f32>,
    /// VRAM used in bytes
    pub vram_used: Option<u64>,
    /// VRAM total in bytes
    pub vram_total: Option<u64>,
}

/// Read GPU info from nvidia-smi (NVIDIA) or sysfs (AMD/Intel)
/// Try to read NVIDIA GPU info via nvidia-smi.
#[cfg(target_os = "linux")]
fn try_read_nvidia_gpu() -> Option<GpuInfo> {
    use std::process::Command;
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = stdout.lines().next()?.split(", ").collect();
    if parts.len() < 6 {
        return None;
    }
    Some(GpuInfo {
        name: parts[0].trim().to_string(),
        utilization: parts[1].trim().parse().ok(),
        temperature: parts[2].trim().parse().ok(),
        power_watts: parts[3].trim().parse().ok(),
        vram_used: parts[4].trim().parse::<u64>().ok().map(|v| v * 1024 * 1024),
        vram_total: parts[5].trim().parse::<u64>().ok().map(|v| v * 1024 * 1024),
    })
}

/// Read AMD GPU info from hwmon directory.
#[cfg(target_os = "linux")]
fn read_amd_hwmon(hwmon_dir: &std::path::Path, card_path: &str) -> Option<GpuInfo> {
    use std::fs;
    let temp = fs::read_to_string(hwmon_dir.join("temp1_input"))
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .map(|t| t / 1000);
    let power = fs::read_to_string(hwmon_dir.join("power1_average"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|p| p as f32 / 1_000_000.0);
    if temp.is_none() && power.is_none() {
        return None;
    }
    let name = fs::read_to_string(hwmon_dir.join("name"))
        .ok()
        .map_or_else(|| "AMD GPU".to_string(), |s| s.trim().to_string());
    let vram_used = fs::read_to_string(format!("{card_path}/mem_info_vram_used"))
        .ok()
        .and_then(|s| s.trim().parse().ok());
    let vram_total = fs::read_to_string(format!("{card_path}/mem_info_vram_total"))
        .ok()
        .and_then(|s| s.trim().parse().ok());
    let utilization = fs::read_to_string(format!("{card_path}/gpu_busy_percent"))
        .ok()
        .and_then(|s| s.trim().parse().ok());
    Some(GpuInfo {
        name,
        utilization,
        temperature: temp,
        power_watts: power,
        vram_used,
        vram_total,
    })
}

/// Try to read AMD GPU info via sysfs.
#[cfg(target_os = "linux")]
fn try_read_amd_gpu() -> Option<GpuInfo> {
    use std::fs;
    for card in 0..4 {
        let card_path = format!("/sys/class/drm/card{card}/device");
        if !std::path::Path::new(&card_path).exists() {
            continue;
        }
        let hwmon_path = format!("{card_path}/hwmon");
        if let Ok(entries) = fs::read_dir(&hwmon_path) {
            for entry in entries.flatten() {
                if let Some(info) = read_amd_hwmon(&entry.path(), &card_path) {
                    return Some(info);
                }
            }
        }
    }
    None
}

pub fn read_gpu_info() -> Option<GpuInfo> {
    #[cfg(target_os = "linux")]
    {
        try_read_nvidia_gpu().or_else(try_read_amd_gpu)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// F006: GPU Panel - shows GPU utilization, VRAM, temperature
/// Format GPU panel title based on detail level.
fn format_gpu_title(gpu: Option<&GpuInfo>, detail_level: DetailLevel) -> String {
    gpu.map(|g| {
        if detail_level == DetailLevel::Minimal {
            g.name.clone()
        } else {
            let temp_str = g
                .temperature
                .map(|t| format!(" │ {t}°C"))
                .unwrap_or_default();
            let power_str = g
                .power_watts
                .map(|p| format!(" │ {p:.0}W"))
                .unwrap_or_default();
            format!("{}{}{}", g.name, temp_str, power_str)
        }
    })
    .unwrap_or_else(|| "GPU".to_string())
}

/// Draw GPU utilization bar.
fn draw_gpu_util_bar(canvas: &mut DirectTerminalCanvas<'_>, inner: Rect, y: &mut f32, util: u8) {
    let bar_width = (inner.width as usize).min(20);
    let bar = make_bar(util as f64 / 100.0, bar_width);
    canvas.draw_text(
        &format!("GPU  {bar} {util:>3}%"),
        Point::new(inner.x, *y),
        &TextStyle {
            color: percent_color(util as f64),
            ..Default::default()
        },
    );
    *y += 1.0;
}

/// Draw VRAM usage bar.
fn draw_vram_bar(
    canvas: &mut DirectTerminalCanvas<'_>,
    inner: Rect,
    y: &mut f32,
    used: u64,
    total: u64,
) {
    if total == 0 || !can_draw_row(*y, &inner) {
        return;
    }
    let pct = safe_pct(used, total);
    let bar_width = (inner.width as usize).min(20);
    let bar = make_bar(pct / 100.0, bar_width);
    canvas.draw_text(
        &format!(
            "VRAM {bar} {}M/{}M",
            used / 1024 / 1024,
            total / 1024 / 1024
        ),
        Point::new(inner.x, *y),
        &TextStyle {
            color: percent_color(pct),
            ..Default::default()
        },
    );
    *y += 1.0;
}

/// Draw GPU history graphs in exploded mode.
fn draw_gpu_history_graphs(
    app: &App,
    canvas: &mut DirectTerminalCanvas<'_>,
    inner: Rect,
    y: &mut f32,
) {
    let gpu_history: Vec<f64> = app.gpu_history.as_slice().to_vec();
    if !gpu_history.is_empty() {
        let mut graph = BrailleGraph::new(gpu_history)
            .with_color(GPU_COLOR)
            .with_label("GPU History")
            .with_range(0.0, 100.0);
        graph.layout(Rect::new(inner.x, *y, inner.width, 6.0));
        graph.paint(canvas);
        *y += 7.0;
    }
    let vram_history: Vec<f64> = app.vram_history.as_slice().to_vec();
    if !vram_history.is_empty() {
        let mut graph = BrailleGraph::new(vram_history)
            .with_color(VRAM_GRAPH_COLOR)
            .with_label("VRAM History")
            .with_range(0.0, 100.0);
        graph.layout(Rect::new(inner.x, *y, inner.width, 6.0));
        graph.paint(canvas);
        *y += 7.0;
    }
}

/// Draw GPU processes list.
fn draw_gpu_procs(app: &App, canvas: &mut DirectTerminalCanvas<'_>, inner: Rect, y: &mut f32) {
    let Some(gpu_data) = app.analyzers.gpu_procs_data() else {
        return;
    };
    if gpu_data.processes.is_empty() {
        return;
    }
    *y += 1.0;
    canvas.draw_text(
        "TY  PID   SM%  MEM%  CMD",
        Point::new(inner.x, *y),
        &TextStyle {
            color: HEADER_COLOR,
            ..Default::default()
        },
    );
    *y += 1.0;
    for proc in gpu_data.processes.iter().take(3) {
        if *y >= inner.y + inner.height {
            break;
        }
        let (type_badge, badge_color) = gpu_proc_badge(proc.proc_type.as_str());
        canvas.draw_text(
            type_badge,
            Point::new(inner.x, *y),
            &TextStyle {
                color: badge_color,
                ..Default::default()
            },
        );
        let sm_str = format_proc_util(proc.gpu_util());
        let mem_str = format_proc_util(if proc.mem_util > 0 {
            Some(proc.mem_util as f32)
        } else {
            None
        });
        let proc_info = format!(
            " {:>5} {}%  {}%  {}",
            proc.pid,
            sm_str,
            mem_str,
            truncate_name(&proc.name, 12)
        );
        canvas.draw_text(
            &proc_info,
            Point::new(inner.x + 1.0, *y),
            &TextStyle {
                color: PROC_INFO_COLOR,
                ..Default::default()
            },
        );
        *y += 1.0;
    }
}

/// Draw GPU temperature if available and within bounds.
fn draw_gpu_temp(canvas: &mut DirectTerminalCanvas<'_>, inner: Rect, y: &mut f32, temp: u32) {
    if *y >= inner.y + inner.height {
        return;
    }
    canvas.draw_text(
        &format!("Temp {temp}°C"),
        Point::new(inner.x, *y),
        &TextStyle {
            color: gpu_temp_color(temp),
            ..Default::default()
        },
    );
    *y += 1.0;
}

/// Draw GPU power if available and within bounds.
fn draw_gpu_power(canvas: &mut DirectTerminalCanvas<'_>, inner: Rect, y: &mut f32, power: f32) {
    if *y >= inner.y + inner.height {
        return;
    }
    canvas.draw_text(
        &format!("Power {power:.0}W"),
        Point::new(inner.x, *y),
        &TextStyle {
            color: POWER_COLOR,
            ..Default::default()
        },
    );
    *y += 1.0;
}

/// Draw GPU panel content when GPU is present.
fn draw_gpu_content(
    app: &App,
    canvas: &mut DirectTerminalCanvas<'_>,
    inner: Rect,
    gpu: &GpuInfo,
    detail_level: DetailLevel,
) {
    let mut y = inner.y;
    if let Some(util) = gpu.utilization {
        draw_gpu_util_bar(canvas, inner, &mut y, util);
    }
    if let (Some(used), Some(total)) = (gpu.vram_used, gpu.vram_total) {
        draw_vram_bar(canvas, inner, &mut y, used, total);
    }
    if let Some(temp) = gpu.temperature {
        draw_gpu_temp(canvas, inner, &mut y, temp);
    }
    if let Some(power) = gpu.power_watts {
        draw_gpu_power(canvas, inner, &mut y, power);
    }
    if detail_level == DetailLevel::Exploded && y < inner.y + inner.height - 10.0 {
        draw_gpu_history_graphs(app, canvas, inner, &mut y);
    }
    if detail_level >= DetailLevel::Expanded && y < inner.y + inner.height - 3.0 {
        draw_gpu_procs(app, canvas, inner, &mut y);
    }
}

pub(super) fn draw_gpu_panel(app: &App, canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    let detail_level = DetailLevel::for_height(bounds.height as u16);
    let gpu = app.gpu_info.clone();
    let title = format_gpu_title(gpu.as_ref(), detail_level);

    let is_focused = app.is_panel_focused(PanelType::Gpu);
    let mut border = create_panel_border(&title, GPU_COLOR, is_focused);
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();
    if inner.height < 1.0 {
        return;
    }

    canvas.push_clip(inner);

    if let Some(ref g) = gpu {
        draw_gpu_content(app, canvas, inner, g, detail_level);
    } else if !app.deterministic {
        canvas.draw_text(
            "No GPU detected or nvidia-smi not available",
            Point::new(inner.x, inner.y),
            &TextStyle {
                color: HEADER_COLOR,
                ..Default::default()
            },
        );
    }

    canvas.pop_clip();
}

/// Battery information from /`sys/class/power_supply`
#[derive(Debug, Default)]
struct BatteryInfo {
    /// Capacity as percentage (0-100)
    capacity: u8,
    /// Status: Charging, Discharging, Full, Not charging, Unknown
    status: String,
    /// Time remaining in minutes (if available)
    time_remaining_mins: Option<u32>,
    /// Whether battery is present
    #[allow(dead_code)]
    present: bool,
}

/// Read battery info from /`sys/class/power_supply` (Linux only)
fn read_battery_info() -> Option<BatteryInfo> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        use std::path::Path;

        // Look for BAT0, BAT1, etc.
        for i in 0..4 {
            let bat_path = format!("/sys/class/power_supply/BAT{i}");
            let path = Path::new(&bat_path);

            if !path.exists() {
                continue;
            }

            // Read capacity
            let capacity = fs::read_to_string(format!("{bat_path}/capacity"))
                .ok()
                .and_then(|s| s.trim().parse::<u8>().ok())
                .unwrap_or(0);

            // Read status
            let status = fs::read_to_string(format!("{bat_path}/status"))
                .ok()
                .map_or_else(|| "Unknown".to_string(), |s| s.trim().to_string());

            // Try to calculate time remaining
            // Uses energy_now/power_now or charge_now/current_now
            let time_remaining_mins = {
                let energy_now = fs::read_to_string(format!("{bat_path}/energy_now"))
                    .ok()
                    .and_then(|s| s.trim().parse::<u64>().ok());
                let power_now = fs::read_to_string(format!("{bat_path}/power_now"))
                    .ok()
                    .and_then(|s| s.trim().parse::<u64>().ok());
                let energy_full = fs::read_to_string(format!("{bat_path}/energy_full"))
                    .ok()
                    .and_then(|s| s.trim().parse::<u64>().ok());

                match (energy_now, power_now, energy_full, status.as_str()) {
                    (Some(en), Some(pn), _, "Discharging") if pn > 0 => Some((en * 60 / pn) as u32),
                    (Some(en), Some(pn), Some(ef), "Charging") if pn > 0 => {
                        let remaining = ef.saturating_sub(en);
                        Some((remaining * 60 / pn) as u32)
                    }
                    _ => None,
                }
            };

            return Some(BatteryInfo {
                capacity,
                status,
                time_remaining_mins,
                present: true,
            });
        }

        None
    }

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// F007: Battery Panel - shows charge level and state
pub(super) fn draw_battery_panel(app: &App, canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    let battery = read_battery_info();

    // ttop-style title (Border adds outer spaces)
    let title = battery
        .as_ref()
        .map(|b| {
            let time_str = b
                .time_remaining_mins
                .map(|m| {
                    if m >= 60 {
                        format!(" │ {}h{}m", m / 60, m % 60)
                    } else {
                        format!(" │ {m}m")
                    }
                })
                .unwrap_or_default();
            format!("Battery │ {}% │ {}{}", b.capacity, b.status, time_str)
        })
        .unwrap_or_else(|| "Battery │ No battery".to_string());

    // Check if this panel is focused (SPEC-024 v5.0 Feature D)
    let is_focused = app.is_panel_focused(PanelType::Battery);
    let mut border = create_panel_border(&title, BATTERY_COLOR, is_focused);
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }

    if let Some(bat) = battery {
        // Draw charge bar with inverted color (red=low, green=full)
        let bar_width = (inner.width as usize).min(30);
        let bar = make_bar(bat.capacity as f64 / 100.0, bar_width);

        // Color: red when low, yellow when medium, green when high
        let color = if bat.capacity < 20 {
            Color {
                r: 1.0,
                g: 0.3,
                b: 0.3,
                a: 1.0,
            } // Red
        } else if bat.capacity < 50 {
            Color {
                r: 1.0,
                g: 0.8,
                b: 0.2,
                a: 1.0,
            } // Yellow
        } else {
            Color {
                r: 0.3,
                g: 0.9,
                b: 0.3,
                a: 1.0,
            } // Green
        };

        canvas.draw_text(
            &bar,
            Point::new(inner.x, inner.y),
            &TextStyle {
                color,
                ..Default::default()
            },
        );

        // Show status icon
        if inner.height >= 2.0 {
            let status_icon = match bat.status.as_str() {
                "Charging" => "⚡ Charging",
                "Discharging" => "🔋 Discharging",
                "Full" => "✓ Full",
                "Not charging" => "— Idle",
                _ => "? Unknown",
            };
            canvas.draw_text(
                status_icon,
                Point::new(inner.x, inner.y + 1.0),
                &TextStyle {
                    color: Color {
                        r: 0.7,
                        g: 0.7,
                        b: 0.7,
                        a: 1.0,
                    },
                    ..Default::default()
                },
            );
        }
    } else {
        canvas.draw_text(
            "No battery detected",
            Point::new(inner.x, inner.y),
            &TextStyle {
                color: Color {
                    r: 0.5,
                    g: 0.5,
                    b: 0.5,
                    a: 1.0,
                },
                ..Default::default()
            },
        );
    }
}

/// Get temperature indicator and color based on threshold.
fn temp_indicator_color(temp: f32) -> (&'static str, Color) {
    if temp > 85.0 {
        (
            "✗",
            Color {
                r: 1.0,
                g: 0.3,
                b: 0.3,
                a: 1.0,
            },
        )
    } else if temp > 70.0 {
        (
            "⚠",
            Color {
                r: 1.0,
                g: 0.8,
                b: 0.2,
                a: 1.0,
            },
        )
    } else {
        (
            "✓",
            Color {
                r: 0.3,
                g: 0.9,
                b: 0.3,
                a: 1.0,
            },
        )
    }
}

/// Get sensor status indicator and color.
fn sensor_status_indicator(
    status: crate::ptop::analyzers::SensorStatus,
    is_fan: bool,
) -> (&'static str, Color) {
    use crate::ptop::analyzers::SensorStatus;
    match status {
        SensorStatus::Critical | SensorStatus::Fault => (
            "✗",
            Color {
                r: 1.0,
                g: 0.3,
                b: 0.3,
                a: 1.0,
            },
        ),
        SensorStatus::Warning | SensorStatus::Low => (
            "⚠",
            Color {
                r: 1.0,
                g: 0.8,
                b: 0.2,
                a: 1.0,
            },
        ),
        SensorStatus::Normal => {
            if is_fan {
                (
                    "✓",
                    Color {
                        r: 0.3,
                        g: 0.8,
                        b: 0.9,
                        a: 1.0,
                    },
                )
            } else {
                (
                    "✓",
                    Color {
                        r: 0.9,
                        g: 0.7,
                        b: 0.3,
                        a: 1.0,
                    },
                )
            }
        }
    }
}

/// Draw a sensor row.
fn draw_sensor_row(canvas: &mut dyn Canvas, x: f32, y: f32, text: &str, color: Color) {
    canvas.draw_text(
        text,
        Point::new(x, y),
        &TextStyle {
            color,
            ..Default::default()
        },
    );
}

/// Build sensor title extra info string from health data.
fn build_sensor_extra_info(
    health_data: Option<&crate::ptop::analyzers::SensorHealthData>,
) -> String {
    use crate::ptop::analyzers::SensorType;
    let Some(data) = health_data else {
        return String::new();
    };
    let fan_count = data.type_counts.get(&SensorType::Fan).copied().unwrap_or(0);
    let volt_count = data
        .type_counts
        .get(&SensorType::Voltage)
        .copied()
        .unwrap_or(0);
    if fan_count > 0 || volt_count > 0 {
        format!(" │ {fan_count}F {volt_count}V")
    } else {
        String::new()
    }
}

/// Collect system components if not in deterministic mode.
fn collect_sensor_components(deterministic: bool) -> (Option<sysinfo::Components>, f32) {
    use sysinfo::{Component, Components};
    if deterministic {
        (None, 0.0_f32)
    } else {
        let comps = Components::new_with_refreshed_list();
        let temp = comps
            .iter()
            .filter_map(Component::temperature)
            .fold(0.0_f32, f32::max);
        (Some(comps), temp)
    }
}

/// Draw temperature sensors from sysinfo Components.
fn draw_temp_sensors(
    canvas: &mut dyn Canvas,
    comps: &sysinfo::Components,
    inner: Rect,
    y: &mut f32,
    rows_used: &mut usize,
    max_rows: usize,
) {
    for component in comps {
        if *rows_used >= max_rows {
            break;
        }
        let Some(temp) = component.temperature() else {
            continue;
        };
        let label_short: String = component.label().chars().take(12).collect();
        let (indicator, color) = temp_indicator_color(temp);
        let text = format!("{indicator} {label_short:<12} {temp:>5.1}°C");
        draw_sensor_row(canvas, inner.x, *y, &text, color);
        *y += 1.0;
        *rows_used += 1;
    }
}

/// Draw fan and voltage sensors from health data.
fn draw_health_sensors(
    canvas: &mut dyn Canvas,
    health_data: &crate::ptop::analyzers::SensorHealthData,
    inner: Rect,
    y: &mut f32,
    rows_used: &mut usize,
    max_rows: usize,
) {
    use crate::ptop::analyzers::SensorType;
    // Fan sensors
    for fan in health_data.fans() {
        if *rows_used >= max_rows {
            break;
        }
        let (indicator, color) = sensor_status_indicator(fan.status, true);
        let text = format!(
            "{indicator} {:<12} {:>5.0} RPM",
            fan.short_label(),
            fan.value
        );
        draw_sensor_row(canvas, inner.x, *y, &text, color);
        *y += 1.0;
        *rows_used += 1;
    }
    // Voltage sensors
    for volt in health_data.by_type(SensorType::Voltage) {
        if *rows_used >= max_rows {
            break;
        }
        let (indicator, color) = sensor_status_indicator(volt.status, false);
        let text = format!(
            "{indicator} {:<12} {:>6.2}V",
            volt.short_label(),
            volt.value
        );
        draw_sensor_row(canvas, inner.x, *y, &text, color);
        *y += 1.0;
        *rows_used += 1;
    }
}

/// F008: Sensors Panel - shows temperature sensors with health indicators
pub(super) fn draw_sensors_panel(app: &App, canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    let (components, max_temp) = collect_sensor_components(app.deterministic);
    let sensor_health_data = app.snapshot_sensor_health.as_ref();
    let extra_info = build_sensor_extra_info(sensor_health_data);
    let title = format!("Sensors │ {max_temp:.0}°C{extra_info}");

    // Check if this panel is focused (SPEC-024 v5.0 Feature D)
    let is_focused = app.is_panel_focused(PanelType::Sensors);
    let mut border = create_panel_border(&title, SENSORS_COLOR, is_focused);
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }
    let Some(ref comps) = components else {
        return;
    };

    let mut y = inner.y;
    let max_rows = inner.height as usize;
    let mut rows_used = 0;

    draw_temp_sensors(canvas, comps, inner, &mut y, &mut rows_used, max_rows);
    if let Some(health_data) = sensor_health_data {
        draw_health_sensors(canvas, health_data, inner, &mut y, &mut rows_used, max_rows);
    }

    if comps.is_empty() && sensor_health_data.is_none() {
        draw_sensor_row(
            canvas,
            inner.x,
            inner.y,
            "No sensors detected",
            Color {
                r: 0.5,
                g: 0.5,
                b: 0.5,
                a: 1.0,
            },
        );
    }
}

/// Containers Panel - shows Docker/Podman containers (ttop style)
pub(super) fn draw_containers_panel(
    app: &App,
    canvas: &mut DirectTerminalCanvas<'_>,
    bounds: Rect,
) {
    // ttop-style title
    let title = "Containers";

    // Check if this panel is focused (SPEC-024 v5.0 Feature D)
    let is_focused = app.is_panel_focused(PanelType::Containers);
    let mut border = create_panel_border(title, CONTAINERS_COLOR, is_focused);
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }

    // In deterministic mode, show "No running containers" like ttop
    if app.deterministic {
        canvas.draw_text(
            "No running containers",
            Point::new(inner.x, inner.y),
            &TextStyle {
                color: Color {
                    r: 0.5,
                    g: 0.5,
                    b: 0.5,
                    a: 1.0,
                },
                ..Default::default()
            },
        );
        return;
    }

    // Get containers data from analyzer
    if let Some(data) = app.analyzers.containers_data() {
        if data.containers.is_empty() {
            canvas.draw_text(
                "No running containers",
                Point::new(inner.x, inner.y),
                &TextStyle {
                    color: Color {
                        r: 0.5,
                        g: 0.5,
                        b: 0.5,
                        a: 1.0,
                    },
                    ..Default::default()
                },
            );
        } else {
            let mut y = inner.y;
            for container in data.containers.iter().take(inner.height as usize) {
                let status_icon = match container.state {
                    ContainerState::Running => "●",
                    ContainerState::Paused => "◐",
                    ContainerState::Exited => "○",
                    ContainerState::Created => "◎",
                    ContainerState::Restarting => "↻",
                    ContainerState::Removing => "⊘",
                    ContainerState::Dead => "✗",
                    ContainerState::Unknown => "?",
                };
                let name: String = container.name.chars().take(20).collect();
                let cpu = container.stats.cpu_percent;
                let mem_mb = container.stats.memory_bytes / (1024 * 1024);

                let text = format!("{status_icon} {name:<20} {cpu:>5.1}% {mem_mb:>4}MB");
                canvas.draw_text(
                    &text,
                    Point::new(inner.x, y),
                    &TextStyle {
                        color: CONTAINERS_COLOR,
                        ..Default::default()
                    },
                );
                y += 1.0;
            }
        }
    } else {
        canvas.draw_text(
            "No container runtime",
            Point::new(inner.x, inner.y),
            &TextStyle {
                color: Color {
                    r: 0.5,
                    g: 0.5,
                    b: 0.5,
                    a: 1.0,
                },
                ..Default::default()
            },
        );
    }
}

/// F010: PSI Panel - shows CPU/Memory/IO pressure (Linux only)
pub(super) fn draw_psi_panel(app: &App, canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    // ttop-style title (Border adds outer spaces)
    let title = "Pressure │ —";

    // Check if this panel is focused (SPEC-024 v5.0 Feature D)
    let is_focused = app.is_panel_focused(PanelType::Psi);
    let mut border = create_panel_border(title, PSI_COLOR, is_focused);
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }

    // Use PSI data from analyzer
    if let Some(psi) = app.psi_data() {
        if psi.available {
            let mut y = inner.y;

            // CPU pressure
            let cpu = psi.cpu.some.avg10;
            let cpu_symbol = pressure_symbol(cpu);
            let cpu_color = pressure_color(cpu);
            canvas.draw_text(
                &format!("CPU  {cpu_symbol} {cpu:>5.1}%"),
                Point::new(inner.x, y),
                &TextStyle {
                    color: cpu_color,
                    ..Default::default()
                },
            );
            y += 1.0;

            // Memory pressure
            if y < inner.y + inner.height {
                let mem = psi.memory.some.avg10;
                let mem_symbol = pressure_symbol(mem);
                let mem_color = pressure_color(mem);
                canvas.draw_text(
                    &format!("MEM  {mem_symbol} {mem:>5.1}%"),
                    Point::new(inner.x, y),
                    &TextStyle {
                        color: mem_color,
                        ..Default::default()
                    },
                );
                y += 1.0;
            }

            // I/O pressure
            if y < inner.y + inner.height {
                let io = psi.io.some.avg10;
                let io_symbol = pressure_symbol(io);
                let io_color = pressure_color(io);
                canvas.draw_text(
                    &format!("I/O  {io_symbol} {io:>5.1}%"),
                    Point::new(inner.x, y),
                    &TextStyle {
                        color: io_color,
                        ..Default::default()
                    },
                );
            }
        } else {
            canvas.draw_text(
                "PSI not available",
                Point::new(inner.x, inner.y),
                &TextStyle {
                    color: Color {
                        r: 0.5,
                        g: 0.5,
                        b: 0.5,
                        a: 1.0,
                    },
                    ..Default::default()
                },
            );
        }
    } else {
        canvas.draw_text(
            "PSI not available",
            Point::new(inner.x, inner.y),
            &TextStyle {
                color: Color {
                    r: 0.5,
                    g: 0.5,
                    b: 0.5,
                    a: 1.0,
                },
                ..Default::default()
            },
        );
    }
}

fn pressure_symbol(pct: f64) -> &'static str {
    if pct > 50.0 {
        "▲▲"
    } else if pct > 20.0 {
        "▲"
    } else if pct > 5.0 {
        "▼"
    } else if pct > 1.0 {
        "◐"
    } else {
        "—"
    }
}

fn pressure_color(pct: f64) -> Color {
    if pct > 50.0 {
        Color {
            r: 1.0,
            g: 0.2,
            b: 0.2,
            a: 1.0,
        } // Critical red
    } else if pct > 20.0 {
        Color {
            r: 1.0,
            g: 0.5,
            b: 0.3,
            a: 1.0,
        } // High orange
    } else if pct > 5.0 {
        Color {
            r: 1.0,
            g: 0.8,
            b: 0.2,
            a: 1.0,
        } // Medium yellow
    } else if pct > 1.0 {
        Color {
            r: 0.3,
            g: 0.9,
            b: 0.3,
            a: 1.0,
        } // Low green
    } else {
        Color {
            r: 0.4,
            g: 0.4,
            b: 0.4,
            a: 1.0,
        } // None gray
    }
}

/// Get service name from port
fn port_to_service(port: u16) -> &'static str {
    match port {
        22 => "SSH",
        80 => "HTTP",
        443 => "HTTPS",
        53 => "DNS",
        25 => "SMTP",
        21 => "FTP",
        3306 => "MySQL",
        5432 => "Pgsql",
        6379 => "Redis",
        27017 => "Mongo",
        8080 => "HTTP",
        8443 => "HTTPS",
        9000..=9999 => "App",
        _ => "",
    }
}

/// F012: Connections Panel - shows active network connections
/// Get connection counts from snapshot data.
fn get_connection_counts(
    conn_data: Option<&crate::ptop::analyzers::ConnectionsData>,
) -> (usize, usize) {
    let Some(data) = conn_data else {
        return (0, 0);
    };
    let listen = data
        .connections
        .iter()
        .filter(|c| c.state == TcpState::Listen)
        .count();
    let active = data
        .connections
        .iter()
        .filter(|c| c.state == TcpState::Established)
        .count();
    (listen, active)
}

/// Get state display color.
fn state_display_color(state: TcpState) -> Color {
    match state {
        TcpState::Established => ACTIVE_COLOR,
        TcpState::Listen => LISTEN_COLOR,
        _ => CONN_DIM_COLOR,
    }
}

/// Get state short code.
fn state_short_code(state: TcpState) -> &'static str {
    match state {
        TcpState::Established => "E",
        TcpState::Listen => "L",
        TcpState::TimeWait => "T",
        TcpState::CloseWait => "C",
        TcpState::SynSent => "S",
        _ => "?",
    }
}

/// Check if address is local (loopback, private, or link-local).
fn is_local_address(addr: &std::net::IpAddr) -> bool {
    match addr {
        std::net::IpAddr::V4(ip) => ip.is_loopback() || ip.is_private() || ip.is_link_local(),
        std::net::IpAddr::V6(ip) => ip.is_loopback(),
    }
}

/// Format remote address for display.
fn format_remote_addr(conn: &crate::ptop::analyzers::TcpConnection) -> String {
    if conn.state == TcpState::Listen {
        "*".to_string()
    } else {
        let addr_str = format!("{}:{}", conn.remote_addr, conn.remote_port);
        if addr_str.len() > 17 {
            format!("{}…", &addr_str[..16])
        } else {
            addr_str
        }
    }
}

/// Format process name for display.
fn format_process_name(conn: &crate::ptop::analyzers::TcpConnection) -> String {
    conn.process_name
        .as_ref()
        .map(|s| {
            if s.len() > 10 {
                format!("{}…", &s[..9])
            } else {
                s.clone()
            }
        })
        .or_else(|| conn.pid.map(|p| p.to_string()))
        .unwrap_or_else(|| "-".to_string())
}

/// Get hot indicator color.
fn hot_indicator_color(indicator: &str) -> Color {
    if indicator == "●" {
        Color {
            r: 1.0,
            g: 0.4,
            b: 0.2,
            a: 1.0,
        }
    }
    // Orange
    else {
        Color {
            r: 1.0,
            g: 0.7,
            b: 0.3,
            a: 1.0,
        }
    } // Yellow
}

/// Get geo indicator (L=local, R=remote, -=listen).
fn get_geo_indicator(state: TcpState, addr: &std::net::IpAddr) -> &'static str {
    if state == TcpState::Listen {
        "-"
    } else if is_local_address(addr) {
        "L"
    } else {
        "R"
    }
}

/// Sort order for TCP state (LISTEN first, then ESTABLISHED, then others).
fn tcp_state_order(state: TcpState) -> u8 {
    match state {
        TcpState::Listen => 0,
        TcpState::Established => 1,
        _ => 2,
    }
}

/// Draw deterministic mode header for connections panel.
fn draw_connections_deterministic(canvas: &mut dyn Canvas, inner: Rect) {
    let header = "SVC   LOCA REMOT GE ST AGE   PROC";
    canvas.draw_text(
        header,
        Point::new(inner.x, inner.y),
        &TextStyle {
            color: CONNECTIONS_COLOR,
            ..Default::default()
        },
    );
}

pub(super) fn draw_connections_panel(
    app: &App,
    canvas: &mut DirectTerminalCanvas<'_>,
    bounds: Rect,
) {
    let (listen_count, active_count) = get_connection_counts(app.snapshot_connections.as_ref());
    let connections = app.snapshot_connections.as_ref().map(|c| &c.connections);

    // Generate sparkline for connection history (CB-CONN-007) using helper
    let sparkline_str = app
        .snapshot_connections
        .as_ref()
        .map(|conn_data| build_sparkline(&conn_data.established_sparkline(), 12))
        .unwrap_or_default();

    // ttop-style title (Border adds outer spaces)
    let title =
        format!("Connections │ {active_count} active │ {listen_count} listen{sparkline_str}");

    // Check if this panel is focused (SPEC-024 v5.0 Feature D)
    let is_focused = app.is_panel_focused(PanelType::Connections);
    let mut border = create_panel_border(&title, CONNECTIONS_COLOR, is_focused);
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }
    if app.deterministic {
        draw_connections_deterministic(canvas, inner);
        return;
    }

    // Header for real data mode
    let header = "SVC   LOCAL        REMOTE            GE ST  AGE   PROC";
    canvas.draw_text(
        header,
        Point::new(inner.x, inner.y),
        &TextStyle {
            color: CONNECTIONS_COLOR,
            ..Default::default()
        },
    );

    let Some(conns) = connections else {
        canvas.draw_text(
            "No data",
            Point::new(inner.x, inner.y + 1.0),
            &TextStyle {
                color: CONN_DIM_COLOR,
                ..Default::default()
            },
        );
        return;
    };

    // Show connections (skip loopback, prioritize ESTABLISHED and LISTEN)
    use std::net::{IpAddr, Ipv4Addr};
    let loopback_v4: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

    let mut display_conns: Vec<_> = conns
        .iter()
        .filter(|c| c.remote_addr != loopback_v4 || c.state == TcpState::Listen)
        .collect();
    display_conns.sort_by(|a, b| tcp_state_order(a.state).cmp(&tcp_state_order(b.state)));

    let max_rows = (inner.height as usize).saturating_sub(1);

    for (i, conn) in display_conns.iter().take(max_rows).enumerate() {
        let y = inner.y + 1.0 + i as f32;
        if y >= inner.y + inner.height {
            break;
        }

        let svc = port_to_service(conn.local_port);
        let local = format!(":{}", conn.local_port);
        let remote = format_remote_addr(conn);
        let geo = get_geo_indicator(conn.state, &conn.remote_addr);
        let state_short = state_short_code(conn.state);
        let proc_name = format_process_name(conn);
        let state_color = state_display_color(conn.state);

        // Get connection age (CB-CONN-001)
        let age = conn.age_display();

        // Get hot indicator (CB-CONN-006)
        let (hot_indicator, _hot_level) = conn.hot_indicator();

        // Format: SVC   LOCAL        REMOTE            GE ST  AGE   PROC
        let line = format!(
            "{svc:<5} {local:<12} {remote:<17} {geo:<2} {state_short:<3} {age:<5} {proc_name}"
        );
        canvas.draw_text(
            &line,
            Point::new(inner.x, y),
            &TextStyle {
                color: state_color,
                ..Default::default()
            },
        );

        // Draw hot indicator after the line (CB-CONN-006)
        if !hot_indicator.is_empty() {
            let hot_x = inner.x + 56.0;
            if hot_x < inner.x + inner.width {
                canvas.draw_text(
                    hot_indicator,
                    Point::new(hot_x, y),
                    &TextStyle {
                        color: hot_indicator_color(hot_indicator),
                        ..Default::default()
                    },
                );
            }
        }
    }

    // If no connections, show message
    if display_conns.is_empty() && inner.height > 1.0 {
        canvas.draw_text(
            "No active connections",
            Point::new(inner.x, inner.y + 1.0),
            &TextStyle {
                color: CONN_DIM_COLOR,
                ..Default::default()
            },
        );
    }
}

/// Get type character for sensor based on label: C (CPU), G (GPU), D (Disk), F (Fan), M (Mobo)
fn sensor_type_char(label: &str) -> char {
    if label.contains("CPU") || label.contains("Core") {
        'C'
    } else if label.contains("GPU") {
        'G'
    } else if label.contains("nvme") || label.contains("SSD") || label.contains("HDD") {
        'D'
    } else if label.contains("fan") || label.contains("Fan") {
        'F'
    } else {
        'M'
    }
}

/// Get color for sensor temperature display
fn sensor_temp_display_color(temp: f32) -> Color {
    if temp > 85.0 {
        Color {
            r: 1.0,
            g: 0.3,
            b: 0.3,
            a: 1.0,
        }
    } else if temp > 70.0 {
        Color {
            r: 1.0,
            g: 0.8,
            b: 0.2,
            a: 1.0,
        }
    } else {
        Color {
            r: 0.3,
            g: 0.9,
            b: 0.3,
            a: 1.0,
        }
    }
}

/// Build 4-char dual-color bar for temperature display
fn build_temp_bar(temp: f32) -> String {
    let pct = (temp / 100.0).clamp(0.0, 1.0);
    let filled = (pct * 4.0).round() as usize;
    (0..4).map(|i| if i < filled { '▄' } else { '░' }).collect()
}

/// Draw a sensor row in compact view
fn draw_sensor_compact_row(canvas: &mut dyn Canvas, x: f32, y: f32, label: &str, temp: f32) {
    let type_char = sensor_type_char(label);
    let bar = build_temp_bar(temp);
    let label_short: String = label.chars().take(8).collect();
    let text = format!("{type_char} {bar} {temp:>4.0}°C {label_short}");
    let color = sensor_temp_display_color(temp);
    canvas.draw_text(
        &text,
        Point::new(x, y),
        &TextStyle {
            color,
            ..Default::default()
        },
    );
}

/// F009: Sensors Compact Panel - compact sensor display with dual-color bars
pub(super) fn draw_sensors_compact_panel(
    _app: &App,
    canvas: &mut DirectTerminalCanvas<'_>,
    bounds: Rect,
) {
    use sysinfo::{Component, Components};

    let components = Components::new_with_refreshed_list();
    let max_temp = components
        .iter()
        .filter_map(Component::temperature)
        .fold(0.0_f32, f32::max);

    let title = format!("Sensors │ {max_temp:.0}°C");

    let mut border = Border::new()
        .with_title(&title)
        .with_style(BorderStyle::Rounded)
        .with_color(SENSORS_COLOR)
        .with_title_left_aligned();
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }

    let mut y = inner.y;
    for component in components.iter().take(inner.height as usize) {
        let label = component.label();
        let Some(temp) = component.temperature() else {
            continue;
        };
        draw_sensor_compact_row(canvas, inner.x, y, label, temp);
        y += 1.0;
    }

    if components.is_empty() {
        canvas.draw_text(
            "No sensors",
            Point::new(inner.x, inner.y),
            &TextStyle {
                color: Color {
                    r: 0.5,
                    g: 0.5,
                    b: 0.5,
                    a: 1.0,
                },
                ..Default::default()
            },
        );
    }
}

/// F011: System Panel - composite view with `sensors_compact` + containers
pub(super) fn draw_system_panel(app: &App, canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    let title = "System";

    let mut border = Border::new()
        .with_title(title)
        .with_style(BorderStyle::Rounded)
        .with_color(Color {
            r: 0.5,
            g: 0.7,
            b: 0.9,
            a: 1.0,
        })
        .with_title_left_aligned();
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }

    // Show system info (O(1) render - uses cached data from App)
    let mut y = inner.y;

    // Hostname (cached at startup)
    if !app.hostname.is_empty() {
        canvas.draw_text(
            &format!("Host: {}", app.hostname),
            Point::new(inner.x, y),
            &TextStyle {
                color: Color {
                    r: 0.7,
                    g: 0.9,
                    b: 1.0,
                    a: 1.0,
                },
                ..Default::default()
            },
        );
        y += 1.0;
    }

    // Kernel version (cached at startup)
    if !app.kernel_version.is_empty() && y < inner.y + inner.height {
        canvas.draw_text(
            &app.kernel_version,
            Point::new(inner.x, y),
            &TextStyle {
                color: Color {
                    r: 0.6,
                    g: 0.8,
                    b: 0.9,
                    a: 1.0,
                },
                ..Default::default()
            },
        );
        y += 1.0;
    }

    // Container detection (cached at startup)
    if y < inner.y + inner.height {
        let container_text = if app.in_container {
            "Container: Yes"
        } else {
            "Container: No"
        };
        canvas.draw_text(
            container_text,
            Point::new(inner.x, y),
            &TextStyle {
                color: Color {
                    r: 0.5,
                    g: 0.7,
                    b: 0.5,
                    a: 1.0,
                },
                ..Default::default()
            },
        );
    }
}

/// Get short name and color for a mount point
fn mount_point_style(mount: &str) -> (&str, Color) {
    if mount == "/" {
        ("/", Color::new(0.8, 0.5, 0.3, 1.0)) // Root: orange
    } else if mount.contains("nvme") || mount.starts_with("/dev/nvme") {
        ("nvme", Color::new(0.3, 0.8, 0.3, 1.0)) // NVMe: green
    } else if mount.contains("/home") {
        ("home", Color::new(0.5, 0.5, 0.9, 1.0)) // Home: blue
    } else if mount.contains("/tmp") || mount.contains("/var/tmp") {
        ("tmp", Color::new(0.9, 0.9, 0.3, 1.0)) // Tmp: yellow
    } else if mount.contains("/boot") {
        ("boot", Color::new(0.9, 0.3, 0.3, 1.0)) // Boot: red
    } else {
        ("other", Color::new(0.6, 0.6, 0.6, 1.0)) // Other: gray
    }
}

/// Extract short name from mount point path, max 6 chars
fn mount_short_name(mount: &str) -> &str {
    let name = mount.split('/').next_back().unwrap_or("disk");
    if name.len() > 6 {
        &name[..6]
    } else {
        name
    }
}

/// Build treemap node from disk info
fn build_disk_node(disk: &sysinfo::Disk) -> Option<TreemapNode> {
    let mount = disk.mount_point().to_string_lossy();
    let used = disk.total_space() - disk.available_space();
    let total = disk.total_space();

    if total == 0 {
        return None;
    }

    let (known_name, color) = mount_point_style(&mount);
    let short_name = if known_name == "other" {
        mount_short_name(&mount)
    } else {
        known_name
    };

    let used_pct = (used as f64 / total as f64) * 100.0;
    let used_color = percent_color(used_pct);
    let free_color = Color::new(0.2, 0.3, 0.2, 1.0);

    let children = vec![
        TreemapNode::leaf_colored("used", used as f64, used_color),
        TreemapNode::leaf_colored("free", disk.available_space() as f64, free_color),
    ];

    let mut node = TreemapNode::branch(short_name, children);
    node.color = Some(color);
    Some(node)
}

/// F013: Treemap Panel - file system treemap visualization
pub(super) fn draw_treemap_panel(app: &App, canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    let (total_used, total_space): (u64, u64) = app
        .disks
        .iter()
        .map(|d| (d.total_space() - d.available_space(), d.total_space()))
        .fold((0, 0), |(au, at), (u, t)| (au + u, at + t));

    let disk_count = app.disks.iter().count();
    let title = format!(
        "Treemap │ {} disk{} │ {:.0}G / {:.0}G",
        disk_count,
        if disk_count == 1 { "" } else { "s" },
        total_used as f64 / 1024.0 / 1024.0 / 1024.0,
        total_space as f64 / 1024.0 / 1024.0 / 1024.0,
    );

    let mut border = Border::new()
        .with_title(&title)
        .with_style(BorderStyle::Rounded)
        .with_color(FILES_COLOR)
        .with_title_left_aligned();
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 2.0 || inner.width < 4.0 {
        return;
    }

    let disk_nodes: Vec<TreemapNode> = app.disks.iter().filter_map(build_disk_node).collect();

    if disk_nodes.is_empty() {
        canvas.draw_text(
            "No disks found",
            Point::new(inner.x + 1.0, inner.y),
            &TextStyle {
                color: Color::new(0.5, 0.5, 0.5, 1.0),
                ..Default::default()
            },
        );
        return;
    }

    let root = TreemapNode::branch("Disks", disk_nodes);
    let mut treemap = Treemap::new()
        .with_root(root)
        .with_max_depth(2)
        .with_labels(inner.width >= 8.0);

    treemap.layout(inner);
    treemap.paint(canvas);
}

/// Display item for files panel
struct FilesDisplayItem {
    name: String,
    size: u64,
    is_dir: bool,
    ratio: f64,
}

/// Build display items from file analyzer data
fn build_file_items_from_analyzer(
    fd: &crate::ptop::analyzers::FileAnalyzerData,
    max_rows: usize,
) -> Vec<FilesDisplayItem> {
    let max_size = fd
        .hot_files
        .iter()
        .map(|f| f.size)
        .max()
        .unwrap_or(1)
        .max(1);
    fd.hot_files
        .iter()
        .take(max_rows)
        .map(|f| FilesDisplayItem {
            name: f
                .path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            size: f.size,
            is_dir: f.path.is_dir(),
            ratio: (f.size as f64 / max_size as f64).min(1.0),
        })
        .collect()
}

/// Build display items from treemap data
fn build_file_items_from_treemap(
    td: &crate::ptop::analyzers::TreemapData,
    max_rows: usize,
) -> Vec<FilesDisplayItem> {
    let max_size = td.top_items.first().map_or(1, |i| i.size).max(1);
    td.top_items
        .iter()
        .take(max_rows)
        .map(|i| FilesDisplayItem {
            name: i.name.clone(),
            size: i.size,
            is_dir: i.is_dir,
            ratio: (i.size as f64 / max_size as f64).min(1.0),
        })
        .collect()
}

/// Draw a single file row in files panel
#[allow(clippy::too_many_arguments)]
fn draw_file_row(
    canvas: &mut dyn Canvas,
    x: f32,
    y: f32,
    item: &FilesDisplayItem,
    name_width: usize,
    bar_width: usize,
    file_color: Color,
    dir_color: Color,
    dim_color: Color,
) {
    use crate::widgets::display_rules::{format_column, ColumnAlign, TruncateStrategy};

    let item_color = if item.is_dir { dir_color } else { file_color };
    let name = format_column(
        &item.name,
        name_width,
        ColumnAlign::Left,
        TruncateStrategy::Path,
    );
    let size_str = format_column(
        &format_bytes(item.size),
        7,
        ColumnAlign::Right,
        TruncateStrategy::End,
    );
    let bar = make_bar(item.ratio, bar_width);

    canvas.draw_text(
        &name,
        Point::new(x, y),
        &TextStyle {
            color: item_color,
            ..Default::default()
        },
    );
    canvas.draw_text(
        &size_str,
        Point::new(x + name_width as f32, y),
        &TextStyle {
            color: dim_color,
            ..Default::default()
        },
    );
    let bar_color = Color::new(
        0.4 + 0.4 * item.ratio as f32,
        0.6 - 0.3 * item.ratio as f32,
        0.3,
        1.0,
    );
    canvas.draw_text(
        &format!("  {}", bar),
        Point::new(x + name_width as f32 + 7.0, y),
        &TextStyle {
            color: bar_color,
            ..Default::default()
        },
    );
}

/// Compute total file size and count from app state.
#[inline]
fn compute_files_totals(app: &App) -> (u64, u32) {
    if let Some(t) = app.snapshot_treemap.as_ref() {
        (t.total_size, t.total_files)
    } else {
        let total_size = app.disks.iter().map(sysinfo::Disk::total_space).sum();
        let file_count = app
            .snapshot_file_analyzer
            .as_ref()
            .map_or(0, |f| f.total_open_files as u32);
        (total_size, file_count)
    }
}

/// Build file items from available data sources.
fn build_files_items(
    file_data: Option<&crate::ptop::analyzers::FileAnalyzerData>,
    treemap_data: Option<&crate::ptop::analyzers::TreemapData>,
    max_rows: usize,
) -> Vec<FilesDisplayItem> {
    file_data
        .map(|fd| build_file_items_from_analyzer(fd, max_rows))
        .or_else(|| treemap_data.map(|td| build_file_items_from_treemap(td, max_rows)))
        .unwrap_or_default()
}

/// Get message for empty files state.
#[inline]
fn files_empty_message(has_file_data: bool, has_treemap_data: bool) -> &'static str {
    if !has_file_data && !has_treemap_data {
        "Scanning filesystem..."
    } else {
        "No files found"
    }
}

/// F014: Files Panel - Tufte-style file/directory visualization
pub(super) fn draw_files_panel(app: &App, canvas: &mut DirectTerminalCanvas<'_>, bounds: Rect) {
    use crate::widgets::display_rules::{format_column, ColumnAlign, TruncateStrategy};

    let file_data = app.snapshot_file_analyzer.as_ref();
    let treemap_data = app.snapshot_treemap.as_ref();
    let disk_entropy = app.snapshot_disk_entropy.as_ref();

    let (total_size, file_count) = compute_files_totals(app);

    let encryption_indicator = if disk_entropy.map_or(0, |d| d.encrypted_count) > 0 {
        "🔒"
    } else {
        ""
    };
    let title = format!(
        "Files │ {} {} │ {} files",
        format_bytes(total_size),
        encryption_indicator,
        file_count
    );

    let is_focused = app.is_panel_focused(PanelType::Files);
    let mut border = create_panel_border(&title, FILES_COLOR, is_focused);
    border.layout(bounds);
    border.paint(canvas);
    let inner = border.inner_rect();

    if inner.height < 1.0 {
        return;
    }

    let dim_color = Color::new(0.5, 0.5, 0.5, 1.0);
    let file_color = Color::new(0.7, 0.7, 0.5, 1.0);
    let dir_color = Color::new(0.5, 0.7, 0.9, 1.0);
    let bg_color = Color::new(0.05, 0.05, 0.07, 1.0);

    canvas.fill_rect(inner, bg_color);

    if app.deterministic {
        canvas.draw_text(
            "...",
            Point::new(inner.x, inner.y),
            &TextStyle {
                color: dim_color,
                ..Default::default()
            },
        );
        return;
    }

    let width = inner.width as usize;
    let bar_width = 12.min(width.saturating_sub(20));
    let name_width = width.saturating_sub(bar_width + 10);
    let header = format!(
        "{}{}  {}",
        format_column("NAME", name_width, ColumnAlign::Left, TruncateStrategy::End),
        format_column("SIZE", 7, ColumnAlign::Right, TruncateStrategy::End),
        format_column("%", bar_width, ColumnAlign::Left, TruncateStrategy::End)
    );
    canvas.draw_text(
        &header,
        Point::new(inner.x, inner.y),
        &TextStyle {
            color: FILES_COLOR,
            ..Default::default()
        },
    );

    let max_rows = inner.height as usize;
    let items = build_files_items(file_data, treemap_data, max_rows);

    if items.is_empty() {
        let msg = files_empty_message(file_data.is_some(), treemap_data.is_some());
        canvas.draw_text(
            msg,
            Point::new(inner.x, inner.y + 1.0),
            &TextStyle {
                color: dim_color,
                ..Default::default()
            },
        );
        return;
    }

    for (i, item) in items
        .iter()
        .take((inner.height as usize).saturating_sub(1))
        .enumerate()
    {
        let y = inner.y + 1.0 + i as f32;
        if y >= inner.y + inner.height {
            break;
        }
        draw_file_row(
            canvas, inner.x, y, item, name_width, bar_width, file_color, dir_color, dim_color,
        );
    }
}
