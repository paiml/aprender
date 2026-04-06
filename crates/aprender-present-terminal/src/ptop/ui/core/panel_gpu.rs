//! GPU Panel Rendering Components
//!
//! Extracted from render.rs to reduce cyclomatic complexity.
//! Contains helper functions for GPU panel colors and formatting.

#![allow(dead_code)]

use presentar_core::Color;

/// Get color for GPU temperature.
///
/// - Red: temp > 85°C (hot)
/// - Yellow: temp > 70°C (warm)
/// - Green: temp <= 70°C (cool)
#[must_use]
pub(crate) fn gpu_temp_color(temp: u32) -> Color {
    if temp > 85 {
        Color::new(1.0, 0.3, 0.3, 1.0) // Red - hot
    } else if temp > 70 {
        Color::new(1.0, 0.8, 0.2, 1.0) // Yellow - warm
    } else {
        Color::new(0.3, 0.9, 0.3, 1.0) // Green - cool
    }
}

/// Get badge character and color for GPU process type.
///
/// - C (Cyan): Compute workloads
/// - G (Magenta): Graphics workloads
/// - ? (Gray): Unknown type
#[must_use]
pub(crate) fn gpu_proc_badge(process_type: &str) -> (&'static str, Color) {
    match process_type.to_uppercase().as_str() {
        "C" | "COMPUTE" => ("C", Color::new(0.0, 0.8, 1.0, 1.0)), // Cyan
        "G" | "GRAPHICS" => ("G", Color::new(1.0, 0.0, 1.0, 1.0)), // Magenta
        _ => ("?", Color::new(0.5, 0.5, 0.5, 1.0)),               // Gray
    }
}

/// Standard gray color for power display.
pub(crate) const POWER_COLOR: Color = Color {
    r: 0.7,
    g: 0.7,
    b: 0.7,
    a: 1.0,
};

/// Gray color for labels/headers.
pub(crate) const HEADER_COLOR: Color = Color {
    r: 0.5,
    g: 0.5,
    b: 0.5,
    a: 1.0,
};

/// Light gray for process info text.
pub(crate) const PROC_INFO_COLOR: Color = Color {
    r: 0.8,
    g: 0.8,
    b: 0.8,
    a: 1.0,
};

/// VRAM history graph color (purple).
pub(crate) const VRAM_GRAPH_COLOR: Color = Color {
    r: 0.6,
    g: 0.4,
    b: 1.0,
    a: 1.0,
};

/// Build GPU utilization bar string.
#[must_use]
pub(crate) fn build_gpu_bar(percent: f32, bar_width: usize) -> String {
    let filled = (((percent / 100.0) * bar_width as f32) as usize).min(bar_width);
    "█".repeat(filled) + &"░".repeat(bar_width.saturating_sub(filled))
}

/// Format GPU utilization row.
#[must_use]
pub(crate) fn format_gpu_row(label: &str, bar: &str, percent: f32) -> String {
    format!("{label}  {bar} {:>3}%", percent as u32)
}

/// Format VRAM usage row.
#[must_use]
pub(crate) fn format_vram_row(bar: &str, used_mb: u64, total_mb: u64) -> String {
    format!("VRAM {bar} {used_mb}M/{total_mb}M")
}

/// Format process utilization value (with "-" for None).
#[must_use]
pub(crate) fn format_proc_util(util: Option<f32>) -> String {
    util.map_or_else(|| "  -".to_string(), |u| format!("{u:>3.0}"))
}

/// Truncate process name to max length.
#[must_use]
pub(crate) fn truncate_name(name: &str, max_len: usize) -> &str {
    if name.len() > max_len {
        &name[..max_len]
    } else {
        name
    }
}

/// Build GPU panel title with optional temperature and power.
#[must_use]
pub(crate) fn build_gpu_title(
    name: &str,
    temp: Option<u32>,
    power: Option<f32>,
    minimal: bool,
) -> String {
    if minimal {
        name.to_string()
    } else {
        let temp_str = temp.map(|t| format!(" │ {t}°C")).unwrap_or_default();
        let power_str = power.map(|p| format!(" │ {p:.0}W")).unwrap_or_default();
        format!("{name}{temp_str}{power_str}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // F-GPU-001: Temperature color red when hot
    #[test]
    fn test_temp_color_red() {
        let color = gpu_temp_color(90);
        assert!(color.r > 0.8 && color.g < 0.5, "Should be red above 85°C");
    }

    // F-GPU-002: Temperature color yellow when warm
    #[test]
    fn test_temp_color_yellow() {
        let color = gpu_temp_color(75);
        assert!(color.r > 0.8 && color.g > 0.5, "Should be yellow 70-85°C");
    }

    // F-GPU-003: Temperature color green when cool
    #[test]
    fn test_temp_color_green() {
        let color = gpu_temp_color(60);
        assert!(color.g > 0.8 && color.r < 0.5, "Should be green below 70°C");
    }

    // F-GPU-004: Process badge C for Compute
    #[test]
    fn test_proc_badge_compute() {
        let (badge, color) = gpu_proc_badge("C");
        assert_eq!(badge, "C");
        assert!(color.b > 0.8, "Compute should be cyan");
    }

    // F-GPU-005: Process badge C for COMPUTE
    #[test]
    fn test_proc_badge_compute_full() {
        let (badge, _) = gpu_proc_badge("COMPUTE");
        assert_eq!(badge, "C");
    }

    // F-GPU-006: Process badge G for Graphics
    #[test]
    fn test_proc_badge_graphics() {
        let (badge, color) = gpu_proc_badge("G");
        assert_eq!(badge, "G");
        assert!(color.r > 0.8 && color.b > 0.8, "Graphics should be magenta");
    }

    // F-GPU-007: Process badge G for GRAPHICS
    #[test]
    fn test_proc_badge_graphics_full() {
        let (badge, _) = gpu_proc_badge("GRAPHICS");
        assert_eq!(badge, "G");
    }

    // F-GPU-008: Process badge ? for unknown
    #[test]
    fn test_proc_badge_unknown() {
        let (badge, color) = gpu_proc_badge("unknown");
        assert_eq!(badge, "?");
        assert!(
            color.r == color.g && color.g == color.b,
            "Unknown should be gray"
        );
    }

    // F-GPU-009: Power color is gray
    #[test]
    fn test_power_color() {
        assert!((POWER_COLOR.r - POWER_COLOR.g).abs() < 0.01);
        assert!((POWER_COLOR.g - POWER_COLOR.b).abs() < 0.01);
    }

    // F-GPU-010: Header color is gray
    #[test]
    fn test_header_color() {
        assert!(HEADER_COLOR.r == HEADER_COLOR.g && HEADER_COLOR.g == HEADER_COLOR.b);
    }

    // F-GPU-011: Proc info color is light gray
    #[test]
    fn test_proc_info_color() {
        assert!(PROC_INFO_COLOR.r > 0.7);
        assert!(PROC_INFO_COLOR.g > 0.7);
        assert!(PROC_INFO_COLOR.b > 0.7);
    }

    // F-GPU-012: VRAM graph color is purple
    #[test]
    fn test_vram_graph_color() {
        assert!(VRAM_GRAPH_COLOR.r > 0.5 && VRAM_GRAPH_COLOR.b > 0.8);
    }

    // F-GPU-013: GPU bar full
    #[test]
    fn test_gpu_bar_full() {
        let bar = build_gpu_bar(100.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 10);
    }

    // F-GPU-014: GPU bar empty
    #[test]
    fn test_gpu_bar_empty() {
        let bar = build_gpu_bar(0.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '░').count(), 10);
    }

    // F-GPU-015: GPU bar half
    #[test]
    fn test_gpu_bar_half() {
        let bar = build_gpu_bar(50.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 5);
    }

    // F-GPU-016: Format GPU row
    #[test]
    fn test_format_gpu_row() {
        let bar = build_gpu_bar(75.0, 10);
        let row = format_gpu_row("GPU", &bar, 75.0);
        assert!(row.contains("GPU"));
        assert!(row.contains("75%"));
    }

    // F-GPU-017: Format VRAM row
    #[test]
    fn test_format_vram_row() {
        let bar = build_gpu_bar(50.0, 10);
        let row = format_vram_row(&bar, 4096, 8192);
        assert!(row.contains("VRAM"));
        assert!(row.contains("4096M"));
        assert!(row.contains("8192M"));
    }

    // F-GPU-018: Format proc util with value
    #[test]
    fn test_format_proc_util_value() {
        let s = format_proc_util(Some(75.0));
        assert!(s.contains("75"));
    }

    // F-GPU-019: Format proc util with None
    #[test]
    fn test_format_proc_util_none() {
        let s = format_proc_util(None);
        assert!(s.contains("-"));
    }

    // F-GPU-020: Truncate name short
    #[test]
    fn test_truncate_name_short() {
        let name = truncate_name("firefox", 12);
        assert_eq!(name, "firefox");
    }

    // F-GPU-021: Truncate name long
    #[test]
    fn test_truncate_name_long() {
        let name = truncate_name("verylongprocessname", 12);
        assert_eq!(name.len(), 12);
    }

    // F-GPU-022: Build GPU title minimal
    #[test]
    fn test_build_title_minimal() {
        let title = build_gpu_title("GTX 1080", Some(65), Some(150.0), true);
        assert_eq!(title, "GTX 1080");
    }

    // F-GPU-023: Build GPU title full
    #[test]
    fn test_build_title_full() {
        let title = build_gpu_title("RTX 3080", Some(70), Some(200.0), false);
        assert!(title.contains("RTX 3080"));
        assert!(title.contains("70°C"));
        assert!(title.contains("200W"));
    }

    // F-GPU-024: Build GPU title no temp
    #[test]
    fn test_build_title_no_temp() {
        let title = build_gpu_title("GPU", None, Some(100.0), false);
        assert!(!title.contains("°C"));
        assert!(title.contains("100W"));
    }

    // F-GPU-025: Build GPU title no power
    #[test]
    fn test_build_title_no_power() {
        let title = build_gpu_title("GPU", Some(50), None, false);
        assert!(title.contains("50°C"));
        assert!(!title.contains("W"));
    }

    // F-GPU-026: Temperature boundary at 85
    #[test]
    fn test_temp_boundary_85() {
        let below = gpu_temp_color(84);
        let above = gpu_temp_color(86);
        assert!(above.r > below.r || below.g > above.g);
    }

    // F-GPU-027: Temperature boundary at 70
    #[test]
    fn test_temp_boundary_70() {
        let below = gpu_temp_color(69);
        let above = gpu_temp_color(71);
        assert!(above.r > below.r);
    }

    // F-GPU-028: GPU bar handles overflow (capped at full)
    #[test]
    fn test_gpu_bar_overflow() {
        let bar = build_gpu_bar(150.0, 10);
        // With 150%, filled is capped to bar_width (10)
        assert_eq!(bar.chars().count(), 10, "Bar should not overflow width");
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 10);
    }

    // F-GPU-029: Process badge case insensitive
    #[test]
    fn test_proc_badge_case() {
        let (badge1, _) = gpu_proc_badge("compute");
        let (badge2, _) = gpu_proc_badge("COMPUTE");
        assert_eq!(badge1, badge2);
    }

    // F-GPU-030: Temperature zero is green
    #[test]
    fn test_temp_zero() {
        let color = gpu_temp_color(0);
        assert!(color.g > 0.8);
    }
}
