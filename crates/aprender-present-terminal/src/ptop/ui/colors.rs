//! Panel color constants for ptop UI.
//!
//! Exact RGB values from ttop theme.rs for pixel-perfect matching.

use presentar_core::Color;

// Re-export selection colors from framework
pub use crate::widgets::selection::{SELECTION_ACCENT, SELECTION_BG};

// =============================================================================
// PANEL BORDER COLORS (ttop exact RGB values)
// =============================================================================

/// CPU panel color - #64C8FF (100,200,255)
pub const CPU_COLOR: Color = Color {
    r: 0.392,
    g: 0.784,
    b: 1.0,
    a: 1.0,
};

/// Memory panel color - #B478FF (180,120,255)
pub const MEMORY_COLOR: Color = Color {
    r: 0.706,
    g: 0.471,
    b: 1.0,
    a: 1.0,
};

/// Disk panel color - #64B4FF (100,180,255)
pub const DISK_COLOR: Color = Color {
    r: 0.392,
    g: 0.706,
    b: 1.0,
    a: 1.0,
};

/// Network panel color - #FF9664 (255,150,100)
pub const NETWORK_COLOR: Color = Color {
    r: 1.0,
    g: 0.588,
    b: 0.392,
    a: 1.0,
};

/// Process panel color - #DCC464 (220,180,100)
pub const PROCESS_COLOR: Color = Color {
    r: 0.863,
    g: 0.706,
    b: 0.392,
    a: 1.0,
};

/// GPU panel color - #64FF96 (100,255,150)
pub const GPU_COLOR: Color = Color {
    r: 0.392,
    g: 1.0,
    b: 0.588,
    a: 1.0,
};

/// Battery panel color - #FFDC64 (255,220,100)
pub const BATTERY_COLOR: Color = Color {
    r: 1.0,
    g: 0.863,
    b: 0.392,
    a: 1.0,
};

/// Sensors panel color - #FF6496 (255,100,150)
pub const SENSORS_COLOR: Color = Color {
    r: 1.0,
    g: 0.392,
    b: 0.588,
    a: 1.0,
};

/// PSI panel color - #C85050 (200,80,80)
pub const PSI_COLOR: Color = Color {
    r: 0.784,
    g: 0.314,
    b: 0.314,
    a: 1.0,
};

/// Connections panel color - #78B4DC (120,180,220)
pub const CONNECTIONS_COLOR: Color = Color {
    r: 0.471,
    g: 0.706,
    b: 0.863,
    a: 1.0,
};

/// Files panel color - #B48C64 (180,140,100)
pub const FILES_COLOR: Color = Color {
    r: 0.706,
    g: 0.549,
    b: 0.392,
    a: 1.0,
};

/// Containers panel color - #64B4DC (100,180,220) - Docker blue
pub const CONTAINERS_COLOR: Color = Color {
    r: 0.392,
    g: 0.706,
    b: 0.863,
    a: 1.0,
};

// =============================================================================
// NETWORK GRAPH COLORS
// =============================================================================

/// Network RX (download) color - Cyan
pub const NET_RX_COLOR: Color = Color {
    r: 0.392,
    g: 0.784,
    b: 1.0,
    a: 1.0,
};

/// Network TX (upload) color - Red
pub const NET_TX_COLOR: Color = Color {
    r: 1.0,
    g: 0.392,
    b: 0.392,
    a: 1.0,
};

// =============================================================================
// SELECTION AND FOCUS COLORS
// =============================================================================

/// Focus accent color (bright green) - re-export from framework
pub const FOCUS_ACCENT_COLOR: Color = SELECTION_ACCENT;

/// Row selection background - re-export from framework
pub const ROW_SELECT_BG: Color = SELECTION_BG;

/// Column header selection background (slightly different from row)
pub const COL_SELECT_BG: Color = Color {
    r: 0.15,
    g: 0.4,
    b: 0.65,
    a: 1.0,
};

/// Status bar background
pub const STATUS_BAR_BG: Color = Color {
    r: 0.08,
    g: 0.08,
    b: 0.12,
    a: 1.0,
};

// =============================================================================
// PERCENTAGE-BASED COLORS
// =============================================================================

/// btop-style color gradient for percentage values (0-100).
/// Uses smooth transition: cyan -> green -> yellow -> orange -> red.
///
/// # Arguments
/// * `percent` - Value from 0.0 to 100.0 (clamped if outside range)
///
/// # Returns
/// Color interpolated based on percentage value
#[must_use]
pub fn percent_color(percent: f64) -> Color {
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

/// Color for swap usage percentage.
/// Green (low) -> Yellow (medium) -> Red (high)
#[must_use]
pub fn swap_color(pct: f64) -> Color {
    let p = pct.clamp(0.0, 100.0);

    if p >= 80.0 {
        // Critical: red
        Color {
            r: 1.0,
            g: 0.3,
            b: 0.3,
            a: 1.0,
        }
    } else if p >= 50.0 {
        // Warning: orange/yellow
        let t = (p - 50.0) / 30.0;
        Color {
            r: 1.0,
            g: (0.8 - t * 0.5) as f32,
            b: 0.3,
            a: 1.0,
        }
    } else if p >= 25.0 {
        // Medium: yellow-green
        let t = (p - 25.0) / 25.0;
        Color {
            r: (0.4 + t * 0.6) as f32,
            g: 0.8,
            b: 0.3,
            a: 1.0,
        }
    } else {
        // Low: green
        Color {
            r: 0.4,
            g: 0.8,
            b: 0.4,
            a: 1.0,
        }
    }
}

/// Color for PSI (Pressure Stall Information) percentage.
#[must_use]
pub fn pressure_color(pct: f64) -> Color {
    let p = pct.clamp(0.0, 100.0);

    if p <= 1.0 {
        // No pressure: dim gray
        Color {
            r: 0.4,
            g: 0.4,
            b: 0.4,
            a: 1.0,
        }
    } else if p <= 5.0 {
        // Low: green
        Color {
            r: 0.4,
            g: 0.8,
            b: 0.4,
            a: 1.0,
        }
    } else if p <= 20.0 {
        // Medium: yellow
        Color {
            r: 1.0,
            g: 0.8,
            b: 0.3,
            a: 1.0,
        }
    } else if p <= 50.0 {
        // High: orange
        Color {
            r: 1.0,
            g: 0.5,
            b: 0.2,
            a: 1.0,
        }
    } else {
        // Critical: red
        Color {
            r: 1.0,
            g: 0.3,
            b: 0.3,
            a: 1.0,
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Panel color constant tests
    // =========================================================================

    #[test]
    fn test_cpu_color_is_cyan() {
        assert!(CPU_COLOR.b > 0.9, "CPU should be cyan (high blue)");
        assert!(CPU_COLOR.g > 0.7, "CPU should be cyan (high green)");
        assert!(CPU_COLOR.r < 0.5, "CPU should be cyan (low red)");
    }

    #[test]
    fn test_memory_color_is_purple() {
        assert!(MEMORY_COLOR.b > 0.9, "Memory should be purple (high blue)");
        assert!(
            MEMORY_COLOR.r > 0.6,
            "Memory should be purple (medium-high red)"
        );
        assert!(MEMORY_COLOR.g < 0.5, "Memory should be purple (low green)");
    }

    #[test]
    fn test_disk_color_is_light_blue() {
        assert!(DISK_COLOR.b > 0.9, "Disk should be light blue");
        assert!(DISK_COLOR.g > 0.6, "Disk should have medium green");
    }

    #[test]
    fn test_network_color_is_orange() {
        assert!(NETWORK_COLOR.r > 0.9, "Network should be orange (high red)");
        assert!(
            NETWORK_COLOR.g > 0.5,
            "Network should be orange (medium green)"
        );
        assert!(NETWORK_COLOR.b < 0.5, "Network should be orange (low blue)");
    }

    #[test]
    fn test_process_color_is_yellow() {
        assert!(PROCESS_COLOR.r > 0.8, "Process should be yellow (high red)");
        assert!(
            PROCESS_COLOR.g > 0.6,
            "Process should be yellow (medium-high green)"
        );
    }

    #[test]
    fn test_gpu_color_is_green() {
        assert!(GPU_COLOR.g > 0.9, "GPU should be green (high green)");
        assert!(GPU_COLOR.b > 0.5, "GPU should have some blue/cyan tint");
    }

    #[test]
    fn test_battery_color_is_yellow_gold() {
        assert!(BATTERY_COLOR.r > 0.9, "Battery should be yellow/gold");
        assert!(BATTERY_COLOR.g > 0.8, "Battery should be yellow/gold");
    }

    #[test]
    fn test_sensors_color_is_pink() {
        assert!(SENSORS_COLOR.r > 0.9, "Sensors should be pink (high red)");
        assert!(
            SENSORS_COLOR.b > 0.5,
            "Sensors should be pink (medium blue)"
        );
    }

    #[test]
    fn test_psi_color_is_red() {
        assert!(PSI_COLOR.r > 0.7, "PSI should be red");
        assert!(PSI_COLOR.g < 0.4, "PSI should be red (low green)");
    }

    #[test]
    fn test_connections_color_is_light_blue() {
        assert!(
            CONNECTIONS_COLOR.b > 0.8,
            "Connections should be light blue"
        );
    }

    #[test]
    fn test_files_color_is_brown() {
        assert!(FILES_COLOR.r > 0.6, "Files should be brown");
        assert!(FILES_COLOR.g > 0.4, "Files should be brown");
        assert!(FILES_COLOR.b < 0.5, "Files should be brown");
    }

    #[test]
    fn test_containers_color_is_docker_blue() {
        assert!(CONTAINERS_COLOR.b > 0.8, "Containers should be Docker blue");
    }

    // =========================================================================
    // Network graph color tests
    // =========================================================================

    #[test]
    fn test_net_rx_color_is_cyan() {
        assert!(NET_RX_COLOR.b > 0.9);
        assert!(NET_RX_COLOR.g > 0.7);
    }

    #[test]
    fn test_net_tx_color_is_red() {
        assert!(NET_TX_COLOR.r > 0.9);
        assert!(NET_TX_COLOR.b < 0.5);
    }

    // =========================================================================
    // Selection color tests
    // =========================================================================

    #[test]
    fn test_status_bar_bg_is_dark() {
        assert!(STATUS_BAR_BG.r < 0.15);
        assert!(STATUS_BAR_BG.g < 0.15);
        assert!(STATUS_BAR_BG.b < 0.15);
    }

    #[test]
    fn test_col_select_bg_is_blue() {
        assert!(COL_SELECT_BG.b > COL_SELECT_BG.r);
        assert!(COL_SELECT_BG.b > COL_SELECT_BG.g);
    }

    // =========================================================================
    // percent_color tests
    // =========================================================================

    #[test]
    fn test_percent_color_zero() {
        let color = percent_color(0.0);
        // Should be cyan-ish (low usage)
        assert!(color.b > 0.8, "0% should have high blue (cyan)");
        assert!(color.g > 0.6, "0% should have medium-high green");
    }

    #[test]
    fn test_percent_color_low() {
        let color = percent_color(10.0);
        assert!(color.b > 0.5, "Low should be blue/cyan-ish");
        assert!(color.g > 0.5, "Low should have green component");
    }

    #[test]
    fn test_percent_color_medium_low() {
        let color = percent_color(35.0);
        assert!(color.g > 0.7, "Medium-low should be greenish");
    }

    #[test]
    fn test_percent_color_medium() {
        let color = percent_color(50.0);
        assert_eq!(color.r, 1.0, "50% should have full red (yellow)");
    }

    #[test]
    fn test_percent_color_medium_high() {
        let color = percent_color(60.0);
        assert!(color.r > 0.7, "Medium-high should have high red");
        assert!(color.g > 0.5, "Medium-high should have some green");
    }

    #[test]
    fn test_percent_color_high() {
        let color = percent_color(80.0);
        assert_eq!(color.r, 1.0, "High should have full red");
    }

    #[test]
    fn test_percent_color_critical() {
        let color = percent_color(95.0);
        assert_eq!(color.r, 1.0, "Critical should be full red");
        assert!(color.g < 0.3, "Critical should have low green");
    }

    #[test]
    fn test_percent_color_max() {
        let color = percent_color(100.0);
        assert_eq!(color.r, 1.0, "100% should be full red");
        assert_eq!(color.g, 0.25, "100% should have specific green");
    }

    #[test]
    fn test_percent_color_clamped_negative() {
        let neg = percent_color(-10.0);
        let zero = percent_color(0.0);
        assert_eq!(neg.r, zero.r, "Negative should clamp to 0");
        assert_eq!(neg.g, zero.g);
        assert_eq!(neg.b, zero.b);
    }

    #[test]
    fn test_percent_color_clamped_over() {
        let over = percent_color(150.0);
        let hundred = percent_color(100.0);
        assert_eq!(over.r, hundred.r, "Over 100 should clamp to 100");
        assert_eq!(over.g, hundred.g);
        assert_eq!(over.b, hundred.b);
    }

    #[test]
    fn test_percent_color_boundary_25() {
        let color = percent_color(25.0);
        assert!(color.g > 0.8, "25% boundary should be green-ish");
    }

    #[test]
    fn test_percent_color_boundary_75() {
        let color = percent_color(75.0);
        assert_eq!(color.r, 1.0, "75% should have full red");
    }

    #[test]
    fn test_percent_color_boundary_90() {
        let color = percent_color(90.0);
        assert_eq!(color.r, 1.0);
        assert_eq!(color.g, 0.25);
    }

    // =========================================================================
    // swap_color tests
    // =========================================================================

    #[test]
    fn test_swap_color_low() {
        let color = swap_color(10.0);
        assert!(color.g > 0.7, "Low swap should be green");
        assert!(color.r < 0.5, "Low swap should have low red");
    }

    #[test]
    fn test_swap_color_medium() {
        let color = swap_color(40.0);
        assert!(color.g > 0.7, "Medium swap should be yellow-green");
    }

    #[test]
    fn test_swap_color_high() {
        let color = swap_color(80.0);
        assert!(color.r > 0.9, "High swap should be red");
        assert!(color.g < 0.4, "High swap should have low green");
    }

    #[test]
    fn test_swap_color_critical() {
        let color = swap_color(95.0);
        assert!(color.r > 0.9, "Critical swap should be red");
    }

    #[test]
    fn test_swap_color_clamped() {
        let neg = swap_color(-10.0);
        let over = swap_color(110.0);
        // Should clamp and not panic
        assert!(neg.r >= 0.0 && neg.r <= 1.0);
        assert!(over.r >= 0.0 && over.r <= 1.0);
    }

    // =========================================================================
    // pressure_color tests
    // =========================================================================

    #[test]
    fn test_pressure_color_none() {
        let color = pressure_color(0.0);
        // Should be dim gray
        assert!(color.r < 0.5);
        assert!(color.g < 0.5);
        assert!(color.b < 0.5);
    }

    #[test]
    fn test_pressure_color_low() {
        let color = pressure_color(3.0);
        // Low pressure should be green
        assert!(color.g > 0.7);
    }

    #[test]
    fn test_pressure_color_medium() {
        let color = pressure_color(15.0);
        // Medium should be yellow
        assert!(color.r > 0.9);
        assert!(color.g > 0.7);
    }

    #[test]
    fn test_pressure_color_high() {
        let color = pressure_color(35.0);
        // High should be orange
        assert!(color.r > 0.9);
        assert!(color.g > 0.4 && color.g < 0.6);
    }

    #[test]
    fn test_pressure_color_critical() {
        let color = pressure_color(60.0);
        // Critical should be red
        assert!(color.r > 0.9);
        assert!(color.g < 0.4);
    }

    // =========================================================================
    // All colors have valid alpha
    // =========================================================================

    #[test]
    fn test_all_colors_have_full_alpha() {
        assert_eq!(CPU_COLOR.a, 1.0);
        assert_eq!(MEMORY_COLOR.a, 1.0);
        assert_eq!(DISK_COLOR.a, 1.0);
        assert_eq!(NETWORK_COLOR.a, 1.0);
        assert_eq!(PROCESS_COLOR.a, 1.0);
        assert_eq!(GPU_COLOR.a, 1.0);
        assert_eq!(BATTERY_COLOR.a, 1.0);
        assert_eq!(SENSORS_COLOR.a, 1.0);
        assert_eq!(PSI_COLOR.a, 1.0);
        assert_eq!(CONNECTIONS_COLOR.a, 1.0);
        assert_eq!(FILES_COLOR.a, 1.0);
        assert_eq!(CONTAINERS_COLOR.a, 1.0);
        assert_eq!(NET_RX_COLOR.a, 1.0);
        assert_eq!(NET_TX_COLOR.a, 1.0);
        assert_eq!(COL_SELECT_BG.a, 1.0);
        assert_eq!(STATUS_BAR_BG.a, 1.0);
    }

    // =========================================================================
    // Color gradient continuity tests
    // =========================================================================

    #[test]
    fn test_percent_color_gradient_is_continuous() {
        // Test that colors change smoothly (no sudden jumps > 0.5)
        let mut prev = percent_color(0.0);
        for i in 1..=100 {
            let curr = percent_color(i as f64);
            let dr = (curr.r - prev.r).abs();
            let dg = (curr.g - prev.g).abs();
            let db = (curr.b - prev.b).abs();
            assert!(
                dr < 0.15 && dg < 0.15 && db < 0.15,
                "Color jump too large at {}%: dr={}, dg={}, db={}",
                i,
                dr,
                dg,
                db
            );
            prev = curr;
        }
    }

    #[test]
    fn test_swap_color_gradient_is_continuous() {
        let mut prev = swap_color(0.0);
        for i in 1..=100 {
            let curr = swap_color(i as f64);
            let dr = (curr.r - prev.r).abs();
            let dg = (curr.g - prev.g).abs();
            let db = (curr.b - prev.b).abs();
            assert!(
                dr < 0.15 && dg < 0.15 && db < 0.15,
                "Swap color jump too large at {}%",
                i
            );
            prev = curr;
        }
    }
}
