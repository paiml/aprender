//! Panel color constants for ptop.
//!
//! Exact RGB values matching ttop/trueno-viz theme.rs.

#![allow(dead_code)]

use presentar_core::Color;

/// CPU panel border color - #64C8FF (100,200,255)
pub const CPU_COLOR: Color = Color {
    r: 0.392,
    g: 0.784,
    b: 1.0,
    a: 1.0,
};

/// Memory panel border color - #B478FF (180,120,255)
pub const MEMORY_COLOR: Color = Color {
    r: 0.706,
    g: 0.471,
    b: 1.0,
    a: 1.0,
};

/// Disk panel border color - #64B4FF (100,180,255)
pub const DISK_COLOR: Color = Color {
    r: 0.392,
    g: 0.706,
    b: 1.0,
    a: 1.0,
};

/// Network panel border color - #FF9664 (255,150,100)
pub const NETWORK_COLOR: Color = Color {
    r: 1.0,
    g: 0.588,
    b: 0.392,
    a: 1.0,
};

/// Process panel border color - #DCC464 (220,180,100)
pub const PROCESS_COLOR: Color = Color {
    r: 0.863,
    g: 0.706,
    b: 0.392,
    a: 1.0,
};

/// GPU panel border color - #64FF96 (100,255,150)
pub const GPU_COLOR: Color = Color {
    r: 0.392,
    g: 1.0,
    b: 0.588,
    a: 1.0,
};

/// Battery panel border color - #FFDC64 (255,220,100)
pub const BATTERY_COLOR: Color = Color {
    r: 1.0,
    g: 0.863,
    b: 0.392,
    a: 1.0,
};

/// Sensors panel border color - #FF6496 (255,100,150)
pub const SENSORS_COLOR: Color = Color {
    r: 1.0,
    g: 0.392,
    b: 0.588,
    a: 1.0,
};

/// PSI panel border color - #C85050 (200,80,80)
pub const PSI_COLOR: Color = Color {
    r: 0.784,
    g: 0.314,
    b: 0.314,
    a: 1.0,
};

/// Connections panel border color - #78B4DC (120,180,220)
pub const CONNECTIONS_COLOR: Color = Color {
    r: 0.471,
    g: 0.706,
    b: 0.863,
    a: 1.0,
};

/// Files panel border color - #B48C64 (180,140,100)
pub const FILES_COLOR: Color = Color {
    r: 0.706,
    g: 0.549,
    b: 0.392,
    a: 1.0,
};

/// Containers panel border color - #64B4DC (100,180,220) - Docker blue
pub const CONTAINERS_COLOR: Color = Color {
    r: 0.392,
    g: 0.706,
    b: 0.863,
    a: 1.0,
};

/// Network RX (receive) color - green
pub const NET_RX_COLOR: Color = Color {
    r: 0.0,
    g: 0.8,
    b: 0.4,
    a: 1.0,
};

/// Network TX (transmit) color - magenta
pub const NET_TX_COLOR: Color = Color {
    r: 0.8,
    g: 0.4,
    b: 0.8,
    a: 1.0,
};

/// Disk read color - cyan
pub const DISK_READ_COLOR: Color = Color {
    r: 0.0,
    g: 0.8,
    b: 0.8,
    a: 1.0,
};

/// Disk write color - orange
pub const DISK_WRITE_COLOR: Color = Color {
    r: 1.0,
    g: 0.6,
    b: 0.2,
    a: 1.0,
};

/// White color for text
pub const WHITE: Color = Color {
    r: 1.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};

/// Gray color for dimmed text
pub const GRAY: Color = Color {
    r: 0.5,
    g: 0.5,
    b: 0.5,
    a: 1.0,
};

/// Dark gray for backgrounds
pub const DARK_GRAY: Color = Color {
    r: 0.2,
    g: 0.2,
    b: 0.2,
    a: 1.0,
};

/// Black color
pub const BLACK: Color = Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};

/// Green for healthy/good states
pub const GREEN: Color = Color {
    r: 0.0,
    g: 0.8,
    b: 0.0,
    a: 1.0,
};

/// Yellow for warning states
pub const YELLOW: Color = Color {
    r: 1.0,
    g: 0.8,
    b: 0.0,
    a: 1.0,
};

/// Red for critical/error states
pub const RED: Color = Color {
    r: 1.0,
    g: 0.2,
    b: 0.2,
    a: 1.0,
};

/// Cyan for information
pub const CYAN: Color = Color {
    r: 0.0,
    g: 0.8,
    b: 0.8,
    a: 1.0,
};

/// Magenta for special highlights
pub const MAGENTA: Color = Color {
    r: 0.8,
    g: 0.4,
    b: 0.8,
    a: 1.0,
};

/// Selection highlight color
pub const SELECTION_BG: Color = Color {
    r: 0.2,
    g: 0.3,
    b: 0.5,
    a: 1.0,
};

/// Header background color
pub const HEADER_BG: Color = Color {
    r: 0.15,
    g: 0.15,
    b: 0.2,
    a: 1.0,
};

#[cfg(test)]
mod tests {
    use super::*;

    // F-CONST-001: CPU color matches ttop exactly
    #[test]
    fn test_cpu_color_matches_ttop() {
        assert!((CPU_COLOR.r - 0.392).abs() < 0.001);
        assert!((CPU_COLOR.g - 0.784).abs() < 0.001);
        assert!((CPU_COLOR.b - 1.0).abs() < 0.001);
    }

    // F-CONST-002: Memory color matches ttop exactly
    #[test]
    fn test_memory_color_matches_ttop() {
        assert!((MEMORY_COLOR.r - 0.706).abs() < 0.001);
        assert!((MEMORY_COLOR.g - 0.471).abs() < 0.001);
        assert!((MEMORY_COLOR.b - 1.0).abs() < 0.001);
    }

    // F-CONST-003: All panel colors have full alpha
    #[test]
    fn test_all_panel_colors_full_alpha() {
        let colors = [
            CPU_COLOR,
            MEMORY_COLOR,
            DISK_COLOR,
            NETWORK_COLOR,
            PROCESS_COLOR,
            GPU_COLOR,
            BATTERY_COLOR,
            SENSORS_COLOR,
            PSI_COLOR,
            CONNECTIONS_COLOR,
            FILES_COLOR,
            CONTAINERS_COLOR,
        ];
        for color in colors {
            assert!(
                (color.a - 1.0).abs() < 0.001,
                "Panel color must have full alpha"
            );
        }
    }

    // F-CONST-004: Network colors are distinct
    #[test]
    fn test_network_colors_distinct() {
        assert!(
            (NET_RX_COLOR.r - NET_TX_COLOR.r).abs() > 0.1
                || (NET_RX_COLOR.g - NET_TX_COLOR.g).abs() > 0.1
                || (NET_RX_COLOR.b - NET_TX_COLOR.b).abs() > 0.1,
            "RX and TX colors must be visually distinct"
        );
    }

    // F-CONST-005: Disk colors are distinct
    #[test]
    fn test_disk_colors_distinct() {
        assert!(
            (DISK_READ_COLOR.r - DISK_WRITE_COLOR.r).abs() > 0.1
                || (DISK_READ_COLOR.g - DISK_WRITE_COLOR.g).abs() > 0.1
                || (DISK_READ_COLOR.b - DISK_WRITE_COLOR.b).abs() > 0.1,
            "Read and Write colors must be visually distinct"
        );
    }

    // F-CONST-006: Status colors follow traffic light convention
    #[test]
    fn test_status_colors_traffic_light() {
        // Green should be mostly green
        assert!(GREEN.g > GREEN.r && GREEN.g > GREEN.b);
        // Yellow should be red + green dominant
        assert!(YELLOW.r > 0.5 && YELLOW.g > 0.5 && YELLOW.b < 0.5);
        // Red should be mostly red
        assert!(RED.r > RED.g && RED.r > RED.b);
    }

    // F-CONST-007: White is actually white
    #[test]
    fn test_white_is_white() {
        assert!((WHITE.r - 1.0).abs() < 0.001);
        assert!((WHITE.g - 1.0).abs() < 0.001);
        assert!((WHITE.b - 1.0).abs() < 0.001);
    }

    // F-CONST-008: Black is actually black
    #[test]
    fn test_black_is_black() {
        assert!(BLACK.r.abs() < 0.001);
        assert!(BLACK.g.abs() < 0.001);
        assert!(BLACK.b.abs() < 0.001);
    }

    // F-CONST-009: Gray is balanced
    #[test]
    fn test_gray_is_balanced() {
        assert!((GRAY.r - GRAY.g).abs() < 0.001);
        assert!((GRAY.g - GRAY.b).abs() < 0.001);
    }

    // F-CONST-010: Selection background is distinguishable
    #[test]
    fn test_selection_bg_distinguishable() {
        // Selection should be darker than white but lighter than black
        let avg = (SELECTION_BG.r + SELECTION_BG.g + SELECTION_BG.b) / 3.0;
        assert!(avg > 0.1 && avg < 0.9, "Selection BG should be mid-tone");
    }

    // F-CONST-011: Hex conversion CPU color
    #[test]
    fn test_cpu_color_hex() {
        // #64C8FF = (100,200,255)
        let r = (CPU_COLOR.r * 255.0).round() as u8;
        let g = (CPU_COLOR.g * 255.0).round() as u8;
        let b = (CPU_COLOR.b * 255.0).round() as u8;
        assert_eq!(r, 100);
        assert_eq!(g, 200);
        assert_eq!(b, 255);
    }

    // F-CONST-012: Hex conversion Memory color
    #[test]
    fn test_memory_color_hex() {
        // #B478FF = (180,120,255)
        let r = (MEMORY_COLOR.r * 255.0).round() as u8;
        let g = (MEMORY_COLOR.g * 255.0).round() as u8;
        let b = (MEMORY_COLOR.b * 255.0).round() as u8;
        assert_eq!(r, 180);
        assert_eq!(g, 120);
        assert_eq!(b, 255);
    }

    // F-CONST-013: Hex conversion Process color
    #[test]
    fn test_process_color_hex() {
        // #DCC464 = (220,180,100)
        let r = (PROCESS_COLOR.r * 255.0).round() as u8;
        let g = (PROCESS_COLOR.g * 255.0).round() as u8;
        let b = (PROCESS_COLOR.b * 255.0).round() as u8;
        assert_eq!(r, 220);
        assert_eq!(g, 180);
        assert_eq!(b, 100);
    }

    // F-CONST-014: Hex conversion GPU color
    #[test]
    fn test_gpu_color_hex() {
        // #64FF96 = (100,255,150)
        let r = (GPU_COLOR.r * 255.0).round() as u8;
        let g = (GPU_COLOR.g * 255.0).round() as u8;
        let b = (GPU_COLOR.b * 255.0).round() as u8;
        assert_eq!(r, 100);
        assert_eq!(g, 255);
        assert_eq!(b, 150);
    }

    // F-CONST-015: Hex conversion Network color
    #[test]
    fn test_network_color_hex() {
        // #FF9664 = (255,150,100)
        let r = (NETWORK_COLOR.r * 255.0).round() as u8;
        let g = (NETWORK_COLOR.g * 255.0).round() as u8;
        let b = (NETWORK_COLOR.b * 255.0).round() as u8;
        assert_eq!(r, 255);
        assert_eq!(g, 150);
        assert_eq!(b, 100);
    }

    // F-CONST-016: All colors are valid (no NaN or Inf)
    #[test]
    fn test_all_colors_valid() {
        let colors = [
            CPU_COLOR,
            MEMORY_COLOR,
            DISK_COLOR,
            NETWORK_COLOR,
            PROCESS_COLOR,
            GPU_COLOR,
            BATTERY_COLOR,
            SENSORS_COLOR,
            PSI_COLOR,
            CONNECTIONS_COLOR,
            FILES_COLOR,
            CONTAINERS_COLOR,
            NET_RX_COLOR,
            NET_TX_COLOR,
            DISK_READ_COLOR,
            DISK_WRITE_COLOR,
            WHITE,
            GRAY,
            DARK_GRAY,
            BLACK,
            GREEN,
            YELLOW,
            RED,
            CYAN,
            MAGENTA,
            SELECTION_BG,
            HEADER_BG,
        ];
        for color in colors {
            assert!(!color.r.is_nan() && !color.r.is_infinite());
            assert!(!color.g.is_nan() && !color.g.is_infinite());
            assert!(!color.b.is_nan() && !color.b.is_infinite());
            assert!(!color.a.is_nan() && !color.a.is_infinite());
        }
    }

    // F-CONST-017: All colors in 0-1 range
    #[test]
    fn test_all_colors_in_range() {
        let colors = [
            CPU_COLOR,
            MEMORY_COLOR,
            DISK_COLOR,
            NETWORK_COLOR,
            PROCESS_COLOR,
            GPU_COLOR,
            BATTERY_COLOR,
            SENSORS_COLOR,
            PSI_COLOR,
            CONNECTIONS_COLOR,
            FILES_COLOR,
            CONTAINERS_COLOR,
        ];
        for color in colors {
            assert!(color.r >= 0.0 && color.r <= 1.0);
            assert!(color.g >= 0.0 && color.g <= 1.0);
            assert!(color.b >= 0.0 && color.b <= 1.0);
            assert!(color.a >= 0.0 && color.a <= 1.0);
        }
    }

    // F-CONST-018: Panel colors are distinguishable from each other
    // Note: Some colors (CPU/Disk) are intentionally similar blue variants
    #[test]
    fn test_panel_colors_distinguishable() {
        let colors = [
            ("CPU", CPU_COLOR),
            ("Memory", MEMORY_COLOR),
            ("Network", NETWORK_COLOR),
            ("Process", PROCESS_COLOR),
            ("GPU", GPU_COLOR),
        ];

        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                let (name1, c1) = colors[i];
                let (name2, c2) = colors[j];
                let diff = (c1.r - c2.r).abs() + (c1.g - c2.g).abs() + (c1.b - c2.b).abs();
                assert!(
                    diff > 0.15,
                    "{} and {} colors should be distinguishable (diff={})",
                    name1,
                    name2,
                    diff
                );
            }
        }
    }

    // F-CONST-019: Cyan is actually cyan
    #[test]
    fn test_cyan_is_cyan() {
        assert!(CYAN.g > 0.5 && CYAN.b > 0.5);
        assert!(CYAN.r < CYAN.g && CYAN.r < CYAN.b);
    }

    // F-CONST-020: Magenta is actually magenta
    #[test]
    fn test_magenta_is_magenta() {
        assert!(MAGENTA.r > 0.5 && MAGENTA.b > 0.5);
    }
}
