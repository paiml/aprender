//! Panel rendering modules for ptop.
//!
//! Each panel has its own module with:
//! - Drawing function
//! - Panel-specific helpers
//! - Comprehensive tests

pub mod battery;
pub mod connections;
pub mod cpu;
pub mod disk;
pub mod memory;
pub mod network;
pub mod process;
pub mod psi;
pub mod sensors;

use crate::{Border, BorderStyle};
use presentar_core::Color;

use super::colors::FOCUS_ACCENT_COLOR;

// =============================================================================
// COMMON PANEL UTILITIES
// =============================================================================

/// Focus indicator character
pub const FOCUS_INDICATOR: &str = "►";

/// Create a panel border with focus indication.
///
/// # Focus Indicators (WCAG AAA compliant):
/// 1. Double-line border for focused (vs rounded for unfocused)
/// 2. Bright accent color blend for focused
/// 3. Focus indicator arrow `►` prepended to title
/// 4. Unfocused panels are dimmed for contrast
#[must_use]
pub fn create_panel_border(title: &str, color: Color, is_focused: bool) -> Border {
    let style = if is_focused {
        BorderStyle::Double
    } else {
        BorderStyle::Rounded
    };

    let border_color = if is_focused {
        // Blend panel color with accent for focused state
        blend_with_accent(color)
    } else {
        // Dim unfocused panels
        dim_color(color)
    };

    let display_title = if is_focused {
        format!("{FOCUS_INDICATOR} {title}")
    } else {
        title.to_string()
    };

    Border::new()
        .with_title(&display_title)
        .with_style(style)
        .with_color(border_color)
        .with_title_left_aligned()
}

/// Blend a color with the focus accent color (50/50 blend).
#[must_use]
pub fn blend_with_accent(color: Color) -> Color {
    Color {
        r: (color.r * 0.5 + FOCUS_ACCENT_COLOR.r * 0.5).min(1.0),
        g: (color.g * 0.5 + FOCUS_ACCENT_COLOR.g * 0.5).min(1.0),
        b: (color.b * 0.5 + FOCUS_ACCENT_COLOR.b * 0.5).min(1.0),
        a: color.a,
    }
}

/// Dim a color by 50% (for unfocused panels).
#[must_use]
pub fn dim_color(color: Color) -> Color {
    Color {
        r: color.r * 0.5,
        g: color.g * 0.5,
        b: color.b * 0.5,
        a: color.a,
    }
}

/// Calculate inner rect from border bounds.
///
/// Returns the area inside the border (1 cell margin on each side).
#[must_use]
pub fn inner_rect(outer: presentar_core::Rect) -> presentar_core::Rect {
    presentar_core::Rect::new(
        outer.x + 1.0,
        outer.y + 1.0,
        (outer.width - 2.0).max(0.0),
        (outer.height - 2.0).max(0.0),
    )
}

/// Check if panel has enough space to render content.
#[must_use]
pub fn has_space(inner_height: f32, min_lines: usize) -> bool {
    inner_height >= min_lines as f32
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use presentar_core::Rect;

    // =========================================================================
    // create_panel_border tests
    // =========================================================================

    #[test]
    fn test_create_panel_border_unfocused_style() {
        // Verify unfocused border can be created (uses Rounded style internally)
        let _border = create_panel_border("Test", Color::new(1.0, 0.0, 0.0, 1.0), false);
    }

    #[test]
    fn test_create_panel_border_focused_style() {
        // Verify focused border can be created (uses Double style internally)
        let _border = create_panel_border("Test", Color::new(1.0, 0.0, 0.0, 1.0), true);
    }

    #[test]
    fn test_create_panel_border_unfocused_creates_border() {
        // Just verify we can create a border without panic
        let _border = create_panel_border("CPU", Color::new(1.0, 0.0, 0.0, 1.0), false);
    }

    #[test]
    fn test_create_panel_border_focused_creates_border() {
        // Just verify we can create a border without panic
        let _border = create_panel_border("CPU", Color::new(1.0, 0.0, 0.0, 1.0), true);
    }

    #[test]
    fn test_create_panel_border_with_title() {
        // Verify border can be created with various titles
        let _border1 = create_panel_border("Memory", Color::new(1.0, 0.0, 0.0, 1.0), false);
        let _border2 = create_panel_border("CPU │ 8 cores", Color::new(0.5, 0.5, 1.0, 1.0), true);
        let _border3 = create_panel_border("", Color::new(0.0, 1.0, 0.0, 1.0), false);
    }

    // =========================================================================
    // blend_with_accent tests
    // =========================================================================

    #[test]
    fn test_blend_with_accent_pure_red() {
        let red = Color::new(1.0, 0.0, 0.0, 1.0);
        let blended = blend_with_accent(red);
        // Should be average of red and accent
        assert!(blended.r > 0.4 && blended.r < 0.8);
        assert!(blended.g > 0.3); // Accent has green
    }

    #[test]
    fn test_blend_with_accent_preserves_alpha() {
        let color = Color::new(1.0, 0.0, 0.0, 0.5);
        let blended = blend_with_accent(color);
        assert_eq!(blended.a, 0.5, "Alpha should be preserved");
    }

    #[test]
    fn test_blend_with_accent_clamps_to_1() {
        let bright = Color::new(1.0, 1.0, 1.0, 1.0);
        let blended = blend_with_accent(bright);
        assert!(blended.r <= 1.0);
        assert!(blended.g <= 1.0);
        assert!(blended.b <= 1.0);
    }

    #[test]
    fn test_blend_with_accent_black() {
        let black = Color::new(0.0, 0.0, 0.0, 1.0);
        let blended = blend_with_accent(black);
        // Should be half of accent color
        assert!(blended.r > 0.0 || blended.g > 0.0 || blended.b > 0.0);
    }

    // =========================================================================
    // dim_color tests
    // =========================================================================

    #[test]
    fn test_dim_color_halves_rgb() {
        let color = Color::new(1.0, 0.8, 0.6, 1.0);
        let dimmed = dim_color(color);
        assert_eq!(dimmed.r, 0.5);
        assert_eq!(dimmed.g, 0.4);
        assert!((dimmed.b - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_dim_color_preserves_alpha() {
        let color = Color::new(1.0, 0.0, 0.0, 0.8);
        let dimmed = dim_color(color);
        assert_eq!(dimmed.a, 0.8, "Alpha should be preserved");
    }

    #[test]
    fn test_dim_color_already_dim() {
        let dim = Color::new(0.2, 0.2, 0.2, 1.0);
        let dimmed = dim_color(dim);
        assert_eq!(dimmed.r, 0.1);
        assert_eq!(dimmed.g, 0.1);
        assert_eq!(dimmed.b, 0.1);
    }

    #[test]
    fn test_dim_color_black_stays_black() {
        let black = Color::new(0.0, 0.0, 0.0, 1.0);
        let dimmed = dim_color(black);
        assert_eq!(dimmed.r, 0.0);
        assert_eq!(dimmed.g, 0.0);
        assert_eq!(dimmed.b, 0.0);
    }

    // =========================================================================
    // inner_rect tests
    // =========================================================================

    #[test]
    fn test_inner_rect_normal() {
        let outer = Rect::new(0.0, 0.0, 10.0, 10.0);
        let inner = inner_rect(outer);
        assert_eq!(inner.x, 1.0);
        assert_eq!(inner.y, 1.0);
        assert_eq!(inner.width, 8.0);
        assert_eq!(inner.height, 8.0);
    }

    #[test]
    fn test_inner_rect_offset() {
        let outer = Rect::new(5.0, 10.0, 20.0, 15.0);
        let inner = inner_rect(outer);
        assert_eq!(inner.x, 6.0);
        assert_eq!(inner.y, 11.0);
        assert_eq!(inner.width, 18.0);
        assert_eq!(inner.height, 13.0);
    }

    #[test]
    fn test_inner_rect_too_small() {
        let outer = Rect::new(0.0, 0.0, 2.0, 2.0);
        let inner = inner_rect(outer);
        assert_eq!(inner.width, 0.0);
        assert_eq!(inner.height, 0.0);
    }

    #[test]
    fn test_inner_rect_minimal() {
        let outer = Rect::new(0.0, 0.0, 3.0, 3.0);
        let inner = inner_rect(outer);
        assert_eq!(inner.width, 1.0);
        assert_eq!(inner.height, 1.0);
    }

    // =========================================================================
    // has_space tests
    // =========================================================================

    #[test]
    fn test_has_space_yes() {
        assert!(has_space(10.0, 5));
    }

    #[test]
    fn test_has_space_exact() {
        assert!(has_space(5.0, 5));
    }

    #[test]
    fn test_has_space_no() {
        assert!(!has_space(3.0, 5));
    }

    #[test]
    fn test_has_space_zero_lines() {
        assert!(has_space(0.0, 0));
    }

    #[test]
    fn test_has_space_negative_height() {
        // Negative height treated as no space
        assert!(!has_space(-1.0, 1));
    }

    // =========================================================================
    // FOCUS_INDICATOR tests
    // =========================================================================

    #[test]
    fn test_focus_indicator_is_arrow() {
        assert_eq!(FOCUS_INDICATOR, "►");
    }

    #[test]
    fn test_focus_indicator_single_char() {
        assert_eq!(FOCUS_INDICATOR.chars().count(), 1);
    }

    // =========================================================================
    // Color range tests
    // =========================================================================

    #[test]
    fn test_blend_colors_in_valid_range() {
        let colors = [
            Color::new(0.0, 0.0, 0.0, 1.0),
            Color::new(1.0, 1.0, 1.0, 1.0),
            Color::new(0.5, 0.5, 0.5, 1.0),
            Color::new(1.0, 0.0, 0.0, 1.0),
            Color::new(0.0, 1.0, 0.0, 1.0),
            Color::new(0.0, 0.0, 1.0, 1.0),
        ];

        for c in colors {
            let blended = blend_with_accent(c);
            assert!(blended.r >= 0.0 && blended.r <= 1.0);
            assert!(blended.g >= 0.0 && blended.g <= 1.0);
            assert!(blended.b >= 0.0 && blended.b <= 1.0);
        }
    }

    #[test]
    fn test_dim_colors_in_valid_range() {
        let colors = [
            Color::new(0.0, 0.0, 0.0, 1.0),
            Color::new(1.0, 1.0, 1.0, 1.0),
            Color::new(2.0, 2.0, 2.0, 1.0), // Out of range input
        ];

        for c in colors {
            let dimmed = dim_color(c);
            // Dimming should always produce valid colors (assuming input was valid)
            assert!(dimmed.r >= 0.0);
            assert!(dimmed.g >= 0.0);
            assert!(dimmed.b >= 0.0);
        }
    }
}
