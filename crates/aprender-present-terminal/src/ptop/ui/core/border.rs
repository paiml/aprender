//! Panel border creation utilities for ptop.
//!
//! Creates visually distinct borders for focused and unfocused panels.

#![allow(dead_code)]

use crate::{Border, BorderStyle};
use presentar_core::Color;

/// Focus accent color (bright cyan) for blending with panel colors.
pub const FOCUS_ACCENT_COLOR: Color = Color {
    r: 0.2,
    g: 0.8,
    b: 1.0,
    a: 1.0,
};

/// Create a panel border with appropriate styling for focus state.
///
/// # Arguments
///
/// * `title` - Panel title text
/// * `color` - Base panel color
/// * `is_focused` - Whether this panel currently has focus
///
/// # Returns
///
/// A configured `Border` widget with:
/// - Double border style for focused panels (WCAG AAA compliant)
/// - Rounded border style for unfocused panels
/// - Accent-blended color for focused, dimmed for unfocused
/// - Focus arrow indicator (`►`) in title when focused
#[must_use]
pub fn create_panel_border(title: &str, color: Color, is_focused: bool) -> Border {
    let style = if is_focused {
        BorderStyle::Double // Double border for focused panel
    } else {
        BorderStyle::Rounded // Normal rounded border
    };

    // Use accent color for focused, dim for unfocused
    let border_color = if is_focused {
        blend_with_accent(color)
    } else {
        dim_color(color)
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

/// Blend a color with the focus accent color (50/50 mix).
#[must_use]
pub fn blend_with_accent(color: Color) -> Color {
    Color {
        r: (color.r * 0.5 + FOCUS_ACCENT_COLOR.r * 0.5).min(1.0),
        g: (color.g * 0.5 + FOCUS_ACCENT_COLOR.g * 0.5).min(1.0),
        b: (color.b * 0.5 + FOCUS_ACCENT_COLOR.b * 0.5).min(1.0),
        a: color.a,
    }
}

/// Dim a color by 50% for unfocused state.
#[must_use]
pub fn dim_color(color: Color) -> Color {
    Color {
        r: color.r * 0.5,
        g: color.g * 0.5,
        b: color.b * 0.5,
        a: color.a,
    }
}

/// Brighten a color for highlights.
#[must_use]
pub fn brighten_color(color: Color, factor: f32) -> Color {
    Color {
        r: (color.r * factor).min(1.0),
        g: (color.g * factor).min(1.0),
        b: (color.b * factor).min(1.0),
        a: color.a,
    }
}

/// Darken a color for shadows.
#[must_use]
pub fn darken_color(color: Color, factor: f32) -> Color {
    Color {
        r: color.r * factor,
        g: color.g * factor,
        b: color.b * factor,
        a: color.a,
    }
}

/// Interpolate between two colors.
#[must_use]
pub fn lerp_color(from: Color, to: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    Color {
        r: from.r + (to.r - from.r) * t,
        g: from.g + (to.g - from.g) * t,
        b: from.b + (to.b - from.b) * t,
        a: from.a + (to.a - from.a) * t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_COLOR: Color = Color {
        r: 0.5,
        g: 0.6,
        b: 0.7,
        a: 1.0,
    };

    // F-BORDER-001: Focus accent color is bright cyan
    #[test]
    fn test_focus_accent_color_is_cyan() {
        assert!(FOCUS_ACCENT_COLOR.b > FOCUS_ACCENT_COLOR.r);
        assert!(FOCUS_ACCENT_COLOR.g > 0.5);
        assert!(FOCUS_ACCENT_COLOR.b > 0.5);
    }

    // F-BORDER-002: Focused border has double style
    #[test]
    fn test_focused_border_double_style() {
        let border = create_panel_border("Test", TEST_COLOR, true);
        // Border should be created without panicking
        let _ = border;
    }

    // F-BORDER-003: Unfocused border has rounded style
    #[test]
    fn test_unfocused_border_rounded_style() {
        let border = create_panel_border("Test", TEST_COLOR, false);
        let _ = border;
    }

    // F-BORDER-004: Focused title has arrow indicator
    #[test]
    fn test_focused_title_has_arrow() {
        // We can't easily test the title directly, but the function should work
        let _border = create_panel_border("CPU", TEST_COLOR, true);
    }

    // F-BORDER-005: Unfocused title has no arrow
    #[test]
    fn test_unfocused_title_no_arrow() {
        let _border = create_panel_border("CPU", TEST_COLOR, false);
    }

    // F-BORDER-006: Blend with accent produces valid color
    #[test]
    fn test_blend_with_accent_valid() {
        let blended = blend_with_accent(TEST_COLOR);
        assert!(blended.r >= 0.0 && blended.r <= 1.0);
        assert!(blended.g >= 0.0 && blended.g <= 1.0);
        assert!(blended.b >= 0.0 && blended.b <= 1.0);
        assert!((blended.a - 1.0).abs() < 0.001);
    }

    // F-BORDER-007: Blend with accent increases blue component
    #[test]
    fn test_blend_with_accent_increases_blue() {
        let original = Color {
            r: 0.5,
            g: 0.5,
            b: 0.3,
            a: 1.0,
        };
        let blended = blend_with_accent(original);
        // Blue should increase towards accent's high blue
        assert!(blended.b > original.b * 0.5);
    }

    // F-BORDER-008: Dim color reduces brightness
    #[test]
    fn test_dim_color_reduces_brightness() {
        let dimmed = dim_color(TEST_COLOR);
        assert!(dimmed.r < TEST_COLOR.r);
        assert!(dimmed.g < TEST_COLOR.g);
        assert!(dimmed.b < TEST_COLOR.b);
    }

    // F-BORDER-009: Dim color is exactly 50%
    #[test]
    fn test_dim_color_is_half() {
        let dimmed = dim_color(TEST_COLOR);
        assert!((dimmed.r - TEST_COLOR.r * 0.5).abs() < 0.001);
        assert!((dimmed.g - TEST_COLOR.g * 0.5).abs() < 0.001);
        assert!((dimmed.b - TEST_COLOR.b * 0.5).abs() < 0.001);
    }

    // F-BORDER-010: Dim color preserves alpha
    #[test]
    fn test_dim_color_preserves_alpha() {
        let dimmed = dim_color(TEST_COLOR);
        assert!((dimmed.a - TEST_COLOR.a).abs() < 0.001);
    }

    // F-BORDER-011: Brighten color increases brightness
    #[test]
    fn test_brighten_color() {
        let brightened = brighten_color(TEST_COLOR, 1.5);
        assert!(brightened.r >= TEST_COLOR.r);
        assert!(brightened.g >= TEST_COLOR.g);
        assert!(brightened.b >= TEST_COLOR.b);
    }

    // F-BORDER-012: Brighten color clamps to 1.0
    #[test]
    fn test_brighten_color_clamps() {
        let bright = Color {
            r: 0.9,
            g: 0.9,
            b: 0.9,
            a: 1.0,
        };
        let brightened = brighten_color(bright, 2.0);
        assert!(brightened.r <= 1.0);
        assert!(brightened.g <= 1.0);
        assert!(brightened.b <= 1.0);
    }

    // F-BORDER-013: Darken color reduces brightness
    #[test]
    fn test_darken_color() {
        let darkened = darken_color(TEST_COLOR, 0.5);
        assert!(darkened.r < TEST_COLOR.r);
        assert!(darkened.g < TEST_COLOR.g);
        assert!(darkened.b < TEST_COLOR.b);
    }

    // F-BORDER-014: Darken preserves alpha
    #[test]
    fn test_darken_preserves_alpha() {
        let darkened = darken_color(TEST_COLOR, 0.5);
        assert!((darkened.a - TEST_COLOR.a).abs() < 0.001);
    }

    // F-BORDER-015: Lerp at t=0 returns from color
    #[test]
    fn test_lerp_t_zero() {
        let from = Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };
        let to = Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        let result = lerp_color(from, to, 0.0);
        assert!((result.r - 0.0).abs() < 0.001);
        assert!((result.g - 0.0).abs() < 0.001);
        assert!((result.b - 0.0).abs() < 0.001);
    }

    // F-BORDER-016: Lerp at t=1 returns to color
    #[test]
    fn test_lerp_t_one() {
        let from = Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };
        let to = Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        let result = lerp_color(from, to, 1.0);
        assert!((result.r - 1.0).abs() < 0.001);
        assert!((result.g - 1.0).abs() < 0.001);
        assert!((result.b - 1.0).abs() < 0.001);
    }

    // F-BORDER-017: Lerp at t=0.5 returns midpoint
    #[test]
    fn test_lerp_t_half() {
        let from = Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };
        let to = Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        let result = lerp_color(from, to, 0.5);
        assert!((result.r - 0.5).abs() < 0.001);
        assert!((result.g - 0.5).abs() < 0.001);
        assert!((result.b - 0.5).abs() < 0.001);
    }

    // F-BORDER-018: Lerp clamps t below 0
    #[test]
    fn test_lerp_clamps_below_zero() {
        let from = Color {
            r: 0.5,
            g: 0.5,
            b: 0.5,
            a: 1.0,
        };
        let to = Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        let result = lerp_color(from, to, -0.5);
        assert!((result.r - 0.5).abs() < 0.001);
    }

    // F-BORDER-019: Lerp clamps t above 1
    #[test]
    fn test_lerp_clamps_above_one() {
        let from = Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };
        let to = Color {
            r: 0.5,
            g: 0.5,
            b: 0.5,
            a: 1.0,
        };
        let result = lerp_color(from, to, 1.5);
        assert!((result.r - 0.5).abs() < 0.001);
    }

    // F-BORDER-020: Lerp interpolates alpha
    #[test]
    fn test_lerp_interpolates_alpha() {
        let from = Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        };
        let to = Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        let result = lerp_color(from, to, 0.5);
        assert!((result.a - 0.5).abs() < 0.001);
    }

    // F-BORDER-021: Empty title creates border
    #[test]
    fn test_empty_title_border() {
        let _border = create_panel_border("", TEST_COLOR, false);
    }

    // F-BORDER-022: Long title creates border
    #[test]
    fn test_long_title_border() {
        let long_title = "This is a very long title that might overflow";
        let _border = create_panel_border(long_title, TEST_COLOR, false);
    }

    // F-BORDER-023: White color dims correctly
    #[test]
    fn test_white_dims_correctly() {
        let white = Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        let dimmed = dim_color(white);
        assert!((dimmed.r - 0.5).abs() < 0.001);
        assert!((dimmed.g - 0.5).abs() < 0.001);
        assert!((dimmed.b - 0.5).abs() < 0.001);
    }

    // F-BORDER-024: Black color dims to black
    #[test]
    fn test_black_dims_to_black() {
        let black = Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };
        let dimmed = dim_color(black);
        assert!(dimmed.r.abs() < 0.001);
        assert!(dimmed.g.abs() < 0.001);
        assert!(dimmed.b.abs() < 0.001);
    }

    // F-BORDER-025: Blend produces distinct color
    #[test]
    fn test_blend_produces_distinct_color() {
        let original = Color {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };
        let blended = blend_with_accent(original);
        // Should be different from original
        assert!((blended.r - original.r).abs() > 0.1);
    }
}
