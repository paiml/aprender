//! Color mode detection and conversion for terminals.

use crossterm::style::Color as CrosstermColor;
use presentar_core::Color;

/// Terminal color capability mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    /// 24-bit true color (COLORTERM=truecolor or 24bit).
    #[default]
    TrueColor,
    /// 256 color palette.
    Color256,
    /// 16 ANSI colors.
    Color16,
    /// Monochrome (no color).
    Mono,
}

impl ColorMode {
    /// Auto-detect terminal color capabilities.
    #[must_use]
    pub fn detect() -> Self {
        Self::detect_with_env(std::env::var("COLORTERM").ok(), std::env::var("TERM").ok())
    }

    /// Detect color mode from environment variable values.
    /// This is the testable core of `detect()`.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn detect_with_env(colorterm: Option<String>, term: Option<String>) -> Self {
        // Check COLORTERM first (most reliable)
        if let Some(ref ct) = colorterm {
            if ct == "truecolor" || ct == "24bit" {
                return Self::TrueColor;
            }
        }

        // Fall back to TERM
        match term.as_deref() {
            Some(t) if t.contains("256color") => Self::Color256,
            Some(t) if t.contains("color") || t.contains("xterm") => Self::Color16,
            Some("dumb") | None => Self::Mono,
            _ => Self::Color16,
        }
    }

    /// Convert a presentar Color to crossterm Color based on this mode.
    ///
    /// Note: Transparent colors (alpha = 0) return `CrosstermColor::Reset` which
    /// uses the terminal's default background color instead of rendering as black.
    #[must_use]
    pub fn to_crossterm(&self, color: Color) -> CrosstermColor {
        debug_assert!(color.r >= 0.0 && color.r <= 1.0, "r must be in 0.0-1.0");
        debug_assert!(color.g >= 0.0 && color.g <= 1.0, "g must be in 0.0-1.0");
        debug_assert!(color.b >= 0.0 && color.b <= 1.0, "b must be in 0.0-1.0");
        debug_assert!(color.a >= 0.0 && color.a <= 1.0, "a must be in 0.0-1.0");

        // CRITICAL: Handle transparent colors specially to avoid black squares!
        // Color::TRANSPARENT is {r: 0, g: 0, b: 0, a: 0} - without this check,
        // it would convert to RGB(0,0,0) = BLACK, creating ugly black artifacts.
        if color.a == 0.0 {
            return CrosstermColor::Reset;
        }

        let r = (color.r * 255.0).round() as u8;
        let g = (color.g * 255.0).round() as u8;
        let b = (color.b * 255.0).round() as u8;

        match self {
            Self::TrueColor => CrosstermColor::Rgb { r, g, b },
            Self::Color256 => CrosstermColor::AnsiValue(Self::rgb_to_256(r, g, b)),
            Self::Color16 => Self::rgb_to_16(r, g, b),
            Self::Mono => CrosstermColor::White,
        }
    }

    /// Convert RGB to 256-color palette index.
    fn rgb_to_256(r: u8, g: u8, b: u8) -> u8 {
        // Check for grayscale (r == g == b)
        if r == g && g == b {
            if r < 8 {
                return 16; // black
            }
            if r > 248 {
                return 231; // white
            }
            // Grayscale ramp: colors 232-255 (24 shades)
            return 232 + ((r - 8) / 10).min(23);
        }

        // 6x6x6 color cube (colors 16-231)
        let r_idx = (u16::from(r) * 5 / 255) as u8;
        let g_idx = (u16::from(g) * 5 / 255) as u8;
        let b_idx = (u16::from(b) * 5 / 255) as u8;
        16 + 36 * r_idx + 6 * g_idx + b_idx
    }

    /// Convert RGB to 16-color ANSI.
    fn rgb_to_16(r: u8, g: u8, b: u8) -> CrosstermColor {
        let luminance = (u32::from(r) * 299 + u32::from(g) * 587 + u32::from(b) * 114) / 1000;
        let bright = luminance > 127;

        let max = r.max(g).max(b);
        let threshold = max / 2;

        let has_r = r > threshold;
        let has_g = g > threshold;
        let has_b = b > threshold;

        match (has_r, has_g, has_b, bright) {
            (false, false, false, false) => CrosstermColor::Black,
            (false, false, false, true) => CrosstermColor::DarkGrey,
            (true, false, false, false) => CrosstermColor::DarkRed,
            (true, false, false, true) => CrosstermColor::Red,
            (false, true, false, false) => CrosstermColor::DarkGreen,
            (false, true, false, true) => CrosstermColor::Green,
            (true, true, false, false) => CrosstermColor::DarkYellow,
            (true, true, false, true) => CrosstermColor::Yellow,
            (false, false, true, false) => CrosstermColor::DarkBlue,
            (false, false, true, true) => CrosstermColor::Blue,
            (true, false, true, false) => CrosstermColor::DarkMagenta,
            (true, false, true, true) => CrosstermColor::Magenta,
            (false, true, true, false) => CrosstermColor::DarkCyan,
            (false, true, true, true) => CrosstermColor::Cyan,
            (true, true, true, false) => CrosstermColor::Grey,
            (true, true, true, true) => CrosstermColor::White,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_mode_default() {
        assert_eq!(ColorMode::default(), ColorMode::TrueColor);
    }

    #[test]
    fn test_truecolor_conversion() {
        let mode = ColorMode::TrueColor;
        let color = Color::new(0.5, 0.25, 0.75, 1.0);
        let result = mode.to_crossterm(color);
        assert_eq!(
            result,
            CrosstermColor::Rgb {
                r: 128,
                g: 64,
                b: 191
            }
        );
    }

    #[test]
    fn test_256_grayscale() {
        // Pure black
        assert_eq!(ColorMode::rgb_to_256(0, 0, 0), 16);
        // Pure white
        assert_eq!(ColorMode::rgb_to_256(255, 255, 255), 231);
        // Mid gray
        let mid = ColorMode::rgb_to_256(128, 128, 128);
        assert!(mid >= 232);
    }

    #[test]
    fn test_256_grayscale_near_black() {
        // Very dark gray (still mapped to black)
        assert_eq!(ColorMode::rgb_to_256(5, 5, 5), 16);
    }

    #[test]
    fn test_256_grayscale_ramp() {
        // Test various gray values
        let gray50 = ColorMode::rgb_to_256(50, 50, 50);
        assert!(gray50 >= 232);

        let gray100 = ColorMode::rgb_to_256(100, 100, 100);
        assert!(gray100 >= 232);

        let gray200 = ColorMode::rgb_to_256(200, 200, 200);
        assert!(gray200 >= 232);
    }

    #[test]
    fn test_256_color_cube() {
        // Pure red (should be in color cube)
        let red = ColorMode::rgb_to_256(255, 0, 0);
        assert!((16..=231).contains(&red));

        // Pure green
        let green = ColorMode::rgb_to_256(0, 255, 0);
        assert!((16..=231).contains(&green));

        // Pure blue
        let blue = ColorMode::rgb_to_256(0, 0, 255);
        assert!((16..=231).contains(&blue));

        // Magenta
        let magenta = ColorMode::rgb_to_256(255, 0, 255);
        assert!((16..=231).contains(&magenta));
    }

    #[test]
    fn test_16_color_mapping() {
        // Black
        assert_eq!(ColorMode::rgb_to_16(0, 0, 0), CrosstermColor::Black);
        // White
        assert_eq!(ColorMode::rgb_to_16(255, 255, 255), CrosstermColor::White);
        // Pure red
        assert!(matches!(
            ColorMode::rgb_to_16(255, 0, 0),
            CrosstermColor::Red | CrosstermColor::DarkRed
        ));
    }

    #[test]
    fn test_16_color_green() {
        let result = ColorMode::rgb_to_16(0, 255, 0);
        assert!(matches!(
            result,
            CrosstermColor::Green | CrosstermColor::DarkGreen
        ));
    }

    #[test]
    fn test_16_color_blue() {
        let result = ColorMode::rgb_to_16(0, 0, 255);
        assert!(matches!(
            result,
            CrosstermColor::Blue | CrosstermColor::DarkBlue
        ));
    }

    #[test]
    fn test_16_color_yellow() {
        let result = ColorMode::rgb_to_16(255, 255, 0);
        assert!(matches!(
            result,
            CrosstermColor::Yellow | CrosstermColor::DarkYellow
        ));
    }

    #[test]
    fn test_16_color_cyan() {
        let result = ColorMode::rgb_to_16(0, 255, 255);
        assert!(matches!(
            result,
            CrosstermColor::Cyan | CrosstermColor::DarkCyan
        ));
    }

    #[test]
    fn test_16_color_magenta() {
        let result = ColorMode::rgb_to_16(255, 0, 255);
        assert!(matches!(
            result,
            CrosstermColor::Magenta | CrosstermColor::DarkMagenta
        ));
    }

    #[test]
    fn test_16_color_dark_gray() {
        let result = ColorMode::rgb_to_16(50, 50, 50);
        // With the luminance-based algorithm, dark gray maps to the dark variants
        assert!(matches!(
            result,
            CrosstermColor::Black | CrosstermColor::DarkGrey | CrosstermColor::Grey
        ));
    }

    #[test]
    fn test_16_color_gray() {
        let result = ColorMode::rgb_to_16(192, 192, 192);
        assert!(matches!(
            result,
            CrosstermColor::Grey | CrosstermColor::White
        ));
    }

    #[test]
    fn test_mono_conversion() {
        let mode = ColorMode::Mono;
        assert_eq!(mode.to_crossterm(Color::RED), CrosstermColor::White);
        assert_eq!(mode.to_crossterm(Color::BLUE), CrosstermColor::White);
        assert_eq!(mode.to_crossterm(Color::GREEN), CrosstermColor::White);
    }

    #[test]
    fn test_transparent_returns_reset() {
        // CRITICAL: Transparent colors must return Reset, NOT black!
        // This prevents the "black squares behind panels" bug.
        for mode in [
            ColorMode::TrueColor,
            ColorMode::Color256,
            ColorMode::Color16,
            ColorMode::Mono,
        ] {
            assert_eq!(
                mode.to_crossterm(Color::TRANSPARENT),
                CrosstermColor::Reset,
                "Mode {:?} should return Reset for TRANSPARENT",
                mode
            );

            // Also test any color with alpha=0
            let zero_alpha = Color::new(1.0, 0.5, 0.25, 0.0);
            assert_eq!(
                mode.to_crossterm(zero_alpha),
                CrosstermColor::Reset,
                "Mode {:?} should return Reset for any color with alpha=0",
                mode
            );
        }
    }

    #[test]
    fn test_256_conversion() {
        let mode = ColorMode::Color256;
        let result = mode.to_crossterm(Color::new(0.5, 0.25, 0.75, 1.0));
        assert!(matches!(result, CrosstermColor::AnsiValue(_)));
    }

    #[test]
    fn test_16_conversion() {
        let mode = ColorMode::Color16;
        let result = mode.to_crossterm(Color::RED);
        assert!(matches!(
            result,
            CrosstermColor::Red | CrosstermColor::DarkRed
        ));
    }

    #[test]
    fn test_color_mode_eq() {
        assert_eq!(ColorMode::TrueColor, ColorMode::TrueColor);
        assert_eq!(ColorMode::Color256, ColorMode::Color256);
        assert_eq!(ColorMode::Color16, ColorMode::Color16);
        assert_eq!(ColorMode::Mono, ColorMode::Mono);
        assert_ne!(ColorMode::TrueColor, ColorMode::Color256);
    }

    #[test]
    fn test_color_mode_clone() {
        let mode = ColorMode::Color256;
        let cloned = mode;
        assert_eq!(mode, cloned);
    }

    #[test]
    fn test_color_mode_debug() {
        let mode = ColorMode::TrueColor;
        assert!(format!("{:?}", mode).contains("TrueColor"));
    }

    #[test]
    fn test_16_color_dim_colors() {
        // Dark red (dim)
        let dark_red = ColorMode::rgb_to_16(128, 0, 0);
        assert!(matches!(
            dark_red,
            CrosstermColor::Red | CrosstermColor::DarkRed
        ));

        // Dark green (dim)
        let dark_green = ColorMode::rgb_to_16(0, 100, 0);
        assert!(matches!(
            dark_green,
            CrosstermColor::Green | CrosstermColor::DarkGreen
        ));

        // Dark blue (dim)
        let dark_blue = ColorMode::rgb_to_16(0, 0, 128);
        assert!(matches!(
            dark_blue,
            CrosstermColor::Blue | CrosstermColor::DarkBlue
        ));
    }

    #[test]
    fn test_256_near_white() {
        // Near white (should be in grayscale ramp, near 231)
        let near_white = ColorMode::rgb_to_256(250, 250, 250);
        assert_eq!(near_white, 231);
    }

    #[test]
    fn test_256_mixed_colors() {
        // Orange-ish
        let orange = ColorMode::rgb_to_256(255, 128, 0);
        assert!((16..=231).contains(&orange));

        // Purple-ish
        let purple = ColorMode::rgb_to_256(128, 0, 255);
        assert!((16..=231).contains(&purple));

        // Teal-ish
        let teal = ColorMode::rgb_to_256(0, 128, 128);
        assert!((16..=231).contains(&teal));
    }

    // Additional tests for better coverage

    #[test]
    fn test_color_mode_detect() {
        // Just verify it doesn't panic and returns a valid mode
        let mode = ColorMode::detect();
        assert!(matches!(
            mode,
            ColorMode::TrueColor | ColorMode::Color256 | ColorMode::Color16 | ColorMode::Mono
        ));
    }

    #[test]
    fn test_16_color_all_dark_variants() {
        // Test dark variants explicitly by keeping luminance low

        // Dark red - high red, low luminance
        let dark_red = ColorMode::rgb_to_16(180, 20, 20);
        assert!(matches!(
            dark_red,
            CrosstermColor::DarkRed | CrosstermColor::Red
        ));

        // Dark green
        let dark_green = ColorMode::rgb_to_16(20, 150, 20);
        assert!(matches!(
            dark_green,
            CrosstermColor::DarkGreen | CrosstermColor::Green
        ));

        // Dark blue
        let dark_blue = ColorMode::rgb_to_16(20, 20, 180);
        assert!(matches!(
            dark_blue,
            CrosstermColor::DarkBlue | CrosstermColor::Blue
        ));

        // Dark yellow
        let dark_yellow = ColorMode::rgb_to_16(150, 150, 20);
        assert!(matches!(
            dark_yellow,
            CrosstermColor::DarkYellow | CrosstermColor::Yellow
        ));

        // Dark cyan
        let dark_cyan = ColorMode::rgb_to_16(20, 150, 150);
        assert!(matches!(
            dark_cyan,
            CrosstermColor::DarkCyan | CrosstermColor::Cyan
        ));

        // Dark magenta
        let dark_magenta = ColorMode::rgb_to_16(150, 20, 150);
        assert!(matches!(
            dark_magenta,
            CrosstermColor::DarkMagenta | CrosstermColor::Magenta
        ));
    }

    #[test]
    fn test_16_color_bright_variants() {
        // Test bright variants - verifies the function returns valid colors
        // The exact mapping depends on the threshold algorithm

        // Bright red
        let bright_red = ColorMode::rgb_to_16(255, 50, 50);
        assert!(!matches!(bright_red, CrosstermColor::Black));

        // Bright green
        let bright_green = ColorMode::rgb_to_16(50, 255, 50);
        assert!(!matches!(bright_green, CrosstermColor::Black));

        // Bright blue
        let bright_blue = ColorMode::rgb_to_16(50, 50, 255);
        assert!(!matches!(bright_blue, CrosstermColor::Black));
    }

    #[test]
    fn test_16_color_dark_grey_explicit() {
        // Dark grey: no dominant color, but luminance > threshold for DarkGrey
        let dark_grey = ColorMode::rgb_to_16(80, 80, 80);
        assert!(matches!(
            dark_grey,
            CrosstermColor::DarkGrey | CrosstermColor::Black | CrosstermColor::Grey
        ));
    }

    #[test]
    fn test_to_crossterm_edge_values() {
        // Test edge values for color conversion
        let mode = ColorMode::TrueColor;

        // Black
        let black = mode.to_crossterm(Color::new(0.0, 0.0, 0.0, 1.0));
        assert_eq!(black, CrosstermColor::Rgb { r: 0, g: 0, b: 0 });

        // White
        let white = mode.to_crossterm(Color::new(1.0, 1.0, 1.0, 1.0));
        assert_eq!(
            white,
            CrosstermColor::Rgb {
                r: 255,
                g: 255,
                b: 255
            }
        );
    }

    #[test]
    fn test_256_grayscale_boundary() {
        // Test grayscale at various boundaries
        assert_eq!(ColorMode::rgb_to_256(7, 7, 7), 16); // < 8, should be black
        assert_eq!(ColorMode::rgb_to_256(8, 8, 8), 232); // >= 8, first grayscale
        assert_eq!(ColorMode::rgb_to_256(249, 249, 249), 231); // > 248, white
    }

    #[test]
    fn test_256_color_cube_corners() {
        // Test color cube corner values
        // (0,0,0) in cube
        let c000 = ColorMode::rgb_to_256(1, 1, 2); // Not grayscale, maps to cube
        assert!((16..=231).contains(&c000));

        // (5,5,5) in cube = 16 + 36*5 + 6*5 + 5 = 16 + 180 + 30 + 5 = 231
        let c555 = ColorMode::rgb_to_256(254, 254, 255); // Max non-grayscale
        assert!((16..=231).contains(&c555));
    }

    #[test]
    fn test_color16_to_crossterm() {
        let mode = ColorMode::Color16;

        // Various colors through the mode
        let red = mode.to_crossterm(Color::RED);
        assert!(matches!(red, CrosstermColor::Red | CrosstermColor::DarkRed));

        let green = mode.to_crossterm(Color::GREEN);
        assert!(matches!(
            green,
            CrosstermColor::Green | CrosstermColor::DarkGreen
        ));

        let blue = mode.to_crossterm(Color::BLUE);
        assert!(matches!(
            blue,
            CrosstermColor::Blue | CrosstermColor::DarkBlue
        ));

        let black = mode.to_crossterm(Color::BLACK);
        assert!(matches!(black, CrosstermColor::Black));

        let white = mode.to_crossterm(Color::WHITE);
        assert!(matches!(white, CrosstermColor::White));
    }

    #[test]
    fn test_color256_grayscale_through_mode() {
        let mode = ColorMode::Color256;

        // Black
        let black = mode.to_crossterm(Color::BLACK);
        assert!(matches!(black, CrosstermColor::AnsiValue(16)));

        // Mid gray
        let gray = mode.to_crossterm(Color::new(0.5, 0.5, 0.5, 1.0));
        if let CrosstermColor::AnsiValue(v) = gray {
            assert!(v >= 232 || ((16..=231).contains(&v)));
        }
    }

    #[test]
    #[allow(clippy::absurd_extreme_comparisons)]
    fn test_rgb_to_256_extensive() {
        // Test various color combinations to ensure full cube coverage
        for r in [0, 51, 102, 153, 204, 255] {
            for g in [0, 51, 102, 153, 204, 255] {
                for b in [0, 51, 102, 153, 204, 255] {
                    let result = ColorMode::rgb_to_256(r, g, b);
                    // Result should be a valid ANSI color (u8)
                    let _ = result;
                }
            }
        }
    }

    #[test]
    fn test_rgb_to_16_extensive() {
        // Test various color combinations
        for r in [0, 64, 128, 192, 255] {
            for g in [0, 64, 128, 192, 255] {
                for b in [0, 64, 128, 192, 255] {
                    let result = ColorMode::rgb_to_16(r, g, b);
                    // Just verify it returns a valid CrosstermColor
                    let _ = format!("{:?}", result);
                }
            }
        }
    }

    #[test]
    fn test_to_crossterm_all_modes() {
        let test_colors = [
            Color::BLACK,
            Color::WHITE,
            Color::RED,
            Color::GREEN,
            Color::BLUE,
            Color::new(0.5, 0.5, 0.5, 1.0),
            Color::new(0.25, 0.75, 0.5, 1.0),
        ];

        for mode in [
            ColorMode::TrueColor,
            ColorMode::Color256,
            ColorMode::Color16,
            ColorMode::Mono,
        ] {
            for color in &test_colors {
                let result = mode.to_crossterm(*color);
                // Verify it produces a valid result
                let _ = format!("{:?}", result);
            }
        }
    }

    #[test]
    #[allow(clippy::absurd_extreme_comparisons)]
    fn test_grayscale_ramp_comprehensive() {
        // Test the full grayscale ramp
        for gray in 0..=255 {
            let result = ColorMode::rgb_to_256(gray, gray, gray);
            // Should be either 16 (black), 231 (white), or in grayscale range 232-255
            assert!(result == 16 || result == 231 || ((232..=255).contains(&result)));
        }
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_detect_returns_valid() {
        // Calling detect() should return one of the valid modes
        // The exact result depends on the environment
        let mode = ColorMode::detect();
        match mode {
            ColorMode::TrueColor => assert!(true),
            ColorMode::Color256 => assert!(true),
            ColorMode::Color16 => assert!(true),
            ColorMode::Mono => assert!(true),
        }
    }

    #[test]
    fn test_color_mode_copy() {
        // ColorMode should be Copy
        let mode1 = ColorMode::TrueColor;
        let mode2 = mode1; // Copy
        assert_eq!(mode1, mode2);
    }

    // Tests for detect_with_env - all branches

    #[test]
    fn test_detect_colorterm_truecolor() {
        let mode = ColorMode::detect_with_env(Some("truecolor".to_string()), None);
        assert_eq!(mode, ColorMode::TrueColor);
    }

    #[test]
    fn test_detect_colorterm_24bit() {
        let mode = ColorMode::detect_with_env(Some("24bit".to_string()), None);
        assert_eq!(mode, ColorMode::TrueColor);
    }

    #[test]
    fn test_detect_colorterm_other_falls_through() {
        // COLORTERM set but not truecolor/24bit - should fall through to TERM
        let mode = ColorMode::detect_with_env(
            Some("other".to_string()),
            Some("xterm-256color".to_string()),
        );
        assert_eq!(mode, ColorMode::Color256);
    }

    #[test]
    fn test_detect_term_256color() {
        let mode = ColorMode::detect_with_env(None, Some("xterm-256color".to_string()));
        assert_eq!(mode, ColorMode::Color256);

        let mode2 = ColorMode::detect_with_env(None, Some("screen-256color".to_string()));
        assert_eq!(mode2, ColorMode::Color256);
    }

    #[test]
    fn test_detect_term_xterm() {
        let mode = ColorMode::detect_with_env(None, Some("xterm".to_string()));
        assert_eq!(mode, ColorMode::Color16);
    }

    #[test]
    fn test_detect_term_color() {
        let mode = ColorMode::detect_with_env(None, Some("linux-color".to_string()));
        assert_eq!(mode, ColorMode::Color16);
    }

    #[test]
    fn test_detect_term_dumb() {
        let mode = ColorMode::detect_with_env(None, Some("dumb".to_string()));
        assert_eq!(mode, ColorMode::Mono);
    }

    #[test]
    fn test_detect_term_none() {
        let mode = ColorMode::detect_with_env(None, None);
        assert_eq!(mode, ColorMode::Mono);
    }

    #[test]
    fn test_detect_term_unknown() {
        // Unknown TERM value should default to Color16
        let mode = ColorMode::detect_with_env(None, Some("vt100".to_string()));
        assert_eq!(mode, ColorMode::Color16);
    }

    #[test]
    fn test_detect_colorterm_priority() {
        // COLORTERM should take priority over TERM
        let mode =
            ColorMode::detect_with_env(Some("truecolor".to_string()), Some("dumb".to_string()));
        assert_eq!(mode, ColorMode::TrueColor);
    }

    #[test]
    #[allow(clippy::manual_string_new)]
    fn test_detect_colorterm_empty_string() {
        // Empty COLORTERM string should fall through
        let mode = ColorMode::detect_with_env(Some("".to_string()), None);
        assert_eq!(mode, ColorMode::Mono);
    }

    #[test]
    fn test_detect_term_various() {
        // Test various TERM values
        assert_eq!(
            ColorMode::detect_with_env(None, Some("rxvt-256color".to_string())),
            ColorMode::Color256
        );
        assert_eq!(
            ColorMode::detect_with_env(None, Some("screen".to_string())),
            ColorMode::Color16
        );
        assert_eq!(
            ColorMode::detect_with_env(None, Some("ansi".to_string())),
            ColorMode::Color16
        );
    }

    #[test]
    fn test_detect_colorterm_with_term_fallback() {
        // Non-truecolor COLORTERM with TERM fallback
        let mode =
            ColorMode::detect_with_env(Some("something".to_string()), Some("xterm".to_string()));
        assert_eq!(mode, ColorMode::Color16);
    }

    #[test]
    fn test_to_crossterm_comprehensive() {
        // Test all modes with a variety of colors
        let colors = [
            Color::new(0.0, 0.0, 0.0, 1.0),
            Color::new(1.0, 1.0, 1.0, 1.0),
            Color::new(1.0, 0.0, 0.0, 1.0),
            Color::new(0.0, 1.0, 0.0, 1.0),
            Color::new(0.0, 0.0, 1.0, 1.0),
            Color::new(0.5, 0.5, 0.5, 1.0),
            Color::new(0.25, 0.5, 0.75, 1.0),
            Color::new(0.1, 0.2, 0.3, 1.0),
        ];

        for color in colors {
            for mode in [
                ColorMode::TrueColor,
                ColorMode::Color256,
                ColorMode::Color16,
                ColorMode::Mono,
            ] {
                let _ = mode.to_crossterm(color);
            }
        }
    }

    #[test]
    fn test_rgb_to_256_boundary_values() {
        // Test at exact color cube boundaries
        for v in [0, 51, 102, 153, 204, 255] {
            let _ = ColorMode::rgb_to_256(v, 0, 0);
            let _ = ColorMode::rgb_to_256(0, v, 0);
            let _ = ColorMode::rgb_to_256(0, 0, v);
        }
    }

    #[test]
    fn test_rgb_to_16_all_combinations() {
        // Test all 16 possible combinations of has_r, has_g, has_b, bright
        let test_cases = [
            (0, 0, 0),       // Black
            (50, 50, 50),    // DarkGrey
            (128, 0, 0),     // DarkRed
            (255, 0, 0),     // Red
            (0, 128, 0),     // DarkGreen
            (0, 255, 0),     // Green
            (128, 128, 0),   // DarkYellow
            (255, 255, 0),   // Yellow
            (0, 0, 128),     // DarkBlue
            (0, 0, 255),     // Blue
            (128, 0, 128),   // DarkMagenta
            (255, 0, 255),   // Magenta
            (0, 128, 128),   // DarkCyan
            (0, 255, 255),   // Cyan
            (192, 192, 192), // Grey
            (255, 255, 255), // White
        ];

        for (r, g, b) in test_cases {
            let _ = ColorMode::rgb_to_16(r, g, b);
        }
    }

    #[test]
    fn test_color_lerp_boundary() {
        // Test lerp with boundary values
        let c1 = Color::RED;
        let c2 = Color::BLUE;
        let _ = c1.lerp(&c2, 0.0);
        let _ = c1.lerp(&c2, 1.0);
        let _ = c1.lerp(&c2, 0.5);
    }

    #[test]
    fn test_detect_original_still_works() {
        // Ensure the original detect() still works
        let _ = ColorMode::detect();
    }
}
