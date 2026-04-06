//! Shared Style Constants for TUI/WASM Parity
//!
//! This module provides unified color and style constants to ensure
//! visual consistency between TUI (ratatui) and WASM (Canvas/CSS) renders.
//!
//! # Architecture (OR-001 Compliant)
//!
//! All visual constants are defined ONCE here and used by both:
//! - TUI: via `ratatui::style::Color` conversions
//! - WASM: via CSS variable injection / Canvas fill styles
//!
//! # References
//!
//! - WCAG 2.1 AA contrast requirements
//! - TPS Visual Control (見える化)

/// Primary background color (darkest)
pub const BG_PRIMARY: &str = "#0a0a1a";
/// Secondary background color (panels)
pub const BG_SECONDARY: &str = "#1a1a2e";
/// Tertiary background color (canvas/content)
pub const BG_TERTIARY: &str = "#0f0f23";

/// Primary border color
pub const BORDER: &str = "#333333";

/// Primary text color (high contrast)
pub const TEXT_PRIMARY: &str = "#e0e0e0";
/// Secondary text color (labels, muted)
pub const TEXT_SECONDARY: &str = "#888888";

/// Accent color (teal/cyan) - success, highlights
pub const ACCENT: &str = "#4ecdc4";
/// Warning color (yellow/gold)
pub const ACCENT_WARN: &str = "#ffd93d";
/// Error color (red/coral)
pub const ACCENT_ERROR: &str = "#ff6b6b";

/// Probar brand color (testing indicator)
pub const PROBAR: &str = "#ff6b6b";

/// City node color (yellow dots on map)
pub const CITY_NODE: &str = "#ffd93d";
/// Tour path color (teal lines)
pub const TOUR_PATH: &str = "#4ecdc4";

/// Font family for monospace text
pub const FONT_MONO: &str = "'JetBrains Mono', 'Fira Code', monospace";
/// Font size base (rem)
pub const FONT_SIZE_BASE: f64 = 0.75;
/// Font size small (rem)
pub const FONT_SIZE_SMALL: f64 = 0.65;

/// Parse hex color to RGB tuple
#[must_use]
pub const fn hex_to_rgb(hex: &str) -> (u8, u8, u8) {
    // Skip the '#' prefix
    let bytes = hex.as_bytes();
    let offset = if bytes[0] == b'#' { 1 } else { 0 };

    let r = hex_byte(bytes[offset], bytes[offset + 1]);
    let g = hex_byte(bytes[offset + 2], bytes[offset + 3]);
    let b = hex_byte(bytes[offset + 4], bytes[offset + 5]);

    (r, g, b)
}

const fn hex_byte(hi: u8, lo: u8) -> u8 {
    hex_digit(hi) * 16 + hex_digit(lo)
}

const fn hex_digit(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => 0,
    }
}

/// Generate CSS custom properties string for injection
#[must_use]
pub fn css_variables() -> String {
    format!(
        r":root {{
    --bg-primary: {BG_PRIMARY};
    --bg-secondary: {BG_SECONDARY};
    --bg-tertiary: {BG_TERTIARY};
    --border: {BORDER};
    --text-primary: {TEXT_PRIMARY};
    --text-secondary: {TEXT_SECONDARY};
    --accent: {ACCENT};
    --accent-warn: {ACCENT_WARN};
    --accent-error: {ACCENT_ERROR};
    --probar: {PROBAR};
}}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_to_rgb_with_hash() {
        assert_eq!(hex_to_rgb("#4ecdc4"), (78, 205, 196));
        assert_eq!(hex_to_rgb("#ffd93d"), (255, 217, 61));
        assert_eq!(hex_to_rgb("#ff6b6b"), (255, 107, 107));
    }

    #[test]
    fn test_hex_to_rgb_bg_colors() {
        assert_eq!(hex_to_rgb(BG_PRIMARY), (10, 10, 26));
        assert_eq!(hex_to_rgb(BG_SECONDARY), (26, 26, 46));
    }

    #[test]
    fn test_css_variables_contains_all_colors() {
        let css = css_variables();
        assert!(css.contains("--bg-primary"));
        assert!(css.contains("--accent"));
        assert!(css.contains("#4ecdc4"));
    }

    #[test]
    fn test_colors_are_valid_hex() {
        // All color constants should be 7 chars (#RRGGBB)
        assert_eq!(BG_PRIMARY.len(), 7);
        assert_eq!(ACCENT.len(), 7);
        assert_eq!(ACCENT_WARN.len(), 7);
        assert!(BG_PRIMARY.starts_with('#'));
    }

    #[test]
    fn test_contrast_accessibility() {
        // WCAG AA requires 4.5:1 contrast ratio for text
        // TEXT_PRIMARY (#e0e0e0) on BG_PRIMARY (#0a0a1a) should pass
        let (tr, tg, tb) = hex_to_rgb(TEXT_PRIMARY);
        let (br, bg, bb) = hex_to_rgb(BG_PRIMARY);

        // Simplified relative luminance (not exact WCAG formula)
        let text_lum = (f64::from(tr) + f64::from(tg) + f64::from(tb)) / 3.0 / 255.0;
        let bg_lum = (f64::from(br) + f64::from(bg) + f64::from(bb)) / 3.0 / 255.0;

        // Contrast ratio approximation
        let ratio = (text_lum + 0.05) / (bg_lum + 0.05);
        assert!(
            ratio > 4.5,
            "Text contrast ratio should exceed WCAG AA: {ratio}"
        );
    }
}
