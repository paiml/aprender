//! Material Design 3 Color Palette
//!
//! Defines color schemes based on Material Design 3 specification.

use std::fmt;

/// An RGBA color
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    /// Create a new color with full opacity
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Create a new color with custom opacity
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create from hex string (e.g., "#6750A4" or "6750A4")
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return None;
        }

        let r = u8::from_str_radix(hex.get(0..2)?, 16).ok()?;
        let g = u8::from_str_radix(hex.get(2..4)?, 16).ok()?;
        let b = u8::from_str_radix(hex.get(4..6)?, 16).ok()?;

        Some(Self::rgb(r, g, b))
    }

    /// Convert to hex string (without #)
    #[allow(clippy::wrong_self_convention)]
    pub fn to_hex(&self) -> String {
        format!("{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    /// Convert to CSS hex string (with #)
    #[allow(clippy::wrong_self_convention)]
    pub fn to_css_hex(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    /// Convert to CSS rgba string
    #[allow(clippy::wrong_self_convention)]
    pub fn to_css_rgba(&self) -> String {
        if self.a == 255 {
            format!("rgb({}, {}, {})", self.r, self.g, self.b)
        } else {
            format!("rgba({}, {}, {}, {:.2})", self.r, self.g, self.b, self.a as f32 / 255.0)
        }
    }

    /// Apply opacity (0.0 - 1.0) to this color
    pub fn with_opacity(&self, opacity: f32) -> Self {
        Self { r: self.r, g: self.g, b: self.b, a: (opacity.clamp(0.0, 1.0) * 255.0) as u8 }
    }

    /// Lighten the color by a percentage (0.0 - 1.0)
    pub fn lighten(&self, amount: f32) -> Self {
        let amount = amount.clamp(0.0, 1.0);
        Self {
            r: (self.r as f32 + (255.0 - self.r as f32) * amount) as u8,
            g: (self.g as f32 + (255.0 - self.g as f32) * amount) as u8,
            b: (self.b as f32 + (255.0 - self.b as f32) * amount) as u8,
            a: self.a,
        }
    }

    /// Darken the color by a percentage (0.0 - 1.0)
    pub fn darken(&self, amount: f32) -> Self {
        let amount = amount.clamp(0.0, 1.0);
        Self {
            r: (self.r as f32 * (1.0 - amount)) as u8,
            g: (self.g as f32 * (1.0 - amount)) as u8,
            b: (self.b as f32 * (1.0 - amount)) as u8,
            a: self.a,
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_css_hex())
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::rgb(0, 0, 0)
    }
}

/// Material Design 3 color palette
#[derive(Debug, Clone)]
pub struct MaterialPalette {
    /// Primary color - #6750A4
    pub primary: Color,
    /// On-primary text color
    pub on_primary: Color,
    /// Primary container
    pub primary_container: Color,
    /// On-primary container text
    pub on_primary_container: Color,

    /// Secondary color
    pub secondary: Color,
    /// On-secondary text color
    pub on_secondary: Color,

    /// Tertiary color
    pub tertiary: Color,
    /// On-tertiary text color
    pub on_tertiary: Color,

    /// Error color
    pub error: Color,
    /// On-error text color
    pub on_error: Color,

    /// Surface color - #FFFBFE
    pub surface: Color,
    /// On-surface text color
    pub on_surface: Color,
    /// Surface variant
    pub surface_variant: Color,
    /// On-surface variant text
    pub on_surface_variant: Color,

    /// Outline color - #79747E
    pub outline: Color,
    /// Outline variant (lighter)
    pub outline_variant: Color,

    /// Background color
    pub background: Color,
    /// On-background text color
    pub on_background: Color,
}

impl MaterialPalette {
    /// Create the default Material Design 3 light palette
    pub fn light() -> Self {
        Self {
            // Primary (Purple)
            primary: Color::rgb(103, 80, 164),
            on_primary: Color::rgb(255, 255, 255),
            primary_container: Color::rgb(234, 221, 255),
            on_primary_container: Color::rgb(33, 0, 93),

            // Secondary (Pink-purple)
            secondary: Color::rgb(98, 91, 113),
            on_secondary: Color::rgb(255, 255, 255),

            // Tertiary (Teal)
            tertiary: Color::rgb(125, 82, 96),
            on_tertiary: Color::rgb(255, 255, 255),

            // Error (Red)
            error: Color::rgb(179, 38, 30),
            on_error: Color::rgb(255, 255, 255),

            // Surface
            surface: Color::rgb(255, 251, 254),
            on_surface: Color::rgb(28, 27, 31),
            surface_variant: Color::rgb(231, 224, 236),
            on_surface_variant: Color::rgb(73, 69, 79),

            // Outline
            outline: Color::rgb(121, 116, 126),
            outline_variant: Color::rgb(202, 196, 208),

            // Background
            background: Color::rgb(255, 251, 254),
            on_background: Color::rgb(28, 27, 31),
        }
    }

    /// Create the Material Design 3 dark palette
    pub fn dark() -> Self {
        Self {
            // Primary (Purple)
            primary: Color::rgb(208, 188, 255),
            on_primary: Color::rgb(56, 30, 114),
            primary_container: Color::rgb(79, 55, 139),
            on_primary_container: Color::rgb(234, 221, 255),

            // Secondary
            secondary: Color::rgb(204, 194, 220),
            on_secondary: Color::rgb(51, 45, 65),

            // Tertiary
            tertiary: Color::rgb(239, 184, 200),
            on_tertiary: Color::rgb(73, 37, 50),

            // Error
            error: Color::rgb(242, 184, 181),
            on_error: Color::rgb(96, 20, 16),

            // Surface
            surface: Color::rgb(28, 27, 31),
            on_surface: Color::rgb(230, 225, 229),
            surface_variant: Color::rgb(73, 69, 79),
            on_surface_variant: Color::rgb(202, 196, 208),

            // Outline
            outline: Color::rgb(147, 143, 153),
            outline_variant: Color::rgb(73, 69, 79),

            // Background
            background: Color::rgb(28, 27, 31),
            on_background: Color::rgb(230, 225, 229),
        }
    }

    /// Create a custom palette with a primary color
    pub fn with_primary(primary: Color) -> Self {
        let mut palette = Self::light();
        palette.primary = primary;
        palette
    }

    /// Validate that a color is in this palette
    pub fn is_valid_color(&self, color: &Color) -> bool {
        color == &self.primary
            || color == &self.on_primary
            || color == &self.primary_container
            || color == &self.on_primary_container
            || color == &self.secondary
            || color == &self.on_secondary
            || color == &self.tertiary
            || color == &self.on_tertiary
            || color == &self.error
            || color == &self.on_error
            || color == &self.surface
            || color == &self.on_surface
            || color == &self.surface_variant
            || color == &self.on_surface_variant
            || color == &self.outline
            || color == &self.outline_variant
            || color == &self.background
            || color == &self.on_background
    }

    /// Get all colors in the palette
    pub fn all_colors(&self) -> Vec<Color> {
        vec![
            self.primary,
            self.on_primary,
            self.primary_container,
            self.on_primary_container,
            self.secondary,
            self.on_secondary,
            self.tertiary,
            self.on_tertiary,
            self.error,
            self.on_error,
            self.surface,
            self.on_surface,
            self.surface_variant,
            self.on_surface_variant,
            self.outline,
            self.outline_variant,
            self.background,
            self.on_background,
        ]
    }
}

impl Default for MaterialPalette {
    fn default() -> Self {
        Self::light()
    }
}

/// Sovereign AI Stack color scheme (extends Material Design 3)
#[derive(Debug, Clone)]
pub struct SovereignPalette {
    /// Base Material palette
    pub material: MaterialPalette,

    /// Trueno component color (orange)
    pub trueno: Color,
    /// Aprender component color (blue)
    pub aprender: Color,
    /// Realizar component color (green)
    pub realizar: Color,
    /// Batuta component color (purple)
    pub batuta: Color,

    /// Success color
    pub success: Color,
    /// Warning color
    pub warning: Color,
    /// Info color
    pub info: Color,
}

impl SovereignPalette {
    /// Create the light sovereign palette
    pub fn light() -> Self {
        Self {
            material: MaterialPalette::light(),
            trueno: Color::rgb(255, 109, 0),   // Deep Orange A400
            aprender: Color::rgb(41, 98, 255), // Blue A700
            realizar: Color::rgb(0, 200, 83),  // Green A700
            batuta: Color::rgb(103, 80, 164),  // Primary Purple
            success: Color::rgb(0, 200, 83),   // Green A700
            warning: Color::rgb(255, 214, 0),  // Yellow A700
            info: Color::rgb(0, 176, 255),     // Light Blue A400
        }
    }

    /// Create the dark sovereign palette
    pub fn dark() -> Self {
        Self {
            material: MaterialPalette::dark(),
            trueno: Color::rgb(255, 171, 64),    // Orange A200
            aprender: Color::rgb(130, 177, 255), // Blue A100
            realizar: Color::rgb(105, 240, 174), // Green A200
            batuta: Color::rgb(208, 188, 255),   // Primary Purple
            success: Color::rgb(105, 240, 174),  // Green A200
            warning: Color::rgb(255, 229, 127),  // Amber A200
            info: Color::rgb(128, 216, 255),     // Light Blue A100
        }
    }

    /// Get color for a stack component
    pub fn component_color(&self, component: &str) -> Color {
        match component.to_lowercase().as_str() {
            "trueno" => self.trueno,
            "aprender" => self.aprender,
            "realizar" => self.realizar,
            "batuta" => self.batuta,
            _ => self.material.outline,
        }
    }
}

impl Default for SovereignPalette {
    fn default() -> Self {
        Self::light()
    }
}

/// Pre-verified video palette for 1080p presentation SVGs.
///
/// All text/background pairings meet WCAG AA 4.5:1 contrast ratio.
/// Forbidden pairings are documented and checked by the linter.
#[derive(Debug, Clone)]
pub struct VideoPalette {
    // Backgrounds
    /// Canvas background
    pub canvas: Color,
    /// Surface (card/box) background
    pub surface: Color,
    /// Grey badge background
    pub badge_grey: Color,
    /// Blue badge background
    pub badge_blue: Color,
    /// Green badge background
    pub badge_green: Color,
    /// Gold badge background
    pub badge_gold: Color,

    // Text
    /// Primary heading text
    pub heading: Color,
    /// Secondary heading text
    pub heading_secondary: Color,
    /// Body text
    pub body: Color,
    /// Blue accent text
    pub accent_blue: Color,
    /// Green accent text
    pub accent_green: Color,
    /// Gold accent text
    pub accent_gold: Color,
    /// Red accent text
    pub accent_red: Color,

    // Strokes
    /// Outline/stroke color
    pub outline: Color,
}

impl VideoPalette {
    /// Dark palette — light text on dark backgrounds.
    pub fn dark() -> Self {
        Self {
            canvas: Color::from_hex("#0f172a").expect("valid hex"),
            surface: Color::from_hex("#1e293b").expect("valid hex"),
            badge_grey: Color::from_hex("#374151").expect("valid hex"),
            badge_blue: Color::from_hex("#1e3a5f").expect("valid hex"),
            badge_green: Color::from_hex("#14532d").expect("valid hex"),
            badge_gold: Color::from_hex("#713f12").expect("valid hex"),
            heading: Color::from_hex("#f1f5f9").expect("valid hex"),
            heading_secondary: Color::from_hex("#d1d5db").expect("valid hex"),
            body: Color::from_hex("#94a3b8").expect("valid hex"),
            accent_blue: Color::from_hex("#60a5fa").expect("valid hex"),
            accent_green: Color::from_hex("#4ade80").expect("valid hex"),
            accent_gold: Color::from_hex("#fde047").expect("valid hex"),
            accent_red: Color::from_hex("#ef4444").expect("valid hex"),
            outline: Color::from_hex("#475569").expect("valid hex"),
        }
    }

    /// Light palette — dark text on light backgrounds.
    pub fn light() -> Self {
        Self {
            canvas: Color::from_hex("#f8fafc").expect("valid hex"),
            surface: Color::from_hex("#ffffff").expect("valid hex"),
            badge_grey: Color::from_hex("#e5e7eb").expect("valid hex"),
            badge_blue: Color::from_hex("#dbeafe").expect("valid hex"),
            badge_green: Color::from_hex("#dcfce7").expect("valid hex"),
            badge_gold: Color::from_hex("#fef9c3").expect("valid hex"),
            heading: Color::from_hex("#0f172a").expect("valid hex"),
            heading_secondary: Color::from_hex("#374151").expect("valid hex"),
            body: Color::from_hex("#475569").expect("valid hex"),
            accent_blue: Color::from_hex("#2563eb").expect("valid hex"),
            accent_green: Color::from_hex("#16a34a").expect("valid hex"),
            accent_gold: Color::from_hex("#ca8a04").expect("valid hex"),
            accent_red: Color::from_hex("#dc2626").expect("valid hex"),
            outline: Color::from_hex("#94a3b8").expect("valid hex"),
        }
    }

    /// Check if a text/background pairing meets WCAG AA 4.5:1 contrast.
    pub fn verify_contrast(text: &Color, bg: &Color) -> bool {
        contrast_ratio(text, bg) >= 4.5
    }
}

impl Default for VideoPalette {
    fn default() -> Self {
        Self::dark()
    }
}

/// Known-bad color pairings that fail WCAG AA 4.5:1 contrast.
pub const FORBIDDEN_PAIRINGS: &[(&str, &str)] = &[
    ("#64748b", "#0f172a"), // slate-500 on navy: ~3.75:1
    ("#6b7280", "#1e293b"), // grey-500 on slate: ~3.03:1
    ("#3b82f6", "#1e293b"), // blue-500 on slate: ~3.98:1
    ("#475569", "#0f172a"), // slate-600 on navy: ~2.58:1
];

/// Calculate WCAG relative luminance for a color channel (sRGB).
fn channel_luminance(c: u8) -> f64 {
    let c = c as f64 / 255.0;
    if c <= 0.03928 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Calculate WCAG relative luminance for a color.
fn relative_luminance(color: &Color) -> f64 {
    0.2126 * channel_luminance(color.r)
        + 0.7152 * channel_luminance(color.g)
        + 0.0722 * channel_luminance(color.b)
}

/// Calculate WCAG contrast ratio between two colors.
pub fn contrast_ratio(c1: &Color, c2: &Color) -> f64 {
    let l1 = relative_luminance(c1);
    let l2 = relative_luminance(c2);
    let lighter = l1.max(l2);
    let darker = l1.min(l2);
    (lighter + 0.05) / (darker + 0.05)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_hex() {
        let color = Color::rgb(103, 80, 164);
        assert_eq!(color.r, 103);
        assert_eq!(color.g, 80);
        assert_eq!(color.b, 164);
    }

    #[test]
    fn test_color_from_hex_no_hash() {
        let color = Color::from_hex("6750A4").expect("unexpected failure");
        assert_eq!(color.r, 103);
        assert_eq!(color.g, 80);
        assert_eq!(color.b, 164);
    }

    #[test]
    fn test_color_to_hex() {
        let color = Color::rgb(103, 80, 164);
        assert_eq!(color.to_hex(), "6750A4");
        assert_eq!(color.to_css_hex(), "#6750A4");
    }

    #[test]
    fn test_color_to_css_rgba() {
        let color = Color::rgb(103, 80, 164);
        assert_eq!(color.to_css_rgba(), "rgb(103, 80, 164)");

        let color_alpha = Color::rgba(103, 80, 164, 128);
        assert!(color_alpha.to_css_rgba().starts_with("rgba(103, 80, 164,"));
    }

    #[test]
    fn test_color_with_opacity() {
        let color = Color::rgb(255, 255, 255);
        let semi = color.with_opacity(0.5);
        assert_eq!(semi.a, 127);
    }

    #[test]
    fn test_color_lighten() {
        let color = Color::rgb(100, 100, 100);
        let lighter = color.lighten(0.5);
        assert!(lighter.r > color.r);
        assert!(lighter.g > color.g);
        assert!(lighter.b > color.b);
    }

    #[test]
    fn test_color_darken() {
        let color = Color::rgb(100, 100, 100);
        let darker = color.darken(0.5);
        assert!(darker.r < color.r);
        assert!(darker.g < color.g);
        assert!(darker.b < color.b);
    }

    #[test]
    fn test_material_palette_light() {
        let palette = MaterialPalette::light();
        assert_eq!(palette.primary.to_css_hex(), "#6750A4");
        assert_eq!(palette.surface.to_css_hex(), "#FFFBFE");
        assert_eq!(palette.outline.to_css_hex(), "#79747E");
    }

    #[test]
    fn test_material_palette_dark() {
        let palette = MaterialPalette::dark();
        assert_eq!(palette.primary.to_css_hex(), "#D0BCFF");
        assert_eq!(palette.surface.to_css_hex(), "#1C1B1F");
    }

    #[test]
    fn test_material_palette_validation() {
        let palette = MaterialPalette::light();
        assert!(palette.is_valid_color(&palette.primary));
        assert!(palette.is_valid_color(&palette.surface));
        assert!(!palette.is_valid_color(&Color::rgb(1, 2, 3)));
    }

    #[test]
    fn test_sovereign_palette() {
        let palette = SovereignPalette::light();
        assert_eq!(palette.trueno.to_css_hex(), "#FF6D00");
        assert_eq!(palette.aprender.to_css_hex(), "#2962FF");
        assert_eq!(palette.realizar.to_css_hex(), "#00C853");
    }

    #[test]
    fn test_sovereign_component_color() {
        let palette = SovereignPalette::light();
        assert_eq!(palette.component_color("trueno"), palette.trueno);
        assert_eq!(palette.component_color("APRENDER"), palette.aprender);
        assert_eq!(palette.component_color("unknown"), palette.material.outline);
    }

    #[test]
    fn test_color_display() {
        let color = Color::rgb(103, 80, 164);
        assert_eq!(format!("{}", color), "#6750A4");
    }

    #[test]
    fn test_color_default() {
        let color = Color::default();
        assert_eq!(color.r, 0);
        assert_eq!(color.g, 0);
        assert_eq!(color.b, 0);
        assert_eq!(color.a, 255);
    }

    #[test]
    fn test_material_palette_with_primary() {
        let custom_primary = Color::rgb(255, 0, 0);
        let palette = MaterialPalette::with_primary(custom_primary);
        assert_eq!(palette.primary, custom_primary);
    }

    #[test]
    fn test_material_palette_all_colors() {
        let palette = MaterialPalette::light();
        let colors = palette.all_colors();
        assert_eq!(colors.len(), 18);
        assert!(colors.contains(&palette.primary));
        assert!(colors.contains(&palette.surface));
        assert!(colors.contains(&palette.error));
    }

    #[test]
    fn test_material_palette_default() {
        let palette = MaterialPalette::default();
        let light = MaterialPalette::light();
        assert_eq!(palette.primary, light.primary);
    }

    #[test]
    fn test_sovereign_palette_dark() {
        let palette = SovereignPalette::dark();
        assert_eq!(palette.trueno.to_css_hex(), "#FFAB40");
        assert_eq!(palette.aprender.to_css_hex(), "#82B1FF");
    }

    #[test]
    fn test_sovereign_palette_default() {
        let palette = SovereignPalette::default();
        let light = SovereignPalette::light();
        assert_eq!(palette.trueno, light.trueno);
    }

    #[test]
    fn test_color_from_hex_invalid_length() {
        assert!(Color::from_hex("#12").is_none());
        assert!(Color::from_hex("#1234567").is_none());
    }

    #[test]
    fn test_color_from_hex_invalid_chars() {
        assert!(Color::from_hex("#GGHHII").is_none());
    }

    #[test]
    fn test_color_equality() {
        let c1 = Color::rgb(100, 200, 50);
        let c2 = Color::rgb(100, 200, 50);
        let c3 = Color::rgb(100, 200, 51);
        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_color_lighten_clamp() {
        let white = Color::rgb(255, 255, 255);
        let lightened = white.lighten(0.5);
        assert_eq!(lightened.r, 255);
        assert_eq!(lightened.g, 255);
        assert_eq!(lightened.b, 255);
    }

    #[test]
    fn test_color_darken_clamp() {
        let black = Color::rgb(0, 0, 0);
        let darkened = black.darken(0.5);
        assert_eq!(darkened.r, 0);
        assert_eq!(darkened.g, 0);
        assert_eq!(darkened.b, 0);
    }

    #[test]
    fn test_color_with_opacity_clamp() {
        let color = Color::rgb(100, 100, 100);
        let over = color.with_opacity(1.5);
        assert_eq!(over.a, 255);
        let under = color.with_opacity(-0.5);
        assert_eq!(under.a, 0);
    }

    // ── VideoPalette tests ──────────────────────────────────────────────

    #[test]
    fn test_video_palette_dark() {
        let vp = VideoPalette::dark();
        assert_eq!(vp.canvas.to_css_hex(), "#0F172A");
        assert_eq!(vp.surface.to_css_hex(), "#1E293B");
        assert_eq!(vp.heading.to_css_hex(), "#F1F5F9");
    }

    #[test]
    fn test_video_palette_light() {
        let vp = VideoPalette::light();
        assert_eq!(vp.canvas.to_css_hex(), "#F8FAFC");
        assert_eq!(vp.surface.to_css_hex(), "#FFFFFF");
        assert_eq!(vp.heading.to_css_hex(), "#0F172A");
    }

    #[test]
    fn test_video_palette_default() {
        let vp = VideoPalette::default();
        // Default is dark
        assert_eq!(vp.canvas, VideoPalette::dark().canvas);
    }

    #[test]
    fn test_video_palette_verify_contrast_passes() {
        let dark = VideoPalette::dark();
        // heading (#f1f5f9) on canvas (#0f172a) should pass
        assert!(VideoPalette::verify_contrast(&dark.heading, &dark.canvas));
        // heading (#f1f5f9) on surface (#1e293b) should pass
        assert!(VideoPalette::verify_contrast(&dark.heading, &dark.surface));
        // accent_gold (#fde047) on canvas (#0f172a) should pass
        assert!(VideoPalette::verify_contrast(&dark.accent_gold, &dark.canvas));
    }

    #[test]
    fn test_video_palette_verify_contrast_fails_for_forbidden() {
        for (text_hex, bg_hex) in FORBIDDEN_PAIRINGS {
            let text = Color::from_hex(text_hex).expect("unexpected failure");
            let bg = Color::from_hex(bg_hex).expect("unexpected failure");
            assert!(
                !VideoPalette::verify_contrast(&text, &bg),
                "Expected forbidden pairing {} on {} to fail contrast check, ratio: {:.2}",
                text_hex,
                bg_hex,
                contrast_ratio(&text, &bg)
            );
        }
    }

    #[test]
    fn test_contrast_ratio_black_on_white() {
        let ratio = contrast_ratio(&Color::rgb(0, 0, 0), &Color::rgb(255, 255, 255));
        assert!(ratio > 20.0 && ratio < 22.0, "Expected ~21:1, got {:.2}", ratio);
    }

    #[test]
    fn test_contrast_ratio_same_color() {
        let c = Color::rgb(128, 128, 128);
        let ratio = contrast_ratio(&c, &c);
        assert!((ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_forbidden_pairings_count() {
        assert_eq!(FORBIDDEN_PAIRINGS.len(), 4);
    }

    #[test]
    fn test_video_palette_light_contrast() {
        let light = VideoPalette::light();
        // heading (#0f172a) on canvas (#f8fafc) should pass
        assert!(VideoPalette::verify_contrast(&light.heading, &light.canvas));
        // body (#475569) on surface (#ffffff) should pass
        assert!(VideoPalette::verify_contrast(&light.body, &light.surface));
    }
}
