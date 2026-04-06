//! Theme system for consistent styling.

use crate::color::Color;
use serde::{Deserialize, Serialize};

/// A color palette for theming.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColorPalette {
    /// Primary brand color
    pub primary: Color,
    /// Secondary brand color
    pub secondary: Color,
    /// Surface/background color
    pub surface: Color,
    /// Background color
    pub background: Color,
    /// Error/danger color
    pub error: Color,
    /// Warning color
    pub warning: Color,
    /// Success color
    pub success: Color,
    /// Text on primary
    pub on_primary: Color,
    /// Text on secondary
    pub on_secondary: Color,
    /// Text on surface
    pub on_surface: Color,
    /// Text on background
    pub on_background: Color,
    /// Text on error
    pub on_error: Color,
}

impl Default for ColorPalette {
    fn default() -> Self {
        Self::light()
    }
}

/// Result of a WCAG contrast check.
#[derive(Debug, Clone)]
pub struct ContrastCheck {
    /// Name of the color pair
    pub name: String,
    /// Foreground color
    pub foreground: Color,
    /// Background color
    pub background: Color,
    /// Calculated contrast ratio
    pub ratio: f32,
    /// Passes WCAG AA for normal text (4.5:1)
    pub passes_aa: bool,
    /// Passes WCAG AAA for normal text (7:1)
    pub passes_aaa: bool,
}

impl ColorPalette {
    /// Check all foreground/background combinations for WCAG compliance.
    /// Returns a list of contrast checks for each semantic color pair.
    #[must_use]
    pub fn check_contrast(&self) -> Vec<ContrastCheck> {
        let checks = [
            ("on_primary/primary", self.on_primary, self.primary),
            ("on_secondary/secondary", self.on_secondary, self.secondary),
            ("on_surface/surface", self.on_surface, self.surface),
            (
                "on_background/background",
                self.on_background,
                self.background,
            ),
            ("on_error/error", self.on_error, self.error),
        ];

        checks
            .into_iter()
            .map(|(name, fg, bg)| {
                let ratio = fg.contrast_ratio(&bg);
                ContrastCheck {
                    name: name.to_string(),
                    foreground: fg,
                    background: bg,
                    ratio,
                    passes_aa: ratio >= 4.5,
                    passes_aaa: ratio >= 7.0,
                }
            })
            .collect()
    }

    /// Check if all color pairs pass WCAG AA.
    #[must_use]
    pub fn passes_wcag_aa(&self) -> bool {
        self.check_contrast().iter().all(|c| c.passes_aa)
    }

    /// Check if all color pairs pass WCAG AAA.
    #[must_use]
    pub fn passes_wcag_aaa(&self) -> bool {
        self.check_contrast().iter().all(|c| c.passes_aaa)
    }

    /// Get any failing contrast pairs for WCAG AA.
    #[must_use]
    pub fn failing_aa(&self) -> Vec<ContrastCheck> {
        self.check_contrast()
            .into_iter()
            .filter(|c| !c.passes_aa)
            .collect()
    }

    /// Create a light color palette.
    /// All color combinations pass WCAG AA (4.5:1 contrast ratio).
    #[must_use]
    pub fn light() -> Self {
        Self {
            primary: Color::new(0.0, 0.35, 0.75, 1.0), // Darker blue for AA compliance
            secondary: Color::new(0.0, 0.40, 0.60, 1.0), // Darker teal for AA compliance
            surface: Color::WHITE,
            background: Color::new(0.98, 0.98, 0.98, 1.0), // Light gray
            error: Color::new(0.69, 0.18, 0.18, 1.0),      // Red (passes with white)
            warning: Color::new(0.70, 0.45, 0.0, 1.0),     // Darker orange for AA
            success: Color::new(0.18, 0.55, 0.34, 1.0),    // Green
            on_primary: Color::WHITE,
            on_secondary: Color::WHITE,
            on_surface: Color::new(0.13, 0.13, 0.13, 1.0), // Dark gray
            on_background: Color::new(0.13, 0.13, 0.13, 1.0),
            on_error: Color::WHITE,
        }
    }

    /// Create a dark color palette.
    #[must_use]
    pub fn dark() -> Self {
        Self {
            primary: Color::new(0.51, 0.71, 1.0, 1.0),     // Light blue
            secondary: Color::new(0.31, 0.82, 0.71, 1.0),  // Teal
            surface: Color::new(0.14, 0.14, 0.14, 1.0),    // Dark gray
            background: Color::new(0.07, 0.07, 0.07, 1.0), // Near black
            error: Color::new(0.94, 0.47, 0.47, 1.0),      // Light red
            warning: Color::new(1.0, 0.78, 0.35, 1.0),     // Light orange
            success: Color::new(0.51, 0.78, 0.58, 1.0),    // Light green
            on_primary: Color::BLACK,
            on_secondary: Color::BLACK,
            on_surface: Color::WHITE,
            on_background: Color::WHITE,
            on_error: Color::BLACK,
        }
    }
}

/// Typography scale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Typography {
    /// Base font size
    pub base_size: f32,
    /// H1 scale (relative to base)
    pub h1_scale: f32,
    /// H2 scale
    pub h2_scale: f32,
    /// H3 scale
    pub h3_scale: f32,
    /// H4 scale
    pub h4_scale: f32,
    /// H5 scale
    pub h5_scale: f32,
    /// H6 scale
    pub h6_scale: f32,
    /// Body scale
    pub body_scale: f32,
    /// Caption scale
    pub caption_scale: f32,
    /// Line height
    pub line_height: f32,
}

impl Default for Typography {
    fn default() -> Self {
        Self::standard()
    }
}

impl Typography {
    /// Standard typography scale (based on 16px base).
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            base_size: 16.0,
            h1_scale: 2.5,       // 40px
            h2_scale: 2.0,       // 32px
            h3_scale: 1.75,      // 28px
            h4_scale: 1.5,       // 24px
            h5_scale: 1.25,      // 20px
            h6_scale: 1.125,     // 18px
            body_scale: 1.0,     // 16px
            caption_scale: 0.75, // 12px
            line_height: 1.5,
        }
    }

    /// Compact typography scale (based on 14px base).
    #[must_use]
    pub const fn compact() -> Self {
        Self {
            base_size: 14.0,
            h1_scale: 2.286,      // 32px
            h2_scale: 1.857,      // 26px
            h3_scale: 1.571,      // 22px
            h4_scale: 1.286,      // 18px
            h5_scale: 1.143,      // 16px
            h6_scale: 1.0,        // 14px
            body_scale: 1.0,      // 14px
            caption_scale: 0.786, // 11px
            line_height: 1.4,
        }
    }

    /// Get size for a heading level (1-6).
    #[must_use]
    pub fn heading_size(&self, level: u8) -> f32 {
        let scale = match level {
            1 => self.h1_scale,
            2 => self.h2_scale,
            3 => self.h3_scale,
            4 => self.h4_scale,
            5 => self.h5_scale,
            _ => self.h6_scale,
        };
        self.base_size * scale
    }

    /// Get body text size.
    #[must_use]
    pub fn body_size(&self) -> f32 {
        self.base_size * self.body_scale
    }

    /// Get caption text size.
    #[must_use]
    pub fn caption_size(&self) -> f32 {
        self.base_size * self.caption_scale
    }
}

/// Spacing scale.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Spacing {
    /// Base spacing unit
    pub unit: f32,
}

impl Default for Spacing {
    fn default() -> Self {
        Self::standard()
    }
}

impl Spacing {
    /// Standard spacing (8px base unit).
    #[must_use]
    pub const fn standard() -> Self {
        Self { unit: 8.0 }
    }

    /// Compact spacing (4px base unit).
    #[must_use]
    pub const fn compact() -> Self {
        Self { unit: 4.0 }
    }

    /// Get spacing for a given multiplier.
    #[must_use]
    pub fn get(&self, multiplier: f32) -> f32 {
        self.unit * multiplier
    }

    /// None/zero spacing.
    #[must_use]
    pub const fn none(&self) -> f32 {
        0.0
    }

    /// Extra small spacing (0.5x).
    #[must_use]
    pub fn xs(&self) -> f32 {
        self.unit * 0.5
    }

    /// Small spacing (1x).
    #[must_use]
    pub const fn sm(&self) -> f32 {
        self.unit
    }

    /// Medium spacing (2x).
    #[must_use]
    pub fn md(&self) -> f32 {
        self.unit * 2.0
    }

    /// Large spacing (3x).
    #[must_use]
    pub fn lg(&self) -> f32 {
        self.unit * 3.0
    }

    /// Extra large spacing (4x).
    #[must_use]
    pub fn xl(&self) -> f32 {
        self.unit * 4.0
    }

    /// 2XL spacing (6x).
    #[must_use]
    pub fn xl2(&self) -> f32 {
        self.unit * 6.0
    }

    /// 3XL spacing (8x).
    #[must_use]
    pub fn xl3(&self) -> f32 {
        self.unit * 8.0
    }
}

/// Border radius presets.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Radii {
    /// Base radius unit
    pub unit: f32,
}

impl Default for Radii {
    fn default() -> Self {
        Self::standard()
    }
}

impl Radii {
    /// Standard radii (4px base).
    #[must_use]
    pub const fn standard() -> Self {
        Self { unit: 4.0 }
    }

    /// No radius.
    #[must_use]
    pub const fn none(&self) -> f32 {
        0.0
    }

    /// Small radius (1x).
    #[must_use]
    pub const fn sm(&self) -> f32 {
        self.unit
    }

    /// Medium radius (2x).
    #[must_use]
    pub fn md(&self) -> f32 {
        self.unit * 2.0
    }

    /// Large radius (3x).
    #[must_use]
    pub fn lg(&self) -> f32 {
        self.unit * 3.0
    }

    /// Extra large radius (4x).
    #[must_use]
    pub fn xl(&self) -> f32 {
        self.unit * 4.0
    }

    /// Full/pill radius.
    #[must_use]
    pub const fn full(&self) -> f32 {
        9999.0
    }
}

/// Shadow presets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shadows {
    /// Shadow color
    pub color: Color,
}

impl Default for Shadows {
    fn default() -> Self {
        Self::standard()
    }
}

impl Shadows {
    /// Standard shadows.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            color: Color::new(0.0, 0.0, 0.0, 0.1),
        }
    }

    /// Small shadow parameters (blur, y offset).
    #[must_use]
    pub const fn sm(&self) -> (f32, f32) {
        (2.0, 1.0)
    }

    /// Medium shadow parameters.
    #[must_use]
    pub const fn md(&self) -> (f32, f32) {
        (4.0, 2.0)
    }

    /// Large shadow parameters.
    #[must_use]
    pub const fn lg(&self) -> (f32, f32) {
        (8.0, 4.0)
    }

    /// XL shadow parameters.
    #[must_use]
    pub const fn xl(&self) -> (f32, f32) {
        (16.0, 8.0)
    }
}

/// Complete theme definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Theme {
    /// Theme name
    pub name: String,
    /// Color palette
    pub colors: ColorPalette,
    /// Typography
    pub typography: Typography,
    /// Spacing
    pub spacing: Spacing,
    /// Border radii
    pub radii: Radii,
    /// Shadows
    pub shadows: Shadows,
}

impl Default for Theme {
    fn default() -> Self {
        Self::light()
    }
}

impl Theme {
    /// Create a light theme.
    #[must_use]
    pub fn light() -> Self {
        Self {
            name: "Light".to_string(),
            colors: ColorPalette::light(),
            typography: Typography::standard(),
            spacing: Spacing::standard(),
            radii: Radii::standard(),
            shadows: Shadows::standard(),
        }
    }

    /// Create a dark theme.
    #[must_use]
    pub fn dark() -> Self {
        Self {
            name: "Dark".to_string(),
            colors: ColorPalette::dark(),
            typography: Typography::standard(),
            spacing: Spacing::standard(),
            radii: Radii::standard(),
            shadows: Shadows::standard(),
        }
    }

    /// Create a theme with a custom name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Create a theme with custom colors.
    #[must_use]
    pub const fn with_colors(mut self, colors: ColorPalette) -> Self {
        self.colors = colors;
        self
    }

    /// Create a theme with custom typography.
    #[must_use]
    pub const fn with_typography(mut self, typography: Typography) -> Self {
        self.typography = typography;
        self
    }

    /// Create a theme with custom spacing.
    #[must_use]
    pub const fn with_spacing(mut self, spacing: Spacing) -> Self {
        self.spacing = spacing;
        self
    }

    /// Create a theme with custom radii.
    #[must_use]
    pub const fn with_radii(mut self, radii: Radii) -> Self {
        self.radii = radii;
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    // =========================================================================
    // ColorPalette Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_color_palette_default() {
        let palette = ColorPalette::default();
        assert_eq!(palette, ColorPalette::light());
    }

    #[test]
    fn test_color_palette_light() {
        let palette = ColorPalette::light();
        // Primary should be a blue color
        assert!(palette.primary.b > palette.primary.r);
        // Surface should be white
        assert_eq!(palette.surface, Color::WHITE);
        // On-primary should be white (for contrast)
        assert_eq!(palette.on_primary, Color::WHITE);
    }

    #[test]
    fn test_color_palette_dark() {
        let palette = ColorPalette::dark();
        // Surface should be dark
        assert!(palette.surface.r < 0.5);
        // On-surface should be white
        assert_eq!(palette.on_surface, Color::WHITE);
        // On-primary should be black (for contrast on light primary)
        assert_eq!(palette.on_primary, Color::BLACK);
    }

    // =========================================================================
    // Typography Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_typography_default() {
        let typo = Typography::default();
        assert_eq!(typo.base_size, 16.0);
    }

    #[test]
    fn test_typography_standard() {
        let typo = Typography::standard();
        assert_eq!(typo.base_size, 16.0);
        assert_eq!(typo.h1_scale, 2.5);
        assert_eq!(typo.line_height, 1.5);
    }

    #[test]
    fn test_typography_compact() {
        let typo = Typography::compact();
        assert_eq!(typo.base_size, 14.0);
        assert!(typo.line_height < Typography::standard().line_height);
    }

    #[test]
    fn test_typography_heading_size() {
        let typo = Typography::standard();
        assert_eq!(typo.heading_size(1), 40.0); // 16 * 2.5
        assert_eq!(typo.heading_size(2), 32.0); // 16 * 2.0
        assert_eq!(typo.heading_size(3), 28.0); // 16 * 1.75
        assert_eq!(typo.heading_size(4), 24.0); // 16 * 1.5
        assert_eq!(typo.heading_size(5), 20.0); // 16 * 1.25
        assert_eq!(typo.heading_size(6), 18.0); // 16 * 1.125
    }

    #[test]
    fn test_typography_heading_size_out_of_range() {
        let typo = Typography::standard();
        // Level > 6 should use h6 scale
        assert_eq!(typo.heading_size(7), typo.heading_size(6));
        assert_eq!(typo.heading_size(0), typo.heading_size(6));
    }

    #[test]
    fn test_typography_body_size() {
        let typo = Typography::standard();
        assert_eq!(typo.body_size(), 16.0);
    }

    #[test]
    fn test_typography_caption_size() {
        let typo = Typography::standard();
        assert_eq!(typo.caption_size(), 12.0); // 16 * 0.75
    }

    // =========================================================================
    // Spacing Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_spacing_default() {
        let spacing = Spacing::default();
        assert_eq!(spacing.unit, 8.0);
    }

    #[test]
    fn test_spacing_standard() {
        let spacing = Spacing::standard();
        assert_eq!(spacing.unit, 8.0);
    }

    #[test]
    fn test_spacing_compact() {
        let spacing = Spacing::compact();
        assert_eq!(spacing.unit, 4.0);
    }

    #[test]
    fn test_spacing_get() {
        let spacing = Spacing::standard();
        assert_eq!(spacing.get(0.0), 0.0);
        assert_eq!(spacing.get(1.0), 8.0);
        assert_eq!(spacing.get(2.0), 16.0);
        assert_eq!(spacing.get(0.5), 4.0);
    }

    #[test]
    fn test_spacing_presets() {
        let spacing = Spacing::standard();
        assert_eq!(spacing.none(), 0.0);
        assert_eq!(spacing.xs(), 4.0); // 0.5x
        assert_eq!(spacing.sm(), 8.0); // 1x
        assert_eq!(spacing.md(), 16.0); // 2x
        assert_eq!(spacing.lg(), 24.0); // 3x
        assert_eq!(spacing.xl(), 32.0); // 4x
        assert_eq!(spacing.xl2(), 48.0); // 6x
        assert_eq!(spacing.xl3(), 64.0); // 8x
    }

    // =========================================================================
    // Radii Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_radii_default() {
        let radii = Radii::default();
        assert_eq!(radii.unit, 4.0);
    }

    #[test]
    fn test_radii_presets() {
        let radii = Radii::standard();
        assert_eq!(radii.none(), 0.0);
        assert_eq!(radii.sm(), 4.0);
        assert_eq!(radii.md(), 8.0);
        assert_eq!(radii.lg(), 12.0);
        assert_eq!(radii.xl(), 16.0);
        assert_eq!(radii.full(), 9999.0);
    }

    // =========================================================================
    // Shadows Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_shadows_default() {
        let shadows = Shadows::default();
        assert!(shadows.color.a < 0.5); // Shadow should be semi-transparent
    }

    #[test]
    fn test_shadows_presets() {
        let shadows = Shadows::standard();
        let (blur_sm, offset_sm) = shadows.sm();
        let (blur_md, offset_md) = shadows.md();
        let (blur_lg, offset_lg) = shadows.lg();
        let (blur_xl, offset_xl) = shadows.xl();

        // Each level should be larger than the previous
        assert!(blur_md > blur_sm);
        assert!(blur_lg > blur_md);
        assert!(blur_xl > blur_lg);

        assert!(offset_md > offset_sm);
        assert!(offset_lg > offset_md);
        assert!(offset_xl > offset_lg);
    }

    // =========================================================================
    // Theme Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_theme_default() {
        let theme = Theme::default();
        assert_eq!(theme.name, "Light");
    }

    #[test]
    fn test_theme_light() {
        let theme = Theme::light();
        assert_eq!(theme.name, "Light");
        assert_eq!(theme.colors, ColorPalette::light());
    }

    #[test]
    fn test_theme_dark() {
        let theme = Theme::dark();
        assert_eq!(theme.name, "Dark");
        assert_eq!(theme.colors, ColorPalette::dark());
    }

    #[test]
    fn test_theme_with_name() {
        let theme = Theme::light().with_name("Custom");
        assert_eq!(theme.name, "Custom");
    }

    #[test]
    fn test_theme_with_colors() {
        let theme = Theme::light().with_colors(ColorPalette::dark());
        assert_eq!(theme.colors, ColorPalette::dark());
    }

    #[test]
    fn test_theme_with_typography() {
        let theme = Theme::light().with_typography(Typography::compact());
        assert_eq!(theme.typography, Typography::compact());
    }

    #[test]
    fn test_theme_with_spacing() {
        let theme = Theme::light().with_spacing(Spacing::compact());
        assert_eq!(theme.spacing, Spacing::compact());
    }

    #[test]
    fn test_theme_with_radii() {
        let custom_radii = Radii { unit: 2.0 };
        let theme = Theme::light().with_radii(custom_radii);
        assert_eq!(theme.radii.unit, 2.0);
    }

    #[test]
    fn test_theme_builder_chain() {
        let theme = Theme::light()
            .with_name("My Theme")
            .with_colors(ColorPalette::dark())
            .with_typography(Typography::compact())
            .with_spacing(Spacing::compact());

        assert_eq!(theme.name, "My Theme");
        assert_eq!(theme.colors, ColorPalette::dark());
        assert_eq!(theme.typography, Typography::compact());
        assert_eq!(theme.spacing, Spacing::compact());
    }

    #[test]
    fn test_theme_serialization() {
        let theme = Theme::dark();
        let json = serde_json::to_string(&theme).expect("serialize");
        let restored: Theme = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(theme, restored);
    }

    // =========================================================================
    // Contrast Check Tests
    // =========================================================================

    #[test]
    fn test_light_palette_contrast_aa() {
        let palette = ColorPalette::light();
        let checks = palette.check_contrast();

        // Should have checks for all pairs
        assert_eq!(checks.len(), 5);

        // on_primary/primary should pass (white on blue)
        let primary_check = checks.iter().find(|c| c.name.contains("primary")).unwrap();
        assert!(
            primary_check.passes_aa,
            "on_primary/primary ratio: {:.2}",
            primary_check.ratio
        );
    }

    #[test]
    fn test_dark_palette_contrast_aa() {
        let palette = ColorPalette::dark();
        let checks = palette.check_contrast();

        // on_surface/surface should pass (white on dark)
        let surface_check = checks.iter().find(|c| c.name.contains("surface")).unwrap();
        assert!(
            surface_check.passes_aa,
            "on_surface/surface ratio: {:.2}",
            surface_check.ratio
        );
    }

    #[test]
    fn test_passes_wcag_aa() {
        let light = ColorPalette::light();
        let dark = ColorPalette::dark();

        // Built-in palettes should be accessible
        assert!(
            light.passes_wcag_aa(),
            "Light palette should pass AA: {:?}",
            light.failing_aa()
        );
        assert!(
            dark.passes_wcag_aa(),
            "Dark palette should pass AA: {:?}",
            dark.failing_aa()
        );
    }

    #[test]
    fn test_failing_aa() {
        // Create an intentionally inaccessible palette
        let bad_palette = ColorPalette {
            primary: Color::rgb(0.5, 0.5, 0.5),
            secondary: Color::rgb(0.5, 0.5, 0.5),
            surface: Color::rgb(0.6, 0.6, 0.6), // Similar to on_surface
            background: Color::rgb(0.6, 0.6, 0.6),
            error: Color::rgb(0.5, 0.5, 0.5),
            warning: Color::rgb(0.5, 0.5, 0.5),
            success: Color::rgb(0.5, 0.5, 0.5),
            on_primary: Color::rgb(0.6, 0.6, 0.6), // Low contrast
            on_secondary: Color::rgb(0.6, 0.6, 0.6),
            on_surface: Color::rgb(0.5, 0.5, 0.5), // Low contrast
            on_background: Color::rgb(0.5, 0.5, 0.5),
            on_error: Color::rgb(0.6, 0.6, 0.6),
        };

        assert!(!bad_palette.passes_wcag_aa());
        let failures = bad_palette.failing_aa();
        assert!(!failures.is_empty());
    }

    #[test]
    fn test_contrast_check_ratios() {
        let palette = ColorPalette::light();
        let checks = palette.check_contrast();

        for check in checks {
            // All ratios should be >= 1.0 (minimum possible)
            assert!(
                check.ratio >= 1.0,
                "{} has invalid ratio {}",
                check.name,
                check.ratio
            );
            // Consistency check
            assert_eq!(check.passes_aa, check.ratio >= 4.5);
            assert_eq!(check.passes_aaa, check.ratio >= 7.0);
        }
    }
}
