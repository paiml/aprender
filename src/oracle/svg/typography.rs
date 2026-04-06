//! Typography Styles
//!
//! Material Design 3 typography system with Roboto font specifications.

use super::palette::Color;
use std::fmt;

/// Font family options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FontFamily {
    /// Roboto (Material default)
    #[default]
    Roboto,
    /// Roboto Mono (code)
    RobotoMono,
    /// System sans-serif fallback
    SansSerif,
    /// System monospace fallback
    Monospace,
    /// Segoe UI (video-optimized sans-serif)
    SegoeUI,
    /// Cascadia Code (video-optimized monospace)
    CascadiaCode,
}

impl FontFamily {
    /// Get the CSS font-family value
    #[allow(clippy::wrong_self_convention)]
    pub fn to_css(&self) -> &'static str {
        match self {
            Self::Roboto => "Roboto, sans-serif",
            Self::RobotoMono => "'Roboto Mono', monospace",
            Self::SansSerif => "system-ui, -apple-system, sans-serif",
            Self::Monospace => "ui-monospace, 'Cascadia Code', monospace",
            Self::SegoeUI => "'Segoe UI', 'Helvetica Neue', sans-serif",
            Self::CascadiaCode => "'Cascadia Code', 'Fira Code', 'Consolas', monospace",
        }
    }
}

impl fmt::Display for FontFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_css())
    }
}

/// Font weight options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FontWeight {
    /// Thin (100)
    Thin,
    /// Light (300)
    Light,
    /// Regular (400)
    #[default]
    Regular,
    /// Medium (500)
    Medium,
    /// SemiBold (600)
    SemiBold,
    /// Bold (700)
    Bold,
    /// Black (900)
    Black,
}

impl FontWeight {
    /// Get the numeric weight value
    pub fn value(&self) -> u16 {
        match self {
            Self::Thin => 100,
            Self::Light => 300,
            Self::Regular => 400,
            Self::Medium => 500,
            Self::SemiBold => 600,
            Self::Bold => 700,
            Self::Black => 900,
        }
    }
}

impl fmt::Display for FontWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value())
    }
}

/// Text alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextAlign {
    #[default]
    Start,
    Middle,
    End,
}

impl TextAlign {
    /// Get the SVG text-anchor value
    pub fn as_svg_anchor(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Middle => "middle",
            Self::End => "end",
        }
    }
}

impl fmt::Display for TextAlign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_svg_anchor())
    }
}

/// A typography style definition
#[derive(Debug, Clone)]
pub struct TextStyle {
    /// Font family
    pub family: FontFamily,
    /// Font size in pixels
    pub size: f32,
    /// Font weight
    pub weight: FontWeight,
    /// Line height multiplier
    pub line_height: f32,
    /// Letter spacing in ems
    pub letter_spacing: f32,
    /// Text color
    pub color: Color,
    /// Text alignment
    pub align: TextAlign,
}

impl TextStyle {
    /// Create a new text style
    pub fn new(size: f32, weight: FontWeight) -> Self {
        Self {
            family: FontFamily::default(),
            size,
            weight,
            line_height: 1.5,
            letter_spacing: 0.0,
            color: Color::rgb(0, 0, 0),
            align: TextAlign::default(),
        }
    }

    /// Set the font family
    pub fn with_family(mut self, family: FontFamily) -> Self {
        self.family = family;
        self
    }

    /// Set the line height
    pub fn with_line_height(mut self, height: f32) -> Self {
        self.line_height = height;
        self
    }

    /// Set the letter spacing
    pub fn with_letter_spacing(mut self, spacing: f32) -> Self {
        self.letter_spacing = spacing;
        self
    }

    /// Set the color
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set the alignment
    pub fn with_align(mut self, align: TextAlign) -> Self {
        self.align = align;
        self
    }

    /// Generate SVG style attributes
    pub fn to_svg_attrs(&self) -> String {
        let mut attrs = format!(
            "font-family=\"{}\" font-size=\"{}\" font-weight=\"{}\" fill=\"{}\"",
            self.family,
            self.size,
            self.weight,
            self.color.to_css_hex()
        );

        if self.letter_spacing != 0.0 {
            attrs.push_str(&format!(" letter-spacing=\"{}em\"", self.letter_spacing));
        }

        if self.align != TextAlign::Start {
            attrs.push_str(&format!(" text-anchor=\"{}\"", self.align));
        }

        attrs
    }
}

impl Default for TextStyle {
    fn default() -> Self {
        Self::new(14.0, FontWeight::Regular)
    }
}

/// Material Design 3 typography scale
#[derive(Debug, Clone)]
pub struct MaterialTypography {
    /// Display Large - 57px
    pub display_large: TextStyle,
    /// Display Medium - 45px
    pub display_medium: TextStyle,
    /// Display Small - 36px
    pub display_small: TextStyle,

    /// Headline Large - 32px
    pub headline_large: TextStyle,
    /// Headline Medium - 28px
    pub headline_medium: TextStyle,
    /// Headline Small - 24px
    pub headline_small: TextStyle,

    /// Title Large - 22px
    pub title_large: TextStyle,
    /// Title Medium - 16px
    pub title_medium: TextStyle,
    /// Title Small - 14px
    pub title_small: TextStyle,

    /// Body Large - 16px
    pub body_large: TextStyle,
    /// Body Medium - 14px
    pub body_medium: TextStyle,
    /// Body Small - 12px
    pub body_small: TextStyle,

    /// Label Large - 14px
    pub label_large: TextStyle,
    /// Label Medium - 12px
    pub label_medium: TextStyle,
    /// Label Small - 11px
    pub label_small: TextStyle,

    /// Code (monospace) - 14px
    pub code: TextStyle,
}

impl MaterialTypography {
    /// Create the Material Design 3 typography scale with a text color
    pub fn with_color(color: Color) -> Self {
        Self {
            // Display
            display_large: TextStyle::new(57.0, FontWeight::Regular)
                .with_line_height(1.12)
                .with_letter_spacing(-0.014)
                .with_color(color),
            display_medium: TextStyle::new(45.0, FontWeight::Regular)
                .with_line_height(1.16)
                .with_color(color),
            display_small: TextStyle::new(36.0, FontWeight::Regular)
                .with_line_height(1.22)
                .with_color(color),

            // Headline
            headline_large: TextStyle::new(32.0, FontWeight::Regular)
                .with_line_height(1.25)
                .with_color(color),
            headline_medium: TextStyle::new(28.0, FontWeight::Regular)
                .with_line_height(1.29)
                .with_color(color),
            headline_small: TextStyle::new(24.0, FontWeight::Regular)
                .with_line_height(1.33)
                .with_color(color),

            // Title
            title_large: TextStyle::new(22.0, FontWeight::Regular)
                .with_line_height(1.27)
                .with_color(color),
            title_medium: TextStyle::new(16.0, FontWeight::Medium)
                .with_line_height(1.5)
                .with_letter_spacing(0.009)
                .with_color(color),
            title_small: TextStyle::new(14.0, FontWeight::Medium)
                .with_line_height(1.43)
                .with_letter_spacing(0.007)
                .with_color(color),

            // Body
            body_large: TextStyle::new(16.0, FontWeight::Regular)
                .with_line_height(1.5)
                .with_letter_spacing(0.031)
                .with_color(color),
            body_medium: TextStyle::new(14.0, FontWeight::Regular)
                .with_line_height(1.43)
                .with_letter_spacing(0.018)
                .with_color(color),
            body_small: TextStyle::new(12.0, FontWeight::Regular)
                .with_line_height(1.33)
                .with_letter_spacing(0.033)
                .with_color(color),

            // Label
            label_large: TextStyle::new(14.0, FontWeight::Medium)
                .with_line_height(1.43)
                .with_letter_spacing(0.007)
                .with_color(color),
            label_medium: TextStyle::new(12.0, FontWeight::Medium)
                .with_line_height(1.33)
                .with_letter_spacing(0.042)
                .with_color(color),
            label_small: TextStyle::new(11.0, FontWeight::Medium)
                .with_line_height(1.45)
                .with_letter_spacing(0.045)
                .with_color(color),

            // Code
            code: TextStyle::new(14.0, FontWeight::Regular)
                .with_family(FontFamily::RobotoMono)
                .with_line_height(1.5)
                .with_color(color),
        }
    }
}

impl Default for MaterialTypography {
    fn default() -> Self {
        Self::with_color(Color::rgb(28, 27, 31))
    }
}

/// Video-optimized typography for 1080p presentation SVGs.
///
/// All sizes >= 18px (hard minimum for readability at 1080p).
/// Uses Segoe UI for body text and Cascadia Code for code.
#[derive(Debug, Clone)]
pub struct VideoTypography {
    /// Slide title — 56px, Bold (700), Segoe UI
    pub slide_title: TextStyle,
    /// Section header — 36px, SemiBold (600), Segoe UI
    pub section_header: TextStyle,
    /// Body text — 24px, Regular (400), Segoe UI
    pub body: TextStyle,
    /// Labels — 18px, Regular (400), Segoe UI
    pub label: TextStyle,
    /// Code — 22px, Regular (400), Cascadia Code
    pub code: TextStyle,
    /// Icon text — 18px, Bold (700), Segoe UI
    pub icon_text: TextStyle,
}

impl VideoTypography {
    /// Minimum font size for video mode.
    pub const MIN_FONT_SIZE: f32 = 18.0;

    /// Video typography with colors for dark backgrounds.
    pub fn dark() -> Self {
        let heading = Color::rgb(241, 245, 249); // #f1f5f9
        let body_color = Color::rgb(148, 163, 184); // #94a3b8
        let accent = Color::rgb(96, 165, 250); // #60a5fa

        Self {
            slide_title: TextStyle::new(56.0, FontWeight::Bold)
                .with_family(FontFamily::SegoeUI)
                .with_color(heading)
                .with_line_height(1.15),
            section_header: TextStyle::new(36.0, FontWeight::SemiBold)
                .with_family(FontFamily::SegoeUI)
                .with_color(heading)
                .with_line_height(1.2),
            body: TextStyle::new(24.0, FontWeight::Regular)
                .with_family(FontFamily::SegoeUI)
                .with_color(body_color)
                .with_line_height(1.4),
            label: TextStyle::new(18.0, FontWeight::Regular)
                .with_family(FontFamily::SegoeUI)
                .with_color(body_color)
                .with_line_height(1.4),
            code: TextStyle::new(22.0, FontWeight::Regular)
                .with_family(FontFamily::CascadiaCode)
                .with_color(accent)
                .with_line_height(1.5),
            icon_text: TextStyle::new(18.0, FontWeight::Bold)
                .with_family(FontFamily::SegoeUI)
                .with_color(heading)
                .with_line_height(1.4),
        }
    }

    /// Video typography with colors for light backgrounds.
    pub fn light() -> Self {
        let heading = Color::rgb(15, 23, 42); // #0f172a
        let body_color = Color::rgb(71, 85, 105); // #475569
        let accent = Color::rgb(37, 99, 235); // #2563eb

        Self {
            slide_title: TextStyle::new(56.0, FontWeight::Bold)
                .with_family(FontFamily::SegoeUI)
                .with_color(heading)
                .with_line_height(1.15),
            section_header: TextStyle::new(36.0, FontWeight::SemiBold)
                .with_family(FontFamily::SegoeUI)
                .with_color(heading)
                .with_line_height(1.2),
            body: TextStyle::new(24.0, FontWeight::Regular)
                .with_family(FontFamily::SegoeUI)
                .with_color(body_color)
                .with_line_height(1.4),
            label: TextStyle::new(18.0, FontWeight::Regular)
                .with_family(FontFamily::SegoeUI)
                .with_color(body_color)
                .with_line_height(1.4),
            code: TextStyle::new(22.0, FontWeight::Regular)
                .with_family(FontFamily::CascadiaCode)
                .with_color(accent)
                .with_line_height(1.5),
            icon_text: TextStyle::new(18.0, FontWeight::Bold)
                .with_family(FontFamily::SegoeUI)
                .with_color(heading)
                .with_line_height(1.4),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_font_family_css() {
        assert_eq!(FontFamily::Roboto.to_css(), "Roboto, sans-serif");
        assert_eq!(FontFamily::RobotoMono.to_css(), "'Roboto Mono', monospace");
        assert_eq!(FontFamily::SegoeUI.to_css(), "'Segoe UI', 'Helvetica Neue', sans-serif");
        assert_eq!(
            FontFamily::CascadiaCode.to_css(),
            "'Cascadia Code', 'Fira Code', 'Consolas', monospace"
        );
    }

    #[test]
    fn test_font_weight_value() {
        assert_eq!(FontWeight::Regular.value(), 400);
        assert_eq!(FontWeight::Bold.value(), 700);
    }

    #[test]
    fn test_text_align_svg() {
        assert_eq!(TextAlign::Start.as_svg_anchor(), "start");
        assert_eq!(TextAlign::Middle.as_svg_anchor(), "middle");
        assert_eq!(TextAlign::End.as_svg_anchor(), "end");
    }

    #[test]
    fn test_text_style_creation() {
        let style = TextStyle::new(16.0, FontWeight::Bold);
        assert_eq!(style.size, 16.0);
        assert_eq!(style.weight, FontWeight::Bold);
    }

    #[test]
    fn test_text_style_builder() {
        let style = TextStyle::new(14.0, FontWeight::Regular)
            .with_family(FontFamily::RobotoMono)
            .with_color(Color::rgb(255, 0, 0))
            .with_align(TextAlign::Middle);

        assert_eq!(style.family, FontFamily::RobotoMono);
        assert_eq!(style.color, Color::rgb(255, 0, 0));
        assert_eq!(style.align, TextAlign::Middle);
    }

    #[test]
    fn test_text_style_to_svg_attrs() {
        let style = TextStyle::new(16.0, FontWeight::Bold).with_color(Color::rgb(0, 0, 0));

        let attrs = style.to_svg_attrs();
        assert!(attrs.contains("font-size=\"16\""));
        assert!(attrs.contains("font-weight=\"700\""));
        assert!(attrs.contains("fill=\"#000000\""));
    }

    #[test]
    fn test_material_typography_scale() {
        let typo = MaterialTypography::default();

        assert_eq!(typo.display_large.size, 57.0);
        assert_eq!(typo.headline_large.size, 32.0);
        assert_eq!(typo.body_medium.size, 14.0);
        assert_eq!(typo.label_small.size, 11.0);
        assert_eq!(typo.code.family, FontFamily::RobotoMono);
    }

    #[test]
    fn test_material_typography_with_color() {
        let color = Color::rgb(255, 255, 255);
        let typo = MaterialTypography::with_color(color);

        assert_eq!(typo.body_medium.color, color);
        assert_eq!(typo.headline_large.color, color);
    }

    #[test]
    fn test_font_family_display() {
        assert_eq!(format!("{}", FontFamily::Roboto), "Roboto, sans-serif");
        assert_eq!(format!("{}", FontFamily::SansSerif), "system-ui, -apple-system, sans-serif");
        assert_eq!(
            format!("{}", FontFamily::Monospace),
            "ui-monospace, 'Cascadia Code', monospace"
        );
    }

    #[test]
    fn test_font_family_default() {
        assert_eq!(FontFamily::default(), FontFamily::Roboto);
    }

    #[test]
    fn test_font_weight_display() {
        assert_eq!(format!("{}", FontWeight::Thin), "100");
        assert_eq!(format!("{}", FontWeight::Light), "300");
        assert_eq!(format!("{}", FontWeight::Regular), "400");
        assert_eq!(format!("{}", FontWeight::Medium), "500");
        assert_eq!(format!("{}", FontWeight::SemiBold), "600");
        assert_eq!(format!("{}", FontWeight::Bold), "700");
        assert_eq!(format!("{}", FontWeight::Black), "900");
    }

    #[test]
    fn test_font_weight_default() {
        assert_eq!(FontWeight::default(), FontWeight::Regular);
    }

    #[test]
    fn test_text_align_display() {
        assert_eq!(format!("{}", TextAlign::Start), "start");
        assert_eq!(format!("{}", TextAlign::Middle), "middle");
        assert_eq!(format!("{}", TextAlign::End), "end");
    }

    #[test]
    fn test_text_align_default() {
        assert_eq!(TextAlign::default(), TextAlign::Start);
    }

    #[test]
    fn test_text_style_with_line_height() {
        let style = TextStyle::new(14.0, FontWeight::Regular).with_line_height(2.0);
        assert_eq!(style.line_height, 2.0);
    }

    #[test]
    fn test_text_style_with_letter_spacing() {
        let style = TextStyle::new(14.0, FontWeight::Regular).with_letter_spacing(0.05);
        assert_eq!(style.letter_spacing, 0.05);
    }

    #[test]
    fn test_text_style_default() {
        let style = TextStyle::default();
        assert_eq!(style.size, 14.0);
        assert_eq!(style.weight, FontWeight::Regular);
        assert_eq!(style.family, FontFamily::Roboto);
        assert_eq!(style.line_height, 1.5);
        assert_eq!(style.letter_spacing, 0.0);
        assert_eq!(style.align, TextAlign::Start);
    }

    #[test]
    fn test_text_style_svg_attrs_with_letter_spacing() {
        let style = TextStyle::new(14.0, FontWeight::Regular).with_letter_spacing(0.05);
        let attrs = style.to_svg_attrs();
        assert!(attrs.contains("letter-spacing=\"0.05em\""));
    }

    #[test]
    fn test_text_style_svg_attrs_with_alignment() {
        let style = TextStyle::new(14.0, FontWeight::Regular).with_align(TextAlign::End);
        let attrs = style.to_svg_attrs();
        assert!(attrs.contains("text-anchor=\"end\""));
    }

    #[test]
    fn test_text_style_svg_attrs_no_optional() {
        let style = TextStyle::new(14.0, FontWeight::Regular);
        let attrs = style.to_svg_attrs();
        assert!(!attrs.contains("letter-spacing"));
        assert!(!attrs.contains("text-anchor"));
    }

    #[test]
    fn test_font_weight_all_values() {
        assert_eq!(FontWeight::Thin.value(), 100);
        assert_eq!(FontWeight::Light.value(), 300);
        assert_eq!(FontWeight::Medium.value(), 500);
        assert_eq!(FontWeight::SemiBold.value(), 600);
        assert_eq!(FontWeight::Black.value(), 900);
    }

    #[test]
    fn test_font_family_segoe_ui_display() {
        let display = format!("{}", FontFamily::SegoeUI);
        assert!(display.contains("Segoe UI"));
    }

    #[test]
    fn test_font_family_cascadia_code_display() {
        let display = format!("{}", FontFamily::CascadiaCode);
        assert!(display.contains("Cascadia Code"));
    }

    #[test]
    fn test_video_typography_dark() {
        let vt = VideoTypography::dark();
        assert_eq!(vt.slide_title.size, 56.0);
        assert_eq!(vt.slide_title.weight, FontWeight::Bold);
        assert_eq!(vt.slide_title.family, FontFamily::SegoeUI);

        assert_eq!(vt.section_header.size, 36.0);
        assert_eq!(vt.section_header.weight, FontWeight::SemiBold);

        assert_eq!(vt.body.size, 24.0);
        assert_eq!(vt.body.weight, FontWeight::Regular);

        assert_eq!(vt.label.size, 18.0);
        assert!(vt.label.size >= VideoTypography::MIN_FONT_SIZE);

        assert_eq!(vt.code.size, 22.0);
        assert_eq!(vt.code.family, FontFamily::CascadiaCode);

        assert_eq!(vt.icon_text.size, 18.0);
        assert_eq!(vt.icon_text.weight, FontWeight::Bold);
    }

    #[test]
    fn test_video_typography_light() {
        let vt = VideoTypography::light();
        assert_eq!(vt.slide_title.size, 56.0);
        assert_eq!(vt.body.size, 24.0);
        assert_eq!(vt.code.family, FontFamily::CascadiaCode);
    }

    #[test]
    fn test_video_typography_all_sizes_meet_minimum() {
        for vt in &[VideoTypography::dark(), VideoTypography::light()] {
            assert!(vt.slide_title.size >= VideoTypography::MIN_FONT_SIZE);
            assert!(vt.section_header.size >= VideoTypography::MIN_FONT_SIZE);
            assert!(vt.body.size >= VideoTypography::MIN_FONT_SIZE);
            assert!(vt.label.size >= VideoTypography::MIN_FONT_SIZE);
            assert!(vt.code.size >= VideoTypography::MIN_FONT_SIZE);
            assert!(vt.icon_text.size >= VideoTypography::MIN_FONT_SIZE);
        }
    }

    #[test]
    fn test_video_typography_min_font_size_constant() {
        assert_eq!(VideoTypography::MIN_FONT_SIZE, 18.0);
    }
}
