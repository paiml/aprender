//! SVG output encoder.
//!
//! Provides both raster (embedded PNG) and vector SVG output.
//! Vector output preserves scalability for print and web.

use crate::color::Rgba;
use crate::error::Result;
use crate::framebuffer::Framebuffer;
use base64::{engine::general_purpose::STANDARD, Engine};
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// SVG encoder for framebuffer and vector output.
#[derive(Debug, Clone)]
pub struct SvgEncoder {
    /// SVG width
    width: u32,
    /// SVG height
    height: u32,
    /// Background color (None for transparent)
    background: Option<Rgba>,
    /// SVG elements
    elements: Vec<SvgElement>,
}

/// An SVG element.
///
/// Field names are self-documenting and match SVG attribute names.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum SvgElement {
    /// Rectangle
    Rect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        fill: Rgba,
        stroke: Option<Rgba>,
        stroke_width: f32,
    },
    /// Circle
    Circle { cx: f32, cy: f32, r: f32, fill: Rgba, stroke: Option<Rgba>, stroke_width: f32 },
    /// Line
    Line { x1: f32, y1: f32, x2: f32, y2: f32, stroke: Rgba, stroke_width: f32 },
    /// Polyline (connected line segments)
    Polyline { points: Vec<(f32, f32)>, stroke: Rgba, stroke_width: f32, fill: Option<Rgba> },
    /// Path (SVG path data)
    Path { d: String, fill: Option<Rgba>, stroke: Option<Rgba>, stroke_width: f32 },
    /// Text
    Text { x: f32, y: f32, text: String, font_size: f32, fill: Rgba, anchor: TextAnchor },
    /// Embedded raster image (base64 PNG)
    Image { x: f32, y: f32, width: f32, height: f32, data: String },
}

/// Text anchor position for SVG text alignment.
#[derive(Debug, Clone, Copy, Default)]
#[allow(missing_docs)]
pub enum TextAnchor {
    /// Align text start at position (left-aligned for LTR)
    #[default]
    Start,
    /// Center text at position
    Middle,
    /// Align text end at position (right-aligned for LTR)
    End,
}

impl Default for SvgEncoder {
    fn default() -> Self {
        Self::new(800, 600)
    }
}

impl SvgEncoder {
    /// Create a new SVG encoder with given dimensions.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, background: Some(Rgba::WHITE), elements: Vec::new() }
    }

    /// Create from a framebuffer (embeds as raster image).
    ///
    /// # Errors
    ///
    /// Returns an error if PNG encoding fails.
    pub fn from_framebuffer(fb: &Framebuffer) -> Result<Self> {
        let mut encoder = Self::new(fb.width(), fb.height());
        encoder.background = None; // Image provides background

        // Encode framebuffer as PNG and embed
        let png_bytes = super::PngEncoder::to_bytes(fb)?;
        let base64_data = STANDARD.encode(&png_bytes);
        let data_uri = format!("data:image/png;base64,{base64_data}");

        encoder.elements.push(SvgElement::Image {
            x: 0.0,
            y: 0.0,
            width: fb.width() as f32,
            height: fb.height() as f32,
            data: data_uri,
        });

        Ok(encoder)
    }

    /// Set background color (None for transparent).
    #[must_use]
    pub fn background(mut self, color: Option<Rgba>) -> Self {
        self.background = color;
        self
    }

    /// Add a rectangle.
    #[must_use]
    pub fn rect(mut self, x: f32, y: f32, width: f32, height: f32, fill: Rgba) -> Self {
        self.elements.push(SvgElement::Rect {
            x,
            y,
            width,
            height,
            fill,
            stroke: None,
            stroke_width: 1.0,
        });
        self
    }

    /// Add a rectangle with stroke.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn rect_outlined(
        mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        fill: Rgba,
        stroke: Rgba,
        stroke_width: f32,
    ) -> Self {
        self.elements.push(SvgElement::Rect {
            x,
            y,
            width,
            height,
            fill,
            stroke: Some(stroke),
            stroke_width,
        });
        self
    }

    /// Add a circle.
    #[must_use]
    pub fn circle(mut self, cx: f32, cy: f32, r: f32, fill: Rgba) -> Self {
        self.elements.push(SvgElement::Circle { cx, cy, r, fill, stroke: None, stroke_width: 1.0 });
        self
    }

    /// Add a circle with stroke.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn circle_outlined(
        mut self,
        cx: f32,
        cy: f32,
        r: f32,
        fill: Rgba,
        stroke: Rgba,
        stroke_width: f32,
    ) -> Self {
        self.elements.push(SvgElement::Circle {
            cx,
            cy,
            r,
            fill,
            stroke: Some(stroke),
            stroke_width,
        });
        self
    }

    /// Add a line.
    #[must_use]
    pub fn line(
        mut self,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        stroke: Rgba,
        stroke_width: f32,
    ) -> Self {
        self.elements.push(SvgElement::Line { x1, y1, x2, y2, stroke, stroke_width });
        self
    }

    /// Add a polyline.
    #[must_use]
    pub fn polyline(mut self, points: &[(f32, f32)], stroke: Rgba, stroke_width: f32) -> Self {
        self.elements.push(SvgElement::Polyline {
            points: points.to_vec(),
            stroke,
            stroke_width,
            fill: None,
        });
        self
    }

    /// Add a filled polygon.
    #[must_use]
    pub fn polygon(
        mut self,
        points: &[(f32, f32)],
        fill: Rgba,
        stroke: Option<Rgba>,
        stroke_width: f32,
    ) -> Self {
        self.elements.push(SvgElement::Polyline {
            points: points.to_vec(),
            stroke: stroke.unwrap_or(fill),
            stroke_width,
            fill: Some(fill),
        });
        self
    }

    /// Add an SVG path.
    #[must_use]
    pub fn path(
        mut self,
        d: &str,
        fill: Option<Rgba>,
        stroke: Option<Rgba>,
        stroke_width: f32,
    ) -> Self {
        self.elements.push(SvgElement::Path { d: d.to_string(), fill, stroke, stroke_width });
        self
    }

    /// Add text.
    #[must_use]
    pub fn text(mut self, x: f32, y: f32, text: &str, font_size: f32, fill: Rgba) -> Self {
        self.elements.push(SvgElement::Text {
            x,
            y,
            text: text.to_string(),
            font_size,
            fill,
            anchor: TextAnchor::Start,
        });
        self
    }

    /// Add text with anchor.
    #[must_use]
    pub fn text_anchored(
        mut self,
        x: f32,
        y: f32,
        text: &str,
        font_size: f32,
        fill: Rgba,
        anchor: TextAnchor,
    ) -> Self {
        self.elements.push(SvgElement::Text {
            x,
            y,
            text: text.to_string(),
            font_size,
            fill,
            anchor,
        });
        self
    }

    /// Add a raw element.
    pub fn add_element(&mut self, element: SvgElement) {
        self.elements.push(element);
    }

    /// Get the SVG width.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the SVG height.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Render to SVG string.
    #[must_use]
    pub fn render(&self) -> String {
        let mut svg = String::with_capacity(4096);

        // SVG header
        let _ = writeln!(
            svg,
            r#"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            self.width, self.height, self.width, self.height
        );

        // Background
        if let Some(bg) = self.background {
            let _ =
                writeln!(svg, r#"  <rect width="100%" height="100%" fill="{}"/>"#, rgba_to_css(bg));
        }

        // Elements
        for element in &self.elements {
            let _ = writeln!(svg, "  {}", element_to_svg(element));
        }

        // Close SVG
        svg.push_str("</svg>\n");
        svg
    }

    /// Write to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if file writing fails.
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file = File::create(path)?;
        file.write_all(self.render().as_bytes())?;
        Ok(())
    }
}

/// Convert RGBA to CSS color string.
fn rgba_to_css(color: Rgba) -> String {
    if color.a == 255 {
        format!("rgb({},{},{})", color.r, color.g, color.b)
    } else {
        format!("rgba({},{},{},{:.3})", color.r, color.g, color.b, f32::from(color.a) / 255.0)
    }
}

/// Convert an SVG element to its string representation.
fn element_to_svg(element: &SvgElement) -> String {
    match element {
        SvgElement::Rect { x, y, width, height, fill, stroke, stroke_width } => {
            let stroke_attr = stroke
                .map(|s| format!(r#" stroke="{}" stroke-width="{}""#, rgba_to_css(s), stroke_width))
                .unwrap_or_default();
            format!(
                r#"<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{}"{stroke_attr}/>"#,
                rgba_to_css(*fill)
            )
        }
        SvgElement::Circle { cx, cy, r, fill, stroke, stroke_width } => {
            let stroke_attr = stroke
                .map(|s| format!(r#" stroke="{}" stroke-width="{}""#, rgba_to_css(s), stroke_width))
                .unwrap_or_default();
            format!(
                r#"<circle cx="{cx}" cy="{cy}" r="{r}" fill="{}"{stroke_attr}/>"#,
                rgba_to_css(*fill)
            )
        }
        SvgElement::Line { x1, y1, x2, y2, stroke, stroke_width } => {
            format!(
                r#"<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{}" stroke-width="{stroke_width}"/>"#,
                rgba_to_css(*stroke)
            )
        }
        SvgElement::Polyline { points, stroke, stroke_width, fill } => {
            let points_str: String =
                points.iter().map(|(x, y)| format!("{x},{y}")).collect::<Vec<_>>().join(" ");
            let fill_attr = fill.map(rgba_to_css).unwrap_or_else(|| "none".to_string());
            let tag = if fill.is_some() { "polygon" } else { "polyline" };
            format!(
                r#"<{tag} points="{points_str}" fill="{fill_attr}" stroke="{}" stroke-width="{stroke_width}"/>"#,
                rgba_to_css(*stroke)
            )
        }
        SvgElement::Path { d, fill, stroke, stroke_width } => {
            let fill_attr = fill.map(rgba_to_css).unwrap_or_else(|| "none".to_string());
            let stroke_attr = stroke
                .map(|s| format!(r#" stroke="{}" stroke-width="{}""#, rgba_to_css(s), stroke_width))
                .unwrap_or_default();
            format!(r#"<path d="{d}" fill="{fill_attr}"{stroke_attr}/>"#)
        }
        SvgElement::Text { x, y, text, font_size, fill, anchor } => {
            let anchor_str = match anchor {
                TextAnchor::Start => "start",
                TextAnchor::Middle => "middle",
                TextAnchor::End => "end",
            };
            // Escape XML special characters
            let escaped_text = text
                .replace('&', "&amp;")
                .replace('<', "&lt;")
                .replace('>', "&gt;")
                .replace('"', "&quot;");
            format!(
                r#"<text x="{x}" y="{y}" font-size="{font_size}" fill="{}" text-anchor="{anchor_str}" font-family="sans-serif">{escaped_text}</text>"#,
                rgba_to_css(*fill)
            )
        }
        SvgElement::Image { x, y, width, height, data } => {
            format!(
                r#"<image x="{x}" y="{y}" width="{width}" height="{height}" xlink:href="{data}"/>"#
            )
        }
    }
}

impl std::fmt::Display for SvgEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.render())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svg_encoder_new() {
        let encoder = SvgEncoder::new(800, 600);
        let svg = encoder.render();

        assert!(svg.contains("width=\"800\""));
        assert!(svg.contains("height=\"600\""));
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_svg_rect() {
        let svg = SvgEncoder::new(100, 100).rect(10.0, 20.0, 30.0, 40.0, Rgba::RED).render();

        assert!(svg.contains("<rect"));
        assert!(svg.contains("x=\"10\""));
        assert!(svg.contains("y=\"20\""));
        assert!(svg.contains("width=\"30\""));
        assert!(svg.contains("height=\"40\""));
        assert!(svg.contains("rgb(255,0,0)"));
    }

    #[test]
    fn test_svg_circle() {
        let svg = SvgEncoder::new(100, 100).circle(50.0, 50.0, 25.0, Rgba::BLUE).render();

        assert!(svg.contains("<circle"));
        assert!(svg.contains("cx=\"50\""));
        assert!(svg.contains("cy=\"50\""));
        assert!(svg.contains("r=\"25\""));
        assert!(svg.contains("rgb(0,0,255)"));
    }

    #[test]
    fn test_svg_line() {
        let svg = SvgEncoder::new(100, 100).line(0.0, 0.0, 100.0, 100.0, Rgba::BLACK, 2.0).render();

        assert!(svg.contains("<line"));
        assert!(svg.contains("x1=\"0\""));
        assert!(svg.contains("y1=\"0\""));
        assert!(svg.contains("x2=\"100\""));
        assert!(svg.contains("y2=\"100\""));
        assert!(svg.contains("stroke-width=\"2\""));
    }

    #[test]
    fn test_svg_polyline() {
        let points = vec![(0.0, 0.0), (50.0, 100.0), (100.0, 0.0)];
        let svg = SvgEncoder::new(100, 100).polyline(&points, Rgba::GREEN, 1.5).render();

        assert!(svg.contains("<polyline"));
        assert!(svg.contains("points=\"0,0 50,100 100,0\""));
        assert!(svg.contains("fill=\"none\""));
    }

    #[test]
    fn test_svg_polygon() {
        let points = vec![(0.0, 0.0), (50.0, 100.0), (100.0, 0.0)];
        let svg = SvgEncoder::new(100, 100)
            .polygon(&points, Rgba::GREEN, Some(Rgba::BLACK), 1.0)
            .render();

        assert!(svg.contains("<polygon"));
        assert!(svg.contains("points=\"0,0 50,100 100,0\""));
    }

    #[test]
    fn test_svg_text() {
        let svg = SvgEncoder::new(100, 100).text(10.0, 50.0, "Hello", 12.0, Rgba::BLACK).render();

        assert!(svg.contains("<text"));
        assert!(svg.contains("Hello"));
        assert!(svg.contains("font-size=\"12\""));
    }

    #[test]
    fn test_svg_text_escaping() {
        let svg = SvgEncoder::new(100, 100)
            .text(10.0, 50.0, "<script>alert('xss')</script>", 12.0, Rgba::BLACK)
            .render();

        assert!(!svg.contains("<script>"));
        assert!(svg.contains("&lt;script&gt;"));
    }

    #[test]
    fn test_svg_transparent_background() {
        let svg = SvgEncoder::new(100, 100).background(None).render();

        // Should not have background rect
        let rect_count = svg.matches("<rect").count();
        assert_eq!(rect_count, 0);
    }

    #[test]
    fn test_svg_rgba_alpha() {
        let color = Rgba::new(255, 0, 0, 128);
        let css = rgba_to_css(color);
        assert!(css.contains("rgba"));
        assert!(css.contains("0.502")); // 128/255 ≈ 0.502
    }

    #[test]
    fn test_svg_from_framebuffer() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::RED);

        let encoder = SvgEncoder::from_framebuffer(&fb).expect("encoding should succeed");
        let svg = encoder.render();

        assert!(svg.contains("<image"));
        assert!(svg.contains("data:image/png;base64,"));
    }

    #[test]
    fn test_svg_path() {
        let svg = SvgEncoder::new(100, 100)
            .path("M 10 10 L 90 90", None, Some(Rgba::BLACK), 2.0)
            .render();

        assert!(svg.contains("<path"));
        assert!(svg.contains("d=\"M 10 10 L 90 90\""));
    }

    #[test]
    fn test_svg_rect_outlined() {
        let svg = SvgEncoder::new(100, 100)
            .rect_outlined(10.0, 20.0, 30.0, 40.0, Rgba::RED, Rgba::BLACK, 2.0)
            .render();

        assert!(svg.contains("<rect"));
        assert!(svg.contains("stroke=\"rgb(0,0,0)\""));
        assert!(svg.contains("stroke-width=\"2\""));
    }

    #[test]
    fn test_svg_circle_outlined() {
        let svg = SvgEncoder::new(100, 100)
            .circle_outlined(50.0, 50.0, 25.0, Rgba::BLUE, Rgba::BLACK, 2.0)
            .render();

        assert!(svg.contains("<circle"));
        assert!(svg.contains("stroke=\"rgb(0,0,0)\""));
        assert!(svg.contains("stroke-width=\"2\""));
    }

    #[test]
    fn test_svg_text_anchored_middle() {
        let svg = SvgEncoder::new(100, 100)
            .text_anchored(50.0, 50.0, "Centered", 12.0, Rgba::BLACK, TextAnchor::Middle)
            .render();

        assert!(svg.contains("<text"));
        assert!(svg.contains("text-anchor=\"middle\""));
    }

    #[test]
    fn test_svg_text_anchored_end() {
        let svg = SvgEncoder::new(100, 100)
            .text_anchored(90.0, 50.0, "Right", 12.0, Rgba::BLACK, TextAnchor::End)
            .render();

        assert!(svg.contains("<text"));
        assert!(svg.contains("text-anchor=\"end\""));
    }

    #[test]
    fn test_svg_add_element() {
        let mut encoder = SvgEncoder::new(100, 100);
        encoder.add_element(SvgElement::Circle {
            cx: 50.0,
            cy: 50.0,
            r: 10.0,
            fill: Rgba::RED,
            stroke: None,
            stroke_width: 1.0,
        });
        let svg = encoder.render();
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn test_svg_write_to_file() {
        let encoder = SvgEncoder::new(100, 100).rect(10.0, 10.0, 80.0, 80.0, Rgba::BLUE);

        let temp_path = std::env::temp_dir().join("test_svg_write.svg");
        encoder.write_to_file(&temp_path).expect("file write should succeed");

        let content = std::fs::read_to_string(&temp_path).expect("file read should succeed");
        assert!(content.contains("<svg"));
        assert!(content.contains("</svg>"));

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_svg_encoder_default() {
        let encoder = SvgEncoder::default();
        let svg = encoder.render();
        assert!(svg.contains("width=\"800\""));
        assert!(svg.contains("height=\"600\""));
    }

    #[test]
    fn test_text_anchor_default() {
        let anchor = TextAnchor::default();
        assert!(matches!(anchor, TextAnchor::Start));
    }

    #[test]
    fn test_svg_path_with_fill() {
        let svg = SvgEncoder::new(100, 100)
            .path("M 10 10 L 90 90 L 50 50 Z", Some(Rgba::GREEN), Some(Rgba::BLACK), 1.0)
            .render();

        assert!(svg.contains("<path"));
        // Rgba::GREEN is rgb(0,255,0) - pure green
        assert!(svg.contains("fill=\"rgb(0,255,0)\""));
        assert!(svg.contains("stroke=\"rgb(0,0,0)\""));
    }

    #[test]
    fn test_svg_path_no_stroke() {
        let svg =
            SvgEncoder::new(100, 100).path("M 10 10 L 90 90", Some(Rgba::RED), None, 0.0).render();

        assert!(svg.contains("<path"));
        assert!(svg.contains("fill=\"rgb(255,0,0)\""));
        assert!(!svg.contains("stroke="));
    }

    #[test]
    fn test_svg_polygon_no_stroke() {
        let points = vec![(0.0, 0.0), (50.0, 100.0), (100.0, 0.0)];
        let svg = SvgEncoder::new(100, 100).polygon(&points, Rgba::RED, None, 1.0).render();

        assert!(svg.contains("<polygon"));
    }

    #[test]
    fn test_svg_debug_clone() {
        let encoder = SvgEncoder::new(100, 100).rect(10.0, 10.0, 80.0, 80.0, Rgba::BLUE);
        let encoder2 = encoder.clone();
        let _ = format!("{encoder2:?}");
    }

    #[test]
    fn test_svg_element_debug_clone() {
        let element = SvgElement::Circle {
            cx: 50.0,
            cy: 50.0,
            r: 10.0,
            fill: Rgba::RED,
            stroke: Some(Rgba::BLACK),
            stroke_width: 2.0,
        };
        let element2 = element.clone();
        let _ = format!("{element2:?}");
    }

    #[test]
    fn test_text_anchor_debug_clone() {
        let anchor = TextAnchor::Middle;
        let anchor2 = anchor;
        let _ = format!("{anchor2:?}");
    }

    #[test]
    fn test_svg_text_xml_entities() {
        let svg = SvgEncoder::new(100, 100)
            .text(10.0, 50.0, "A & B \"quoted\"", 12.0, Rgba::BLACK)
            .render();

        assert!(svg.contains("&amp;"));
        assert!(svg.contains("&quot;"));
    }

    #[test]
    fn test_svg_encoder_width_height() {
        let encoder = SvgEncoder::new(800, 600);
        assert_eq!(encoder.width(), 800);
        assert_eq!(encoder.height(), 600);
    }

    #[test]
    fn test_svg_encoder_display() {
        let encoder = SvgEncoder::new(400, 300).rect(10.0, 10.0, 100.0, 50.0, Rgba::BLUE);
        let display_str = format!("{encoder}");

        assert!(display_str.contains("<svg"));
        assert!(display_str.contains("</svg>"));
        assert!(display_str.contains("width=\"400\""));
        assert!(display_str.contains("height=\"300\""));
    }
}
