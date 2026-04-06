//! Text-Heavy Renderer
//!
//! Optimized for documentation diagrams and flowcharts with lots of text.

use crate::oracle::svg::builder::SvgBuilder;
use crate::oracle::svg::grid_protocol::LayoutTemplate;
use crate::oracle::svg::layout::{Viewport, GRID_SIZE};
use crate::oracle::svg::palette::SovereignPalette;
// Point may be needed for future layout features
#[allow(unused_imports)]
use crate::oracle::svg::shapes::Point;
// Typography types available for advanced text configuration
#[allow(unused_imports)]
use crate::oracle::svg::typography::{FontWeight, TextAlign, TextStyle};

/// Text-heavy renderer for documentation diagrams
#[derive(Debug)]
pub struct TextHeavyRenderer {
    /// SVG builder
    builder: SvgBuilder,
    /// Palette
    palette: SovereignPalette,
    /// Current y position for sequential text
    current_y: f32,
    /// Line height
    line_height: f32,
    /// Left margin
    margin_left: f32,
}

impl TextHeavyRenderer {
    /// Create a new text-heavy renderer
    pub fn new() -> Self {
        let viewport = Viewport::document();
        Self {
            builder: SvgBuilder::new().viewport(viewport),
            palette: SovereignPalette::light(),
            current_y: viewport.padding + GRID_SIZE * 4.0,
            line_height: GRID_SIZE * 3.0, // 24px
            margin_left: viewport.padding + GRID_SIZE * 2.0,
        }
    }

    /// Set the viewport
    pub fn viewport(mut self, viewport: Viewport) -> Self {
        self.current_y = viewport.padding + GRID_SIZE * 4.0;
        self.margin_left = viewport.padding + GRID_SIZE * 2.0;
        self.builder = self.builder.viewport(viewport);
        self
    }

    /// Use dark mode
    pub fn dark_mode(mut self) -> Self {
        self.palette = SovereignPalette::dark();
        self.builder = self.builder.dark_mode();
        self
    }

    /// Set line height
    pub fn line_height(mut self, height: f32) -> Self {
        self.line_height = height;
        self
    }

    /// Add a title
    pub fn title(mut self, text: &str) -> Self {
        self.builder = self.builder.title(text);

        let style = self
            .builder
            .get_typography()
            .headline_large
            .clone()
            .with_color(self.palette.material.on_background);

        self.builder = self.builder.text_styled(self.margin_left, self.current_y, text, style);
        self.current_y += GRID_SIZE * 6.0; // Extra space after title

        self
    }

    /// Add a section heading
    pub fn heading(mut self, text: &str) -> Self {
        self.current_y += GRID_SIZE * 2.0; // Space before heading

        let style = self
            .builder
            .get_typography()
            .headline_small
            .clone()
            .with_color(self.palette.material.on_background);

        self.builder = self.builder.text_styled(self.margin_left, self.current_y, text, style);
        self.current_y += GRID_SIZE * 4.0;

        self
    }

    /// Add a paragraph of text
    pub fn paragraph(mut self, text: &str) -> Self {
        let style = self
            .builder
            .get_typography()
            .body_medium
            .clone()
            .with_color(self.palette.material.on_surface);

        // Simple word wrapping (basic implementation)
        let max_width = 600.0; // Approximate max text width
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut current_line = String::new();
        let char_width = 8.0; // Approximate character width at 14px

        for word in words {
            let test_line = if current_line.is_empty() {
                word.to_string()
            } else {
                format!("{} {}", current_line, word)
            };

            if test_line.len() as f32 * char_width > max_width && !current_line.is_empty() {
                // Output current line and start new one
                self.builder = self.builder.text_styled(
                    self.margin_left,
                    self.current_y,
                    &current_line,
                    style.clone(),
                );
                self.current_y += self.line_height;
                current_line = word.to_string();
            } else {
                current_line = test_line;
            }
        }

        // Output remaining text
        if !current_line.is_empty() {
            self.builder =
                self.builder.text_styled(self.margin_left, self.current_y, &current_line, style);
            self.current_y += self.line_height;
        }

        self.current_y += GRID_SIZE; // Extra space after paragraph

        self
    }

    /// Add a bullet point
    pub fn bullet(mut self, text: &str) -> Self {
        let style = self
            .builder
            .get_typography()
            .body_medium
            .clone()
            .with_color(self.palette.material.on_surface);

        // Bullet character
        self.builder =
            self.builder.text_styled(self.margin_left, self.current_y, "•", style.clone());

        // Text after bullet
        self.builder = self.builder.text_styled(
            self.margin_left + GRID_SIZE * 2.0,
            self.current_y,
            text,
            style,
        );

        self.current_y += self.line_height;

        self
    }

    /// Add a numbered item
    pub fn numbered(mut self, number: u32, text: &str) -> Self {
        let style = self
            .builder
            .get_typography()
            .body_medium
            .clone()
            .with_color(self.palette.material.on_surface);

        // Number
        self.builder = self.builder.text_styled(
            self.margin_left,
            self.current_y,
            &format!("{}.", number),
            style.clone(),
        );

        // Text after number
        self.builder = self.builder.text_styled(
            self.margin_left + GRID_SIZE * 3.0,
            self.current_y,
            text,
            style,
        );

        self.current_y += self.line_height;

        self
    }

    /// Add a code block
    pub fn code(mut self, code: &str) -> Self {
        self.current_y += GRID_SIZE;

        // Code background
        let bg_color = self.palette.material.surface_variant;
        let lines: Vec<&str> = code.lines().collect();
        let code_height = (lines.len() as f32 * self.line_height) + GRID_SIZE * 2.0;

        self.builder = self.builder.rect_styled(
            "_code_bg",
            self.margin_left,
            self.current_y,
            500.0,
            code_height,
            bg_color,
            None,
            GRID_SIZE / 2.0,
        );

        self.current_y += GRID_SIZE;

        let style =
            self.builder.get_typography().code.clone().with_color(self.palette.material.on_surface);

        for line in lines {
            self.builder = self.builder.text_styled(
                self.margin_left + GRID_SIZE,
                self.current_y,
                line,
                style.clone(),
            );
            self.current_y += self.line_height;
        }

        self.current_y += GRID_SIZE * 2.0; // Extra space after code

        self
    }

    /// Add a labeled box (for key-value pairs)
    pub fn labeled_box(mut self, label: &str, value: &str) -> Self {
        let label_style = self
            .builder
            .get_typography()
            .label_medium
            .clone()
            .with_color(self.palette.material.on_surface_variant);

        let value_style = self
            .builder
            .get_typography()
            .body_medium
            .clone()
            .with_color(self.palette.material.on_surface);

        // Label
        self.builder =
            self.builder.text_styled(self.margin_left, self.current_y, label, label_style);

        // Value
        self.builder = self.builder.text_styled(
            self.margin_left + GRID_SIZE * 15.0,
            self.current_y,
            value,
            value_style,
        );

        self.current_y += self.line_height;

        self
    }

    /// Add a divider line
    pub fn divider(mut self) -> Self {
        self.current_y += GRID_SIZE;

        self.builder = self.builder.line_styled(
            self.margin_left,
            self.current_y,
            self.margin_left + 600.0,
            self.current_y,
            self.palette.material.outline_variant,
            1.0,
        );

        self.current_y += GRID_SIZE * 2.0;

        self
    }

    /// Add vertical space
    pub fn space(mut self, lines: u32) -> Self {
        self.current_y += self.line_height * lines as f32;
        self
    }

    /// Enable grid protocol mode with video palette and typography.
    pub fn grid_protocol(mut self) -> Self {
        self.builder = self.builder.grid_protocol().video_styles();
        self.palette = SovereignPalette::dark();
        self
    }

    /// Apply a layout template, allocating all regions.
    pub fn template(mut self, template: LayoutTemplate) -> Self {
        if !self.builder.is_grid_mode() {
            self = self.grid_protocol();
        }

        let allocations = template.allocations();
        for (name, span) in allocations {
            let _ = self.builder.allocate(name, span);
        }

        self
    }

    /// Build the SVG
    pub fn build(self) -> String {
        self.builder.build()
    }
}

impl Default for TextHeavyRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_heavy_renderer_creation() {
        let renderer = TextHeavyRenderer::new();
        assert_eq!(renderer.line_height, 24.0);
    }

    #[test]
    fn test_text_heavy_title() {
        let svg = TextHeavyRenderer::new().title("Documentation").build();

        assert!(svg.contains("<title>Documentation</title>"));
        assert!(svg.contains("Documentation"));
    }

    #[test]
    fn test_text_heavy_heading() {
        let svg = TextHeavyRenderer::new().title("Doc").heading("Section 1").build();

        assert!(svg.contains("Section 1"));
    }

    #[test]
    fn test_text_heavy_paragraph() {
        let svg = TextHeavyRenderer::new()
            .paragraph("This is a test paragraph with some text content.")
            .build();

        assert!(svg.contains("test paragraph"));
    }

    #[test]
    fn test_text_heavy_bullets() {
        let svg = TextHeavyRenderer::new().bullet("First item").bullet("Second item").build();

        assert!(svg.contains("First item"));
        assert!(svg.contains("Second item"));
        assert!(svg.contains("•"));
    }

    #[test]
    fn test_text_heavy_numbered() {
        let svg =
            TextHeavyRenderer::new().numbered(1, "First step").numbered(2, "Second step").build();

        assert!(svg.contains("1."));
        assert!(svg.contains("First step"));
    }

    #[test]
    fn test_text_heavy_code() {
        let svg = TextHeavyRenderer::new().code("let x = 42;\nprintln!(\"{}\", x);").build();

        assert!(svg.contains("let x = 42"));
    }

    #[test]
    fn test_text_heavy_labeled_box() {
        let svg = TextHeavyRenderer::new()
            .labeled_box("Version:", "1.0.0")
            .labeled_box("Author:", "John Doe")
            .build();

        assert!(svg.contains("Version:"));
        assert!(svg.contains("1.0.0"));
    }

    #[test]
    fn test_text_heavy_divider() {
        let svg = TextHeavyRenderer::new().divider().build();

        assert!(svg.contains("<line"));
    }

    #[test]
    fn test_text_heavy_dark_mode() {
        let renderer = TextHeavyRenderer::new().dark_mode();
        let _svg = renderer.build();
        // Just verify it doesn't panic
    }

    #[test]
    fn test_text_heavy_complete_document() {
        let svg = TextHeavyRenderer::new()
            .title("API Documentation")
            .paragraph("This document describes the API.")
            .heading("Endpoints")
            .bullet("GET /api/users")
            .bullet("POST /api/users")
            .divider()
            .heading("Examples")
            .code("curl https://api.example.com/users")
            .build();

        assert!(svg.contains("API Documentation"));
        assert!(svg.contains("Endpoints"));
        assert!(svg.contains("GET /api/users"));
    }

    #[test]
    fn test_text_heavy_default() {
        let renderer = TextHeavyRenderer::default();
        assert_eq!(renderer.line_height, 24.0);
    }

    #[test]
    fn test_text_heavy_viewport() {
        let viewport = Viewport::new(800.0, 600.0);
        let renderer = TextHeavyRenderer::new().viewport(viewport);
        let svg = renderer.build();
        // Should contain some width indicator
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn test_text_heavy_line_height() {
        let renderer = TextHeavyRenderer::new().line_height(32.0);
        assert_eq!(renderer.line_height, 32.0);
    }

    #[test]
    fn test_text_heavy_space() {
        let initial_y = TextHeavyRenderer::new().current_y;
        let renderer = TextHeavyRenderer::new().space(2);
        // Space should increase current_y
        assert!(renderer.current_y > initial_y);
    }

    #[test]
    fn test_text_heavy_long_paragraph() {
        // Test word wrapping with a long paragraph
        let long_text = "This is a very long paragraph that should trigger word wrapping because it exceeds the maximum width allowed for a single line in the text renderer.";
        let svg = TextHeavyRenderer::new().paragraph(long_text).build();
        // Should contain multiple text elements due to wrapping
        assert!(svg.contains("paragraph"));
    }

    #[test]
    fn test_text_heavy_empty_paragraph() {
        let svg = TextHeavyRenderer::new().paragraph("").build();
        // Should not crash on empty input
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn test_text_heavy_multiline_code() {
        let code = "line 1\nline 2\nline 3";
        let svg = TextHeavyRenderer::new().code(code).build();
        assert!(svg.contains("line 1"));
        assert!(svg.contains("line 2"));
        assert!(svg.contains("line 3"));
    }

    #[test]
    fn test_text_heavy_chain_methods() {
        let svg = TextHeavyRenderer::new()
            .line_height(28.0)
            .title("Test")
            .space(1)
            .paragraph("Content")
            .divider()
            .bullet("Item")
            .numbered(1, "Step")
            .code("code")
            .labeled_box("Key", "Value")
            .build();

        assert!(svg.contains("Test"));
        assert!(svg.contains("Content"));
        assert!(svg.contains("Item"));
    }

    #[test]
    fn test_text_heavy_viewport_updates_margins() {
        let viewport = Viewport::new(1024.0, 768.0);
        let renderer = TextHeavyRenderer::new().viewport(viewport);
        // Margins should be updated based on viewport padding
        assert!(renderer.margin_left > 0.0);
    }

    #[test]
    fn test_text_heavy_viewport_document() {
        let viewport = Viewport::document();
        let renderer = TextHeavyRenderer::new().viewport(viewport);
        let svg = renderer.build();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn test_text_heavy_heading_increases_y() {
        let initial = TextHeavyRenderer::new();
        let after_heading = TextHeavyRenderer::new().heading("Section");
        // Heading should increase y position
        assert!(after_heading.current_y > initial.current_y);
    }

    #[test]
    fn test_text_heavy_bullet_increases_y() {
        let initial = TextHeavyRenderer::new();
        let after_bullet = TextHeavyRenderer::new().bullet("Item");
        assert!(after_bullet.current_y > initial.current_y);
    }

    #[test]
    fn test_text_heavy_numbered_increases_y() {
        let initial = TextHeavyRenderer::new();
        let after_numbered = TextHeavyRenderer::new().numbered(1, "Step");
        assert!(after_numbered.current_y > initial.current_y);
    }

    // ── Grid Protocol Renderer Tests ───────────────────────────────────

    #[test]
    fn test_text_heavy_grid_protocol() {
        let svg = TextHeavyRenderer::new().grid_protocol().title("Grid Doc").build();

        assert!(svg.contains("viewBox=\"0 0 1920 1080\""));
        assert!(svg.contains("GRID PROTOCOL MANIFEST"));
    }

    #[test]
    fn test_text_heavy_template() {
        let svg = TextHeavyRenderer::new().template(LayoutTemplate::TwoColumn).build();

        assert!(svg.contains("GRID PROTOCOL MANIFEST"));
        assert!(svg.contains("\"header\""));
        assert!(svg.contains("\"left\""));
        assert!(svg.contains("\"right\""));
    }

    #[test]
    fn test_text_heavy_template_auto_enables_grid() {
        let svg = TextHeavyRenderer::new().template(LayoutTemplate::ReflectionReadings).build();

        assert!(svg.contains("viewBox=\"0 0 1920 1080\""));
    }
}
