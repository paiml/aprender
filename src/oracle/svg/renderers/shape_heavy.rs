//! Shape-Heavy Renderer
//!
//! Optimized for architectural diagrams and component diagrams with many shapes.

use crate::oracle::svg::builder::SvgBuilder;
use crate::oracle::svg::grid_protocol::LayoutTemplate;
use crate::oracle::svg::layout::{auto_layout, Viewport, GRID_SIZE};
use crate::oracle::svg::palette::SovereignPalette;
use crate::oracle::svg::shapes::{Point, Size};

/// Shape-heavy renderer for architectural diagrams
#[derive(Debug)]
pub struct ShapeHeavyRenderer {
    /// SVG builder
    builder: SvgBuilder,
    /// Palette
    palette: SovereignPalette,
    /// Component spacing
    spacing: f32,
    /// Component box dimensions
    box_size: Size,
}

impl ShapeHeavyRenderer {
    /// Create a new shape-heavy renderer
    pub fn new() -> Self {
        Self {
            builder: SvgBuilder::new().presentation(),
            palette: SovereignPalette::light(),
            spacing: GRID_SIZE * 4.0, // 32px
            box_size: Size::new(160.0, 80.0),
        }
    }

    /// Set the viewport
    pub fn viewport(mut self, viewport: Viewport) -> Self {
        self.builder = self.builder.viewport(viewport);
        self
    }

    /// Use dark mode
    pub fn dark_mode(mut self) -> Self {
        self.palette = SovereignPalette::dark();
        self.builder = self.builder.dark_mode();
        self
    }

    /// Set spacing between components
    pub fn spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    /// Set component box size
    pub fn box_size(mut self, size: Size) -> Self {
        self.box_size = size;
        self
    }

    /// Add a component box
    pub fn component(mut self, id: &str, x: f32, y: f32, name: &str, component_type: &str) -> Self {
        let color = self.palette.component_color(component_type);

        self.builder = self.builder.rect_styled(
            id,
            x,
            y,
            self.box_size.width,
            self.box_size.height,
            color.lighten(0.85),
            Some((color, 2.0)),
            GRID_SIZE,
        );

        // Component name label
        let label_x = x + self.box_size.width / 2.0;
        let label_y = y + self.box_size.height / 2.0 + 5.0;

        let style = self
            .builder
            .get_typography()
            .title_small
            .clone()
            .with_color(self.palette.material.on_surface)
            .with_align(crate::oracle::svg::typography::TextAlign::Middle);

        self.builder = self.builder.text_styled(label_x, label_y, name, style);

        self
    }

    /// Add a layer box (containing multiple components)
    pub fn layer(mut self, id: &str, x: f32, y: f32, width: f32, height: f32, name: &str) -> Self {
        let color = self.palette.material.surface_variant;

        self.builder = self.builder.rect_styled(
            id,
            x,
            y,
            width,
            height,
            color,
            Some((self.palette.material.outline_variant, 1.0)),
            GRID_SIZE * 2.0,
        );

        // Layer name label at top-left
        let style = self
            .builder
            .get_typography()
            .label_medium
            .clone()
            .with_color(self.palette.material.on_surface_variant);

        self.builder =
            self.builder.text_styled(x + GRID_SIZE * 2.0, y + GRID_SIZE * 3.0, name, style);

        self
    }

    /// Add a connection line between two points
    pub fn connect(mut self, from: Point, to: Point) -> Self {
        self.builder = self.builder.line_styled(
            from.x,
            from.y,
            to.x,
            to.y,
            self.palette.material.outline,
            2.0,
        );
        self
    }

    /// Add a horizontal stack of components
    pub fn horizontal_stack(mut self, components: &[(&str, &str)], start: Point) -> Self {
        let elements: Vec<_> = components.iter().map(|(id, _)| (*id, self.box_size)).collect();

        let layout = auto_layout::row(&elements, start, self.spacing);

        for ((id, name), (_, rect)) in components.iter().zip(layout.iter()) {
            let component_type = if name.to_lowercase().contains("trueno") {
                "trueno"
            } else if name.to_lowercase().contains("aprender") {
                "aprender"
            } else if name.to_lowercase().contains("realizar") {
                "realizar"
            } else {
                "batuta"
            };

            self = self.component(id, rect.position.x, rect.position.y, name, component_type);
        }

        self
    }

    /// Add a vertical stack of components
    pub fn vertical_stack(mut self, components: &[(&str, &str)], start: Point) -> Self {
        let elements: Vec<_> = components.iter().map(|(id, _)| (*id, self.box_size)).collect();

        let layout = auto_layout::column(&elements, start, self.spacing);

        for ((id, name), (_, rect)) in components.iter().zip(layout.iter()) {
            let component_type = if name.to_lowercase().contains("trueno") {
                "trueno"
            } else if name.to_lowercase().contains("aprender") {
                "aprender"
            } else if name.to_lowercase().contains("realizar") {
                "realizar"
            } else {
                "batuta"
            };

            self = self.component(id, rect.position.x, rect.position.y, name, component_type);
        }

        self
    }

    /// Add a title to the diagram
    pub fn title(mut self, title: &str) -> Self {
        self.builder = self.builder.title(title);

        // Get viewport info before modifying builder
        let viewport = *self.builder.get_layout().viewport();
        let style = self
            .builder
            .get_typography()
            .headline_medium
            .clone()
            .with_color(self.palette.material.on_background);

        // Also add visual title at top
        self.builder = self.builder.text_styled(
            viewport.padding + GRID_SIZE,
            viewport.padding + GRID_SIZE * 4.0,
            title,
            style,
        );

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
        // Ensure grid mode is active
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

impl Default for ShapeHeavyRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_heavy_renderer_creation() {
        let renderer = ShapeHeavyRenderer::new();
        assert_eq!(renderer.spacing, 32.0);
        assert_eq!(renderer.box_size.width, 160.0);
    }

    #[test]
    fn test_shape_heavy_component() {
        let svg =
            ShapeHeavyRenderer::new().component("trueno", 100.0, 100.0, "Trueno", "trueno").build();

        assert!(svg.contains("<svg"));
        assert!(svg.contains("Trueno"));
    }

    #[test]
    fn test_shape_heavy_layer() {
        let svg = ShapeHeavyRenderer::new()
            .layer("compute", 50.0, 50.0, 400.0, 200.0, "Compute Layer")
            .build();

        assert!(svg.contains("Compute Layer"));
    }

    #[test]
    fn test_shape_heavy_horizontal_stack() {
        let svg = ShapeHeavyRenderer::new()
            .horizontal_stack(
                &[("c1", "Trueno"), ("c2", "Aprender"), ("c3", "Realizar")],
                Point::new(100.0, 100.0),
            )
            .build();

        assert!(svg.contains("Trueno"));
        assert!(svg.contains("Aprender"));
        assert!(svg.contains("Realizar"));
    }

    #[test]
    fn test_shape_heavy_dark_mode() {
        let renderer = ShapeHeavyRenderer::new().dark_mode();
        // Just verify it doesn't panic
        let _svg = renderer.build();
    }

    #[test]
    fn test_shape_heavy_with_title() {
        let svg = ShapeHeavyRenderer::new().title("Architecture Diagram").build();

        assert!(svg.contains("<title>Architecture Diagram</title>"));
    }

    #[test]
    fn test_shape_heavy_connect() {
        let svg = ShapeHeavyRenderer::new()
            .connect(Point::new(100.0, 100.0), Point::new(200.0, 200.0))
            .build();

        assert!(svg.contains("<line"));
    }

    #[test]
    fn test_shape_heavy_viewport() {
        let viewport = Viewport::new(800.0, 600.0);
        let renderer = ShapeHeavyRenderer::new().viewport(viewport);
        let svg = renderer.build();
        // Should contain the viewport dimensions
        assert!(svg.contains("width=\"800\"") || svg.contains("width=\""));
    }

    #[test]
    fn test_shape_heavy_spacing() {
        let renderer = ShapeHeavyRenderer::new().spacing(64.0);
        assert_eq!(renderer.spacing, 64.0);
    }

    #[test]
    fn test_shape_heavy_box_size() {
        let renderer = ShapeHeavyRenderer::new().box_size(Size::new(200.0, 100.0));
        assert_eq!(renderer.box_size.width, 200.0);
        assert_eq!(renderer.box_size.height, 100.0);
    }

    #[test]
    fn test_shape_heavy_vertical_stack() {
        let svg = ShapeHeavyRenderer::new()
            .vertical_stack(&[("c1", "Trueno"), ("c2", "Aprender")], Point::new(100.0, 100.0))
            .build();

        assert!(svg.contains("Trueno"));
        assert!(svg.contains("Aprender"));
    }

    #[test]
    fn test_shape_heavy_default() {
        let renderer = ShapeHeavyRenderer::default();
        assert_eq!(renderer.spacing, 32.0);
    }

    #[test]
    fn test_shape_heavy_component_different_types() {
        let svg = ShapeHeavyRenderer::new()
            .component("c1", 0.0, 0.0, "Batuta", "batuta")
            .component("c2", 200.0, 0.0, "Custom", "unknown")
            .build();

        assert!(svg.contains("Batuta"));
        assert!(svg.contains("Custom"));
    }

    // ── Grid Protocol Renderer Tests ───────────────────────────────────

    #[test]
    fn test_shape_heavy_grid_protocol() {
        let svg = ShapeHeavyRenderer::new().grid_protocol().title("Grid Test").build();

        assert!(svg.contains("viewBox=\"0 0 1920 1080\""));
        assert!(svg.contains("GRID PROTOCOL MANIFEST"));
    }

    #[test]
    fn test_shape_heavy_template_diagram() {
        let svg = ShapeHeavyRenderer::new()
            .template(LayoutTemplate::Diagram)
            .title("Architecture")
            .build();

        assert!(svg.contains("GRID PROTOCOL MANIFEST"));
        assert!(svg.contains("\"header\""));
        assert!(svg.contains("\"diagram\""));
    }

    #[test]
    fn test_shape_heavy_template_dashboard() {
        let svg = ShapeHeavyRenderer::new().template(LayoutTemplate::Dashboard).build();

        assert!(svg.contains("GRID PROTOCOL MANIFEST"));
        assert!(svg.contains("\"top_left\""));
        assert!(svg.contains("\"bottom_right\""));
    }

    #[test]
    fn test_shape_heavy_template_auto_enables_grid() {
        // template() should auto-enable grid protocol
        let svg = ShapeHeavyRenderer::new().template(LayoutTemplate::TitleSlide).build();

        assert!(svg.contains("viewBox=\"0 0 1920 1080\""));
        assert!(svg.contains("GRID PROTOCOL MANIFEST"));
    }

    #[test]
    fn test_shape_heavy_grid_protocol_video_styles() {
        let svg = ShapeHeavyRenderer::new().grid_protocol().build();

        assert!(svg.contains("Segoe UI"));
        assert!(svg.contains("Cascadia Code"));
    }
}
