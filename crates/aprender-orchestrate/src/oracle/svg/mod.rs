//! SVG Generation Module
//!
//! SVG diagram generation with Material Design 3 and Grid Protocol modes:
//! - Color palettes (light/dark themes, VideoPalette for 1080p)
//! - Typography scales (Material + VideoTypography for presentations)
//! - Shape primitives
//! - Layout engine with collision detection
//! - Grid Protocol: cell-based 16x9 layout with provable non-overlap
//! - Specialized renderers for different diagram types
//! - Linting and validation (including video-mode rules)
//!
//! # Toyota Production System Principles
//!
//! - **Poka-Yoke**: Grid protocol prevents cell overlap via occupied-set tracking
//! - **Jidoka**: Lint validation stops on errors
//! - **Kaizen**: Continuous improvement via validation feedback

pub mod builder;
pub mod grid_protocol;
pub mod layout;
pub mod lint;
pub mod palette;
pub mod renderers;
pub mod shapes;
pub mod typography;

// Re-exports for convenience
#[allow(unused_imports)]
pub use builder::{ComponentDiagramBuilder, SvgBuilder, SvgElement};
#[allow(unused_imports)]
pub use grid_protocol::{
    GridError, GridProtocol, GridSpan, LayoutTemplate, PixelBounds, CANVAS_HEIGHT, CANVAS_WIDTH,
    CELL_PADDING, CELL_SIZE, GRID_COLS, GRID_ROWS, INTERNAL_PADDING, MIN_BLOCK_GAP, TOTAL_CELLS,
};
#[allow(unused_imports)]
pub use layout::{auto_layout, LayoutEngine, LayoutError, Viewport, GRID_SIZE};
#[allow(unused_imports)]
pub use lint::{LintConfig, LintResult, LintRule, LintSeverity, LintViolation, SvgLinter};
#[allow(unused_imports)]
pub use palette::{Color, MaterialPalette, SovereignPalette, VideoPalette};
#[allow(unused_imports)]
pub use renderers::{RenderMode, ShapeHeavyRenderer, TextHeavyRenderer};
#[allow(unused_imports)]
pub use shapes::{ArrowMarker, Circle, Line, Path, PathCommand, Point, Rect, Size, Text};
#[allow(unused_imports)]
pub use typography::{
    FontFamily, FontWeight, MaterialTypography, TextAlign, TextStyle, VideoTypography,
};

/// Generate a simple component diagram for the Sovereign AI Stack
///
/// Uses the grid protocol with Template E (Diagram) for provable
/// non-overlap layout at 1920x1080.
pub fn sovereign_stack_diagram() -> String {
    ShapeHeavyRenderer::new()
        .template(LayoutTemplate::Diagram)
        .title("Sovereign AI Stack Architecture")
        .layer("compute", 50.0, 100.0, 1820.0, 200.0, "Compute Layer")
        .horizontal_stack(
            &[("trueno", "Trueno"), ("repartir", "Repartir"), ("trueno_db", "Trueno DB")],
            Point::new(100.0, 150.0),
        )
        .layer("ml", 50.0, 350.0, 1820.0, 200.0, "ML Layer")
        .horizontal_stack(
            &[("aprender", "Aprender"), ("entrenar", "Entrenar"), ("realizar", "Realizar")],
            Point::new(100.0, 400.0),
        )
        .layer("orch", 50.0, 600.0, 1820.0, 150.0, "Orchestration")
        .component("batuta", 100.0, 650.0, "Batuta", "batuta")
        .build()
}

/// Generate a documentation diagram
pub fn documentation_diagram(title: &str, content: &[(&str, &str)]) -> String {
    let mut renderer = TextHeavyRenderer::new().title(title);

    for (heading, text) in content {
        renderer = renderer.heading(heading).paragraph(text);
    }

    renderer.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sovereign_stack_diagram() {
        let svg = sovereign_stack_diagram();

        assert!(svg.contains("<svg"));
        assert!(svg.contains("Trueno"));
        assert!(svg.contains("Aprender"));
        assert!(svg.contains("Batuta"));
        // Grid protocol is now active
        assert!(svg.contains("GRID PROTOCOL MANIFEST"));
        assert!(svg.contains("viewBox=\"0 0 1920 1080\""));
    }

    #[test]
    fn test_documentation_diagram() {
        let svg = documentation_diagram(
            "Test Doc",
            &[("Section 1", "Content for section 1."), ("Section 2", "Content for section 2.")],
        );

        assert!(svg.contains("Test Doc"));
        assert!(svg.contains("Section 1"));
        assert!(svg.contains("Section 2"));
    }

    #[test]
    fn test_svg_builder_quick() {
        let svg = SvgBuilder::new()
            .size(400.0, 300.0)
            .title("Quick Test")
            .rect("box", 50.0, 50.0, 100.0, 80.0)
            .text(60.0, 90.0, "Hello")
            .build();

        assert!(svg.contains("viewBox=\"0 0 400 300\""));
        assert!(svg.contains("<title>Quick Test</title>"));
    }

    #[test]
    fn test_render_mode_selection() {
        let mode: RenderMode = "shape-heavy".parse().expect("parse failed");
        assert_eq!(mode, RenderMode::ShapeHeavy);

        let mode: RenderMode = "text-heavy".parse().expect("parse failed");
        assert_eq!(mode, RenderMode::TextHeavy);
    }

    #[test]
    fn test_color_palette() {
        let palette = MaterialPalette::light();
        assert_eq!(palette.primary.to_css_hex(), "#6750A4");

        let palette = MaterialPalette::dark();
        assert_eq!(palette.primary.to_css_hex(), "#D0BCFF");
    }

    #[test]
    fn test_typography_scale() {
        let typo = MaterialTypography::default();
        assert_eq!(typo.display_large.size, 57.0);
        assert_eq!(typo.body_medium.size, 14.0);
        assert_eq!(typo.label_small.size, 11.0);
    }

    #[test]
    fn test_layout_grid_snap() {
        let engine = LayoutEngine::new(Viewport::default());

        // 8px grid
        assert_eq!(engine.snap_to_grid(10.0), 8.0);
        assert_eq!(engine.snap_to_grid(15.0), 16.0);
        assert_eq!(engine.snap_to_grid(8.0), 8.0);
    }

    #[test]
    fn test_lint_basic() {
        let linter = SvgLinter::new();

        // Valid color
        let palette = MaterialPalette::light();
        assert!(linter.lint_color(&palette.primary, None).is_none());

        // Invalid color
        assert!(linter.lint_color(&Color::rgb(1, 2, 3), None).is_some());
    }

    #[test]
    fn test_shapes() {
        let rect = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(rect.center().x, 60.0);
        assert_eq!(rect.center().y, 45.0);

        let circle = Circle::new(50.0, 50.0, 25.0);
        assert!(circle.contains(&Point::new(50.0, 50.0)));
    }

    #[test]
    fn test_output_size_limit() {
        // Generate a complex diagram
        let svg = ShapeHeavyRenderer::new()
            .title("Large Diagram")
            .horizontal_stack(
                &[
                    ("c1", "Component 1"),
                    ("c2", "Component 2"),
                    ("c3", "Component 3"),
                    ("c4", "Component 4"),
                    ("c5", "Component 5"),
                ],
                Point::new(50.0, 200.0),
            )
            .build();

        // Must be under 100KB
        assert!(svg.len() < 100_000, "SVG too large: {} bytes", svg.len());
    }
}
