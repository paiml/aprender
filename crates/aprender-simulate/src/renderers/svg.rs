//! SVG Renderer for simulation visualization.
//!
//! Consumes `Vec<RenderCommand>` and outputs Grid Protocol SVG strings
//! for consumption by rmedia's native SVG producer (`type="svg"`).
//!
//! # Grid Protocol Compliance
//!
//! Output SVGs follow the Grid Protocol specification:
//! - Canvas: 1920x1080, `viewBox` matches `width`/`height`
//! - 16x9 grid (120px cells) for element positioning
//! - Dark palette (`#0f172a` canvas, `#1e293b` panels)
//! - 18px minimum font size, 4.5:1 WCAG AA contrast
//! - Element IDs on every `<g>` group for rmedia targeting
//!
//! # References
//!
//! - SVG Grid Protocol: `docs/specifications/svg-grid-protocol.md`
//! - rmedia SVG producer: `paiml/rmedia#6`, `paiml/rmedia#7`

use crate::orbit::render::{Camera, Color, RenderCommand};
use serde::{Deserialize, Serialize};
use std::fmt::Write;

/// Grid Protocol canvas dimensions.
const CANVAS_WIDTH: f64 = 1920.0;
const CANVAS_HEIGHT: f64 = 1080.0;

/// Grid Protocol dark palette — canvas background.
const BG_CANVAS: &str = "#0f172a";

/// Grid Protocol dark palette — primary text.
const TEXT_PRIMARY: &str = "#f1f5f9";

/// SVG renderer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SvgConfig {
    /// Canvas width in pixels.
    pub width: f64,
    /// Canvas height in pixels.
    pub height: f64,
    /// Background color hex.
    pub background: String,
    /// Font family for text elements.
    pub font_family: String,
    /// Minimum font size (Grid Protocol: 18px).
    pub min_font_size: f64,
    /// Whether to include the grid manifest comment.
    pub include_manifest: bool,
}

impl Default for SvgConfig {
    fn default() -> Self {
        Self {
            width: CANVAS_WIDTH,
            height: CANVAS_HEIGHT,
            background: BG_CANVAS.to_string(),
            font_family: "'Segoe UI', 'Helvetica Neue', sans-serif".to_string(),
            min_font_size: 18.0,
            include_manifest: true,
        }
    }
}

/// SVG renderer that consumes render commands and produces SVG strings.
#[derive(Debug, Clone)]
pub struct SvgRenderer {
    config: SvgConfig,
    camera: Camera,
}

impl SvgRenderer {
    /// Create a new SVG renderer with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(SvgConfig::default())
    }

    /// Create a new SVG renderer with custom configuration.
    #[must_use]
    pub fn with_config(config: SvgConfig) -> Self {
        let camera = Camera {
            width: config.width,
            height: config.height,
            ..Camera::default()
        };
        Self { config, camera }
    }

    /// Render a set of commands to an SVG string.
    #[must_use]
    pub fn render(&mut self, commands: &[RenderCommand]) -> String {
        let mut elements: Vec<String> = Vec::new();
        let mut element_count = 0u32;

        for cmd in commands {
            match cmd {
                RenderCommand::SetCamera {
                    center_x,
                    center_y,
                    zoom,
                } => {
                    self.camera.center_x = *center_x;
                    self.camera.center_y = *center_y;
                    self.camera.zoom = *zoom;
                }
                RenderCommand::Clear { color } => {
                    let w = self.config.width;
                    let h = self.config.height;
                    let hex = color_to_hex(*color);
                    elements.push(format!(
                        r#"  <rect id="bg" x="0" y="0" width="{w}" height="{h}" fill="{hex}"/>"#
                    ));
                }
                RenderCommand::DrawCircle {
                    x,
                    y,
                    radius,
                    color,
                    filled,
                } => {
                    let (sx, sy) = self.camera.world_to_screen(*x, *y);
                    let hex = color_to_hex(*color);
                    let id = element_count;
                    element_count += 1;

                    if *filled {
                        elements.push(format!(
                            r#"  <circle id="circle-{id}" cx="{sx:.1}" cy="{sy:.1}" r="{radius:.1}" fill="{hex}"/>"#
                        ));
                    } else {
                        elements.push(format!(
                            r#"  <circle id="circle-{id}" cx="{sx:.1}" cy="{sy:.1}" r="{radius:.1}" fill="none" stroke="{hex}" stroke-width="2"/>"#
                        ));
                    }
                }
                RenderCommand::DrawLine {
                    x1,
                    y1,
                    x2,
                    y2,
                    color,
                } => {
                    let (sx1, sy1) = self.camera.world_to_screen(*x1, *y1);
                    let (sx2, sy2) = self.camera.world_to_screen(*x2, *y2);
                    let hex = color_to_hex(*color);
                    let id = element_count;
                    element_count += 1;

                    elements.push(format!(
                        r#"  <line id="line-{id}" x1="{sx1:.1}" y1="{sy1:.1}" x2="{sx2:.1}" y2="{sy2:.1}" stroke="{hex}" stroke-width="2"/>"#
                    ));
                }
                RenderCommand::DrawOrbitPath { points, color } => {
                    if points.len() < 2 {
                        continue;
                    }
                    let hex = color_to_hex(*color);
                    let id = element_count;
                    element_count += 1;

                    let mut d = String::new();
                    for (i, (x, y)) in points.iter().enumerate() {
                        let (sx, sy) = self.camera.world_to_screen(*x, *y);
                        if i == 0 {
                            let _ = write!(d, "M{sx:.1},{sy:.1}");
                        } else {
                            let _ = write!(d, " L{sx:.1},{sy:.1}");
                        }
                    }

                    elements.push(format!(
                        r#"  <path id="orbit-path-{id}" d="{d}" fill="none" stroke="{hex}" stroke-width="2" stroke-opacity="0.7"/>"#
                    ));
                }
                RenderCommand::DrawText { x, y, text, color } => {
                    let (sx, sy) = self.camera.world_to_screen(*x, *y);
                    let hex = color_to_hex(*color);
                    let id = element_count;
                    element_count += 1;
                    let escaped = xml_escape(text);
                    let font = &self.config.font_family;
                    let size = self.config.min_font_size;

                    elements.push(format!(
                        r#"  <text id="text-{id}" x="{sx:.1}" y="{sy:.1}" font-family="{font}" font-size="{size}" fill="{hex}">{escaped}</text>"#
                    ));
                }
                RenderCommand::DrawVelocity {
                    x,
                    y,
                    vx,
                    vy,
                    scale,
                    color,
                } => {
                    let (sx, sy) = self.camera.world_to_screen(*x, *y);
                    let ex = sx + vx * scale;
                    let ey = sy + vy * scale;
                    let hex = color_to_hex(*color);
                    let id = element_count;
                    element_count += 1;

                    elements.push(format!(
                        r#"  <line id="velocity-{id}" x1="{sx:.1}" y1="{sy:.1}" x2="{ex:.1}" y2="{ey:.1}" stroke="{hex}" stroke-width="2" marker-end="url(#arrowhead)"/>"#
                    ));
                }
                RenderCommand::HighlightBody {
                    x,
                    y,
                    radius,
                    color,
                } => {
                    let (sx, sy) = self.camera.world_to_screen(*x, *y);
                    let hex = color_to_hex(*color);
                    let id = element_count;
                    element_count += 1;
                    let outer_r = radius * 1.5;

                    elements.push(format!(
                        r#"  <circle id="highlight-{id}" cx="{sx:.1}" cy="{sy:.1}" r="{outer_r:.1}" fill="none" stroke="{hex}" stroke-width="3" stroke-dasharray="4,4"/>"#
                    ));
                }
            }
        }

        self.build_svg(&elements)
    }

    /// Build the complete SVG document from rendered elements.
    fn build_svg(&self, elements: &[String]) -> String {
        let w = self.config.width;
        let h = self.config.height;

        let mut svg = String::with_capacity(4096);

        let _ = writeln!(
            svg,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\">"
        );

        if self.config.include_manifest {
            let _ = writeln!(
                svg,
                "<!-- GRID PROTOCOL MANIFEST\n     Canvas: {w}x{h} | Grid: 16x9 | Cell: 120px\n     Renderer: simular SVG\n-->"
            );
        }

        // Arrowhead marker definition
        svg.push_str("  <defs>\n");
        svg.push_str("    <marker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" refX=\"10\" refY=\"3.5\" orient=\"auto\">\n");
        let _ = writeln!(
            svg,
            "      <polygon points=\"0 0, 10 3.5, 0 7\" fill=\"{TEXT_PRIMARY}\"/>"
        );
        svg.push_str("    </marker>\n");
        svg.push_str("  </defs>\n");

        for element in elements {
            svg.push_str(element);
            svg.push('\n');
        }

        svg.push_str("</svg>\n");
        svg
    }
}

impl Default for SvgRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert RGBA color to hex string.
#[must_use]
pub fn color_to_hex(color: Color) -> String {
    if color.a == 255 {
        format!("#{:02x}{:02x}{:02x}", color.r, color.g, color.b)
    } else {
        format!(
            "#{:02x}{:02x}{:02x}{:02x}",
            color.r, color.g, color.b, color.a
        )
    }
}

/// Escape XML special characters in text content.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit::render::Color;

    #[test]
    fn test_color_to_hex_opaque() {
        let c = Color::rgb(255, 128, 0);
        assert_eq!(color_to_hex(c), "#ff8000");
    }

    #[test]
    fn test_color_to_hex_transparent() {
        let c = Color::new(255, 0, 0, 128);
        assert_eq!(color_to_hex(c), "#ff000080");
    }

    #[test]
    fn test_color_to_hex_black() {
        assert_eq!(color_to_hex(Color::BLACK), "#000000");
    }

    #[test]
    fn test_color_to_hex_white() {
        assert_eq!(color_to_hex(Color::WHITE), "#ffffff");
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("a<b>c"), "a&lt;b&gt;c");
        assert_eq!(xml_escape("a&b"), "a&amp;b");
        assert_eq!(xml_escape(r#"a"b"#), "a&quot;b");
    }

    #[test]
    fn test_svg_config_default() {
        let config = SvgConfig::default();
        assert!((config.width - 1920.0).abs() < f64::EPSILON);
        assert!((config.height - 1080.0).abs() < f64::EPSILON);
        assert!((config.min_font_size - 18.0).abs() < f64::EPSILON);
        assert!(config.include_manifest);
    }

    #[test]
    fn test_renderer_new() {
        let renderer = SvgRenderer::new();
        assert!((renderer.config.width - 1920.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_renderer_default() {
        let renderer = SvgRenderer::default();
        assert!((renderer.config.width - 1920.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_render_empty_commands() {
        let mut renderer = SvgRenderer::new();
        let svg = renderer.render(&[]);
        assert!(svg.contains("viewBox=\"0 0 1920 1080\""));
        assert!(svg.contains("width=\"1920\""));
        assert!(svg.contains("height=\"1080\""));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_render_clear() {
        let mut renderer = SvgRenderer::new();
        let svg = renderer.render(&[RenderCommand::Clear {
            color: Color::BLACK,
        }]);
        assert!(svg.contains("id=\"bg\""));
        assert!(svg.contains("#000000"));
    }

    #[test]
    fn test_render_filled_circle() {
        let mut renderer = SvgRenderer::new();
        let commands = vec![
            RenderCommand::SetCamera {
                center_x: 0.0,
                center_y: 0.0,
                zoom: 1.0,
            },
            RenderCommand::DrawCircle {
                x: 0.0,
                y: 0.0,
                radius: 10.0,
                color: Color::SUN,
                filled: true,
            },
        ];
        let svg = renderer.render(&commands);
        assert!(svg.contains("circle"));
        assert!(svg.contains("id=\"circle-0\""));
        assert!(svg.contains("#ffcc00"));
    }

    #[test]
    fn test_render_unfilled_circle() {
        let mut renderer = SvgRenderer::new();
        let commands = vec![RenderCommand::DrawCircle {
            x: 0.0,
            y: 0.0,
            radius: 50.0,
            color: Color::WHITE,
            filled: false,
        }];
        let svg = renderer.render(&commands);
        assert!(svg.contains("fill=\"none\""));
        assert!(svg.contains("stroke="));
    }

    #[test]
    fn test_render_line() {
        let mut renderer = SvgRenderer::new();
        let commands = vec![RenderCommand::DrawLine {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 100.0,
            color: Color::GREEN,
        }];
        let svg = renderer.render(&commands);
        assert!(svg.contains("line"));
        assert!(svg.contains("id=\"line-0\""));
    }

    #[test]
    fn test_render_orbit_path() {
        let mut renderer = SvgRenderer::new();
        let commands = vec![RenderCommand::DrawOrbitPath {
            points: vec![(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)],
            color: Color::EARTH,
        }];
        let svg = renderer.render(&commands);
        assert!(svg.contains("path"));
        assert!(svg.contains("id=\"orbit-path-0\""));
        assert!(svg.contains("M"));
        assert!(svg.contains("L"));
    }

    #[test]
    fn test_render_orbit_path_single_point_skipped() {
        let mut renderer = SvgRenderer::new();
        let commands = vec![RenderCommand::DrawOrbitPath {
            points: vec![(0.0, 0.0)],
            color: Color::EARTH,
        }];
        let svg = renderer.render(&commands);
        assert!(!svg.contains("<path"));
    }

    #[test]
    fn test_render_text() {
        let mut renderer = SvgRenderer::new();
        let commands = vec![RenderCommand::DrawText {
            x: 10.0,
            y: 10.0,
            text: "Hello World".to_string(),
            color: Color::WHITE,
        }];
        let svg = renderer.render(&commands);
        assert!(svg.contains("<text"));
        assert!(svg.contains("id=\"text-0\""));
        assert!(svg.contains("Hello World"));
        assert!(svg.contains("font-size=\"18\""));
    }

    #[test]
    fn test_render_text_escaping() {
        let mut renderer = SvgRenderer::new();
        let commands = vec![RenderCommand::DrawText {
            x: 10.0,
            y: 10.0,
            text: "E=1e-9 & L<1e-12".to_string(),
            color: Color::WHITE,
        }];
        let svg = renderer.render(&commands);
        assert!(svg.contains("&amp;"));
        assert!(svg.contains("&lt;"));
    }

    #[test]
    fn test_render_velocity() {
        let mut renderer = SvgRenderer::new();
        let commands = vec![RenderCommand::DrawVelocity {
            x: 0.0,
            y: 0.0,
            vx: 50.0,
            vy: 30.0,
            scale: 1.0,
            color: Color::GREEN,
        }];
        let svg = renderer.render(&commands);
        assert!(svg.contains("<line"));
        assert!(svg.contains("id=\"velocity-0\""));
        assert!(svg.contains("marker-end"));
    }

    #[test]
    fn test_render_highlight() {
        let mut renderer = SvgRenderer::new();
        let commands = vec![RenderCommand::HighlightBody {
            x: 0.0,
            y: 0.0,
            radius: 20.0,
            color: Color::RED,
        }];
        let svg = renderer.render(&commands);
        assert!(svg.contains("<circle"));
        assert!(svg.contains("id=\"highlight-0\""));
        assert!(svg.contains("stroke-dasharray"));
    }

    #[test]
    fn test_render_camera_transform() {
        let mut renderer = SvgRenderer::new();
        let commands = vec![
            RenderCommand::SetCamera {
                center_x: 0.0,
                center_y: 0.0,
                zoom: 2.0,
            },
            RenderCommand::DrawCircle {
                x: 100.0,
                y: 0.0,
                radius: 5.0,
                color: Color::WHITE,
                filled: true,
            },
        ];
        let svg = renderer.render(&commands);
        // At zoom=2, x=100 should map to 960 + 100*2 = 1160
        assert!(svg.contains("1160.0"));
    }

    #[test]
    fn test_manifest_included() {
        let mut renderer = SvgRenderer::new();
        let svg = renderer.render(&[]);
        assert!(svg.contains("GRID PROTOCOL MANIFEST"));
    }

    #[test]
    fn test_manifest_excluded() {
        let config = SvgConfig {
            include_manifest: false,
            ..SvgConfig::default()
        };
        let mut renderer = SvgRenderer::with_config(config);
        let svg = renderer.render(&[]);
        assert!(!svg.contains("GRID PROTOCOL MANIFEST"));
    }

    #[test]
    fn test_arrowhead_marker_defined() {
        let mut renderer = SvgRenderer::new();
        let svg = renderer.render(&[]);
        assert!(svg.contains("marker id=\"arrowhead\""));
    }

    #[test]
    fn test_viewbox_parity() {
        let mut renderer = SvgRenderer::new();
        let svg = renderer.render(&[]);
        // Grid Protocol check #20: viewBox must match width/height
        assert!(svg.contains("viewBox=\"0 0 1920 1080\""));
        assert!(svg.contains("width=\"1920\""));
        assert!(svg.contains("height=\"1080\""));
    }

    #[test]
    fn test_element_ids_unique() {
        let mut renderer = SvgRenderer::new();
        let commands = vec![
            RenderCommand::DrawCircle {
                x: 0.0,
                y: 0.0,
                radius: 5.0,
                color: Color::WHITE,
                filled: true,
            },
            RenderCommand::DrawCircle {
                x: 10.0,
                y: 10.0,
                radius: 5.0,
                color: Color::RED,
                filled: true,
            },
            RenderCommand::DrawText {
                x: 0.0,
                y: 0.0,
                text: "A".to_string(),
                color: Color::WHITE,
            },
        ];
        let svg = renderer.render(&commands);
        assert!(svg.contains("id=\"circle-0\""));
        assert!(svg.contains("id=\"circle-1\""));
        assert!(svg.contains("id=\"text-2\""));
    }

    #[test]
    fn test_full_orbit_render() {
        let mut renderer = SvgRenderer::new();
        let commands = vec![
            RenderCommand::Clear {
                color: Color::BLACK,
            },
            RenderCommand::SetCamera {
                center_x: 0.0,
                center_y: 0.0,
                zoom: 300.0,
            },
            RenderCommand::DrawCircle {
                x: 0.0,
                y: 0.0,
                radius: 15.0,
                color: Color::SUN,
                filled: true,
            },
            RenderCommand::DrawOrbitPath {
                points: vec![(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)],
                color: Color::EARTH,
            },
            RenderCommand::DrawCircle {
                x: 1.0,
                y: 0.0,
                radius: 5.0,
                color: Color::EARTH,
                filled: true,
            },
            RenderCommand::DrawText {
                x: 1.1,
                y: 0.0,
                text: "Earth".to_string(),
                color: Color::WHITE,
            },
        ];
        let svg = renderer.render(&commands);

        // Structural checks
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>\n"));
        assert!(svg.contains("#ffcc00")); // Sun color
        assert!(svg.contains("Earth"));

        // Count elements
        let circle_count = svg.matches("<circle").count();
        assert_eq!(circle_count, 2); // Sun + Earth
        let path_count = svg.matches("<path").count();
        assert_eq!(path_count, 1); // Orbit trail
        let text_count = svg.matches("<text").count();
        assert_eq!(text_count, 1); // Earth label
    }
}
