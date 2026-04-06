//! Draw commands for GPU rendering.
//!
//! All rendering reduces to these primitives.

use crate::{Color, CornerRadius, Point, Rect};
use serde::{Deserialize, Serialize};

/// Reference to a path in the path buffer.
pub type PathRef = u32;

/// Reference to a tensor in the tensor buffer.
pub type TensorRef = u32;

/// Fill rule for path filling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FillRule {
    /// Non-zero winding rule
    #[default]
    NonZero,
    /// Even-odd rule
    EvenOdd,
}

/// Stroke style for path rendering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrokeStyle {
    /// Stroke color
    pub color: Color,
    /// Stroke width in pixels
    pub width: f32,
    /// Line cap style
    pub cap: LineCap,
    /// Line join style
    pub join: LineJoin,
    /// Dash pattern (empty = solid)
    pub dash: Vec<f32>,
}

impl Default for StrokeStyle {
    fn default() -> Self {
        Self {
            color: Color::BLACK,
            width: 1.0,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
            dash: Vec::new(),
        }
    }
}

/// Line cap style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LineCap {
    /// Flat cap at endpoint
    #[default]
    Butt,
    /// Rounded cap
    Round,
    /// Square cap extending beyond endpoint
    Square,
}

/// Line join style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LineJoin {
    /// Sharp corner
    #[default]
    Miter,
    /// Rounded corner
    Round,
    /// Beveled corner
    Bevel,
}

/// Box style for rectangles and circles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoxStyle {
    /// Fill color (None = no fill)
    pub fill: Option<Color>,
    /// Stroke style (None = no stroke)
    pub stroke: Option<StrokeStyle>,
    /// Shadow (None = no shadow)
    pub shadow: Option<Shadow>,
}

impl Default for BoxStyle {
    fn default() -> Self {
        Self {
            fill: Some(Color::WHITE),
            stroke: None,
            shadow: None,
        }
    }
}

impl BoxStyle {
    /// Create a box with only fill color.
    #[must_use]
    pub const fn fill(color: Color) -> Self {
        Self {
            fill: Some(color),
            stroke: None,
            shadow: None,
        }
    }

    /// Create a box with only stroke.
    #[must_use]
    pub const fn stroke(style: StrokeStyle) -> Self {
        Self {
            fill: None,
            stroke: Some(style),
            shadow: None,
        }
    }

    /// Add a shadow to the box.
    #[must_use]
    pub const fn with_shadow(mut self, shadow: Shadow) -> Self {
        self.shadow = Some(shadow);
        self
    }
}

/// Shadow configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shadow {
    /// Shadow color
    pub color: Color,
    /// Horizontal offset
    pub offset_x: f32,
    /// Vertical offset
    pub offset_y: f32,
    /// Blur radius
    pub blur: f32,
}

impl Default for Shadow {
    fn default() -> Self {
        Self {
            color: Color::rgba(0.0, 0.0, 0.0, 0.3),
            offset_x: 0.0,
            offset_y: 2.0,
            blur: 4.0,
        }
    }
}

/// Image sampling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Sampling {
    /// Nearest neighbor (pixelated)
    Nearest,
    /// Bilinear interpolation (smooth)
    #[default]
    Bilinear,
    /// Trilinear with mipmaps
    Trilinear,
}

/// 2D transformation matrix.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform2D {
    /// Matrix elements [a, b, c, d, e, f]
    /// | a c e |
    /// | b d f |
    /// | 0 0 1 |
    pub matrix: [f32; 6],
}

impl Default for Transform2D {
    fn default() -> Self {
        Self::identity()
    }
}

impl Transform2D {
    /// Identity transformation.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            matrix: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }

    /// Create a translation transform.
    #[must_use]
    pub const fn translate(x: f32, y: f32) -> Self {
        Self {
            matrix: [1.0, 0.0, 0.0, 1.0, x, y],
        }
    }

    /// Create a scale transform.
    #[must_use]
    pub const fn scale(sx: f32, sy: f32) -> Self {
        Self {
            matrix: [sx, 0.0, 0.0, sy, 0.0, 0.0],
        }
    }

    /// Create a rotation transform (radians).
    #[must_use]
    pub fn rotate(angle: f32) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Self {
            matrix: [cos, sin, -sin, cos, 0.0, 0.0],
        }
    }

    /// Chain transforms: first apply self, then apply other.
    ///
    /// For point p: `a.then(b).apply(p)` == `b.apply(a.apply(p))`
    #[must_use]
    pub fn then(&self, other: &Self) -> Self {
        // For "first self, then other" semantics: result = other * self
        let a = other.matrix;
        let b = self.matrix;
        Self {
            matrix: [
                a[0].mul_add(b[0], a[2] * b[1]),
                a[1].mul_add(b[0], a[3] * b[1]),
                a[0].mul_add(b[2], a[2] * b[3]),
                a[1].mul_add(b[2], a[3] * b[3]),
                a[0].mul_add(b[4], a[2] * b[5]) + a[4],
                a[1].mul_add(b[4], a[3] * b[5]) + a[5],
            ],
        }
    }

    /// Transform a point.
    #[must_use]
    pub fn apply(&self, point: Point) -> Point {
        let m = self.matrix;
        Point::new(
            m[0].mul_add(point.x, m[2] * point.y) + m[4],
            m[1].mul_add(point.x, m[3] * point.y) + m[5],
        )
    }
}

/// Drawing primitive - all rendering reduces to these.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DrawCommand {
    /// Draw a path (polyline or polygon)
    Path {
        /// Points defining the path
        points: Vec<Point>,
        /// Whether the path is closed
        closed: bool,
        /// Stroke style
        style: StrokeStyle,
    },

    /// Fill a path
    Fill {
        /// Reference to path in buffer
        path: PathRef,
        /// Fill color
        color: Color,
        /// Fill rule
        rule: FillRule,
    },

    /// Draw a rectangle
    Rect {
        /// Rectangle bounds
        bounds: Rect,
        /// Corner radius
        radius: CornerRadius,
        /// Box style
        style: BoxStyle,
    },

    /// Draw a circle
    Circle {
        /// Center point
        center: Point,
        /// Radius
        radius: f32,
        /// Box style
        style: BoxStyle,
    },

    /// Draw an arc (pie slice)
    Arc {
        /// Center point
        center: Point,
        /// Radius
        radius: f32,
        /// Start angle in radians
        start_angle: f32,
        /// End angle in radians
        end_angle: f32,
        /// Fill color
        color: Color,
    },

    /// Draw text
    Text {
        /// Text content
        content: String,
        /// Position
        position: Point,
        /// Text style
        style: crate::widget::TextStyle,
    },

    /// Draw an image from tensor
    Image {
        /// Reference to tensor in buffer
        tensor: TensorRef,
        /// Destination bounds
        bounds: Rect,
        /// Sampling mode
        sampling: Sampling,
    },

    /// Group of commands with transform
    Group {
        /// Child commands
        children: Vec<Self>,
        /// Transform to apply
        transform: Transform2D,
    },

    /// Clip to bounds
    Clip {
        /// Clip bounds
        bounds: Rect,
        /// Child command
        child: Box<Self>,
    },

    /// Apply opacity
    Opacity {
        /// Alpha value (0.0 - 1.0)
        alpha: f32,
        /// Child command
        child: Box<Self>,
    },
}

impl DrawCommand {
    /// Create a filled rectangle.
    #[must_use]
    pub const fn filled_rect(bounds: Rect, color: Color) -> Self {
        Self::Rect {
            bounds,
            radius: CornerRadius::ZERO,
            style: BoxStyle::fill(color),
        }
    }

    /// Create a rounded rectangle.
    #[must_use]
    pub const fn rounded_rect(bounds: Rect, radius: f32, color: Color) -> Self {
        Self::Rect {
            bounds,
            radius: CornerRadius::uniform(radius),
            style: BoxStyle::fill(color),
        }
    }

    /// Create a stroked rectangle.
    #[must_use]
    pub const fn stroked_rect(bounds: Rect, stroke: StrokeStyle) -> Self {
        Self::Rect {
            bounds,
            radius: CornerRadius::ZERO,
            style: BoxStyle::stroke(stroke),
        }
    }

    /// Create a filled circle.
    #[must_use]
    pub const fn filled_circle(center: Point, radius: f32, color: Color) -> Self {
        Self::Circle {
            center,
            radius,
            style: BoxStyle::fill(color),
        }
    }

    /// Create a line between two points.
    #[must_use]
    pub fn line(from: Point, to: Point, style: StrokeStyle) -> Self {
        Self::Path {
            points: vec![from, to],
            closed: false,
            style,
        }
    }

    /// Wrap in a group with transform.
    #[must_use]
    pub fn with_transform(self, transform: Transform2D) -> Self {
        Self::Group {
            children: vec![self],
            transform,
        }
    }

    /// Wrap with opacity.
    #[must_use]
    pub fn with_opacity(self, alpha: f32) -> Self {
        Self::Opacity {
            alpha,
            child: Box::new(self),
        }
    }

    /// Wrap with clip bounds.
    #[must_use]
    pub fn with_clip(self, bounds: Rect) -> Self {
        Self::Clip {
            bounds,
            child: Box::new(self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // StrokeStyle Tests
    // =========================================================================

    #[test]
    fn test_stroke_style_default() {
        let style = StrokeStyle::default();
        assert_eq!(style.color, Color::BLACK);
        assert_eq!(style.width, 1.0);
        assert_eq!(style.cap, LineCap::Butt);
        assert_eq!(style.join, LineJoin::Miter);
        assert!(style.dash.is_empty());
    }

    #[test]
    fn test_line_cap_variants() {
        assert_eq!(LineCap::default(), LineCap::Butt);
        let _ = LineCap::Round;
        let _ = LineCap::Square;
    }

    #[test]
    fn test_line_join_variants() {
        assert_eq!(LineJoin::default(), LineJoin::Miter);
        let _ = LineJoin::Round;
        let _ = LineJoin::Bevel;
    }

    // =========================================================================
    // BoxStyle Tests
    // =========================================================================

    #[test]
    fn test_box_style_default() {
        let style = BoxStyle::default();
        assert_eq!(style.fill, Some(Color::WHITE));
        assert!(style.stroke.is_none());
        assert!(style.shadow.is_none());
    }

    #[test]
    fn test_box_style_fill() {
        let style = BoxStyle::fill(Color::RED);
        assert_eq!(style.fill, Some(Color::RED));
        assert!(style.stroke.is_none());
    }

    #[test]
    fn test_box_style_stroke() {
        let stroke = StrokeStyle {
            color: Color::BLUE,
            width: 2.0,
            ..Default::default()
        };
        let style = BoxStyle::stroke(stroke.clone());
        assert!(style.fill.is_none());
        assert_eq!(style.stroke, Some(stroke));
    }

    #[test]
    fn test_box_style_with_shadow() {
        let style = BoxStyle::fill(Color::WHITE).with_shadow(Shadow::default());
        assert!(style.shadow.is_some());
    }

    // =========================================================================
    // Shadow Tests
    // =========================================================================

    #[test]
    fn test_shadow_default() {
        let shadow = Shadow::default();
        assert_eq!(shadow.offset_x, 0.0);
        assert_eq!(shadow.offset_y, 2.0);
        assert_eq!(shadow.blur, 4.0);
    }

    // =========================================================================
    // Transform2D Tests
    // =========================================================================

    #[test]
    fn test_transform_identity() {
        let t = Transform2D::identity();
        assert_eq!(t.matrix, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_transform_translate() {
        let t = Transform2D::translate(10.0, 20.0);
        let p = t.apply(Point::new(0.0, 0.0));
        assert_eq!(p, Point::new(10.0, 20.0));
    }

    #[test]
    fn test_transform_scale() {
        let t = Transform2D::scale(2.0, 3.0);
        let p = t.apply(Point::new(5.0, 10.0));
        assert_eq!(p, Point::new(10.0, 30.0));
    }

    #[test]
    fn test_transform_rotate_90() {
        let t = Transform2D::rotate(std::f32::consts::FRAC_PI_2);
        let p = t.apply(Point::new(1.0, 0.0));
        assert!((p.x - 0.0).abs() < 0.0001);
        assert!((p.y - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_transform_chain() {
        let t1 = Transform2D::translate(10.0, 0.0);
        let t2 = Transform2D::scale(2.0, 2.0);
        let combined = t1.then(&t2);
        let p = combined.apply(Point::new(0.0, 0.0));
        assert_eq!(p, Point::new(20.0, 0.0));
    }

    // =========================================================================
    // FillRule Tests
    // =========================================================================

    #[test]
    fn test_fill_rule_default() {
        assert_eq!(FillRule::default(), FillRule::NonZero);
    }

    // =========================================================================
    // Sampling Tests
    // =========================================================================

    #[test]
    fn test_sampling_default() {
        assert_eq!(Sampling::default(), Sampling::Bilinear);
    }

    // =========================================================================
    // DrawCommand Tests
    // =========================================================================

    #[test]
    fn test_draw_command_filled_rect() {
        let cmd = DrawCommand::filled_rect(Rect::new(0.0, 0.0, 100.0, 50.0), Color::RED);
        match cmd {
            DrawCommand::Rect {
                bounds,
                radius,
                style,
            } => {
                assert_eq!(bounds.width, 100.0);
                assert_eq!(bounds.height, 50.0);
                assert!(radius.is_zero());
                assert_eq!(style.fill, Some(Color::RED));
            }
            _ => panic!("Expected Rect command"),
        }
    }

    #[test]
    fn test_draw_command_rounded_rect() {
        let cmd = DrawCommand::rounded_rect(Rect::new(0.0, 0.0, 100.0, 50.0), 8.0, Color::BLUE);
        match cmd {
            DrawCommand::Rect { radius, .. } => {
                assert!(radius.is_uniform());
                assert_eq!(radius.top_left, 8.0);
            }
            _ => panic!("Expected Rect command"),
        }
    }

    #[test]
    fn test_draw_command_stroked_rect() {
        let stroke = StrokeStyle {
            color: Color::GREEN,
            width: 3.0,
            ..Default::default()
        };
        let cmd = DrawCommand::stroked_rect(Rect::new(0.0, 0.0, 100.0, 50.0), stroke);
        match cmd {
            DrawCommand::Rect { style, .. } => {
                assert!(style.fill.is_none());
                assert!(style.stroke.is_some());
            }
            _ => panic!("Expected Rect command"),
        }
    }

    #[test]
    fn test_draw_command_filled_circle() {
        let cmd = DrawCommand::filled_circle(Point::new(50.0, 50.0), 25.0, Color::YELLOW);
        match cmd {
            DrawCommand::Circle {
                center,
                radius,
                style,
            } => {
                assert_eq!(center, Point::new(50.0, 50.0));
                assert_eq!(radius, 25.0);
                assert_eq!(style.fill, Some(Color::YELLOW));
            }
            _ => panic!("Expected Circle command"),
        }
    }

    #[test]
    fn test_draw_command_line() {
        let style = StrokeStyle::default();
        let cmd = DrawCommand::line(Point::new(0.0, 0.0), Point::new(100.0, 100.0), style);
        match cmd {
            DrawCommand::Path { points, closed, .. } => {
                assert_eq!(points.len(), 2);
                assert!(!closed);
            }
            _ => panic!("Expected Path command"),
        }
    }

    #[test]
    fn test_draw_command_with_transform() {
        let rect = DrawCommand::filled_rect(Rect::new(0.0, 0.0, 10.0, 10.0), Color::RED);
        let cmd = rect.with_transform(Transform2D::translate(5.0, 5.0));
        match cmd {
            DrawCommand::Group {
                children,
                transform,
            } => {
                assert_eq!(children.len(), 1);
                assert_eq!(transform.matrix[4], 5.0);
                assert_eq!(transform.matrix[5], 5.0);
            }
            _ => panic!("Expected Group command"),
        }
    }

    #[test]
    fn test_draw_command_with_opacity() {
        let rect = DrawCommand::filled_rect(Rect::new(0.0, 0.0, 10.0, 10.0), Color::RED);
        let cmd = rect.with_opacity(0.5);
        match cmd {
            DrawCommand::Opacity { alpha, .. } => {
                assert_eq!(alpha, 0.5);
            }
            _ => panic!("Expected Opacity command"),
        }
    }

    #[test]
    fn test_draw_command_with_clip() {
        let rect = DrawCommand::filled_rect(Rect::new(0.0, 0.0, 100.0, 100.0), Color::RED);
        let cmd = rect.with_clip(Rect::new(10.0, 10.0, 50.0, 50.0));
        match cmd {
            DrawCommand::Clip { bounds, .. } => {
                assert_eq!(bounds.x, 10.0);
                assert_eq!(bounds.width, 50.0);
            }
            _ => panic!("Expected Clip command"),
        }
    }

    #[test]
    fn test_draw_command_path() {
        let cmd = DrawCommand::Path {
            points: vec![
                Point::new(0.0, 0.0),
                Point::new(100.0, 0.0),
                Point::new(50.0, 100.0),
            ],
            closed: true,
            style: StrokeStyle::default(),
        };
        match cmd {
            DrawCommand::Path { points, closed, .. } => {
                assert_eq!(points.len(), 3);
                assert!(closed);
            }
            _ => panic!("Expected Path command"),
        }
    }

    #[test]
    fn test_draw_command_text() {
        let cmd = DrawCommand::Text {
            content: "Hello".to_string(),
            position: Point::new(10.0, 20.0),
            style: crate::widget::TextStyle::default(),
        };
        match cmd {
            DrawCommand::Text {
                content, position, ..
            } => {
                assert_eq!(content, "Hello");
                assert_eq!(position.x, 10.0);
            }
            _ => panic!("Expected Text command"),
        }
    }

    #[test]
    fn test_draw_command_image() {
        let cmd = DrawCommand::Image {
            tensor: 42,
            bounds: Rect::new(0.0, 0.0, 200.0, 150.0),
            sampling: Sampling::Bilinear,
        };
        match cmd {
            DrawCommand::Image {
                tensor,
                bounds,
                sampling,
            } => {
                assert_eq!(tensor, 42);
                assert_eq!(bounds.width, 200.0);
                assert_eq!(sampling, Sampling::Bilinear);
            }
            _ => panic!("Expected Image command"),
        }
    }

    #[test]
    fn test_draw_command_fill() {
        let cmd = DrawCommand::Fill {
            path: 1,
            color: Color::GREEN,
            rule: FillRule::EvenOdd,
        };
        match cmd {
            DrawCommand::Fill { path, color, rule } => {
                assert_eq!(path, 1);
                assert_eq!(color, Color::GREEN);
                assert_eq!(rule, FillRule::EvenOdd);
            }
            _ => panic!("Expected Fill command"),
        }
    }

    #[test]
    fn test_draw_command_nested_group() {
        let inner = DrawCommand::filled_rect(Rect::new(0.0, 0.0, 10.0, 10.0), Color::RED);
        let outer = DrawCommand::Group {
            children: vec![inner.with_transform(Transform2D::translate(5.0, 5.0))],
            transform: Transform2D::scale(2.0, 2.0),
        };
        match outer {
            DrawCommand::Group {
                children,
                transform,
            } => {
                assert_eq!(children.len(), 1);
                assert_eq!(transform.matrix[0], 2.0);
            }
            _ => panic!("Expected Group command"),
        }
    }
}
