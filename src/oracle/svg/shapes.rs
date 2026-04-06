//! SVG Shape Primitives
//!
//! Basic shapes for building diagrams: rectangles, circles, paths, text.

use super::palette::Color;
use super::typography::TextStyle;

/// Point in 2D space
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Distance to another point
    pub fn distance(&self, other: &Point) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    /// Midpoint between two points
    pub fn midpoint(&self, other: &Point) -> Point {
        Point::new(f32::midpoint(self.x, other.x), f32::midpoint(self.y, other.y))
    }
}

/// Size (width and height)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Size {
    pub width: f32,
    pub height: f32,
}

impl Size {
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    /// Area
    pub fn area(&self) -> f32 {
        self.width * self.height
    }
}

/// A rectangle
#[derive(Debug, Clone, PartialEq)]
pub struct Rect {
    /// Top-left corner position
    pub position: Point,
    /// Size
    pub size: Size,
    /// Corner radius (0 for sharp corners)
    pub corner_radius: f32,
    /// Fill color
    pub fill: Option<Color>,
    /// Stroke color
    pub stroke: Option<Color>,
    /// Stroke width
    pub stroke_width: f32,
}

impl Rect {
    /// Create a new rectangle
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            position: Point::new(x, y),
            size: Size::new(width, height),
            corner_radius: 0.0,
            fill: None,
            stroke: None,
            stroke_width: 1.0,
        }
    }

    /// Set corner radius
    pub fn with_radius(mut self, radius: f32) -> Self {
        self.corner_radius = radius;
        self
    }

    /// Set fill color
    pub fn with_fill(mut self, color: Color) -> Self {
        self.fill = Some(color);
        self
    }

    /// Set stroke
    pub fn with_stroke(mut self, color: Color, width: f32) -> Self {
        self.stroke = Some(color);
        self.stroke_width = width;
        self
    }

    /// Get center point
    pub fn center(&self) -> Point {
        Point::new(
            self.position.x + self.size.width / 2.0,
            self.position.y + self.size.height / 2.0,
        )
    }

    /// Get right edge x coordinate
    pub fn right(&self) -> f32 {
        self.position.x + self.size.width
    }

    /// Get bottom edge y coordinate
    pub fn bottom(&self) -> f32 {
        self.position.y + self.size.height
    }

    /// Check if a point is inside the rectangle
    pub fn contains(&self, point: &Point) -> bool {
        point.x >= self.position.x
            && point.x <= self.right()
            && point.y >= self.position.y
            && point.y <= self.bottom()
    }

    /// Check if two rectangles overlap
    pub fn intersects(&self, other: &Rect) -> bool {
        self.position.x < other.right()
            && self.right() > other.position.x
            && self.position.y < other.bottom()
            && self.bottom() > other.position.y
    }

    /// Render to SVG element
    pub fn to_svg(&self) -> String {
        let mut attrs = format!(
            "x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"",
            self.position.x, self.position.y, self.size.width, self.size.height
        );

        if self.corner_radius > 0.0 {
            attrs.push_str(&format!(" rx=\"{}\"", self.corner_radius));
        }

        if let Some(fill) = &self.fill {
            attrs.push_str(&format!(" fill=\"{}\"", fill.to_css_hex()));
        } else {
            attrs.push_str(" fill=\"none\"");
        }

        if let Some(stroke) = &self.stroke {
            attrs.push_str(&format!(
                " stroke=\"{}\" stroke-width=\"{}\"",
                stroke.to_css_hex(),
                self.stroke_width
            ));
        }

        format!("<rect {}/>", attrs)
    }
}

impl Default for Rect {
    fn default() -> Self {
        Self::new(0.0, 0.0, 100.0, 100.0)
    }
}

/// A circle
#[derive(Debug, Clone, PartialEq)]
pub struct Circle {
    /// Center position
    pub center: Point,
    /// Radius
    pub radius: f32,
    /// Fill color
    pub fill: Option<Color>,
    /// Stroke color
    pub stroke: Option<Color>,
    /// Stroke width
    pub stroke_width: f32,
}

impl Circle {
    /// Create a new circle
    pub fn new(cx: f32, cy: f32, r: f32) -> Self {
        Self { center: Point::new(cx, cy), radius: r, fill: None, stroke: None, stroke_width: 1.0 }
    }

    /// Set fill color
    pub fn with_fill(mut self, color: Color) -> Self {
        self.fill = Some(color);
        self
    }

    /// Set stroke
    pub fn with_stroke(mut self, color: Color, width: f32) -> Self {
        self.stroke = Some(color);
        self.stroke_width = width;
        self
    }

    /// Get bounding rectangle
    pub fn bounds(&self) -> Rect {
        Rect::new(
            self.center.x - self.radius,
            self.center.y - self.radius,
            self.radius * 2.0,
            self.radius * 2.0,
        )
    }

    /// Check if a point is inside the circle
    pub fn contains(&self, point: &Point) -> bool {
        self.center.distance(point) <= self.radius
    }

    /// Check if two circles overlap
    pub fn intersects(&self, other: &Circle) -> bool {
        self.center.distance(&other.center) < self.radius + other.radius
    }

    /// Render to SVG element
    pub fn to_svg(&self) -> String {
        let mut attrs =
            format!("cx=\"{}\" cy=\"{}\" r=\"{}\"", self.center.x, self.center.y, self.radius);

        if let Some(fill) = &self.fill {
            attrs.push_str(&format!(" fill=\"{}\"", fill.to_css_hex()));
        } else {
            attrs.push_str(" fill=\"none\"");
        }

        if let Some(stroke) = &self.stroke {
            attrs.push_str(&format!(
                " stroke=\"{}\" stroke-width=\"{}\"",
                stroke.to_css_hex(),
                self.stroke_width
            ));
        }

        format!("<circle {}/>", attrs)
    }
}

impl Default for Circle {
    fn default() -> Self {
        Self::new(50.0, 50.0, 25.0)
    }
}

/// A line segment
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    /// Start point
    pub start: Point,
    /// End point
    pub end: Point,
    /// Stroke color
    pub stroke: Color,
    /// Stroke width
    pub stroke_width: f32,
    /// Dash array (for dashed lines)
    pub dash_array: Option<String>,
}

impl Line {
    /// Create a new line
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            start: Point::new(x1, y1),
            end: Point::new(x2, y2),
            stroke: Color::rgb(0, 0, 0),
            stroke_width: 1.0,
            dash_array: None,
        }
    }

    /// Set stroke color
    pub fn with_stroke(mut self, color: Color) -> Self {
        self.stroke = color;
        self
    }

    /// Set stroke width
    pub fn with_stroke_width(mut self, width: f32) -> Self {
        self.stroke_width = width;
        self
    }

    /// Set dash pattern
    pub fn with_dash(mut self, pattern: &str) -> Self {
        self.dash_array = Some(pattern.to_string());
        self
    }

    /// Get the length of the line
    pub fn length(&self) -> f32 {
        self.start.distance(&self.end)
    }

    /// Get the midpoint
    pub fn midpoint(&self) -> Point {
        self.start.midpoint(&self.end)
    }

    /// Render to SVG element
    pub fn to_svg(&self) -> String {
        let mut attrs = format!(
            "x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\"",
            self.start.x,
            self.start.y,
            self.end.x,
            self.end.y,
            self.stroke.to_css_hex(),
            self.stroke_width
        );

        if let Some(dash) = &self.dash_array {
            attrs.push_str(&format!(" stroke-dasharray=\"{}\"", dash));
        }

        format!("<line {}/>", attrs)
    }
}

/// SVG path commands
#[derive(Debug, Clone)]
pub enum PathCommand {
    /// Move to (x, y)
    MoveTo(f32, f32),
    /// Line to (x, y)
    LineTo(f32, f32),
    /// Horizontal line to x
    HorizontalTo(f32),
    /// Vertical line to y
    VerticalTo(f32),
    /// Quadratic curve to (x, y) with control point (cx, cy)
    QuadraticTo { cx: f32, cy: f32, x: f32, y: f32 },
    /// Cubic curve to (x, y) with control points
    CubicTo { cx1: f32, cy1: f32, cx2: f32, cy2: f32, x: f32, y: f32 },
    /// Arc to (x, y)
    ArcTo { rx: f32, ry: f32, rotation: f32, large_arc: bool, sweep: bool, x: f32, y: f32 },
    /// Close path
    Close,
}

impl PathCommand {
    /// Convert to SVG path data string
    pub fn to_svg(&self) -> String {
        match self {
            Self::MoveTo(x, y) => format!("M {} {}", x, y),
            Self::LineTo(x, y) => format!("L {} {}", x, y),
            Self::HorizontalTo(x) => format!("H {}", x),
            Self::VerticalTo(y) => format!("V {}", y),
            Self::QuadraticTo { cx, cy, x, y } => format!("Q {} {} {} {}", cx, cy, x, y),
            Self::CubicTo { cx1, cy1, cx2, cy2, x, y } => {
                format!("C {} {} {} {} {} {}", cx1, cy1, cx2, cy2, x, y)
            }
            Self::ArcTo { rx, ry, rotation, large_arc, sweep, x, y } => format!(
                "A {} {} {} {} {} {} {}",
                rx,
                ry,
                rotation,
                i32::from(*large_arc),
                i32::from(*sweep),
                x,
                y
            ),
            Self::Close => "Z".to_string(),
        }
    }
}

/// A path shape
#[derive(Debug, Clone)]
pub struct Path {
    /// Path commands
    pub commands: Vec<PathCommand>,
    /// Fill color
    pub fill: Option<Color>,
    /// Stroke color
    pub stroke: Option<Color>,
    /// Stroke width
    pub stroke_width: f32,
}

impl Path {
    /// Create a new empty path
    pub fn new() -> Self {
        Self { commands: Vec::new(), fill: None, stroke: None, stroke_width: 1.0 }
    }

    /// Move to a point
    pub fn move_to(mut self, x: f32, y: f32) -> Self {
        self.commands.push(PathCommand::MoveTo(x, y));
        self
    }

    /// Line to a point
    pub fn line_to(mut self, x: f32, y: f32) -> Self {
        self.commands.push(PathCommand::LineTo(x, y));
        self
    }

    /// Quadratic curve to a point
    pub fn quad_to(mut self, cx: f32, cy: f32, x: f32, y: f32) -> Self {
        self.commands.push(PathCommand::QuadraticTo { cx, cy, x, y });
        self
    }

    /// Cubic curve to a point
    pub fn cubic_to(mut self, cx1: f32, cy1: f32, cx2: f32, cy2: f32, x: f32, y: f32) -> Self {
        self.commands.push(PathCommand::CubicTo { cx1, cy1, cx2, cy2, x, y });
        self
    }

    /// Close the path
    pub fn close(mut self) -> Self {
        self.commands.push(PathCommand::Close);
        self
    }

    /// Set fill color
    pub fn with_fill(mut self, color: Color) -> Self {
        self.fill = Some(color);
        self
    }

    /// Set stroke
    pub fn with_stroke(mut self, color: Color, width: f32) -> Self {
        self.stroke = Some(color);
        self.stroke_width = width;
        self
    }

    /// Get the path data string
    pub fn to_path_data(&self) -> String {
        self.commands.iter().map(|c| c.to_svg()).collect::<Vec<_>>().join(" ")
    }

    /// Render to SVG element
    pub fn to_svg(&self) -> String {
        let mut attrs = format!("d=\"{}\"", self.to_path_data());

        if let Some(fill) = &self.fill {
            attrs.push_str(&format!(" fill=\"{}\"", fill.to_css_hex()));
        } else {
            attrs.push_str(" fill=\"none\"");
        }

        if let Some(stroke) = &self.stroke {
            attrs.push_str(&format!(
                " stroke=\"{}\" stroke-width=\"{}\"",
                stroke.to_css_hex(),
                self.stroke_width
            ));
        }

        format!("<path {}/>", attrs)
    }
}

impl Default for Path {
    fn default() -> Self {
        Self::new()
    }
}

/// A text element
#[derive(Debug, Clone)]
pub struct Text {
    /// Position
    pub position: Point,
    /// Text content
    pub content: String,
    /// Style
    pub style: TextStyle,
}

impl Text {
    /// Create a new text element
    pub fn new(x: f32, y: f32, content: &str) -> Self {
        Self {
            position: Point::new(x, y),
            content: content.to_string(),
            style: TextStyle::default(),
        }
    }

    /// Set the text style
    pub fn with_style(mut self, style: TextStyle) -> Self {
        self.style = style;
        self
    }

    /// Render to SVG element
    pub fn to_svg(&self) -> String {
        let style_attrs = self.style.to_svg_attrs();
        format!(
            "<text x=\"{}\" y=\"{}\" {}>{}</text>",
            self.position.x,
            self.position.y,
            style_attrs,
            html_escape(&self.content)
        )
    }
}

/// Escape HTML special characters
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Arrow marker for line endings
#[derive(Debug, Clone)]
pub struct ArrowMarker {
    /// Marker ID
    pub id: String,
    /// Arrow color
    pub color: Color,
    /// Arrow size
    pub size: f32,
}

impl ArrowMarker {
    /// Create a new arrow marker
    pub fn new(id: &str, color: Color) -> Self {
        Self { id: id.to_string(), color, size: 10.0 }
    }

    /// Set the arrow size
    pub fn with_size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    /// Render to SVG marker definition
    pub fn to_svg_def(&self) -> String {
        format!(
            r#"<marker id="{}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="{}" markerHeight="{}" orient="auto-start-reverse">
  <path d="M 0 0 L 10 5 L 0 10 z" fill="{}"/>
</marker>"#,
            self.id,
            self.size,
            self.size,
            self.color.to_css_hex()
        )
    }
}

#[cfg(test)]
#[path = "shapes_tests.rs"]
mod tests;
