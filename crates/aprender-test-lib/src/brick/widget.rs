//! Widget Integration: Verify-Measure-Layout-Paint (PROBAR-SPEC-009-P12)
//!
//! This module unifies the Brick and Widget concepts so that every widget
//! IS a brick, ensuring all UI components have verifiable assertions.

// Allow missing docs for geometry primitives - fields are self-explanatory
#![allow(missing_docs)]
//!
//! # Widget Lifecycle
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    WIDGET LIFECYCLE                          │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  1. VERIFY (Brick)                                          │
//! │     ├── Check all assertions                                │
//! │     ├── Validate budget constraints                         │
//! │     └── Return BrickVerification                            │
//! │              │                                               │
//! │              ▼ (only if valid)                              │
//! │  2. MEASURE (Widget)                                        │
//! │     ├── Compute intrinsic size                              │
//! │     └── Return Size                                         │
//! │              │                                               │
//! │              ▼                                               │
//! │  3. LAYOUT (Widget)                                         │
//! │     ├── Position self within bounds                         │
//! │     ├── Layout children recursively                         │
//! │     └── Return LayoutResult                                 │
//! │              │                                               │
//! │              ▼                                               │
//! │  4. PAINT (Widget)                                          │
//! │     ├── Generate DrawCommands                               │
//! │     ├── Record to Canvas                                    │
//! │     └── Batch for GPU rendering                             │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Jidoka Pattern
//!
//! The verify step implements Jidoka (stop-the-line): if any assertion
//! fails, rendering is blocked. This prevents invalid UI states from
//! ever being displayed.
//!
//! # References
//!
//! - PROBAR-SPEC-009-P12: Widget Integration - Presentar Unification

use std::any::Any;
use std::time::Duration;

use super::{Brick, BrickBudget};

/// 2D point coordinate for widget rendering
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WidgetPoint {
    /// X coordinate
    pub x: f32,
    /// Y coordinate
    pub y: f32,
}

impl WidgetPoint {
    /// Create a new point
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Origin point (0, 0)
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
}

/// 2D size (width, height)
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Size {
    /// Width in pixels
    pub width: f32,
    /// Height in pixels
    pub height: f32,
}

impl Size {
    /// Create a new size
    #[must_use]
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    /// Zero size
    pub const ZERO: Self = Self {
        width: 0.0,
        height: 0.0,
    };

    /// Check if size has positive area
    #[must_use]
    pub fn has_area(&self) -> bool {
        self.width > 0.0 && self.height > 0.0
    }
}

/// Rectangle with position and size
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    /// Create a new rectangle
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Create rectangle from origin and size
    #[must_use]
    pub const fn from_size(size: Size) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: size.width,
            height: size.height,
        }
    }

    /// Get the size of this rectangle
    #[must_use]
    pub const fn size(&self) -> Size {
        Size {
            width: self.width,
            height: self.height,
        }
    }

    /// Get the origin point
    #[must_use]
    pub const fn origin(&self) -> WidgetPoint {
        WidgetPoint {
            x: self.x,
            y: self.y,
        }
    }

    /// Check if point is inside rectangle
    #[must_use]
    pub fn contains(&self, point: WidgetPoint) -> bool {
        point.x >= self.x
            && point.x < self.x + self.width
            && point.y >= self.y
            && point.y < self.y + self.height
    }

    /// Convert to array [x, y, width, height]
    #[must_use]
    pub const fn to_array(&self) -> [f32; 4] {
        [self.x, self.y, self.width, self.height]
    }
}

/// WidgetColor with RGBA components for widget rendering
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WidgetColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl WidgetColor {
    /// Create a new color
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Create from RGB with full opacity
    #[must_use]
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Pure white
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };

    /// Pure black
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    /// Transparent
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    /// Convert to array [r, g, b, a]
    #[must_use]
    pub const fn to_array(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Create from hex (0xRRGGBB)
    #[must_use]
    pub fn from_hex(hex: u32) -> Self {
        let r = ((hex >> 16) & 0xFF) as f32 / 255.0;
        let g = ((hex >> 8) & 0xFF) as f32 / 255.0;
        let b = (hex & 0xFF) as f32 / 255.0;
        Self::rgb(r, g, b)
    }
}

/// Corner radius for rounded rectangles
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CornerRadius {
    pub top_left: f32,
    pub top_right: f32,
    pub bottom_left: f32,
    pub bottom_right: f32,
}

impl CornerRadius {
    /// Create uniform corner radius
    #[must_use]
    pub const fn uniform(radius: f32) -> Self {
        Self {
            top_left: radius,
            top_right: radius,
            bottom_left: radius,
            bottom_right: radius,
        }
    }

    /// No rounding
    pub const ZERO: Self = Self::uniform(0.0);
}

/// Text styling options
#[derive(Debug, Clone, Default)]
pub struct TextStyle {
    /// Font family
    pub font_family: String,
    /// Font size in pixels
    pub font_size: f32,
    /// Font weight (100-900)
    pub font_weight: u16,
    /// Text color
    pub color: WidgetColor,
    /// Line height multiplier
    pub line_height: f32,
}

impl TextStyle {
    /// Create a basic text style
    #[must_use]
    pub fn new(font_size: f32, color: WidgetColor) -> Self {
        Self {
            font_family: "sans-serif".into(),
            font_size,
            font_weight: 400,
            color,
            line_height: 1.2,
        }
    }
}

/// Stroke styling for paths and shapes
#[derive(Debug, Clone, Default)]
pub struct StrokeStyle {
    /// Stroke color
    pub color: WidgetColor,
    /// Stroke width
    pub width: f32,
    /// Line cap style
    pub line_cap: LineCap,
    /// Line join style
    pub line_join: LineJoin,
}

/// Line cap style
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LineCap {
    /// Flat line end
    #[default]
    Butt,
    /// Rounded line end
    Round,
    /// Square line end
    Square,
}

/// Line join style
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LineJoin {
    /// Mitered corner
    #[default]
    Miter,
    /// Rounded corner
    Round,
    /// Beveled corner
    Bevel,
}

/// 2D transformation matrix
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform2D {
    /// Matrix components [a, b, c, d, e, f]
    pub matrix: [f32; 6],
}

impl Default for Transform2D {
    fn default() -> Self {
        Self::identity()
    }
}

impl Transform2D {
    /// Identity transform
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            matrix: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }

    /// Translation transform
    #[must_use]
    pub fn translate(x: f32, y: f32) -> Self {
        Self {
            matrix: [1.0, 0.0, 0.0, 1.0, x, y],
        }
    }

    /// Scale transform
    #[must_use]
    pub fn scale(sx: f32, sy: f32) -> Self {
        Self {
            matrix: [sx, 0.0, 0.0, sy, 0.0, 0.0],
        }
    }

    /// Rotation transform (radians)
    #[must_use]
    pub fn rotate(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            matrix: [c, s, -s, c, 0.0, 0.0],
        }
    }
}

/// Draw command for GPU batching
///
/// All paint operations become commands that can be batched
/// for efficient GPU rendering.
#[derive(Debug, Clone)]
pub enum DrawCommand {
    /// Draw a rectangle
    Rect {
        bounds: Rect,
        color: WidgetColor,
        radius: CornerRadius,
    },
    /// Draw a circle
    Circle {
        center: WidgetPoint,
        radius: f32,
        color: WidgetColor,
    },
    /// Draw text
    Text {
        content: String,
        position: WidgetPoint,
        style: TextStyle,
    },
    /// Draw a path
    Path {
        points: Vec<WidgetPoint>,
        style: StrokeStyle,
        closed: bool,
    },
    /// Draw an image/tensor
    Image { data: Vec<u8>, bounds: Rect },
    /// Group of commands with transform
    Group {
        children: Vec<DrawCommand>,
        transform: Transform2D,
    },
    /// Fill rect with gradient
    Gradient {
        bounds: Rect,
        start_color: WidgetColor,
        end_color: WidgetColor,
        angle: f32,
    },
    /// Clear a region
    Clear { bounds: Rect, color: WidgetColor },
}

/// GPU instance for batched rendering
#[derive(Debug, Clone, Default)]
pub struct GpuInstance {
    /// Bounds [x, y, width, height]
    pub bounds: [f32; 4],
    /// WidgetColor [r, g, b, a]
    pub color: [f32; 4],
    /// Shape type (0=rect, 1=circle, 2=text, etc.)
    pub shape_type: u32,
    /// Corner radius for rects
    pub corner_radius: f32,
    /// Additional parameters
    pub params: [f32; 4],
}

/// Convert draw commands to GPU instances for batched rendering
#[must_use]
pub fn commands_to_gpu_instances(commands: &[DrawCommand]) -> Vec<GpuInstance> {
    let mut instances = Vec::with_capacity(commands.len());

    for cmd in commands {
        match cmd {
            DrawCommand::Rect {
                bounds,
                color,
                radius,
            } => {
                instances.push(GpuInstance {
                    bounds: bounds.to_array(),
                    color: color.to_array(),
                    shape_type: 0,
                    corner_radius: radius.top_left,
                    params: [0.0; 4],
                });
            }
            DrawCommand::Circle {
                center,
                radius,
                color,
            } => {
                instances.push(GpuInstance {
                    bounds: [
                        center.x - radius,
                        center.y - radius,
                        radius * 2.0,
                        radius * 2.0,
                    ],
                    color: color.to_array(),
                    shape_type: 1,
                    corner_radius: *radius,
                    params: [0.0; 4],
                });
            }
            DrawCommand::Clear { bounds, color } => {
                instances.push(GpuInstance {
                    bounds: bounds.to_array(),
                    color: color.to_array(),
                    shape_type: 3,
                    corner_radius: 0.0,
                    params: [0.0; 4],
                });
            }
            DrawCommand::Gradient {
                bounds,
                start_color,
                end_color,
                angle,
            } => {
                instances.push(GpuInstance {
                    bounds: bounds.to_array(),
                    color: start_color.to_array(),
                    shape_type: 4,
                    corner_radius: 0.0,
                    params: [end_color.r, end_color.g, end_color.b, *angle],
                });
            }
            // Text and Path require more complex handling
            DrawCommand::Text { .. } | DrawCommand::Path { .. } | DrawCommand::Image { .. } => {
                // These need separate render passes
            }
            DrawCommand::Group { children, .. } => {
                // Recursively process children
                instances.extend(commands_to_gpu_instances(children));
            }
        }
    }

    instances
}

/// Layout constraints from parent
#[derive(Debug, Clone, Copy, Default)]
pub struct Constraints {
    /// Minimum width
    pub min_width: f32,
    /// Maximum width
    pub max_width: f32,
    /// Minimum height
    pub min_height: f32,
    /// Maximum height
    pub max_height: f32,
}

impl Constraints {
    /// Create unbounded constraints
    #[must_use]
    pub fn unbounded() -> Self {
        Self {
            min_width: 0.0,
            max_width: f32::INFINITY,
            min_height: 0.0,
            max_height: f32::INFINITY,
        }
    }

    /// Create tight constraints (exact size)
    #[must_use]
    pub fn tight(size: Size) -> Self {
        Self {
            min_width: size.width,
            max_width: size.width,
            min_height: size.height,
            max_height: size.height,
        }
    }

    /// Create loose constraints (max size)
    #[must_use]
    pub fn loose(size: Size) -> Self {
        Self {
            min_width: 0.0,
            max_width: size.width,
            min_height: 0.0,
            max_height: size.height,
        }
    }

    /// Constrain a size to these constraints
    #[must_use]
    pub fn constrain(&self, size: Size) -> Size {
        Size {
            width: size.width.clamp(self.min_width, self.max_width),
            height: size.height.clamp(self.min_height, self.max_height),
        }
    }

    /// Check if size satisfies constraints
    #[must_use]
    pub fn is_satisfied_by(&self, size: Size) -> bool {
        size.width >= self.min_width
            && size.width <= self.max_width
            && size.height >= self.min_height
            && size.height <= self.max_height
    }
}

/// Result of layout phase
#[derive(Debug, Clone, Default)]
pub struct LayoutResult {
    /// Final bounds after layout
    pub bounds: Rect,
    /// Whether layout succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl LayoutResult {
    /// Create a successful layout result
    #[must_use]
    pub fn success(bounds: Rect) -> Self {
        Self {
            bounds,
            success: true,
            error: None,
        }
    }

    /// Create a failed layout result
    #[must_use]
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            bounds: Rect::default(),
            success: false,
            error: Some(error.into()),
        }
    }
}

/// UI event for widget interaction
#[derive(Debug, Clone)]
pub enum Event {
    /// Mouse click
    Click {
        position: WidgetPoint,
        button: WidgetMouseButton,
    },
    /// Mouse move
    MouseMove { position: WidgetPoint },
    /// Key press
    KeyPress { key: String, modifiers: Modifiers },
    /// Focus gained
    Focus,
    /// Focus lost
    Blur,
    /// Scroll
    Scroll { delta_x: f32, delta_y: f32 },
    /// Touch start
    TouchStart { position: WidgetPoint, id: u32 },
    /// Touch move
    TouchMove { position: WidgetPoint, id: u32 },
    /// Touch end
    TouchEnd { id: u32 },
}

/// Mouse button for widget events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WidgetMouseButton {
    Left,
    Right,
    Middle,
}

/// Keyboard modifiers
#[derive(Debug, Clone, Copy, Default)]
pub struct Modifiers {
    /// Shift key pressed
    pub shift: bool,
    /// Control key pressed
    pub ctrl: bool,
    /// Alt/Option key pressed
    pub alt: bool,
    /// Meta/Command key pressed
    pub meta: bool,
}

/// Canvas for recording draw commands
pub trait Canvas: Send + Sync {
    /// Record a draw command
    fn draw(&mut self, command: DrawCommand);

    /// Get all recorded commands
    fn commands(&self) -> &[DrawCommand];

    /// Clear all commands
    fn clear(&mut self);

    /// Create a sub-canvas with transform
    fn with_transform(&self, transform: Transform2D) -> Box<dyn Canvas>;

    /// Get canvas size
    fn size(&self) -> Size;
}

/// Simple canvas implementation for recording commands
#[derive(Debug)]
pub struct RecordingCanvas {
    commands: Vec<DrawCommand>,
    size: Size,
}

impl RecordingCanvas {
    /// Create a new recording canvas
    #[must_use]
    pub fn new(size: Size) -> Self {
        Self {
            commands: Vec::new(),
            size,
        }
    }
}

impl Canvas for RecordingCanvas {
    fn draw(&mut self, command: DrawCommand) {
        self.commands.push(command);
    }

    fn commands(&self) -> &[DrawCommand] {
        &self.commands
    }

    fn clear(&mut self) {
        self.commands.clear();
    }

    fn with_transform(&self, transform: Transform2D) -> Box<dyn Canvas> {
        Box::new(TransformedCanvas {
            inner: RecordingCanvas::new(self.size),
            transform,
        })
    }

    fn size(&self) -> Size {
        self.size
    }
}

/// Canvas with applied transform
struct TransformedCanvas {
    inner: RecordingCanvas,
    transform: Transform2D,
}

impl Canvas for TransformedCanvas {
    fn draw(&mut self, command: DrawCommand) {
        // Wrap command in group with transform
        self.inner.draw(DrawCommand::Group {
            children: vec![command],
            transform: self.transform,
        });
    }

    fn commands(&self) -> &[DrawCommand] {
        self.inner.commands()
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn with_transform(&self, transform: Transform2D) -> Box<dyn Canvas> {
        // Compose transforms
        let composed = Transform2D {
            matrix: [
                self.transform.matrix[0] * transform.matrix[0],
                self.transform.matrix[1],
                self.transform.matrix[2],
                self.transform.matrix[3] * transform.matrix[3],
                self.transform.matrix[4] + transform.matrix[4],
                self.transform.matrix[5] + transform.matrix[5],
            ],
        };
        Box::new(TransformedCanvas {
            inner: RecordingCanvas::new(self.inner.size),
            transform: composed,
        })
    }

    fn size(&self) -> Size {
        self.inner.size
    }
}

/// Widget trait: Every widget must also be a Brick (PROBAR-SPEC-009)
///
/// This unifies UI components with verifiable assertions, ensuring
/// that no widget can render without passing its assertions.
pub trait Widget: Brick {
    /// Step 1: Compute intrinsic size given constraints
    fn measure(&self, constraints: Constraints) -> Size;

    /// Step 2: Position self and children within bounds
    fn layout(&mut self, bounds: Rect) -> LayoutResult;

    /// Step 3: Generate draw commands (only called if verified!)
    fn paint(&self, canvas: &mut dyn Canvas);

    /// Handle interaction events
    fn event(&mut self, event: &Event) -> Option<Box<dyn Any>>;

    /// Get children widgets (for tree traversal)
    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    /// Get mutable children
    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

/// Extension trait for verified rendering
pub trait WidgetExt: Widget {
    /// Render widget with Jidoka verification
    ///
    /// Only paints if all assertions pass. This is the Jidoka
    /// (stop-the-line) pattern for UI rendering.
    fn render(&self, canvas: &mut dyn Canvas) {
        let verification = self.verify();
        if !verification.is_valid() {
            // Jidoka: Stop the line - don't render invalid state
            return;
        }
        self.paint(canvas);
    }

    /// Render with timing and return metrics
    fn render_timed(&self, canvas: &mut dyn Canvas) -> RenderMetrics {
        let start = std::time::Instant::now();

        let verification = self.verify();
        let verify_time = start.elapsed();

        if !verification.is_valid() {
            return RenderMetrics {
                verify_time,
                paint_time: Duration::ZERO,
                total_time: verify_time,
                valid: false,
                command_count: 0,
            };
        }

        let paint_start = std::time::Instant::now();
        self.paint(canvas);
        let paint_time = paint_start.elapsed();

        RenderMetrics {
            verify_time,
            paint_time,
            total_time: start.elapsed(),
            valid: true,
            command_count: canvas.commands().len(),
        }
    }

    /// Full layout-paint cycle with verification
    fn render_full(&mut self, bounds: Rect, canvas: &mut dyn Canvas) -> LayoutResult
    where
        Self: Sized,
    {
        // Verify first
        let verification = self.verify();
        if !verification.is_valid() {
            return LayoutResult::failure("Brick verification failed");
        }

        // Layout
        let layout_result = self.layout(bounds);
        if !layout_result.success {
            return layout_result;
        }

        // Paint
        self.paint(canvas);

        layout_result
    }
}

impl<W: Widget> WidgetExt for W {}

/// Metrics from rendering
#[derive(Debug, Clone, Default)]
pub struct RenderMetrics {
    /// Time spent verifying
    pub verify_time: Duration,
    /// Time spent painting
    pub paint_time: Duration,
    /// Total render time
    pub total_time: Duration,
    /// Whether verification passed
    pub valid: bool,
    /// Number of draw commands generated
    pub command_count: usize,
}

impl RenderMetrics {
    /// Check if render was within budget
    #[must_use]
    pub fn within_budget(&self, budget: BrickBudget) -> bool {
        self.total_time <= budget.as_duration()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::brick::{BrickAssertion, BrickVerification};

    // ============================================================
    // Test Widget Implementation
    // ============================================================

    /// Test widget implementation
    struct TestWidget {
        text: String,
        size: Size,
        assertions: Vec<BrickAssertion>,
    }

    impl TestWidget {
        fn new(text: &str) -> Self {
            Self {
                text: text.to_string(),
                size: Size::new(100.0, 50.0),
                assertions: vec![
                    BrickAssertion::TextVisible,
                    BrickAssertion::ContrastRatio(4.5),
                ],
            }
        }
    }

    impl Brick for TestWidget {
        fn brick_name(&self) -> &'static str {
            "TestWidget"
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &self.assertions
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(16)
        }

        fn verify(&self) -> BrickVerification {
            let mut passed = Vec::new();
            let mut failed = Vec::new();

            for assertion in &self.assertions {
                match assertion {
                    BrickAssertion::TextVisible => {
                        if !self.text.is_empty() {
                            passed.push(assertion.clone());
                        } else {
                            failed.push((assertion.clone(), "Empty text".into()));
                        }
                    }
                    _ => passed.push(assertion.clone()),
                }
            }

            BrickVerification {
                passed,
                failed,
                verification_time: Duration::from_micros(50),
            }
        }

        fn to_html(&self) -> String {
            format!("<div class=\"widget\">{}</div>", self.text)
        }

        fn to_css(&self) -> String {
            ".widget { display: flex; }".into()
        }
    }

    impl Widget for TestWidget {
        fn measure(&self, constraints: Constraints) -> Size {
            constraints.constrain(self.size)
        }

        fn layout(&mut self, bounds: Rect) -> LayoutResult {
            LayoutResult::success(bounds)
        }

        fn paint(&self, canvas: &mut dyn Canvas) {
            canvas.draw(DrawCommand::Rect {
                bounds: Rect::new(0.0, 0.0, self.size.width, self.size.height),
                color: WidgetColor::WHITE,
                radius: CornerRadius::ZERO,
            });
            canvas.draw(DrawCommand::Text {
                content: self.text.clone(),
                position: WidgetPoint::new(10.0, 25.0),
                style: TextStyle::new(16.0, WidgetColor::BLACK),
            });
        }

        fn event(&mut self, event: &Event) -> Option<Box<dyn Any>> {
            match event {
                Event::Click { .. } => Some(Box::new("clicked")),
                _ => None,
            }
        }
    }

    // ============================================================
    // WidgetPoint tests
    // ============================================================

    #[test]
    fn test_point() {
        let p = WidgetPoint::new(10.0, 20.0);
        assert_eq!(p.x, 10.0);
        assert_eq!(p.y, 20.0);
        assert_eq!(WidgetPoint::ZERO, WidgetPoint::new(0.0, 0.0));
    }

    #[test]
    fn test_point_default() {
        let p = WidgetPoint::default();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
    }

    #[test]
    fn test_point_debug_and_clone() {
        let p = WidgetPoint::new(1.0, 2.0);
        let cloned = p;
        assert!(format!("{:?}", cloned).contains("WidgetPoint"));
    }

    #[test]
    fn test_point_equality() {
        let p1 = WidgetPoint::new(10.0, 20.0);
        let p2 = WidgetPoint::new(10.0, 20.0);
        let p3 = WidgetPoint::new(10.0, 30.0);

        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    // ============================================================
    // Size tests
    // ============================================================

    #[test]
    fn test_size() {
        let s = Size::new(100.0, 50.0);
        assert!(s.has_area());
        assert!(!Size::ZERO.has_area());
    }

    #[test]
    fn test_size_default() {
        let s = Size::default();
        assert_eq!(s.width, 0.0);
        assert_eq!(s.height, 0.0);
    }

    #[test]
    fn test_size_has_area_edge_cases() {
        assert!(!Size::new(0.0, 100.0).has_area());
        assert!(!Size::new(100.0, 0.0).has_area());
        assert!(!Size::new(-1.0, 100.0).has_area());
        assert!(Size::new(0.001, 0.001).has_area());
    }

    #[test]
    fn test_size_debug_and_clone() {
        let s = Size::new(50.0, 100.0);
        let cloned = s;
        assert!(format!("{:?}", cloned).contains("Size"));
    }

    #[test]
    fn test_size_equality() {
        assert_eq!(Size::new(10.0, 20.0), Size::new(10.0, 20.0));
        assert_ne!(Size::new(10.0, 20.0), Size::new(10.0, 30.0));
    }

    // ============================================================
    // Rect tests
    // ============================================================

    #[test]
    fn test_rect() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(r.contains(WidgetPoint::new(50.0, 30.0)));
        assert!(!r.contains(WidgetPoint::new(5.0, 30.0)));
        assert_eq!(r.size(), Size::new(100.0, 50.0));
    }

    #[test]
    fn test_rect_default() {
        let r = Rect::default();
        assert_eq!(r.x, 0.0);
        assert_eq!(r.y, 0.0);
        assert_eq!(r.width, 0.0);
        assert_eq!(r.height, 0.0);
    }

    #[test]
    fn test_rect_from_size() {
        let r = Rect::from_size(Size::new(100.0, 50.0));
        assert_eq!(r.x, 0.0);
        assert_eq!(r.y, 0.0);
        assert_eq!(r.width, 100.0);
        assert_eq!(r.height, 50.0);
    }

    #[test]
    fn test_rect_origin() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        let origin = r.origin();
        assert_eq!(origin.x, 10.0);
        assert_eq!(origin.y, 20.0);
    }

    #[test]
    fn test_rect_contains_edge_cases() {
        let r = Rect::new(0.0, 0.0, 100.0, 100.0);

        // Inside
        assert!(r.contains(WidgetPoint::new(50.0, 50.0)));

        // On edges (inclusive at start, exclusive at end)
        assert!(r.contains(WidgetPoint::new(0.0, 0.0)));
        assert!(r.contains(WidgetPoint::new(99.9, 99.9)));
        assert!(!r.contains(WidgetPoint::new(100.0, 50.0)));
        assert!(!r.contains(WidgetPoint::new(50.0, 100.0)));

        // Outside
        assert!(!r.contains(WidgetPoint::new(-1.0, 50.0)));
        assert!(!r.contains(WidgetPoint::new(50.0, -1.0)));
    }

    #[test]
    fn test_rect_to_array() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(r.to_array(), [10.0, 20.0, 100.0, 50.0]);
    }

    #[test]
    fn test_rect_debug_and_clone() {
        let r = Rect::new(1.0, 2.0, 3.0, 4.0);
        let cloned = r;
        assert!(format!("{:?}", cloned).contains("Rect"));
    }

    // ============================================================
    // WidgetColor tests
    // ============================================================

    #[test]
    fn test_color() {
        let c = WidgetColor::from_hex(0xFF0000);
        assert!((c.r - 1.0).abs() < f32::EPSILON);
        assert!(c.g.abs() < f32::EPSILON);
        assert!(c.b.abs() < f32::EPSILON);
    }

    #[test]
    fn test_color_new() {
        let c = WidgetColor::new(0.5, 0.6, 0.7, 0.8);
        assert!((c.r - 0.5).abs() < f32::EPSILON);
        assert!((c.g - 0.6).abs() < f32::EPSILON);
        assert!((c.b - 0.7).abs() < f32::EPSILON);
        assert!((c.a - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_color_rgb() {
        let c = WidgetColor::rgb(0.1, 0.2, 0.3);
        assert!((c.r - 0.1).abs() < f32::EPSILON);
        assert!((c.g - 0.2).abs() < f32::EPSILON);
        assert!((c.b - 0.3).abs() < f32::EPSILON);
        assert!((c.a - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_color_constants() {
        assert_eq!(WidgetColor::WHITE.r, 1.0);
        assert_eq!(WidgetColor::WHITE.g, 1.0);
        assert_eq!(WidgetColor::WHITE.b, 1.0);
        assert_eq!(WidgetColor::WHITE.a, 1.0);

        assert_eq!(WidgetColor::BLACK.r, 0.0);
        assert_eq!(WidgetColor::BLACK.g, 0.0);
        assert_eq!(WidgetColor::BLACK.b, 0.0);
        assert_eq!(WidgetColor::BLACK.a, 1.0);

        assert_eq!(WidgetColor::TRANSPARENT.a, 0.0);
    }

    #[test]
    fn test_color_to_array() {
        let c = WidgetColor::new(0.1, 0.2, 0.3, 0.4);
        let arr = c.to_array();
        assert!((arr[0] - 0.1).abs() < f32::EPSILON);
        assert!((arr[1] - 0.2).abs() < f32::EPSILON);
        assert!((arr[2] - 0.3).abs() < f32::EPSILON);
        assert!((arr[3] - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_color_from_hex_all_colors() {
        // Red
        let red = WidgetColor::from_hex(0xFF0000);
        assert!((red.r - 1.0).abs() < f32::EPSILON);

        // Green
        let green = WidgetColor::from_hex(0x00FF00);
        assert!((green.g - 1.0).abs() < f32::EPSILON);

        // Blue
        let blue = WidgetColor::from_hex(0x0000FF);
        assert!((blue.b - 1.0).abs() < f32::EPSILON);

        // Gray
        let gray = WidgetColor::from_hex(0x808080);
        assert!((gray.r - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_color_default() {
        let c = WidgetColor::default();
        assert_eq!(c.r, 0.0);
        assert_eq!(c.g, 0.0);
        assert_eq!(c.b, 0.0);
        assert_eq!(c.a, 0.0);
    }

    // ============================================================
    // CornerRadius tests
    // ============================================================

    #[test]
    fn test_corner_radius_uniform() {
        let r = CornerRadius::uniform(10.0);
        assert_eq!(r.top_left, 10.0);
        assert_eq!(r.top_right, 10.0);
        assert_eq!(r.bottom_left, 10.0);
        assert_eq!(r.bottom_right, 10.0);
    }

    #[test]
    fn test_corner_radius_zero() {
        let r = CornerRadius::ZERO;
        assert_eq!(r.top_left, 0.0);
        assert_eq!(r.top_right, 0.0);
        assert_eq!(r.bottom_left, 0.0);
        assert_eq!(r.bottom_right, 0.0);
    }

    #[test]
    fn test_corner_radius_default() {
        let r = CornerRadius::default();
        assert_eq!(r.top_left, 0.0);
    }

    #[test]
    fn test_corner_radius_debug_and_clone() {
        let r = CornerRadius::uniform(5.0);
        let cloned = r;
        assert!(format!("{:?}", cloned).contains("CornerRadius"));
    }

    // ============================================================
    // TextStyle tests
    // ============================================================

    #[test]
    fn test_text_style() {
        let style = TextStyle::new(16.0, WidgetColor::BLACK);
        assert_eq!(style.font_size, 16.0);
        assert_eq!(style.font_family, "sans-serif");
    }

    #[test]
    fn test_text_style_default() {
        let style = TextStyle::default();
        assert!(style.font_family.is_empty());
        assert_eq!(style.font_size, 0.0);
        assert_eq!(style.font_weight, 0);
    }

    #[test]
    fn test_text_style_full() {
        let style = TextStyle::new(24.0, WidgetColor::WHITE);
        assert_eq!(style.font_size, 24.0);
        assert_eq!(style.font_weight, 400);
        assert!((style.line_height - 1.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_text_style_debug_and_clone() {
        let style = TextStyle::new(12.0, WidgetColor::BLACK);
        let cloned = style;
        assert!(format!("{:?}", cloned).contains("TextStyle"));
    }

    // ============================================================
    // StrokeStyle tests
    // ============================================================

    #[test]
    fn test_stroke_style_defaults() {
        let style = StrokeStyle::default();
        assert!(matches!(style.line_cap, LineCap::Butt));
        assert!(matches!(style.line_join, LineJoin::Miter));
    }

    #[test]
    fn test_stroke_style_debug_and_clone() {
        let style = StrokeStyle::default();
        let cloned = style;
        assert!(format!("{:?}", cloned).contains("StrokeStyle"));
    }

    // ============================================================
    // LineCap and LineJoin tests
    // ============================================================

    #[test]
    fn test_line_cap_variants() {
        let butt = LineCap::Butt;
        let round = LineCap::Round;
        let square = LineCap::Square;

        assert_eq!(butt, LineCap::default());
        assert_ne!(round, square);
    }

    #[test]
    fn test_line_join_variants() {
        let miter = LineJoin::Miter;
        let round = LineJoin::Round;
        let bevel = LineJoin::Bevel;

        assert_eq!(miter, LineJoin::default());
        assert_ne!(round, bevel);
    }

    // ============================================================
    // Transform2D tests
    // ============================================================

    #[test]
    fn test_transform() {
        let t = Transform2D::translate(10.0, 20.0);
        assert_eq!(t.matrix[4], 10.0);
        assert_eq!(t.matrix[5], 20.0);

        let s = Transform2D::scale(2.0, 3.0);
        assert_eq!(s.matrix[0], 2.0);
        assert_eq!(s.matrix[3], 3.0);
    }

    #[test]
    fn test_transform_identity() {
        let t = Transform2D::identity();
        assert_eq!(t.matrix, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_transform_default() {
        let t = Transform2D::default();
        assert_eq!(t.matrix, Transform2D::identity().matrix);
    }

    #[test]
    fn test_transform_rotate() {
        use std::f32::consts::PI;

        // 90 degree rotation
        let t = Transform2D::rotate(PI / 2.0);
        assert!((t.matrix[0]).abs() < 0.0001); // cos(90) = 0
        assert!((t.matrix[1] - 1.0).abs() < 0.0001); // sin(90) = 1
    }

    #[test]
    fn test_transform_debug_and_clone() {
        let t = Transform2D::translate(1.0, 2.0);
        let cloned = t;
        assert!(format!("{:?}", cloned).contains("Transform2D"));
    }

    #[test]
    fn test_transform_equality() {
        let t1 = Transform2D::translate(10.0, 20.0);
        let t2 = Transform2D::translate(10.0, 20.0);
        let t3 = Transform2D::scale(2.0, 2.0);

        assert_eq!(t1, t2);
        assert_ne!(t1, t3);
    }

    // ============================================================
    // DrawCommand tests
    // ============================================================

    #[test]
    fn test_draw_command_rect() {
        let cmd = DrawCommand::Rect {
            bounds: Rect::new(0.0, 0.0, 100.0, 50.0),
            color: WidgetColor::WHITE,
            radius: CornerRadius::uniform(5.0),
        };

        assert!(format!("{:?}", cmd).contains("Rect"));
    }

    #[test]
    fn test_draw_command_circle() {
        let cmd = DrawCommand::Circle {
            center: WidgetPoint::new(50.0, 50.0),
            radius: 25.0,
            color: WidgetColor::BLACK,
        };

        assert!(format!("{:?}", cmd).contains("Circle"));
    }

    #[test]
    fn test_draw_command_text() {
        let cmd = DrawCommand::Text {
            content: "Hello".to_string(),
            position: WidgetPoint::new(10.0, 20.0),
            style: TextStyle::new(16.0, WidgetColor::BLACK),
        };

        assert!(format!("{:?}", cmd).contains("Text"));
    }

    #[test]
    fn test_draw_command_path() {
        let cmd = DrawCommand::Path {
            points: vec![WidgetPoint::new(0.0, 0.0), WidgetPoint::new(100.0, 100.0)],
            style: StrokeStyle::default(),
            closed: false,
        };

        assert!(format!("{:?}", cmd).contains("Path"));
    }

    #[test]
    fn test_draw_command_image() {
        let cmd = DrawCommand::Image {
            data: vec![0, 1, 2, 3],
            bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
        };

        assert!(format!("{:?}", cmd).contains("Image"));
    }

    #[test]
    fn test_draw_command_group() {
        let cmd = DrawCommand::Group {
            children: vec![DrawCommand::Clear {
                bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
                color: WidgetColor::WHITE,
            }],
            transform: Transform2D::identity(),
        };

        assert!(format!("{:?}", cmd).contains("Group"));
    }

    #[test]
    fn test_draw_command_gradient() {
        let cmd = DrawCommand::Gradient {
            bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
            start_color: WidgetColor::WHITE,
            end_color: WidgetColor::BLACK,
            angle: 45.0,
        };

        assert!(format!("{:?}", cmd).contains("Gradient"));
    }

    #[test]
    fn test_draw_command_clear() {
        let cmd = DrawCommand::Clear {
            bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
            color: WidgetColor::TRANSPARENT,
        };

        assert!(format!("{:?}", cmd).contains("Clear"));
    }

    #[test]
    fn test_draw_command_clone() {
        let cmd = DrawCommand::Rect {
            bounds: Rect::new(0.0, 0.0, 100.0, 50.0),
            color: WidgetColor::WHITE,
            radius: CornerRadius::ZERO,
        };

        let _cloned = cmd;
    }

    // ============================================================
    // GpuInstance tests
    // ============================================================

    #[test]
    fn test_gpu_instance_default() {
        let instance = GpuInstance::default();
        assert_eq!(instance.shape_type, 0);
        assert_eq!(instance.corner_radius, 0.0);
    }

    #[test]
    fn test_gpu_instance_debug_and_clone() {
        let instance = GpuInstance {
            bounds: [0.0, 0.0, 100.0, 50.0],
            color: [1.0, 1.0, 1.0, 1.0],
            shape_type: 0,
            corner_radius: 5.0,
            params: [0.0; 4],
        };

        let cloned = instance;
        assert!(format!("{:?}", cloned).contains("GpuInstance"));
    }

    // ============================================================
    // commands_to_gpu_instances tests
    // ============================================================

    #[test]
    fn test_gpu_instances() {
        let commands = vec![
            DrawCommand::Rect {
                bounds: Rect::new(0.0, 0.0, 100.0, 50.0),
                color: WidgetColor::WHITE,
                radius: CornerRadius::uniform(5.0),
            },
            DrawCommand::Circle {
                center: WidgetPoint::new(50.0, 50.0),
                radius: 25.0,
                color: WidgetColor::BLACK,
            },
        ];

        let instances = commands_to_gpu_instances(&commands);
        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].shape_type, 0); // Rect
        assert_eq!(instances[1].shape_type, 1); // Circle
    }

    #[test]
    fn test_gpu_instances_clear() {
        let commands = vec![DrawCommand::Clear {
            bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
            color: WidgetColor::WHITE,
        }];

        let instances = commands_to_gpu_instances(&commands);
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].shape_type, 3); // Clear
    }

    #[test]
    fn test_gpu_instances_gradient() {
        let commands = vec![DrawCommand::Gradient {
            bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
            start_color: WidgetColor::WHITE,
            end_color: WidgetColor::BLACK,
            angle: 45.0,
        }];

        let instances = commands_to_gpu_instances(&commands);
        assert_eq!(instances.len(), 1);
        assert_eq!(instances[0].shape_type, 4); // Gradient
    }

    #[test]
    fn test_gpu_instances_group_recursive() {
        let commands = vec![DrawCommand::Group {
            children: vec![
                DrawCommand::Rect {
                    bounds: Rect::new(0.0, 0.0, 50.0, 50.0),
                    color: WidgetColor::WHITE,
                    radius: CornerRadius::ZERO,
                },
                DrawCommand::Circle {
                    center: WidgetPoint::new(25.0, 25.0),
                    radius: 10.0,
                    color: WidgetColor::BLACK,
                },
            ],
            transform: Transform2D::identity(),
        }];

        let instances = commands_to_gpu_instances(&commands);
        assert_eq!(instances.len(), 2); // Children are flattened
    }

    #[test]
    fn test_gpu_instances_skips_text_path_image() {
        let commands = vec![
            DrawCommand::Text {
                content: "Hello".to_string(),
                position: WidgetPoint::ZERO,
                style: TextStyle::default(),
            },
            DrawCommand::Path {
                points: vec![],
                style: StrokeStyle::default(),
                closed: false,
            },
            DrawCommand::Image {
                data: vec![],
                bounds: Rect::default(),
            },
        ];

        let instances = commands_to_gpu_instances(&commands);
        assert_eq!(instances.len(), 0); // These need separate render passes
    }

    // ============================================================
    // Constraints tests
    // ============================================================

    #[test]
    fn test_constraints() {
        let constraints = Constraints::loose(Size::new(200.0, 100.0));
        let result = constraints.constrain(Size::new(300.0, 50.0));
        assert_eq!(result.width, 200.0);
        assert_eq!(result.height, 50.0);
    }

    #[test]
    fn test_constraints_unbounded() {
        let c = Constraints::unbounded();
        assert_eq!(c.min_width, 0.0);
        assert_eq!(c.min_height, 0.0);
        assert_eq!(c.max_width, f32::INFINITY);
        assert_eq!(c.max_height, f32::INFINITY);
    }

    #[test]
    fn test_constraints_tight() {
        let c = Constraints::tight(Size::new(100.0, 50.0));
        assert_eq!(c.min_width, 100.0);
        assert_eq!(c.max_width, 100.0);
        assert_eq!(c.min_height, 50.0);
        assert_eq!(c.max_height, 50.0);
    }

    #[test]
    fn test_constraints_loose() {
        let c = Constraints::loose(Size::new(100.0, 50.0));
        assert_eq!(c.min_width, 0.0);
        assert_eq!(c.max_width, 100.0);
        assert_eq!(c.min_height, 0.0);
        assert_eq!(c.max_height, 50.0);
    }

    #[test]
    fn test_constraints_constrain() {
        let c = Constraints {
            min_width: 50.0,
            max_width: 150.0,
            min_height: 25.0,
            max_height: 75.0,
        };

        // Below min
        let result = c.constrain(Size::new(10.0, 10.0));
        assert_eq!(result, Size::new(50.0, 25.0));

        // Above max
        let result = c.constrain(Size::new(200.0, 200.0));
        assert_eq!(result, Size::new(150.0, 75.0));

        // Within range
        let result = c.constrain(Size::new(100.0, 50.0));
        assert_eq!(result, Size::new(100.0, 50.0));
    }

    #[test]
    fn test_constraints_is_satisfied_by() {
        let c = Constraints {
            min_width: 50.0,
            max_width: 150.0,
            min_height: 25.0,
            max_height: 75.0,
        };

        assert!(c.is_satisfied_by(Size::new(100.0, 50.0)));
        assert!(c.is_satisfied_by(Size::new(50.0, 25.0)));
        assert!(c.is_satisfied_by(Size::new(150.0, 75.0)));
        assert!(!c.is_satisfied_by(Size::new(40.0, 50.0)));
        assert!(!c.is_satisfied_by(Size::new(100.0, 80.0)));
    }

    #[test]
    fn test_constraints_default() {
        let c = Constraints::default();
        assert_eq!(c.min_width, 0.0);
        assert_eq!(c.min_height, 0.0);
        assert_eq!(c.max_width, 0.0);
        assert_eq!(c.max_height, 0.0);
    }

    #[test]
    fn test_constraints_debug_and_clone() {
        let c = Constraints::loose(Size::new(100.0, 100.0));
        let cloned = c;
        assert!(format!("{:?}", cloned).contains("Constraints"));
    }

    // ============================================================
    // LayoutResult tests
    // ============================================================

    #[test]
    fn test_layout_result() {
        let success = LayoutResult::success(Rect::new(0.0, 0.0, 100.0, 50.0));
        assert!(success.success);

        let failure = LayoutResult::failure("Test error");
        assert!(!failure.success);
        assert_eq!(failure.error, Some("Test error".to_string()));
    }

    #[test]
    fn test_layout_result_default() {
        let r = LayoutResult::default();
        assert!(!r.success);
        assert!(r.error.is_none());
    }

    #[test]
    fn test_layout_result_debug_and_clone() {
        let r = LayoutResult::success(Rect::default());
        let cloned = r;
        assert!(format!("{:?}", cloned).contains("LayoutResult"));
    }

    // ============================================================
    // Event tests
    // ============================================================

    #[test]
    fn test_event_click() {
        let event = Event::Click {
            position: WidgetPoint::new(10.0, 20.0),
            button: WidgetMouseButton::Left,
        };

        assert!(format!("{:?}", event).contains("Click"));
    }

    #[test]
    fn test_event_mouse_move() {
        let event = Event::MouseMove {
            position: WidgetPoint::new(50.0, 50.0),
        };

        assert!(format!("{:?}", event).contains("MouseMove"));
    }

    #[test]
    fn test_event_key_press() {
        let event = Event::KeyPress {
            key: "Enter".to_string(),
            modifiers: Modifiers {
                shift: true,
                ctrl: false,
                alt: false,
                meta: false,
            },
        };

        assert!(format!("{:?}", event).contains("KeyPress"));
    }

    #[test]
    fn test_event_focus_blur() {
        let focus = Event::Focus;
        let blur = Event::Blur;

        assert!(format!("{:?}", focus).contains("Focus"));
        assert!(format!("{:?}", blur).contains("Blur"));
    }

    #[test]
    fn test_event_scroll() {
        let event = Event::Scroll {
            delta_x: 10.0,
            delta_y: -20.0,
        };

        assert!(format!("{:?}", event).contains("Scroll"));
    }

    #[test]
    fn test_event_touch() {
        let start = Event::TouchStart {
            position: WidgetPoint::new(100.0, 200.0),
            id: 1,
        };
        let move_ev = Event::TouchMove {
            position: WidgetPoint::new(110.0, 210.0),
            id: 1,
        };
        let end = Event::TouchEnd { id: 1 };

        assert!(format!("{:?}", start).contains("TouchStart"));
        assert!(format!("{:?}", move_ev).contains("TouchMove"));
        assert!(format!("{:?}", end).contains("TouchEnd"));
    }

    #[test]
    fn test_event_clone() {
        let event = Event::Click {
            position: WidgetPoint::ZERO,
            button: WidgetMouseButton::Right,
        };

        let _cloned = event;
    }

    // ============================================================
    // WidgetMouseButton tests
    // ============================================================

    #[test]
    fn test_mouse_button() {
        assert_eq!(WidgetMouseButton::Left, WidgetMouseButton::Left);
        assert_ne!(WidgetMouseButton::Left, WidgetMouseButton::Right);
        assert_ne!(WidgetMouseButton::Right, WidgetMouseButton::Middle);
    }

    #[test]
    fn test_mouse_button_debug_and_clone() {
        let btn = WidgetMouseButton::Middle;
        let cloned = btn;
        assert!(format!("{:?}", cloned).contains("Middle"));
    }

    // ============================================================
    // Modifiers tests
    // ============================================================

    #[test]
    fn test_modifiers_default() {
        let m = Modifiers::default();
        assert!(!m.shift);
        assert!(!m.ctrl);
        assert!(!m.alt);
        assert!(!m.meta);
    }

    #[test]
    fn test_modifiers_debug_and_clone() {
        let m = Modifiers {
            shift: true,
            ctrl: true,
            alt: false,
            meta: true,
        };
        let cloned = m;
        assert!(format!("{:?}", cloned).contains("Modifiers"));
    }

    // ============================================================
    // RecordingCanvas tests
    // ============================================================

    #[test]
    fn test_recording_canvas_new() {
        let canvas = RecordingCanvas::new(Size::new(800.0, 600.0));
        assert_eq!(canvas.size(), Size::new(800.0, 600.0));
        assert!(canvas.commands().is_empty());
    }

    #[test]
    fn test_canvas_clear() {
        let mut canvas = RecordingCanvas::new(Size::new(100.0, 100.0));
        canvas.draw(DrawCommand::Clear {
            bounds: Rect::new(0.0, 0.0, 100.0, 100.0),
            color: WidgetColor::WHITE,
        });
        assert_eq!(canvas.commands().len(), 1);
        canvas.clear();
        assert_eq!(canvas.commands().len(), 0);
    }

    #[test]
    fn test_canvas_draw_multiple() {
        let mut canvas = RecordingCanvas::new(Size::new(100.0, 100.0));

        canvas.draw(DrawCommand::Rect {
            bounds: Rect::new(0.0, 0.0, 50.0, 50.0),
            color: WidgetColor::WHITE,
            radius: CornerRadius::ZERO,
        });
        canvas.draw(DrawCommand::Circle {
            center: WidgetPoint::new(75.0, 75.0),
            radius: 20.0,
            color: WidgetColor::BLACK,
        });

        assert_eq!(canvas.commands().len(), 2);
    }

    #[test]
    fn test_canvas_with_transform() {
        let canvas = RecordingCanvas::new(Size::new(100.0, 100.0));
        let mut transformed = canvas.with_transform(Transform2D::translate(10.0, 20.0));

        transformed.draw(DrawCommand::Rect {
            bounds: Rect::new(0.0, 0.0, 50.0, 50.0),
            color: WidgetColor::WHITE,
            radius: CornerRadius::ZERO,
        });

        assert_eq!(transformed.commands().len(), 1);
        // The command should be wrapped in a Group
        if let DrawCommand::Group { transform, .. } = &transformed.commands()[0] {
            assert_eq!(transform.matrix[4], 10.0);
            assert_eq!(transform.matrix[5], 20.0);
        } else {
            panic!("Expected Group command");
        }
    }

    #[test]
    fn test_transformed_canvas_with_nested_transform() {
        let canvas = RecordingCanvas::new(Size::new(100.0, 100.0));
        let transformed = canvas.with_transform(Transform2D::translate(10.0, 10.0));
        let nested = transformed.with_transform(Transform2D::translate(5.0, 5.0));

        assert_eq!(nested.size(), Size::new(100.0, 100.0));
    }

    #[test]
    fn test_transformed_canvas_clear() {
        let canvas = RecordingCanvas::new(Size::new(100.0, 100.0));
        let mut transformed = canvas.with_transform(Transform2D::identity());

        transformed.draw(DrawCommand::Rect {
            bounds: Rect::default(),
            color: WidgetColor::WHITE,
            radius: CornerRadius::ZERO,
        });
        transformed.clear();

        assert_eq!(transformed.commands().len(), 0);
    }

    #[test]
    fn test_recording_canvas_debug() {
        let canvas = RecordingCanvas::new(Size::new(100.0, 100.0));
        assert!(format!("{:?}", canvas).contains("RecordingCanvas"));
    }

    // ============================================================
    // Widget trait tests
    // ============================================================

    #[test]
    fn test_widget_render() {
        let widget = TestWidget::new("Hello");
        let mut canvas = RecordingCanvas::new(Size::new(800.0, 600.0));

        widget.render(&mut canvas);

        assert_eq!(canvas.commands().len(), 2);
    }

    #[test]
    fn test_widget_render_invalid() {
        let widget = TestWidget::new(""); // Empty text = invalid
        let mut canvas = RecordingCanvas::new(Size::new(800.0, 600.0));

        widget.render(&mut canvas);

        // Should not paint due to failed verification
        assert_eq!(canvas.commands().len(), 0);
    }

    #[test]
    fn test_widget_render_timed() {
        let widget = TestWidget::new("Hello");
        let mut canvas = RecordingCanvas::new(Size::new(800.0, 600.0));

        let metrics = widget.render_timed(&mut canvas);

        assert!(metrics.valid);
        assert!(metrics.total_time >= Duration::ZERO);
        assert_eq!(metrics.command_count, 2);
    }

    #[test]
    fn test_widget_render_timed_invalid() {
        let widget = TestWidget::new("");
        let mut canvas = RecordingCanvas::new(Size::new(800.0, 600.0));

        let metrics = widget.render_timed(&mut canvas);

        assert!(!metrics.valid);
        assert_eq!(metrics.command_count, 0);
        assert_eq!(metrics.paint_time, Duration::ZERO);
    }

    #[test]
    fn test_widget_render_full() {
        let mut widget = TestWidget::new("Hello");
        let mut canvas = RecordingCanvas::new(Size::new(800.0, 600.0));

        let result = widget.render_full(Rect::new(0.0, 0.0, 100.0, 50.0), &mut canvas);

        assert!(result.success);
        assert_eq!(canvas.commands().len(), 2);
    }

    #[test]
    fn test_widget_render_full_invalid() {
        let mut widget = TestWidget::new("");
        let mut canvas = RecordingCanvas::new(Size::new(800.0, 600.0));

        let result = widget.render_full(Rect::new(0.0, 0.0, 100.0, 50.0), &mut canvas);

        assert!(!result.success);
        assert_eq!(result.error, Some("Brick verification failed".to_string()));
    }

    #[test]
    fn test_widget_event() {
        let mut widget = TestWidget::new("Hello");

        let result = widget.event(&Event::Click {
            position: WidgetPoint::new(50.0, 25.0),
            button: WidgetMouseButton::Left,
        });

        assert!(result.is_some());
    }

    #[test]
    fn test_widget_event_unhandled() {
        let mut widget = TestWidget::new("Hello");

        let result = widget.event(&Event::Focus);
        assert!(result.is_none());

        let result = widget.event(&Event::Blur);
        assert!(result.is_none());

        let result = widget.event(&Event::Scroll {
            delta_x: 0.0,
            delta_y: 10.0,
        });
        assert!(result.is_none());
    }

    #[test]
    fn test_widget_measure() {
        let widget = TestWidget::new("Hello");
        let size = widget.measure(Constraints::unbounded());
        assert_eq!(size, Size::new(100.0, 50.0));
    }

    #[test]
    fn test_widget_measure_constrained() {
        let widget = TestWidget::new("Hello");
        let size = widget.measure(Constraints::tight(Size::new(50.0, 25.0)));
        assert_eq!(size, Size::new(50.0, 25.0));
    }

    #[test]
    fn test_widget_layout() {
        let mut widget = TestWidget::new("Hello");
        let result = widget.layout(Rect::new(10.0, 20.0, 100.0, 50.0));

        assert!(result.success);
        assert_eq!(result.bounds, Rect::new(10.0, 20.0, 100.0, 50.0));
    }

    #[test]
    fn test_widget_children_default() {
        let widget = TestWidget::new("Hello");
        assert!(widget.children().is_empty());
    }

    #[test]
    fn test_widget_children_mut_default() {
        let mut widget = TestWidget::new("Hello");
        assert!(widget.children_mut().is_empty());
    }

    // ============================================================
    // RenderMetrics tests
    // ============================================================

    #[test]
    fn test_render_metrics_budget() {
        let metrics = RenderMetrics {
            verify_time: Duration::from_millis(1),
            paint_time: Duration::from_millis(5),
            total_time: Duration::from_millis(6),
            valid: true,
            command_count: 10,
        };

        assert!(metrics.within_budget(BrickBudget::uniform(16)));
        assert!(!metrics.within_budget(BrickBudget::uniform(5)));
    }

    #[test]
    fn test_render_metrics_default() {
        let metrics = RenderMetrics::default();
        assert!(!metrics.valid);
        assert_eq!(metrics.command_count, 0);
        assert_eq!(metrics.total_time, Duration::ZERO);
    }

    #[test]
    fn test_render_metrics_debug_and_clone() {
        let metrics = RenderMetrics {
            verify_time: Duration::from_millis(1),
            paint_time: Duration::from_millis(2),
            total_time: Duration::from_millis(3),
            valid: true,
            command_count: 5,
        };

        let cloned = metrics;
        assert!(format!("{:?}", cloned).contains("RenderMetrics"));
    }
}
