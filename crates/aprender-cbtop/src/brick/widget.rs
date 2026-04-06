//! Widget trait and layout primitives for TUI rendering.

/// Widget trait - measure/layout/paint cycle
pub trait Widget {
    /// Measure desired size given constraints
    fn measure(&self, constraints: &Constraints) -> Size;

    /// Layout with allocated size
    fn layout(&mut self, size: Size);

    /// Paint to canvas
    fn paint(&self, canvas: &mut dyn Canvas);
}

/// Size in terminal cells
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Size {
    pub width: f32,
    pub height: f32,
}

impl Size {
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    pub const ZERO: Self = Self {
        width: 0.0,
        height: 0.0,
    };
}

/// Point in terminal cells
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
}

/// Rectangle in terminal cells
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn from_size(size: Size) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: size.width,
            height: size.height,
        }
    }

    pub fn size(&self) -> Size {
        Size::new(self.width, self.height)
    }

    pub fn top_left(&self) -> Point {
        Point::new(self.x, self.y)
    }
}

/// Layout constraints
#[derive(Debug, Clone, Copy)]
pub struct Constraints {
    pub min_width: f32,
    pub max_width: f32,
    pub min_height: f32,
    pub max_height: f32,
}

impl Constraints {
    pub const fn new(min_width: f32, max_width: f32, min_height: f32, max_height: f32) -> Self {
        Self {
            min_width,
            max_width,
            min_height,
            max_height,
        }
    }

    pub fn tight(size: Size) -> Self {
        Self {
            min_width: size.width,
            max_width: size.width,
            min_height: size.height,
            max_height: size.height,
        }
    }

    pub fn loose(size: Size) -> Self {
        Self {
            min_width: 0.0,
            max_width: size.width,
            min_height: 0.0,
            max_height: size.height,
        }
    }

    pub fn constrain(&self, size: Size) -> Size {
        Size {
            width: size.width.clamp(self.min_width, self.max_width),
            height: size.height.clamp(self.min_height, self.max_height),
        }
    }
}

impl Default for Constraints {
    fn default() -> Self {
        Self {
            min_width: 0.0,
            max_width: f32::INFINITY,
            min_height: 0.0,
            max_height: f32::INFINITY,
        }
    }
}

/// Color representation (24-bit RGB)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    // Standard colors
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    pub const RED: Self = Self::rgb(255, 0, 0);
    pub const GREEN: Self = Self::rgb(0, 255, 0);
    pub const BLUE: Self = Self::rgb(0, 0, 255);
    pub const YELLOW: Self = Self::rgb(255, 255, 0);
    pub const CYAN: Self = Self::rgb(0, 255, 255);
    pub const MAGENTA: Self = Self::rgb(255, 0, 255);
    pub const GRAY: Self = Self::rgb(128, 128, 128);
    pub const DARK_GRAY: Self = Self::rgb(64, 64, 64);
    pub const LIGHT_GRAY: Self = Self::rgb(192, 192, 192);

    // Andon colors (Toyota Way visual management)
    pub const ANDON_GREEN: Self = Self::rgb(0, 200, 0);
    pub const ANDON_YELLOW: Self = Self::rgb(255, 200, 0);
    pub const ANDON_RED: Self = Self::rgb(255, 50, 50);
}

/// Text style for rendering
#[derive(Debug, Clone, Copy, Default)]
pub struct TextStyle {
    pub color: Color,
    pub background: Option<Color>,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
}

impl TextStyle {
    pub const fn new() -> Self {
        Self {
            color: Color::WHITE,
            background: None,
            bold: false,
            italic: false,
            underline: false,
        }
    }

    pub const fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub const fn background(mut self, color: Color) -> Self {
        self.background = Some(color);
        self
    }

    pub const fn bold(mut self) -> Self {
        self.bold = true;
        self
    }
}

/// Canvas trait for rendering (presentar-style)
pub trait Canvas {
    /// Fill rectangle with solid color
    fn fill_rect(&mut self, rect: Rect, color: Color);

    /// Stroke rectangle outline
    fn stroke_rect(&mut self, rect: Rect, color: Color, width: f32);

    /// Draw text at position
    fn draw_text(&mut self, text: &str, pos: Point, style: &TextStyle);

    /// Draw line between points
    fn draw_line(&mut self, from: Point, to: Point, color: Color, width: f32);

    /// Fill circle
    fn fill_circle(&mut self, center: Point, radius: f32, color: Color);

    /// Stroke circle outline
    fn stroke_circle(&mut self, center: Point, radius: f32, color: Color, width: f32);

    /// Draw path (connected line segments)
    fn draw_path(&mut self, points: &[Point], color: Color, width: f32);

    /// Get canvas size
    fn size(&self) -> Size;

    /// Clear canvas with color
    fn clear(&mut self, color: Color);
}
