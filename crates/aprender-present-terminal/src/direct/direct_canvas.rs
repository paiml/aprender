//! Direct terminal canvas implementing the Canvas trait.
//!
//! This writes directly to a `CellBuffer`,
//! which is then rendered via the `DiffRenderer`.

use super::cell_buffer::{CellBuffer, Modifiers};
use crate::color::ColorMode;
use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Transform2D};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Direct terminal canvas that implements presentar's Canvas trait.
///
/// This canvas writes directly to a `CellBuffer`,
/// enabling zero-allocation steady-state rendering.
pub struct DirectTerminalCanvas<'a> {
    /// The cell buffer to write to.
    buffer: &'a mut CellBuffer,
    /// Clip region stack.
    clip_stack: Vec<ClipRect>,
    /// Transform stack.
    transform_stack: Vec<Transform2D>,
    /// Current accumulated transform.
    current_transform: Transform2D,
    /// Color mode for palette mapping.
    color_mode: ColorMode,
}

/// Simple clip rectangle.
#[derive(Clone, Copy, Debug)]
struct ClipRect {
    x: u16,
    y: u16,
    width: u16,
    height: u16,
}

impl ClipRect {
    const fn new(x: u16, y: u16, width: u16, height: u16) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    const fn contains(self, x: u16, y: u16) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    #[allow(clippy::cast_possible_wrap)]
    fn intersect(self, other: Self) -> Option<Self> {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        if x2 > x1 && y2 > y1 {
            Some(Self::new(x1, y1, x2 - x1, y2 - y1))
        } else {
            None
        }
    }

    const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

impl<'a> DirectTerminalCanvas<'a> {
    /// Create a new direct canvas.
    #[must_use]
    pub fn new(buffer: &'a mut CellBuffer) -> Self {
        let clip = ClipRect::new(0, 0, buffer.width(), buffer.height());
        Self {
            buffer,
            clip_stack: vec![clip],
            transform_stack: Vec::new(),
            current_transform: Transform2D::IDENTITY,
            color_mode: ColorMode::detect(),
        }
    }

    /// Create a canvas with a specific color mode.
    #[must_use]
    pub fn with_color_mode(mut self, mode: ColorMode) -> Self {
        self.color_mode = mode;
        self
    }

    /// Get the current color mode.
    #[must_use]
    pub const fn color_mode(&self) -> ColorMode {
        self.color_mode
    }

    /// Get the buffer width.
    #[must_use]
    pub fn width(&self) -> u16 {
        self.buffer.width()
    }

    /// Get the buffer height.
    #[must_use]
    pub fn height(&self) -> u16 {
        self.buffer.height()
    }

    /// Get the current clip region.
    fn clip(&self) -> ClipRect {
        self.clip_stack
            .last()
            .copied()
            .unwrap_or_else(|| ClipRect::new(0, 0, self.buffer.width(), self.buffer.height()))
    }

    /// Transform a point using the current transform.
    fn transform_point(&self, p: Point) -> Point {
        let m = &self.current_transform.matrix;
        Point::new(
            m[0] * p.x + m[2] * p.y + m[4],
            m[1] * p.x + m[3] * p.y + m[5],
        )
    }

    /// Convert a presentar Rect to terminal coordinates, applying transform and clipping.
    fn to_terminal_rect(&self, rect: Rect) -> Option<ClipRect> {
        let top_left = self.transform_point(Point::new(rect.x, rect.y));
        let bottom_right =
            self.transform_point(Point::new(rect.x + rect.width, rect.y + rect.height));

        let x = top_left.x.round() as i32;
        let y = top_left.y.round() as i32;
        let w = (bottom_right.x - top_left.x).round() as i32;
        let h = (bottom_right.y - top_left.y).round() as i32;

        if x < 0 || y < 0 || w <= 0 || h <= 0 {
            // Handle negative coordinates by adjusting
            let x = x.max(0) as u16;
            let y = y.max(0) as u16;
            let w = w.max(0) as u16;
            let h = h.max(0) as u16;

            if w == 0 || h == 0 {
                return None;
            }

            let rect = ClipRect::new(x, y, w, h);
            self.clip().intersect(rect)
        } else {
            let rect = ClipRect::new(x as u16, y as u16, w as u16, h as u16);
            self.clip().intersect(rect)
        }
    }

    /// Set a cell with clipping.
    fn set_cell(
        &mut self,
        x: u16,
        y: u16,
        symbol: &str,
        fg: Color,
        bg: Color,
        modifiers: Modifiers,
    ) {
        let clip = self.clip();
        if clip.contains(x, y) && x < self.buffer.width() && y < self.buffer.height() {
            self.buffer.update(x, y, symbol, fg, bg, modifiers);

            // Handle wide characters
            let width = UnicodeWidthStr::width(symbol);
            if width > 1 && x + 1 < self.buffer.width() {
                if let Some(cell) = self.buffer.get_mut(x + 1, y) {
                    cell.make_continuation();
                }
                self.buffer.mark_dirty(x + 1, y);
            }
        }
    }

    /// Convert text style to modifiers.
    fn style_to_modifiers(style: &TextStyle) -> Modifiers {
        let mut modifiers = Modifiers::NONE;
        if matches!(style.weight, presentar_core::FontWeight::Bold) {
            modifiers = modifiers.with(Modifiers::BOLD);
        }
        if matches!(style.style, presentar_core::FontStyle::Italic) {
            modifiers = modifiers.with(Modifiers::ITALIC);
        }
        modifiers
    }
}

impl Canvas for DirectTerminalCanvas<'_> {
    fn fill_rect(&mut self, rect: Rect, color: Color) {
        let Some(r) = self.to_terminal_rect(rect) else {
            return;
        };

        // Skip if clipped to empty
        if r.is_empty() {
            return;
        }

        for y in r.y..r.y + r.height {
            for x in r.x..r.x + r.width {
                self.set_cell(x, y, " ", color, color, Modifiers::NONE);
            }
        }
    }

    fn stroke_rect(&mut self, rect: Rect, color: Color, _width: f32) {
        let Some(r) = self.to_terminal_rect(rect) else {
            return;
        };

        // Top and bottom edges
        for x in r.x..r.x + r.width {
            self.set_cell(x, r.y, "─", color, Color::TRANSPARENT, Modifiers::NONE);
            if r.height > 1 {
                self.set_cell(
                    x,
                    r.y + r.height - 1,
                    "─",
                    color,
                    Color::TRANSPARENT,
                    Modifiers::NONE,
                );
            }
        }

        // Left and right edges
        for y in r.y..r.y + r.height {
            self.set_cell(r.x, y, "│", color, Color::TRANSPARENT, Modifiers::NONE);
            if r.width > 1 {
                self.set_cell(
                    r.x + r.width - 1,
                    y,
                    "│",
                    color,
                    Color::TRANSPARENT,
                    Modifiers::NONE,
                );
            }
        }

        // Corners
        self.set_cell(r.x, r.y, "┌", color, Color::TRANSPARENT, Modifiers::NONE);
        if r.width > 1 {
            self.set_cell(
                r.x + r.width - 1,
                r.y,
                "┐",
                color,
                Color::TRANSPARENT,
                Modifiers::NONE,
            );
        }
        if r.height > 1 {
            self.set_cell(
                r.x,
                r.y + r.height - 1,
                "└",
                color,
                Color::TRANSPARENT,
                Modifiers::NONE,
            );
            if r.width > 1 {
                self.set_cell(
                    r.x + r.width - 1,
                    r.y + r.height - 1,
                    "┘",
                    color,
                    Color::TRANSPARENT,
                    Modifiers::NONE,
                );
            }
        }
    }

    #[allow(clippy::cast_possible_wrap)]
    fn draw_text(&mut self, text: &str, position: Point, style: &TextStyle) {
        let p = self.transform_point(position);
        let mut x = p.x.round() as i32;
        let y = p.y.round() as i32;

        if y < 0 {
            return;
        }
        let y = y as u16;

        let clip = self.clip();
        if y < clip.y || y >= clip.y + clip.height {
            return;
        }

        let modifiers = Self::style_to_modifiers(style);
        let fg = style.color;

        // Render grapheme by grapheme
        for grapheme in text.graphemes(true) {
            if x < 0 {
                x += UnicodeWidthStr::width(grapheme) as i32;
                continue;
            }

            let xu = x as u16;
            if xu >= clip.x + clip.width {
                break;
            }

            if xu >= clip.x {
                // CRITICAL: Preserve existing background when drawing text.
                // Text rendering should NOT overwrite background colors set by fill_rect.
                // Without this, backgrounds would be lost during text rendering.
                let existing_bg = self
                    .buffer
                    .get(xu, y)
                    .map(|c| c.bg)
                    .unwrap_or(Color::TRANSPARENT);
                self.set_cell(xu, y, grapheme, fg, existing_bg, modifiers);
            }

            x += UnicodeWidthStr::width(grapheme) as i32;
        }
    }

    fn draw_line(&mut self, from: Point, to: Point, color: Color, _width: f32) {
        let from = self.transform_point(from);
        let to = self.transform_point(to);

        // Bresenham's line algorithm
        let x0 = from.x.round() as i32;
        let y0 = from.y.round() as i32;
        let x1 = to.x.round() as i32;
        let y1 = to.y.round() as i32;

        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        let mut x = x0;
        let mut y = y0;

        loop {
            if x >= 0 && y >= 0 {
                let ch = if dx > (-dy) * 2 {
                    "─"
                } else if (-dy) > dx * 2 {
                    "│"
                } else if (sx > 0) == (sy > 0) {
                    "╲"
                } else {
                    "╱"
                };
                self.set_cell(
                    x as u16,
                    y as u16,
                    ch,
                    color,
                    Color::TRANSPARENT,
                    Modifiers::NONE,
                );
            }

            if x == x1 && y == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                if x == x1 {
                    break;
                }
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                if y == y1 {
                    break;
                }
                err += dx;
                y += sy;
            }
        }
    }

    fn fill_circle(&mut self, center: Point, radius: f32, color: Color) {
        let c = self.transform_point(center);
        let r = radius.round() as i32;
        let cx = c.x.round() as i32;
        let cy = c.y.round() as i32;

        // Midpoint circle algorithm with fill
        for y in (cy - r)..=(cy + r) {
            let dy = (y - cy).abs();
            let dx = ((r * r - dy * dy) as f32).sqrt() as i32;
            for x in (cx - dx)..=(cx + dx) {
                if x >= 0 && y >= 0 {
                    self.set_cell(x as u16, y as u16, " ", color, color, Modifiers::NONE);
                }
            }
        }
    }

    fn stroke_circle(&mut self, center: Point, radius: f32, color: Color, _width: f32) {
        let c = self.transform_point(center);
        let r = radius.round() as i32;
        let cx = c.x.round() as i32;
        let cy = c.y.round() as i32;

        // Midpoint circle algorithm
        let mut x = r;
        let mut y = 0;
        let mut err = 0;

        while x >= y {
            let points = [
                (cx + x, cy + y),
                (cx + y, cy + x),
                (cx - y, cy + x),
                (cx - x, cy + y),
                (cx - x, cy - y),
                (cx - y, cy - x),
                (cx + y, cy - x),
                (cx + x, cy - y),
            ];

            for (px, py) in points {
                if px >= 0 && py >= 0 {
                    self.set_cell(
                        px as u16,
                        py as u16,
                        "●",
                        color,
                        Color::TRANSPARENT,
                        Modifiers::NONE,
                    );
                }
            }

            y += 1;
            err += 1 + 2 * y;
            if 2 * (err - x) + 1 > 0 {
                x -= 1;
                err += 1 - 2 * x;
            }
        }
    }

    fn fill_arc(
        &mut self,
        center: Point,
        radius: f32,
        start_angle: f32,
        end_angle: f32,
        color: Color,
    ) {
        let c = self.transform_point(center);
        let cx = c.x.round() as i32;
        let cy = c.y.round() as i32;

        let steps = (radius * 4.0) as i32;
        if steps <= 0 {
            return;
        }

        let angle_step = (end_angle - start_angle) / steps as f32;

        for i in 0..=steps {
            let angle = start_angle + i as f32 * angle_step;
            let x = cx + (radius * angle.cos()).round() as i32;
            let y = cy + (radius * angle.sin()).round() as i32;
            if x >= 0 && y >= 0 {
                self.set_cell(x as u16, y as u16, " ", color, color, Modifiers::NONE);
            }
        }
    }

    fn draw_path(&mut self, points: &[Point], color: Color, width: f32) {
        if points.len() < 2 {
            return;
        }

        for window in points.windows(2) {
            self.draw_line(window[0], window[1], color, width);
        }
    }

    fn fill_polygon(&mut self, points: &[Point], color: Color) {
        if points.len() < 3 {
            return;
        }

        // Transform points
        let transformed: Vec<Point> = points.iter().map(|p| self.transform_point(*p)).collect();

        // Find bounding box
        let min_y = transformed
            .iter()
            .map(|p| p.y.round() as i32)
            .min()
            .unwrap_or(0);
        let max_y = transformed
            .iter()
            .map(|p| p.y.round() as i32)
            .max()
            .unwrap_or(0);

        // Scanline fill
        for y in min_y..=max_y {
            let mut intersections: Vec<i32> = Vec::new();

            for i in 0..transformed.len() {
                let p1 = transformed[i];
                let p2 = transformed[(i + 1) % transformed.len()];

                let y1 = p1.y.round() as i32;
                let y2 = p2.y.round() as i32;

                if (y1 <= y && y < y2) || (y2 <= y && y < y1) {
                    let t = (y as f32 - p1.y) / (p2.y - p1.y);
                    let x = (p1.x + t * (p2.x - p1.x)).round() as i32;
                    intersections.push(x);
                }
            }

            intersections.sort_unstable();

            for chunk in intersections.chunks(2) {
                if chunk.len() == 2 {
                    for x in chunk[0]..=chunk[1] {
                        if x >= 0 && y >= 0 {
                            self.set_cell(x as u16, y as u16, " ", color, color, Modifiers::NONE);
                        }
                    }
                }
            }
        }
    }

    fn push_clip(&mut self, rect: Rect) {
        if let Some(r) = self.to_terminal_rect(rect) {
            if let Some(clipped) = self.clip().intersect(r) {
                self.clip_stack.push(clipped);
            } else {
                // Empty clip
                self.clip_stack.push(ClipRect::new(0, 0, 0, 0));
            }
        } else {
            // Empty clip
            self.clip_stack.push(ClipRect::new(0, 0, 0, 0));
        }
    }

    fn pop_clip(&mut self) {
        if self.clip_stack.len() > 1 {
            self.clip_stack.pop();
        }
    }

    fn push_transform(&mut self, transform: Transform2D) {
        self.transform_stack.push(self.current_transform);

        // Multiply transforms
        let a = &self.current_transform.matrix;
        let b = &transform.matrix;
        self.current_transform = Transform2D {
            matrix: [
                a[0] * b[0] + a[2] * b[1],
                a[1] * b[0] + a[3] * b[1],
                a[0] * b[2] + a[2] * b[3],
                a[1] * b[2] + a[3] * b[3],
                a[0] * b[4] + a[2] * b[5] + a[4],
                a[1] * b[4] + a[3] * b[5] + a[5],
            ],
        };
    }

    fn pop_transform(&mut self) {
        if let Some(t) = self.transform_stack.pop() {
            self.current_transform = t;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use presentar_core::{FontStyle, FontWeight};

    fn create_canvas(width: u16, height: u16) -> CellBuffer {
        CellBuffer::new(width, height)
    }

    #[test]
    fn test_canvas_creation() {
        let mut buffer = create_canvas(80, 24);
        let canvas = DirectTerminalCanvas::new(&mut buffer);
        assert_eq!(canvas.width(), 80);
        assert_eq!(canvas.height(), 24);
    }

    #[test]
    fn test_canvas_with_color_mode() {
        let mut buffer = create_canvas(80, 24);
        let canvas = DirectTerminalCanvas::new(&mut buffer).with_color_mode(ColorMode::Color256);
        assert_eq!(canvas.color_mode(), ColorMode::Color256);
    }

    #[test]
    fn test_fill_rect() {
        let mut buffer = create_canvas(20, 10);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.fill_rect(Rect::new(1.0, 1.0, 3.0, 3.0), Color::RED);
        }

        let cell = buffer.get(2, 2).unwrap();
        assert_eq!(cell.bg, Color::RED);
    }

    #[test]
    fn test_fill_rect_outside_bounds() {
        let mut buffer = create_canvas(10, 10);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.fill_rect(Rect::new(100.0, 100.0, 3.0, 3.0), Color::RED);
        }
        // Should not panic
    }

    #[test]
    fn test_stroke_rect() {
        let mut buffer = create_canvas(20, 10);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.stroke_rect(Rect::new(1.0, 1.0, 5.0, 5.0), Color::GREEN, 1.0);
        }

        assert_eq!(buffer.get(1, 1).unwrap().symbol.as_str(), "┌");
        assert_eq!(buffer.get(5, 1).unwrap().symbol.as_str(), "┐");
        assert_eq!(buffer.get(1, 5).unwrap().symbol.as_str(), "└");
        assert_eq!(buffer.get(5, 5).unwrap().symbol.as_str(), "┘");
    }

    #[test]
    fn test_stroke_rect_single_cell() {
        let mut buffer = create_canvas(10, 10);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.stroke_rect(Rect::new(1.0, 1.0, 1.0, 1.0), Color::GREEN, 1.0);
        }
    }

    #[test]
    fn test_draw_text() {
        let mut buffer = create_canvas(20, 5);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.draw_text("Hello", Point::new(0.0, 0.0), &TextStyle::default());
        }

        assert_eq!(buffer.get(0, 0).unwrap().symbol.as_str(), "H");
        assert_eq!(buffer.get(1, 0).unwrap().symbol.as_str(), "e");
    }

    #[test]
    fn test_draw_text_bold_italic() {
        let mut buffer = create_canvas(20, 5);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            let style = TextStyle {
                weight: FontWeight::Bold,
                style: FontStyle::Italic,
                ..Default::default()
            };
            canvas.draw_text("Hi", Point::new(0.0, 0.0), &style);
        }

        let cell = buffer.get(0, 0).unwrap();
        assert!(cell.modifiers.contains(Modifiers::BOLD));
        assert!(cell.modifiers.contains(Modifiers::ITALIC));
    }

    #[test]
    fn test_draw_text_clipped_y() {
        let mut buffer = create_canvas(20, 5);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.draw_text("Hello", Point::new(0.0, 10.0), &TextStyle::default());
        }
        // Should not render
    }

    #[test]
    fn test_draw_text_negative_y() {
        let mut buffer = create_canvas(20, 5);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.draw_text("Hello", Point::new(0.0, -5.0), &TextStyle::default());
        }
    }

    #[test]
    fn test_draw_text_partial_clip() {
        let mut buffer = create_canvas(5, 5);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.draw_text("Hello World", Point::new(0.0, 0.0), &TextStyle::default());
        }
        // Should clip at width
        assert_eq!(buffer.get(4, 0).unwrap().symbol.as_str(), "o");
    }

    #[test]
    fn test_draw_text_negative_x() {
        let mut buffer = create_canvas(10, 5);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.draw_text("Hello", Point::new(-2.0, 0.0), &TextStyle::default());
        }
        // First visible should be 'l'
        assert_eq!(buffer.get(0, 0).unwrap().symbol.as_str(), "l");
    }

    #[test]
    fn test_draw_line_horizontal() {
        let mut buffer = create_canvas(20, 10);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.draw_line(
                Point::new(0.0, 5.0),
                Point::new(10.0, 5.0),
                Color::WHITE,
                1.0,
            );
        }
    }

    #[test]
    fn test_draw_line_vertical() {
        let mut buffer = create_canvas(20, 20);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.draw_line(
                Point::new(5.0, 0.0),
                Point::new(5.0, 10.0),
                Color::WHITE,
                1.0,
            );
        }
    }

    #[test]
    fn test_draw_line_diagonal() {
        let mut buffer = create_canvas(20, 20);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.draw_line(
                Point::new(0.0, 0.0),
                Point::new(10.0, 10.0),
                Color::WHITE,
                1.0,
            );
        }
    }

    #[test]
    fn test_draw_line_same_point() {
        let mut buffer = create_canvas(20, 20);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.draw_line(
                Point::new(5.0, 5.0),
                Point::new(5.0, 5.0),
                Color::WHITE,
                1.0,
            );
        }
    }

    #[test]
    fn test_fill_circle() {
        let mut buffer = create_canvas(20, 20);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.fill_circle(Point::new(10.0, 10.0), 5.0, Color::BLUE);
        }
        // Center should be filled
        assert_eq!(buffer.get(10, 10).unwrap().bg, Color::BLUE);
    }

    #[test]
    fn test_stroke_circle() {
        let mut buffer = create_canvas(20, 20);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.stroke_circle(Point::new(10.0, 10.0), 5.0, Color::GREEN, 1.0);
        }
    }

    #[test]
    fn test_fill_arc() {
        let mut buffer = create_canvas(20, 20);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.fill_arc(
                Point::new(10.0, 10.0),
                5.0,
                0.0,
                std::f32::consts::PI,
                Color::RED,
            );
        }
    }

    #[test]
    fn test_fill_arc_zero_radius() {
        let mut buffer = create_canvas(20, 20);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.fill_arc(
                Point::new(10.0, 10.0),
                0.0,
                0.0,
                std::f32::consts::PI,
                Color::RED,
            );
        }
    }

    #[test]
    fn test_draw_path() {
        let mut buffer = create_canvas(20, 20);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            let points = [
                Point::new(0.0, 0.0),
                Point::new(5.0, 5.0),
                Point::new(10.0, 0.0),
            ];
            canvas.draw_path(&points, Color::WHITE, 1.0);
        }
    }

    #[test]
    fn test_draw_path_empty() {
        let mut buffer = create_canvas(20, 20);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.draw_path(&[], Color::WHITE, 1.0);
        }
    }

    #[test]
    fn test_draw_path_single_point() {
        let mut buffer = create_canvas(20, 20);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.draw_path(&[Point::new(5.0, 5.0)], Color::WHITE, 1.0);
        }
    }

    #[test]
    fn test_fill_polygon() {
        let mut buffer = create_canvas(20, 20);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            let points = [
                Point::new(5.0, 0.0),
                Point::new(10.0, 10.0),
                Point::new(0.0, 10.0),
            ];
            canvas.fill_polygon(&points, Color::BLUE);
        }
    }

    #[test]
    fn test_fill_polygon_insufficient_points() {
        let mut buffer = create_canvas(20, 20);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.fill_polygon(&[Point::new(5.0, 5.0)], Color::BLUE);
            canvas.fill_polygon(&[Point::new(5.0, 5.0), Point::new(10.0, 10.0)], Color::BLUE);
        }
    }

    #[test]
    fn test_push_pop_clip() {
        let mut buffer = create_canvas(20, 10);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.push_clip(Rect::new(5.0, 5.0, 10.0, 5.0));
            canvas.fill_rect(Rect::new(0.0, 0.0, 20.0, 10.0), Color::RED);
            canvas.pop_clip();
        }

        // Outside clip should be unchanged
        assert_eq!(buffer.get(0, 0).unwrap().bg, Color::TRANSPARENT);
        // Inside clip should be filled
        assert_eq!(buffer.get(7, 7).unwrap().bg, Color::RED);
    }

    #[test]
    fn test_push_clip_empty() {
        let mut buffer = create_canvas(20, 10);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.push_clip(Rect::new(100.0, 100.0, 10.0, 10.0));
            canvas.fill_rect(Rect::new(0.0, 0.0, 20.0, 10.0), Color::RED);
            canvas.pop_clip();
        }
        // Nothing should be filled
        assert_eq!(buffer.get(0, 0).unwrap().bg, Color::TRANSPARENT);
    }

    #[test]
    fn test_pop_clip_at_root() {
        let mut buffer = create_canvas(20, 10);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.pop_clip();
            canvas.pop_clip();
        }
    }

    #[test]
    fn test_push_pop_transform() {
        let mut buffer = create_canvas(20, 10);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.push_transform(Transform2D::translate(5.0, 5.0));
            canvas.fill_rect(Rect::new(0.0, 0.0, 2.0, 2.0), Color::BLUE);
            canvas.pop_transform();
        }

        assert_eq!(buffer.get(5, 5).unwrap().bg, Color::BLUE);
        assert_eq!(buffer.get(0, 0).unwrap().bg, Color::TRANSPARENT);
    }

    #[test]
    fn test_transform_stack() {
        let mut buffer = create_canvas(20, 20);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.push_transform(Transform2D::translate(5.0, 5.0));
            canvas.push_transform(Transform2D::translate(2.0, 2.0));
            canvas.fill_rect(Rect::new(0.0, 0.0, 2.0, 2.0), Color::GREEN);
            canvas.pop_transform();
            canvas.pop_transform();
        }

        assert_eq!(buffer.get(7, 7).unwrap().bg, Color::GREEN);
    }

    #[test]
    fn test_pop_transform_empty() {
        let mut buffer = create_canvas(20, 10);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.pop_transform();
        }
    }

    #[test]
    fn test_wide_character() {
        let mut buffer = create_canvas(20, 5);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.draw_text("日本", Point::new(0.0, 0.0), &TextStyle::default());
        }

        assert_eq!(buffer.get(0, 0).unwrap().symbol.as_str(), "日");
        assert!(buffer.get(1, 0).unwrap().is_continuation());
        assert_eq!(buffer.get(2, 0).unwrap().symbol.as_str(), "本");
    }

    #[test]
    fn test_clip_rect_methods() {
        let r1 = ClipRect::new(0, 0, 10, 10);
        let r2 = ClipRect::new(5, 5, 10, 10);

        assert!(r1.contains(5, 5));
        assert!(!r1.contains(10, 10));

        let intersect = r1.intersect(r2).unwrap();
        assert_eq!(intersect.x, 5);
        assert_eq!(intersect.y, 5);
        assert_eq!(intersect.width, 5);
        assert_eq!(intersect.height, 5);
    }

    #[test]
    fn test_clip_rect_no_intersect() {
        let r1 = ClipRect::new(0, 0, 5, 5);
        let r2 = ClipRect::new(10, 10, 5, 5);

        assert!(r1.intersect(r2).is_none());
    }

    #[test]
    fn test_clip_rect_empty() {
        let r = ClipRect::new(0, 0, 0, 0);
        assert!(r.is_empty());
    }

    #[test]
    fn test_negative_rect() {
        let mut buffer = create_canvas(20, 10);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.fill_rect(Rect::new(-5.0, -5.0, 10.0, 10.0), Color::RED);
        }
        // Should still fill visible portion
        assert_eq!(buffer.get(0, 0).unwrap().bg, Color::RED);
    }

    #[test]
    fn test_to_terminal_rect_zero_size() {
        let mut buffer = create_canvas(20, 10);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.fill_rect(Rect::new(5.0, 5.0, 0.0, 0.0), Color::RED);
        }
    }

    // =========================================================================
    // REGRESSION TESTS
    // =========================================================================

    /// Regression test for selection artifact bug.
    ///
    /// Bug: When a row was selected (blue background), then deselected,
    /// the blue background persisted because draw_text was overwriting
    /// the fill_rect background with TRANSPARENT.
    ///
    /// Fix: draw_text now preserves the existing background color instead
    /// of setting it to TRANSPARENT.
    #[test]
    fn test_draw_text_preserves_fill_rect_background() {
        let mut buffer = create_canvas(20, 5);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);

            // Step 1: Fill row with selection background (simulates selected row)
            canvas.fill_rect(Rect::new(0.0, 1.0, 20.0, 1.0), Color::BLUE);

            // Step 2: Draw text on top (simulates rendering row content)
            canvas.draw_text("Process 1234", Point::new(0.0, 1.0), &TextStyle::default());

            // CRITICAL: Text characters MUST preserve the BLUE background
            // This was the bug - draw_text was setting bg to TRANSPARENT
            let cell = buffer.get(0, 1).unwrap();
            assert_eq!(
                cell.bg,
                Color::BLUE,
                "Text draw MUST preserve fill_rect background, got {:?}",
                cell.bg
            );
        }
    }

    /// Regression test: Selection change clears old selection properly.
    ///
    /// Simulates the full selection lifecycle:
    /// 1. Row 1 selected (highlighted bg)
    /// 2. Selection moves to row 2
    /// 3. Row 1 should now have dimmed bg (not highlighted!)
    #[test]
    fn test_selection_change_clears_old_background() {
        let dimmed_bg = Color::new(0.08, 0.08, 0.1, 1.0);
        let selection_bg = Color::new(0.15, 0.12, 0.22, 1.0); // ttop-style subtle selection

        let mut buffer = create_canvas(20, 5);

        // Frame 1: Row 1 selected
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.fill_rect(Rect::new(0.0, 1.0, 20.0, 1.0), selection_bg);
            canvas.draw_text("Selected Row", Point::new(0.0, 1.0), &TextStyle::default());

            canvas.fill_rect(Rect::new(0.0, 2.0, 20.0, 1.0), dimmed_bg);
            canvas.draw_text("Normal Row", Point::new(0.0, 2.0), &TextStyle::default());
        }

        // Verify frame 1
        assert_eq!(buffer.get(0, 1).unwrap().bg, selection_bg);
        assert_eq!(buffer.get(0, 2).unwrap().bg, dimmed_bg);

        // Frame 2: Selection moves to row 2
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            // Row 1 now dimmed
            canvas.fill_rect(Rect::new(0.0, 1.0, 20.0, 1.0), dimmed_bg);
            canvas.draw_text("Normal Row", Point::new(0.0, 1.0), &TextStyle::default());

            // Row 2 now selected
            canvas.fill_rect(Rect::new(0.0, 2.0, 20.0, 1.0), selection_bg);
            canvas.draw_text("Selected Row", Point::new(0.0, 2.0), &TextStyle::default());
        }

        // CRITICAL: Row 1 must now have dimmed background, NOT selection_bg
        assert_eq!(
            buffer.get(0, 1).unwrap().bg,
            dimmed_bg,
            "Old selection must be cleared to dimmed_bg"
        );
        assert_eq!(
            buffer.get(0, 2).unwrap().bg,
            selection_bg,
            "New selection must have selection_bg"
        );
    }

    /// Test that text color is set correctly while preserving background.
    #[test]
    fn test_draw_text_sets_foreground_color() {
        let mut buffer = create_canvas(20, 5);
        {
            let mut canvas = DirectTerminalCanvas::new(&mut buffer);
            canvas.fill_rect(Rect::new(0.0, 0.0, 20.0, 1.0), Color::BLACK);

            let style = TextStyle {
                color: Color::YELLOW,
                ..Default::default()
            };
            canvas.draw_text("Test", Point::new(0.0, 0.0), &style);
        }

        let cell = buffer.get(0, 0).unwrap();
        assert_eq!(cell.fg, Color::YELLOW);
        assert_eq!(cell.bg, Color::BLACK);
    }
}
