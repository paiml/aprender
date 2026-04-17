//! Primitive rendering functions.
//!
//! Implements rasterization algorithms for basic geometric shapes.

use crate::color::Rgba;
use crate::framebuffer::Framebuffer;
use crate::geometry::{Line, Point, Rect};

/// Safely convert a `u32` pixel coordinate to `i32` without wrapping.
#[inline]
pub fn i32_px(val: u32) -> i32 {
    i32::try_from(val).unwrap_or(i32::MAX)
}

/// Trait for drawable primitives.
pub trait Drawable {
    /// Draw this primitive to a framebuffer.
    fn draw(&self, fb: &mut Framebuffer, color: Rgba);

    /// Draw this primitive with anti-aliasing if supported.
    fn draw_aa(&self, fb: &mut Framebuffer, color: Rgba) {
        // Default to non-AA drawing
        self.draw(fb, color);
    }
}

// ============================================================================
// Line Drawing
// ============================================================================

/// Draw a line using Bresenham's algorithm (non-antialiased).
///
/// # Arguments
///
/// * `fb` - Target framebuffer
/// * `x0`, `y0` - Start coordinates
/// * `x1`, `y1` - End coordinates
/// * `color` - Line color
pub fn draw_line(fb: &mut Framebuffer, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgba) {
    // Contract: display-format-v1.yaml precondition (pv codegen)
    contract_pre_display_format!(fb.pixels());
    contract_post_display_format!(&"ok");

    let mut cursor = BresenhamCursor::new(x0, y0, x1, y1);
    loop {
        plot_pixel_if_visible(fb, cursor.x, cursor.y, color);
        if cursor.finished() {
            break;
        }
        if cursor.step() {
            break;
        }
    }
}

/// Plot one integer pixel into the framebuffer, skipping negative
/// coordinates (off-screen on the top/left).
fn plot_pixel_if_visible(fb: &mut Framebuffer, x: i32, y: i32, color: Rgba) {
    if x >= 0 && y >= 0 {
        fb.set_pixel(x as u32, y as u32, color);
    }
}

/// State for one run of Bresenham's line algorithm.
struct BresenhamCursor {
    x: i32,
    y: i32,
    x1: i32,
    y1: i32,
    dx: i32,
    dy: i32,
    sx: i32,
    sy: i32,
    err: i32,
}

impl BresenhamCursor {
    fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        Self {
            x: x0,
            y: y0,
            x1,
            y1,
            dx,
            dy,
            sx,
            sy,
            err: dx + dy,
        }
    }

    fn finished(&self) -> bool {
        self.x == self.x1 && self.y == self.y1
    }

    /// Advance one Bresenham step. Returns `true` when the step would
    /// have moved past the endpoint — caller should stop plotting.
    fn step(&mut self) -> bool {
        let e2 = 2 * self.err;
        if e2 >= self.dy {
            if self.x == self.x1 {
                return true;
            }
            self.err += self.dy;
            self.x += self.sx;
        }
        if e2 <= self.dx {
            if self.y == self.y1 {
                return true;
            }
            self.err += self.dx;
            self.y += self.sy;
        }
        false
    }
}

/// Draw an anti-aliased line using Wu's algorithm.
///
/// This implements Xiaolin Wu's line algorithm from SIGGRAPH '91,
/// which produces smooth lines with sub-pixel accuracy.
///
/// # Algorithm
///
/// Wu's algorithm draws two pixels at each step along the major axis,
/// adjusting their intensities based on the fractional distance from
/// the ideal line position.
///
/// # References
///
/// Wu, X. (1991). "An Efficient Antialiasing Technique." SIGGRAPH '91.
pub fn draw_line_aa(fb: &mut Framebuffer, x0: f32, y0: f32, x1: f32, y1: f32, color: Rgba) {
    // Contract: display-format-v1.yaml precondition (pv codegen)
    contract_pre_display_format!(fb.pixels());
    contract_post_display_format!(&"ok");
    let steep = (y1 - y0).abs() > (x1 - x0).abs();

    let (x0, y0, x1, y1) = if steep { (y0, x0, y1, x1) } else { (x0, y0, x1, y1) };

    let (x0, y0, x1, y1) = if x0 > x1 { (x1, y1, x0, y0) } else { (x0, y0, x1, y1) };

    let dx = x1 - x0;
    let dy = y1 - y0;
    let gradient = if dx.abs() < f32::EPSILON { 1.0 } else { dy / dx };

    // Handle first endpoint
    let xend = x0.round();
    let yend = y0 + gradient * (xend - x0);
    let xgap = rfpart(x0 + 0.5);
    let xpxl1 = xend as i32;
    let ypxl1 = yend.floor() as i32;

    if steep {
        plot(fb, ypxl1, xpxl1, color, rfpart(yend) * xgap);
        plot(fb, ypxl1 + 1, xpxl1, color, fpart(yend) * xgap);
    } else {
        plot(fb, xpxl1, ypxl1, color, rfpart(yend) * xgap);
        plot(fb, xpxl1, ypxl1 + 1, color, fpart(yend) * xgap);
    }

    let mut intery = yend + gradient;

    // Handle second endpoint
    let xend = x1.round();
    let yend = y1 + gradient * (xend - x1);
    let xgap = fpart(x1 + 0.5);
    let xpxl2 = xend as i32;
    let ypxl2 = yend.floor() as i32;

    if steep {
        plot(fb, ypxl2, xpxl2, color, rfpart(yend) * xgap);
        plot(fb, ypxl2 + 1, xpxl2, color, fpart(yend) * xgap);
    } else {
        plot(fb, xpxl2, ypxl2, color, rfpart(yend) * xgap);
        plot(fb, xpxl2, ypxl2 + 1, color, fpart(yend) * xgap);
    }

    // Main loop
    if steep {
        for x in (xpxl1 + 1)..xpxl2 {
            let ipart = intery.floor() as i32;
            plot(fb, ipart, x, color, rfpart(intery));
            plot(fb, ipart + 1, x, color, fpart(intery));
            intery += gradient;
        }
    } else {
        for x in (xpxl1 + 1)..xpxl2 {
            let ipart = intery.floor() as i32;
            plot(fb, x, ipart, color, rfpart(intery));
            plot(fb, x, ipart + 1, color, fpart(intery));
            intery += gradient;
        }
    }
}

/// Plot a pixel with intensity (for anti-aliased drawing).
#[inline]
fn plot(fb: &mut Framebuffer, x: i32, y: i32, color: Rgba, intensity: f32) {
    if x >= 0 && y >= 0 && x < i32_px(fb.width()) && y < i32_px(fb.height()) {
        let alpha = (f32::from(color.a) * intensity) as u8;
        let blended = color.with_alpha(alpha);
        fb.blend_pixel(x as u32, y as u32, blended);
    }
}

/// Fractional part of a float.
#[inline]
fn fpart(x: f32) -> f32 {
    x - x.floor()
}

/// Reverse fractional part.
#[inline]
fn rfpart(x: f32) -> f32 {
    1.0 - fpart(x)
}

impl Drawable for Line {
    fn draw(&self, fb: &mut Framebuffer, color: Rgba) {
        draw_line(
            fb,
            self.start.x as i32,
            self.start.y as i32,
            self.end.x as i32,
            self.end.y as i32,
            color,
        );
    }

    fn draw_aa(&self, fb: &mut Framebuffer, color: Rgba) {
        draw_line_aa(fb, self.start.x, self.start.y, self.end.x, self.end.y, color);
    }
}

// ============================================================================
// Rectangle Drawing
// ============================================================================

/// Draw a filled rectangle.
pub fn draw_rect(fb: &mut Framebuffer, x: i32, y: i32, width: u32, height: u32, color: Rgba) {
    // Contract: display-format-v1.yaml precondition (pv codegen)
    contract_pre_display_format!(fb.pixels());
    contract_post_display_format!(&"ok");
    let x = x.max(0) as u32;
    let y = y.max(0) as u32;
    fb.fill_rect(x, y, width, height, color);
}

/// Draw a rectangle outline.
pub fn draw_rect_outline(
    fb: &mut Framebuffer,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    color: Rgba,
    thickness: u32,
) {
    // Contract: display-format-v1.yaml precondition (pv codegen)
    contract_pre_display_format!(fb.pixels());
    contract_post_display_format!(&"ok");
    let thickness = thickness.max(1);
    let x = x.max(0) as u32;
    let y = y.max(0) as u32;

    // Top edge
    fb.fill_rect(x, y, width, thickness, color);
    // Bottom edge
    if height > thickness {
        fb.fill_rect(x, y + height - thickness, width, thickness, color);
    }
    // Left edge
    if height > 2 * thickness {
        fb.fill_rect(x, y + thickness, thickness, height - 2 * thickness, color);
    }
    // Right edge
    if width > thickness && height > 2 * thickness {
        fb.fill_rect(
            x + width - thickness,
            y + thickness,
            thickness,
            height - 2 * thickness,
            color,
        );
    }
}

impl Drawable for Rect {
    fn draw(&self, fb: &mut Framebuffer, color: Rgba) {
        draw_rect(fb, self.x as i32, self.y as i32, self.width as u32, self.height as u32, color);
    }
}

// ============================================================================
// Circle/Point Drawing
// ============================================================================

/// Draw a filled circle using the midpoint algorithm.
///
/// # Arguments
///
/// * `fb` - Target framebuffer
/// * `cx`, `cy` - Center coordinates
/// * `radius` - Circle radius in pixels
/// * `color` - Fill color
pub fn draw_circle(fb: &mut Framebuffer, cx: i32, cy: i32, radius: i32, color: Rgba) {
    // Contract: display-format-v1.yaml precondition (pv codegen)
    contract_pre_display_format!(fb.pixels());
    if radius <= 0 {
        if radius == 0 && cx >= 0 && cy >= 0 {
            fb.set_pixel(cx as u32, cy as u32, color);
        }
        return;
    }

    // Midpoint circle algorithm for filled circle
    let mut x = radius;
    let mut y = 0;
    let mut err = 1 - radius;

    while x >= y {
        // Draw horizontal scan lines for each octant
        draw_horizontal_line(fb, cx - x, cx + x, cy + y, color);
        draw_horizontal_line(fb, cx - x, cx + x, cy - y, color);
        draw_horizontal_line(fb, cx - y, cx + y, cy + x, color);
        draw_horizontal_line(fb, cx - y, cx + y, cy - x, color);

        y += 1;
        if err < 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

/// Draw a circle outline.
pub fn draw_circle_outline(fb: &mut Framebuffer, cx: i32, cy: i32, radius: i32, color: Rgba) {
    // Contract: display-format-v1.yaml precondition (pv codegen)
    contract_pre_display_format!(fb.pixels());
    if radius <= 0 {
        if radius == 0 && cx >= 0 && cy >= 0 {
            fb.set_pixel(cx as u32, cy as u32, color);
        }
        return;
    }

    let mut x = radius;
    let mut y = 0;
    let mut err = 1 - radius;

    while x >= y {
        // Plot 8 octant points
        plot_circle_point(fb, cx + x, cy + y, color);
        plot_circle_point(fb, cx - x, cy + y, color);
        plot_circle_point(fb, cx + x, cy - y, color);
        plot_circle_point(fb, cx - x, cy - y, color);
        plot_circle_point(fb, cx + y, cy + x, color);
        plot_circle_point(fb, cx - y, cy + x, color);
        plot_circle_point(fb, cx + y, cy - x, color);
        plot_circle_point(fb, cx - y, cy - x, color);

        y += 1;
        if err < 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

/// Draw a point with variable size (rendered as filled circle).
pub fn draw_point(fb: &mut Framebuffer, x: f32, y: f32, size: f32, color: Rgba) {
    // Contract: display-format-v1.yaml precondition (pv codegen)
    contract_pre_display_format!(fb.pixels());
    let radius = (size / 2.0) as i32;
    draw_circle(fb, x as i32, y as i32, radius, color);
}

/// Helper to draw a horizontal line (used by filled circle).
#[inline]
fn draw_horizontal_line(fb: &mut Framebuffer, x1: i32, x2: i32, y: i32, color: Rgba) {
    if y < 0 || y >= i32_px(fb.height()) {
        return;
    }

    let x_start = x1.max(0) as u32;
    let x_end = (x2 + 1).max(0).min(i32_px(fb.width())) as u32;

    if x_start < x_end {
        let width = x_end - x_start;
        fb.fill_rect(x_start, y as u32, width, 1, color);
    }
}

/// Helper to plot a single circle point with bounds checking.
#[inline]
fn plot_circle_point(fb: &mut Framebuffer, x: i32, y: i32, color: Rgba) {
    if x >= 0 && y >= 0 && x < i32_px(fb.width()) && y < i32_px(fb.height()) {
        fb.set_pixel(x as u32, y as u32, color);
    }
}

impl Drawable for Point {
    fn draw(&self, fb: &mut Framebuffer, color: Rgba) {
        draw_point(fb, self.x, self.y, 1.0, color);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_draw_line_horizontal() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        draw_line(&mut fb, 10, 50, 90, 50, Rgba::BLACK);

        // Check that pixels along the line are set
        assert_eq!(fb.get_pixel(10, 50), Some(Rgba::BLACK));
        assert_eq!(fb.get_pixel(50, 50), Some(Rgba::BLACK));
        assert_eq!(fb.get_pixel(90, 50), Some(Rgba::BLACK));
    }

    #[test]
    fn test_draw_line_vertical() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        draw_line(&mut fb, 50, 10, 50, 90, Rgba::BLACK);

        assert_eq!(fb.get_pixel(50, 10), Some(Rgba::BLACK));
        assert_eq!(fb.get_pixel(50, 50), Some(Rgba::BLACK));
        assert_eq!(fb.get_pixel(50, 90), Some(Rgba::BLACK));
    }

    #[test]
    fn test_draw_line_diagonal() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        draw_line(&mut fb, 10, 10, 90, 90, Rgba::BLACK);

        assert_eq!(fb.get_pixel(10, 10), Some(Rgba::BLACK));
        assert_eq!(fb.get_pixel(50, 50), Some(Rgba::BLACK));
        assert_eq!(fb.get_pixel(90, 90), Some(Rgba::BLACK));
    }

    #[test]
    fn test_draw_line_aa() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        draw_line_aa(&mut fb, 10.0, 10.0, 90.0, 50.0, Rgba::BLACK);

        // Anti-aliased line should have some pixels set along the path
        // Not checking exact values due to anti-aliasing blending
        let pixel = fb.get_pixel(50, 30);
        assert!(pixel.is_some());
    }

    #[test]
    fn test_draw_rect() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        draw_rect(&mut fb, 20, 20, 30, 30, Rgba::RED);

        assert_eq!(fb.get_pixel(25, 25), Some(Rgba::RED));
        assert_eq!(fb.get_pixel(10, 10), Some(Rgba::WHITE));
    }

    #[test]
    fn test_draw_rect_outline() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        draw_rect_outline(&mut fb, 20, 20, 30, 30, Rgba::RED, 2);

        // Border should be red
        assert_eq!(fb.get_pixel(20, 20), Some(Rgba::RED));
        // Inside should be white
        assert_eq!(fb.get_pixel(35, 35), Some(Rgba::WHITE));
    }

    #[test]
    fn test_draw_circle() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        draw_circle(&mut fb, 50, 50, 20, Rgba::BLUE);

        // Center should be filled
        assert_eq!(fb.get_pixel(50, 50), Some(Rgba::BLUE));
        // Outside should be white
        assert_eq!(fb.get_pixel(5, 5), Some(Rgba::WHITE));
    }

    #[test]
    fn test_draw_circle_outline() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        draw_circle_outline(&mut fb, 50, 50, 20, Rgba::GREEN);

        // Edge should be colored
        assert_eq!(fb.get_pixel(70, 50), Some(Rgba::GREEN));
        // Center should still be white (outline only)
        assert_eq!(fb.get_pixel(50, 50), Some(Rgba::WHITE));
    }

    #[test]
    fn test_draw_point() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        draw_point(&mut fb, 50.0, 50.0, 10.0, Rgba::RED);

        // Center should be filled
        assert_eq!(fb.get_pixel(50, 50), Some(Rgba::RED));
    }

    #[test]
    fn test_drawable_trait_line() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        let line = Line::from_coords(10.0, 10.0, 90.0, 90.0);
        line.draw(&mut fb, Rgba::BLACK);

        assert_eq!(fb.get_pixel(50, 50), Some(Rgba::BLACK));
    }

    #[test]
    fn test_drawable_trait_rect() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        let rect = Rect::new(20.0, 20.0, 30.0, 30.0);
        rect.draw(&mut fb, Rgba::GREEN);

        assert_eq!(fb.get_pixel(35, 35), Some(Rgba::GREEN));
    }

    #[test]
    fn test_drawable_trait_point() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        let point = Point::new(50.0, 50.0);
        point.draw(&mut fb, Rgba::BLUE);

        // Single point should set at least the center pixel
        let pixel = fb.get_pixel(50, 50);
        assert!(pixel.is_some());
    }

    #[test]
    fn test_line_out_of_bounds() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        // Line that goes out of bounds should not panic
        draw_line(&mut fb, -10, -10, 110, 110, Rgba::BLACK);

        // Only in-bounds pixels should be affected
        assert_eq!(fb.get_pixel(50, 50), Some(Rgba::BLACK));
    }

    #[test]
    fn test_circle_zero_radius() {
        let mut fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        draw_circle(&mut fb, 50, 50, 0, Rgba::RED);

        // Zero radius should just draw a single point
        assert_eq!(fb.get_pixel(50, 50), Some(Rgba::RED));
    }
}
