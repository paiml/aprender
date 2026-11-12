//! Primitive rendering functions.
//!
//! Implements rasterization algorithms for basic geometric shapes.

use super::clip::canvas_segment;
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

    // Clip to the canvas (plus a guard) first: a segment that starts or ends far off the canvas
    // would otherwise walk up to 2^32 invisible pixels, and `x1 - x0` could overflow `i32`.
    let Some((cx0, cy0, cx1, cy1)) =
        canvas_segment(fb, (f64::from(x0), f64::from(y0), f64::from(x1), f64::from(y1)))
    else {
        return;
    };
    // The clipped ends are within the guard box, so the casts below are exact (an unclipped
    // segment round-trips through `f64` unchanged).
    let mut cursor = BresenhamCursor::new(
        cx0.round() as i32,
        cy0.round() as i32,
        cx1.round() as i32,
        cy1.round() as i32,
    );
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
        Self { x: x0, y: y0, x1, y1, dx, dy, sx, sy, err: dx + dy }
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
    // Total for every f32: a non-finite end draws nothing, and a segment far off the canvas is
    // clipped before any loop (see `clip`). A segment already near the canvas is unchanged.
    let Some((x0, y0, x1, y1)) =
        canvas_segment(fb, (f64::from(x0), f64::from(y0), f64::from(x1), f64::from(y1)))
    else {
        return;
    };
    let (x0, y0, x1, y1) = (x0 as f32, y0 as f32, x1 as f32, y1 as f32);
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
///
/// A rectangle that starts left of or above the canvas is clipped there: its far edge stays at
/// `origin + extent`. (Before paiml/aprender#6949 it was shifted onto the canvas with its full
/// width, so a bar whose left edge was off canvas was drawn too wide.)
pub fn draw_rect(fb: &mut Framebuffer, x: i32, y: i32, width: u32, height: u32, color: Rgba) {
    // Contract: display-format-v1.yaml precondition (pv codegen)
    contract_pre_display_format!(fb.pixels());
    contract_post_display_format!(&"ok");
    // Clip in i64 so `x + width` cannot overflow, and so a rectangle that starts left of or
    // above the canvas keeps its far edge where it is instead of being shifted onto the canvas.
    let clip = |origin: i32, extent: u32| -> (u32, u32) {
        let start = i64::from(origin);
        let end = (start + i64::from(extent)).clamp(0, i64::from(u32::MAX));
        let start = start.clamp(0, i64::from(u32::MAX));
        (start as u32, (end - start) as u32)
    };
    let (x, w) = clip(x, width);
    let (y, h) = clip(y, height);
    fb.fill_rect(x, y, w, h, color);
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
    // Saturating throughout: `y + height` and `2 * thickness` can exceed `u32` for extreme
    // sizes, and `fill_rect` clips whatever lands past the canvas.
    let double = thickness.saturating_mul(2);

    // Top edge
    fb.fill_rect(x, y, width, thickness, color);
    // Bottom edge
    if height > thickness {
        fb.fill_rect(x, y.saturating_add(height - thickness), width, thickness, color);
    }
    // Left edge
    if height > double {
        fb.fill_rect(x, y.saturating_add(thickness), thickness, height - double, color);
    }
    // Right edge
    if width > thickness && height > double {
        fb.fill_rect(
            x.saturating_add(width - thickness),
            y.saturating_add(thickness),
            thickness,
            height - double,
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

/// Radii up to this walk the midpoint algorithm (at most about 2 900 steps); a larger circle that
/// reaches the canvas is filled row by row over the canvas rows only, so the work is bounded by
/// the canvas height whatever the radius.
const MAX_MIDPOINT_RADIUS: i32 = 4096;

/// Whether the circle's bounding square misses the canvas entirely.
fn circle_off_canvas(fb: &Framebuffer, cx: i64, cy: i64, r: i64) -> bool {
    cx.saturating_add(r) < 0
        || cx.saturating_sub(r) >= i64::from(fb.width())
        || cy.saturating_add(r) < 0
        || cy.saturating_sub(r) >= i64::from(fb.height())
}

/// Largest `s` with `s * s <= n` (`n >= 0`).
fn isqrt(n: i64) -> i64 {
    // The f64 root is off by at most one for n <= 2^62, in either direction.
    let s = (n as f64).sqrt() as i64;
    (s - 1..=s + 1).rev().find(|c| c * c <= n).unwrap_or(0)
}

/// The canvas rows a circle of radius `r` reaches, each with its half-width.
fn circle_rows(fb: &Framebuffer, cy: i64, r: i64) -> impl Iterator<Item = (i64, i64)> {
    let r2 = r * r;
    let first = cy.saturating_sub(r).max(0);
    let last = cy.saturating_add(r).min(i64::from(fb.height()) - 1);
    (first..=last).map(move |py| {
        let dy = py - cy;
        (py, isqrt(r2 - dy * dy))
    })
}

/// Draw a filled circle using the midpoint algorithm.
///
/// Total for every `i32` centre and radius: arithmetic is done in `i64`, a circle that misses the
/// canvas draws nothing, and a radius above [`MAX_MIDPOINT_RADIUS`] is filled over the canvas rows
/// only. Output for a radius up to that limit is the midpoint algorithm's, unchanged.
///
/// The two paths are not pixel-identical at the threshold: the row path is the exact integer disc,
/// the midpoint circle differs from it on pixels within one pixel of the edge (measured: about 300
/// pixels on a 400x300 canvas at radius 4096). Contract `render-total-v1`, RT-DEV-002.
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
    let (cx, cy, r) = (i64::from(cx), i64::from(cy), i64::from(radius));
    if circle_off_canvas(fb, cx, cy, r) {
        return;
    }
    if radius > MAX_MIDPOINT_RADIUS {
        for (py, half) in circle_rows(fb, cy, r) {
            draw_horizontal_line(fb, cx - half, cx + half, py, color);
        }
        return;
    }

    // Midpoint circle algorithm for filled circle
    let mut x = i64::from(radius);
    let mut y = 0_i64;
    let mut err = 1 - x;

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
///
/// Total like [`draw_circle`]. A radius above the midpoint limit is drawn as the two end points of
/// each canvas row it crosses, so near the top and bottom of such a circle the outline has gaps
/// that the midpoint outline does not (RT-DEV-002).
pub fn draw_circle_outline(fb: &mut Framebuffer, cx: i32, cy: i32, radius: i32, color: Rgba) {
    // Contract: display-format-v1.yaml precondition (pv codegen)
    contract_pre_display_format!(fb.pixels());
    if radius <= 0 {
        if radius == 0 && cx >= 0 && cy >= 0 {
            fb.set_pixel(cx as u32, cy as u32, color);
        }
        return;
    }
    let (cx, cy, r) = (i64::from(cx), i64::from(cy), i64::from(radius));
    if circle_off_canvas(fb, cx, cy, r) {
        return;
    }
    if radius > MAX_MIDPOINT_RADIUS {
        for (py, half) in circle_rows(fb, cy, r) {
            plot_circle_point(fb, cx - half, py, color);
            plot_circle_point(fb, cx + half, py, color);
        }
        return;
    }

    let mut x = i64::from(radius);
    let mut y = 0_i64;
    let mut err = 1 - x;

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
///
/// A non-finite `x`, `y` or `size` casts to a finite `i32` (NaN to 0, infinities to the ends), so
/// the circle is still total; callers that must not draw such a point drop it first.
pub fn draw_point(fb: &mut Framebuffer, x: f32, y: f32, size: f32, color: Rgba) {
    // Contract: display-format-v1.yaml precondition (pv codegen)
    contract_pre_display_format!(fb.pixels());
    let radius = (size / 2.0) as i32;
    draw_circle(fb, x as i32, y as i32, radius, color);
}

/// Helper to draw a horizontal line (used by filled circle); `i64` so no sum can overflow.
#[inline]
fn draw_horizontal_line(fb: &mut Framebuffer, x1: i64, x2: i64, y: i64, color: Rgba) {
    if y < 0 || y >= i64::from(fb.height()) {
        return;
    }

    let x_start = x1.max(0);
    let x_end = (x2 + 1).max(0).min(i64::from(fb.width()));

    if x_start < x_end {
        let width = x_end - x_start;
        fb.fill_rect(x_start as u32, y as u32, width as u32, 1, color);
    }
}

/// Helper to plot a single circle point with bounds checking.
#[inline]
fn plot_circle_point(fb: &mut Framebuffer, x: i64, y: i64, color: Rgba) {
    if x >= 0 && y >= 0 && x < i64::from(fb.width()) && y < i64::from(fb.height()) {
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

    #[test]
    fn draw_rect_clips_a_negative_origin_and_keeps_the_far_edge() {
        // Columns -100..200 on a 300x100 canvas: 0..200 are ink, 200.. are not. Main shifted the
        // rectangle to start at 0, which painted 0..300 (the far edge moved right by 100).
        let mut fb = Framebuffer::new(300, 100).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);
        draw_rect(&mut fb, -100, 10, 300, 20, Rgba::BLACK);
        for x in 0..300 {
            for y in 0..100 {
                let want = x < 200 && (10..30).contains(&y);
                assert_eq!(fb.get_pixel(x, y) == Some(Rgba::BLACK), want, "({x}, {y})");
            }
        }
        // The same along y: rows -5..15 clip to 0..15.
        let mut fb = Framebuffer::new(50, 50).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);
        draw_rect(&mut fb, 5, -5, 10, 20, Rgba::BLACK);
        for y in 0..50 {
            assert_eq!(fb.get_pixel(7, y) == Some(Rgba::BLACK), y < 15, "row {y}");
        }
    }

    #[test]
    fn draw_rect_extreme_extents_cover_what_they_reach() {
        let mut fb = Framebuffer::new(20, 20).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);
        // i32::MIN + u32::MAX = i32::MAX: the rectangle spans the whole canvas.
        draw_rect(&mut fb, i32::MIN, i32::MIN, u32::MAX, u32::MAX, Rgba::BLACK);
        assert!((0..20).all(|x| (0..20).all(|y| fb.get_pixel(x, y) == Some(Rgba::BLACK))));
        // Ends exactly at the canvas edge (-1 + 1 = 0): nothing is drawn.
        let mut fb = Framebuffer::new(20, 20).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);
        draw_rect(&mut fb, -1, -1, 1, 1, Rgba::BLACK);
        assert!((0..20).all(|x| (0..20).all(|y| fb.get_pixel(x, y) == Some(Rgba::WHITE))));
    }

    #[test]
    fn isqrt_corrects_a_rounded_f64_square_root() {
        // 2^62 - 1 rounds up to 2^62 as an f64, whose root is one too large.
        assert_eq!(isqrt((1_i64 << 62) - 1), (1_i64 << 31) - 1);
        assert_eq!(isqrt(1_i64 << 62), 1_i64 << 31);
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(15), 3);
        assert_eq!(isqrt(16), 4);
        // Near 2^62 the f64 square root is off by one in either direction; both are corrected.
        for k in ((1_i64 << 31) - 2000)..(1_i64 << 31) {
            assert_eq!(isqrt(k * k), k);
            assert_eq!(isqrt(k * k + 2 * k), k);
            assert_eq!(isqrt(k * k - 1), k - 1);
        }
    }
}
