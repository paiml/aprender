//! Geometric primitives for visualization.
//!
//! Provides basic geometric types used for rendering plots.

/// A 2D point with floating-point coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

impl Point {
    /// Origin point (0, 0).
    pub const ORIGIN: Self = Self::new(0.0, 0.0);

    /// Create a new point.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Calculate the distance to another point.
    #[must_use]
    pub fn distance(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Linear interpolation between two points.
    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self::new(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)
    }
}

/// A line segment between two points.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Line {
    /// Start point.
    pub start: Point,
    /// End point.
    pub end: Point,
}

impl Line {
    /// Create a new line segment.
    #[must_use]
    pub const fn new(start: Point, end: Point) -> Self {
        Self { start, end }
    }

    /// Create a line from coordinates.
    #[must_use]
    pub const fn from_coords(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self::new(Point::new(x0, y0), Point::new(x1, y1))
    }

    /// Get the length of the line.
    #[must_use]
    pub fn length(&self) -> f32 {
        self.start.distance(self.end)
    }
}

/// A rectangle defined by position and size.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rect {
    /// X coordinate of the top-left corner.
    pub x: f32,
    /// Y coordinate of the top-left corner.
    pub y: f32,
    /// Width of the rectangle.
    pub width: f32,
    /// Height of the rectangle.
    pub height: f32,
}

impl Rect {
    /// Create a new rectangle.
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Create a rectangle from two corner points.
    #[must_use]
    pub fn from_corners(top_left: Point, bottom_right: Point) -> Self {
        Self::new(top_left.x, top_left.y, bottom_right.x - top_left.x, bottom_right.y - top_left.y)
    }

    /// Check if a point is inside the rectangle.
    #[must_use]
    pub fn contains(&self, point: Point) -> bool {
        point.x >= self.x
            && point.x <= self.x + self.width
            && point.y >= self.y
            && point.y <= self.y + self.height
    }

    /// Get the center point of the rectangle.
    #[must_use]
    pub fn center(&self) -> Point {
        Point::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Get the area of the rectangle.
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);
        assert!((p1.distance(p2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_point_lerp() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(10.0, 10.0);
        let mid = p1.lerp(p2, 0.5);
        assert!((mid.x - 5.0).abs() < 0.001);
        assert!((mid.y - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_line_length() {
        let line = Line::from_coords(0.0, 0.0, 3.0, 4.0);
        assert!((line.length() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_rect_contains() {
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        assert!(rect.contains(Point::new(5.0, 5.0)));
        assert!(!rect.contains(Point::new(15.0, 5.0)));
    }

    #[test]
    fn test_rect_area() {
        let rect = Rect::new(0.0, 0.0, 10.0, 5.0);
        assert!((rect.area() - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_point_origin() {
        assert!(Point::ORIGIN.x.abs() < f32::EPSILON);
        assert!(Point::ORIGIN.y.abs() < f32::EPSILON);
    }

    #[test]
    fn test_point_new() {
        let p = Point::new(3.5, 7.2);
        assert!((p.x - 3.5).abs() < 0.001);
        assert!((p.y - 7.2).abs() < 0.001);
    }

    #[test]
    fn test_point_default() {
        let p = Point::default();
        assert!(p.x.abs() < f32::EPSILON);
        assert!(p.y.abs() < f32::EPSILON);
    }

    #[test]
    fn test_point_eq() {
        let p1 = Point::new(1.0, 2.0);
        let p2 = Point::new(1.0, 2.0);
        let p3 = Point::new(3.0, 4.0);
        assert_eq!(p1, p2);
        assert_ne!(p1, p3);
    }

    #[test]
    fn test_point_debug_clone() {
        let p = Point::new(1.0, 2.0);
        let p2 = p;
        let _ = format!("{p2:?}");
    }

    #[test]
    fn test_line_new() {
        let start = Point::new(1.0, 2.0);
        let end = Point::new(3.0, 4.0);
        let line = Line::new(start, end);
        assert_eq!(line.start, start);
        assert_eq!(line.end, end);
    }

    #[test]
    fn test_line_default() {
        let line = Line::default();
        assert_eq!(line.start, Point::default());
        assert_eq!(line.end, Point::default());
    }

    #[test]
    fn test_line_eq() {
        let l1 = Line::from_coords(0.0, 0.0, 1.0, 1.0);
        let l2 = Line::from_coords(0.0, 0.0, 1.0, 1.0);
        let l3 = Line::from_coords(0.0, 0.0, 2.0, 2.0);
        assert_eq!(l1, l2);
        assert_ne!(l1, l3);
    }

    #[test]
    fn test_line_debug_clone() {
        let line = Line::from_coords(0.0, 0.0, 1.0, 1.0);
        let line2 = line;
        let _ = format!("{line2:?}");
    }

    #[test]
    fn test_rect_from_corners() {
        let rect = Rect::from_corners(Point::new(10.0, 20.0), Point::new(50.0, 60.0));
        assert!((rect.x - 10.0).abs() < 0.001);
        assert!((rect.y - 20.0).abs() < 0.001);
        assert!((rect.width - 40.0).abs() < 0.001);
        assert!((rect.height - 40.0).abs() < 0.001);
    }

    #[test]
    fn test_rect_center() {
        let rect = Rect::new(10.0, 20.0, 40.0, 60.0);
        let center = rect.center();
        assert!((center.x - 30.0).abs() < 0.001);
        assert!((center.y - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_rect_default() {
        let rect = Rect::default();
        assert!(rect.x.abs() < f32::EPSILON);
        assert!(rect.y.abs() < f32::EPSILON);
        assert!(rect.width.abs() < f32::EPSILON);
        assert!(rect.height.abs() < f32::EPSILON);
    }

    #[test]
    fn test_rect_eq() {
        let r1 = Rect::new(0.0, 0.0, 10.0, 10.0);
        let r2 = Rect::new(0.0, 0.0, 10.0, 10.0);
        let r3 = Rect::new(5.0, 5.0, 10.0, 10.0);
        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
    }

    #[test]
    fn test_rect_debug_clone() {
        let rect = Rect::new(1.0, 2.0, 3.0, 4.0);
        let rect2 = rect;
        let _ = format!("{rect2:?}");
    }

    #[test]
    fn test_rect_contains_boundary() {
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        // On boundaries
        assert!(rect.contains(Point::new(0.0, 0.0))); // top-left
        assert!(rect.contains(Point::new(10.0, 10.0))); // bottom-right
        assert!(rect.contains(Point::new(0.0, 10.0))); // bottom-left
        assert!(rect.contains(Point::new(10.0, 0.0))); // top-right
    }

    #[test]
    fn test_rect_contains_outside() {
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        assert!(!rect.contains(Point::new(-1.0, 5.0))); // left
        assert!(!rect.contains(Point::new(11.0, 5.0))); // right
        assert!(!rect.contains(Point::new(5.0, -1.0))); // above
        assert!(!rect.contains(Point::new(5.0, 11.0))); // below
    }
}
