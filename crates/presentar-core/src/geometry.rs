//! Geometric primitives: Point, Size, Rect, `CornerRadius`.
//!
//! This module provides the fundamental geometric types used throughout Presentar
//! for layout calculations and rendering.
//!
//! # Examples
//!
//! ```
//! use presentar_core::{Point, Size, Rect};
//!
//! // Create points and calculate distances
//! let p1 = Point::new(0.0, 0.0);
//! let p2 = Point::new(3.0, 4.0);
//! assert_eq!(p1.distance(&p2), 5.0);
//!
//! // Create sizes
//! let size = Size::new(100.0, 50.0);
//! assert_eq!(size.area(), 5000.0);
//!
//! // Create rectangles
//! let rect = Rect::new(10.0, 20.0, 100.0, 50.0);
//! assert!(rect.contains_point(&Point::new(50.0, 40.0)));
//!
//! // Rectangle intersection
//! let r1 = Rect::new(0.0, 0.0, 100.0, 100.0);
//! let r2 = Rect::new(50.0, 50.0, 100.0, 100.0);
//! let inter = r1.intersection(&r2).expect("should intersect");
//! assert_eq!(inter.width, 50.0);
//! ```

use provable_contracts_macros::contract;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Sub};

/// A 2D point with x and y coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// X coordinate
    pub x: f32,
    /// Y coordinate
    pub y: f32,
}

impl Point {
    /// Origin point (0, 0)
    pub const ORIGIN: Self = Self { x: 0.0, y: 0.0 };

    /// Create a new point.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Calculate Euclidean distance to another point.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx.hypot(dy)
    }

    /// Linear interpolation between two points.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self::new(
            (other.x - self.x).mul_add(t, self.x),
            (other.y - self.y).mul_add(t, self.y),
        )
    }
}

impl Default for Point {
    fn default() -> Self {
        Self::ORIGIN
    }
}

impl Add for Point {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Point {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

/// A 2D size with width and height.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Size {
    /// Width
    pub width: f32,
    /// Height
    pub height: f32,
}

impl Size {
    /// Zero size
    pub const ZERO: Self = Self {
        width: 0.0,
        height: 0.0,
    };

    /// Create a new size.
    #[must_use]
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    /// Calculate area.
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Calculate aspect ratio (width / height).
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0.0 {
            0.0
        } else {
            self.width / self.height
        }
    }

    /// Check if this size can contain another size.
    #[must_use]
    pub fn contains(&self, other: &Self) -> bool {
        self.width >= other.width && self.height >= other.height
    }

    /// Scale size by a factor.
    #[must_use]
    pub fn scale(&self, factor: f32) -> Self {
        Self::new(self.width * factor, self.height * factor)
    }
}

impl Default for Size {
    fn default() -> Self {
        Self::ZERO
    }
}

/// A rectangle defined by position and size.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    /// X position of top-left corner
    pub x: f32,
    /// Y position of top-left corner
    pub y: f32,
    /// Width
    pub width: f32,
    /// Height
    pub height: f32,
}

impl Rect {
    /// Create a new rectangle.
    #[must_use]
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Create from two corner points.
    #[must_use]
    pub fn from_points(top_left: Point, bottom_right: Point) -> Self {
        Self::new(
            top_left.x,
            top_left.y,
            bottom_right.x - top_left.x,
            bottom_right.y - top_left.y,
        )
    }

    /// Create from size at origin.
    #[must_use]
    pub const fn from_size(size: Size) -> Self {
        Self::new(0.0, 0.0, size.width, size.height)
    }

    /// Get the origin (top-left) point.
    #[must_use]
    pub const fn origin(&self) -> Point {
        Point::new(self.x, self.y)
    }

    /// Get the size.
    #[must_use]
    pub const fn size(&self) -> Size {
        Size::new(self.width, self.height)
    }

    /// Get the area.
    #[must_use]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Get top-left corner.
    #[must_use]
    pub const fn top_left(&self) -> Point {
        Point::new(self.x, self.y)
    }

    /// Get top-right corner.
    #[must_use]
    pub fn top_right(&self) -> Point {
        Point::new(self.x + self.width, self.y)
    }

    /// Get bottom-left corner.
    #[must_use]
    pub fn bottom_left(&self) -> Point {
        Point::new(self.x, self.y + self.height)
    }

    /// Get bottom-right corner.
    #[must_use]
    pub fn bottom_right(&self) -> Point {
        Point::new(self.x + self.width, self.y + self.height)
    }

    /// Get center point.
    #[must_use]
    pub fn center(&self) -> Point {
        Point::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Check if a point is inside the rectangle (inclusive).
    #[must_use]
    pub fn contains_point(&self, point: &Point) -> bool {
        point.x >= self.x
            && point.x <= self.x + self.width
            && point.y >= self.y
            && point.y <= self.y + self.height
    }

    /// Check if this rectangle intersects another.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }

    /// Calculate intersection with another rectangle.
    #[must_use]
    #[contract("rect-geometry-v1", equation = "intersection")]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = (self.x + self.width).min(other.x + other.width);
        let bottom = (self.y + self.height).min(other.y + other.height);

        if right > x && bottom > y {
            Some(Self::new(x, y, right - x, bottom - y))
        } else {
            None
        }
    }

    /// Calculate union with another rectangle.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = (self.x + self.width).max(other.x + other.width);
        let bottom = (self.y + self.height).max(other.y + other.height);

        Self::new(x, y, right - x, bottom - y)
    }

    /// Create a new rectangle inset by the given amount on all sides.
    #[must_use]
    #[provable_contracts_macros::contract("rect-geometry-v1", equation = "inset")]
    pub fn inset(&self, amount: f32) -> Self {
        Self::new(
            self.x + amount,
            self.y + amount,
            2.0f32.mul_add(-amount, self.width).max(0.0),
            2.0f32.mul_add(-amount, self.height).max(0.0),
        )
    }

    /// Create a new rectangle with the given position.
    #[must_use]
    pub const fn with_origin(&self, origin: Point) -> Self {
        Self::new(origin.x, origin.y, self.width, self.height)
    }

    /// Create a new rectangle with the given size.
    #[must_use]
    pub const fn with_size(&self, size: Size) -> Self {
        Self::new(self.x, self.y, size.width, size.height)
    }
}

impl Default for Rect {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

/// Corner radii for rounded rectangles.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CornerRadius {
    /// Top-left radius
    pub top_left: f32,
    /// Top-right radius
    pub top_right: f32,
    /// Bottom-right radius
    pub bottom_right: f32,
    /// Bottom-left radius
    pub bottom_left: f32,
}

impl CornerRadius {
    /// Zero radius
    pub const ZERO: Self = Self {
        top_left: 0.0,
        top_right: 0.0,
        bottom_right: 0.0,
        bottom_left: 0.0,
    };

    /// Create corner radii with individual values.
    #[must_use]
    pub const fn new(top_left: f32, top_right: f32, bottom_right: f32, bottom_left: f32) -> Self {
        Self {
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        }
    }

    /// Create uniform corner radius.
    #[must_use]
    pub const fn uniform(radius: f32) -> Self {
        Self::new(radius, radius, radius, radius)
    }

    /// Check if all corners have zero radius.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.top_left == 0.0
            && self.top_right == 0.0
            && self.bottom_right == 0.0
            && self.bottom_left == 0.0
    }

    /// Check if all corners have the same radius.
    #[must_use]
    pub fn is_uniform(&self) -> bool {
        self.top_left == self.top_right
            && self.top_right == self.bottom_right
            && self.bottom_right == self.bottom_left
    }
}

impl Default for CornerRadius {
    fn default() -> Self {
        Self::ZERO
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn test_point_default() {
        assert_eq!(Point::default(), Point::ORIGIN);
    }

    #[test]
    fn test_point_lerp() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(10.0, 10.0);
        let mid = p1.lerp(&p2, 0.5);
        assert_eq!(mid, Point::new(5.0, 5.0));
    }

    #[test]
    fn test_size_default() {
        assert_eq!(Size::default(), Size::ZERO);
    }

    #[test]
    fn test_size_scale() {
        let s = Size::new(10.0, 20.0);
        assert_eq!(s.scale(2.0), Size::new(20.0, 40.0));
    }

    #[test]
    fn test_rect_default() {
        let r = Rect::default();
        assert_eq!(r.x, 0.0);
        assert_eq!(r.area(), 0.0);
    }

    #[test]
    fn test_corner_radius_is_uniform() {
        assert!(CornerRadius::uniform(10.0).is_uniform());
        assert!(!CornerRadius::new(1.0, 2.0, 3.0, 4.0).is_uniform());
    }

    #[test]
    fn test_corner_radius_is_zero() {
        assert!(CornerRadius::ZERO.is_zero());
        assert!(!CornerRadius::uniform(1.0).is_zero());
    }

    // ===== Point Tests =====

    #[test]
    fn test_point_new() {
        let p = Point::new(10.0, 20.0);
        assert_eq!(p.x, 10.0);
        assert_eq!(p.y, 20.0);
    }

    #[test]
    fn test_point_distance() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);
        assert_eq!(p1.distance(&p2), 5.0);
    }

    #[test]
    fn test_point_add() {
        let p1 = Point::new(1.0, 2.0);
        let p2 = Point::new(3.0, 4.0);
        assert_eq!(p1 + p2, Point::new(4.0, 6.0));
    }

    #[test]
    fn test_point_sub() {
        let p1 = Point::new(5.0, 7.0);
        let p2 = Point::new(2.0, 3.0);
        assert_eq!(p1 - p2, Point::new(3.0, 4.0));
    }

    // ===== Size Tests =====

    #[test]
    fn test_size_new() {
        let s = Size::new(100.0, 50.0);
        assert_eq!(s.width, 100.0);
        assert_eq!(s.height, 50.0);
    }

    #[test]
    fn test_size_area() {
        let s = Size::new(10.0, 20.0);
        assert_eq!(s.area(), 200.0);
    }

    #[test]
    fn test_size_aspect_ratio() {
        let s = Size::new(16.0, 9.0);
        assert!((s.aspect_ratio() - 1.777).abs() < 0.01);
    }

    #[test]
    fn test_size_aspect_ratio_zero_height() {
        let s = Size::new(10.0, 0.0);
        assert_eq!(s.aspect_ratio(), 0.0);
    }

    #[test]
    fn test_size_contains() {
        let big = Size::new(100.0, 100.0);
        let small = Size::new(50.0, 50.0);
        assert!(big.contains(&small));
        assert!(!small.contains(&big));
    }

    // ===== Rect Tests =====

    #[test]
    fn test_rect_from_points() {
        let r = Rect::from_points(Point::new(10.0, 20.0), Point::new(50.0, 70.0));
        assert_eq!(r.x, 10.0);
        assert_eq!(r.y, 20.0);
        assert_eq!(r.width, 40.0);
        assert_eq!(r.height, 50.0);
    }

    #[test]
    fn test_rect_from_size() {
        let r = Rect::from_size(Size::new(100.0, 50.0));
        assert_eq!(r.x, 0.0);
        assert_eq!(r.y, 0.0);
        assert_eq!(r.width, 100.0);
    }

    #[test]
    fn test_rect_corners() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(r.top_left(), Point::new(10.0, 20.0));
        assert_eq!(r.top_right(), Point::new(110.0, 20.0));
        assert_eq!(r.bottom_left(), Point::new(10.0, 70.0));
        assert_eq!(r.bottom_right(), Point::new(110.0, 70.0));
    }

    #[test]
    fn test_rect_center() {
        let r = Rect::new(0.0, 0.0, 100.0, 50.0);
        assert_eq!(r.center(), Point::new(50.0, 25.0));
    }

    #[test]
    fn test_rect_contains_point() {
        let r = Rect::new(10.0, 10.0, 100.0, 100.0);
        assert!(r.contains_point(&Point::new(50.0, 50.0)));
        assert!(r.contains_point(&Point::new(10.0, 10.0))); // Edge
        assert!(!r.contains_point(&Point::new(5.0, 50.0)));
    }

    #[test]
    fn test_rect_intersects() {
        let r1 = Rect::new(0.0, 0.0, 100.0, 100.0);
        let r2 = Rect::new(50.0, 50.0, 100.0, 100.0);
        let r3 = Rect::new(200.0, 200.0, 50.0, 50.0);
        assert!(r1.intersects(&r2));
        assert!(!r1.intersects(&r3));
    }

    #[test]
    fn test_rect_intersection() {
        let r1 = Rect::new(0.0, 0.0, 100.0, 100.0);
        let r2 = Rect::new(50.0, 50.0, 100.0, 100.0);
        let inter = r1.intersection(&r2).unwrap();
        assert_eq!(inter, Rect::new(50.0, 50.0, 50.0, 50.0));
    }

    #[test]
    fn test_rect_intersection_none() {
        let r1 = Rect::new(0.0, 0.0, 50.0, 50.0);
        let r2 = Rect::new(100.0, 100.0, 50.0, 50.0);
        assert!(r1.intersection(&r2).is_none());
    }

    #[test]
    fn test_rect_union() {
        let r1 = Rect::new(0.0, 0.0, 50.0, 50.0);
        let r2 = Rect::new(25.0, 25.0, 50.0, 50.0);
        let u = r1.union(&r2);
        assert_eq!(u, Rect::new(0.0, 0.0, 75.0, 75.0));
    }

    #[test]
    fn test_rect_inset() {
        let r = Rect::new(0.0, 0.0, 100.0, 100.0);
        let inset = r.inset(10.0);
        assert_eq!(inset, Rect::new(10.0, 10.0, 80.0, 80.0));
    }

    #[test]
    fn test_rect_inset_clamps() {
        let r = Rect::new(0.0, 0.0, 20.0, 20.0);
        let inset = r.inset(15.0);
        assert_eq!(inset.width, 0.0);
        assert_eq!(inset.height, 0.0);
    }

    #[test]
    fn test_rect_with_origin() {
        let r = Rect::new(0.0, 0.0, 100.0, 50.0);
        let moved = r.with_origin(Point::new(20.0, 30.0));
        assert_eq!(moved.x, 20.0);
        assert_eq!(moved.y, 30.0);
        assert_eq!(moved.width, 100.0);
    }

    #[test]
    fn test_rect_with_size() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        let resized = r.with_size(Size::new(200.0, 100.0));
        assert_eq!(resized.x, 10.0);
        assert_eq!(resized.width, 200.0);
    }
}
