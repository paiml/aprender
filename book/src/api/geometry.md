# Geometry Types

This chapter documents the geometric primitives in trueno-viz.

## Point

A 2D point with floating-point coordinates.

```rust
use trueno_viz::geometry::Point;

// Construction
let p = Point::new(10.0, 20.0);
let origin = Point::ORIGIN;  // (0, 0)

// Field access
println!("x: {}, y: {}", p.x, p.y);

// Operations
let other = Point::new(13.0, 24.0);
let dist = p.distance(other);  // 5.0 (3-4-5 triangle)

// Linear interpolation
let mid = p.lerp(other, 0.5);  // Midpoint
```

## Line

A line segment between two points.

```rust
use trueno_viz::geometry::{Line, Point};

// Construction
let line = Line::new(
    Point::new(0.0, 0.0),
    Point::new(10.0, 10.0)
);

// From coordinates
let line = Line::from_coords(0.0, 0.0, 10.0, 10.0);

// Properties
let len = line.length();  // ~14.14
```

## Rect

A rectangle defined by position and size.

```rust
use trueno_viz::geometry::{Rect, Point};

// Construction
let rect = Rect::new(10.0, 20.0, 100.0, 50.0);

// From corners
let rect = Rect::from_corners(
    Point::new(10.0, 20.0),   // Top-left
    Point::new(110.0, 70.0)   // Bottom-right
);

// Properties
let center = rect.center();
let area = rect.area();

// Hit testing
let inside = rect.contains(Point::new(50.0, 40.0));  // true
let outside = rect.contains(Point::new(200.0, 40.0)); // false
```

## Complete API

```rust
impl Point {
    pub const ORIGIN: Self;
    pub const fn new(x: f32, y: f32) -> Self;
    pub fn distance(self, other: Self) -> f32;
    pub fn lerp(self, other: Self, t: f32) -> Self;
}

impl Line {
    pub const fn new(start: Point, end: Point) -> Self;
    pub const fn from_coords(x0: f32, y0: f32, x1: f32, y1: f32) -> Self;
    pub fn length(&self) -> f32;
}

impl Rect {
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self;
    pub fn from_corners(top_left: Point, bottom_right: Point) -> Self;
    pub fn contains(&self, point: Point) -> bool;
    pub fn center(&self) -> Point;
    pub fn area(&self) -> f32;
}
```
