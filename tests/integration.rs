//! Integration tests for the Presentar UI framework.
//!
//! These tests verify the public API across all presentar crates.

use presentar_core::{Color, Constraints, Point, Rect, Size};

#[test]
fn test_size_operations() {
    let size = Size::new(100.0, 50.0);
    assert_eq!(size.width, 100.0);
    assert_eq!(size.height, 50.0);

    let scaled = Size::new(size.width * 2.0, size.height * 2.0);
    assert_eq!(scaled.width, 200.0);
    assert_eq!(scaled.height, 100.0);
}

#[test]
fn test_point_operations() {
    let p1 = Point::new(10.0, 20.0);
    let p2 = Point::new(5.0, 10.0);

    assert_eq!(p1.x, 10.0);
    assert_eq!(p1.y, 20.0);
    assert_eq!(p2.x, 5.0);
    assert_eq!(p2.y, 10.0);
}

#[test]
fn test_rect_creation() {
    let rect = Rect::new(Point::new(0.0, 0.0), Size::new(100.0, 50.0));
    assert_eq!(rect.origin.x, 0.0);
    assert_eq!(rect.origin.y, 0.0);
    assert_eq!(rect.size.width, 100.0);
    assert_eq!(rect.size.height, 50.0);
}

#[test]
fn test_color_constants() {
    let red = Color::RED;
    assert_eq!(red.r, 1.0);
    assert_eq!(red.g, 0.0);
    assert_eq!(red.b, 0.0);

    let white = Color::WHITE;
    assert_eq!(white.r, 1.0);
    assert_eq!(white.g, 1.0);
    assert_eq!(white.b, 1.0);

    let black = Color::BLACK;
    assert_eq!(black.r, 0.0);
    assert_eq!(black.g, 0.0);
    assert_eq!(black.b, 0.0);
}

#[test]
fn test_color_rgba() {
    let color = Color::rgba(0.5, 0.6, 0.7, 0.8);
    assert!((color.r - 0.5).abs() < 0.001);
    assert!((color.g - 0.6).abs() < 0.001);
    assert!((color.b - 0.7).abs() < 0.001);
    assert!((color.a - 0.8).abs() < 0.001);
}

#[test]
fn test_constraints_constrain() {
    let constraints = Constraints::new(10.0, 200.0, 10.0, 100.0);

    // Test constraining within bounds
    let size = Size::new(50.0, 50.0);
    let bounded = constraints.constrain(size);
    assert_eq!(bounded.width, 50.0);
    assert_eq!(bounded.height, 50.0);

    // Test constraining larger size
    let large = Size::new(300.0, 200.0);
    let bounded_large = constraints.constrain(large);
    assert_eq!(bounded_large.width, 200.0);
    assert_eq!(bounded_large.height, 100.0);

    // Test constraining smaller size
    let small = Size::new(5.0, 5.0);
    let bounded_small = constraints.constrain(small);
    assert_eq!(bounded_small.width, 10.0);
    assert_eq!(bounded_small.height, 10.0);
}

#[test]
fn test_constraints_tight() {
    let size = Size::new(100.0, 50.0);
    let tight = Constraints::tight(size);

    assert_eq!(tight.min_width, 100.0);
    assert_eq!(tight.max_width, 100.0);
    assert_eq!(tight.min_height, 50.0);
    assert_eq!(tight.max_height, 50.0);
}

#[test]
fn test_constraints_unbounded() {
    let unbounded = Constraints::unbounded();

    assert_eq!(unbounded.min_width, 0.0);
    assert_eq!(unbounded.min_height, 0.0);
    assert!(unbounded.max_width.is_infinite());
    assert!(unbounded.max_height.is_infinite());
}
