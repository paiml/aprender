use super::*;

#[test]
fn test_point_distance() {
    let p1 = Point::new(0.0, 0.0);
    let p2 = Point::new(3.0, 4.0);
    assert!((p1.distance(&p2) - 5.0).abs() < 0.001);
}

#[test]
fn test_point_midpoint() {
    let p1 = Point::new(0.0, 0.0);
    let p2 = Point::new(10.0, 10.0);
    let mid = p1.midpoint(&p2);
    assert_eq!(mid.x, 5.0);
    assert_eq!(mid.y, 5.0);
}

#[test]
fn test_rect_creation() {
    let rect = Rect::new(10.0, 20.0, 100.0, 50.0);
    assert_eq!(rect.position.x, 10.0);
    assert_eq!(rect.position.y, 20.0);
    assert_eq!(rect.size.width, 100.0);
    assert_eq!(rect.size.height, 50.0);
}

#[test]
fn test_rect_center() {
    let rect = Rect::new(0.0, 0.0, 100.0, 50.0);
    let center = rect.center();
    assert_eq!(center.x, 50.0);
    assert_eq!(center.y, 25.0);
}

#[test]
fn test_rect_contains() {
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    assert!(rect.contains(&Point::new(50.0, 50.0)));
    assert!(!rect.contains(&Point::new(150.0, 50.0)));
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
fn test_rect_to_svg() {
    let rect = Rect::new(10.0, 20.0, 100.0, 50.0)
        .with_fill(Color::rgb(255, 0, 0))
        .with_stroke(Color::rgb(0, 0, 0), 2.0);

    let svg = rect.to_svg();
    assert!(svg.contains("x=\"10\""));
    assert!(svg.contains("fill=\"#FF0000\""));
    assert!(svg.contains("stroke=\"#000000\""));
}

#[test]
fn test_circle_creation() {
    let circle = Circle::new(50.0, 50.0, 25.0);
    assert_eq!(circle.center.x, 50.0);
    assert_eq!(circle.radius, 25.0);
}

#[test]
fn test_circle_contains() {
    let circle = Circle::new(50.0, 50.0, 25.0);
    assert!(circle.contains(&Point::new(50.0, 50.0)));
    assert!(circle.contains(&Point::new(60.0, 50.0)));
    assert!(!circle.contains(&Point::new(100.0, 100.0)));
}

#[test]
fn test_circle_to_svg() {
    let circle = Circle::new(50.0, 50.0, 25.0).with_fill(Color::rgb(0, 255, 0));

    let svg = circle.to_svg();
    assert!(svg.contains("cx=\"50\""));
    assert!(svg.contains("r=\"25\""));
    assert!(svg.contains("fill=\"#00FF00\""));
}

#[test]
fn test_line_creation() {
    let line = Line::new(0.0, 0.0, 100.0, 100.0);
    assert!((line.length() - 141.42).abs() < 0.1);
}

#[test]
fn test_path_builder() {
    let path = Path::new().move_to(0.0, 0.0).line_to(100.0, 0.0).line_to(100.0, 100.0).close();

    let data = path.to_path_data();
    assert!(data.contains("M 0 0"));
    assert!(data.contains("L 100 0"));
    assert!(data.ends_with('Z'));
}

#[test]
fn test_text_creation() {
    let text = Text::new(10.0, 20.0, "Hello");
    assert_eq!(text.content, "Hello");
    assert_eq!(text.position.x, 10.0);
}

#[test]
fn test_text_escaping() {
    let text = Text::new(0.0, 0.0, "<script>");
    let svg = text.to_svg();
    assert!(svg.contains("&lt;script&gt;"));
}

#[test]
fn test_arrow_marker() {
    let arrow = ArrowMarker::new("arrow1", Color::rgb(0, 0, 0));
    let svg = arrow.to_svg_def();
    assert!(svg.contains("id=\"arrow1\""));
    assert!(svg.contains("fill=\"#000000\""));
}

#[test]
fn test_point_new() {
    let p = Point::new(5.0, 10.0);
    assert_eq!(p.x, 5.0);
    assert_eq!(p.y, 10.0);
}

#[test]
fn test_point_default() {
    let p = Point::default();
    assert_eq!(p.x, 0.0);
    assert_eq!(p.y, 0.0);
}

#[test]
fn test_size_new() {
    let s = Size::new(100.0, 50.0);
    assert_eq!(s.width, 100.0);
    assert_eq!(s.height, 50.0);
}

#[test]
fn test_size_area() {
    let s = Size::new(10.0, 5.0);
    assert_eq!(s.area(), 50.0);
}

#[test]
fn test_size_default() {
    let s = Size::default();
    assert_eq!(s.width, 0.0);
    assert_eq!(s.height, 0.0);
}

#[test]
fn test_rect_right_bottom() {
    let rect = Rect::new(10.0, 20.0, 100.0, 50.0);
    assert_eq!(rect.right(), 110.0);
    assert_eq!(rect.bottom(), 70.0);
}

#[test]
fn test_rect_default() {
    let rect = Rect::default();
    assert_eq!(rect.position.x, 0.0);
    assert_eq!(rect.size.width, 100.0);
}

#[test]
fn test_rect_with_radius() {
    let rect = Rect::new(0.0, 0.0, 50.0, 50.0).with_radius(8.0);
    assert_eq!(rect.corner_radius, 8.0);
    let svg = rect.to_svg();
    assert!(svg.contains("rx=\"8\""));
}

#[test]
fn test_circle_bounds() {
    let circle = Circle::new(50.0, 50.0, 25.0);
    let bounds = circle.bounds();
    assert_eq!(bounds.position.x, 25.0);
    assert_eq!(bounds.position.y, 25.0);
    assert_eq!(bounds.size.width, 50.0);
    assert_eq!(bounds.size.height, 50.0);
}

#[test]
fn test_circle_intersects() {
    let c1 = Circle::new(0.0, 0.0, 10.0);
    let c2 = Circle::new(15.0, 0.0, 10.0);
    let c3 = Circle::new(50.0, 0.0, 10.0);
    assert!(c1.intersects(&c2)); // Overlapping
    assert!(!c1.intersects(&c3)); // Not touching
}

#[test]
fn test_circle_with_stroke() {
    let circle = Circle::new(50.0, 50.0, 25.0).with_stroke(Color::rgb(0, 0, 255), 3.0);
    let svg = circle.to_svg();
    assert!(svg.contains("stroke=\"#0000FF\""));
    assert!(svg.contains("stroke-width=\"3\""));
}

#[test]
fn test_circle_default() {
    let circle = Circle::default();
    assert_eq!(circle.center.x, 50.0);
    assert_eq!(circle.radius, 25.0);
}

#[test]
fn test_line_midpoint() {
    let line = Line::new(0.0, 0.0, 100.0, 100.0);
    let mid = line.midpoint();
    assert_eq!(mid.x, 50.0);
    assert_eq!(mid.y, 50.0);
}

#[test]
fn test_line_with_stroke() {
    let line = Line::new(0.0, 0.0, 100.0, 0.0).with_stroke(Color::rgb(255, 0, 0));
    assert_eq!(line.stroke, Color::rgb(255, 0, 0));
}

#[test]
fn test_line_with_stroke_width() {
    let line = Line::new(0.0, 0.0, 100.0, 0.0).with_stroke_width(5.0);
    assert_eq!(line.stroke_width, 5.0);
}

#[test]
fn test_line_with_dash() {
    let line = Line::new(0.0, 0.0, 100.0, 0.0).with_dash("5,5");
    let svg = line.to_svg();
    assert!(svg.contains("stroke-dasharray=\"5,5\""));
}

#[test]
fn test_line_to_svg() {
    let line = Line::new(10.0, 20.0, 30.0, 40.0);
    let svg = line.to_svg();
    assert!(svg.contains("x1=\"10\""));
    assert!(svg.contains("y1=\"20\""));
    assert!(svg.contains("x2=\"30\""));
    assert!(svg.contains("y2=\"40\""));
}

#[test]
fn test_path_quad_curve() {
    let path = Path::new().move_to(0.0, 0.0).quad_to(50.0, 50.0, 100.0, 0.0);
    let data = path.to_path_data();
    assert!(data.contains("Q 50 50"));
}

#[test]
fn test_path_cubic_curve() {
    let path = Path::new().move_to(0.0, 0.0).cubic_to(25.0, 50.0, 75.0, 50.0, 100.0, 0.0);
    let data = path.to_path_data();
    assert!(data.contains("C 25 50"));
}

#[test]
fn test_path_with_fill_stroke() {
    let path = Path::new()
        .move_to(0.0, 0.0)
        .line_to(100.0, 100.0)
        .with_fill(Color::rgb(255, 0, 0))
        .with_stroke(Color::rgb(0, 0, 0), 2.0);
    let svg = path.to_svg();
    assert!(svg.contains("fill=\"#FF0000\""));
    assert!(svg.contains("stroke=\"#000000\""));
}

#[test]
fn test_path_default() {
    let path = Path::default();
    assert!(path.commands.is_empty());
}

#[test]
fn test_text_with_style() {
    use crate::oracle::svg::typography::{FontWeight, TextStyle};
    let style = TextStyle::new(16.0, FontWeight::Bold);
    let text = Text::new(0.0, 0.0, "Hello").with_style(style);
    let svg = text.to_svg();
    assert!(svg.contains("font-weight=\"700\""));
}

#[test]
fn test_arrow_marker_with_size() {
    let arrow = ArrowMarker::new("a1", Color::rgb(0, 0, 0)).with_size(20.0);
    assert_eq!(arrow.size, 20.0);
    let svg = arrow.to_svg_def();
    assert!(svg.contains("markerWidth=\"20\""));
}

#[test]
fn test_html_escape() {
    let text = Text::new(0.0, 0.0, "a & b < c > d \"e\" 'f'");
    let svg = text.to_svg();
    assert!(svg.contains("&amp;"));
    assert!(svg.contains("&lt;"));
    assert!(svg.contains("&gt;"));
    assert!(svg.contains("&quot;"));
    assert!(svg.contains("&#39;"));
}

#[test]
fn test_rect_no_fill() {
    let rect = Rect::new(0.0, 0.0, 50.0, 50.0);
    let svg = rect.to_svg();
    assert!(svg.contains("fill=\"none\""));
}

#[test]
fn test_circle_no_fill() {
    let circle = Circle::new(50.0, 50.0, 25.0);
    let svg = circle.to_svg();
    assert!(svg.contains("fill=\"none\""));
}

#[test]
fn test_path_no_fill() {
    let path = Path::new().move_to(0.0, 0.0).line_to(100.0, 100.0);
    let svg = path.to_svg();
    assert!(svg.contains("fill=\"none\""));
}

#[test]
fn test_rect_with_fill_style() {
    let rect = Rect::new(0.0, 0.0, 50.0, 50.0).with_fill(Color::rgb(255, 0, 0));
    let svg = rect.to_svg();
    assert!(svg.contains("fill=\"#FF0000\""));
}

#[test]
fn test_rect_with_stroke_style() {
    let rect = Rect::new(0.0, 0.0, 50.0, 50.0).with_stroke(Color::rgb(0, 255, 0), 3.0);
    let svg = rect.to_svg();
    assert!(svg.contains("stroke=\"#00FF00\""));
    assert!(svg.contains("stroke-width=\"3\""));
}

#[test]
fn test_circle_with_fill_color() {
    let circle = Circle::new(50.0, 50.0, 25.0).with_fill(Color::rgb(0, 0, 255));
    let svg = circle.to_svg();
    assert!(svg.contains("fill=\"#0000FF\""));
}

#[test]
fn test_line_length_calc() {
    let line = Line::new(0.0, 0.0, 3.0, 4.0);
    assert!((line.length() - 5.0).abs() < 0.0001);
}

#[test]
fn test_line_defaults() {
    let line = Line::new(0.0, 0.0, 100.0, 100.0);
    assert_eq!(line.stroke_width, 1.0);
    assert!(line.dash_array.is_none());
}

#[test]
fn test_path_close_command() {
    let path = Path::new().move_to(0.0, 0.0).line_to(100.0, 0.0).line_to(100.0, 100.0).close();
    let data = path.to_path_data();
    assert!(data.contains('Z'));
}

#[test]
fn test_point_equality_check() {
    let p1 = Point::new(1.0, 2.0);
    let p2 = Point::new(1.0, 2.0);
    let p3 = Point::new(3.0, 4.0);
    assert_eq!(p1, p2);
    assert_ne!(p1, p3);
}

#[test]
fn test_size_equality_check() {
    let s1 = Size::new(10.0, 20.0);
    let s2 = Size::new(10.0, 20.0);
    assert_eq!(s1, s2);
}

#[test]
fn test_rect_equality() {
    let r1 = Rect::new(0.0, 0.0, 100.0, 100.0);
    let r2 = Rect::new(0.0, 0.0, 100.0, 100.0);
    assert_eq!(r1, r2);
}

#[test]
fn test_circle_equality() {
    let c1 = Circle::new(50.0, 50.0, 25.0);
    let c2 = Circle::new(50.0, 50.0, 25.0);
    assert_eq!(c1, c2);
}

#[test]
fn test_line_equality() {
    let l1 = Line::new(0.0, 0.0, 100.0, 100.0);
    let l2 = Line::new(0.0, 0.0, 100.0, 100.0);
    assert_eq!(l1, l2);
}
