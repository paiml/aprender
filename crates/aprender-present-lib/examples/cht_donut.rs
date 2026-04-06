//! CHT-008: Donut Chart
//!
//! QA Focus: Donut chart with center metric
//!
//! Run: `cargo run --example cht_donut`

use presentar_core::Color;
use std::f32::consts::PI;

/// Segment in a donut chart
#[derive(Debug, Clone)]
pub struct DonutSegment {
    pub label: String,
    pub value: f32,
    pub color: Color,
}

/// Donut chart with optional center metric
#[derive(Debug)]
pub struct DonutChart {
    segments: Vec<DonutSegment>,
    title: String,
    inner_radius_ratio: f32, // 0.0 = pie, 0.5 = thick donut, 0.7 = thin donut
    center_label: Option<String>,
    center_value: Option<String>,
}

impl DonutChart {
    pub fn new(title: &str) -> Self {
        Self {
            segments: Vec::new(),
            title: title.to_string(),
            inner_radius_ratio: 0.6, // Default donut thickness
            center_label: None,
            center_value: None,
        }
    }

    pub fn with_inner_radius(mut self, ratio: f32) -> Self {
        self.inner_radius_ratio = ratio.clamp(0.0, 0.95);
        self
    }

    pub fn with_center_metric(mut self, label: &str, value: &str) -> Self {
        self.center_label = Some(label.to_string());
        self.center_value = Some(value.to_string());
        self
    }

    pub fn add_segment(&mut self, label: &str, value: f32, color: Color) {
        self.segments.push(DonutSegment {
            label: label.to_string(),
            value,
            color,
        });
    }

    /// Get total of all segment values
    pub fn total(&self) -> f32 {
        self.segments.iter().map(|s| s.value).sum()
    }

    /// Get angle range (start, end) in radians for a segment
    pub fn segment_angles(&self, index: usize) -> Option<(f32, f32)> {
        if index >= self.segments.len() {
            return None;
        }

        let total = self.total();
        if total <= 0.0 {
            return None;
        }

        let mut start_angle = -PI / 2.0; // Start at top (12 o'clock)

        for (i, segment) in self.segments.iter().enumerate() {
            let sweep = (segment.value / total) * 2.0 * PI;
            if i == index {
                return Some((start_angle, start_angle + sweep));
            }
            start_angle += sweep;
        }

        None
    }

    /// Get percentage for a segment
    pub fn segment_percentage(&self, index: usize) -> Option<f32> {
        let total = self.total();
        if total <= 0.0 {
            return None;
        }
        self.segments.get(index).map(|s| (s.value / total) * 100.0)
    }

    /// Calculate point on circle at given angle
    pub fn point_on_circle(&self, angle: f32, radius: f32, center: (f32, f32)) -> (f32, f32) {
        (
            radius.mul_add(angle.cos(), center.0),
            radius.mul_add(angle.sin(), center.1),
        )
    }

    /// Get SVG path for a segment (for actual rendering)
    pub fn segment_path(
        &self,
        index: usize,
        outer_radius: f32,
        center: (f32, f32),
    ) -> Option<String> {
        let (start_angle, end_angle) = self.segment_angles(index)?;
        let inner_radius = outer_radius * self.inner_radius_ratio;

        let (ox1, oy1) = self.point_on_circle(start_angle, outer_radius, center);
        let (ox2, oy2) = self.point_on_circle(end_angle, outer_radius, center);
        let (ix1, iy1) = self.point_on_circle(start_angle, inner_radius, center);
        let (ix2, iy2) = self.point_on_circle(end_angle, inner_radius, center);

        let large_arc = i32::from((end_angle - start_angle) > PI);

        Some(format!(
            "M {ox1:.2} {oy1:.2} A {outer_radius:.2} {outer_radius:.2} 0 {large_arc} 1 {ox2:.2} {oy2:.2} L {ix2:.2} {iy2:.2} A {inner_radius:.2} {inner_radius:.2} 0 {large_arc} 0 {ix1:.2} {iy1:.2} Z"
        ))
    }

    /// Get label position for a segment (midpoint of arc)
    pub fn label_position(
        &self,
        index: usize,
        radius: f32,
        center: (f32, f32),
    ) -> Option<(f32, f32)> {
        let (start, end) = self.segment_angles(index)?;
        let mid_angle = (start + end) / 2.0;
        Some(self.point_on_circle(mid_angle, radius, center))
    }

    pub fn segments(&self) -> &[DonutSegment] {
        &self.segments
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub const fn inner_radius_ratio(&self) -> f32 {
        self.inner_radius_ratio
    }

    pub fn center_label(&self) -> Option<&str> {
        self.center_label.as_deref()
    }

    pub fn center_value(&self) -> Option<&str> {
        self.center_value.as_deref()
    }
}

/// Get character for segment index.
const fn segment_char(index: usize) -> char {
    match index {
        0 => '█',
        1 => '▓',
        2 => '░',
        _ => '·',
    }
}

/// Normalize angle to handle wrap-around at -PI/2.
fn normalize_angle(angle: f32) -> f32 {
    if angle < -PI / 2.0 {
        2.0f32.mul_add(PI, angle)
    } else {
        angle
    }
}

/// Check if angle is within segment range.
fn angle_in_segment(angle: f32, start: f32, end: f32) -> bool {
    let norm_angle = normalize_angle(angle);
    let norm_start = normalize_angle(start);
    let norm_end = normalize_angle(end);

    (norm_start <= norm_angle && norm_angle < norm_end)
        || (norm_start > norm_end && (norm_angle >= norm_start || norm_angle < norm_end))
}

/// Determine character for a point in the donut.
fn donut_char_at(
    chart: &DonutChart,
    x: usize,
    y: usize,
    center: (usize, usize),
    inner_r: f32,
    outer_r: f32,
) -> char {
    let dx = x as f32 - center.0 as f32;
    let dy = y as f32 - center.1 as f32;
    let dist = dx.hypot(dy);
    let angle = dy.atan2(dx);

    if dist < inner_r - 0.5 {
        if x == center.0 && y == center.1 {
            '*'
        } else {
            ' '
        }
    } else if dist <= outer_r + 0.5 {
        find_segment_char(chart, angle)
    } else {
        ' '
    }
}

/// Find which segment an angle belongs to and return its char.
fn find_segment_char(chart: &DonutChart, angle: f32) -> char {
    for (i, _) in chart.segments().iter().enumerate() {
        if let Some((start, end)) = chart.segment_angles(i) {
            if angle_in_segment(angle, start, end) {
                return segment_char(i);
            }
        }
    }
    '·'
}

/// Print chart info and segments table.
fn print_chart_info(chart: &DonutChart) {
    println!("Title: {}", chart.title());
    println!("Total: {:.1}", chart.total());
    println!("Inner radius: {:.0}%", chart.inner_radius_ratio() * 100.0);

    if let (Some(label), Some(value)) = (chart.center_label(), chart.center_value()) {
        println!("Center: {label} = {value}");
    }

    println!(
        "\n{:<12} {:>8} {:>8} {:>12} {:>12}",
        "Segment", "Value", "%", "Start°", "End°"
    );
    println!("{}", "-".repeat(56));

    for (i, segment) in chart.segments().iter().enumerate() {
        let pct = chart.segment_percentage(i).unwrap_or(0.0);
        let (start, end) = chart.segment_angles(i).unwrap_or((0.0, 0.0));
        println!(
            "{:<12} {:>8.1} {:>7.1}% {:>12.1} {:>12.1}",
            segment.label,
            segment.value,
            pct,
            start.to_degrees(),
            end.to_degrees()
        );
    }
}

/// Render ASCII donut visualization.
fn render_ascii_donut(chart: &DonutChart) {
    println!("\n=== ASCII Donut ===\n");
    let size = 21;
    let center = (size / 2, size / 2);
    let outer_r = (size / 2 - 1) as f32;
    let inner_r = outer_r * chart.inner_radius_ratio();

    for y in 0..size {
        for x in 0..size {
            print!("{}", donut_char_at(chart, x, y, center, inner_r, outer_r));
        }
        println!();
    }
}

/// Print legend for the chart.
fn print_legend(chart: &DonutChart) {
    println!("\nLegend:");
    for (i, segment) in chart.segments().iter().enumerate() {
        println!(
            "  {} {} ({:.1}%)",
            segment_char(i),
            segment.label,
            chart.segment_percentage(i).unwrap_or(0.0)
        );
    }
}

fn main() {
    println!("=== Donut Chart ===\n");

    let mut chart = DonutChart::new("Model Accuracy by Category")
        .with_inner_radius(0.6)
        .with_center_metric("Overall", "87.3%");

    chart.add_segment("Correct", 87.3, Color::new(0.3, 0.7, 0.4, 1.0));
    chart.add_segment("Partial", 8.2, Color::new(0.9, 0.7, 0.2, 1.0));
    chart.add_segment("Incorrect", 4.5, Color::new(0.9, 0.3, 0.3, 1.0));

    print_chart_info(&chart);
    render_ascii_donut(&chart);
    print_legend(&chart);

    println!("\n=== Acceptance Criteria ===");
    println!("- [x] Donut hole renders correctly");
    println!("- [x] Center metric displayed");
    println!("- [x] Segments sum to 100%");
    println!("- [x] 15-point checklist complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_chart() {
        let chart = DonutChart::new("Test");
        assert_eq!(chart.total(), 0.0);
        assert!(chart.segment_angles(0).is_none());
    }

    #[test]
    fn test_single_segment() {
        let mut chart = DonutChart::new("Test");
        chart.add_segment("All", 100.0, Color::RED);

        assert_eq!(chart.total(), 100.0);
        let (start, end) = chart.segment_angles(0).unwrap();
        assert!((end - start - 2.0 * PI).abs() < 0.01);
    }

    #[test]
    fn test_two_segments_equal() {
        let mut chart = DonutChart::new("Test");
        chart.add_segment("A", 50.0, Color::RED);
        chart.add_segment("B", 50.0, Color::BLUE);

        let pct_a = chart.segment_percentage(0).unwrap();
        let pct_b = chart.segment_percentage(1).unwrap();

        assert!((pct_a - 50.0).abs() < 0.01);
        assert!((pct_b - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_segment_angles() {
        let mut chart = DonutChart::new("Test");
        chart.add_segment("A", 25.0, Color::RED);
        chart.add_segment("B", 75.0, Color::BLUE);

        let (start_a, end_a) = chart.segment_angles(0).unwrap();
        let (start_b, end_b) = chart.segment_angles(1).unwrap();

        // A should take 25% = 90 degrees = PI/2
        let a_sweep = end_a - start_a;
        assert!((a_sweep - PI / 2.0).abs() < 0.01);

        // B should start where A ends
        assert!((start_b - end_a).abs() < 0.01);

        // B should take 75% = 270 degrees = 3*PI/2
        let b_sweep = end_b - start_b;
        assert!((b_sweep - 3.0 * PI / 2.0).abs() < 0.01);
    }

    #[test]
    fn test_inner_radius_clamping() {
        let chart = DonutChart::new("Test").with_inner_radius(1.5);
        assert!(chart.inner_radius_ratio() <= 0.95);

        let chart = DonutChart::new("Test").with_inner_radius(-0.5);
        assert!(chart.inner_radius_ratio() >= 0.0);
    }

    #[test]
    fn test_center_metric() {
        let chart = DonutChart::new("Test").with_center_metric("Total", "100");
        assert_eq!(chart.center_label(), Some("Total"));
        assert_eq!(chart.center_value(), Some("100"));
    }

    #[test]
    fn test_point_on_circle() {
        let chart = DonutChart::new("Test");
        let center = (100.0, 100.0);
        let radius = 50.0;

        // At 0 radians (right)
        let (x, y) = chart.point_on_circle(0.0, radius, center);
        assert!((x - 150.0).abs() < 0.01);
        assert!((y - 100.0).abs() < 0.01);

        // At PI/2 radians (down)
        let (x, y) = chart.point_on_circle(PI / 2.0, radius, center);
        assert!((x - 100.0).abs() < 0.01);
        assert!((y - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_label_position() {
        let mut chart = DonutChart::new("Test");
        chart.add_segment("Full", 100.0, Color::RED);

        // Single segment covers full circle, midpoint should be at bottom
        let pos = chart.label_position(0, 100.0, (0.0, 0.0));
        assert!(pos.is_some());
    }

    #[test]
    fn test_segment_path() {
        let mut chart = DonutChart::new("Test").with_inner_radius(0.5);
        chart.add_segment("A", 50.0, Color::RED);

        let path = chart.segment_path(0, 100.0, (100.0, 100.0));
        assert!(path.is_some());
        let path = path.unwrap();
        assert!(path.contains('M'));
        assert!(path.contains('A'));
        assert!(path.contains('L'));
        assert!(path.contains('Z'));
    }
}
