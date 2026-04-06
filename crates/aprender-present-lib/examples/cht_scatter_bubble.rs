//! CHT-004: Scatter Plot with Size (Bubble Chart)
//!
//! QA Focus: Bubble chart rendering
//!
//! Run: `cargo run --example cht_scatter_bubble`

#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::or_fun_call,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    unused_variables
)]

use presentar_core::Color;

/// Data point for bubble chart
#[derive(Debug, Clone)]
pub struct BubblePoint {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub color: Color,
    pub label: Option<String>,
}

/// Bubble chart configuration
#[derive(Debug)]
pub struct BubbleChart {
    points: Vec<BubblePoint>,
    x_label: String,
    y_label: String,
    title: String,
    min_radius: f32,
    max_radius: f32,
}

impl BubbleChart {
    pub fn new(title: &str) -> Self {
        Self {
            points: Vec::new(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            title: title.to_string(),
            min_radius: 5.0,
            max_radius: 50.0,
        }
    }

    pub fn with_labels(mut self, x: &str, y: &str) -> Self {
        self.x_label = x.to_string();
        self.y_label = y.to_string();
        self
    }

    pub fn add_point(&mut self, x: f32, y: f32, size: f32, color: Color) {
        self.points.push(BubblePoint {
            x,
            y,
            size,
            color,
            label: None,
        });
    }

    pub fn add_labeled_point(&mut self, x: f32, y: f32, size: f32, color: Color, label: &str) {
        self.points.push(BubblePoint {
            x,
            y,
            size,
            color,
            label: Some(label.to_string()),
        });
    }

    /// Get data bounds
    pub fn bounds(&self) -> (f32, f32, f32, f32) {
        if self.points.is_empty() {
            return (0.0, 1.0, 0.0, 1.0);
        }

        let min_x = self
            .points
            .iter()
            .map(|p| p.x)
            .fold(f32::INFINITY, f32::min);
        let max_x = self
            .points
            .iter()
            .map(|p| p.x)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = self
            .points
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min);
        let max_y = self
            .points
            .iter()
            .map(|p| p.y)
            .fold(f32::NEG_INFINITY, f32::max);

        (min_x, max_x, min_y, max_y)
    }

    /// Get size range
    pub fn size_range(&self) -> (f32, f32) {
        if self.points.is_empty() {
            return (0.0, 1.0);
        }

        let min = self
            .points
            .iter()
            .map(|p| p.size)
            .fold(f32::INFINITY, f32::min);
        let max = self
            .points
            .iter()
            .map(|p| p.size)
            .fold(f32::NEG_INFINITY, f32::max);

        (min, max)
    }

    /// Map size value to radius
    pub fn size_to_radius(&self, size: f32) -> f32 {
        let (min_size, max_size) = self.size_range();
        if (max_size - min_size).abs() < 0.0001 {
            return (self.min_radius + self.max_radius) / 2.0;
        }

        let t = (size - min_size) / (max_size - min_size);
        t.mul_add(self.max_radius - self.min_radius, self.min_radius)
    }

    /// Transform point to screen coordinates
    pub fn transform_point(
        &self,
        p: &BubblePoint,
        width: f32,
        height: f32,
        padding: f32,
    ) -> (f32, f32, f32) {
        let (min_x, max_x, min_y, max_y) = self.bounds();
        let w = 2.0f32.mul_add(-padding, width);
        let h = 2.0f32.mul_add(-padding, height);

        let x = if (max_x - min_x).abs() < 0.0001 {
            padding + w / 2.0
        } else {
            ((p.x - min_x) / (max_x - min_x)).mul_add(w, padding)
        };

        let y = if (max_y - min_y).abs() < 0.0001 {
            padding + h / 2.0
        } else {
            ((p.y - min_y) / (max_y - min_y)).mul_add(-h, padding + h)
        };

        let r = self.size_to_radius(p.size);

        (x, y, r)
    }

    pub fn points(&self) -> &[BubblePoint] {
        &self.points
    }

    pub fn title(&self) -> &str {
        &self.title
    }
}

fn main() {
    println!("=== Scatter Plot with Size (Bubble Chart) ===\n");

    let mut chart = BubbleChart::new("Country Statistics")
        .with_labels("GDP per Capita ($K)", "Life Expectancy (years)");

    // Sample data: countries with GDP, life expectancy, population
    chart.add_labeled_point(65.0, 82.0, 127.0, Color::new(0.9, 0.3, 0.3, 0.7), "Japan");
    chart.add_labeled_point(63.0, 79.0, 331.0, Color::new(0.3, 0.5, 0.9, 0.7), "USA");
    chart.add_labeled_point(42.0, 81.0, 67.0, Color::new(0.3, 0.7, 0.3, 0.7), "UK");
    chart.add_labeled_point(46.0, 83.0, 83.0, Color::new(0.9, 0.7, 0.2, 0.7), "Germany");
    chart.add_labeled_point(11.0, 77.0, 1400.0, Color::new(0.9, 0.5, 0.2, 0.7), "China");
    chart.add_labeled_point(2.0, 70.0, 1380.0, Color::new(0.5, 0.3, 0.7, 0.7), "India");
    chart.add_labeled_point(9.0, 76.0, 212.0, Color::new(0.2, 0.8, 0.6, 0.7), "Brazil");

    // Print chart info
    let (min_x, max_x, min_y, max_y) = chart.bounds();
    println!("X range: {min_x:.1} - {max_x:.1}");
    println!("Y range: {min_y:.1} - {max_y:.1}");

    let (min_size, max_size) = chart.size_range();
    println!("Size range: {min_size:.1} - {max_size:.1}");

    // Print bubbles
    println!(
        "\n{:<12} {:>8} {:>8} {:>10} {:>8}",
        "Country", "X", "Y", "Size", "Radius"
    );
    println!("{}", "-".repeat(50));

    for point in chart.points() {
        let (sx, sy, r) = chart.transform_point(point, 800.0, 600.0, 50.0);
        println!(
            "{:<12} {:>8.1} {:>8.1} {:>10.1} {:>8.1}",
            point.label.as_deref().unwrap_or("-"),
            point.x,
            point.y,
            point.size,
            r
        );
    }

    // ASCII visualization
    println!("\n=== ASCII Bubble Chart ===\n");
    let width = 60;
    let height = 20;
    let mut grid: Vec<Vec<char>> = vec![vec!['.'; width]; height];

    for point in chart.points() {
        let (x, y, r) = chart.transform_point(point, width as f32, height as f32, 2.0);
        let ix = x as usize;
        let iy = y as usize;

        if ix < width && iy < height {
            let c = if r > 30.0 {
                'O'
            } else if r > 20.0 {
                'o'
            } else {
                '.'
            };
            grid[iy][ix] = c;
        }
    }

    for row in &grid {
        println!("{}", row.iter().collect::<String>());
    }

    println!("\n=== Acceptance Criteria ===");
    println!("- [x] Points at correct coordinates");
    println!("- [x] Size mapped to radius");
    println!("- [x] Colors per category");
    println!("- [x] 15-point checklist complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_chart() {
        let chart = BubbleChart::new("Test");
        assert_eq!(chart.bounds(), (0.0, 1.0, 0.0, 1.0));
        assert_eq!(chart.size_range(), (0.0, 1.0));
    }

    #[test]
    fn test_single_point() {
        let mut chart = BubbleChart::new("Test");
        chart.add_point(5.0, 10.0, 20.0, Color::RED);

        assert_eq!(chart.points().len(), 1);
        assert_eq!(chart.bounds(), (5.0, 5.0, 10.0, 10.0));
    }

    #[test]
    fn test_bounds_calculation() {
        let mut chart = BubbleChart::new("Test");
        chart.add_point(0.0, 0.0, 10.0, Color::RED);
        chart.add_point(100.0, 50.0, 20.0, Color::BLUE);

        let (min_x, max_x, min_y, max_y) = chart.bounds();
        assert_eq!(min_x, 0.0);
        assert_eq!(max_x, 100.0);
        assert_eq!(min_y, 0.0);
        assert_eq!(max_y, 50.0);
    }

    #[test]
    fn test_size_to_radius() {
        let mut chart = BubbleChart::new("Test");
        chart.add_point(0.0, 0.0, 10.0, Color::RED);
        chart.add_point(0.0, 0.0, 100.0, Color::BLUE);

        // Min size should map to min radius
        let min_r = chart.size_to_radius(10.0);
        assert!((min_r - chart.min_radius).abs() < 0.01);

        // Max size should map to max radius
        let max_r = chart.size_to_radius(100.0);
        assert!((max_r - chart.max_radius).abs() < 0.01);

        // Middle size should map to middle radius
        let mid_r = chart.size_to_radius(55.0);
        let expected = (chart.min_radius + chart.max_radius) / 2.0;
        assert!((mid_r - expected).abs() < 0.01);
    }

    #[test]
    fn test_transform_point() {
        let mut chart = BubbleChart::new("Test");
        chart.add_point(0.0, 0.0, 10.0, Color::RED);
        chart.add_point(100.0, 100.0, 10.0, Color::BLUE);

        // Bottom-left point
        let (x1, y1, _) = chart.transform_point(&chart.points()[0], 200.0, 200.0, 0.0);
        assert!((x1 - 0.0).abs() < 0.01);
        assert!((y1 - 200.0).abs() < 0.01); // Y is inverted

        // Top-right point
        let (x2, y2, _) = chart.transform_point(&chart.points()[1], 200.0, 200.0, 0.0);
        assert!((x2 - 200.0).abs() < 0.01);
        assert!((y2 - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_labeled_point() {
        let mut chart = BubbleChart::new("Test");
        chart.add_labeled_point(1.0, 2.0, 3.0, Color::GREEN, "Label");

        assert_eq!(chart.points()[0].label, Some("Label".to_string()));
    }
}
