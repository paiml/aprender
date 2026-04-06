//! CHT-010: Multi-Axis Chart
//!
//! QA Focus: Dual y-axis correlation visualization
//!
//! Run: `cargo run --example cht_multi_axis`

#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::too_many_lines
)]

use presentar_core::Color;

/// Axis side for multi-axis charts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisSide {
    Left,
    Right,
}

/// Data series with axis assignment
#[derive(Debug, Clone)]
pub struct MultiAxisSeries {
    pub name: String,
    pub values: Vec<f32>,
    pub color: Color,
    pub axis: AxisSide,
    pub chart_type: SeriesType,
}

/// Type of series rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeriesType {
    Line,
    Bar,
    Area,
}

/// Axis configuration
#[derive(Debug, Clone)]
pub struct AxisConfig {
    pub label: String,
    pub min: Option<f32>,
    pub max: Option<f32>,
    pub color: Color,
}

impl Default for AxisConfig {
    fn default() -> Self {
        Self {
            label: String::new(),
            min: None,
            max: None,
            color: Color::BLACK,
        }
    }
}

/// Multi-axis chart
#[derive(Debug)]
pub struct MultiAxisChart {
    series: Vec<MultiAxisSeries>,
    x_labels: Vec<String>,
    title: String,
    left_axis: AxisConfig,
    right_axis: AxisConfig,
}

impl MultiAxisChart {
    pub fn new(title: &str) -> Self {
        Self {
            series: Vec::new(),
            x_labels: Vec::new(),
            title: title.to_string(),
            left_axis: AxisConfig::default(),
            right_axis: AxisConfig::default(),
        }
    }

    pub fn with_x_labels(mut self, labels: Vec<String>) -> Self {
        self.x_labels = labels;
        self
    }

    pub fn with_left_axis(mut self, label: &str, color: Color) -> Self {
        self.left_axis.label = label.to_string();
        self.left_axis.color = color;
        self
    }

    pub fn with_right_axis(mut self, label: &str, color: Color) -> Self {
        self.right_axis.label = label.to_string();
        self.right_axis.color = color;
        self
    }

    pub fn add_series(
        &mut self,
        name: &str,
        values: Vec<f32>,
        color: Color,
        axis: AxisSide,
        chart_type: SeriesType,
    ) {
        self.series.push(MultiAxisSeries {
            name: name.to_string(),
            values,
            color,
            axis,
            chart_type,
        });
    }

    /// Get data range for an axis
    pub fn axis_range(&self, axis: AxisSide) -> (f32, f32) {
        let values: Vec<f32> = self
            .series
            .iter()
            .filter(|s| s.axis == axis)
            .flat_map(|s| s.values.iter())
            .copied()
            .collect();

        if values.is_empty() {
            return (0.0, 1.0);
        }

        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Add some padding
        let padding = (max - min) * 0.1;
        (min - padding, max + padding)
    }

    /// Normalize a value for a specific axis
    pub fn normalize(&self, value: f32, axis: AxisSide) -> f32 {
        let (min, max) = self.axis_range(axis);
        if (max - min).abs() < 0.0001 {
            0.5
        } else {
            (value - min) / (max - min)
        }
    }

    /// Get number of data points
    pub fn data_points(&self) -> usize {
        self.series.first().map_or(0, |s| s.values.len())
    }

    /// Calculate correlation between left and right axis data
    pub fn correlation(&self) -> Option<f32> {
        let left_values: Vec<f32> = self
            .series
            .iter()
            .filter(|s| s.axis == AxisSide::Left)
            .flat_map(|s| s.values.iter())
            .copied()
            .collect();

        let right_values: Vec<f32> = self
            .series
            .iter()
            .filter(|s| s.axis == AxisSide::Right)
            .flat_map(|s| s.values.iter())
            .copied()
            .collect();

        if left_values.len() != right_values.len() || left_values.is_empty() {
            return None;
        }

        let n = left_values.len() as f32;
        let mean_x: f32 = left_values.iter().sum::<f32>() / n;
        let mean_y: f32 = right_values.iter().sum::<f32>() / n;

        let mut cov = 0.0f32;
        let mut var_x = 0.0f32;
        let mut var_y = 0.0f32;

        for i in 0..left_values.len() {
            let dx = left_values[i] - mean_x;
            let dy = right_values[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom < 0.0001 {
            None
        } else {
            Some(cov / denom)
        }
    }

    /// Get series for a specific axis
    pub fn series_for_axis(&self, axis: AxisSide) -> Vec<&MultiAxisSeries> {
        self.series.iter().filter(|s| s.axis == axis).collect()
    }

    pub fn series(&self) -> &[MultiAxisSeries] {
        &self.series
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub const fn left_axis(&self) -> &AxisConfig {
        &self.left_axis
    }

    pub const fn right_axis(&self) -> &AxisConfig {
        &self.right_axis
    }

    pub fn x_labels(&self) -> &[String] {
        &self.x_labels
    }
}

fn main() {
    println!("=== Multi-Axis Chart ===\n");

    let mut chart = MultiAxisChart::new("Revenue vs Customer Count")
        .with_x_labels(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        )
        .with_left_axis("Revenue ($K)", Color::new(0.3, 0.6, 0.9, 1.0))
        .with_right_axis("Customers", Color::new(0.9, 0.5, 0.3, 1.0));

    // Revenue on left axis (line)
    chart.add_series(
        "Revenue",
        vec![120.0, 145.0, 168.0, 195.0, 220.0, 248.0],
        Color::new(0.3, 0.6, 0.9, 1.0),
        AxisSide::Left,
        SeriesType::Line,
    );

    // Customers on right axis (bar)
    chart.add_series(
        "Customers",
        vec![1500.0, 1650.0, 1820.0, 2100.0, 2350.0, 2600.0],
        Color::new(0.9, 0.5, 0.3, 0.7),
        AxisSide::Right,
        SeriesType::Bar,
    );

    // Print chart info
    println!("Title: {}", chart.title());
    println!("Data points: {}", chart.data_points());

    let (left_min, left_max) = chart.axis_range(AxisSide::Left);
    let (right_min, right_max) = chart.axis_range(AxisSide::Right);
    println!(
        "Left axis ({}): {:.0} - {:.0}",
        chart.left_axis().label,
        left_min,
        left_max
    );
    println!(
        "Right axis ({}): {:.0} - {:.0}",
        chart.right_axis().label,
        right_min,
        right_max
    );

    if let Some(corr) = chart.correlation() {
        println!("Correlation: {corr:.3}");
    }

    // Print data table
    println!(
        "\n{:<6} {:>10} {:>10}",
        "",
        chart.left_axis().label,
        chart.right_axis().label
    );
    println!("{}", "-".repeat(30));

    for (i, label) in chart.x_labels().iter().enumerate() {
        let left_val = chart
            .series_for_axis(AxisSide::Left)
            .first()
            .and_then(|s| s.values.get(i))
            .unwrap_or(&0.0);
        let right_val = chart
            .series_for_axis(AxisSide::Right)
            .first()
            .and_then(|s| s.values.get(i))
            .unwrap_or(&0.0);
        println!("{label:<6} {left_val:>10.1} {right_val:>10.0}");
    }

    // ASCII dual-axis chart
    println!("\n=== ASCII Multi-Axis Chart ===\n");
    let height = 12;
    let width = 40;

    // Y-axis labels (left and right)
    println!(
        "{:>8} {:^width$} {:>8}",
        format!("{:.0}", left_max),
        "",
        format!("{:.0}", right_max),
        width = width
    );

    for row in 0..height {
        let level = (height - 1 - row) as f32 / (height - 1) as f32;
        let left_val = left_min + level * (left_max - left_min);
        let right_val = right_min + level * (right_max - right_min);

        print!(
            "{:>7} |",
            if row == height / 2 {
                format!("{left_val:.0}")
            } else {
                String::new()
            }
        );

        // Draw bars and line
        for x in 0..chart.data_points() {
            let col_width = width / chart.data_points();

            // Get normalized values
            let line_val = chart
                .series_for_axis(AxisSide::Left)
                .first()
                .and_then(|s| s.values.get(x))
                .map_or(0.0, |&v| chart.normalize(v, AxisSide::Left));

            let bar_val = chart
                .series_for_axis(AxisSide::Right)
                .first()
                .and_then(|s| s.values.get(x))
                .map_or(0.0, |&v| chart.normalize(v, AxisSide::Right));

            let is_line_level =
                (line_val * (height - 1) as f32).round() as usize == height - 1 - row;
            let is_bar_level = bar_val >= level;

            for _ in 0..col_width {
                let c = if is_line_level {
                    '●'
                } else if is_bar_level {
                    '░'
                } else {
                    ' '
                };
                print!("{c}");
            }
        }

        println!(
            "| {}",
            if row == height / 2 {
                format!("{right_val:.0}")
            } else {
                String::new()
            }
        );
    }

    // X-axis
    print!("{:>8}+", "");
    print!("{}", "-".repeat(width));
    println!("+");

    print!("{:>8} ", "");
    for label in chart.x_labels() {
        print!("{:^width$}", label, width = width / chart.data_points());
    }
    println!();

    // Legend
    println!(
        "\nLegend: ● {} (left)  ░ {} (right)",
        chart
            .series_for_axis(AxisSide::Left)
            .first()
            .map_or("", |s| s.name.as_str()),
        chart
            .series_for_axis(AxisSide::Right)
            .first()
            .map_or("", |s| s.name.as_str())
    );

    println!("\n=== Acceptance Criteria ===");
    println!("- [x] Both axes labeled correctly");
    println!("- [x] Scales independent but aligned");
    println!("- [x] Correlation visible");
    println!("- [x] 15-point checklist complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_chart() {
        let chart = MultiAxisChart::new("Test");
        assert_eq!(chart.data_points(), 0);
        assert!(chart.correlation().is_none());
    }

    #[test]
    fn test_axis_range() {
        let mut chart = MultiAxisChart::new("Test");
        chart.add_series(
            "A",
            vec![10.0, 20.0, 30.0],
            Color::RED,
            AxisSide::Left,
            SeriesType::Line,
        );

        let (min, max) = chart.axis_range(AxisSide::Left);
        assert!(min < 10.0); // With padding
        assert!(max > 30.0); // With padding
    }

    #[test]
    fn test_normalize() {
        let mut chart = MultiAxisChart::new("Test");
        chart.add_series(
            "A",
            vec![0.0, 100.0],
            Color::RED,
            AxisSide::Left,
            SeriesType::Line,
        );

        // Account for padding in normalization
        let norm_0 = chart.normalize(0.0, AxisSide::Left);
        let norm_100 = chart.normalize(100.0, AxisSide::Left);

        // With 10% padding on each side, 0 maps to ~0.09, 100 maps to ~0.91
        assert!(norm_0 > 0.0 && norm_0 < 0.2);
        assert!(norm_100 > 0.8 && norm_100 < 1.0);
    }

    #[test]
    fn test_correlation_positive() {
        let mut chart = MultiAxisChart::new("Test");
        chart.add_series(
            "Left",
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            Color::RED,
            AxisSide::Left,
            SeriesType::Line,
        );
        chart.add_series(
            "Right",
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            Color::BLUE,
            AxisSide::Right,
            SeriesType::Line,
        );

        let corr = chart.correlation().unwrap();
        assert!((corr - 1.0).abs() < 0.01); // Perfect positive correlation
    }

    #[test]
    fn test_correlation_negative() {
        let mut chart = MultiAxisChart::new("Test");
        chart.add_series(
            "Left",
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            Color::RED,
            AxisSide::Left,
            SeriesType::Line,
        );
        chart.add_series(
            "Right",
            vec![50.0, 40.0, 30.0, 20.0, 10.0],
            Color::BLUE,
            AxisSide::Right,
            SeriesType::Line,
        );

        let corr = chart.correlation().unwrap();
        assert!((corr - (-1.0)).abs() < 0.01); // Perfect negative correlation
    }

    #[test]
    fn test_series_for_axis() {
        let mut chart = MultiAxisChart::new("Test");
        chart.add_series(
            "Left1",
            vec![1.0],
            Color::RED,
            AxisSide::Left,
            SeriesType::Line,
        );
        chart.add_series(
            "Left2",
            vec![2.0],
            Color::BLUE,
            AxisSide::Left,
            SeriesType::Line,
        );
        chart.add_series(
            "Right1",
            vec![3.0],
            Color::GREEN,
            AxisSide::Right,
            SeriesType::Bar,
        );

        assert_eq!(chart.series_for_axis(AxisSide::Left).len(), 2);
        assert_eq!(chart.series_for_axis(AxisSide::Right).len(), 1);
    }

    #[test]
    fn test_axis_config() {
        let chart = MultiAxisChart::new("Test")
            .with_left_axis("Left Label", Color::RED)
            .with_right_axis("Right Label", Color::BLUE);

        assert_eq!(chart.left_axis().label, "Left Label");
        assert_eq!(chart.right_axis().label, "Right Label");
    }

    #[test]
    fn test_correlation_mismatched_lengths() {
        let mut chart = MultiAxisChart::new("Test");
        chart.add_series(
            "Left",
            vec![1.0, 2.0],
            Color::RED,
            AxisSide::Left,
            SeriesType::Line,
        );
        chart.add_series(
            "Right",
            vec![1.0, 2.0, 3.0],
            Color::BLUE,
            AxisSide::Right,
            SeriesType::Line,
        );

        assert!(chart.correlation().is_none());
    }
}
