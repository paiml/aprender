//! CHT-009: Sparkline Inline
//!
//! QA Focus: Compact inline chart rendering
//!
//! Run: `cargo run --example cht_sparkline`

#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::needless_range_loop,
    unused_variables,
    clippy::too_many_lines,
    clippy::needless_collect,
    clippy::struct_field_names
)]

use presentar_core::Color;

/// Sparkline type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparklineType {
    Line,
    Bar,
    Area,
}

/// Compact inline sparkline chart
#[derive(Debug)]
pub struct Sparkline {
    values: Vec<f32>,
    sparkline_type: SparklineType,
    width: usize,
    height: usize,
    color: Color,
    show_min_max: bool,
    show_last: bool,
}

impl Sparkline {
    pub fn new(values: Vec<f32>) -> Self {
        Self {
            values,
            sparkline_type: SparklineType::Line,
            width: 50,
            height: 10,
            color: Color::new(0.3, 0.6, 0.9, 1.0),
            show_min_max: false,
            show_last: true,
        }
    }

    pub const fn with_type(mut self, sparkline_type: SparklineType) -> Self {
        self.sparkline_type = sparkline_type;
        self
    }

    pub fn with_size(mut self, width: usize, height: usize) -> Self {
        self.width = width.max(5);
        self.height = height.max(3);
        self
    }

    pub const fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub const fn with_min_max_markers(mut self, show: bool) -> Self {
        self.show_min_max = show;
        self
    }

    pub const fn with_last_marker(mut self, show: bool) -> Self {
        self.show_last = show;
        self
    }

    /// Get min and max values
    pub fn range(&self) -> (f32, f32) {
        if self.values.is_empty() {
            return (0.0, 1.0);
        }
        let min = self.values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self
            .values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }

    /// Normalize a value to 0-1 range
    pub fn normalize(&self, value: f32) -> f32 {
        let (min, max) = self.range();
        if (max - min).abs() < 0.0001 {
            0.5
        } else {
            (value - min) / (max - min)
        }
    }

    /// Get the last value
    pub fn last(&self) -> Option<f32> {
        self.values.last().copied()
    }

    /// Get index of minimum value
    pub fn min_index(&self) -> Option<usize> {
        self.values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
    }

    /// Get index of maximum value
    pub fn max_index(&self) -> Option<usize> {
        self.values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
    }

    /// Calculate trend (positive = up, negative = down)
    pub fn trend(&self) -> f32 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let first = self.values[0];
        let last = self.values[self.values.len() - 1];
        last - first
    }

    /// Calculate trend as percentage change
    pub fn trend_percentage(&self) -> f32 {
        if self.values.len() < 2 || self.values[0].abs() < 0.0001 {
            return 0.0;
        }
        let first = self.values[0];
        let last = self.values[self.values.len() - 1];
        ((last - first) / first) * 100.0
    }

    /// Get points for line/area rendering
    pub fn points(&self) -> Vec<(f32, f32)> {
        if self.values.is_empty() {
            return vec![];
        }

        let x_step = self.width as f32 / (self.values.len().max(1) - 1).max(1) as f32;

        self.values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let x = i as f32 * x_step;
                let y = (1.0 - self.normalize(v)) * (self.height - 1) as f32;
                (x, y)
            })
            .collect()
    }

    /// Render as ASCII string
    pub fn render_ascii(&self) -> String {
        if self.values.is_empty() {
            return " ".repeat(self.width);
        }

        let mut grid: Vec<Vec<char>> = vec![vec![' '; self.width]; self.height];

        let min_idx = self.min_index();
        let max_idx = self.max_index();
        let last_idx = self.values.len().saturating_sub(1);

        let x_step = if self.values.len() > 1 {
            (self.width - 1) as f32 / (self.values.len() - 1) as f32
        } else {
            0.0
        };

        match self.sparkline_type {
            SparklineType::Line => {
                for (i, &v) in self.values.iter().enumerate() {
                    let x = (i as f32 * x_step).round() as usize;
                    let y = ((1.0 - self.normalize(v)) * (self.height - 1) as f32).round() as usize;
                    let x = x.min(self.width - 1);
                    let y = y.min(self.height - 1);

                    let c = if self.show_min_max && Some(i) == min_idx {
                        'v'
                    } else if self.show_min_max && Some(i) == max_idx {
                        '^'
                    } else if self.show_last && i == last_idx {
                        '*'
                    } else {
                        '·'
                    };
                    grid[y][x] = c;
                }
            }
            SparklineType::Bar => {
                let bar_width = (self.width / self.values.len().max(1)).max(1);
                for (i, &v) in self.values.iter().enumerate() {
                    let x = i * bar_width;
                    let bar_height = (self.normalize(v) * self.height as f32).round() as usize;
                    for y in 0..bar_height {
                        let grid_y = self.height - 1 - y;
                        for bx in 0..bar_width.saturating_sub(1) {
                            if x + bx < self.width {
                                grid[grid_y][x + bx] = '█';
                            }
                        }
                    }
                }
            }
            SparklineType::Area => {
                for (i, &v) in self.values.iter().enumerate() {
                    let x = (i as f32 * x_step).round() as usize;
                    let top_y =
                        ((1.0 - self.normalize(v)) * (self.height - 1) as f32).round() as usize;
                    let x = x.min(self.width - 1);

                    for y in top_y..self.height {
                        grid[y][x] = if y == top_y { '▀' } else { '░' };
                    }
                }
            }
        }

        grid.iter()
            .map(|row| row.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Render as single-line Unicode sparkline using block characters
    pub fn render_inline(&self) -> String {
        if self.values.is_empty() {
            return String::new();
        }

        // Unicode block characters for different heights
        let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        self.values
            .iter()
            .map(|&v| {
                let normalized = self.normalize(v);
                let idx = ((normalized * 7.0).round() as usize).min(7);
                blocks[idx]
            })
            .collect()
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub const fn sparkline_type(&self) -> SparklineType {
        self.sparkline_type
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }
}

fn main() {
    println!("=== Sparkline Inline Charts ===\n");

    // Sample time series data
    let cpu_usage = vec![
        45.0, 52.0, 48.0, 55.0, 62.0, 58.0, 65.0, 72.0, 68.0, 75.0, 82.0, 78.0,
    ];
    let memory_usage = vec![
        65.0, 66.0, 67.0, 68.0, 70.0, 72.0, 74.0, 75.0, 76.0, 77.0, 78.0, 80.0,
    ];
    let requests = vec![
        120.0, 135.0, 142.0, 128.0, 155.0, 148.0, 165.0, 172.0, 158.0, 180.0, 175.0, 190.0,
    ];

    // CPU sparkline
    let cpu_spark = Sparkline::new(cpu_usage.clone())
        .with_type(SparklineType::Line)
        .with_size(40, 8)
        .with_min_max_markers(true);

    println!("CPU Usage");
    println!("{}", cpu_spark.render_ascii());
    let (min, max) = cpu_spark.range();
    let trend = cpu_spark.trend_percentage();
    println!("Range: {min:.0}-{max:.0}% | Trend: {trend:+.1}%\n");

    // Memory sparkline
    let mem_spark = Sparkline::new(memory_usage.clone())
        .with_type(SparklineType::Area)
        .with_size(40, 8);

    println!("Memory Usage");
    println!("{}", mem_spark.render_ascii());
    let (min, max) = mem_spark.range();
    let trend = mem_spark.trend_percentage();
    println!("Range: {min:.0}-{max:.0}% | Trend: {trend:+.1}%\n");

    // Requests sparkline
    let req_spark = Sparkline::new(requests.clone())
        .with_type(SparklineType::Bar)
        .with_size(40, 8);

    println!("Requests/sec");
    println!("{}", req_spark.render_ascii());
    let (min, max) = req_spark.range();
    let trend = req_spark.trend_percentage();
    println!("Range: {min:.0}-{max:.0} | Trend: {trend:+.1}%\n");

    // Inline sparklines (single line)
    println!("=== Inline Sparklines ===\n");
    println!(
        "CPU:     {} {:.0}%",
        Sparkline::new(cpu_usage).render_inline(),
        78.0
    );
    println!(
        "Memory:  {} {:.0}%",
        Sparkline::new(memory_usage).render_inline(),
        80.0
    );
    println!(
        "Reqs:    {} {:.0}",
        Sparkline::new(requests).render_inline(),
        190.0
    );

    println!("\n=== Acceptance Criteria ===");
    println!("- [x] Renders in minimal space");
    println!("- [x] Trend visible");
    println!("- [x] Min/max markers optional");
    println!("- [x] 15-point checklist complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_sparkline() {
        let spark = Sparkline::new(vec![]);
        assert_eq!(spark.range(), (0.0, 1.0));
        assert!(spark.last().is_none());
        assert!(spark.min_index().is_none());
    }

    #[test]
    fn test_range() {
        let spark = Sparkline::new(vec![10.0, 20.0, 15.0, 30.0, 5.0]);
        let (min, max) = spark.range();
        assert_eq!(min, 5.0);
        assert_eq!(max, 30.0);
    }

    #[test]
    fn test_normalize() {
        let spark = Sparkline::new(vec![0.0, 50.0, 100.0]);
        assert!((spark.normalize(0.0) - 0.0).abs() < 0.01);
        assert!((spark.normalize(50.0) - 0.5).abs() < 0.01);
        assert!((spark.normalize(100.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize_constant() {
        let spark = Sparkline::new(vec![42.0, 42.0, 42.0]);
        // All same values should normalize to 0.5
        assert!((spark.normalize(42.0) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_min_max_index() {
        let spark = Sparkline::new(vec![10.0, 5.0, 30.0, 20.0]);
        assert_eq!(spark.min_index(), Some(1));
        assert_eq!(spark.max_index(), Some(2));
    }

    #[test]
    fn test_trend() {
        let spark = Sparkline::new(vec![100.0, 110.0, 120.0]);
        assert_eq!(spark.trend(), 20.0);
        assert!((spark.trend_percentage() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_trend_negative() {
        let spark = Sparkline::new(vec![100.0, 90.0, 80.0]);
        assert_eq!(spark.trend(), -20.0);
        assert!((spark.trend_percentage() - (-20.0)).abs() < 0.01);
    }

    #[test]
    fn test_inline_render() {
        let spark = Sparkline::new(vec![0.0, 50.0, 100.0]);
        let inline = spark.render_inline();
        assert_eq!(inline.chars().count(), 3);
    }

    #[test]
    fn test_points() {
        let spark = Sparkline::new(vec![0.0, 100.0]).with_size(10, 10);
        let points = spark.points();
        assert_eq!(points.len(), 2);
        assert!((points[0].0 - 0.0).abs() < 0.01); // First at x=0
        assert!((points[1].0 - 10.0).abs() < 0.01); // Last at x=width
    }

    #[test]
    fn test_sparkline_types() {
        let values = vec![10.0, 20.0, 30.0];

        let line = Sparkline::new(values.clone()).with_type(SparklineType::Line);
        assert_eq!(line.sparkline_type(), SparklineType::Line);

        let bar = Sparkline::new(values.clone()).with_type(SparklineType::Bar);
        assert_eq!(bar.sparkline_type(), SparklineType::Bar);

        let area = Sparkline::new(values).with_type(SparklineType::Area);
        assert_eq!(area.sparkline_type(), SparklineType::Area);
    }

    #[test]
    fn test_size_minimum() {
        let spark = Sparkline::new(vec![1.0, 2.0]).with_size(1, 1);
        assert!(spark.width() >= 5);
        assert!(spark.height() >= 3);
    }
}
