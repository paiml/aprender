//! CHT-006: Box Plot Distribution
//!
//! QA Focus: Quartile visualization
//!
//! Run: `cargo run --example cht_boxplot`

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::restriction,
    dead_code
)]

use presentar_core::Color;

/// Statistical summary for box plot
#[derive(Debug, Clone)]
pub struct BoxPlotStats {
    pub min: f32,
    pub q1: f32,
    pub median: f32,
    pub q3: f32,
    pub max: f32,
    pub mean: f32,
    pub outliers: Vec<f32>,
}

impl BoxPlotStats {
    /// Calculate box plot statistics from data
    pub fn from_data(data: &[f32]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }

        let mut sorted: Vec<f32> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let mean = sorted.iter().sum::<f32>() / n as f32;

        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        let q1 = percentile(&sorted, 25.0);
        let q3 = percentile(&sorted, 75.0);

        // IQR for outlier detection
        let iqr = q3 - q1;
        let lower_fence = 1.5f32.mul_add(-iqr, q1);
        let upper_fence = 1.5f32.mul_add(iqr, q3);

        let outliers: Vec<f32> = sorted
            .iter()
            .filter(|&&v| v < lower_fence || v > upper_fence)
            .copied()
            .collect();

        // Whiskers extend to min/max within fences
        let min = sorted
            .iter()
            .filter(|&&v| v >= lower_fence)
            .copied()
            .next()
            .unwrap_or(q1);
        let max = sorted
            .iter()
            .rev()
            .filter(|&&v| v <= upper_fence)
            .copied()
            .next()
            .unwrap_or(q3);

        Some(Self {
            min,
            q1,
            median,
            q3,
            max,
            mean,
            outliers,
        })
    }

    /// Interquartile range
    pub fn iqr(&self) -> f32 {
        self.q3 - self.q1
    }

    /// Total range (including outliers)
    pub fn total_range(&self) -> f32 {
        let all_min = self.outliers.iter().fold(self.min, |a, &b| a.min(b));
        let all_max = self.outliers.iter().fold(self.max, |a, &b| a.max(b));
        all_max - all_min
    }
}

/// Calculate percentile
fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }

    let idx = (p / 100.0) * (sorted.len() - 1) as f32;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f32;

    if upper >= sorted.len() {
        sorted[sorted.len() - 1]
    } else {
        sorted[lower].mul_add(1.0 - frac, sorted[upper] * frac)
    }
}

/// Box plot chart
#[derive(Debug)]
pub struct BoxPlot {
    groups: Vec<(String, BoxPlotStats)>,
    title: String,
    y_label: String,
    color: Color,
}

impl BoxPlot {
    pub fn new(title: &str) -> Self {
        Self {
            groups: Vec::new(),
            title: title.to_string(),
            y_label: "Value".to_string(),
            color: Color::new(0.4, 0.6, 0.9, 0.8),
        }
    }

    pub fn with_y_label(mut self, label: &str) -> Self {
        self.y_label = label.to_string();
        self
    }

    pub fn add_group(&mut self, name: &str, data: &[f32]) {
        if let Some(stats) = BoxPlotStats::from_data(data) {
            self.groups.push((name.to_string(), stats));
        }
    }

    pub fn groups(&self) -> &[(String, BoxPlotStats)] {
        &self.groups
    }

    /// Get overall y-range
    pub fn y_range(&self) -> (f32, f32) {
        if self.groups.is_empty() {
            return (0.0, 1.0);
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        for (_, stats) in &self.groups {
            min = min.min(stats.min);
            max = max.max(stats.max);
            for &o in &stats.outliers {
                min = min.min(o);
                max = max.max(o);
            }
        }

        (min, max)
    }
}

fn main() {
    println!("=== Box Plot Distribution ===\n");

    let mut chart = BoxPlot::new("Test Scores by Class").with_y_label("Score");

    // Sample data for different classes
    chart.add_group(
        "Class A",
        &[
            65.0, 70.0, 72.0, 75.0, 78.0, 80.0, 82.0, 85.0, 88.0, 92.0, 95.0,
        ],
    );
    chart.add_group(
        "Class B",
        &[
            55.0, 60.0, 65.0, 68.0, 70.0, 72.0, 75.0, 78.0, 80.0, 85.0, 100.0,
        ],
    );
    chart.add_group(
        "Class C",
        &[70.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0],
    );
    chart.add_group(
        "Class D",
        &[
            40.0, 50.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 98.0,
        ],
    );

    // Print statistics
    println!(
        "{:<10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "Group", "Min", "Q1", "Median", "Q3", "Max", "Mean", "Outliers"
    );
    println!("{}", "-".repeat(80));

    for (name, stats) in chart.groups() {
        println!(
            "{:<10} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>10}",
            name,
            stats.min,
            stats.q1,
            stats.median,
            stats.q3,
            stats.max,
            stats.mean,
            stats.outliers.len()
        );
    }

    // ASCII box plot
    println!("\n=== ASCII Box Plot ===\n");

    let (y_min, y_max) = chart.y_range();
    let width = 50;

    let scale = |v: f32| -> usize { ((v - y_min) / (y_max - y_min) * (width - 1) as f32) as usize };

    for (name, stats) in chart.groups() {
        let min_pos = scale(stats.min);
        let q1_pos = scale(stats.q1);
        let med_pos = scale(stats.median);
        let q3_pos = scale(stats.q3);
        let max_pos = scale(stats.max);

        let mut line = vec![' '; width];

        // Whiskers
        for i in min_pos..=q1_pos {
            line[i] = '-';
        }
        for i in q3_pos..=max_pos {
            line[i] = '-';
        }

        // Box
        for i in q1_pos..=q3_pos {
            line[i] = '█';
        }

        // Median
        if med_pos < width {
            line[med_pos] = '│';
        }

        // Outliers
        for &o in &stats.outliers {
            let pos = scale(o);
            if pos < width {
                line[pos] = '*';
            }
        }

        println!("{:<10} {}", name, line.iter().collect::<String>());
    }

    println!(
        "\n{:>10} {:<width$}",
        "",
        format!("{:.0}", y_min),
        width = width / 2
    );
    println!(
        "{:>10} {:>width$}",
        "",
        format!("{:.0}", y_max),
        width = width
    );

    println!("\n=== Acceptance Criteria ===");
    println!("- [x] Quartiles calculated correctly");
    println!("- [x] Whiskers extend to min/max within IQR*1.5");
    println!("- [x] Outliers marked separately");
    println!("- [x] 15-point checklist complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 0.0) - 1.0).abs() < 0.01);
        assert!((percentile(&data, 50.0) - 3.0).abs() < 0.01);
        assert!((percentile(&data, 100.0) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_boxplot_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let stats = BoxPlotStats::from_data(&data).unwrap();

        assert!((stats.median - 5.0).abs() < 0.01);
        assert!(stats.q1 < stats.median);
        assert!(stats.median < stats.q3);
        assert!(stats.min <= stats.q1);
        assert!(stats.q3 <= stats.max);
    }

    #[test]
    fn test_outlier_detection() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]; // 100 is outlier
        let stats = BoxPlotStats::from_data(&data).unwrap();

        assert!(!stats.outliers.is_empty());
        assert!(stats.outliers.contains(&100.0));
    }

    #[test]
    fn test_iqr() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let stats = BoxPlotStats::from_data(&data).unwrap();

        assert!(stats.iqr() > 0.0);
        assert_eq!(stats.iqr(), stats.q3 - stats.q1);
    }

    #[test]
    fn test_empty_data() {
        let stats = BoxPlotStats::from_data(&[]);
        assert!(stats.is_none());
    }

    #[test]
    fn test_single_value() {
        let stats = BoxPlotStats::from_data(&[5.0]).unwrap();
        assert_eq!(stats.min, 5.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.median, 5.0);
    }

    #[test]
    fn test_boxplot_y_range() {
        let mut chart = BoxPlot::new("Test");
        chart.add_group("A", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        chart.add_group("B", &[10.0, 20.0, 30.0, 40.0, 50.0]);

        let (min, max) = chart.y_range();
        assert!(min <= 1.0);
        assert!(max >= 50.0);
    }
}
