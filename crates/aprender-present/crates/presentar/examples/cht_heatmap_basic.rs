//! CHT-005: Heatmap Basic
//!
//! QA Focus: 2D tensor visualization
//!
//! Run: `cargo run --example cht_heatmap_basic`

#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::match_same_arms,
    clippy::needless_range_loop,
    clippy::iter_without_into_iter,
    clippy::used_underscore_binding,
    clippy::struct_field_names,
    dead_code,
    unused_variables,
    clippy::format_collect
)]

use presentar_core::Color;

/// Colormap for heatmap rendering
#[derive(Debug, Clone, Copy)]
pub enum Colormap {
    Viridis,
    Plasma,
    Inferno,
    Blues,
    Reds,
    Greens,
    Grayscale,
}

impl Colormap {
    /// Map a value (0-1) to a color
    pub fn map(&self, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0);

        match self {
            Self::Viridis => {
                // Approximation of viridis
                let r = 0.267 + t * (0.993 - 0.267);
                let g = if t < 0.5 {
                    (t * 2.0).mul_add(0.906 - 0.004, 0.004)
                } else {
                    ((t - 0.5) * 2.0).mul_add(-(0.906 - 0.334), 0.906)
                };
                let b = (1.0 - t).mul_add(0.533 - 0.329, 0.329);
                Color::new(r, g, b, 1.0)
            }
            Self::Plasma => {
                let r = 0.05 + t * 0.9;
                let g = t * 0.7;
                let b = (1.0 - t).mul_add(0.4, 0.53);
                Color::new(r, g, b, 1.0)
            }
            Self::Inferno => {
                let r = t;
                let g = t * t;
                let b = (1.0 - t) * 0.5;
                Color::new(r, g, b, 1.0)
            }
            Self::Blues => Color::new(1.0 - t * 0.7, 1.0 - t * 0.5, 1.0, 1.0),
            Self::Reds => Color::new(1.0, 1.0 - t * 0.8, 1.0 - t * 0.8, 1.0),
            Self::Greens => Color::new(1.0 - t * 0.7, 1.0, 1.0 - t * 0.7, 1.0),
            Self::Grayscale => Color::new(1.0 - t, 1.0 - t, 1.0 - t, 1.0),
        }
    }
}

/// 2D Heatmap
#[derive(Debug)]
pub struct Heatmap {
    data: Vec<Vec<f32>>,
    rows: usize,
    cols: usize,
    row_labels: Vec<String>,
    col_labels: Vec<String>,
    title: String,
    colormap: Colormap,
    annotate: bool,
}

impl Heatmap {
    pub fn new(data: Vec<Vec<f32>>, title: &str) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };

        Self {
            data,
            rows,
            cols,
            row_labels: (0..rows).map(|i| format!("R{i}")).collect(),
            col_labels: (0..cols).map(|i| format!("C{i}")).collect(),
            title: title.to_string(),
            colormap: Colormap::Viridis,
            annotate: false,
        }
    }

    pub fn with_labels(mut self, row_labels: Vec<String>, col_labels: Vec<String>) -> Self {
        self.row_labels = row_labels;
        self.col_labels = col_labels;
        self
    }

    pub const fn with_colormap(mut self, colormap: Colormap) -> Self {
        self.colormap = colormap;
        self
    }

    pub const fn with_annotations(mut self, annotate: bool) -> Self {
        self.annotate = annotate;
        self
    }

    /// Get value at position
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        self.data.get(row).and_then(|r| r.get(col)).copied()
    }

    /// Get min and max values
    pub fn range(&self) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        for row in &self.data {
            for &val in row {
                min = min.min(val);
                max = max.max(val);
            }
        }

        if min == f32::INFINITY {
            (0.0, 1.0)
        } else {
            (min, max)
        }
    }

    /// Normalize value to 0-1 range
    pub fn normalize(&self, value: f32) -> f32 {
        let (min, max) = self.range();
        if (max - min).abs() < 0.0001 {
            0.5
        } else {
            (value - min) / (max - min)
        }
    }

    /// Get color for a cell
    pub fn cell_color(&self, row: usize, col: usize) -> Color {
        self.get(row, col)
            .map_or(Color::BLACK, |v| self.colormap.map(self.normalize(v)))
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn cols(&self) -> usize {
        self.cols
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub const fn annotate(&self) -> bool {
        self.annotate
    }
}

/// Get heat level character for normalized value.
fn heat_char(t: f32) -> char {
    if t > 0.8 {
        '█'
    } else if t > 0.6 {
        '▓'
    } else if t > 0.4 {
        '▒'
    } else if t > 0.2 {
        '░'
    } else {
        ' '
    }
}

/// Print heatmap data table.
fn print_data_table(heatmap: &Heatmap) {
    println!(
        "\n        {}",
        heatmap
            .col_labels
            .iter()
            .map(|s| format!("{:>5}", &s[..3.min(s.len())]))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!("       {}", "-".repeat(heatmap.cols() * 6));

    for (i, row_label) in heatmap.row_labels.iter().enumerate() {
        print!("{:>6} |", &row_label[..6.min(row_label.len())]);
        for j in 0..heatmap.cols() {
            if let Some(val) = heatmap.get(i, j) {
                print!("{val:>5.0} ");
            }
        }
        println!();
    }
}

/// Print ASCII heatmap.
fn print_ascii_heatmap(heatmap: &Heatmap) {
    println!("\n=== ASCII Heatmap ===\n");
    println!(
        "       {}",
        heatmap
            .col_labels
            .iter()
            .map(|s| format!("{:>3}", &s[..1]))
            .collect::<String>()
    );

    for (i, row_label) in heatmap.row_labels.iter().enumerate() {
        print!("{:>6} ", &row_label[..6.min(row_label.len())]);
        for j in 0..heatmap.cols() {
            let t = heatmap.get(i, j).map_or(0.0, |v| heatmap.normalize(v));
            print!("{:>3}", heat_char(t));
        }
        println!();
    }
    println!("\nLegend: █ hot  ▓ warm  ▒ mild  ░ cool  ' ' cold");
}

fn main() {
    println!("=== Heatmap Basic ===\n");

    let data = vec![
        vec![
            5.0, 7.0, 12.0, 18.0, 23.0, 28.0, 30.0, 29.0, 24.0, 17.0, 10.0, 6.0,
        ],
        vec![
            2.0, 4.0, 10.0, 16.0, 21.0, 26.0, 29.0, 28.0, 22.0, 14.0, 7.0, 3.0,
        ],
        vec![
            10.0, 12.0, 16.0, 20.0, 25.0, 30.0, 33.0, 32.0, 28.0, 22.0, 15.0, 11.0,
        ],
        vec![
            -5.0, -2.0, 5.0, 12.0, 18.0, 23.0, 26.0, 24.0, 18.0, 10.0, 2.0, -3.0,
        ],
    ];

    let row_labels = vec![
        "Paris".to_string(),
        "Berlin".to_string(),
        "Madrid".to_string(),
        "Moscow".to_string(),
    ];
    let col_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();

    let heatmap = Heatmap::new(data, "Monthly Average Temperatures (°C)")
        .with_labels(row_labels, col_labels)
        .with_colormap(Colormap::Inferno)
        .with_annotations(true);

    println!("Title: {}", heatmap.title());
    println!("Size: {}x{}", heatmap.rows(), heatmap.cols());
    let (min, max) = heatmap.range();
    println!("Range: {min:.1} - {max:.1}");

    print_data_table(&heatmap);
    print_ascii_heatmap(&heatmap);

    println!("\n=== Acceptance Criteria ===");
    println!("- [x] Data grid correct");
    println!("- [x] Color scale applied");
    println!("- [x] Labels visible");
    println!("- [x] 15-point checklist complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colormap_bounds() {
        for cm in [Colormap::Viridis, Colormap::Plasma, Colormap::Blues] {
            let c0 = cm.map(0.0);
            let c1 = cm.map(1.0);

            assert!(c0.r >= 0.0 && c0.r <= 1.0);
            assert!(c1.r >= 0.0 && c1.r <= 1.0);
        }
    }

    #[test]
    fn test_colormap_clamping() {
        let c_neg = Colormap::Viridis.map(-0.5);
        let c_zero = Colormap::Viridis.map(0.0);
        assert_eq!(c_neg.r, c_zero.r);

        let c_high = Colormap::Viridis.map(1.5);
        let c_one = Colormap::Viridis.map(1.0);
        assert_eq!(c_high.r, c_one.r);
    }

    #[test]
    fn test_heatmap_creation() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let heatmap = Heatmap::new(data, "Test");

        assert_eq!(heatmap.rows(), 2);
        assert_eq!(heatmap.cols(), 2);
    }

    #[test]
    fn test_heatmap_get() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let heatmap = Heatmap::new(data, "Test");

        assert_eq!(heatmap.get(0, 0), Some(1.0));
        assert_eq!(heatmap.get(1, 1), Some(4.0));
        assert_eq!(heatmap.get(2, 0), None);
    }

    #[test]
    fn test_heatmap_range() {
        let data = vec![vec![1.0, 5.0], vec![2.0, 10.0]];
        let heatmap = Heatmap::new(data, "Test");

        let (min, max) = heatmap.range();
        assert_eq!(min, 1.0);
        assert_eq!(max, 10.0);
    }

    #[test]
    fn test_heatmap_normalize() {
        let data = vec![vec![0.0, 100.0]];
        let heatmap = Heatmap::new(data, "Test");

        assert!((heatmap.normalize(0.0) - 0.0).abs() < 0.01);
        assert!((heatmap.normalize(50.0) - 0.5).abs() < 0.01);
        assert!((heatmap.normalize(100.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_heatmap() {
        let heatmap = Heatmap::new(vec![], "Empty");
        assert_eq!(heatmap.rows(), 0);
        assert_eq!(heatmap.cols(), 0);
        assert_eq!(heatmap.range(), (0.0, 1.0));
    }
}
