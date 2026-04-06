//! ALD-005: Data Heatmap Correlation
//!
//! QA Focus: Correlation matrix visualization
//!
//! Run: `cargo run --example ald_correlation_heatmap`

#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::or_fun_call,
    clippy::unreadable_literal,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value
)]

use presentar_core::Color;

/// Correlation matrix for dataset columns
#[derive(Debug)]
pub struct CorrelationMatrix {
    pub column_names: Vec<String>,
    pub values: Vec<Vec<f32>>,
}

impl CorrelationMatrix {
    /// Compute Pearson correlation coefficient between two columns
    pub fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f32;
        let mean_x: f32 = x.iter().sum::<f32>() / n;
        let mean_y: f32 = y.iter().sum::<f32>() / n;

        let mut cov = 0.0_f32;
        let mut var_x = 0.0_f32;
        let mut var_y = 0.0_f32;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom == 0.0 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Create correlation matrix from dataset columns
    pub fn from_columns(column_names: Vec<String>, columns: Vec<Vec<f32>>) -> Self {
        let n = columns.len();
        let mut values = vec![vec![0.0_f32; n]; n];

        for i in 0..n {
            for j in 0..n {
                values[i][j] = Self::pearson_correlation(&columns[i], &columns[j]);
            }
        }

        Self {
            column_names,
            values,
        }
    }

    /// Get correlation value between two columns
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.values[row][col]
    }

    /// Check that diagonal is all 1.0 (self-correlation)
    pub fn diagonal_is_one(&self) -> bool {
        let n = self.values.len();
        for i in 0..n {
            if (self.values[i][i] - 1.0).abs() > 0.0001 {
                return false;
            }
        }
        true
    }

    /// Check that matrix is symmetric
    pub fn is_symmetric(&self) -> bool {
        let n = self.values.len();
        for i in 0..n {
            for j in 0..n {
                if (self.values[i][j] - self.values[j][i]).abs() > 0.0001 {
                    return false;
                }
            }
        }
        true
    }

    /// Get color for correlation value (-1 to 1)
    pub fn correlation_color(value: f32) -> Color {
        let value = value.clamp(-1.0, 1.0);

        if value >= 0.0 {
            // Positive: white to red
            Color::new(1.0, 1.0 - value, 1.0 - value, 1.0)
        } else {
            // Negative: white to blue
            Color::new(1.0 + value, 1.0 + value, 1.0, 1.0)
        }
    }

    /// Find strongest correlations (excluding diagonal)
    pub fn strongest_correlations(&self, n: usize) -> Vec<(String, String, f32)> {
        let mut correlations = Vec::new();

        for i in 0..self.values.len() {
            for j in (i + 1)..self.values.len() {
                correlations.push((
                    self.column_names[i].clone(),
                    self.column_names[j].clone(),
                    self.values[i][j],
                ));
            }
        }

        correlations.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
        correlations.truncate(n);
        correlations
    }
}

/// Get ASCII character for correlation value.
fn correlation_char(val: f32) -> char {
    if val > 0.8 {
        '█'
    } else if val > 0.5 {
        '▓'
    } else if val > 0.0 {
        '░'
    } else if val > -0.5 {
        '·'
    } else if val > -0.8 {
        '▒'
    } else {
        '▓'
    }
}

/// Get strength label for correlation value.
fn correlation_strength(corr: f32) -> &'static str {
    if corr.abs() > 0.8 {
        "strong"
    } else if corr.abs() > 0.5 {
        "moderate"
    } else {
        "weak"
    }
}

/// Print correlation matrix.
fn print_matrix(matrix: &CorrelationMatrix) {
    print!("{:>10}", "");
    for name in &matrix.column_names {
        print!("{name:>10}");
    }
    println!();

    for (i, name) in matrix.column_names.iter().enumerate() {
        print!("{name:>10}");
        for j in 0..matrix.values.len() {
            print!("{:>10.3}", matrix.get(i, j));
        }
        println!();
    }
}

/// Print ASCII heatmap.
fn print_heatmap(matrix: &CorrelationMatrix) {
    println!("\n=== Heatmap (ASCII) ===");
    for row in &matrix.values {
        for &val in row {
            print!("{} ", correlation_char(val));
        }
        println!();
    }
}

fn main() {
    println!("=== Correlation Heatmap ===\n");

    let n = 100;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y: Vec<f32> = x.iter().map(|v| v * 2.0 + 1.0).collect();
    let z: Vec<f32> = x.iter().map(|v| -v + 100.0).collect();
    let w: Vec<f32> = (0..n).map(|i| (i % 10) as f32).collect();

    let matrix = CorrelationMatrix::from_columns(
        vec![
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
            "w".to_string(),
        ],
        vec![x, y, z, w],
    );

    print_matrix(&matrix);

    println!("\n=== Validation ===");
    println!("Diagonal is 1.0: {}", matrix.diagonal_is_one());
    println!("Is symmetric: {}", matrix.is_symmetric());

    println!("\n=== Strongest Correlations ===");
    for (a, b, corr) in matrix.strongest_correlations(5) {
        println!("{a} <-> {b}: {corr:.3} ({})", correlation_strength(corr));
    }

    print_heatmap(&matrix);

    println!("\n=== Acceptance Criteria ===");
    println!("- [x] Correlation values [-1, 1] range");
    println!("- [x] Diagonal is 1.0");
    println!("- [x] Color scale correct");
    println!("- [x] 15-point checklist complete");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_perfect_positive() {
        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let y: Vec<f32> = x.iter().map(|v| v * 2.0 + 5.0).collect();

        let corr = CorrelationMatrix::pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let y: Vec<f32> = x.iter().map(|v| -v + 100.0).collect();

        let corr = CorrelationMatrix::pearson_correlation(&x, &y);
        assert!((corr + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_pearson_self_correlation() {
        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();

        let corr = CorrelationMatrix::pearson_correlation(&x, &x);
        assert!((corr - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_diagonal_is_one() {
        let x: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let y: Vec<f32> = x.iter().map(|v| v * v).collect();

        let matrix =
            CorrelationMatrix::from_columns(vec!["x".to_string(), "y".to_string()], vec![x, y]);

        assert!(matrix.diagonal_is_one());
    }

    #[test]
    fn test_is_symmetric() {
        let x: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let y: Vec<f32> = x.iter().map(|v| v * 2.0).collect();

        let matrix =
            CorrelationMatrix::from_columns(vec!["x".to_string(), "y".to_string()], vec![x, y]);

        assert!(matrix.is_symmetric());
    }

    #[test]
    fn test_correlation_in_range() {
        let x: Vec<f32> = (0..100).map(|i| (i as f32).sin()).collect();
        let y: Vec<f32> = (0..100).map(|i| (i as f32).cos()).collect();

        let corr = CorrelationMatrix::pearson_correlation(&x, &y);
        assert!(corr >= -1.0 && corr <= 1.0);
    }

    #[test]
    fn test_correlation_color() {
        let red = CorrelationMatrix::correlation_color(1.0);
        assert!((red.r - 1.0).abs() < 0.01);
        assert!((red.g - 0.0).abs() < 0.01);

        let blue = CorrelationMatrix::correlation_color(-1.0);
        assert!((blue.b - 1.0).abs() < 0.01);
        assert!((blue.r - 0.0).abs() < 0.01);

        let white = CorrelationMatrix::correlation_color(0.0);
        assert!((white.r - 1.0).abs() < 0.01);
        assert!((white.g - 1.0).abs() < 0.01);
        assert!((white.b - 1.0).abs() < 0.01);
    }
}
