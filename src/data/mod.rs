//! DataFrame module for named column containers.
//!
//! Provides a minimal DataFrame implementation (~300 LOC) for ML workflows.
//! Heavy data wrangling should be delegated to ruchy/polars.

use crate::error::Result;
use crate::primitives::{Matrix, Vector};

/// A minimal DataFrame with named columns.
///
/// This is a thin wrapper around `Vec<(String, Vector<f32>)>` with
/// convenience methods for ML workflows.
///
/// # Examples
///
/// ```
/// use aprender::data::DataFrame;
/// use aprender::primitives::Vector;
///
/// let columns = vec![
///     ("x".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0])),
///     ("y".to_string(), Vector::from_slice(&[4.0, 5.0, 6.0])),
/// ];
/// let df = DataFrame::new(columns).unwrap();
/// assert_eq!(df.shape(), (3, 2));
/// ```
#[derive(Debug, Clone)]
pub struct DataFrame {
    columns: Vec<(String, Vector<f32>)>,
    n_rows: usize,
}

impl DataFrame {
    /// Creates a new DataFrame from named columns.
    ///
    /// # Errors
    ///
    /// Returns an error if columns have different lengths or if empty.
    pub fn new(columns: Vec<(String, Vector<f32>)>) -> Result<Self> {
        if columns.is_empty() {
            return Err("DataFrame must have at least one column".into());
        }

        let n_rows = columns[0].1.len();

        // Verify all columns have same length
        for (name, col) in &columns {
            if col.len() != n_rows {
                return Err("All columns must have the same length".into());
            }
            if name.is_empty() {
                return Err("Column names cannot be empty".into());
            }
        }

        // Check for duplicate column names
        let mut names: Vec<&str> = columns.iter().map(|(n, _)| n.as_str()).collect();
        names.sort_unstable();
        for i in 1..names.len() {
            if names[i] == names[i - 1] {
                return Err("Duplicate column names not allowed".into());
            }
        }

        Ok(Self { columns, n_rows })
    }

    /// Returns the shape as (n_rows, n_cols).
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows, self.columns.len())
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    /// Returns the column names.
    #[must_use]
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Returns a reference to a column by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the column doesn't exist.
    pub fn column(&self, name: &str) -> Result<&Vector<f32>> {
        self.columns
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v)
            .ok_or_else(|| "Column not found".into())
    }

    /// Selects multiple columns by name, returning a new DataFrame.
    ///
    /// # Errors
    ///
    /// Returns an error if any column doesn't exist.
    pub fn select(&self, names: &[&str]) -> Result<Self> {
        if names.is_empty() {
            return Err("Must select at least one column".into());
        }

        let mut selected = Vec::with_capacity(names.len());

        for &name in names {
            let col = self.column(name)?;
            selected.push((name.to_string(), col.clone()));
        }

        Self::new(selected)
    }

    /// Returns a row as a Vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of bounds.
    pub fn row(&self, idx: usize) -> Result<Vector<f32>> {
        if idx >= self.n_rows {
            return Err("Row index out of bounds".into());
        }

        let data: Vec<f32> = self.columns.iter().map(|(_, col)| col[idx]).collect();
        Ok(Vector::from_vec(data))
    }

    /// Converts the DataFrame to a Matrix (column-major stacking).
    ///
    /// Returns a Matrix with shape (n_rows, n_cols).
    #[must_use]
    pub fn to_matrix(&self) -> Matrix<f32> {
        let mut data = Vec::with_capacity(self.n_rows * self.columns.len());

        for row_idx in 0..self.n_rows {
            for (_, col) in &self.columns {
                data.push(col[row_idx]);
            }
        }

        Matrix::from_vec(self.n_rows, self.columns.len(), data)
            .expect("Internal error: data size mismatch")
    }

    /// Returns an iterator over columns as (name, vector) pairs.
    pub fn iter_columns(&self) -> impl Iterator<Item = (&str, &Vector<f32>)> {
        self.columns.iter().map(|(n, v)| (n.as_str(), v))
    }

    /// Adds a new column to the DataFrame.
    ///
    /// # Errors
    ///
    /// Returns an error if column length doesn't match or name already exists.
    pub fn add_column(&mut self, name: String, data: Vector<f32>) -> Result<()> {
        if data.len() != self.n_rows {
            return Err("Column length must match existing rows".into());
        }

        if self.columns.iter().any(|(n, _)| n == &name) {
            return Err("Column name already exists".into());
        }

        if name.is_empty() {
            return Err("Column name cannot be empty".into());
        }

        self.columns.push((name, data));
        Ok(())
    }

    /// Drops a column by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the column doesn't exist or is the last column.
    pub fn drop_column(&mut self, name: &str) -> Result<()> {
        if self.columns.len() == 1 {
            return Err("Cannot drop the last column".into());
        }

        let idx = self
            .columns
            .iter()
            .position(|(n, _)| n == name)
            .ok_or("Column not found")?;

        self.columns.remove(idx);
        Ok(())
    }

    /// Returns descriptive statistics for all columns.
    #[must_use]
    pub fn describe(&self) -> Vec<ColumnStats> {
        self.columns
            .iter()
            .map(|(name, col)| {
                let mean = col.mean();
                let variance = col.variance();
                let std = variance.sqrt();

                let mut sorted: Vec<f32> = col.as_slice().to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let min = sorted.first().copied().unwrap_or(0.0);
                let max = sorted.last().copied().unwrap_or(0.0);
                let median = if sorted.is_empty() {
                    0.0
                } else if sorted.len() % 2 == 0 {
                    (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
                } else {
                    sorted[sorted.len() / 2]
                };

                ColumnStats {
                    name: name.clone(),
                    count: col.len(),
                    mean,
                    std,
                    min,
                    median,
                    max,
                }
            })
            .collect()
    }
}

/// Descriptive statistics for a column.
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Column name.
    pub name: String,
    /// Number of elements.
    pub count: usize,
    /// Mean value.
    pub mean: f32,
    /// Standard deviation.
    pub std: f32,
    /// Minimum value.
    pub min: f32,
    /// Median value.
    pub median: f32,
    /// Maximum value.
    pub max: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_df() -> DataFrame {
        let columns = vec![
            ("a".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0])),
            ("b".to_string(), Vector::from_slice(&[4.0, 5.0, 6.0])),
            ("c".to_string(), Vector::from_slice(&[7.0, 8.0, 9.0])),
        ];
        DataFrame::new(columns).unwrap()
    }

    #[test]
    fn test_new() {
        let df = sample_df();
        assert_eq!(df.shape(), (3, 3));
        assert_eq!(df.n_rows(), 3);
        assert_eq!(df.n_cols(), 3);
    }

    #[test]
    fn test_new_empty_error() {
        let result = DataFrame::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_mismatched_lengths_error() {
        let columns = vec![
            ("a".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0])),
            ("b".to_string(), Vector::from_slice(&[4.0, 5.0])),
        ];
        let result = DataFrame::new(columns);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_duplicate_names_error() {
        let columns = vec![
            ("a".to_string(), Vector::from_slice(&[1.0, 2.0])),
            ("a".to_string(), Vector::from_slice(&[3.0, 4.0])),
        ];
        let result = DataFrame::new(columns);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_empty_name_error() {
        let columns = vec![("".to_string(), Vector::from_slice(&[1.0, 2.0]))];
        let result = DataFrame::new(columns);
        assert!(result.is_err());
    }

    #[test]
    fn test_column_names() {
        let df = sample_df();
        let names = df.column_names();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_column() {
        let df = sample_df();
        let col = df.column("b").unwrap();
        assert_eq!(col.len(), 3);
        assert!((col[0] - 4.0).abs() < 1e-6);
        assert!((col[1] - 5.0).abs() < 1e-6);
        assert!((col[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_column_not_found() {
        let df = sample_df();
        let result = df.column("z");
        assert!(result.is_err());
    }

    #[test]
    fn test_select() {
        let df = sample_df();
        let selected = df.select(&["a", "c"]).unwrap();
        assert_eq!(selected.shape(), (3, 2));
        assert_eq!(selected.column_names(), vec!["a", "c"]);
    }

    #[test]
    fn test_select_empty_error() {
        let df = sample_df();
        let result = df.select(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_select_not_found_error() {
        let df = sample_df();
        let result = df.select(&["a", "z"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_row() {
        let df = sample_df();
        let row = df.row(1).unwrap();
        assert_eq!(row.len(), 3);
        assert!((row[0] - 2.0).abs() < 1e-6);
        assert!((row[1] - 5.0).abs() < 1e-6);
        assert!((row[2] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_row_out_of_bounds() {
        let df = sample_df();
        let result = df.row(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_matrix() {
        let df = sample_df();
        let matrix = df.to_matrix();
        assert_eq!(matrix.shape(), (3, 3));

        // Row 0: [1, 4, 7]
        assert!((matrix.get(0, 0) - 1.0).abs() < 1e-6);
        assert!((matrix.get(0, 1) - 4.0).abs() < 1e-6);
        assert!((matrix.get(0, 2) - 7.0).abs() < 1e-6);

        // Row 1: [2, 5, 8]
        assert!((matrix.get(1, 0) - 2.0).abs() < 1e-6);
        assert!((matrix.get(1, 1) - 5.0).abs() < 1e-6);
        assert!((matrix.get(1, 2) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_column() {
        let mut df = sample_df();
        let new_col = Vector::from_slice(&[10.0, 11.0, 12.0]);
        df.add_column("d".to_string(), new_col).unwrap();

        assert_eq!(df.n_cols(), 4);
        let col = df.column("d").unwrap();
        assert!((col[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_column_wrong_length() {
        let mut df = sample_df();
        let new_col = Vector::from_slice(&[10.0, 11.0]);
        let result = df.add_column("d".to_string(), new_col);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_column_duplicate_name() {
        let mut df = sample_df();
        let new_col = Vector::from_slice(&[10.0, 11.0, 12.0]);
        let result = df.add_column("a".to_string(), new_col);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_column_empty_name() {
        let mut df = sample_df();
        let new_col = Vector::from_slice(&[10.0, 11.0, 12.0]);
        let result = df.add_column(String::new(), new_col);
        assert!(result.is_err());
    }

    #[test]
    fn test_drop_column() {
        let mut df = sample_df();
        df.drop_column("b").unwrap();

        assert_eq!(df.n_cols(), 2);
        assert!(df.column("b").is_err());
    }

    #[test]
    fn test_drop_column_not_found() {
        let mut df = sample_df();
        let result = df.drop_column("z");
        assert!(result.is_err());
    }

    #[test]
    fn test_drop_last_column_error() {
        let columns = vec![("a".to_string(), Vector::from_slice(&[1.0, 2.0]))];
        let mut df = DataFrame::new(columns).unwrap();
        let result = df.drop_column("a");
        assert!(result.is_err());
    }

    #[test]
    fn test_describe() {
        let columns = vec![(
            "x".to_string(),
            Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]),
        )];
        let df = DataFrame::new(columns).unwrap();
        let stats = df.describe();

        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].name, "x");
        assert_eq!(stats[0].count, 5);
        assert!((stats[0].mean - 3.0).abs() < 1e-6);
        assert!((stats[0].min - 1.0).abs() < 1e-6);
        assert!((stats[0].max - 5.0).abs() < 1e-6);
        assert!((stats[0].median - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_iter_columns() {
        let df = sample_df();
        let cols: Vec<_> = df.iter_columns().collect();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].0, "a");
        assert_eq!(cols[1].0, "b");
        assert_eq!(cols[2].0, "c");
    }

    #[test]
    fn test_select_preserves_property() {
        // Property: select(names).column(name) == original.column(name)
        let df = sample_df();
        let selected = df.select(&["a", "c"]).unwrap();

        let orig_a = df.column("a").unwrap();
        let sel_a = selected.column("a").unwrap();

        assert_eq!(orig_a.len(), sel_a.len());
        for i in 0..orig_a.len() {
            assert!((orig_a[i] - sel_a[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_to_matrix_column_count() {
        // Property: to_matrix().n_cols() == n_selected_columns
        let df = sample_df();
        let selected = df.select(&["a", "b"]).unwrap();
        let matrix = selected.to_matrix();
        assert_eq!(matrix.n_cols(), 2);
    }

    #[test]
    fn test_describe_median_even_length() {
        // Test median calculation for even-length arrays
        // Median of [1, 2, 3, 4] = (2 + 3) / 2 = 2.5
        let columns = vec![("x".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]))];
        let df = DataFrame::new(columns).unwrap();
        let stats = df.describe();

        // This catches mutations in:
        // - sorted.len() % 2 == 0 (% vs + or /)
        // - sorted[sorted.len() / 2 - 1] (index calculation)
        // - + sorted[sorted.len() / 2] (sum of middle values)
        // - / 2.0 (averaging)
        assert!(
            (stats[0].median - 2.5).abs() < 1e-6,
            "Expected median 2.5, got {}",
            stats[0].median
        );
    }

    #[test]
    fn test_describe_median_odd_length() {
        // Test median calculation for odd-length arrays
        // Median of [1, 2, 3] = 2.0 (middle element)
        let columns = vec![("x".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0]))];
        let df = DataFrame::new(columns).unwrap();
        let stats = df.describe();

        // For odd length, median = sorted[len / 2] = sorted[1] = 2.0
        assert!(
            (stats[0].median - 2.0).abs() < 1e-6,
            "Expected median 2.0, got {}",
            stats[0].median
        );
    }

    #[test]
    fn test_describe_median_two_elements() {
        // Test median with exactly 2 elements
        // Median of [10, 20] = (10 + 20) / 2 = 15
        let columns = vec![("x".to_string(), Vector::from_slice(&[10.0, 20.0]))];
        let df = DataFrame::new(columns).unwrap();
        let stats = df.describe();

        // This catches mutations in median averaging
        assert!(
            (stats[0].median - 15.0).abs() < 1e-6,
            "Expected median 15.0, got {}",
            stats[0].median
        );
    }

    #[test]
    fn test_describe_median_arithmetic_mutations() {
        // Test to catch specific arithmetic mutations
        // Using values where wrong operations give different results
        // [2, 4, 6, 8]: median = (4 + 6) / 2 = 5.0
        let columns = vec![("x".to_string(), Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]))];
        let df = DataFrame::new(columns).unwrap();
        let stats = df.describe();

        // If + becomes - in median sum: (4 - 6) / 2 = -1
        // If / 2.0 becomes * 2.0: (4 + 6) * 2 = 20
        // If / 2 - 1 becomes / 2 + 1: would access wrong index
        assert!(
            (stats[0].median - 5.0).abs() < 1e-6,
            "Expected median 5.0, got {}",
            stats[0].median
        );
        assert!(
            stats[0].median > 0.0,
            "Median should be positive, got {}",
            stats[0].median
        );
        assert!(
            stats[0].median < 10.0,
            "Median should be < 10, got {}",
            stats[0].median
        );
    }

    #[test]
    fn test_describe_median_unsorted_input() {
        // Verify median calculation sorts data correctly
        // Input [5, 1, 3, 2, 4] -> sorted [1, 2, 3, 4, 5] -> median = 3
        let columns = vec![(
            "x".to_string(),
            Vector::from_slice(&[5.0, 1.0, 3.0, 2.0, 4.0]),
        )];
        let df = DataFrame::new(columns).unwrap();
        let stats = df.describe();

        assert!(
            (stats[0].median - 3.0).abs() < 1e-6,
            "Expected median 3.0, got {}",
            stats[0].median
        );
    }

    #[test]
    fn test_describe_six_elements() {
        // Test with 6 elements to ensure index math is correct
        // [1, 2, 3, 4, 5, 6]: median = (3 + 4) / 2 = 3.5
        // len = 6, len/2 = 3, len/2 - 1 = 2
        // sorted[2] = 3, sorted[3] = 4
        let columns = vec![(
            "x".to_string(),
            Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )];
        let df = DataFrame::new(columns).unwrap();
        let stats = df.describe();

        assert!(
            (stats[0].median - 3.5).abs() < 1e-6,
            "Expected median 3.5, got {}",
            stats[0].median
        );
    }
}
