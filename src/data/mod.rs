//! `DataFrame` module for named column containers.
//!
//! Provides a minimal `DataFrame` implementation (~300 LOC) for ML workflows.
//! Heavy data wrangling should be delegated to ruchy/polars.

use crate::error::Result;
use crate::primitives::{Matrix, Vector};

/// A minimal `DataFrame` with named columns.
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
/// let df = DataFrame::new(columns).expect("DataFrame creation should succeed with valid columns");
/// assert_eq!(df.shape(), (3, 2));
/// ```
#[derive(Debug, Clone)]
pub struct DataFrame {
    columns: Vec<(String, Vector<f32>)>,
    n_rows: usize,
}

impl DataFrame {
    /// Creates a new `DataFrame` from named columns.
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

    /// Returns the shape as (`n_rows`, `n_cols`).
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

    /// Selects multiple columns by name, returning a new `DataFrame`.
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

    /// Converts the `DataFrame` to a Matrix (column-major stacking).
    ///
    /// Returns a Matrix with shape (`n_rows`, `n_cols`).
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

    /// Adds a new column to the `DataFrame`.
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
#[path = "data_tests.rs"]
mod tests;
