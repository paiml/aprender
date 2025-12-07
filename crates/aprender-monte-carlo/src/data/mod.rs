//! Data loading and embedded historical data
//!
//! Provides access to:
//! - Embedded S&P 500 historical returns (1928-2024)
//! - CSV file loading for custom data
//! - Data preprocessing utilities

mod sp500;

pub use sp500::{Sp500Data, Sp500Period};

use crate::error::{MonteCarloError, Result};
use std::path::Path;

/// CSV data loader for custom return data
#[derive(Debug, Clone)]
pub struct CsvLoader {
    /// Parsed returns
    pub returns: Vec<f64>,
    /// Column name used
    pub column_name: String,
    /// Number of rows loaded
    pub n_rows: usize,
}

impl CsvLoader {
    /// Load returns from a CSV file
    ///
    /// # Arguments
    /// * `path` - Path to CSV file
    /// * `column` - Optional column name (uses first numeric column if None)
    ///
    /// # Returns
    /// Parsed returns as a vector of f64
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed
    pub fn load<P: AsRef<Path>>(path: P, column: Option<&str>) -> Result<Self> {
        let path = path.as_ref();

        let mut reader = csv::Reader::from_path(path).map_err(|e| MonteCarloError::IoError {
            path: path.display().to_string(),
            message: format!("Failed to open CSV: {e}"),
        })?;

        let headers = reader
            .headers()
            .map_err(|e| MonteCarloError::CsvParse {
                line: 1,
                column: "headers".to_string(),
                message: format!("Failed to read headers: {e}"),
            })?
            .clone();

        // Find the target column
        let column_name = if let Some(col) = column {
            if !headers.iter().any(|h| h == col) {
                return Err(MonteCarloError::MissingField {
                    field: col.to_string(),
                    hint: format!(
                        "Available columns: {:?}",
                        headers.iter().collect::<Vec<_>>()
                    ),
                });
            }
            col.to_string()
        } else {
            // Find first column that looks numeric (not a date)
            headers
                .iter()
                .find(|h| {
                    let h_lower = h.to_lowercase();
                    h_lower.contains("return")
                        || h_lower.contains("pct")
                        || h_lower.contains("change")
                })
                .or_else(|| headers.get(1)) // Second column as fallback
                .ok_or_else(|| MonteCarloError::MissingField {
                    field: "return column".to_string(),
                    hint: "CSV should have a column containing 'return', 'pct', or 'change' in name".to_string(),
                })?
                .to_string()
        };

        let col_idx = headers
            .iter()
            .position(|h| h == column_name)
            .ok_or_else(|| MonteCarloError::MissingField {
                field: column_name.clone(),
                hint: format!("Column '{column_name}' not found in headers"),
            })?;

        let mut returns = Vec::new();
        let mut line_num = 2; // Start after header

        for result in reader.records() {
            let record = result.map_err(|e| MonteCarloError::CsvParse {
                line: line_num,
                column: column_name.clone(),
                message: format!("Failed to read row: {e}"),
            })?;

            if let Some(value) = record.get(col_idx) {
                // Try to parse as number, skip if not valid
                if let Ok(val) = value.trim().replace('%', "").parse::<f64>() {
                    // Handle percentage values
                    let return_val = if val.abs() > 1.0 && value.contains('%') {
                        val / 100.0
                    } else if val.abs() > 1.0 {
                        // Assume percentage if value is large
                        val / 100.0
                    } else {
                        val
                    };
                    returns.push(return_val);
                }
            }
            line_num += 1;
        }

        let n_rows = returns.len();

        if returns.is_empty() {
            return Err(MonteCarloError::EmptyData {
                context: format!("No valid numeric values found in column '{column_name}'"),
            });
        }

        Ok(Self {
            returns,
            column_name,
            n_rows,
        })
    }

    /// Get summary statistics
    #[must_use]
    pub fn stats(&self) -> DataStats {
        if self.returns.is_empty() {
            return DataStats::default();
        }

        let n = self.returns.len() as f64;
        let mean = self.returns.iter().sum::<f64>() / n;
        let variance = self.returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();

        let min = self.returns.iter().copied().fold(f64::INFINITY, f64::min);
        let max = self
            .returns
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        DataStats {
            n: self.returns.len(),
            mean,
            std,
            min,
            max,
        }
    }
}

/// Summary statistics for loaded data
#[derive(Debug, Clone, Default)]
pub struct DataStats {
    /// Number of observations
    pub n: usize,
    /// Mean return
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
}

impl std::fmt::Display for DataStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Data Statistics:")?;
        writeln!(f, "  Observations: {}", self.n)?;
        writeln!(f, "  Mean:         {:.4}", self.mean)?;
        writeln!(f, "  Std Dev:      {:.4}", self.std)?;
        writeln!(f, "  Min:          {:.4}", self.min)?;
        writeln!(f, "  Max:          {:.4}", self.max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_csv_loader() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(file, "date,return").expect("write header");
        writeln!(file, "2020-01,0.05").expect("write row");
        writeln!(file, "2020-02,-0.03").expect("write row");
        writeln!(file, "2020-03,0.02").expect("write row");

        let loader = CsvLoader::load(file.path(), Some("return")).expect("load CSV");

        assert_eq!(loader.n_rows, 3);
        assert_eq!(loader.returns.len(), 3);
        assert!((loader.returns[0] - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_csv_loader_percentage() {
        let mut file = NamedTempFile::new().expect("temp file");
        writeln!(file, "date,pct_change").expect("write header");
        writeln!(file, "2020-01,5%").expect("write row");
        writeln!(file, "2020-02,-3%").expect("write row");

        let loader = CsvLoader::load(file.path(), None).expect("load CSV");

        assert_eq!(loader.n_rows, 2);
        assert!((loader.returns[0] - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_data_stats() {
        let loader = CsvLoader {
            returns: vec![0.01, 0.02, 0.03, 0.04, 0.05],
            column_name: "test".to_string(),
            n_rows: 5,
        };

        let stats = loader.stats();

        assert_eq!(stats.n, 5);
        assert!((stats.mean - 0.03).abs() < 0.001);
        assert!(stats.std > 0.0);
    }
}
