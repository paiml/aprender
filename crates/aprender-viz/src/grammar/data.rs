//! Data abstraction for Grammar of Graphics.
//!
//! Provides a simple columnar data structure for visualization.

use std::collections::HashMap;

/// A value in a data frame.
#[derive(Debug, Clone, PartialEq)]
pub enum DataValue {
    /// A numeric value.
    Number(f32),
    /// A text value.
    Text(String),
    /// A missing value.
    Null,
}

impl DataValue {
    /// Get as f32, or None if not a number.
    #[must_use]
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            DataValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Get as string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            DataValue::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl From<f32> for DataValue {
    fn from(v: f32) -> Self {
        DataValue::Number(v)
    }
}

impl From<&str> for DataValue {
    fn from(s: &str) -> Self {
        DataValue::Text(s.to_string())
    }
}

impl From<String> for DataValue {
    fn from(s: String) -> Self {
        DataValue::Text(s)
    }
}

/// A simple columnar data frame.
#[derive(Debug, Clone, Default)]
pub struct DataFrame {
    /// Column data keyed by column name.
    columns: HashMap<String, Vec<DataValue>>,
    /// Number of rows.
    n_rows: usize,
}

impl DataFrame {
    /// Create a new empty data frame.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from x and y arrays.
    #[must_use]
    pub fn from_xy(x: &[f32], y: &[f32]) -> Self {
        let n = x.len().min(y.len());
        let mut df = Self::new();
        df.add_column_f32("x", &x[..n]);
        df.add_column_f32("y", &y[..n]);
        df
    }

    /// Create from a single data array.
    #[must_use]
    pub fn from_data(data: &[f32]) -> Self {
        let mut df = Self::new();
        df.add_column_f32("data", data);
        df
    }

    /// Add a numeric column.
    pub fn add_column_f32(&mut self, name: &str, data: &[f32]) {
        let values: Vec<DataValue> = data.iter().map(|&v| DataValue::Number(v)).collect();
        self.n_rows = self.n_rows.max(values.len());
        self.columns.insert(name.to_string(), values);
    }

    /// Add a text column.
    pub fn add_column_str(&mut self, name: &str, data: &[&str]) {
        let values: Vec<DataValue> = data.iter().map(|&s| DataValue::Text(s.to_string())).collect();
        self.n_rows = self.n_rows.max(values.len());
        self.columns.insert(name.to_string(), values);
    }

    /// Get a column as f32 values.
    #[must_use]
    pub fn get_f32(&self, name: &str) -> Option<Vec<f32>> {
        self.columns.get(name).map(|col| col.iter().filter_map(DataValue::as_f32).collect())
    }

    /// Get a column.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&[DataValue]> {
        self.columns.get(name).map(std::vec::Vec::as_slice)
    }

    /// Get number of rows.
    #[must_use]
    pub fn nrow(&self) -> usize {
        self.n_rows
    }

    /// Get number of columns.
    #[must_use]
    pub fn ncol(&self) -> usize {
        self.columns.len()
    }

    /// Check if a column exists.
    #[must_use]
    pub fn has_column(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }

    /// Get column names.
    #[must_use]
    pub fn columns(&self) -> Vec<&str> {
        self.columns.keys().map(std::string::String::as_str).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataframe_from_xy() {
        let df = DataFrame::from_xy(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert_eq!(df.nrow(), 3);
        assert_eq!(df.ncol(), 2);
        assert!(df.has_column("x"));
        assert!(df.has_column("y"));
    }

    #[test]
    fn test_dataframe_get_f32() {
        let df = DataFrame::from_xy(&[1.0, 2.0], &[3.0, 4.0]);
        let x = df.get_f32("x").expect("operation should succeed");
        assert_eq!(x, vec![1.0, 2.0]);
    }

    #[test]
    fn test_data_value_conversions() {
        let num: DataValue = 42.0f32.into();
        assert_eq!(num.as_f32(), Some(42.0));

        let text: DataValue = "hello".into();
        assert_eq!(text.as_str(), Some("hello"));
    }

    #[test]
    fn test_dataframe_from_data() {
        let df = DataFrame::from_data(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(df.nrow(), 4);
        assert!(df.has_column("data"));
        let data = df.get_f32("data").expect("operation should succeed");
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dataframe_add_column_str() {
        let mut df = DataFrame::new();
        df.add_column_str("names", &["Alice", "Bob", "Charlie"]);
        assert_eq!(df.nrow(), 3);
        assert!(df.has_column("names"));
    }

    #[test]
    fn test_dataframe_get() {
        let df = DataFrame::from_xy(&[1.0, 2.0], &[3.0, 4.0]);
        let col = df.get("x").expect("key should exist");
        assert_eq!(col.len(), 2);
        assert_eq!(col[0].as_f32(), Some(1.0));
    }

    #[test]
    fn test_dataframe_get_missing() {
        let df = DataFrame::new();
        assert!(df.get("missing").is_none());
        assert!(df.get_f32("missing").is_none());
    }

    #[test]
    fn test_dataframe_columns() {
        let df = DataFrame::from_xy(&[1.0], &[2.0]);
        let cols = df.columns();
        assert_eq!(cols.len(), 2);
        assert!(cols.contains(&"x"));
        assert!(cols.contains(&"y"));
    }

    #[test]
    fn test_dataframe_empty() {
        let df = DataFrame::new();
        assert_eq!(df.nrow(), 0);
        assert_eq!(df.ncol(), 0);
        assert!(!df.has_column("anything"));
    }

    #[test]
    fn test_dataframe_from_xy_unequal() {
        // Different length arrays - should take minimum
        let df = DataFrame::from_xy(&[1.0, 2.0, 3.0], &[4.0, 5.0]);
        let x = df.get_f32("x").expect("operation should succeed");
        let y = df.get_f32("y").expect("operation should succeed");
        assert_eq!(x.len(), 2);
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn test_data_value_null() {
        let null = DataValue::Null;
        assert_eq!(null.as_f32(), None);
        assert_eq!(null.as_str(), None);
    }

    #[test]
    fn test_data_value_from_string() {
        let text: DataValue = String::from("test").into();
        assert_eq!(text.as_str(), Some("test"));
    }

    #[test]
    fn test_data_value_as_f32_on_text() {
        let text: DataValue = "hello".into();
        assert_eq!(text.as_f32(), None);
    }

    #[test]
    fn test_data_value_as_str_on_number() {
        let num: DataValue = 42.0f32.into();
        assert_eq!(num.as_str(), None);
    }

    #[test]
    fn test_data_value_debug_clone_eq() {
        let v1 = DataValue::Number(42.0);
        let v2 = v1.clone();
        assert_eq!(v1, v2);
        let _ = format!("{v1:?}");
    }

    #[test]
    fn test_dataframe_debug_clone() {
        let df = DataFrame::from_xy(&[1.0], &[2.0]);
        let df2 = df.clone();
        assert_eq!(df2.nrow(), 1);
        let _ = format!("{df2:?}");
    }

    #[test]
    fn test_dataframe_default() {
        let df = DataFrame::default();
        assert_eq!(df.nrow(), 0);
    }
}
