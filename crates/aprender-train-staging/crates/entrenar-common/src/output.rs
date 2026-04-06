//! Output formatting and table rendering (Visual Control principle).
//!
//! Provides consistent output formatting across all entrenar tools,
//! supporting both human-readable tables and machine-parseable JSON.

use serde::Serialize;

/// A formatted table for terminal output.
#[derive(Debug, Clone)]
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    column_widths: Vec<usize>,
}

impl Table {
    /// Get the table headers.
    pub fn headers(&self) -> &[String] {
        &self.headers
    }

    /// Get the table rows.
    pub fn rows(&self) -> &[Vec<String>] {
        &self.rows
    }

    /// Render the table as a string.
    pub fn render(&self) -> String {
        if self.headers.is_empty() {
            return String::new();
        }

        let mut output = String::new();

        // Top border
        output.push_str(&self.render_border('┌', '┬', '┐'));

        // Header row
        output.push_str(&self.render_row(&self.headers));

        // Header separator
        output.push_str(&self.render_border('├', '┼', '┤'));

        // Data rows
        for row in &self.rows {
            output.push_str(&self.render_row(row));
        }

        // Bottom border
        output.push_str(&self.render_border('└', '┴', '┘'));

        output
    }

    fn render_border(&self, left: char, mid: char, right: char) -> String {
        let mut line = String::new();
        line.push(left);
        for (i, width) in self.column_widths.iter().enumerate() {
            line.push_str(&"─".repeat(*width + 2));
            if i < self.column_widths.len() - 1 {
                line.push(mid);
            }
        }
        line.push(right);
        line.push('\n');
        line
    }

    fn render_row(&self, values: &[String]) -> String {
        let mut line = String::new();
        line.push('│');
        for (i, (value, width)) in values.iter().zip(&self.column_widths).enumerate() {
            line.push(' ');
            line.push_str(&format!("{:width$}", value, width = *width));
            line.push(' ');
            if i < self.column_widths.len() - 1 {
                line.push('│');
            }
        }
        line.push('│');
        line.push('\n');
        line
    }

    /// Convert to JSON representation.
    pub fn to_json(&self) -> String {
        let records: Vec<_> = self
            .rows
            .iter()
            .map(|row| {
                self.headers
                    .iter()
                    .zip(row)
                    .map(|(h, v)| (h.clone(), v.clone()))
                    .collect::<std::collections::HashMap<_, _>>()
            })
            .collect();

        serde_json::to_string_pretty(&records).unwrap_or_default()
    }
}

/// Builder for creating tables.
#[derive(Debug, Default)]
pub struct TableBuilder {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl TableBuilder {
    /// Create a new table builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the table headers.
    pub fn headers(mut self, headers: Vec<impl Into<String>>) -> Self {
        self.headers = headers.into_iter().map(Into::into).collect();
        self
    }

    /// Add a row to the table.
    pub fn row(mut self, row: Vec<impl Into<String>>) -> Self {
        self.rows.push(row.into_iter().map(Into::into).collect());
        self
    }

    /// Add multiple rows to the table.
    pub fn rows(mut self, rows: Vec<Vec<impl Into<String> + Clone>>) -> Self {
        for row in rows {
            self.rows.push(row.into_iter().map(Into::into).collect());
        }
        self
    }

    /// Build the table.
    pub fn build(self) -> Table {
        let mut column_widths: Vec<usize> = self.headers.iter().map(String::len).collect();

        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < column_widths.len() {
                    column_widths[i] = column_widths[i].max(cell.len());
                }
            }
        }

        Table { headers: self.headers, rows: self.rows, column_widths }
    }
}

/// Format bytes as human-readable size.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.1} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Format a large number with commas.
pub fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Format a duration in human-readable form.
pub fn format_duration(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else if seconds < 3600.0 {
        let mins = (seconds / 60.0).floor();
        let secs = seconds % 60.0;
        format!("{mins}m {secs:.0}s")
    } else {
        let hours = (seconds / 3600.0).floor();
        let mins = ((seconds % 3600.0) / 60.0).floor();
        format!("{hours}h {mins}m")
    }
}

/// Structured output that can be rendered as table or JSON.
#[derive(Debug, Clone, Serialize)]
#[allow(clippy::type_complexity)]
pub struct StructuredOutput<T: Serialize> {
    pub data: T,
    #[serde(skip)]
    pub table_headers: Vec<String>,
    #[serde(skip)]
    pub row_fn: Option<fn(&T) -> Vec<Vec<String>>>,
}

impl<T: Serialize> StructuredOutput<T> {
    /// Create new structured output.
    pub fn new(data: T) -> Self {
        Self { data, table_headers: vec![], row_fn: None }
    }

    /// Render as JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.data).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_builder() {
        let table = TableBuilder::new()
            .headers(vec!["Name", "Value", "Type"])
            .row(vec!["alpha", "0.7", "float"])
            .row(vec!["temperature", "4.0", "float"])
            .build();

        assert_eq!(table.headers().len(), 3);
        assert_eq!(table.rows().len(), 2);
    }

    #[test]
    fn test_table_render() {
        let table = TableBuilder::new().headers(vec!["A", "B"]).row(vec!["1", "2"]).build();

        let rendered = table.render();
        assert!(rendered.contains('┌'));
        assert!(rendered.contains('│'));
        assert!(rendered.contains('└'));
        assert!(rendered.contains('A'));
        assert!(rendered.contains('1'));
    }

    #[test]
    fn test_table_to_json() {
        let table =
            TableBuilder::new().headers(vec!["name", "value"]).row(vec!["test", "123"]).build();

        let json = table.to_json();
        assert!(json.contains("\"name\""));
        assert!(json.contains("\"test\""));
        assert!(json.contains("\"value\""));
        assert!(json.contains("\"123\""));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1500), "1.5 KB");
        assert_eq!(format_bytes(1_500_000), "1.4 MB");
        assert_eq!(format_bytes(1_500_000_000), "1.4 GB");
        assert_eq!(format_bytes(1_500_000_000_000), "1.4 TB");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1_000_000), "1,000,000");
        assert_eq!(format_number(7_000_000_000), "7,000,000,000");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30.0), "30.0s");
        assert_eq!(format_duration(90.0), "1m 30s");
        assert_eq!(format_duration(3700.0), "1h 1m");
    }

    #[test]
    fn test_column_widths_adapt_to_content() {
        let table = TableBuilder::new()
            .headers(vec!["Short", "X"])
            .row(vec!["A very long value here", "Y"])
            .build();

        let rendered = table.render();
        // The long value should fit in the table
        assert!(rendered.contains("A very long value here"));
    }

    #[test]
    fn test_empty_table_renders_empty() {
        let table = TableBuilder::new().build();
        let rendered = table.render();
        assert!(rendered.is_empty());
    }

    #[test]
    fn test_table_with_no_rows() {
        let table = TableBuilder::new().headers(vec!["A", "B"]).build();

        let rendered = table.render();
        assert!(rendered.contains('A'));
        assert!(rendered.contains('B'));
        // Should still have borders
        assert!(rendered.contains('┌'));
        assert!(rendered.contains('└'));
    }

    #[test]
    fn test_table_builder_rows_method() {
        let table = TableBuilder::new()
            .headers(vec!["X", "Y"])
            .rows(vec![vec!["1", "2"], vec!["3", "4"]])
            .build();

        assert_eq!(table.rows().len(), 2);
        assert_eq!(table.rows()[0], vec!["1", "2"]);
        assert_eq!(table.rows()[1], vec!["3", "4"]);
    }

    #[test]
    fn test_format_bytes_edge_cases() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1), "1 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
    }

    #[test]
    fn test_format_number_edge_cases() {
        assert_eq!(format_number(1), "1");
        assert_eq!(format_number(12), "12");
        assert_eq!(format_number(123), "123");
        assert_eq!(format_number(1234), "1,234");
        assert_eq!(format_number(12345), "12,345");
        assert_eq!(format_number(123456), "123,456");
    }

    #[test]
    fn test_format_duration_edge_cases() {
        assert_eq!(format_duration(0.0), "0.0s");
        assert_eq!(format_duration(59.9), "59.9s");
        assert_eq!(format_duration(60.0), "1m 0s");
        assert_eq!(format_duration(3599.0), "59m 59s");
        assert_eq!(format_duration(3600.0), "1h 0m");
    }

    #[test]
    fn test_structured_output_to_json() {
        #[derive(serde::Serialize)]
        struct TestData {
            name: String,
            value: i32,
        }

        let output = StructuredOutput::new(TestData { name: "test".into(), value: 42 });

        let json = output.to_json();
        assert!(json.contains("\"name\""));
        assert!(json.contains("\"test\""));
        assert!(json.contains("42"));
    }

    #[test]
    fn test_table_json_empty_rows() {
        let table = TableBuilder::new().headers(vec!["a", "b"]).build();

        let json = table.to_json();
        assert_eq!(json, "[]");
    }

    #[test]
    fn test_table_json_multiple_rows() {
        let table = TableBuilder::new()
            .headers(vec!["key", "val"])
            .row(vec!["x", "1"])
            .row(vec!["y", "2"])
            .build();

        let json = table.to_json();
        assert!(json.contains("\"key\""));
        assert!(json.contains("\"val\""));
        assert!(json.contains("\"x\""));
        assert!(json.contains("\"y\""));
    }
}
