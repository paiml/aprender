//! Multi-Format Export System (PMAT-036)
//!
//! Unified export system for benchmark results and analysis reports.
//!
//! # Features
//!
//! - Multiple export formats: JSON, CSV, Markdown, HTML
//! - Report types: Benchmark, Comparison, Regression, Summary
//! - File write operations
//! - Fluent report builder API
//!
//! # Falsification Criteria (F1281-F1290)
//!
//! See `tests/export_reporting_f1281.rs` for falsification tests.

mod builder;
mod exporter;
mod types;

pub use builder::ReportBuilder;
pub use exporter::ReportExporter;
pub use types::{
    BenchmarkMetric, BenchmarkReport, ComparisonEntry, ComparisonReport, ExportFormat, ReportType,
};

#[cfg(test)]
mod tests;
