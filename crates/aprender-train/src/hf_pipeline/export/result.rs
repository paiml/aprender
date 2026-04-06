//! Export result types.

use std::path::PathBuf;

use super::format::ExportFormat;

/// Export result
#[derive(Debug, Clone)]
pub struct ExportResult {
    /// Output path
    pub path: PathBuf,
    /// Format used
    pub format: ExportFormat,
    /// File size in bytes
    pub size_bytes: u64,
    /// Number of tensors exported
    pub num_tensors: usize,
}

impl ExportResult {
    /// Format size as human-readable string
    #[must_use]
    pub fn size_human(&self) -> String {
        if self.size_bytes >= 1_000_000_000 {
            format!("{:.2} GB", self.size_bytes as f64 / 1e9)
        } else if self.size_bytes >= 1_000_000 {
            format!("{:.2} MB", self.size_bytes as f64 / 1e6)
        } else if self.size_bytes >= 1_000 {
            format!("{:.2} KB", self.size_bytes as f64 / 1e3)
        } else {
            format!("{} B", self.size_bytes)
        }
    }
}
