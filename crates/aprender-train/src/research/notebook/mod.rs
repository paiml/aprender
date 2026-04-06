//! Notebook Exporter for Jupyter bridge (ENT-024)
//!
//! Provides export to Jupyter notebook format (.ipynb) with
//! evcxr kernel support for Rust.

mod cell;
mod exporter;
mod kernel;

#[cfg(test)]
mod tests;

pub use cell::{Cell, CellMetadata, CellOutput, CellType};
pub use exporter::{NotebookExporter, NotebookMetadata};
pub use kernel::KernelSpec;

/// Split source into lines for Jupyter format
pub(crate) fn split_source(source: String) -> Vec<String> {
    source
        .lines()
        .map(|line| format!("{line}\n"))
        .collect::<Vec<_>>()
        .into_iter()
        .enumerate()
        .map(|(i, mut line)| {
            // Remove trailing newline from last line
            if i == source.lines().count().saturating_sub(1) {
                line.pop();
            }
            line
        })
        .collect()
}
