//! Cell types for Jupyter notebooks.

use serde::{Deserialize, Serialize};

use super::split_source;

/// Cell type in a notebook
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellType {
    /// Executable code cell
    Code,
    /// Markdown documentation cell
    Markdown,
    /// Raw text cell
    Raw,
}

impl std::fmt::Display for CellType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Code => write!(f, "code"),
            Self::Markdown => write!(f, "markdown"),
            Self::Raw => write!(f, "raw"),
        }
    }
}

/// A single cell in a notebook
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cell {
    /// Type of cell
    pub cell_type: CellType,
    /// Source content (lines)
    pub source: Vec<String>,
    /// Cell outputs (for code cells)
    pub outputs: Vec<CellOutput>,
    /// Execution count (for code cells)
    pub execution_count: Option<u32>,
    /// Cell metadata
    pub metadata: CellMetadata,
}

impl Cell {
    /// Create a new code cell
    pub fn code(source: impl Into<String>) -> Self {
        Self {
            cell_type: CellType::Code,
            source: split_source(source.into()),
            outputs: Vec::new(),
            execution_count: None,
            metadata: CellMetadata::default(),
        }
    }

    /// Create a new markdown cell
    pub fn markdown(source: impl Into<String>) -> Self {
        Self {
            cell_type: CellType::Markdown,
            source: split_source(source.into()),
            outputs: Vec::new(),
            execution_count: None,
            metadata: CellMetadata::default(),
        }
    }

    /// Create a new raw cell
    pub fn raw(source: impl Into<String>) -> Self {
        Self {
            cell_type: CellType::Raw,
            source: split_source(source.into()),
            outputs: Vec::new(),
            execution_count: None,
            metadata: CellMetadata::default(),
        }
    }

    /// Add an output to the cell
    pub fn with_output(mut self, output: CellOutput) -> Self {
        self.outputs.push(output);
        self
    }

    /// Set execution count
    pub fn with_execution_count(mut self, count: u32) -> Self {
        self.execution_count = Some(count);
        self
    }

    /// Get the source as a single string
    pub fn source_text(&self) -> String {
        self.source.join("")
    }
}

/// Cell output
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellOutput {
    /// Output type
    pub output_type: String,
    /// Output data
    pub data: Option<serde_json::Value>,
    /// Text output
    pub text: Option<Vec<String>>,
}

impl CellOutput {
    /// Create a stream output (stdout/stderr)
    pub fn stream(_name: &str, text: impl Into<String>) -> Self {
        Self {
            output_type: "stream".to_string(),
            data: None,
            text: Some(split_source(text.into())),
        }
    }

    /// Create an execute result output
    pub fn execute_result(data: serde_json::Value) -> Self {
        Self { output_type: "execute_result".to_string(), data: Some(data), text: None }
    }
}

/// Cell metadata
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellMetadata {
    /// Whether the cell is collapsed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collapsed: Option<bool>,
    /// Whether the cell is scrolled
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scrolled: Option<bool>,
}
